"""Staged agent: compile on a thread, spend the init budget waiting, then play.

Shipped as agent.py for the London final, where the init budget was cut from
90 s to 30 s and the engine's numba compile takes about 40 s on the platform.

    agent.py        this file
    agent_real.py   the real engine, src/agent.py renamed and unchanged
    agent_pure.py   the python-chess fallback

staged/build.py builds that layout from submission.zip.

How it works. The platform measures init from process start to the moment
import returns, then freezes the process whenever it is not our move. So the
init window is the only free CPU in the game: the compile starts on a thread
at import, import blocks until the engine is ready or INIT_WAIT_S has passed,
and whatever compile is left is paid from the clock on move one, where the
book usually answers for free. Out of book, move one waits for the real engine
rather than letting the fallback play. Measured on the platform: ready at
26.4 s of 30, compile finished 12 to 14 s into move one.

Threading and btc_nrt. The engine replaces numba's atomic reference counting
with a non-atomic version, which is only correct single-threaded. That holds
here: the compile thread is the only thread touching numba until it has
finished, and the main thread runs pure python until then.
"""

import os
import sys
import threading
import time

print("init: start", flush=True)
_started = time.perf_counter()

# Set to a value to force a stage, for testing: "pure" never upgrades, "real"
# waits for the engine on the first move.
FORCE = os.environ.get("BTC_STAGE", "")

_engine = None
_ready = threading.Event()
_failed = False


def _compile():
    """Import and warm the real engine. Runs off the main thread."""
    global _engine, _failed
    try:
        import agent_real as real
        _engine = real
        # CPU time is the number that transfers between machines: wall time
        # includes every second the referee kept the process frozen.
        print("stage: real engine ready at %.1fs wall, %.1fs cpu"
              % (time.perf_counter() - _started, time.process_time()),
              flush=True)
    except Exception as exc:
        _failed = True
        print("stage: real engine unavailable, staying pure: %r" % (exc,),
              flush=True)
    finally:
        _ready.set()


def _launch():
    """Start the compile. Called after the book is loaded, not before: the
    real engine's import loads the same book, and two loads at once would
    race on the reader list for a saving of a few hundred milliseconds."""
    if FORCE != "pure":
        threading.Thread(target=_compile, daemon=True).start()


import chess

import agent_pure

# The book needs no numba. btc_book is pure python-chess: load() opens
# polyglot readers and probe() takes a board, so it is available immediately
# and it is the SAME book the real engine plays. While the engine compiles we
# can therefore answer book positions with the real engine's own move rather
# than with a 20k nps guess, which removes most of what staging costs.
try:
    import btc_book
    _BOOK_STATUS = btc_book.load()
except Exception as _exc:
    btc_book = None
    _BOOK_STATUS = "book unavailable: %r" % (_exc,)

_launch()

# While the engine compiles, the fallback searches for a small slice of the
# move budget and then SLEEPS the rest of it.
#
# The sleep is the point, not the throttle. The fallback is CPU-bound pure
# Python and holds the GIL solid, so the compile thread only gets scheduler
# slices: measured, a 42 s compile had not finished after 180 s of back-to-back
# fallback moves, and searching less per move made it worse rather than better,
# because the moves then came faster. Sleeping releases the GIL and hands the
# compile a whole core.
#
# Spending our own clock this way is the right trade: the budget buys a 690k
# nps engine for the rest of the game instead of a few better moves now.
COMPILING_CLOCK_DIV = int(os.environ.get("BTC_STAGE_CLOCK_DIV", "40"))

# The init budget is wall time from process start to the runner's ready line,
# and the referee freezes the process the moment it is ready. So the budget
# is the one place compile time is free: every second of it spent waiting
# for the compile thread is a second not paid from the game clock later.
# Import therefore blocks until the engine is ready or this many seconds
# have passed since this module started, whichever is first. The unmeasured
# part, interpreter start plus the runner's own imports, is well under a
# second, so the margin to the 30 s budget is the rest.
INIT_WAIT_S = float(os.environ.get("BTC_STAGE_INIT_WAIT_S", "26"))

# Donation per move once the game has started. Large slices on purpose: the
# total is fixed by how much compile is left, and one big slice finishes it
# in one move, where small slices spread it over several fallback-quality
# moves. Capped as a fraction of the clock, and never below the floor.
DONATE_MAX_S = float(os.environ.get("BTC_STAGE_DONATE_MAX_S", "30"))
DONATE_FRACTION = int(os.environ.get("BTC_STAGE_DONATE_FRACTION", "4"))
DONATE_FLOOR_MS = int(os.environ.get("BTC_STAGE_DONATE_FLOOR_MS", "30000"))


def _donate(time_left_ms):
    """Idle a slice of our own clock so the compile thread gets the core.

    Sleeping is what hands the GIL over. Returning instantly would leave the
    compile to scrape scheduler slices out of the opponent's thinking time,
    and a measured 42 s compile had not finished after 180 s of back-to-back
    fallback moves for exactly that reason.
    """
    if time_left_ms < DONATE_FLOOR_MS:
        return
    slice_s = min(DONATE_MAX_S, time_left_ms / (1000.0 * max(DONATE_FRACTION, 1)))
    deadline = time.perf_counter() + slice_s
    while _engine is None and not _failed and time.perf_counter() < deadline:
        time.sleep(0.005)


def _book_move(fen):
    """The real engine's book move for this position, or None."""
    if btc_book is None:
        return None
    try:
        board = chess.Board(" ".join(fen.split()[:6]))
        return btc_book.probe(board)
    except Exception:
        return None


def get_move(fen, time_left_ms):
    """Real engine if it is ready, book then pure fallback if it is not.
    Never raises."""
    if FORCE == "real":
        _ready.wait()
    if _engine is not None:
        try:
            return _engine.get_move(fen, time_left_ms)
        except Exception as exc:
            # The real engine is meant to be incapable of this. If it manages
            # it anyway, the game is not over: fall through to the fallback.
            print("stage: real engine raised, falling back: %r" % (exc,),
                  flush=True)
    # Book first. This move is what the real engine would have played, so the
    # clock spent donating below costs nothing in move quality.
    book = _book_move(fen)
    if book is not None:
        _donate(time_left_ms)
        return book
    # Out of book: wait for the engine FIRST, then let it play this move.
    # The compile has the whole init window behind it already, so on the
    # platform it finishes 14 to 16 s into this call, and the real engine
    # then searches the position with its normal budget. The fallback only
    # plays if the compile still is not done at the end of the slice.
    _donate(time_left_ms)
    if _engine is not None:
        try:
            return _engine.get_move(fen, time_left_ms)
        except Exception as exc:
            print("stage: real engine raised, falling back: %r" % (exc,),
                  flush=True)
    try:
        return agent_pure.get_move(
            fen, max(time_left_ms // COMPILING_CLOCK_DIV, 60))
    except Exception as exc:
        print("stage: fallback raised too: %r" % (exc,), flush=True)
    try:
        board = chess.Board(" ".join(fen.split()[:6]))
        moves = list(board.legal_moves)
        if moves:
            return moves[0].uci()
    except Exception:
        pass
    return "0000"


def _process_age():
    """Seconds since this process started, from /proc on Linux, else 0.

    The platform's budget starts before this module's first line runs, so
    the wait is measured from the real process start where that can be read.
    Any failure returns 0, and the caller takes the larger of this and the
    module clock, so this can only ever shorten the wait, never lengthen it.
    """
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        start_ticks = float(fields[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        age = uptime - start_ticks / os.sysconf("SC_CLK_TCK")
        return age if 0.0 <= age < 3600.0 else 0.0
    except Exception:
        return 0.0


def _wait():
    """Spend the init budget on the compile. Event.wait releases the GIL, so
    the compile thread has the whole core until the deadline or until it
    finishes. Returning at the deadline leaves the thread running; the
    game's first moves donate whatever is left."""
    elapsed = max(time.perf_counter() - _started, _process_age())
    remaining = INIT_WAIT_S - elapsed
    if remaining > 0:
        _ready.wait(remaining)


_wait()
print("init: %s" % (_BOOK_STATUS,), flush=True)
print("init: staged agent ready in %.2fs wall, %.2fs process age, %.1fs cpu, "
      "engine %s"
      % (time.perf_counter() - _started, _process_age(), time.process_time(),
         "compiled" if _engine is not None else "still compiling"),
      flush=True)
