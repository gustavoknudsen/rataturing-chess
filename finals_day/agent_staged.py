"""Staged agent: import instantly, compile in the background, upgrade mid-game.

Ship this AS agent.py when the init budget is cut below what numba needs.
Import returns in well under a second, so it satisfies any init budget at all,
including zero. Moves come from the pure-python fallback until the real engine
finishes compiling, then from the real engine for the rest of the game.

The shipped layout is three files, because this one takes the name the platform
imports and the real engine has to move out of the way:

    agent.py        this file
    agent_real.py   the real engine, today's agent.py, renamed and unchanged
    agent_pure.py   the python-chess fallback

finals_day/stage.py builds exactly that layout from submission.zip.

Why this works. The 90 s budget covers `import agent`, and the platform starts
the clock afterwards. The compile is ~31 s of wall time that has to happen
somewhere, and the only requirement is that a legal move comes back on demand.
Running it on a thread moves it out of the import and into the first few moves
of the game, where the opponent's thinking time pays for most of it.

What it costs. The opening moves are played by a ~20k nps python engine instead
of a ~690k nps one. Against a book those moves are usually forced anyway, and
by roughly move 5 the real engine has taken over. That is a far smaller loss
than failing init, which loses every game.

Threading and btc_nrt. The engine replaces numba's atomic reference counting
with a non-atomic version, which is only correct single-threaded. That holds
here: the compile thread is the only thread touching numba, and the main thread
runs pure python until it has joined. The handover is one boolean read, and
after it the compile thread is finished. This is the same argument the UCI
adapter relies on.
"""

import os
import sys
import threading
import time

print("init: start", flush=True)
_started = time.perf_counter()

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

# Set to a value to force a stage, for testing: "pure" never upgrades, "real"
# waits for the engine on the first move.
FORCE = os.environ.get("BTC_STAGE", "")

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
COMPILING_CLOCK_DIV = int(os.environ.get("BTC_STAGE_CLOCK_DIV", "20"))
DONATE_DIV = int(os.environ.get("BTC_STAGE_DONATE_DIV", "30"))

_engine = None
_ready = threading.Event()
_failed = False


def _compile():
    """Import and warm the real engine. Runs off the main thread."""
    global _engine, _failed
    try:
        import agent_real as real
        _engine = real
        print("stage: real engine ready at %.1fs"
              % (time.perf_counter() - _started), flush=True)
    except Exception as exc:
        _failed = True
        print("stage: real engine unavailable, staying pure: %r" % (exc,),
              flush=True)
    finally:
        _ready.set()


def _start():
    if FORCE != "pure":
        threading.Thread(target=_compile, daemon=True).start()


def _donate(time_left_ms):
    """Idle a slice of our own clock so the compile thread gets the core.

    Sleeping is what hands the GIL over. Returning instantly would leave the
    compile to scrape scheduler slices out of the opponent's thinking time,
    and a measured 42 s compile had not finished after 180 s of back-to-back
    fallback moves for exactly that reason.
    """
    deadline = time.perf_counter() + max(time_left_ms, 0) / (
        1000.0 * max(DONATE_DIV, 1))
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
    try:
        move = agent_pure.get_move(
            fen, max(time_left_ms // COMPILING_CLOCK_DIV, 60))
        _donate(time_left_ms)
        return move
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


_start()
print("init: %s" % (_BOOK_STATUS,), flush=True)
print("init: staged agent ready in %.2fs" % (time.perf_counter() - _started),
      flush=True)
