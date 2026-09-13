"""UCI adapter for Rataturing.

The engine speaks get_move(fen, time_left_ms), because that is the entry point
the Chessathon platform required. Chess GUIs speak UCI. This module is the
adapter between them, and it is purely additive: it imports the competition
engine unchanged and drives it through that same entry point, so a game played
in a GUI runs the code path that played the tournament.

Three properties of the engine make that possible without editing the search.

The iterative deepening driver in btc_search.search_position is plain Python,
so wrapping _aspiration_search is enough to emit an info line after every
completed depth.

The search polls a stop flag, state.sc[SC_STOP], at seven points. Writing that
flag from another thread ends the search within microseconds, which is what
makes `stop` and `go infinite` work.

Compilation is the whole startup cost, so it runs on a background thread from
the moment the process starts, and `isready` waits for it. The engine answers
`uci` immediately, so a GUI can register and configure it with no delay.

Startup compiles the engine with numba and takes roughly a minute. That is
paid once per process, and a GUI keeps one process for a whole session.

Only the search thread touches numba arrays. The main thread reads stdin and
formats strings, and its one write into engine memory is the stop flag, a
plain store that takes no reference count. That is what keeps the non-atomic
reference counting in btc_nrt.py correct here.
"""

import os
import sys
import threading
import time

# Must happen before the engine is imported. The engine logs to stdout, which
# would corrupt the protocol, so fd 1 is pointed at stderr and the original is
# kept for UCI replies only. This is what the platform runner does too.
_REAL_OUT = os.dup(1)
os.dup2(2, 1)

# The engine ships as ordinary .py files in an engine/ folder beside the
# executable, never bundled into it. numba locates its on-disk cache from each
# function's source path, so the sources have to be real files that keep the
# same path between runs. Frozen, that anchor is the executable; from the
# repository it is this file, with the engine in ../src.
_BASE = os.path.dirname(os.path.abspath(
    sys.executable if getattr(sys, "frozen", False) else __file__))
ENGINE_DIR = None
for _candidate in (os.path.join(_BASE, "engine"), _BASE,
                   os.path.join(_BASE, os.pardir, "src")):
    if os.path.exists(os.path.join(_candidate, "agent.py")):
        ENGINE_DIR = os.path.abspath(_candidate)
        sys.path.insert(0, ENGINE_DIR)
        break
if ENGINE_DIR is None:
    raise SystemExit("cannot find agent.py: expected it beside this file, "
                     "in ./engine, or in ../src")

import chess

VERSION = "1.1"
AUTHOR = "Gustavo Knudsen"

# Defaults differ from the competition build in one place. BTC_MOVE_OVERHEAD
# is 420 ms there, which is not this engine overshooting its budget: it is the
# platform's referee slack, measured. A local GUI has no such slack, so
# shipping 420 would hand away most of a fast time control.
OPTIONS = {"Hash": 256, "OwnBook": True, "Move Overhead": 50}
LIMITS = {"Hash": (1, 4096), "Move Overhead": (0, 5000)}

ENGINE = None
READY = threading.Event()
OUT_LOCK = threading.Lock()
WORKER = None
BOOK_PROBE = None
HASH_MB = [OPTIONS["Hash"]]
SEARCH = {"started": 0.0, "max_depth": 0, "infos": 0}


def _say(line):
    with OUT_LOCK:
        os.write(_REAL_OUT, (line + "\n").encode("ascii"))


def _note(line):
    print(line, file=sys.stderr)


def _variant():
    """NNUE or Classic, decided the same way the engine decides it: by whether
    net.npz is present. No flag, so the name can never disagree with the
    evaluation actually in use."""
    if os.environ.get("BTC_NNUE", "1") != "1":
        return "Classic"
    return "NNUE" if os.path.exists(os.path.join(ENGINE_DIR, "net.npz")) \
        else "Classic"


# ---------------------------------------------------------------- compilation

def _compile():
    """Import the engine, which compiles it. Runs off the main thread."""
    global ENGINE, BOOK_PROBE
    started = time.perf_counter()
    try:
        import agent
    except Exception as exc:                         # pragma: no cover
        _note("engine failed to load: %r" % (exc,))
        READY.set()
        return
    ENGINE = agent
    BOOK_PROBE = agent.btc_book.probe
    _instrument(agent.search)
    _apply_options()
    _note("uci: ready in %.1fs" % (time.perf_counter() - started))
    READY.set()


def _ready():
    READY.wait()
    if ENGINE is None:
        raise SystemExit("engine unavailable")
    return ENGINE


# -------------------------------------------------------------------- options

def _apply_options():
    if ENGINE is None:
        return
    ENGINE.btc_time.OVERHEAD_MS = int(OPTIONS["Move Overhead"])
    ENGINE.btc_book.probe = BOOK_PROBE if OPTIONS["OwnBook"] \
        else (lambda board: None)
    _set_hash(int(OPTIONS["Hash"]))


def _set_hash(mb):
    """Rebuild the search state around a new table size. Each entry is a key
    and a payload, 16 bytes, and the state asserts a power of two because the
    bucket mask derives from it."""
    if mb == HASH_MB[0]:
        return
    entries = max(8, (mb * 1024 * 1024) // 16)
    entries = 1 << (int(entries).bit_length() - 1)
    ENGINE.STATE = ENGINE.search.SearchState(entries)
    HASH_MB[0] = mb
    _note("uci: hash %d MB (%d entries)" % (mb, entries))


def _setoption(tokens):
    if "name" not in tokens:
        return
    start = tokens.index("name") + 1
    end = tokens.index("value") if "value" in tokens else len(tokens)
    name = " ".join(tokens[start:end])
    raw = " ".join(tokens[end + 1:]) if end < len(tokens) else ""
    if name not in OPTIONS:
        _note("uci: unknown option %r" % (name,))
        return
    if isinstance(OPTIONS[name], bool):
        OPTIONS[name] = raw.strip().lower() == "true"
    else:
        low, high = LIMITS[name]
        try:
            OPTIONS[name] = min(max(int(raw), low), high)
        except ValueError:
            _note("uci: bad value for %r: %r" % (name, raw))
            return
    if READY.is_set():
        _apply_options()


# ------------------------------------------------------------- info reporting

def _instrument(search):
    """Report after every completed depth, and enforce `go depth`.

    The depth limit is applied on entry rather than on exit. Stopping after
    depth N returns depth N-1's move, because search_position checks the stop
    flag before it records the new best move. Refusing to start depth N+1
    leaves depth N recorded, which is what the GUI asked for.
    """
    original = search._aspiration_search

    def wrapper(state, bb, st, prev_score, depth, rep_base):
        limit = SEARCH["max_depth"]
        if limit and depth > limit:
            state.sc[search.SC_STOP] = 1
            return prev_score
        score = original(state, bb, st, prev_score, depth, rep_base)
        if not state.sc[search.SC_STOP]:
            _emit_info(state, depth, score)
        return score

    search._aspiration_search = wrapper


def _score_str(score):
    search = ENGINE.search
    if abs(score) > search.MATE_SCORE:
        plies = search.MATE_VALUE - abs(score)
        moves = (plies + 1) // 2
        return "mate %d" % (moves if score > 0 else -moves)
    return "cp %d" % score


def _pv(state):
    core = ENGINE.core
    count = int(state.pv_len[0])
    return " ".join(core.move_to_uci(int(state.pv_table[0, i]))
                    for i in range(count))


def _emit_info(state, depth, score):
    SEARCH["infos"] += 1
    elapsed = max(time.perf_counter() - SEARCH["started"], 1e-6)
    nodes = int(state.sc[ENGINE.search.SC_NODES])
    line = "info depth %d score %s nodes %d nps %d time %d" % (
        depth, _score_str(score), nodes, int(nodes / elapsed),
        int(elapsed * 1000))
    moves = _pv(state)
    _say(line + " pv " + moves if moves else line)


# --------------------------------------------------------------------- search

def _parse_position(tokens):
    if "fen" in tokens:
        start = tokens.index("fen") + 1
        end = tokens.index("moves") if "moves" in tokens else len(tokens)
        board = chess.Board(" ".join(tokens[start:end]))
    else:
        board = chess.Board()
    if "moves" in tokens:
        for uci in tokens[tokens.index("moves") + 1:]:
            board.push_uci(uci)
    return board


def _parse_go(tokens):
    params = {}
    for i, token in enumerate(tokens):
        if token == "infinite":
            params["infinite"] = 1
        elif token in ("wtime", "btime", "winc", "binc", "movetime", "depth") \
                and i + 1 < len(tokens):
            params[token] = int(tokens[i + 1])
    return params


def _budget_for(params, board):
    """Return (budget function, time_left_ms, max_depth).

    The engine reads its budget only through btc_time.budget, so replacing
    that function is enough to impose any of the UCI limits.
    """
    forever = 3600000
    if "infinite" in params or "depth" in params:
        # soft 0 disables the iterative deepening cutoff, so only the depth
        # limit or an explicit stop ends the search.
        return (lambda left, move, increment_ms=0: (0, forever), forever,
                int(params.get("depth", 0)))
    if "movetime" in params:
        fixed = int(params["movetime"])
        return (lambda left, move, increment_ms=0: (fixed, fixed),
                fixed * 4, 0)
    if board.turn == chess.WHITE:
        left, inc = params.get("wtime", 60000), params.get("winc", 0)
    else:
        left, inc = params.get("btime", 60000), params.get("binc", 0)
    base = ENGINE.btc_time.budget

    def clocked(time_left_ms, move_number, increment_ms=inc):
        return base(time_left_ms, move_number, increment_ms)

    return clocked, left, 0


def _run_search(board, params):
    engine = _ready()
    budget, time_left, max_depth = _budget_for(params, board)
    SEARCH["started"] = time.perf_counter()
    SEARCH["max_depth"] = max_depth
    SEARCH["infos"] = 0
    original = engine.btc_time.budget
    engine.btc_time.budget = budget
    # `infinite` and `depth` are analysis requests, and answering those from
    # the book would return a move with no evaluation and no line. Play the
    # book only when the GUI asked for a move to play.
    probe = engine.btc_book.probe
    if max_depth or "infinite" in params:
        engine.btc_book.probe = lambda position: None
    try:
        move = engine.get_move(board.fen(), int(time_left))
    except Exception as exc:                         # pragma: no cover
        _note("search failed: %r" % (exc,))
        move = None
    finally:
        engine.btc_time.budget = original
        engine.btc_book.probe = probe
        SEARCH["max_depth"] = 0
    if move and not SEARCH["infos"]:
        _say("info string book move")
    _say("bestmove " + (move or "0000"))


def _go(board, tokens):
    global WORKER
    _stop()
    if WORKER is not None and WORKER.is_alive():
        WORKER.join()
    WORKER = threading.Thread(target=_run_search,
                              args=(board.copy(), _parse_go(tokens)),
                              daemon=True)
    WORKER.start()


def _stop():
    """Ask the running search to finish. Safe to call when none is running:
    search_position clears the flag before its first node."""
    if ENGINE is not None:
        ENGINE.STATE.sc[ENGINE.search.SC_STOP] = 1


def _shutdown():
    """End any running search and let it report before the process goes away.

    The search runs on a daemon thread, so returning from main() while it is
    still working would drop its bestmove. A GUI always waits for bestmove
    before sending quit, but stdin reaching end of file mid-search is ordinary
    when the engine is driven from a script.
    """
    _stop()
    if WORKER is not None and WORKER.is_alive():
        WORKER.join(timeout=10)


def _new_game():
    """Reset everything that carries between moves. The platform started a
    process per game; here one process plays many, so game N+1 must not begin
    with game N's table, history or repetition keys."""
    engine = _ready()
    engine.TRACKER = engine.btc_game.GameTracker()
    state = engine.STATE
    for array in (state.main_hist, state.cap_hist, state.cont_hist,
                  state.counters, state.killers, state.tt_key, state.tt_data,
                  state.sc, state.fc, state.played, state.static_evals,
                  state.pv_len):
        array[:] = 0


# ----------------------------------------------------------------------- main

def _identify():
    _say("id name Rataturing %s %s" % (_variant(), VERSION))
    _say("id author " + AUTHOR)
    _say("option name Hash type spin default %d min %d max %d"
         % (OPTIONS["Hash"], LIMITS["Hash"][0], LIMITS["Hash"][1]))
    _say("option name OwnBook type check default true")
    _say("option name Move Overhead type spin default %d min %d max %d"
         % (OPTIONS["Move Overhead"], LIMITS["Move Overhead"][0],
            LIMITS["Move Overhead"][1]))
    _say("uciok")


def main():
    threading.Thread(target=_compile, daemon=True).start()
    board = chess.Board()
    for raw in sys.stdin:
        tokens = raw.split()
        if not tokens:
            continue
        command = tokens[0]
        if command == "uci":
            _identify()
        elif command == "isready":
            _ready()
            _say("readyok")
        elif command == "setoption":
            _setoption(tokens)
        elif command == "ucinewgame":
            _new_game()
        elif command == "position":
            board = _parse_position(tokens)
        elif command == "go":
            _go(board, tokens)
        elif command == "stop":
            _stop()
        elif command == "quit":
            _shutdown()
            return
        else:
            _note("uci: ignored %r" % (raw.strip(),))
    _shutdown()


if __name__ == "__main__":
    main()
