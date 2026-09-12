"""AI Chessathon agent: BetterThanCris ported to Python + numba.

Entry point required by the platform: get_move(fen, time_left_ms) -> UCI.
Everything heavy happens at import, inside the init budget: numba
compilation of the whole engine via a warmup search, and table loading. The
final cut that budget below the compile time; staged/agent_staged.py is the
wrapper that met it.
A python-chess legality check wraps every result so no failure path can
return an illegal move, and no failure path raises: get_move falls back
through our own parser and generator, which read positions python-chess
rejects, before giving up. An exception here would lose the game."""

import os
import time

_import_started = time.perf_counter()

# First, and flushed. stdout is a pipe in the platform container and therefore
# block-buffered, so an unflushed line can sit unsent past the import
# watchdog, and no output inside that window is a loss. This proves the process
# is alive before any of the expensive work starts.
print("init: start", flush=True)

import chess

# Constraint panel. Every BTC_* flag is read with os.environ.get at module
# import, and the engine modules are imported below, so values set here are
# live everywhere. setdefault, so a real environment variable still wins.
# See docs/FINALS_DAY.md for the measurements behind the shipped values.
PANEL = {
    # Time management. The platform charges 1 to 2 ms per move, not the 420
    # the qualification build reserved, and the engine was finishing lost
    # games with a minute unused.
    "BTC_MOVE_OVERHEAD": "100",
    "BTC_MTG": "28",
    # Levers kept for a changed platform, all measured:
    # "BTC_TT_ENTRIES": "2097152",   memory cap 512 MB (uses 459 MB)
    # "BTC_THREADS": "4", "BTC_NRT": "0",   more cores; both are required
    # "BTC_NNUE": "0",   hand-crafted evaluation for irregular start positions
}
for _key, _value in PANEL.items():
    os.environ.setdefault(_key, _value)


def _int_env(name, default):
    """An environment integer that can never stop the import.

    Everything else in this file is wrapped because an exception here loses
    every game rather than one. A bare int() on a panel value is the same
    exposure with a friendlier cause: BTC_TT_ENTRIES=16m is a typo, not a
    reason to forfeit."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        print(f"init: ignoring bad {name}={raw!r}", flush=True)
        return default


def _round_down_pow2(value):
    """Largest power of two at or below value, or 0 if there is no sane one.

    SearchState asserts a power of two of at least 8, and 1000000 is a far
    more natural thing to type than 1048576. Rounding down spends less memory
    than asked for, which is the safe direction when a cap is the reason the
    value was set at all."""
    if value < 8:
        return 0
    return 1 << (int(value).bit_length() - 1)


# 0 keeps the engine's own TT_ENTRIES (256 MiB, the largest allocation).
TT_ENTRIES_OVERRIDE = _round_down_pow2(_int_env("BTC_TT_ENTRIES", 0))

# The increment of the real time control, in ms. Nothing derives it and a
# wrong value mis-budgets every move silently.
INCREMENT_MS = 500

# Must precede every engine import: it rewrites how numba emits reference
# counting, and applying it after the first compile silently does nothing.
import btc_nrt  # noqa: F401
import btc_book
import btc_core as core
import btc_game
import btc_search as search
import btc_time
import numpy as np

_TT = TT_ENTRIES_OVERRIDE or search.TT_ENTRIES

# One core is the shipped condition, so this is 1 and the parallel module is
# never even imported. Raise it only if the hardware changes, and read
# btc_parallel first: threads are refused outright unless BTC_NRT=0, because
# the reference counting patch is memory corruption in a threaded process.
THREADS = _int_env("BTC_THREADS", 1)

# Think on the opponent's clock. Pointless under the tournament rules, which
# say both agents take the core in turns, so there is no CPU to think with
# while the opponent thinks. Needs BTC_NRT=0 like any other thread. See
# btc_parallel.Ponderer for why there is no ponderhit bookkeeping.
PONDER = _int_env("BTC_PONDER", 0)

if THREADS > 1 or PONDER:
    import btc_parallel
    PARALLEL = btc_parallel.ParallelSearch(THREADS, _TT)
    STATE = PARALLEL.main
    print(f"init: {PARALLEL.threads} search threads", flush=True)

    def _search(bb, st, keys, count, soft, hard, max_depth=search.MAX_SEARCH_PLY):
        return btc_parallel.search_position(PARALLEL, bb, st, keys, count,
                                            soft, hard, max_depth)
else:
    PARALLEL = None
    STATE = search.SearchState(_TT)

    def _search(bb, st, keys, count, soft, hard, max_depth=search.MAX_SEARCH_PLY):
        return search.search_position(STATE, bb, st, keys, count, soft, hard,
                                      max_depth)

PONDERER = None
if PONDER and PARALLEL is not None:
    PONDERER = btc_parallel.Ponderer(PARALLEL)
    print(f"init: pondering {'on' if PONDERER.state is not None else 'refused'}",
          flush=True)

TRACKER = btc_game.GameTracker()
BB, ST = core.new_board()

# Effectively no deadline, in ms. Large enough that the warmup can never be
# cut short by compilation, small enough to stay well inside float64 precision
# when added to perf_counter. Overridable only so the tests can force a
# starved warmup and prove it degrades instead of crashing.
HARD_MS_NONE = int(os.environ.get("BTC_WARMUP_HARD_MS", "3600000"))


def _warmup():
    core.warmup()
    core.parse_fen(core.START_FEN, BB, ST)
    keys = STATE.rep[:1].copy()
    keys[0] = BB[core.HASH]
    # No time limit. search_position turns hard_ms into an absolute wall-clock
    # deadline before its first njit call, and that call is where the whole
    # numba compile happens, so any finite budget is spent on compilation and
    # the search aborts at depth 1. This warmup is bounded by max_depth
    # instead. A 60 s budget here passed locally and crashed on the platform,
    # whose core is slower, once the engine grew past it.
    mv, score, depth, nodes = _search(
        BB, ST, keys, 1, soft=0, hard=HARD_MS_NONE, max_depth=4)
    if depth != 4 or nodes <= 0:
        # Never fatal: an exception here loses the game outright, while a
        # partial warmup only means some compilation lands on the first move.
        print(f"warmup incomplete: depth {depth} nodes {nodes}", flush=True)
    STATE.main_hist[:] = 0
    STATE.tt_key[:] = 0
    STATE.tt_data[:] = 0
    STATE.sc[:] = 0


_warmup()
print(f"init: {btc_book.load()}", flush=True)
print(f"init: engine compiled in "
      f"{time.perf_counter() - _import_started:.1f}s", flush=True)


def _normalise_fen(fen):
    """The same position in a form python-chess will parse.

    The referee sends standard FENs, but variant dialects do not parse: a
    three-check FEN carries a seventh field (+0+0), a crazyhouse FEN carries a
    pocket in brackets. python-chess raises on both, and core.parse_fen raises
    on the pocket too, so everything that parses goes through here first. A
    standard six-field FEN passes through byte-identical.
    """
    parts = fen.split()
    if parts and "[" in parts[0]:
        parts[0] = parts[0].split("[", 1)[0]
    return " ".join(parts[:6])


def _core_move(fen):
    """A legal move from our own generator. Returns None if even this fails.

    Takes the normalised FEN: core.parse_fen reads a seventh field harmlessly
    but raises KeyError on a crazyhouse pocket, so the raw string is not
    safer here, only differently unsafe.
    """
    bb, st = core.new_board()
    core.parse_fen(_normalise_fen(fen), bb, st)
    moves = core.legal_moves(bb, st)
    return core.move_to_uci(moves[0]) if moves else None


def _fallback_move(fen):
    """A legal move without searching, by whichever parser can read the
    position. Never raises: an exception here loses the game outright, and
    this is the path that exists to stop that happening."""
    try:
        board = chess.Board(_normalise_fen(fen))
        moves = list(board.legal_moves)
        if moves:
            for mv in moves:
                if board.is_capture(mv):
                    return mv.uci()
            return moves[0].uci()
    except Exception as exc:
        print(f"fallback: python-chess rejected the fen {exc!r}", flush=True)
    try:
        uci = _core_move(fen)
        if uci is not None:
            return uci
    except Exception as exc:
        print(f"fallback: own parser failed too {exc!r}", flush=True)
    return "0000"


def _book_move(fen, bb, st):
    """Book move for this root, or None. Keeps TRACKER in step.

    The tracker bookkeeping is why this lives inside _search_move rather than
    beside it: `update` has already run, and a move played without the matching
    `push_our_move` would desync the repetition history for the rest of the
    game."""
    try:
        board = chess.Board(_normalise_fen(fen))
        uci = btc_book.probe(board)
    except Exception as exc:
        # A book miss costs one opening move. Letting this escape would abort
        # the search that was about to run instead.
        print(f"book: probe skipped {exc!r}", flush=True)
        return None
    if uci is None:
        return None
    for packed in core.legal_moves(bb, st):
        if core.move_to_uci(packed) == uci:
            TRACKER.push_our_move(packed)
            print(f"book {uci} move {board.fullmove_number}", flush=True)
            return uci
    # python-chess called the move legal and our generator did not produce it.
    # That is a disagreement about the position, so trust neither and search.
    print(f"book move {uci} not in our move list", flush=True)
    return None


def _ponder_next(fen, uci):
    """Start thinking about the position we expect to be asked about next.

    The guess is our own principal variation: we play `uci`, the opponent
    replies with the second move of the PV. Everything the ponder thread
    proves goes into the shared table, so a right guess means the next real
    search starts warm and a wrong one means a few entries for a position that
    did not happen. Never raises: this is an optimisation, and an exception
    here would lose a game it was supposed to help win.
    """
    if PONDERER is None or PONDERER.state is None:
        return
    try:
        if int(STATE.pv_len[0]) < 2:
            return
        reply = core.move_to_uci(int(STATE.pv_table[0, 1]))
        board = chess.Board(_normalise_fen(fen))
        board.push(chess.Move.from_uci(uci))
        board.push(chess.Move.from_uci(reply))

        bb, st = core.new_board()
        core.parse_fen(board.fen(), bb, st)
        # Real repetition history, extended by the two positions we are
        # assuming. A truncated history would let the ponder search call
        # something a draw that is not one and store that score in the shared
        # table, which is the one way this could do harm rather than nothing.
        keys, count = TRACKER.history()
        extended = np.zeros(int(count) + 2, dtype=np.uint64)
        extended[:count] = keys[:count]
        mid_bb, mid_st = core.new_board()
        board.pop()
        core.parse_fen(board.fen(), mid_bb, mid_st)
        extended[count] = mid_bb[core.HASH]
        extended[count + 1] = bb[core.HASH]
        PONDERER.start(bb, st, extended, int(count) + 2,
                       int(STATE.sc[search.SC_TT_GEN]) + 1)
    except Exception as exc:
        print(f"ponder: not started {exc!r}", flush=True)


def _search_move(fen, time_left_ms):
    # Before anything else. A ponder thread still running while the real
    # search starts would be a second thread on state the search is about to
    # reset.
    if PONDERER is not None:
        PONDERER.stop()
    TRACKER.update(_normalise_fen(fen))
    bb, st = TRACKER.bb, TRACKER.st
    book = _book_move(fen, bb, st)
    if book is not None:
        return book
    keys, count = TRACKER.history()
    soft, hard = btc_time.budget(time_left_ms, int(st[core.MOVENUM]),
                                 INCREMENT_MS)
    packed, score, depth, nodes = _search(
        bb, st, keys, count, soft, hard)
    if packed == 0:
        return None
    uci = core.move_to_uci(packed)
    print(f"move {uci} score {score} depth {depth} nodes {nodes} "
          f"clock {time_left_ms}", flush=True)
    TRACKER.push_our_move(packed)
    _ponder_next(fen, uci)
    return uci


# Previous call's remaining clock and the wall time we spent on it, so the
# next call can work out what the platform charged us. Lists, not globals, so
# the update needs no global statement.
_LAST_CLOCK = [0]
_LAST_SPENT = [0.0]


def _report_overhead(time_left_ms):
    """Print the platform's per-move cost, derived from the clock.

    Between two calls our clock moves by our own wall time plus the platform's
    overhead minus the increment, and we time our own half, so the overhead
    falls out. Diagnostic only; never raises. Our wall time is captured before
    the legality check, so a fallback move under-counts and the next reading
    comes out high; both paths print loudly.
    """
    try:
        previous = _LAST_CLOCK[0]
        # An unchanged clock means the caller is not running one: the
        # packaging smoke test hands the same 120000 to every move, and the
        # reading there is meaningless rather than merely noisy.
        if previous and previous != time_left_ms:
            # Not only when the clock falls. On a book move we spend nothing
            # and the increment makes the clock grow, which is a negative
            # drain and still a valid reading. Those are in fact the cleanest
            # samples, because our own wall time is near zero and the overhead
            # is almost the whole of the difference.
            drain = previous - time_left_ms
            implied = drain - _LAST_SPENT[0] + INCREMENT_MS
            print(f"clock: drain {drain} ours {_LAST_SPENT[0]:.0f} "
                  f"implied overhead {implied:.0f}", flush=True)
    except Exception:
        pass


def get_move(fen: str, time_left_ms: int) -> str:
    _report_overhead(time_left_ms)
    _call_started = time.perf_counter()
    try:
        uci = _search_move(fen, time_left_ms)
    except Exception as exc:
        print(f"search failed: {exc!r}", flush=True)
        uci = None
    _LAST_CLOCK[0] = time_left_ms
    _LAST_SPENT[0] = (time.perf_counter() - _call_started) * 1000.0
    if uci is not None:
        try:
            board = chess.Board(_normalise_fen(fen))
        except Exception:
            # python-chess cannot read this position but core.parse_fen could,
            # and the move came from our own generator, so it is legal by the
            # only parser that understood the position. Trust it.
            print("verify: python-chess rejected the fen, trusting the search",
                  flush=True)
            return uci
        try:
            if chess.Move.from_uci(uci) in board.legal_moves:
                return uci
            print(f"illegal move from search: {uci}", flush=True)
        except ValueError:
            print(f"malformed move from search: {uci}", flush=True)
    return _fallback_move(fen)
