"""AI Chessathon agent: BetterThanCris ported to Python + numba.

Entry point required by the platform: get_move(fen, time_left_ms) -> UCI.
Everything heavy happens at import inside the 90 s init budget: numba
compilation of the whole engine via a warmup search, and table construction.
A python-chess legality check wraps every result so no failure path can
return an illegal move, and no failure path raises: get_move falls back
through our own parser and generator, which read positions python-chess
rejects, before giving up. An exception here would lose the game."""

import os
import time

_import_started = time.perf_counter()

# First, and flushed. stdout is a pipe in the platform container and therefore
# block-buffered, so an unflushed line can sit unsent past the 90 s import
# watchdog, and no output inside that window is a loss. This proves the process
# is alive before any of the expensive work starts.
print("init: start", flush=True)

import chess

# Constraint panel. Every BTC_* flag in the engine is read with os.environ.get
# at module import, and the engine modules are imported below, so anything set
# here is live everywhere. That makes the flags reachable in competition, where
# nothing outside this file can set an environment variable.
#
# Empty is the shipped configuration and changes nothing. setdefault, so a real
# environment variable still wins and the development tooling keeps working.
# Measured 2026-09-11, so the comments here are figures rather than guesses.
# finals_day/NIGHT_PLAN.md carries the decision tree these belong to.
PANEL = {
    # Memory. The table is the only large allocation: a full agent uses
    # 682 MB, and each halving of the table saves exactly its own size.
    # 1 GB needs nothing. 512 MB needs the line below, which measures 459 MB.
    # 256 MB is unreachable; the floor is 432 MB of numba runtime.
    # "BTC_TT_ENTRIES": "2097152",  # power of two, rounded down if not

    # Cores. BOTH of these are required together. BTC_THREADS alone is
    # refused, because non-atomic refcounting plus threads is memory
    # corruption. Setting BTC_NRT=0 gives back the 32% it is worth, so two
    # threads may be a net loss: prove it in a practice round first.
    # "BTC_THREADS": "4",
    # "BTC_NRT": "0",

    # Time control. Read the real increment into INCREMENT_MS below as well;
    # these two only matter if finals_day/tc_tune.py says the control flags.
    # "BTC_MOVE_OVERHEAD": "420",  # per-move reserve, ms
    # "BTC_MTG": "40",             # assumed moves remaining

    # Network. BTC_NNUE=0 falls back to the hand-crafted evaluation, which is
    # weaker on average but has no distributional blind spots: reach for it
    # only if the starting positions become irregular, Chess960 above all.
    # "BTC_NET": "net_small.npz",  # smaller network, needs BTC_NET_UNITS too
    # "BTC_NET_UNITS": "98",       # per-net eval scale, see btc_eval
    # "BTC_NNUE": "0",

    # Init. Both measured and both disappointing: ENDGAMES saves nothing at
    # all, MINIMAL saves 28% and costs strength for the whole game. If the
    # init budget is cut, ship finals_day/agent_staged.py instead of either.
    # "BTC_MINIMAL": "1",
    # "BTC_ENDGAMES": "0",
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


# 0 keeps the engine's own TT_ENTRIES. The transposition table is 256 MiB and
# is by far the largest thing we allocate, so it is the knob to turn if the
# memory limit drops. Entries are rounded down to a power of two. Edit the
# default here, or set BTC_TT_ENTRIES in PANEL above.
TT_ENTRIES_OVERRIDE = _round_down_pow2(_int_env("BTC_TT_ENTRIES", 0))

# The real increment of the real time control, in ms.
#
# IF THE TIME CONTROL CHANGES, THIS IS THE FIRST THING TO EDIT. Nothing
# derives it and nothing detects a mismatch: the increment and the platform's
# per-move overhead are confounded in the clock, so they cannot be separated
# from observation. A wrong value here mis-budgets every move of every game
# and produces no error, which is the worst failure mode available.
INCREMENT_MS = 500

# Must precede every engine import: it rewrites how numba emits reference
# counting, and applying it after the first compile silently does nothing.
import btc_nrt  # noqa: F401
import btc_book
import btc_core as core
import btc_game
import btc_search as search
import btc_time

_TT = TT_ENTRIES_OVERRIDE or search.TT_ENTRIES

# One core is the shipped condition, so this is 1 and the parallel module is
# never even imported. Raise it only if the hardware changes, and read
# btc_parallel first: threads are refused outright unless BTC_NRT=0, because
# the reference counting patch is memory corruption in a threaded process.
THREADS = _int_env("BTC_THREADS", 1)

if THREADS > 1:
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


def _search_move(fen, time_left_ms):
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
    return uci


# Previous call's remaining clock and the wall time we spent on it, so the
# next call can work out what the platform charged us. Lists, not globals, so
# the update needs no global statement.
_LAST_CLOCK = [0]
_LAST_SPENT = [0.0]


def _report_overhead(time_left_ms):
    """Print the platform's real per move cost, derived from the clock.

    Between two calls our clock moves by exactly

        drain = our own wall time + platform overhead - increment

    and we measured our own wall time, so the overhead falls out. This
    matters because OVERHEAD_MS = 420 was inferred from a rated floor and
    never timed, and every margin in finals_day/tc_tune.py rests on it. If the
    time control changes, one practice round now answers whether 420 is right
    instead of leaving it a guess.

    Diagnostic only. Nothing reads this, and it must never raise.

    One known limitation: our own wall time is captured before the legality
    check and the fallback, so if the search fails or returns an illegal move,
    that move's cost is under-counted and the NEXT reading comes out high.
    Both of those paths print loudly, so a skewed reading is easy to attribute.
    The timing is not moved into a finally block because get_move has three
    exits and is the one function that must never acquire a new way to break.
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
