"""AI Chessathon agent: BetterThanCris ported to Python + numba.

Entry point required by the platform: get_move(fen, time_left_ms) -> UCI.
Everything heavy happens at import inside the 90 s init budget: numba
compilation of the whole engine via a warmup search, and table construction.
A python-chess legality check wraps every result so no failure path can
return an illegal move."""

import os
import time

_import_started = time.perf_counter()

import chess

import btc_core as core
import btc_game
import btc_search as search
import btc_time

STATE = search.SearchState()
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
    mv, score, depth, nodes = search.search_position(
        STATE, BB, ST, keys, 1, soft_ms=0, hard_ms=HARD_MS_NONE, max_depth=4)
    if depth != 4 or nodes <= 0:
        # Never fatal: an exception here loses the game outright, while a
        # partial warmup only means some compilation lands on the first move.
        print(f"warmup incomplete: depth {depth} nodes {nodes}")
    STATE.main_hist[:] = 0
    STATE.tt_key[:] = 0
    STATE.tt_data[:] = 0
    STATE.sc[:] = 0


_warmup()
print(f"init: engine compiled in {time.perf_counter() - _import_started:.1f}s")


def _fallback_move(fen):
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    for mv in moves:
        if board.is_capture(mv):
            return mv.uci()
    return moves[0].uci()


def _search_move(fen, time_left_ms):
    TRACKER.update(fen)
    bb, st = TRACKER.bb, TRACKER.st
    keys, count = TRACKER.history()
    soft, hard = btc_time.budget(time_left_ms, int(st[core.MOVENUM]))
    packed, score, depth, nodes = search.search_position(
        STATE, bb, st, keys, count, soft, hard)
    if packed == 0:
        return None
    uci = core.move_to_uci(packed)
    print(f"move {uci} score {score} depth {depth} nodes {nodes} "
          f"clock {time_left_ms}")
    TRACKER.push_our_move(packed)
    return uci


def get_move(fen: str, time_left_ms: int) -> str:
    try:
        uci = _search_move(fen, time_left_ms)
    except Exception as exc:
        print(f"search failed: {exc!r}")
        uci = None
    if uci is not None:
        try:
            if chess.Move.from_uci(uci) in chess.Board(fen).legal_moves:
                return uci
            print(f"illegal move from search: {uci}")
        except ValueError:
            print(f"malformed move from search: {uci}")
    return _fallback_move(fen)
