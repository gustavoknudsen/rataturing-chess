"""AI Chessathon agent: BetterThanCris ported to Python + numba.

Entry point required by the platform: get_move(fen, time_left_ms) -> UCI.
Everything heavy happens at import inside the 90 s init budget: numba
compilation of the whole engine via a warmup search, and table construction.
A python-chess legality check wraps every result so no failure path can
return an illegal move."""

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


def _warmup():
    core.warmup()
    core.parse_fen(core.START_FEN, BB, ST)
    keys = STATE.rep[:1].copy()
    keys[0] = BB[core.HASH]
    # generous deadline: the first call pays the whole numba compile, and the
    # depth-4 shakedown search must actually run after it
    mv, score, depth, nodes = search.search_position(
        STATE, BB, ST, keys, 1, soft_ms=0, hard_ms=60000, max_depth=4)
    assert depth == 4 and nodes > 0, "warmup search did not complete"
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
