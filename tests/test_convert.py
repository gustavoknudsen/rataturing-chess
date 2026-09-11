"""Endgame conversion. Run: python test_convert.py

The engine plays out won endings against a defender that maximises survival.
This is the acceptance test that matters for specialised endgame knowledge:
not "does the search see mate at depth 12", but "does it actually mate before
the fifty-move rule the referee claims automatically".
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import os
import sys
import time

import chess
import numpy as np

import btc_core as core
import btc_search as se

# mate distances with best play are well under these; the limits are the
# practical bar, not the theoretical one
POSITIONS = [
    ("KRvK", "7k/8/8/8/8/8/8/R3K3 w - - 0 1", 32),
    ("KQvK", "7k/8/8/8/8/8/8/3QK3 w - - 0 1", 20),
    ("KRRvK", "4k3/8/8/8/8/8/8/R2K3R w - - 0 1", 20),
    ("KQRvK", "7k/8/8/8/8/8/8/R2QK3 w - - 0 1", 16),
    ("KBNvK", "7k/8/8/8/8/8/8/1NB1K3 w - - 0 1", 50),
]

passed = 0
failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name} {detail}")


def _defender_move(board):
    """Hardest practical defence: keep the defending king as far from the
    edge, and from the attacking king, as possible."""
    defender = board.turn
    best = None
    best_key = None
    for mv in board.legal_moves:
        board.push(mv)
        king = board.king(defender)
        attacker_king = board.king(not defender)
        rank, file = chess.square_rank(king), chess.square_file(king)
        edge = min(rank, 7 - rank) + min(file, 7 - file)
        dist = chess.square_distance(king, attacker_king)
        board.pop()
        key = (edge, dist)
        if best_key is None or key > best_key:
            best_key = key
            best = mv
    return best


# The harness used to pass rep_count = 1, which hands the engine no history:
# it cannot see that it is repeating, and the threefold is then detected by the
# harness board rather than avoided by the search. A real game passes the whole
# history through GameTracker.
NO_HISTORY = os.environ.get("BTC_CONVERT_NOHIST") == "1"


def position_hash(fen):
    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    return bb[core.HASH]


def _engine_move(state, fen, depth, history):
    """Fixed depth, not fixed time: a wall-clock budget makes this test depend
    on machine load, which produced spurious pass/fail flips."""
    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    if NO_HISTORY:
        keys = np.array([bb[core.HASH]], dtype=np.uint64)
        count = 1
    else:
        keys = np.array(history, dtype=np.uint64)
        count = len(history)
    mv, _, _, _ = se.search_position(state, bb, st, keys, count, 0, 600000,
                                     max_depth=depth)
    return core.move_to_uci(mv)


def _warm(state):
    bb, st = core.new_board()
    core.parse_fen(core.START_FEN, bb, st)
    keys = np.array([bb[core.HASH]], dtype=np.uint64)
    se.search_position(state, bb, st, keys, 1, 0, 600000, max_depth=4)


def convert(name, fen, limit_moves, depth=10):
    state = se.SearchState(tt_entries=1 << 18)
    board = chess.Board(fen)
    moves_played = 0
    # Every position of the game so far, current one last - the same thing
    # GameTracker hands the engine in a real game.
    history = [position_hash(board.fen())]
    while moves_played < limit_moves:
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            return False, moves_played, outcome.termination.name
        uci = _engine_move(state, board.fen(), depth, history)
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            return False, moves_played, "illegal"
        board.push(mv)
        history.append(position_hash(board.fen()))
        moves_played += 1
        if board.is_checkmate():
            return True, moves_played, "mate"
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            return False, moves_played, outcome.termination.name
        reply = _defender_move(board)
        if reply is None:
            break
        board.push(reply)
        history.append(position_hash(board.fen()))
        if board.is_fifty_moves():
            return False, moves_played, "fifty_moves"
    return False, moves_played, "not converted"


def main():
    state = se.SearchState(tt_entries=1 << 16)
    t0 = time.perf_counter()
    _warm(state)
    # report what the modules actually resolved to, not the env defaults
    import btc_eval as ev
    print(f"warmup {time.perf_counter() - t0:.0f}s  "
          f"endgames={ev.USE_ENDGAMES}  threats={ev.USE_THREATS}  "
          f"history={'off' if NO_HISTORY else 'on'}", flush=True)

    for name, fen, limit in POSITIONS:
        # an illegal fixture (side to move able to capture the enemy king)
        # makes pseudo-legal generation look like an engine bug
        board = chess.Board(fen)
        check(f"{name} fixture is legal", board.is_valid(),
              f"{fen}: {board.status()!r}")
        t0 = time.perf_counter()
        ok, moves, why = convert(name, fen, limit)
        print(f"{name:8s} {'MATE in ' + str(moves) if ok else 'FAILED (' + why + ')':22s} "
              f"limit {limit:3d}  {time.perf_counter() - t0:5.1f}s", flush=True)
        check(f"converts {name}", ok, f"{why} after {moves} moves")

    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
