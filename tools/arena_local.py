"""Local strength gate: our agent vs a 2-ply minimax baseline.

Run: python arena_local.py [games] [base_ms]

The baseline mirrors the official starter's minimax bar (material + mobility,
two plies, deterministic tie-break). Referee follows platform rules: wall
clock per move, increment after the move, illegal move or flag loses,
python-chess adjudicates draws. Beating this decisively is the phase 2 bar;
real A/B testing between our own versions uses longer controls.
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import sys
import time

import chess

PIECE_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
               chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
MATE = 10 ** 6


def _material(board):
    score = 0
    for piece_type, value in PIECE_VALUE.items():
        score += value * len(board.pieces(piece_type, board.turn))
        score -= value * len(board.pieces(piece_type, not board.turn))
    return score


def _negamax(board, depth):
    moves = list(board.legal_moves)
    if not moves:
        return -MATE if board.is_check() else 0
    if depth == 0:
        return _material(board) + 4 * len(moves)
    best = -MATE * 2
    for mv in moves:
        board.push(mv)
        best = max(best, -_negamax(board, depth - 1))
        board.pop()
    return best


def baseline_move(fen, _time_left_ms):
    board = chess.Board(fen)
    best_score, best = -MATE * 2, None
    for mv in board.legal_moves:
        board.push(mv)
        score = -_negamax(board, 1)
        board.pop()
        if score > best_score:
            best_score, best = score, mv
    return best.uci()


def play_game(white_fn, black_fn, base_ms, increment_ms, start_fen=None):
    board = chess.Board(start_fen) if start_fen else chess.Board()
    clocks = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    movers = {chess.WHITE: white_fn, chess.BLACK: black_fn}
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                return 0.5, outcome.termination.name
            return (1.0 if outcome.winner == chess.WHITE else 0.0), \
                outcome.termination.name
        if board.ply() >= 600:
            return 0.5, "PLY_CAP"
        side = board.turn
        started = time.perf_counter()
        uci = movers[side](board.fen(), int(clocks[side]))
        clocks[side] -= (time.perf_counter() - started) * 1000.0
        if clocks[side] < 0:
            return (0.0 if side == chess.WHITE else 1.0), "FLAG"
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            return (0.0 if side == chess.WHITE else 1.0), "ILLEGAL"
        board.push(mv)
        clocks[side] += increment_ms


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    base_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    import agent

    # rated games start from curated positions, never the standard start
    import openings as opening_book
    book = opening_book.load()

    score = 0.0
    for game in range(games):
        ours_white = game % 2 == 0
        opening = book[(game // 2) % len(book)]
        if ours_white:
            result, term = play_game(agent.get_move, baseline_move, base_ms,
                                     100, opening)
            ours = result
        else:
            result, term = play_game(baseline_move, agent.get_move, base_ms,
                                     100, opening)
            ours = 1.0 - result
        score += ours
        colour = "white" if ours_white else "black"
        print(f"game {game + 1}: ours({colour}) {ours} [{term}]", flush=True)
    print(f"score vs minimax baseline: {score}/{games}")


if __name__ == "__main__":
    main()
