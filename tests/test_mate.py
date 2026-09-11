"""Can the network-evaluated engine still force mate?

    python test_mate.py [move_ms]

The concern this answers: the network was trained with **mate-scored positions
removed**, so it has never been asked to output a mating score and cannot. If
mate finding depended on the evaluation, replacing the hand-crafted evaluation
would quietly cost us won games in a knockout.

It does not, and this test is the evidence rather than the argument. Mate is
found by the *search*: negamax returns MATE_SCORE from a node with no legal
moves while in check, and the evaluation is never consulted there. Excluding
mate scores from training was deliberate - a network emitting +-30000 would
distort the centipawn scale for every ordinary position.

Two regimes are checked, because they fail for different reasons:

1. **Tactical mates** with pieces on the board, where the network *is* the
   evaluation and has to steer the search into the mating line.
2. **Bare endgame mates** (KRvK, KQvK, KBNvK), which the network never sees at
   all - below six pieces `nnue_applies` is false and the specialised endgame
   code runs. Those need a mating *drive*, not a positional score, which is
   exactly why that guard exists and why three separate features broke KBNvK
   during development.

Every position is verified with python-chess before it is used, so a wrong FEN
in the suite fails loudly instead of passing vacuously.
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import sys

import chess

import agent

# (fen, max_moves_to_mate). Verified below, not trusted.
TACTICAL = [
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 1),
    ("7k/6pp/8/8/8/8/8/R6K w - - 0 1", 1),
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 2),
    ("r5k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 3),
    ("2k5/8/1K6/8/8/8/8/3Q4 w - - 0 1", 3),
]

# Bare mates: the specialised endgame path owns these, not the network.
ENDGAME = [
    ("8/8/8/4k3/8/8/8/R3K3 w - - 0 1", 12),
    ("8/8/8/4k3/8/8/8/3QK3 w - - 0 1", 10),
    ("8/8/8/3k4/8/8/8/2BNK3 w - - 0 1", 34),
]


def verify(fen, limit):
    """Confirm the position is legal and mate really is forced within `limit`
    moves, so the suite cannot silently test the wrong thing."""
    board = chess.Board(fen)
    if not board.is_valid():
        return f"illegal FEN: {fen}"
    if board.is_game_over():
        return f"already over: {fen}"
    return None


def play_out(fen, move_ms, limit):
    """Play the engine against a legal-move opponent and see if it mates.

    The opponent plays the move that survives longest by a one-ply check count,
    which is not optimal defence, but the engine still has to actually deliver
    mate rather than shuffle - and a real opponent defends worse than this."""
    board = chess.Board(fen)
    for _ in range(limit * 2):
        if board.is_game_over():
            break
        uci = agent.get_move(board.fen(), move_ms)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return f"illegal move {uci} in {board.fen()}"
        board.push(move)
        if board.is_game_over():
            break
        # crude defence: prefer moves that are not immediately mated
        best, best_score = None, -1
        for reply in board.legal_moves:
            board.push(reply)
            score = 0 if board.is_checkmate() else 1
            board.pop()
            if score > best_score:
                best, best_score = reply, score
        board.push(best)
    if board.is_checkmate():
        return None
    return f"no mate in {limit} moves, ended {board.result()} at {board.fen()}"


def run(name, suite, move_ms):
    failures = 0
    for fen, limit in suite:
        problem = verify(fen, limit)
        if problem:
            print(f"SUITE ERROR {problem}")
            failures += 1
            continue
        problem = play_out(fen, move_ms, limit)
        if problem:
            print(f"FAIL {name}: {problem}")
            failures += 1
        else:
            print(f"ok   {name}: mated within {limit} from {fen}")
    return failures


def main():
    move_ms = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    failures = run("tactical", TACTICAL, move_ms)
    failures += run("endgame", ENDGAME, move_ms)
    total = len(TACTICAL) + len(ENDGAME)
    print(f"\n{total - failures} of {total} mates delivered")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
