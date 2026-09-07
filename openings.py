"""Opening positions for local testing. Not shipped in the submission.

Rated games start from a curated set that is not published; the eight in the
starter harness are a published sample. Testing only on those eight makes
results correlate once a match passes sixteen games, and tunes the engine to
eight positions. This module generates a larger set of balanced, deduplicated
positions the same way opening books for engine testing are built: play a few
random plies, then keep the position only if a shallow search says neither
side is already better than a pawn.

Run directly to (re)generate the cache:
    python openings.py [count]
"""

import json
import os
import random
import sys

import chess

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "openings.json")

# The eight published in the starter harness. Kept first so results stay
# comparable with earlier matches, and because they are a real sample of the
# curated set's style.
SAMPLE = [
    "r1bqk2r/pp1pppbp/2n2np1/2p5/2P5/2N1PNP1/PP1P1PBP/R1BQK2R b KQkq - 0 6",
    "r1b1k2r/pp2nppp/2n1p3/q1ppP3/P2P4/2P2N2/2PB1PPP/R2QKB1R b KQkq - 4 9",
    "rnbqkb1r/pp3ppp/2p5/1B1p4/3Pn3/5N2/PPP2PPP/RNBQK2R w KQkq - 0 7",
    "r1bq1rk1/pppp1ppp/2n2n2/1Bb5/3NP3/2P5/PP3PPP/RNBQ1RK1 w - - 3 8",
    "rnbqk2r/pp2ppbp/6p1/2p5/3PP3/2P1BN2/P4PPP/R2QKB1R b KQkq - 1 8",
    "r1bqk2r/pp1n1ppp/2n1p3/2bpP3/5P2/2NB4/PPP3PP/R1BQK1NR w KQkq - 0 8",
    "1rbqk1nr/pp2ppbp/2np2p1/2p5/P3P3/2NP2P1/1PP1NPBP/R1BQK2R b KQk - 0 7",
    "r1bqkb1r/pp3ppp/2np4/1N1Pp3/8/8/PPP2PPP/R1BQKB1R b KQkq - 0 8",
]

BALANCE_MARGIN = 90


def _shallow_score(fen):
    """Side-to-move score from a short fixed-depth search."""
    import btc_core as core
    import btc_search as se

    state = se.SearchState(tt_entries=1 << 14)
    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    keys = [int(bb[core.HASH])]
    import numpy as np
    _, score, _, _ = se.search_position(
        state, bb, st, np.array(keys, dtype=np.uint64), 1, 0, 3000,
        max_depth=6)
    return score


def generate(count, seed=1, min_plies=6, max_plies=12):
    """Balanced positions from short random walks, deduplicated."""
    rng = random.Random(seed)
    seen = set()
    out = []
    for fen in SAMPLE:
        key = " ".join(fen.split()[:4])
        seen.add(key)
        out.append(fen)
    attempts = 0
    while len(out) < count and attempts < count * 60:
        attempts += 1
        board = chess.Board()
        plies = rng.randrange(min_plies, max_plies + 1)
        ok = True
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                ok = False
                break
            board.push(rng.choice(moves))
        if not ok or board.is_game_over():
            continue
        fen = board.fen()
        key = " ".join(fen.split()[:4])
        if key in seen:
            continue
        if abs(_shallow_score(fen)) > BALANCE_MARGIN:
            continue
        seen.add(key)
        out.append(fen)
    return out


def load(count=None):
    """Cached opening set, falling back to the published sample."""
    if os.path.exists(CACHE):
        with open(CACHE, encoding="ascii") as handle:
            fens = json.load(handle)
    else:
        fens = list(SAMPLE)
    if count is not None:
        fens = fens[:count]
    return fens


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    fens = generate(count)
    with open(CACHE, "w", encoding="ascii") as handle:
        json.dump(fens, handle, indent=1)
    print(f"wrote {len(fens)} openings to {CACHE}")
    print(f"first non-sample: {fens[len(SAMPLE)] if len(fens) > len(SAMPLE) else 'none'}")


if __name__ == "__main__":
    main()
