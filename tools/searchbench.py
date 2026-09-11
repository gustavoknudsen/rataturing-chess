"""Fixed-depth search benchmark. Run: python searchbench.py [depth]

Reports total nodes and total time over a fixed position set. Nodes to a fixed
depth are deterministic and carry no clock, so they are the metric to trust for
anything that claims to shrink the tree; time is reported alongside but is only
meaningful against a run made back to back on the same machine state.

The transposition table is zeroed between positions so a run does not depend on
the order or on what the previous position left behind.

Use --time to A/B a pure speed change: nodes must be identical, time down.
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import sys
import time

import numpy as np

import btc_nrt  # noqa: F401  - must precede any njit compile
import btc_core as core
import btc_search as search

# Fourteen positions: five opening/early middlegame, five middlegame tactics
# and structure, four endgames. Taken from the classic public benchmark sets
# so the mix is not tuned to our own engine.
POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbq1rk1/pp2ppbp/2pp1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8",
    "r2q1rk1/1b1nbppp/pp1ppn2/6B1/2PNP3/2N2P2/PP2B1PP/R2Q1RK1 w - - 0 12",
    "2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1",
    "4rrk1/pp1n1pp1/2p1b2p/3p4/3P1P2/1QPB1N1P/PP4P1/R4RK1 w - - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 0 9",
    "3r1rk1/p3qppp/1pnbbn2/2p5/2P5/1PN1PN2/PB1QBPPP/3R1RK1 w - - 0 15",
    "r4rk1/1b2qppp/p1n1p3/1pn5/3P4/P1NBPN2/1PQ2PPP/2R2RK1 w - - 0 16",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "8/3k4/8/8/8/4P3/3K4/8 w - - 0 1",
    "6k1/5p1p/6p1/8/8/6P1/5P1P/R5K1 w - - 0 1",
    "6k1/5ppp/4b3/8/8/4B3/5PPP/6K1 w - - 0 1",
    "r5k1/pp3ppp/8/3p4/3P4/2P5/PP3PPP/R5K1 w - - 0 1",
]


def run(depth, positions=None, history=0):
    """Returns (total_nodes, total_seconds, per_position).

    `history` seeds that many prior game positions before the search root.
    They never match anything, so they cannot change the result, but every
    node pays whatever the repetition check costs to walk them - which is the
    cost a real game at move `history` actually pays. Benchmarking from a
    one-move history hides that entirely."""
    positions = positions or POSITIONS
    state = search.SearchState()
    bb, st = core.new_board()
    keys = np.zeros(history + 8, dtype=np.uint64)
    if history:
        rng = np.random.default_rng(12345)
        keys[:history] = rng.integers(1, 2 ** 63, size=history, dtype=np.int64)

    per_position = []
    total_nodes = 0
    total_seconds = 0.0
    for fen in positions:
        state.tt_key[:] = 0
        state.tt_data[:] = 0
        state.main_hist[:] = 0
        state.cap_hist[:] = 0
        state.cont_hist[:] = 0
        state.counters[:] = 0
        core.parse_fen(fen, bb, st)
        keys[history] = bb[core.HASH]
        started = time.perf_counter()
        mv, score, reached, nodes = search.search_position(
            state, bb, st, keys, history + 1, soft_ms=0, hard_ms=3_600_000,
            max_depth=depth)
        elapsed = time.perf_counter() - started
        per_position.append((fen, nodes, score, reached, elapsed))
        total_nodes += nodes
        total_seconds += elapsed
    return total_nodes, total_seconds, per_position


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    history = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    # Compile everything before timing anything.
    run(2, history=history)

    best = None
    for _ in range(repeats):
        total_nodes, total_seconds, per_position = run(depth, history=history)
        if best is None or total_seconds < best[1]:
            best = (total_nodes, total_seconds, per_position)

    total_nodes, total_seconds, per_position = best
    for fen, nodes, score, reached, elapsed in per_position:
        print(f"{nodes:10d}  d{reached:<3d} {score:6d}  {elapsed:6.2f}s  "
              f"{fen.split(' ')[0][:32]}")
    print(f"\ndepth {depth}, game history {history}")
    print(f"nodes {total_nodes}")
    print(f"time  {total_seconds:.2f}s")
    print(f"nps   {total_nodes / total_seconds:,.0f}")


if __name__ == "__main__":
    main()
