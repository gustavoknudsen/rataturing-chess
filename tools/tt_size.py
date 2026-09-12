"""Does a bigger transposition table search fewer nodes?

    .venv/Scripts/python.exe tools/tt_size.py [depth]

The platform gives 2 GB and the engine uses 682 MB of it, of which the table
is 256 MiB. That leaves 1.37 GB idle. A bigger table is not a heuristic
trade: it retains strictly more, and the only cost is memory. But our match
harness runs at 10 s + 0.04 s, where a search never comes close to filling
16.7M entries, so a match cannot show the benefit.

Fixed-depth node count can. Reaching the same depth in fewer nodes means more
of the tree was answered from the table, and unlike a timing measurement it is
deterministic, so it stays valid while the machine is busy with something
else.

Positions are searched in sequence against one table per size, without
clearing between them, because that is what a game does: the table carries
entries from positions that have already gone by.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import btc_nrt  # noqa: F401
import btc_core as core
import btc_search as search

# Midgame positions with enough material left that the tree is large.
POSITIONS = [
    "r1bqkb1r/pp1n1ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 7",
    "r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9",
    "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 9",
    "2rq1rk1/pb1nbppp/1p2pn2/2pp4/2PP4/1PN1PN2/PB2BPPP/R2Q1RK1 w - - 0 11",
    "r1bq1rk1/1p2bppp/p1nppn2/8/3NP3/1BN1B3/PPP2PPP/R2Q1RK1 w - - 0 10",
    "rn1q1rk1/pb2bppp/1p2pn2/2pp4/2PP4/1P2PN2/PB1NBPPP/R2Q1RK1 w - - 0 10",
]

SIZES = [1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 25]


def run(tt_entries, depth):
    """Total nodes to reach `depth` on every position, sharing one table."""
    state = search.SearchState(tt_entries)
    total = 0
    started = time.perf_counter()
    for fen in POSITIONS:
        bb, st = core.new_board()
        core.parse_fen(fen, bb, st)
        keys = np.zeros(8, dtype=np.uint64)
        keys[0] = bb[core.HASH]
        _, _, reached, nodes = search.search_position(
            state, bb, st, keys, 1, 0, 3600000, max_depth=depth)
        if reached != depth:
            print("  WARNING: reached depth {} not {}".format(reached, depth))
        total += nodes
    return total, time.perf_counter() - started


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print("nodes to reach depth {} over {} positions".format(
        depth, len(POSITIONS)))
    print("")

    # Compile before measuring, so the first size does not pay for it.
    warm = search.SearchState(1 << 20)
    bb, st = core.new_board()
    core.parse_fen(POSITIONS[0], bb, st)
    keys = np.zeros(8, dtype=np.uint64)
    keys[0] = bb[core.HASH]
    search.search_position(warm, bb, st, keys, 1, 0, 3600000, max_depth=6)
    del warm

    baseline = None
    for size in SIZES:
        nodes, elapsed = run(size, depth)
        if baseline is None:
            baseline = nodes
            delta = ""
        else:
            delta = "  {:+.2f}% nodes".format((nodes - baseline) * 100.0 / baseline)
        print("  table {:>5} MiB   {:>12,} nodes   {:>6.1f}s{}".format(
            size * 16 // (1 << 20), nodes, elapsed, delta))
    print("")
    print("Fewer nodes for the same depth means more of the tree came from the")
    print("table. Timing here is unreliable if the machine is busy; the node")
    print("counts are not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
