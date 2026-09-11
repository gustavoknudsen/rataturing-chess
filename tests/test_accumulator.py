"""Assert the incremental accumulator equals a full refresh at every node.

    python test_accumulator.py [depth]

An incremental NNUE accumulator is the most dangerous construct in an engine,
because every way it can be wrong produces a *quietly* wrong evaluation rather
than a crash: the search keeps running, the moves stay legal, and the only
symptom is losing games. There is no perft equivalent to catch it.

So this is the perft equivalent. It walks the move tree exactly as
`btc_core.perft` does, maintains the accumulator incrementally down the tree,
and at every single node recomputes it from scratch and compares all 2 * L1
values. A single disagreement anywhere fails the test.

The positions are the standard perft suite precisely because those were chosen
to exercise castling, en passant, promotion and captures - the four cases that
break hand-written incremental updates. `btc_nnue._apply_delta` derives its
changes by XORing piece planes rather than decoding the move, so none of those
should be special at all; this test is what turns "should" into "does".

Both bucket configurations are tested. With king buckets a king move can change
its perspective's bucket or mirror, which invalidates every feature index for
that side and forces a partial rebuild - a branch that simply does not exist in
the unbucketed net, and therefore needs its own coverage.
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import os
import sys

import numpy as np
from numba import njit

import btc_core as core
from btc_core import generate_moves, make_move, unmake
from btc_nnue import load, refresh, update

# Standard perft positions: castling rights, en passant, promotions, captures.
POSITIONS = [
    ("start", core.START_FEN),
    ("kiwipete",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("endgame ep", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("promotions",
     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
    ("promo 2", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
    ("king walk", "8/8/8/4k3/8/2K5/8/8 w - - 0 1"),
]


@njit(cache=False)
def walk(bb, st, undo_bb, undo_st, mls, acc, scratch, depth, ply,
         ft_w, ft_b, l1, table, buckets):
    """Recursive move-tree walk. Returns (nodes, mismatches)."""
    if depth == 0:
        return 1, 0
    nodes = 0
    bad = 0
    count = generate_moves(bb, st, mls[ply])
    for i in range(count):
        if make_move(bb, st, undo_bb, undo_st, ply, mls[ply, i]) == 0:
            continue
        # undo_bb[ply] is the position before the move, bb the position after
        update(acc, ply, ply + 1, undo_bb[ply], bb, ft_w, ft_b, l1,
               table, buckets)
        refresh(bb, ft_w, ft_b, l1, scratch, table, buckets)
        for perspective in range(2):
            for k in range(l1):
                if acc[ply + 1, perspective, k] != scratch[perspective, k]:
                    bad += 1
                    break
        sub_nodes, sub_bad = walk(bb, st, undo_bb, undo_st, mls, acc, scratch,
                                  depth - 1, ply + 1, ft_w, ft_b, l1,
                                  table, buckets)
        nodes += sub_nodes
        bad += sub_bad
        unmake(bb, st, undo_bb, undo_st, ply)
    return nodes, bad


def check(net_path, depth):
    ft_w, ft_b, out_w, out_b, l1, qa, qb, scale, table, buckets = \
        load(net_path)
    bb, st = core.new_board()
    undo_bb, undo_st, mls = core.new_stacks()
    acc = np.zeros((core.MAX_PLY + 1, 2, l1), dtype=np.int16)
    scratch = np.zeros((2, l1), dtype=np.int16)

    failures = 0
    for name, fen in POSITIONS:
        core.parse_fen(fen, bb, st)
        refresh(bb, ft_w, ft_b, l1, scratch, table, buckets)
        acc[0] = scratch
        nodes, bad = walk(bb, st, undo_bb, undo_st, mls, acc, scratch,
                          depth, 0, ft_w, ft_b, l1, table, buckets)
        status = "ok  " if bad == 0 else "FAIL"
        print(f"  {status} {name:12s} {nodes:9,} nodes, "
              f"{bad} accumulator mismatches")
        failures += bad
    return failures


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    # Any candidate net must be walked before it is trusted, so the list is
    # overridable rather than fixed: BTC_TEST_NETS=path1,path2. The defaults
    # are the shipped net and a bucketed one, which are the two shapes the
    # bucketed code path has to get right.
    override = os.environ.get("BTC_TEST_NETS", "")
    if override:
        nets = [(f"net {i + 1}", p) for i, p in enumerate(override.split(","))]
    else:
        _SRC = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            _os.pardir, "src")
        nets = [("no buckets", _os.path.join(_SRC, "net.npz")),
                ("4 buckets ", "D:/chess_nnue/sw_buckets/net.npz")]
    total = 0
    for label, path in nets:
        print(f"{label} ({path}), depth {depth}:")
        total += check(path, depth)
    if total:
        print(f"\nFAIL: {total} nodes disagreed with a full refresh")
        return 1
    print("\nPASS: incremental accumulator matches a full refresh at every node")
    return 0


if __name__ == "__main__":
    sys.exit(main())
