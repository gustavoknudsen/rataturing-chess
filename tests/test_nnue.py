"""Assert the engine's numba inference equals the trainer's reference exactly.

    python test_nnue.py [net.npz] [data_dir] [n]

This is the test that makes the network trustworthy. It is not a tolerance
check: quantised integer inference is exactly reproducible, so the two
implementations must agree on every position or one of them is wrong.

It covers the two independent things that can be wrong:

1. **The arithmetic** - accumulator, SCReLU, output scaling, the side-to-move
   ordering, and the int64 promotion in the output sum.
2. **The feature derivation**, which is the subtler one. nnue_gate.py reads the
   feature indices the trainer actually consumed, straight out of the dataset.
   btc_nnue.py derives them from the engine's bitboards. If the engine's piece
   order, square order or perspective flip disagreed with the trainer's by so
   much as one plane, the network would still return plausible-looking numbers
   and simply play badly. Comparing the two paths on the same positions is what
   rules that out.
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "training"))


import sys

import numpy as np

import btc_core as core
import nnue_data
import nnue_gate
from btc_nnue import load, nnue_eval

PAD = 65535

# Asymmetric on purpose. A mirrored or transposed feature bug survives any
# position that happens to be symmetric, so every one of these breaks at least
# one symmetry: castling rights on one side only, an en passant square, pieces
# off their home ranks, and a promotion-adjacent pawn.
FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 2",
    "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 99 50",
]


def build_board(feats_row, count):
    """Bitboards from stored feature indices, the engine's own layout."""
    bb = np.zeros(16, dtype=np.uint64)
    for i in range(count):
        index = int(feats_row[i])
        if index == PAD:
            continue
        bb[index // 64] |= np.uint64(1) << np.uint64(index % 64)
    for piece in range(6):
        bb[12] |= bb[piece]
        bb[13] |= bb[piece + 6]
    bb[14] = bb[12] | bb[13]
    return bb


def engine_features(bb):
    """The feature index multiset btc_nnue.refresh would walk, from real
    engine bitboards."""
    found = []
    for piece in range(12):
        bits = int(bb[piece])
        while bits:
            square = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            found.append(64 * piece + square)
    return sorted(found)


def check_layout():
    """The engine's board representation must agree with the trainer's.

    test_inference below builds its bitboards *from* dataset feature indices,
    so it round-trips and cannot see a disagreement here. This parses the same
    FEN twice - once with the engine's parse_fen, once with the extractor's
    parse_board - and compares the feature sets. If the engine's piece order or
    square orientation differed from the one the network was trained on, the
    network would return plausible numbers for the wrong board and simply play
    badly, with nothing anywhere reporting an error."""
    bb, st = core.new_board()
    row = np.empty(nnue_data.MAX_PIECES, dtype=np.uint16)
    board = np.empty(64, dtype=np.int8)
    for fen in FENS:
        core.parse_fen(fen, bb, st)
        row[:] = PAD
        count = nnue_data.parse_board(fen.split()[0], row, board)
        trainer = sorted(int(v) for v in row[:count])
        if engine_features(bb) != trainer:
            print(f"FAIL: layout mismatch on {fen}")
            return 1
    print(f"PASS: engine bitboards match the extractor on {len(FENS)} FENs")
    return 0


def mirror(bb):
    """Swap colours and flip the board vertically."""
    out = np.zeros(16, dtype=np.uint64)
    for piece in range(12):
        bits = int(bb[piece])
        while bits:
            square = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            out[(piece + 6) % 12] |= np.uint64(1) << np.uint64(square ^ 56)
    for piece in range(6):
        out[12] |= out[piece]
        out[13] |= out[piece + 6]
    out[14] = out[12] | out[13]
    return out


def check_mirror(net):
    """A side-to-move-relative net must score a colour-mirrored position
    identically. Mirroring swaps the two accumulators and flips the side to
    move, so the concatenation order is unchanged and the output must be exactly
    equal - not approximately. This is the sharpest available test of the
    perspective flip: a wrong `square ^ 56` or a swapped colour plane breaks it,
    while both survive the reference comparison, which shares the same
    formula."""
    ft_w, ft_b, out_w, out_b, l1, qa, qb, scale, table, buckets = net
    bb, st = core.new_board()
    for fen in FENS:
        core.parse_fen(fen, bb, st)
        side = int(st[core.SIDE])
        plain = nnue_eval(bb, side, ft_w, ft_b, out_w, out_b,
                          l1, qa, qb, scale, table, buckets)
        flipped = nnue_eval(mirror(bb), 1 - side, ft_w, ft_b, out_w, out_b,
                            l1, qa, qb, scale, table, buckets)
        if plain != flipped:
            print(f"FAIL: mirror {plain} vs {flipped} on {fen}")
            return 1
    print(f"PASS: colour-mirror invariant on {len(FENS)} FENs")
    return 0


def main():
    net_path = sys.argv[1] if len(sys.argv) > 1 \
        else "D:/chess_nnue/net256/net.npz"
    data_dir = sys.argv[2] if len(sys.argv) > 2 else "D:/chess_nnue/data50m"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    net = load(net_path)
    ft_w, ft_b, out_w, out_b, l1, qa, qb, scale, table, buckets = net
    reference = nnue_gate.load_net(net_path)

    failures = check_layout() + check_mirror(net)

    feats = np.load(f"{data_dir}/feats.npy", mmap_mode="r")
    counts = np.load(f"{data_dir}/counts.npy", mmap_mode="r")
    stm = np.load(f"{data_dir}/stm.npy", mmap_mode="r")

    # the tail is the held-out slice, so this never reads a trained-on position
    lo = len(counts) - n

    mismatches = 0
    spread = 0
    for row in range(lo, lo + n):
        count = int(counts[row])
        side = int(stm[row])
        bb = build_board(feats[row], count)
        got = nnue_eval(bb, side, ft_w, ft_b, out_w, out_b,
                        l1, qa, qb, scale, table, buckets)
        want = nnue_gate.net_eval(feats[row], count, side, reference)
        if got != want:
            mismatches += 1
            if mismatches <= 5:
                print(f"row {row}: engine {got} reference {want}")
        spread = max(spread, abs(int(want)))

    print(f"{n} held-out positions, L1={l1}, largest |eval| {spread} cp")
    if mismatches:
        print(f"FAIL: {mismatches} of {n} disagree")
        failures += 1
    else:
        print("PASS: engine inference is bit-identical to the reference")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
