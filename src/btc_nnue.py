"""Quantised NNUE inference for the engine, in numba nopython mode.

Mirrors the numpy reference in nnue_gate.py exactly. test_nnue.py asserts the
two agree bit for bit on real positions, so a disagreement between what the
network scored in training and what the engine plays is localised here rather
than hidden inside the search.

`refresh` rebuilds both accumulators from the bitboards and seeds ply 0;
`update` applies one move incrementally and is what the search uses per node.

**Weights are arguments, not module globals.** That lets test_nnue.py load a
different network than the engine imported and compare them in one process,
which is what makes the correctness test possible.
"""

import os

import numpy as np
from numba import int64, njit

from btc_core import ONE, ZERO, OCC_A, count_bits, lsb

# Below this the hand-crafted evaluation stops returning material and starts
# returning a mating drive (KPK, KBNvK and friends). The network was never
# trained on that scale and would talk the search out of winning those endings,
# so the specialised path keeps them. Lowered from 6 to 4 when the material-only
# fallback replaced the hand-crafted evaluation: the net was trained on 4+ piece
# positions and evaluates them well, while endgame_probe still owns the bare
# mates below that.
NNUE_MIN_PIECES = 4


def load(path):
    """Read a net.npz produced by nnue_train.py. Returns a tuple ready to pass
    straight into nnue_eval, with dtypes pinned so numba compiles one
    specialisation rather than one per caller.

    The king-bucket table is stored *in the file* rather than duplicated here.
    Trainer and engine disagreeing about it would not crash - it would quietly
    read the wrong 768-row block and evaluate a different position - so there is
    only ever one copy of it. Nets trained before buckets existed carry neither
    field and load as a single bucket with an all-zero table, which makes the
    bucketed code path bit-identical to the unbucketed one."""
    data = np.load(path)
    files = data.files
    buckets = int(data["buckets"]) if "buckets" in files else 1
    if "bucket_table" in files:
        table = np.ascontiguousarray(data["bucket_table"], dtype=np.int32)
    else:
        table = np.zeros(64, dtype=np.int32)
    # Output weights are always handed to numba as (out_buckets, 2 * l1) and
    # the bias as (out_buckets,), so there is one shape to compile against.
    # Nets trained before output buckets stored a flat vector and a scalar;
    # reshaping them to a single bucket makes the bucketed path bit-identical
    # for them rather than a second code path.
    out_w = np.atleast_2d(np.ascontiguousarray(data["out_w"], dtype=np.int16))
    out_b = np.atleast_1d(np.ascontiguousarray(data["out_b"], dtype=np.int32))
    return (np.ascontiguousarray(data["ft_w"], dtype=np.int16),
            np.ascontiguousarray(data["ft_b"], dtype=np.int32),
            out_w, out_b, int(data["l1"]),
            int(data["qa"]), int(data["qb"]), int(data["scale"]),
            table, buckets)


def find_net():
    """Locate the shipped network next to this module, or None."""
    path = os.environ.get("BTC_NET")
    if path and os.path.exists(path):
        return path
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "net.npz")
    return local if os.path.exists(local) else None


@njit(cache=False, fastmath=True, error_model='numpy')
def refresh(bb, ft_w, ft_b, l1, acc, table, buckets):
    """Rebuild both accumulators from the bitboards. acc is int16 (2, l1).

    Each perspective offsets its features by its own king's bucket, which is
    what preserves colour-mirror symmetry: mirroring swaps which king each
    perspective sees, so the accumulators swap and the score is unchanged. An
    all-zero table (a pre-bucket net) reduces this to the unbucketed case.

    Feature index is 64 * piece + square with square 0 = a8, matching both the
    engine's square order and the trainer's. The black perspective mirrors
    vertically and swaps the colour plane; `square ^ 56` inverts the rank bits.
    """
    for k in range(l1):
        acc[0, k] = ft_b[k]
        acc[1, k] = ft_b[k]
    white_king = lsb(bb[5])
    black_king = lsb(bb[11]) ^ 56
    # Horizontal mirror onto files a-d, each perspective keyed on its own king.
    # `^ 7` inverts the file bits. Gated on `buckets` explicitly rather than
    # inferred from the table, so a net trained before buckets existed takes a
    # provably unchanged path instead of one that depends on table contents.
    white_flip = 0
    black_flip = 0
    white_offset = 0
    black_offset = 0
    if buckets > 1:
        if (white_king & 7) >= 4:
            white_flip = 7
        if (black_king & 7) >= 4:
            black_flip = 7
        white_offset = 768 * table[white_king ^ white_flip]
        black_offset = 768 * table[black_king ^ black_flip]
    for piece in range(12):
        bits = bb[piece]
        colour = piece // 6
        base_black = 384 * (1 - colour) + 64 * (piece % 6)
        while bits:
            square = lsb(bits)
            bits &= bits - ONE
            white_row = white_offset + 64 * piece + (square ^ white_flip)
            black_row = black_offset + base_black \
                + ((square ^ 56) ^ black_flip)
            for k in range(l1):
                acc[0, k] += ft_w[white_row, k]
                acc[1, k] += ft_w[black_row, k]


@njit(cache=False, fastmath=True, error_model='numpy')
def propagate(acc, side, out_w, out_b, l1, qa, qb, scale, bucket):
    """SCReLU hidden layer and the single output, in centipawns.

    The side to move's accumulator goes first. That ordering is what makes the
    output side-to-move relative, which is what negamax wants, and it is what
    lets the network express tempo at all.

    int64 accumulation. The worst term is qa * qa * 1.98 * qb summed over 2 * l1,
    about 4.2e9, which overflows int32.

    int32 blocking and float64 accumulation were both tried, at three widths.
    All agree to the last unit and all are slower, by more the wider the net -
    ns per evaluation, int64 / f64 / f64-split / int32-blocked:

        L1=512    1200 / 1358 / 1410 / 1520
        L1=1024   2239 / 2291 / 2669 / 2583
        L1=2048   3340 / 5933 / 7332 / 8560

    Measure before changing this."""
    first = int64(0) if side == 0 else int64(1)
    second = int64(1) - first
    total = int64(0)
    for k in range(l1):
        a = int64(acc[first, k])
        if a < 0:
            a = int64(0)
        elif a > qa:
            a = int64(qa)
        total += a * a * int64(out_w[bucket, k])
        b = int64(acc[second, k])
        if b < 0:
            b = int64(0)
        elif b > qa:
            b = int64(qa)
        total += b * b * int64(out_w[bucket, l1 + k])
    total = total // int64(qa) + int64(out_b[bucket])
    return total * int64(scale) // (int64(qa) * int64(qb))


@njit(cache=False, fastmath=True, error_model='numpy')
def _out_bucket(bb, out_buckets):
    """Piece count -> output bucket. Must match nnue_train.out_bucket_of.

    A single output layer has to map the accumulator to a score across every
    phase of the game, but a pawn up in a rook ending is not worth what a pawn
    up in a middlegame is. Separate output vectors let the net say so, and cost
    nothing: the dot product is the same length either way, only the weight
    vector read differs."""
    if out_buckets <= 1:
        return 0
    index = (count_bits(bb[OCC_A]) - 1) * out_buckets // 32
    if index < 0:
        return 0
    if index >= out_buckets:
        return out_buckets - 1
    return index


@njit(cache=False, fastmath=True, error_model='numpy')
def nnue_eval(bb, side, ft_w, ft_b, out_w, out_b, l1, qa, qb, scale,
              table, buckets):
    """Static evaluation in standard centipawns, side-to-move relative.

    The caller converts to the engine's internal scale, where a pawn is 126
    rather than 100; see NET_UNITS in btc_eval.

    The accumulator is allocated here rather than reused from a module-level
    buffer: numba 0.67 types a global array as readonly, so a shared scratch
    cannot be written from nopython mode.

    **int16, not int32.** The worst reachable accumulator is bounded at 16,384
    against an int16 ceiling of 32,767, and int16 doubles the lanes an AVX2
    register holds for the refresh loop."""
    acc = np.empty((2, l1), dtype=np.int16)
    refresh(bb, ft_w, ft_b, l1, acc, table, buckets)
    return propagate(acc, side, out_w, out_b, l1, qa, qb, scale,
                     _out_bucket(bb, out_w.shape[0]))


@njit(cache=False, fastmath=True, error_model='numpy')
def nnue_applies(bb):
    """False in the endgames the specialised evaluation owns."""
    return count_bits(bb[OCC_A]) >= NNUE_MIN_PIECES


@njit(cache=False, fastmath=True, error_model='numpy')
def _persp_key(king_square, buckets, table):
    """Bucket and mirror state for one perspective, packed into one int.

    Two positions share a key exactly when their features are indexed the same
    way, so comparing keys before and after a move is the test for whether that
    perspective can be updated incrementally or must be rebuilt."""
    if buckets <= 1:
        return 0
    flip = 7 if (king_square & 7) >= 4 else 0
    return 2 * table[king_square ^ flip] + (1 if flip else 0)


@njit(cache=False, fastmath=True, error_model='numpy')
def _refresh_side(bb, perspective, ft_w, ft_b, l1, acc, row, table, buckets):
    """Rebuild one perspective's accumulator into acc[row, perspective]."""
    for k in range(l1):
        acc[row, perspective, k] = ft_b[k]
    if perspective == 0:
        king = lsb(bb[5])
    else:
        king = lsb(bb[11]) ^ 56
    flip = 0
    offset = 0
    if buckets > 1:
        if (king & 7) >= 4:
            flip = 7
        offset = 768 * table[king ^ flip]
    for piece in range(12):
        bits = bb[piece]
        colour = piece // 6
        base = 384 * (1 - colour) + 64 * (piece % 6)
        while bits:
            square = lsb(bits)
            bits &= bits - ONE
            if perspective == 0:
                index = offset + 64 * piece + (square ^ flip)
            else:
                index = offset + base + ((square ^ 56) ^ flip)
            for k in range(l1):
                acc[row, perspective, k] += ft_w[index, k]


@njit(cache=False, fastmath=True, error_model='numpy')
def _apply_delta(bb_before, bb_after, perspective, king, ft_w, l1, acc,
                 src_row, dst_row, table, buckets):
    """Carry one perspective forward across a move by board difference.

    **This is why incremental update is safe here.** It never decodes the move.
    It XORs each piece plane before against after, which yields exactly the
    squares that changed, and adds or subtracts the corresponding weight rows.
    Castling (two pieces move), en passant (a pawn vanishes from a square the
    move never names), promotion (a piece changes type) and ordinary captures
    all fall out of the difference automatically, because the difference is
    derived from what the board actually did rather than from an interpretation
    of the move encoding. The entire class of special-case bugs that makes
    incremental accumulators dangerous cannot arise."""
    for k in range(l1):
        acc[dst_row, perspective, k] = acc[src_row, perspective, k]
    flip = 0
    offset = 0
    if buckets > 1:
        if (king & 7) >= 4:
            flip = 7
        offset = 768 * table[king ^ flip]
    for piece in range(12):
        changed = bb_before[piece] ^ bb_after[piece]
        if changed == ZERO:
            continue
        colour = piece // 6
        base = 384 * (1 - colour) + 64 * (piece % 6)
        added = changed & bb_after[piece]
        removed = changed & bb_before[piece]
        while added:
            square = lsb(added)
            added &= added - ONE
            if perspective == 0:
                index = offset + 64 * piece + (square ^ flip)
            else:
                index = offset + base + ((square ^ 56) ^ flip)
            for k in range(l1):
                acc[dst_row, perspective, k] += ft_w[index, k]
        while removed:
            square = lsb(removed)
            removed &= removed - ONE
            if perspective == 0:
                index = offset + 64 * piece + (square ^ flip)
            else:
                index = offset + base + ((square ^ 56) ^ flip)
            for k in range(l1):
                acc[dst_row, perspective, k] -= ft_w[index, k]


@njit(cache=False, fastmath=True, error_model='numpy')
def update(acc, src_row, dst_row, bb_before, bb_after, ft_w, ft_b, l1,
           table, buckets):
    """Fill acc[dst_row] from acc[src_row] for the move bb_before -> bb_after.

    A king move that changes its own perspective's bucket or mirror invalidates
    every feature index for that side, so that perspective is rebuilt instead.
    The other perspective is still updated incrementally - the two are
    independent, which is what keeps a king move from costing a full refresh of
    both sides."""
    white_king = lsb(bb_after[5])
    black_king = lsb(bb_after[11]) ^ 56
    if buckets > 1 and _persp_key(lsb(bb_before[5]), buckets, table) \
            != _persp_key(white_king, buckets, table):
        _refresh_side(bb_after, 0, ft_w, ft_b, l1, acc, dst_row, table, buckets)
    else:
        _apply_delta(bb_before, bb_after, 0, white_king, ft_w, l1, acc,
                     src_row, dst_row, table, buckets)
    if buckets > 1 and _persp_key(lsb(bb_before[11]) ^ 56, buckets, table) \
            != _persp_key(black_king, buckets, table):
        _refresh_side(bb_after, 1, ft_w, ft_b, l1, acc, dst_row, table, buckets)
    else:
        _apply_delta(bb_before, bb_after, 1, black_king, ft_w, l1, acc,
                     src_row, dst_row, table, buckets)
