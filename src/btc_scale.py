"""Endgame scale factor, ported from BTC evaluation.h.

Discounts the score of the side that is ahead in endings it cannot actually
convert: material edges with no pawns, opposite-coloured bishops, single-flank
rook endings, wrong rook pawn with the wrong-coloured bishop, and dead-drawn
KP vs KP. The factor runs 0 (dead draw) to 64 (no scaling) and is applied to
the endgame component only, so it is full strength in a pure ending and a
no-op in the opening.

This was missing from the port entirely. Without it the engine believes it is
winning in drawn endings, which distorts every trade decision that leads into
one: on a bare K+P vs K+P position the port scored +1068 where BTC scores +611.
"""

import numpy as np
from numba import njit, uint64

# Jit-only helpers. Suppressing the Python-callable wrappers cuts compile
# time, but calling one of these from Python then crashes the process, so a
# function gets this only once every call site is known to be jitted.
_NOWRAP = {"no_cpython_wrapper": True, "no_cfunc_wrapper": True}


from btc_core import (
    B, BLACK, K, KING_ATTACKS, N, OCC_B, OCC_W, ONE, P, Q, SIDE, WHITE, ZERO,
    count_bits, lsb,
)
from btc_endgame import _non_pawn_material
from btc_evalmasks import (
    BLACK_FORWARD_FILE, BLACK_PASSED, FILE_MASK, FILE_OF, RANK_OF,
    RELATIVE_RANK, WHITE_FORWARD_FILE, WHITE_PASSED,
)
from btc_kpk import probe as kpk_probe

NORMAL = 64


BISHOP_V = 825
KNIGHT_V = 781
ROOK_V = 1276

FILE_A = uint64(int(FILE_MASK[0]))
FILE_B = uint64(int(FILE_MASK[1]))
FILE_G = uint64(int(FILE_MASK[6]))
FILE_H = uint64(int(FILE_MASK[7]))
QUEENSIDE = uint64(int(FILE_MASK[0]) | int(FILE_MASK[1])
                   | int(FILE_MASK[2]) | int(FILE_MASK[3]))
KINGSIDE = uint64(int(FILE_MASK[4]) | int(FILE_MASK[5])
                  | int(FILE_MASK[6]) | int(FILE_MASK[7]))


@njit(cache=False, fastmath=True, **_NOWRAP)
def _msb(bbv):
    """Index of the most significant set bit. Caller guarantees bbv != 0."""
    sq = 0
    v = bbv
    while v:
        sq = lsb(v)
        v &= v - ONE
    return sq


@njit(cache=False, fastmath=True, **_NOWRAP)
def _square_colour(sq):
    return (sq & 1) ^ ((sq >> 3) & 1)


@njit(cache=False, fastmath=True, **_NOWRAP)
def _chebyshev(a, b):
    dr = RANK_OF[a] - RANK_OF[b]
    df = FILE_OF[a] - FILE_OF[b]
    if dr < 0:
        dr = -dr
    if df < 0:
        df = -df
    return dr if dr > df else df


@njit(cache=False, fastmath=True, **_NOWRAP)
def _pawns_and_kings(bb, strong):
    weak = 1 - strong
    s_pawns = bb[P] if strong == WHITE else bb[P + 6]
    w_pawns = bb[P] if weak == WHITE else bb[P + 6]
    s_king = lsb(bb[K]) if strong == WHITE else lsb(bb[K + 6])
    w_king = lsb(bb[K]) if weak == WHITE else lsb(bb[K + 6])
    return s_pawns, w_pawns, s_king, w_king


@njit(cache=False, fastmath=True, **_NOWRAP)
def _scale_pawns_vs_bare_king(s_pawns, w_king, weak):
    """K and pawns vs lone K: all pawns on one rook file with the weak king in
    front of them is a draw."""
    span = WHITE_PASSED[w_king] if weak == WHITE else BLACK_PASSED[w_king]
    on_rook_file = not (s_pawns & ~FILE_A) or not (s_pawns & ~FILE_H)
    if on_rook_file and not (s_pawns & ~span):
        return 0
    return -1


@njit(cache=False, fastmath=True, **_NOWRAP)
def _scale_bishop_pawns(bb, strong, s_pawns, w_pawns, s_king, w_king,
                        npm_weak, s_pawn_count, w_pawn_count):
    """KB and pawns vs K: wrong rook pawn with the wrong-coloured bishop, and
    the b/g-file blockade."""
    s_bishop = lsb(bb[B]) if strong == WHITE else lsb(bb[B + 6])

    if not (s_pawns & ~FILE_A) or not (s_pawns & ~FILE_H):
        pawn_file = FILE_OF[lsb(s_pawns)]
        queen_sq = pawn_file if strong == WHITE else 56 + pawn_file
        if _square_colour(queen_sq) != _square_colour(s_bishop) \
                and _chebyshev(queen_sq, w_king) <= 1:
            return 0

    all_pawns = s_pawns | w_pawns
    on_bg = not (all_pawns & ~FILE_B) or not (all_pawns & ~FILE_G)
    if on_bg and npm_weak == 0 and w_pawn_count >= 1:
        wp_sq = lsb(w_pawns) if strong == WHITE else _msb(w_pawns)
        push = wp_sq - 8 if strong == BLACK else wp_sq + 8
        blocked = 0 <= push <= 63 and (s_pawns & (ONE << uint64(push)))
        if RELATIVE_RANK[strong, wp_sq] == 6 and blocked \
                and (_square_colour(s_bishop) != _square_colour(wp_sq)
                     or s_pawn_count == 1):
            sd = _chebyshev(wp_sq, s_king)
            wd = _chebyshev(wp_sq, w_king)
            if RELATIVE_RANK[strong, w_king] >= 6 and wd <= 2 and wd <= sd:
                return 0
    return -1


@njit(cache=False, fastmath=True, **_NOWRAP)
def _scale_bishop_pawn_vs_bishop(bb, strong, weak, s_pawns, w_king):
    """KBP vs KB: king blockading on the wrong colour, or opposite bishops."""
    sp_sq = lsb(s_pawns)
    s_bishop = lsb(bb[B]) if strong == WHITE else lsb(bb[B + 6])
    w_bishop = lsb(bb[B]) if weak == WHITE else lsb(bb[B + 6])
    forward = WHITE_FORWARD_FILE[sp_sq] if strong == WHITE \
        else BLACK_FORWARD_FILE[sp_sq]
    if (forward & (ONE << uint64(w_king))) \
            and (_square_colour(w_king) != _square_colour(s_bishop)
                 or RELATIVE_RANK[strong, w_king] <= 5):
        return 0
    if _square_colour(s_bishop) != _square_colour(w_bishop):
        return 0
    return -1


@njit(cache=False, fastmath=True, **_NOWRAP)
def _scale_bishop_pawn_vs_knight(bb, strong, s_pawns, w_king):
    """KBP vs KN: king blockading the pawn on the wrong colour."""
    sp_sq = lsb(s_pawns)
    s_bishop = lsb(bb[B]) if strong == WHITE else lsb(bb[B + 6])
    if FILE_OF[w_king] == FILE_OF[sp_sq] \
            and RELATIVE_RANK[strong, sp_sq] < RELATIVE_RANK[strong, w_king] \
            and (_square_colour(w_king) != _square_colour(s_bishop)
                 or RELATIVE_RANK[strong, w_king] <= 5):
        return 0
    return -1


@njit(cache=False, fastmath=True, **_NOWRAP)
def _scale_pawn_vs_pawn(bb, st, strong, s_pawns, s_king, w_king):
    """KP vs KP: ignore the weak pawn and probe the bitbase. A position that is
    already drawn without the weak pawn stays drawn with it."""
    sp_sq = lsb(s_pawns)
    # a passer on the 5th or beyond off the a-file is too dangerous to call
    if RELATIVE_RANK[strong, sp_sq] >= 4 and FILE_OF[sp_sq] != 0:
        return -1
    strong_to_move = int(st[SIDE]) == strong
    if kpk_probe(s_king, sp_sq, w_king, strong == WHITE, strong_to_move):
        return -1
    return 0


@njit(cache=False, fastmath=True, **_NOWRAP)
def _specialized_scale(bb, st, strong):
    """Scale factor for exactly drawn or strongly drawish material, or -1 when
    no specialised rule applies."""
    weak = 1 - strong
    npm_strong = _non_pawn_material(bb, strong)
    npm_weak = _non_pawn_material(bb, weak)
    s_pawns, w_pawns, s_king, w_king = _pawns_and_kings(bb, strong)
    s_count = count_bits(s_pawns)
    w_count = count_bits(w_pawns)

    if npm_strong == 0 and s_count >= 2 and npm_weak == 0 and w_count == 0:
        return _scale_pawns_vs_bare_king(s_pawns, w_king, weak)

    # the specific rules are tested before the general KB-and-pawns one. In
    # the C the general branch comes first and returns -1 unconditionally, and
    # its guard is implied by theirs, so KBP vs KB and KBP vs KN can never fire
    # there.
    s_bishops = count_bits(bb[B] if strong == WHITE else bb[B + 6])
    if npm_strong == BISHOP_V and s_count >= 1 and s_bishops == 1:
        if npm_weak == BISHOP_V and s_count == 1 and w_count == 0:
            return _scale_bishop_pawn_vs_bishop(bb, strong, weak, s_pawns,
                                                w_king)
        w_knights = count_bits(bb[N] if weak == WHITE else bb[N + 6])
        if npm_weak == KNIGHT_V and s_count == 1 and w_count == 0 \
                and w_knights == 1:
            return _scale_bishop_pawn_vs_knight(bb, strong, s_pawns, w_king)
        return _scale_bishop_pawns(bb, strong, s_pawns, w_pawns, s_king,
                                   w_king, npm_weak, s_count, w_count)

    if npm_strong == 0 and s_count == 1 and npm_weak == 0 and w_count == 1:
        return _scale_pawn_vs_pawn(bb, st, strong, s_pawns, s_king, w_king)

    return -1


@njit(cache=False, fastmath=True, **_NOWRAP)
def _opposite_bishop_scale(bb, strong, s_pawns, w_pawns, npm_white, npm_black):
    if npm_white == BISHOP_V and npm_black == BISHOP_V:
        passed = 0
        pawns = s_pawns
        while pawns:
            sq = lsb(pawns)
            pawns &= pawns - ONE
            span = WHITE_PASSED[sq] if strong == WHITE else BLACK_PASSED[sq]
            if not (span & w_pawns):
                passed += 1
        return 18 + 4 * passed
    strong_occ = bb[OCC_W] if strong == WHITE else bb[OCC_B]
    return 22 + 3 * count_bits(strong_occ)


@njit(cache=False, fastmath=True, **_NOWRAP)
def _generic_scale(bb, strong, weak, s_pawns, w_pawns, npm_white, npm_black,
                   s_count):
    """Opposite bishops, single-flank rook endings and queen-vs-pieces."""
    weak_occ = bb[OCC_W] if weak == WHITE else bb[OCC_B]
    npm_strong = npm_white if strong == WHITE else npm_black
    if count_bits(weak_occ) == 1 and npm_strong >= ROOK_V:
        return NORMAL

    white_bishops = count_bits(bb[B])
    black_bishops = count_bits(bb[B + 6])
    opposite = white_bishops == 1 and black_bishops == 1 \
        and _square_colour(lsb(bb[B])) != _square_colour(lsb(bb[B + 6]))

    all_pawns = bb[P] | bb[P + 6]
    both_flanks = (all_pawns & QUEENSIDE) != ZERO and (all_pawns & KINGSIDE) != ZERO
    one_flank = 0 if both_flanks else 1

    if opposite:
        sf = _opposite_bishop_scale(bb, strong, s_pawns, w_pawns, npm_white,
                                    npm_black)
    elif npm_white == ROOK_V and npm_black == ROOK_V and s_count - count_bits(w_pawns) <= 1 \
            and ((KINGSIDE & s_pawns) != ZERO) != ((QUEENSIDE & s_pawns) != ZERO) \
            and _weak_king_touches_pawn(bb, weak, w_pawns):
        sf = 36
    elif count_bits(bb[Q]) + count_bits(bb[Q + 6]) == 1:
        if count_bits(bb[Q]) == 1:
            minors = count_bits(bb[B + 6]) + count_bits(bb[N + 6])
        else:
            minors = count_bits(bb[B]) + count_bits(bb[N])
        sf = 37 + 3 * minors
    else:
        sf = NORMAL
        limit = 36 + 7 * s_count
        if limit < sf:
            sf = limit
        sf -= 4 * one_flank

    return sf - 4 * one_flank


@njit(cache=False, fastmath=True, **_NOWRAP)
def _weak_king_touches_pawn(bb, weak, w_pawns):
    w_king = lsb(bb[K]) if weak == WHITE else lsb(bb[K + 6])
    return (KING_ATTACKS[w_king] & w_pawns) != ZERO


@njit(cache=False, fastmath=True, **_NOWRAP)
def endgame_scale(bb, st, score):
    """Scale factor in [0, 64] for the side the score favours."""
    strong = WHITE if score > 0 else BLACK
    weak = 1 - strong
    npm_white = _non_pawn_material(bb, WHITE)
    npm_black = _non_pawn_material(bb, BLACK)
    npm_strong = npm_white if strong == WHITE else npm_black
    npm_weak = npm_white if weak == WHITE else npm_black
    s_pawns, w_pawns, s_king, w_king = _pawns_and_kings(bb, strong)
    s_count = count_bits(s_pawns)

    sf = NORMAL
    # no pawns and only a slim material edge is very hard or impossible to win
    if s_count == 0 and npm_strong - npm_weak <= BISHOP_V:
        if npm_strong < ROOK_V:
            sf = 0
        elif npm_weak <= BISHOP_V:
            sf = 4
        else:
            sf = 14

    if sf == NORMAL:
        spec = _specialized_scale(bb, st, strong)
        if spec >= 0:
            sf = spec

    if sf == NORMAL:
        sf = _generic_scale(bb, strong, weak, s_pawns, w_pawns, npm_white,
                            npm_black, s_count)

    if sf < 0:
        sf = 0
    if sf > NORMAL:
        sf = NORMAL
    return sf
