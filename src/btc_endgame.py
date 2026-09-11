"""Specialised endgame knowledge, ported from BTC evaluation.h.

Exact evaluations for known low-material configurations. These matter more
here than tablebases would: they give the evaluation a gradient toward mating
(drive the weak king to the edge or the right corner, bring the strong king
close), which is what lets a shallow search convert won endings instead of
shuffling until the referee claims fifty moves. Without them the evaluation is
flat across all winning rook moves and the search has nothing to follow.

KP vs K is deliberately not handled: the C engine probes a generated bitbase,
which is a separate piece of work. Those positions fall through to the normal
evaluation.
"""

import numpy as np
from numba import njit, uint64

from btc_core import (
    B, K, N, OCC_A, OCC_B, OCC_W, ONE, P, Q, R, SIDE, WHITE, ZERO, count_bits,
    lsb,
)
from btc_evalmasks import (
    BLACK_FORWARD_FILE, FILE_MASK, FILE_OF, RANK_OF, RELATIVE_RANK,
    WHITE_FORWARD_FILE,
)
from btc_kpk import probe as kpk_probe
KNOWN_WIN = 10000
MATE_SCORE = 48000

PAWN_EG = 208
ROOK_EG = 1380
QUEEN_EG = 2682
ROOK_MG = 1276

NPM_MG = np.array([0, 781, 825, 1276, 2538, 0], dtype=np.int64)

LIGHT_SQUARES = uint64(0x55AA55AA55AA55AA)
DARK_SQUARES = uint64(0xAA55AA55AA55AA55)
BDEG_FILES = uint64(int(FILE_MASK[1]) | int(FILE_MASK[3])
                    | int(FILE_MASK[4]) | int(FILE_MASK[6]))


@njit(cache=False, fastmath=True)
def _chebyshev(a, b):
    dr = RANK_OF[a] - RANK_OF[b]
    df = FILE_OF[a] - FILE_OF[b]
    if dr < 0:
        dr = -dr
    if df < 0:
        df = -df
    return dr if dr > df else df


@njit(cache=False, fastmath=True)
def _push_to_edge(sq):
    rank = RANK_OF[sq]
    file = FILE_OF[sq]
    rd = rank if rank < 7 - rank else 7 - rank
    fd = file if file < 7 - file else 7 - file
    return 90 - (7 * fd * fd // 2 + 7 * rd * rd // 2)


@njit(cache=False, fastmath=True)
def _push_to_corner(sq):
    """0 on the a8-h1 diagonal, 7 in the a1 and h8 corners.

    The C engine writes this as abs(7 - getRank[s] - getFile[s]), but its
    getRank counts from rank 1 (getRank[a8] == 7) while RANK_OF here counts
    from a8 (RANK_OF[a8] == 0). Substituting gives abs(RANK_OF - FILE_OF).
    Using the C expression directly targets a8/h1, the wrong corners, which
    makes the KBN mate unreachable."""
    v = RANK_OF[sq] - FILE_OF[sq]
    return v if v >= 0 else -v


@njit(cache=False, fastmath=True)
def _push_close(a, b):
    return 140 - 20 * _chebyshev(a, b)


@njit(cache=False, fastmath=True)
def _push_away(a, b):
    return 120 - _push_close(a, b)


@njit(cache=False, fastmath=True)
def _square_colour(sq):
    return (sq & 1) ^ ((sq >> 3) & 1)


@njit(cache=False, fastmath=True)
def _non_pawn_material(bb, side):
    base = 0 if side == WHITE else 6
    total = 0
    for pt in range(N, K):
        total += count_bits(bb[base + pt]) * NPM_MG[pt]
    return total


@njit(cache=False, fastmath=True)
def _lone_king_case(bb, strong, s_counts, npm_strong, s_king, w_king):
    """Weak side has a bare king. Returns (handled, score) for the strong side."""
    s_pawns, s_knights, s_bishops, s_rooks, s_queens = s_counts
    base = 0 if strong == WHITE else 6

    if s_pawns == 0 and s_knights == 2 and s_bishops == 0 \
            and s_rooks == 0 and s_queens == 0:
        return True, 0

    if s_pawns == 0 and s_knights == 1 and s_bishops == 1 \
            and s_rooks == 0 and s_queens == 0:
        bishop_sq = lsb(bb[base + B])
        # a1 is square 56; mate must be delivered in a corner the bishop covers
        drive = (w_king ^ 7) if _square_colour(bishop_sq) != _square_colour(56) \
            else w_king
        return True, (KNOWN_WIN + 3520) + _push_close(s_king, w_king) \
            + 420 * _push_to_corner(drive)

    if npm_strong >= ROOK_MG:
        score = npm_strong + s_pawns * PAWN_EG + _push_to_edge(w_king) \
            + _push_close(s_king, w_king)
        bishops = bb[base + B]
        opposite_bishops = (bishops & LIGHT_SQUARES) != ZERO \
            and (bishops & DARK_SQUARES) != ZERO
        if s_queens or s_rooks or (s_bishops and s_knights) or opposite_bishops:
            score += KNOWN_WIN
            if score > MATE_SCORE - 1:
                score = MATE_SCORE - 1
        return True, score

    return False, 0


@njit(cache=False, fastmath=True)
def _side_counts(bb, side):
    base = 0 if side == WHITE else 6
    return (count_bits(bb[base + P]), count_bits(bb[base + N]),
            count_bits(bb[base + B]), count_bits(bb[base + R]),
            count_bits(bb[base + Q]))


# Above this many pieces on the board, probe() cannot recognise anything and
# the evaluation is the ordinary material-plus-positional one. Search code that
# reasons in material units has to know where that stops being true.
SPECIALISED_MAX_PIECES = 5


@njit(cache=False, fastmath=True)
def insufficient_material(bb):
    """King versus king, or king and a single minor versus king."""
    white_count = count_bits(bb[OCC_W])
    black_count = count_bits(bb[OCC_B])
    if white_count == 1 and black_count == 1:
        return True
    if white_count + black_count != 3:
        return False
    minors = bb[N] | bb[B] | bb[N + 6] | bb[B + 6]
    return count_bits(minors) == 1


@njit(cache=False, fastmath=True)
def probe(bb, st):
    """Exact score for a recognised configuration, from white's point of view.
    Returns (handled, score).

    Covers the same set as BTC's probeSpecializedEndgame: bare-king endings,
    the weak side holding a single pawn, rook against a lone minor, and queen
    against rook. These give the evaluation its mating gradient, which is what
    lets a shallow search convert instead of shuffling."""
    if count_bits(bb[OCC_A]) > SPECIALISED_MAX_PIECES:
        return False, 0

    for strong in range(2):
        weak = 1 - strong
        s_counts = _side_counts(bb, strong)
        w_counts = _side_counts(bb, weak)
        npm_strong = _non_pawn_material(bb, strong)
        npm_weak = _non_pawn_material(bb, weak)
        s_king = lsb(bb[K]) if strong == WHITE else lsb(bb[K + 6])
        w_king = lsb(bb[K]) if weak == WHITE else lsb(bb[K + 6])

        handled, score = _dispatch(bb, st, strong, weak, s_counts, w_counts,
                                   npm_strong, npm_weak, s_king, w_king)
        if handled:
            return True, score if strong == WHITE else -score

    return False, 0


@njit(cache=False, fastmath=True)
def _dispatch(bb, st, strong, weak, s_counts, w_counts, npm_strong, npm_weak,
              s_king, w_king):
    """BTC's branch chain, in its order: the later cases are only reached when
    the earlier guards fail."""
    if npm_weak == 0 and w_counts[0] == 0:
        if s_counts[0] == 1 and npm_strong == 0:
            return _kp_vs_k(bb, st, strong, s_king, w_king)
        return _lone_king_case(bb, strong, s_counts, npm_strong, s_king,
                               w_king)

    if npm_weak == 0 and w_counts[0] == 1:
        return _weak_pawn_case(bb, st, strong, weak, s_counts, s_king, w_king)

    only_rook = (s_counts[0] == 0 and s_counts[1] == 0 and s_counts[2] == 0
                 and s_counts[3] == 1 and s_counts[4] == 0)
    if only_rook and w_counts[0] == 0:
        return _rook_vs_minor(bb, weak, w_counts, w_king)

    only_queen = (s_counts[0] == 0 and s_counts[1] == 0 and s_counts[2] == 0
                  and s_counts[3] == 0 and s_counts[4] == 1)
    lone_rook = (w_counts[0] == 0 and w_counts[1] == 0 and w_counts[2] == 0
                 and w_counts[3] == 1 and w_counts[4] == 0)
    if only_queen and lone_rook:
        return True, (QUEEN_EG - ROOK_EG + _push_to_edge(w_king)
                      + _push_close(s_king, w_king))
    return False, 0


@njit(cache=False, fastmath=True)
def _kp_vs_k(bb, st, strong, s_king, w_king):
    """KP vs K, decided exactly by the bitbase."""
    pawn_sq = lsb(bb[P] if strong == WHITE else bb[P + 6])
    if not kpk_probe(s_king, pawn_sq, w_king, strong == WHITE,
                     int(st[SIDE]) == strong):
        return True, 0
    # BTC adds the pawn's rank counted from the strong side's first rank,
    # which its normalisation turns into getRank of a white pawn
    return True, KNOWN_WIN + PAWN_EG + RELATIVE_RANK[strong, pawn_sq]


@njit(cache=False, fastmath=True)
def _weak_pawn_case(bb, st, strong, weak, s_counts, s_king, w_king):
    """Weak side has a king and one pawn: KR vs KP, KQ vs KP, KNN vs KP."""
    wp_sq = lsb(bb[P] if weak == WHITE else bb[P + 6])
    if s_counts[0] != 0:
        return False, 0

    if s_counts[3] == 1 and s_counts[1] == 0 and s_counts[2] == 0 \
            and s_counts[4] == 0:
        return _rook_vs_pawn(bb, st, strong, weak, s_king, w_king, wp_sq)

    if s_counts[4] == 1 and s_counts[1] == 0 and s_counts[2] == 0 \
            and s_counts[3] == 0:
        # won unless a rook or bishop pawn on the 7th is guarded by the king
        score = _push_close(s_king, w_king)
        if RELATIVE_RANK[weak, wp_sq] != 6 \
                or _chebyshev(w_king, wp_sq) != 1 \
                or (BDEG_FILES & (ONE << uint64(wp_sq))):
            score += QUEEN_EG - PAWN_EG
        return True, score

    if s_counts[1] == 2 and s_counts[2] == 0 and s_counts[3] == 0 \
            and s_counts[4] == 0:
        # KNN vs KP: very drawish, small chances only with a cornered king
        return True, (PAWN_EG + 2 * _push_to_edge(w_king)
                      - 10 * RELATIVE_RANK[weak, wp_sq])
    return False, 0


@njit(cache=False, fastmath=True)
def _rook_vs_pawn(bb, st, strong, weak, s_king, w_king, wp_sq):
    """KR vs KP. Four cases: the strong king is in front of the pawn, the weak
    king is too far away to help, the weak king shields a pawn the strong king
    cannot reach, or a race scored by the tempo difference."""
    rook_sq = lsb(bb[R] if strong == WHITE else bb[R + 6])
    stm_weak = 1 if int(st[SIDE]) == weak else 0
    stm_strong = 1 if int(st[SIDE]) == strong else 0
    push = -8 if weak == WHITE else 8
    queen_sq = FILE_OF[wp_sq] if weak == WHITE else 56 + FILE_OF[wp_sq]
    ahead = WHITE_FORWARD_FILE[s_king] if strong == WHITE \
        else BLACK_FORWARD_FILE[s_king]

    if ahead & (ONE << uint64(wp_sq)):
        return True, ROOK_EG - _chebyshev(s_king, wp_sq)
    if _chebyshev(w_king, wp_sq) >= 3 + stm_weak \
            and _chebyshev(w_king, rook_sq) >= 3:
        return True, ROOK_EG - _chebyshev(s_king, wp_sq)
    if RELATIVE_RANK[strong, w_king] <= 2 \
            and _chebyshev(w_king, wp_sq) == 1 \
            and RELATIVE_RANK[strong, s_king] >= 3 \
            and _chebyshev(s_king, wp_sq) > 2 + stm_strong:
        return True, 80 - 8 * _chebyshev(s_king, wp_sq)
    stop = wp_sq + push
    return True, 200 - 8 * (_chebyshev(s_king, stop)
                            - _chebyshev(w_king, stop)
                            - _chebyshev(wp_sq, queen_sq))


@njit(cache=False, fastmath=True)
def _rook_vs_minor(bb, weak, w_counts, w_king):
    """KR vs KB and KR vs KN, both drawish."""
    if w_counts[2] == 1 and w_counts[1] == 0 and w_counts[3] == 0 \
            and w_counts[4] == 0:
        return True, _push_to_edge(w_king)
    if w_counts[1] == 1 and w_counts[2] == 0 and w_counts[3] == 0 \
            and w_counts[4] == 0:
        knight_sq = lsb(bb[N] if weak == WHITE else bb[N + 6])
        return True, _push_to_edge(w_king) + _push_away(w_king, knight_sq)
    return False, 0
