"""Hand-crafted evaluation ported from BTC evaluation.h / eval_constants.cpp.

Structure differs from the C engine in one deliberate way: the C code rebuilds
its attack and mobility accumulators inside every makeMove, including on nodes
that never evaluate. Here they are built inside evaluate() on demand, which
removes that waste and structurally prevents the stale-accumulator class of bug
(see docs/ENGINE_AUDIT.md and docs/BTC_UPSTREAM_ISSUES.md).

Terms: material, piece-square tables, mobility with pin-excluded areas and
x-rays, outposts, minor behind pawn, king protector distance, rook on open and
semi-open files, passed pawns, pawn structure, and king safety. All scores are
on BTC's unified scale (pawn = 126 midgame) from white's point of view until
the final flip.
"""

import numpy as np
from numba import int64, njit, uint64

from btc_core import (
    B, BLACK, K, N, OCC_A, OCC_B, OCC_W, ONE, P, Q, R, SIDE, WHITE, ZERO,
    bishop_attacks, count_bits, KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS,
    lsb, queen_attacks, rook_attacks,
)
from btc_evalmasks import (
    ADJACENT_FILES, BETWEEN, BLACK_FORWARD_FILE, BLACK_KING_ZONE,
    BLACK_PASSED, BLACK_SUPPORT, FILE_MASK, FILE_OF, ISOLATED, KING_FLANK,
    LINE, PHALANX, RANK_MASK, RANK_OF, RELATIVE_RANK, WHITE_FORWARD_FILE,
    WHITE_KING_ZONE, WHITE_PASSED, WHITE_SUPPORT,
)
from btc_psqt import MIRROR, PIECE_TABLES

TEMPO = 28
OPENING_PHASE = 15196
ENDGAME_PHASE = 1841

MATERIAL_MG = np.array([126, 781, 825, 1276, 2538, 10000], dtype=np.int64)
MATERIAL_EG = np.array([208, 854, 915, 1380, 2682, 10000], dtype=np.int64)

MOBILITY_MG = np.array([
    [0] * 32,
    [-62, -53, -12, -3, 3, 12, 21, 28, 37] + [37] * 23,
    [-47, -20, 14, 29, 39, 53, 53, 60, 62, 69, 78, 83, 91, 96] + [96] * 18,
    [-60, -24, 0, 3, 4, 14, 20, 30, 41, 41, 41, 45, 57, 58, 67] + [67] * 17,
    [-29, -16, -8, -8, 18, 25, 23, 37, 41, 54, 65, 68, 69, 70, 70, 70, 71, 72,
     74, 76, 90, 104, 105, 106, 112, 114, 114, 119] + [119] * 4,
    [0] * 32,
], dtype=np.int64)

MOBILITY_EG = np.array([
    [0] * 32,
    [-79, -57, -31, -17, 7, 13, 16, 21, 26] + [26] * 23,
    [-59, -25, -8, 12, 21, 40, 56, 58, 65, 72, 78, 87, 88, 98] + [98] * 18,
    [-82, -15, 17, 43, 72, 100, 102, 122, 133, 139, 153, 160, 165, 170, 175]
    + [175] * 17,
    [-49, -29, -8, 17, 39, 54, 59, 73, 76, 95, 95, 101, 124, 128, 132, 133,
     136, 140, 147, 149, 153, 169, 171, 171, 178, 185, 187, 221] + [221] * 4,
    [0] * 32,
], dtype=np.int64)

PASSED_RANK_MG = np.array([0, 2, 15, 22, 64, 166, 284, 0], dtype=np.int64)
PASSED_RANK_EG = np.array([0, 38, 36, 50, 81, 184, 269, 0], dtype=np.int64)
PASSED_FILE_MG, PASSED_FILE_EG = 13, 8

CONNECTED_RANK = np.array([0, 3, 7, 7, 15, 54, 86, 0], dtype=np.int64)

DOUBLED_MG, DOUBLED_EG = -11, -51
ISOLATED_MG, ISOLATED_EG = -1, -20
BACKWARD_MG, BACKWARD_EG = -6, -19
WEAK_UNOPPOSED_MG, WEAK_UNOPPOSED_EG = -15, -18

SEMI_OPEN_MG, SEMI_OPEN_EG = 18, 7
OPEN_MG, OPEN_EG = 44, 20
ROOK_OPEN_MG = np.array([18, 49], dtype=np.int64)
ROOK_OPEN_EG = np.array([8, 26], dtype=np.int64)

OUTPOST_KNIGHT_MG, OUTPOST_KNIGHT_EG = 54, 34
OUTPOST_BISHOP_MG, OUTPOST_BISHOP_EG = 31, 25
REACHABLE_OUTPOST_MG, REACHABLE_OUTPOST_EG = 33, 19
MINOR_BEHIND_PAWN_MG, MINOR_BEHIND_PAWN_EG = 18, 3
KING_PROTECTOR_KNIGHT_MG, KING_PROTECTOR_KNIGHT_EG = 9, 9
KING_PROTECTOR_BISHOP_MG, KING_PROTECTOR_BISHOP_EG = 7, 9
LONG_DIAGONAL_BISHOP_MG = 45

KING_ATTACK_WEIGHTS = np.array([0, 76, 46, 45, 14, 0], dtype=np.int64)
SAFE_CHECK_MG = np.array([0, 805, 650, 1071, 730, 0], dtype=np.int64)

KD_WEAK_RING = 183
KD_UNSAFE_CHECK = 148
KD_KING_ATTACKS = 69
KD_FLANK_ATTACK = 3
KD_NO_QUEEN = 873
KD_INIT = 37

SHELTER_STRENGTH = np.array([
    [-6, 81, 93, 58, 39, 18, 25],
    [-43, 61, 35, -49, -29, -11, -63],
    [-10, 75, 23, -2, 32, 3, -45],
    [-39, -13, -29, -52, -48, -67, -166],
], dtype=np.int64)

UNBLOCKED_STORM = np.array([
    [89, -285, -185, 93, 57, 45, 51],
    [44, -18, 123, 46, 39, -7, 23],
    [4, 52, 162, 37, 7, -14, -2],
    [-10, -14, 90, 15, 2, -7, -16],
], dtype=np.int64)

LONG_DIAGONALS = uint64(0x8142241818244281)
CENTER = uint64(0x0000001818000000)


@njit(cache=False, fastmath=True)
def c_div(a, b):
    """C integer division: truncation toward zero. Requires b > 0."""
    q = a // b
    if q < 0 and q * b != a:
        q += 1
    return q


@njit(cache=False, fastmath=True)
def _taper(mg, eg, phase):
    return c_div(mg * phase + eg * (OPENING_PHASE - phase), OPENING_PHASE)


@njit(cache=False, fastmath=True)
def game_phase(bb):
    score = 0
    for pc in range(N, Q + 1):
        score += (count_bits(bb[pc]) + count_bits(bb[pc + 6])) * MATERIAL_MG[pc]
    if score > OPENING_PHASE:
        score = OPENING_PHASE
    return score


@njit(cache=False, fastmath=True)
def _pawn_attack_span(bb, side):
    """Squares the side's pawns attack now or could attack by advancing."""
    span = ZERO
    pawns = bb[P] if side == WHITE else bb[P + 6]
    while pawns:
        sq = lsb(pawns)
        pawns &= pawns - ONE
        f = FILE_OF[sq]
        r = RANK_OF[sq]
        if side == WHITE:
            rr = r - 1
            while rr >= 0:
                if f > 0:
                    span |= ONE << uint64(rr * 8 + f - 1)
                if f < 7:
                    span |= ONE << uint64(rr * 8 + f + 1)
                rr -= 1
        else:
            rr = r + 1
            while rr <= 7:
                if f > 0:
                    span |= ONE << uint64(rr * 8 + f - 1)
                if f < 7:
                    span |= ONE << uint64(rr * 8 + f + 1)
                rr += 1
    return span


@njit(cache=False, fastmath=True)
def _pawn_attacks_of(bb, side):
    attacks = ZERO
    doubled = ZERO
    pawns = bb[P] if side == WHITE else bb[P + 6]
    while pawns:
        sq = lsb(pawns)
        pawns &= pawns - ONE
        a = PAWN_ATTACKS[side, sq]
        doubled |= attacks & a
        attacks |= a
    return attacks, doubled


@njit(cache=False, fastmath=True)
def _king_blockers(bb, us):
    """Pieces of either colour pinned against `us`'s king by a single blocker."""
    ksq = lsb(bb[K]) if us == WHITE else lsb(bb[K + 6])
    base = 6 if us == WHITE else 0
    enemy_rq = bb[base + R] | bb[base + Q]
    enemy_bq = bb[base + B] | bb[base + Q]
    snipers = (rook_attacks(ksq, ZERO) & enemy_rq) \
        | (bishop_attacks(ksq, ZERO) & enemy_bq)
    occ = bb[OCC_A] ^ snipers
    blockers = ZERO
    while snipers:
        sniper = lsb(snipers)
        snipers &= snipers - ONE
        between = BETWEEN[ksq, sniper] & occ
        if between and (between & (between - ONE)) == ZERO:
            blockers |= between
    return blockers


@njit(cache=False, fastmath=True)
def _mobility_area(bb, side, enemy_pawn_attacks, blockers):
    """Squares that count toward mobility: not our blocked or low pawns, not
    our king or queen, not pinned pieces, not attacked by an enemy pawn."""
    occ = bb[OCC_A]
    if side == WHITE:
        low_ranks = RANK_MASK[6] | RANK_MASK[5]
        blocked = bb[P] & ((occ << uint64(8)) | low_ranks)
        own = blocked | bb[K] | bb[Q]
    else:
        low_ranks = RANK_MASK[1] | RANK_MASK[2]
        blocked = bb[P + 6] & ((occ >> uint64(8)) | low_ranks)
        own = blocked | bb[K + 6] | bb[Q + 6]
    return ~(own | blockers | enemy_pawn_attacks)


@njit(cache=False, fastmath=True)
def _piece_mobility(bb, piece_type, sq, side, area, blockers):
    """Attacked squares inside the mobility area, x-raying through queens and
    (for rooks) friendly rooks, restricted to the pin line when pinned."""
    occ = bb[OCC_A]
    queens = bb[Q] | bb[Q + 6]
    if piece_type == N:
        att = KNIGHT_ATTACKS[sq]
    elif piece_type == B:
        att = bishop_attacks(sq, occ ^ queens)
    elif piece_type == R:
        own_rooks = bb[R] if side == WHITE else bb[R + 6]
        att = rook_attacks(sq, occ ^ queens ^ own_rooks)
    else:
        att = queen_attacks(sq, occ)
    if blockers & (ONE << uint64(sq)):
        ksq = lsb(bb[K]) if side == WHITE else lsb(bb[K + 6])
        att &= LINE[ksq, sq]
    return count_bits(att & area)


@njit(cache=False, fastmath=True)
def _passed_pawn(bb, sq, side, phase, enemy_king_sq, own_king_sq):
    """Passed pawn bonus by relative rank and file, with a king-race term in
    the endgame. Returns 0 when the pawn is not passed."""
    enemy_pawns = bb[P + 6] if side == WHITE else bb[P]
    mask = WHITE_PASSED[sq] if side == WHITE else BLACK_PASSED[sq]
    if enemy_pawns & mask:
        return 0
    own_pawns = bb[P] if side == WHITE else bb[P + 6]
    front = WHITE_FORWARD_FILE[sq] if side == WHITE else BLACK_FORWARD_FILE[sq]
    if own_pawns & front:
        return 0

    rr = RELATIVE_RANK[side, sq]
    mg = PASSED_RANK_MG[rr]
    eg = PASSED_RANK_EG[rr]
    f = FILE_OF[sq]
    edge = f if f < 7 - f else 7 - f
    mg -= PASSED_FILE_MG * edge
    eg -= PASSED_FILE_EG * edge

    if rr > 2:
        push = sq - 8 if side == WHITE else sq + 8
        if 0 <= push <= 63:
            enemy_dist = _chebyshev(enemy_king_sq, push)
            own_dist = _chebyshev(own_king_sq, push)
            eg += 19 * enemy_dist - 8 * own_dist
    return _taper(mg, eg, phase)


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
def _pawn_structure(bb, side, phase):
    """Doubled, isolated, backward, phalanx and supported pawns."""
    score = 0
    own = bb[P] if side == WHITE else bb[P + 6]
    enemy = bb[P + 6] if side == WHITE else bb[P]
    pawns = own
    while pawns:
        sq = lsb(pawns)
        pawns &= pawns - ONE
        f = FILE_OF[sq]
        front = WHITE_FORWARD_FILE[sq] if side == WHITE else BLACK_FORWARD_FILE[sq]
        support = WHITE_SUPPORT[sq] if side == WHITE else BLACK_SUPPORT[sq]
        opposed = (enemy & front) != ZERO

        if own & front:
            score += _taper(DOUBLED_MG, DOUBLED_EG, phase)

        neighbours = own & ADJACENT_FILES[f]
        if not neighbours:
            score += _taper(ISOLATED_MG, ISOLATED_EG, phase)
            if not opposed:
                score += _taper(WEAK_UNOPPOSED_MG, WEAK_UNOPPOSED_EG, phase)
        else:
            phalanx = own & PHALANX[sq]
            supported = count_bits(own & support)
            if phalanx or supported:
                rr = RELATIVE_RANK[side, sq]
                bonus = CONNECTED_RANK[rr] * (2 if phalanx else 1)
                if opposed:
                    bonus = c_div(bonus, 2)
                score += _taper(bonus + 21 * supported,
                                c_div(bonus * (rr - 2), 4), phase)
            elif not (own & (WHITE_PASSED[sq] if side == BLACK
                             else BLACK_PASSED[sq]) & ADJACENT_FILES[f]):
                score += _taper(BACKWARD_MG, BACKWARD_EG, phase)
    return score


@njit(cache=False, fastmath=True)
def _shelter_storm(bb, side, ksq):
    """King shelter and pawn storm on the king's file and its neighbours."""
    own = bb[P] if side == WHITE else bb[P + 6]
    enemy = bb[P + 6] if side == WHITE else bb[P]
    center = FILE_OF[ksq]
    if center < 1:
        center = 1
    if center > 6:
        center = 6
    score = 0
    for f in range(center - 1, center + 2):
        file_mask = FILE_MASK[f]
        our_pawns = own & file_mask
        their_pawns = enemy & file_mask
        our_rank = 0
        if our_pawns:
            sq = lsb(our_pawns) if side == BLACK else _msb(our_pawns)
            our_rank = RELATIVE_RANK[side, sq]
        their_rank = 0
        if their_pawns:
            sq = lsb(their_pawns) if side == BLACK else _msb(their_pawns)
            their_rank = RELATIVE_RANK[side, sq]
        edge = f if f < 7 - f else 7 - f
        score += SHELTER_STRENGTH[edge, our_rank]
        if our_rank and our_rank == their_rank - 1:
            score -= 82
        else:
            score -= UNBLOCKED_STORM[edge, their_rank]
    return score


@njit(cache=False, fastmath=True)
def _msb(bbv):
    idx = 0
    while bbv:
        idx = lsb(bbv)
        bbv &= bbv - ONE
    return idx


@njit(cache=False, fastmath=True)
def _minor_terms(bb, piece_type, sq, side, phase, own_pawn_att, enemy_pawn_att,
                 enemy_pawn_span):
    """Outposts, minor behind pawn, king protector, long diagonal bishop."""
    score = 0
    bit = ONE << uint64(sq)
    rr = RELATIVE_RANK[side, sq]
    outpost_ok = 3 <= rr <= 5 and (own_pawn_att & bit) and not (enemy_pawn_span & bit)
    if outpost_ok:
        if piece_type == N:
            score += _taper(OUTPOST_KNIGHT_MG, OUTPOST_KNIGHT_EG, phase)
        else:
            score += _taper(OUTPOST_BISHOP_MG, OUTPOST_BISHOP_EG, phase)
    elif piece_type == N:
        own_occ = bb[OCC_W] if side == WHITE else bb[OCC_B]
        ranks = RANK_MASK[2] | RANK_MASK[3] | RANK_MASK[4] if side == WHITE \
            else RANK_MASK[3] | RANK_MASK[4] | RANK_MASK[5]
        targets = KNIGHT_ATTACKS[sq] & ranks & own_pawn_att & ~enemy_pawn_span \
            & ~own_occ
        if targets:
            score += _taper(REACHABLE_OUTPOST_MG, REACHABLE_OUTPOST_EG, phase)

    own_pawns = bb[P] if side == WHITE else bb[P + 6]
    behind = (own_pawns << uint64(8)) if side == WHITE else (own_pawns >> uint64(8))
    if behind & bit:
        score += _taper(MINOR_BEHIND_PAWN_MG, MINOR_BEHIND_PAWN_EG, phase)

    ksq = lsb(bb[K]) if side == WHITE else lsb(bb[K + 6])
    dist = _chebyshev(sq, ksq)
    if piece_type == N:
        score -= _taper(KING_PROTECTOR_KNIGHT_MG, KING_PROTECTOR_KNIGHT_EG,
                        phase) * dist
    else:
        score -= _taper(KING_PROTECTOR_BISHOP_MG, KING_PROTECTOR_BISHOP_EG,
                        phase) * dist
        if (LONG_DIAGONALS & bit) and count_bits(
                bishop_attacks(sq, bb[P] | bb[P + 6]) & CENTER) > 1:
            score += _taper(LONG_DIAGONAL_BISHOP_MG, 0, phase)
    return score


@njit(cache=False, fastmath=True)
def _rook_file_term(bb, sq, side, phase):
    own_pawns = bb[P] if side == WHITE else bb[P + 6]
    enemy_pawns = bb[P + 6] if side == WHITE else bb[P]
    file_mask = FILE_MASK[FILE_OF[sq]]
    if own_pawns & file_mask:
        return 0
    if enemy_pawns & file_mask:
        return _taper(ROOK_OPEN_MG[0], ROOK_OPEN_EG[0], phase)
    return _taper(ROOK_OPEN_MG[1], ROOK_OPEN_EG[1], phase)


@njit(cache=False, fastmath=True)
def _side_score(bb, side, phase, own_pawn_att, enemy_pawn_att, enemy_pawn_span,
                area, blockers):
    """All per-piece terms for one side, from that side's point of view.
    Returns (score, king ring hits, attack units, distinct attackers); these
    are returned rather than written into arrays so the hot path allocates
    nothing."""
    score = 0
    king_zone_hits = 0
    attack_units = 0
    attackers = 0
    base = 0 if side == WHITE else 6
    enemy_ksq = lsb(bb[K + 6]) if side == WHITE else lsb(bb[K])
    enemy_zone = BLACK_KING_ZONE[enemy_ksq] if side == WHITE \
        else WHITE_KING_ZONE[enemy_ksq]
    own_ksq = lsb(bb[K]) if side == WHITE else lsb(bb[K + 6])
    occ = bb[OCC_A]

    for pt in range(P, K + 1):
        pieces = bb[base + pt]
        while pieces:
            sq = lsb(pieces)
            pieces &= pieces - ONE
            pst_sq = sq if side == WHITE else MIRROR[sq]
            score += _taper(MATERIAL_MG[pt], MATERIAL_EG[pt], phase)
            score += _taper(PIECE_TABLES[0, pt, pst_sq],
                            PIECE_TABLES[1, pt, pst_sq], phase)

            if pt == P:
                score += _passed_pawn(bb, sq, side, phase, enemy_ksq, own_ksq)
                continue
            if pt == K:
                continue

            mob = _piece_mobility(bb, pt, sq, side, area, blockers)
            score += _taper(MOBILITY_MG[pt, mob], MOBILITY_EG[pt, mob], phase)

            if pt == N or pt == B:
                score += _minor_terms(bb, pt, sq, side, phase, own_pawn_att,
                                      enemy_pawn_att, enemy_pawn_span)
            elif pt == R:
                score += _rook_file_term(bb, sq, side, phase)

            if pt == N:
                att = KNIGHT_ATTACKS[sq]
            elif pt == B:
                att = bishop_attacks(sq, occ)
            elif pt == R:
                att = rook_attacks(sq, occ)
            else:
                att = queen_attacks(sq, occ)
            hits = count_bits(att & enemy_zone)
            if hits:
                attackers += 1
                attack_units += KING_ATTACK_WEIGHTS[pt] * hits
                king_zone_hits += hits

    score += _pawn_structure(bb, side, phase)
    return score, king_zone_hits, attack_units, attackers


@njit(cache=False, fastmath=True)
def _all_attacks(bb, side):
    occ = bb[OCC_A]
    base = 0 if side == WHITE else 6
    attacks, _ = _pawn_attacks_of(bb, side)
    for pt in range(N, K + 1):
        pieces = bb[base + pt]
        while pieces:
            sq = lsb(pieces)
            pieces &= pieces - ONE
            if pt == N:
                attacks |= KNIGHT_ATTACKS[sq]
            elif pt == B:
                attacks |= bishop_attacks(sq, occ)
            elif pt == R:
                attacks |= rook_attacks(sq, occ)
            elif pt == Q:
                attacks |= queen_attacks(sq, occ)
            else:
                attacks |= KING_ATTACKS[sq]
    return attacks


@njit(cache=False, fastmath=True)
def _king_safety(bb, side, phase, attack_units, attackers, king_ring_attacks,
                 enemy_attacks, our_attacks):
    """Sum-of-contributions king danger, converted to a score."""
    ksq = lsb(bb[K]) if side == WHITE else lsb(bb[K + 6])
    enemy_queen = bb[Q + 6] if side == WHITE else bb[Q]

    danger = KD_INIT
    danger += attack_units
    danger += KD_KING_ATTACKS * king_ring_attacks
    weak_ring = count_bits(enemy_attacks & (WHITE_KING_ZONE[ksq]
                                            if side == WHITE
                                            else BLACK_KING_ZONE[ksq])
                           & ~our_attacks)
    danger += KD_WEAK_RING * weak_ring
    flank = KING_FLANK[FILE_OF[ksq]]
    danger += KD_FLANK_ATTACK * count_bits(enemy_attacks & flank)
    if not enemy_queen:
        danger -= KD_NO_QUEEN

    if attackers < 2:
        danger = c_div(danger, 2)
    if danger <= 0:
        return 0
    mg = -c_div(danger * danger, 4096)
    eg = -c_div(danger, 16)
    return _taper(mg, eg, phase)


@njit(cache=False, fastmath=True)
def evaluate(bb, st):
    """Full evaluation from the side to move's point of view."""
    phase = game_phase(bb)

    white_pawn_att, _ = _pawn_attacks_of(bb, WHITE)
    black_pawn_att, _ = _pawn_attacks_of(bb, BLACK)
    white_span = _pawn_attack_span(bb, WHITE)
    black_span = _pawn_attack_span(bb, BLACK)
    white_blockers = _king_blockers(bb, WHITE)
    black_blockers = _king_blockers(bb, BLACK)
    white_area = _mobility_area(bb, WHITE, black_pawn_att, white_blockers)
    black_area = _mobility_area(bb, BLACK, white_pawn_att, black_blockers)

    white_score, w_hits, w_units, w_attackers = _side_score(
        bb, WHITE, phase, white_pawn_att, black_pawn_att, black_span,
        white_area, white_blockers)
    black_score, b_hits, b_units, b_attackers = _side_score(
        bb, BLACK, phase, black_pawn_att, white_pawn_att, white_span,
        black_area, black_blockers)

    white_attacks = _all_attacks(bb, WHITE)
    black_attacks = _all_attacks(bb, BLACK)

    # danger to white's king comes from black's attack accumulation
    white_score += _king_safety(bb, WHITE, phase, b_units, b_attackers,
                                b_hits, black_attacks, white_attacks)
    black_score += _king_safety(bb, BLACK, phase, w_units, w_attackers,
                                w_hits, white_attacks, black_attacks)

    white_score += _shelter_storm(bb, WHITE, lsb(bb[K]))
    black_score += _shelter_storm(bb, BLACK, lsb(bb[K + 6]))

    score = white_score - black_score
    if st[SIDE] == WHITE:
        return score + TEMPO
    return -score + TEMPO
