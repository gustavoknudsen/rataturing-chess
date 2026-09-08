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

import os

import numpy as np
from numba import int64, njit, uint64

# Compile budget is a live constraint (see docs/PROGRESS.md), so the heavier
# optional terms are toggleable and can be cut without touching the code.
# Threats is on: it scored 63.3% over 30 games once the specialised endgames
# were in place (50.0% without them, because it inflates the score in won
# positions and that hid mates), and its own compile cost is ~1 s.
USE_THREATS = os.environ.get("BTC_THREATS", "1") == "1"
USE_ENDGAMES = os.environ.get("BTC_ENDGAMES", "1") == "1"
USE_SPACE = os.environ.get("BTC_SPACE", "1") == "1"

from btc_core import (
    B, BLACK, K, N, OCC_A, OCC_B, OCC_W, ONE, P, Q, R, SIDE, WHITE, ZERO,
    bishop_attacks, count_bits, KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS,
    lsb, queen_attacks, rook_attacks,
)
from btc_evalmasks import (
    ADJACENT_FILES, BETWEEN, BLACK_FORWARD_FILE, BLACK_KING_ZONE, CENTER_FILES,
    BLACK_PASSED, BLACK_SUPPORT, FILE_MASK, FILE_OF, ISOLATED, KING_FLANK,
    LINE, PHALANX, RANK_MASK, RANK_OF, RELATIVE_RANK, WHITE_FORWARD_FILE,
    WHITE_KING_ZONE, WHITE_PASSED, WHITE_SUPPORT,
)
from btc_endgame import insufficient_material
from btc_endgame import probe as endgame_probe
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

# Polynomial material imbalance, indexed [piece1][piece2][phase].
# Piece index 0 is the "bishop pair" pseudo-piece, then P N B R Q as 1..5.
QUADRATIC_OURS = np.array([
    [[1419, 1455], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[101, 28], [37, 39], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[57, 64], [249, 187], [-49, -62], [0, 0], [0, 0], [0, 0]],
    [[0, 0], [118, 137], [10, 27], [0, 0], [0, 0], [0, 0]],
    [[-63, -68], [-5, 3], [100, 81], [132, 118], [-246, -244], [0, 0]],
    [[-210, -211], [37, 14], [147, 141], [161, 105], [-158, -174], [-9, -31]],
], dtype=np.int64)

QUADRATIC_THEIRS = np.array([
    [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[33, 30], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[46, 18], [106, 84], [0, 0], [0, 0], [0, 0], [0, 0]],
    [[75, 35], [59, 44], [60, 15], [0, 0], [0, 0], [0, 0]],
    [[26, 35], [6, 22], [38, 39], [-12, -2], [0, 0], [0, 0]],
    [[97, 93], [100, 163], [-58, -91], [112, 192], [276, 225], [0, 0]],
], dtype=np.int64)

SPACE_THRESHOLD = 11551

# Threats, indexed by victim piece type (P N B R Q).
THREAT_BY_MINOR_MG = np.array([6, 64, 82, 103, 81, 0], dtype=np.int64)
THREAT_BY_MINOR_EG = np.array([37, 50, 57, 130, 163, 0], dtype=np.int64)
THREAT_BY_ROOK_MG = np.array([54, 56, 66, 86, 0, 0], dtype=np.int64)
THREAT_BY_ROOK_EG = np.array([42, 43, 44, 60, 0, 0], dtype=np.int64)
THREAT_BY_KING_MG, THREAT_BY_KING_EG = 24, 87
HANGING_MG, HANGING_EG = 72, 40
RESTRICTED_MG, RESTRICTED_EG = 6, 7
THREAT_BY_SAFE_PAWN_MG, THREAT_BY_SAFE_PAWN_EG = 167, 99
WEAK_QUEEN_PROTECTION_MG, WEAK_QUEEN_PROTECTION_EG = 14, 0
THREAT_BY_PAWN_PUSH_MG, THREAT_BY_PAWN_PUSH_EG = 48, 39
KNIGHT_ON_QUEEN_MG, KNIGHT_ON_QUEEN_EG = 16, 11
SLIDER_ON_QUEEN_MG, SLIDER_ON_QUEEN_EG = 62, 21


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
def _accumulate(bb, side):
    """Attack sets for one side. Returns
    (minor, rook, queen, king, all, doubly attacked).
    Computed once and shared by king safety and threats; the C engine rebuilt
    equivalent tables inside every makeMove instead."""
    occ = bb[OCC_A]
    base = 0 if side == WHITE else 6
    pawn_att, pawn_double = _pawn_attacks_of(bb, side)

    all_att = pawn_att
    double = pawn_double
    knight_att = ZERO
    bishop_att = ZERO
    rook_att = ZERO
    queen_att = ZERO
    king_att = ZERO

    for pt in range(N, K + 1):
        pieces = bb[base + pt]
        while pieces:
            sq = lsb(pieces)
            pieces &= pieces - ONE
            if pt == N:
                att = KNIGHT_ATTACKS[sq]
                knight_att |= att
            elif pt == B:
                att = bishop_attacks(sq, occ)
                bishop_att |= att
            elif pt == R:
                att = rook_attacks(sq, occ)
                rook_att |= att
            elif pt == Q:
                att = queen_attacks(sq, occ)
                queen_att |= att
            else:
                att = KING_ATTACKS[sq]
                king_att |= att
            double |= all_att & att
            all_att |= att
    return (knight_att, bishop_att, rook_att, queen_att, king_att, all_att,
            double)


@njit(cache=False, fastmath=True)
def _queen_threats(bb, us, our_knight, our_bishop, our_rook, our_double,
                   strongly_protected_them, area):
    """Squares from which we could fork or hit a lone enemy queen next move.
    Returns (knight hits, slider hits, doubled weight)."""
    them_base = 6 if us == WHITE else 0
    our_base = 0 if us == WHITE else 6
    enemy_queen = bb[them_base + Q]
    if count_bits(enemy_queen) != 1:
        return 0, 0, 1
    weight = 2 if count_bits(bb[Q] | bb[Q + 6]) == 1 else 1
    qsq = lsb(enemy_queen)
    occ = bb[OCC_A]
    safe_sq = area & ~bb[our_base + P] & ~strongly_protected_them
    knight_hits = count_bits(our_knight & KNIGHT_ATTACKS[qsq] & safe_sq)
    slider = (our_bishop & bishop_attacks(qsq, occ)) \
        | (our_rook & rook_attacks(qsq, occ))
    slider_hits = count_bits(slider & safe_sq & our_double)
    return knight_hits, slider_hits, weight


@njit(cache=False, fastmath=True)
def _pawn_push_threats(bb, us, our_all, their_all, their_pawn_att,
                       non_pawn_enemies):
    """Enemy pieces attacked by a pawn we could safely push next move."""
    our_base = 0 if us == WHITE else 6
    empty = ~bb[OCC_A]
    pawns = bb[our_base + P]
    if us == WHITE:
        pushes = (pawns >> uint64(8)) & empty
        pushes |= ((pushes & RANK_MASK[5]) >> uint64(8)) & empty
    else:
        pushes = (pawns << uint64(8)) & empty
        pushes |= ((pushes & RANK_MASK[2]) << uint64(8)) & empty
    pushes &= ~their_pawn_att & (~their_all | our_all)
    threatened = ZERO
    while pushes:
        sq = lsb(pushes)
        pushes &= pushes - ONE
        threatened |= PAWN_ATTACKS[us, sq]
    return count_bits(threatened & non_pawn_enemies)


@njit(cache=False, fastmath=True)
def _threats(bb, us, phase, our_knight, our_bishop, our_rook, our_king,
             our_all, our_double, their_all, their_pawn_att, their_double,
             their_queen_att, area):
    """Threats from `us`'s point of view, ported from BTC evaluateThreats
    (evaluation.h): threats by minor, rook and king on weak or defended
    enemies, hanging pieces, pieces protected only by the enemy queen,
    restricted enemy mobility, threats by safe pawns and by safe pawn pushes,
    and knight/slider forks against a lone enemy queen."""
    our_minor = our_knight | our_bishop
    them_base = 6 if us == WHITE else 0
    our_base = 0 if us == WHITE else 6
    enemy_occ = bb[OCC_B] if us == WHITE else bb[OCC_W]
    non_pawn_enemies = enemy_occ & ~bb[them_base + P]

    strongly_protected_them = their_pawn_att | (their_double & ~our_double)
    weak = enemy_occ & ~strongly_protected_them & our_all
    defended = non_pawn_enemies & strongly_protected_them

    mg = 0
    eg = 0
    if defended | weak:
        targets = (defended | weak) & our_minor
        for pt in range(P, K):
            hits = count_bits(targets & bb[them_base + pt])
            if hits:
                mg += THREAT_BY_MINOR_MG[pt] * hits
                eg += THREAT_BY_MINOR_EG[pt] * hits
        targets = weak & our_rook
        for pt in range(P, K):
            hits = count_bits(targets & bb[them_base + pt])
            if hits:
                mg += THREAT_BY_ROOK_MG[pt] * hits
                eg += THREAT_BY_ROOK_EG[pt] * hits
        if weak & our_king:
            mg += THREAT_BY_KING_MG
            eg += THREAT_BY_KING_EG
        hanging = weak & (~their_all | (non_pawn_enemies & our_double))
        hits = count_bits(hanging)
        mg += HANGING_MG * hits
        eg += HANGING_EG * hits
        hits = count_bits(weak & their_queen_att)
        mg += WEAK_QUEEN_PROTECTION_MG * hits
        eg += WEAK_QUEEN_PROTECTION_EG * hits

    restricted = their_all & ~strongly_protected_them & our_all
    hits = count_bits(restricted)
    mg += RESTRICTED_MG * hits
    eg += RESTRICTED_EG * hits

    # threats from pawns that are themselves reasonably safe
    safe = ~their_all | our_all
    our_pawns = bb[our_base + P]
    pawn_threat = ZERO
    tmp = our_pawns & safe
    while tmp:
        sq = lsb(tmp)
        tmp &= tmp - ONE
        pawn_threat |= PAWN_ATTACKS[us, sq]
    hits = count_bits(pawn_threat & non_pawn_enemies)
    mg += THREAT_BY_SAFE_PAWN_MG * hits
    eg += THREAT_BY_SAFE_PAWN_EG * hits

    hits = _pawn_push_threats(bb, us, our_all, their_all, their_pawn_att,
                              non_pawn_enemies)
    mg += THREAT_BY_PAWN_PUSH_MG * hits
    eg += THREAT_BY_PAWN_PUSH_EG * hits

    knight_hits, slider_hits, weight = _queen_threats(
        bb, us, our_knight, our_bishop, our_rook, our_double,
        strongly_protected_them, area)
    mg += KNIGHT_ON_QUEEN_MG * knight_hits * weight
    eg += KNIGHT_ON_QUEEN_EG * knight_hits * weight
    mg += SLIDER_ON_QUEEN_MG * slider_hits * weight
    eg += SLIDER_ON_QUEEN_EG * slider_hits * weight

    return _taper(mg, eg, phase)


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


# The twelve imbalance piece counts are each below 16, so they pack into one
# integer as 4-bit fields. This keeps the term allocation-free on the hot path.
@njit(cache=False, fastmath=True)
def _count_of(packed, side, pt):
    return int64((packed >> uint64(side * 24 + pt * 4)) & uint64(0xF))


@njit(cache=False, fastmath=True)
def _clamp15(v):
    return 15 if v > 15 else v


@njit(cache=False, fastmath=True)
def _pack_side(bb, base, shift):
    bishops = count_bits(bb[base + B])
    packed = uint64(1 if bishops > 1 else 0) << uint64(shift)
    packed |= uint64(_clamp15(count_bits(bb[base + P]))) << uint64(shift + 4)
    packed |= uint64(_clamp15(count_bits(bb[base + N]))) << uint64(shift + 8)
    packed |= uint64(_clamp15(bishops)) << uint64(shift + 12)
    packed |= uint64(_clamp15(count_bits(bb[base + R]))) << uint64(shift + 16)
    packed |= uint64(_clamp15(count_bits(bb[base + Q]))) << uint64(shift + 20)
    return packed


@njit(cache=False, fastmath=True)
def _pack_counts(bb):
    return _pack_side(bb, 0, 0) | _pack_side(bb, 6, 24)


@njit(cache=False, fastmath=True)
def _space(bb, us, phase, their_pawn_att, their_all, white_pawn_double,
           black_pawn_double):
    """Safe central squares in our own half, weighted by how crowded and how
    blocked the position is. Middlegame only: the caller tapers it away."""
    if phase < SPACE_THRESHOLD:
        return 0
    our_base = 0 if us == WHITE else 6
    our_pawns = bb[our_base + P]
    if us == WHITE:
        rank_mask = RANK_MASK[6] | RANK_MASK[5] | RANK_MASK[4]
    else:
        rank_mask = RANK_MASK[1] | RANK_MASK[2] | RANK_MASK[3]
    space_mask = CENTER_FILES & rank_mask
    safe_area = space_mask & ~our_pawns & ~their_pawn_att

    behind = our_pawns
    if us == WHITE:
        behind |= behind << uint64(8)
        behind |= behind << uint64(16)
    else:
        behind |= behind >> uint64(8)
        behind |= behind >> uint64(16)

    bonus = count_bits(safe_area) \
        + count_bits(behind & safe_area & ~their_all)

    # blocked pawns are counted for both colours regardless of `us`: a pawn
    # whose front square holds an enemy pawn or is covered by two enemy pawns
    blocked = count_bits((bb[P] >> uint64(8)) & (bb[P + 6] | black_pawn_double)) \
        + count_bits((bb[P + 6] << uint64(8)) & (bb[P] | white_pawn_double))
    if blocked > 9:
        blocked = 9
    own_occ = bb[OCC_W] if us == WHITE else bb[OCC_B]
    weight = count_bits(own_occ) - 3 + blocked
    return c_div(bonus * weight * weight, 16)


@njit(cache=False, fastmath=True)
def _imbalance_side(packed, us, phase_idx):
    them = 1 - us
    bonus = 0
    for pt1 in range(6):
        count1 = _count_of(packed, us, pt1)
        if count1 == 0:
            continue
        v = QUADRATIC_OURS[pt1, pt1, phase_idx] * count1
        for pt2 in range(pt1):
            v += QUADRATIC_OURS[pt1, pt2, phase_idx] * _count_of(packed, us, pt2) \
                + QUADRATIC_THEIRS[pt1, pt2, phase_idx] * _count_of(packed, them, pt2)
        bonus += count1 * v
    return bonus


@njit(cache=False, fastmath=True)
def _imbalance(bb, phase):
    """Piece-pair polynomial imbalance, from white's point of view. Subsumes
    the bishop pair via the index 0 pseudo-piece."""
    packed = _pack_counts(bb)
    mg = c_div(_imbalance_side(packed, WHITE, 0) - _imbalance_side(packed, BLACK, 0), 16)
    eg = c_div(_imbalance_side(packed, WHITE, 1) - _imbalance_side(packed, BLACK, 1), 16)
    return _taper(mg, eg, phase)


@njit(cache=False, fastmath=True)
def evaluate(bb, st):
    """Full evaluation from the side to move's point of view."""
    if USE_ENDGAMES:
        if insufficient_material(bb):
            return 0
        handled, exact = endgame_probe(bb, st)
        if handled:
            return exact if st[SIDE] == WHITE else -exact

    phase = game_phase(bb)

    white_pawn_att, white_pawn_double = _pawn_attacks_of(bb, WHITE)
    black_pawn_att, black_pawn_double = _pawn_attacks_of(bb, BLACK)
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

    (w_knight, w_bishop, w_rook, w_queen, w_king, white_attacks,
     w_double) = _accumulate(bb, WHITE)
    (b_knight, b_bishop, b_rook, b_queen, b_king, black_attacks,
     b_double) = _accumulate(bb, BLACK)

    # danger to white's king comes from black's attack accumulation
    white_score += _king_safety(bb, WHITE, phase, b_units, b_attackers,
                                b_hits, black_attacks, white_attacks)
    black_score += _king_safety(bb, BLACK, phase, w_units, w_attackers,
                                w_hits, white_attacks, black_attacks)

    white_score += _shelter_storm(bb, WHITE, lsb(bb[K]))
    black_score += _shelter_storm(bb, BLACK, lsb(bb[K + 6]))

    if USE_THREATS:
        white_score += _threats(bb, WHITE, phase, w_knight, w_bishop, w_rook,
                                w_king, white_attacks, w_double,
                                black_attacks, black_pawn_att, b_double,
                                b_queen, white_area)
        black_score += _threats(bb, BLACK, phase, b_knight, b_bishop, b_rook,
                                b_king, black_attacks, b_double,
                                white_attacks, white_pawn_att, w_double,
                                w_queen, black_area)

    score = white_score - black_score
    score += _imbalance(bb, phase)

    if USE_SPACE:
        # middlegame term: taper it away toward the endgame
        space = _space(bb, WHITE, phase, black_pawn_att, black_attacks,
                       white_pawn_double, black_pawn_double) \
            - _space(bb, BLACK, phase, white_pawn_att, white_attacks,
                     white_pawn_double, black_pawn_double)
        score += _taper(space, 0, phase)
    if st[SIDE] == WHITE:
        return score + TEMPO
    return -score + TEMPO
