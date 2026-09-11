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
USE_SCALE = os.environ.get("BTC_SCALE", "1") == "1"
# Initiative. Diverges from the oracle, so it is off by default and the
# oracle comparison must be run with it off.
USE_INITIATIVE = os.environ.get("BTC_INITIATIVE") == "1"

import btc_nnue
from btc_nnue import nnue_applies, nnue_eval

# The self-trained network replaces the whole hand-crafted positional total.
#
# Enabled by the *presence of net.npz*, not by a flag default. The platform
# gives us no way to set an environment variable, so a flag that defaults off
# could never be turned on in the submission; and a flag that defaults on would
# be wrong here in the repo, where there is no net. Keying on the file makes
# both cases right by construction: package.py ships net.npz only when the
# network has earned it, and a zip without it is exactly the hand-crafted
# engine. BTC_NNUE=0 still forces it off, which is what gives an A/B match its
# control arm once net.npz is sitting in the working directory.
_NET_PATH = btc_nnue.find_net() if os.environ.get("BTC_NNUE", "1") == "1" \
    else None
USE_NNUE = _NET_PATH is not None
if USE_NNUE:
    (NET_FT_W, NET_FT_B, NET_OUT_W, NET_OUT_B, NET_L1,
     NET_QA, NET_QB, NET_SCALE, NET_TABLE, NET_BUCKETS) = \
        btc_nnue.load(_NET_PATH)
    # Output buckets are carried by the weight shape rather than a separate
    # field, so a net saved before they existed needs no special case.
    NET_OUT_BUCKETS = int(NET_OUT_W.shape[0])
else:
    # Shapes still have to be valid for numba to type the dead branch.
    NET_FT_W = np.zeros((768, 1), dtype=np.int16)
    NET_FT_B = np.zeros(1, dtype=np.int32)
    NET_OUT_W = np.zeros((1, 2), dtype=np.int16)
    NET_OUT_B = np.zeros(1, dtype=np.int32)
    NET_TABLE = np.zeros(64, dtype=np.int32)
    NET_L1, NET_QA, NET_QB, NET_SCALE = 1, 255, 64, 400
    NET_BUCKETS = 1
    NET_OUT_BUCKETS = 1

# The network is trained on standard centipawns; this engine's scores are
# wider. Every search margin - RFP, futility, delta, the aspiration window -
# is a constant in engine units, so without this conversion the network's
# narrower output silently strengthens all of them and the result is a
# different search, not a different evaluation.
#
# Measured, not nominal: the ratio of the two evaluators' output spreads over
# held-out positions. The nominal pawn ratio of 126 would have made the
# network's scores about 35% too small.
#
# **This number belongs to one network and must be re-measured for any other.**
# Nets measured here have ranged from 96 to 194, and running a net at another
# net's units rescales every margin, so a stronger net can look broken. The
# platform passes no environment variables, so this default is what ships:
#
#     .venv/Scripts/python.exe nnue_gate.py <net.npz> D:/chess_nnue/bp2b 20000
NET_UNITS = int(os.environ.get("BTC_NET_UNITS", "98"))

from btc_core import (
    B, BLACK, K, N, OCC_A, OCC_B, OCC_W, ONE, P, Q, R, SIDE, WHITE, ZERO,
    bishop_attacks, count_bits, KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS,
    lsb, queen_attacks, rook_attacks,
)
from btc_evalmasks import (
    ADJACENT_FILES, BETWEEN, BLACK_FORWARD_FILE, BLACK_KING_ZONE, CAMP,
    CENTER_FILES,
    BLACK_PASSED, BLACK_SUPPORT, FILE_MASK, FILE_OF, ISOLATED, KING_FLANK,
    FORWARD_RANKS, LINE, PHALANX, RANK_MASK, RANK_OF, RELATIVE_RANK,
    WHITE_FORWARD_FILE,
    WHITE_KING_ZONE, WHITE_PASSED, WHITE_SUPPORT,
)
from btc_endgame import insufficient_material
from btc_endgame import probe as endgame_probe
from btc_scale import endgame_scale
from btc_psqt import MIRROR, PIECE_TABLES

# Files a-d and e-h, for the "pawns on both flanks" test in _initiative.
QUEEN_SIDE = FILE_MASK[0] | FILE_MASK[1] | FILE_MASK[2] | FILE_MASK[3]
KING_SIDE = FILE_MASK[4] | FILE_MASK[5] | FILE_MASK[6] | FILE_MASK[7]

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
DOUBLED_EARLY_MG, DOUBLED_EARLY_EG = -17, -7
WEAK_LEVER_MG, WEAK_LEVER_EG = -2, -57
# blocked pawn on the 5th and 6th relative rank, {mg, eg}; subtracted
BLOCKED_PAWN = np.array([[19, 8], [7, -3]], dtype=np.int64)

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
ROOK_CLOSED_MG, ROOK_CLOSED_EG = 10, 5
BISHOP_XRAY_MG, BISHOP_XRAY_EG = 4, 5
# pawns on the bishop's own colour, indexed by the bishop's distance to an edge
BISHOP_PAWNS_MG = np.array([3, 3, 2, 3], dtype=np.int64)
BISHOP_PAWNS_EG = np.array([8, 9, 7, 7], dtype=np.int64)
LIGHT_SQUARES = uint64(0x55AA55AA55AA55AA)
DARK_SQUARES = uint64(0xAA55AA55AA55AA55)

KD_WEAK_RING = 183
KD_UNSAFE_CHECK = 148
KD_KING_ATTACKS = 69
KD_FLANK_ATTACK = 3
KD_NO_QUEEN = 873
KD_INIT = 37
KD_BLOCKER = 98
KD_KNIGHT_DEFENSE = 100
KD_SHELTER = 6
KD_FLANK_DEFENSE = 4
PAWNLESS_FLANK_MG, PAWNLESS_FLANK_EG = 19, 97
FLANK_ATTACKS_MG, FLANK_ATTACKS_EG = 8, 0
# king on a (semi-)open file, indexed [our file has no pawn][theirs has none]
KING_ON_FILE = np.array([[[-18, 11], [-6, -3]],
                         [[-1, 5], [8, -3]]], dtype=np.int64)
# king-ring attacker weight and safe-check bonus, indexed by piece type
# {P, N, B, R, Q, K}; safe check is {single, multiple}
KING_ATTACK_WEIGHTS = np.array([0, 76, 46, 45, 14, 0], dtype=np.int64)
SAFE_CHECK = np.array([[0, 0], [805, 1292], [650, 984], [1071, 1886],
                       [730, 1128], [0, 0]], dtype=np.int64)

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


@njit(cache=False, fastmath=True, error_model='numpy')
def c_div(a, b):
    """C integer division: truncation toward zero. Requires b > 0."""
    q = a // b
    if q < 0 and q * b != a:
        q += 1
    return q


@njit(cache=False, fastmath=True, error_model='numpy')
def _taper(mg, eg, phase):
    return c_div(mg * phase + eg * (OPENING_PHASE - phase), OPENING_PHASE)


@njit(cache=False, fastmath=True, error_model='numpy')
def _stage_value(mg, eg, phase, stage):
    """BTC only interpolates inside the middlegame band.

    Outside it the raw table entry for that stage is used, so above
    OPENING_PHASE material a piece is worth its opening value rather than an
    extrapolation of it. The port tapered unconditionally, which cost about 14
    per queen and 10 per rook whenever the piece count was above the band -
    which is most of the opening and early middlegame. `stage` is 0 opening,
    1 endgame, 2 middlegame, matching PIECE_TABLES' first index."""
    if stage == 2:
        return _taper(mg, eg, phase)
    return mg if stage == 0 else eg


@njit(cache=False, fastmath=True, error_model='numpy')
def game_stage(phase):
    if phase > OPENING_PHASE:
        return 0
    if phase < ENDGAME_PHASE:
        return 1
    return 2


@njit(cache=False, fastmath=True, error_model='numpy')
def game_phase(bb):
    """Non-pawn material on both sides, BTC's getGameStageScore.

    Deliberately unclamped. The starting position holds 16604, above
    OPENING_PHASE of 15196, and BTC feeds that raw value to interpolate(), so
    a term is extrapolated past its own middlegame value while the queens are
    still on: interpolate(mg, eg, 16604) is mg*1.093 - eg*0.093. Clamping here
    made every tapered term disagree with BTC in the opening and early
    middlegame. Only the endgame scale factor floors the weight, and it does
    that itself."""
    score = 0
    for pc in range(N, Q + 1):
        score += (count_bits(bb[pc]) + count_bits(bb[pc + 6])) * MATERIAL_MG[pc]
    return score


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
def _passed_pawn(bb, sq, side, phase, enemy_king_sq, own_king_sq,
                 own_pawn_att, own_attacks, enemy_attacks):
    """Passed pawn bonus by relative rank and file, with a king-race term in
    the endgame. Returns 0 when the pawn is not passed."""
    enemy_pawns = bb[P + 6] if side == WHITE else bb[P]
    mask = WHITE_PASSED[sq] if side == WHITE else BLACK_PASSED[sq]
    if enemy_pawns & mask:
        return 0
    front = WHITE_FORWARD_FILE[sq] if side == WHITE else BLACK_FORWARD_FILE[sq]
    # No doubled-pawn exclusion: BTC tests only enemy pawns in the span, so a
    # pawn with a friendly pawn ahead of it on the same file still scores as
    # passed. The port used to return 0 there, which silently deleted the term
    # for the rear pawn of every doubled passer.

    rr = RELATIVE_RANK[side, sq]
    mg = PASSED_RANK_MG[rr]
    eg = PASSED_RANK_EG[rr]
    f = FILE_OF[sq]
    edge = f if f < 7 - f else 7 - f

    if rr > 2:
        push = sq - 8 if side == WHITE else sq + 8
        if 0 <= push <= 63:
            # BTC weights the whole king race by rank: w = 5r - 13, which runs
            # 2, 7, 12, 17 as the pawn advances. The port previously dropped w
            # and folded a constant 4 into the coefficients (19/4 -> 19,
            # 2 -> 8), which is right only at r == 3 and undervalues an
            # advanced passer badly: a pawn on the 7th was worth 323 here
            # against BTC's 611.
            w = 5 * rr - 13
            enemy_dist = _king_proximity(enemy_king_sq, push)
            own_dist = _king_proximity(own_king_sq, push)
            eg += (enemy_dist * 19 // 4 - own_dist * 2) * w
            # the square two ahead, so our king is rewarded for escorting
            # rather than merely reaching the stop square
            if rr != 6:
                ahead = push - 8 if side == WHITE else push + 8
                if 0 <= ahead <= 63:
                    eg -= _king_proximity(own_king_sq, ahead) * w
            if not (bb[OCC_A] & (ONE << uint64(push))):
                mg, eg = _passer_path(bb, sq, side, push, w, mask, front, mg,
                                      eg, own_pawn_att, own_attacks,
                                      enemy_attacks)
    # BTC tapers the file deduction on its own and multiplies afterwards, so
    # it truncates once rather than once per side of the mg/eg pair
    return _taper(mg, eg, phase) \
        - _taper(PASSED_FILE_MG, PASSED_FILE_EG, phase) * edge


@njit(cache=False, fastmath=True, error_model='numpy')
def _passer_path(bb, sq, side, push, w, span, to_queen, mg, eg, own_pawn_att,
                 own_attacks, enemy_attacks):
    """Reward a passer whose road to promotion is clear or covered.

    Ported from the `if (!getBit(occupancies[both], blockSq))` block of BTC's
    evaluatePassedPawn, which the port had omitted entirely. It is the largest
    part of the term for an advanced passer: k reaches 41 and w reaches 17, so
    it can add nearly 700 to both mg and eg.

    Deliberately not bit-exact with BTC. In the C the local `int r` shadows the
    black-rook piece enum, so `bitboards[r]` reads black pawns and any enemy
    pawn behind the passer is mistaken for a rook controlling the file, which
    deletes the bonus. See docs/BTC_UPSTREAM_ISSUES.md item 4."""
    them_occ = bb[OCC_B] if side == WHITE else bb[OCC_W]
    us_occ = bb[OCC_W] if side == WHITE else bb[OCC_B]
    behind = BLACK_FORWARD_FILE[sq] if side == WHITE else WHITE_FORWARD_FILE[sq]
    rooks_queens = bb[R] | bb[R + 6] | bb[Q] | bb[Q + 6]
    backers = behind & rooks_queens

    unsafe = span
    # an enemy rook or queen behind the pawn controls the whole file, so the
    # span stays unsafe; otherwise only enemy-held or enemy-covered squares are
    if not (them_occ & backers):
        unsafe &= enemy_attacks | them_occ

    block_bit = ONE << uint64(push)
    if not unsafe:
        k = 36
    elif not (unsafe & ~own_pawn_att):
        k = 30
    elif not (unsafe & to_queen):
        k = 17
    elif not (unsafe & block_bit):
        k = 7
    else:
        k = 0

    if (us_occ & backers) or (own_attacks & block_bit):
        k += 5
    return mg + k * w, eg + k * w


@njit(cache=False, fastmath=True, error_model='numpy')
def _chebyshev(a, b):
    dr = RANK_OF[a] - RANK_OF[b]
    df = FILE_OF[a] - FILE_OF[b]
    if dr < 0:
        dr = -dr
    if df < 0:
        df = -df
    return dr if dr > df else df


@njit(cache=False, fastmath=True, error_model='numpy')
def _king_proximity(ksq, sq):
    """Chebyshev distance capped at 5, as BTC's kingProximity. Without the cap
    the king-race term keeps growing past the range the weights were fitted
    for."""
    d = _chebyshev(ksq, sq)
    return d if d < 5 else 5


@njit(cache=False, fastmath=True, error_model='numpy')
def _pawn_structure(bb, side, enemy_att):
    """Per-pawn structure terms, ported from BTC pawnStructureTerms.

    Returns raw (mg, eg). BTC accumulates these over every pawn of both
    colours and interpolates the net once, so this must not taper.
    enemy_att is passed in: evaluate() already has both sides' pawn attacks,
    and recomputing them here cost a full pass over the pawns twice a node."""
    mg = 0
    eg = 0
    own = bb[P] if side == WHITE else bb[P + 6]
    enemy = bb[P + 6] if side == WHITE else bb[P]
    pawns = own
    while pawns:
        sq = lsb(pawns)
        pawns &= pawns - ONE
        m, e = _pawn_terms(sq, side, own, enemy, enemy_att)
        mg += m
        eg += e
    return mg, eg


@njit(cache=False, fastmath=True, error_model='numpy')
def _pawn_terms(sq, side, own, enemy, enemy_att):
    """Structure terms for one pawn, from its own side's point of view."""
    mg = 0
    eg = 0
    f = FILE_OF[sq]
    rr = RELATIVE_RANK[side, sq]
    up = -8 if side == WHITE else 8
    front = sq + up
    behind = sq - up

    forward_file = WHITE_FORWARD_FILE[sq]
    backward_file = BLACK_FORWARD_FILE[sq]
    if side != WHITE:
        forward_file = BLACK_FORWARD_FILE[sq]
        backward_file = WHITE_FORWARD_FILE[sq]
    adj = ADJACENT_FILES[f]

    opposed = (enemy & forward_file) != ZERO
    on_front = 0 <= front <= 63
    on_behind = 0 <= behind <= 63
    blocked = on_front and (enemy & (ONE << uint64(front))) != ZERO
    lever = enemy & PAWN_ATTACKS[side, sq]
    lever_push = ZERO
    if on_front:
        lever_push = enemy & PAWN_ATTACKS[side, front]
    doubled = on_behind and (own & (ONE << uint64(behind))) != ZERO
    neighbours = own & adj
    phalanx = neighbours & RANK_MASK[RANK_OF[sq]]
    support = ZERO
    if on_behind:
        support = neighbours & RANK_MASK[RANK_OF[behind]]

    # an early doubled pawn the enemy has not fixed in place
    if doubled:
        them_control = enemy | enemy_att
        if side == WHITE:
            fixed = own & (them_control << uint64(8))
        else:
            fixed = own & (them_control >> uint64(8))
        if not fixed:
            mg += DOUBLED_EARLY_MG
            eg += DOUBLED_EARLY_EG

    front_ranks_them = ZERO
    if on_front:
        front_ranks_them = FORWARD_RANKS[1 - side, front]
    is_backward = not (neighbours & front_ranks_them)         and (lever_push or blocked)

    if support or phalanx:
        v = CONNECTED_RANK[rr] * (2 + (1 if phalanx else 0)
                                  - (1 if opposed else 0))             + 22 * count_bits(support)
        mg += v
        eg += c_div(v * (rr - 2), 4)
    elif not neighbours:
        # isolated, unless it is a doubled pawn stuck behind an enemy pawn
        if opposed and (own & backward_file) and not (enemy & adj):
            mg += DOUBLED_MG
            eg += DOUBLED_EG
        else:
            unopp = 0 if opposed else 1
            mg += ISOLATED_MG + WEAK_UNOPPOSED_MG * unopp
            eg += ISOLATED_EG + WEAK_UNOPPOSED_EG * unopp
    elif is_backward:
        not_edge = 1 if 0 < f < 7 else 0
        unopp = 0 if opposed else 1
        mg += BACKWARD_MG + WEAK_UNOPPOSED_MG * unopp * not_edge
        eg += BACKWARD_EG + WEAK_UNOPPOSED_EG * unopp * not_edge

    if not support:
        dbl = 1 if doubled else 0
        weak_lever = 1 if count_bits(lever) > 1 else 0
        mg += DOUBLED_MG * dbl + WEAK_LEVER_MG * weak_lever
        eg += DOUBLED_EG * dbl + WEAK_LEVER_EG * weak_lever

    if blocked and (rr == 4 or rr == 5):
        mg -= BLOCKED_PAWN[rr - 4, 0]
        eg -= BLOCKED_PAWN[rr - 4, 1]

    return mg, eg

@njit(cache=False, fastmath=True, error_model='numpy')
def _shelter_storm(bb, side, ksq):
    """King shelter and pawn storm, ported from BTC getKingShelter and the
    tail of getKingSafety. Returns raw (mg, eg); the caller interpolates.

    The port previously returned one untapered number and omitted five things:
    the filter that ignores pawns behind the king, the +5/+5 base, the storm
    penalty's `theirRank == 2` gate (it applied -82 unconditionally), the
    king-on-open-file penalty, and the endgame distance-to-nearest-pawn term.
    The mg half also feeds king danger through kdShelter, so those errors
    propagated into the king-safety score as well."""
    own_all = bb[P] if side == WHITE else bb[P + 6]
    enemy_all = bb[P + 6] if side == WHITE else bb[P]
    # pawns behind our own king shelter nothing
    behind = FORWARD_RANKS[1 - side, ksq]
    own = own_all & ~behind
    enemy = enemy_all & ~behind

    mg = 5
    eg = 5
    center = FILE_OF[ksq]
    if center < 1:
        center = 1
    elif center > 6:
        center = 6

    for f in range(center - 1, center + 2):
        file_mask = FILE_MASK[f]
        ours = own & file_mask
        our_rank = RELATIVE_RANK[side, _front_most(ours, side)] if ours else 0
        theirs = enemy & file_mask
        their_rank = 0
        if theirs:
            their_rank = RELATIVE_RANK[side, _front_most(theirs, side)]
        d = f if f < 7 - f else 7 - f
        mg += SHELTER_STRENGTH[d, our_rank]
        if our_rank and our_rank == their_rank - 1:
            # a blocked storm only costs when the enemy pawn is on its 3rd
            if their_rank == 2:
                mg -= 82
                eg -= 82
        else:
            mg -= UNBLOCKED_STORM[d, their_rank]

    king_file = FILE_MASK[FILE_OF[ksq]]
    our_semi = 0 if (own_all & king_file) else 1
    their_semi = 0 if (enemy_all & king_file) else 1
    mg -= KING_ON_FILE[our_semi, their_semi, 0]
    eg -= KING_ON_FILE[our_semi, their_semi, 1]

    eg -= 16 * _min_pawn_distance(own_all, ksq)
    return mg, eg


@njit(cache=False, fastmath=True, error_model='numpy')
def _front_most(bbv, side):
    """BTC's frontMostSquare(!side, bb): the pawn nearest our own king."""
    return _msb(bbv) if side == WHITE else lsb(bbv)


@njit(cache=False, fastmath=True, error_model='numpy')
def _min_pawn_distance(pawns, ksq):
    """Manhattan distance from the king to its nearest own pawn, BTC's
    distance(). 0 when we have no pawns at all, capped at 8."""
    if not pawns:
        return 0
    if pawns & KING_ATTACKS[ksq]:
        return 1
    best = 8
    rest = pawns
    while rest:
        sq = lsb(rest)
        rest &= rest - ONE
        df = FILE_OF[ksq] - FILE_OF[sq]
        dr = RANK_OF[ksq] - RANK_OF[sq]
        if df < 0:
            df = -df
        if dr < 0:
            dr = -dr
        if df + dr < best:
            best = df + dr
    return best

@njit(cache=False, fastmath=True, error_model='numpy')
def _msb(bbv):
    idx = 0
    while bbv:
        idx = lsb(bbv)
        bbv &= bbv - ONE
    return idx


@njit(cache=False, fastmath=True, error_model='numpy')
def _minor_terms(bb, piece_type, sq, side, phase, own_pawn_att, enemy_pawn_att,
                 enemy_pawn_span):
    """Outposts, minor behind pawn, king protector, long diagonal bishop."""
    score = 0
    bit = ONE << uint64(sq)
    rr = RELATIVE_RANK[side, sq]
    # BTC guards the knight outpost with the enemy's current pawn attacks and
    # the bishop's with the pawn attack span, which is a strict superset. The
    # port used the span for both, so a knight lost its outpost whenever an
    # enemy pawn could still advance to challenge the square - the ordinary
    # case, such as a knight on d5 against a black pawn still on e7.
    challenged = enemy_pawn_att if piece_type == N else enemy_pawn_span
    outpost_ok = 3 <= rr <= 5 and (own_pawn_att & bit) \
        and not (challenged & bit)
    if outpost_ok:
        if piece_type == N:
            score += _taper(OUTPOST_KNIGHT_MG, OUTPOST_KNIGHT_EG, phase)
        else:
            score += _taper(OUTPOST_BISHOP_MG, OUTPOST_BISHOP_EG, phase)
    elif piece_type == N:
        own_occ = bb[OCC_W] if side == WHITE else bb[OCC_B]
        ranks = RANK_MASK[2] | RANK_MASK[3] | RANK_MASK[4] if side == WHITE \
            else RANK_MASK[3] | RANK_MASK[4] | RANK_MASK[5]
        targets = KNIGHT_ATTACKS[sq] & ranks & own_pawn_att & ~enemy_pawn_att \
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
        # BTC scores the knight as a bonus for closeness, (7 - distance), and
        # the bishop as a penalty for distance. The two are inconsistent in the
        # C as well, but the knight form carries a constant 7 * K the port was
        # dropping: 63 per knight, which only cancels when both sides have the
        # same number of them.
        score += _taper(KING_PROTECTOR_KNIGHT_MG, KING_PROTECTOR_KNIGHT_EG,
                        phase) * (7 - dist)
    else:
        score -= _taper(KING_PROTECTOR_BISHOP_MG, KING_PROTECTOR_BISHOP_EG,
                        phase) * dist
        score += _bishop_pawn_terms(bb, sq, side, phase, own_pawn_att)
    return score


@njit(cache=False, fastmath=True, error_model='numpy')
def _bishop_pawn_terms(bb, sq, side, phase, own_pawn_att):
    """Pawns on the bishop's colour, the long diagonal, and enemy pawns it
    x-rays through the queens. The first and last were missing from the port."""
    bit = ONE << uint64(sq)
    occ = bb[OCC_A]
    own_pawns = bb[P] if side == WHITE else bb[P + 6]
    enemy_pawns = bb[P + 6] if side == WHITE else bb[P]

    same_colour = DARK_SQUARES if _square_colour(sq) else LIGHT_SQUARES
    on_colour = count_bits(own_pawns & same_colour)
    ahead = (occ << uint64(8)) if side == WHITE else (occ >> uint64(8))
    blocked_centre = count_bits(own_pawns & ahead & CENTER_FILES)
    unprotected = 0 if (own_pawn_att & bit) else 1
    f = FILE_OF[sq]
    edge = f if f < 7 - f else 7 - f
    score = -_taper(BISHOP_PAWNS_MG[edge], BISHOP_PAWNS_EG[edge], phase)         * on_colour * (unprotected + blocked_centre)

    # long diagonal: BTC clears only its own pawns from the blockers
    through = bishop_attacks(sq, occ ^ own_pawns)
    if count_bits(through & CENTER) >= 2:
        score += _taper(LONG_DIAGONAL_BISHOP_MG, 0, phase)

    xray = bishop_attacks(sq, occ ^ (bb[Q] | bb[Q + 6]))
    score -= _taper(BISHOP_XRAY_MG, BISHOP_XRAY_EG, phase)         * count_bits(xray & enemy_pawns)
    return score


@njit(cache=False, fastmath=True, error_model='numpy')
def _square_colour(sq):
    return (sq & 1) ^ ((sq >> 3) & 1)


@njit(cache=False, fastmath=True, error_model='numpy')
def _rook_file_term(bb, sq, side, phase):
    own_pawns = bb[P] if side == WHITE else bb[P + 6]
    enemy_pawns = bb[P + 6] if side == WHITE else bb[P]
    file_mask = FILE_MASK[FILE_OF[sq]]
    if own_pawns & file_mask:
        # our own pawn on the file: a penalty if any of them is blocked
        occ = bb[OCC_A]
        ahead = (occ << uint64(8)) if side == WHITE else (occ >> uint64(8))
        if own_pawns & file_mask & ahead:
            return -_taper(ROOK_CLOSED_MG, ROOK_CLOSED_EG, phase)
        return 0
    if enemy_pawns & file_mask:
        return _taper(ROOK_OPEN_MG[0], ROOK_OPEN_EG[0], phase)
    return _taper(ROOK_OPEN_MG[1], ROOK_OPEN_EG[1], phase)


@njit(cache=False, fastmath=True, error_model='numpy')
def _side_score(bb, side, phase, own_pawn_att, enemy_pawn_att, enemy_pawn_span,
                area, blockers, own_attacks, enemy_attacks):
    """All per-piece terms for one side, from that side's point of view.

    King-ring attackers used to be accumulated here as well and returned
    alongside the score, which meant re-deriving every slider's attack set a
    second time per node. Nothing read them: king danger counts its own
    attackers in _kd_attackers."""
    score = 0
    base = 0 if side == WHITE else 6
    enemy_ksq = lsb(bb[K + 6]) if side == WHITE else lsb(bb[K])
    enemy_zone = BLACK_KING_ZONE[enemy_ksq] if side == WHITE \
        else WHITE_KING_ZONE[enemy_ksq]
    own_ksq = lsb(bb[K]) if side == WHITE else lsb(bb[K + 6])
    occ = bb[OCC_A]
    stage = game_stage(phase)

    for pt in range(P, K + 1):
        pieces = bb[base + pt]
        while pieces:
            sq = lsb(pieces)
            pieces &= pieces - ONE
            pst_sq = sq if side == WHITE else MIRROR[sq]
            score += _stage_value(MATERIAL_MG[pt], MATERIAL_EG[pt], phase,
                                  stage)
            score += _stage_value(PIECE_TABLES[0, pt, pst_sq],
                                  PIECE_TABLES[1, pt, pst_sq], phase, stage)

            if pt == P:
                score += _passed_pawn(bb, sq, side, phase, enemy_ksq, own_ksq,
                                      own_pawn_att, own_attacks,
                                      enemy_attacks)
                continue
            if pt == K:
                continue

            mob = _piece_mobility(bb, pt, sq, side, area, blockers)
            score += _stage_value(MOBILITY_MG[pt, mob], MOBILITY_EG[pt, mob],
                                  phase, stage)

            if pt == N or pt == B:
                score += _minor_terms(bb, pt, sq, side, phase, own_pawn_att,
                                      enemy_pawn_att, enemy_pawn_span)
            elif pt == R:
                score += _rook_file_term(bb, sq, side, phase)

    return score


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
def _king_safety(bb, side, phase, own_all, own_double, own_king_att,
                 own_queen_att, own_knight_att, own_pawn_double, enemy_all,
                 enemy_double, enemy_rook_att, enemy_queen_att,
                 enemy_bishop_att, enemy_knight_att, enemy_pawn_att,
                 shelter_mg):
    """King danger for `side`, ported from BTC getKingDanger.

    Returns the penalty to add to `side`'s score (so, negative). Sums small
    contributions - attacker weight, weak king-ring squares, safe and unsafe
    checks, slider blockers, king-flank pressure, and credits for no enemy
    queen, a defending knight, shelter and flank defence - then squares the
    total. The port previously had only the attacker, weak-ring, flank and
    no-queen terms, and its flank term was linear rather than quadratic, so it
    under-read an attack badly: a knight near the enemy king with queens on
    the board scored about 900 low against BTC."""
    them = 1 - side
    ksq = lsb(bb[K]) if side == WHITE else lsb(bb[K + 6])
    own_queen_bb = bb[Q] if side == WHITE else bb[Q + 6]
    their_occ = bb[OCC_B] if side == WHITE else bb[OCC_W]
    enemy_queens = bb[Q + 6] if side == WHITE else bb[Q]

    # squares by our king the enemy attacks and we defend at most once
    weak = enemy_all & ~own_double & (~own_all | own_king_att | own_queen_att)
    safe = ~their_occ & (~own_all | (weak & enemy_double))

    # slider rays from our king, with our own queen transparent
    occ = bb[OCC_A] ^ own_queen_bb
    rook_rays = rook_attacks(ksq, occ)
    bishop_rays = bishop_attacks(ksq, occ)

    danger, unsafe = _kd_checks(ksq, rook_rays, bishop_rays, safe,
                                enemy_rook_att, enemy_queen_att,
                                enemy_bishop_att, enemy_knight_att,
                                own_queen_att)

    kf = FILE_OF[ksq]
    if kf < 1:
        kf = 1
    elif kf > 6:
        kf = 6
    kr = 7 - RANK_OF[ksq]
    if kr < 1:
        kr = 1
    elif kr > 6:
        kr = 6
    ring_sq = (7 - kr) * 8 + kf
    king_ring = (KING_ATTACKS[ring_sq] | (ONE << uint64(ring_sq)))         & ~own_pawn_double

    flank = KING_FLANK[FILE_OF[ksq]] & CAMP[side]
    flank_atk = enemy_all & flank
    flank_attack = count_bits(flank_atk) + count_bits(flank_atk & enemy_double)
    flank_defense = count_bits(own_all & flank)

    count, weight, adj_hits = _kd_attackers(bb, them, king_ring,
                                            KING_ATTACKS[ksq], bb[OCC_A])
    count += count_bits(king_ring & enemy_pawn_att)

    danger += count * weight
    danger += KD_WEAK_RING * count_bits(king_ring & weak)
    danger += KD_UNSAFE_CHECK * count_bits(unsafe)
    danger += KD_BLOCKER * _kd_blockers(bb, side, ksq)
    danger += KD_KING_ATTACKS * adj_hits
    danger += c_div(KD_FLANK_ATTACK * flank_attack * flank_attack, 8)
    if not enemy_queens:
        danger -= KD_NO_QUEEN
    if own_knight_att & own_king_att:
        danger -= KD_KNIGHT_DEFENSE
    danger -= c_div(KD_SHELTER * shelter_mg, 8)
    danger -= KD_FLANK_DEFENSE * flank_defense
    danger += KD_INIT

    mg = 0
    eg = 0
    if danger > 100:
        mg = c_div(danger * danger, 4096)
        eg = c_div(danger, 16)
    if not ((bb[P] | bb[P + 6]) & KING_FLANK[FILE_OF[ksq]]):
        mg += PAWNLESS_FLANK_MG
        eg += PAWNLESS_FLANK_EG
    # linear pressure on the king's flank, applied whatever the danger total.
    # In a quiet position, where danger stays under the threshold, this is the
    # only king-danger output there is, and the port had no counterpart.
    mg += FLANK_ATTACKS_MG * flank_attack
    eg += FLANK_ATTACKS_EG * flank_attack
    return -_taper(mg, eg, phase)


@njit(cache=False, fastmath=True, error_model='numpy')
def _kd_checks(ksq, rook_rays, bishop_rays, safe, enemy_rook_att,
               enemy_queen_att, enemy_bishop_att, enemy_knight_att,
               own_queen_att):
    """Safe-check bonuses, and the unsafe checks that remain. Each attacker is
    only credited for squares a more valuable attacker cannot already use."""
    danger = 0
    unsafe = ZERO

    rook_checks = rook_rays & enemy_rook_att & safe
    if rook_checks:
        danger += SAFE_CHECK[R, 1 if count_bits(rook_checks) > 1 else 0]
    else:
        unsafe |= rook_rays & enemy_rook_att

    queen_checks = (rook_rays | bishop_rays) & enemy_queen_att & safe         & ~(own_queen_att | rook_checks)
    if queen_checks:
        danger += SAFE_CHECK[Q, 1 if count_bits(queen_checks) > 1 else 0]

    bishop_checks = bishop_rays & enemy_bishop_att & safe & ~queen_checks
    if bishop_checks:
        danger += SAFE_CHECK[B, 1 if count_bits(bishop_checks) > 1 else 0]
    else:
        unsafe |= bishop_rays & enemy_bishop_att

    knight_checks = KNIGHT_ATTACKS[ksq] & enemy_knight_att
    if knight_checks & safe:
        n = count_bits(knight_checks & safe)
        danger += SAFE_CHECK[N, 1 if n > 1 else 0]
    else:
        unsafe |= knight_checks
    return danger, unsafe


@njit(cache=False, fastmath=True, error_model='numpy')
def _kd_attackers(bb, them, king_ring, king_adj, occ):
    """Enemy pieces whose attacks reach the king ring: how many, their summed
    weight, and how many squares directly beside the king they hit."""
    base = 0 if them == WHITE else 6
    count = 0
    weight = 0
    adj_hits = 0
    for pt in range(N, K):
        pieces = bb[base + pt]
        while pieces:
            sq = lsb(pieces)
            pieces &= pieces - ONE
            if pt == N:
                att = KNIGHT_ATTACKS[sq]
            elif pt == B:
                att = bishop_attacks(sq, occ)
            elif pt == R:
                att = rook_attacks(sq, occ)
            else:
                att = queen_attacks(sq, occ)
            if att & king_ring:
                count += 1
                weight += KING_ATTACK_WEIGHTS[pt]
                adj_hits += count_bits(att & king_adj)
    return count, weight, adj_hits


@njit(cache=False, fastmath=True, error_model='numpy')
def _kd_blockers(bb, side, ksq):
    """Pieces of either colour that alone block an enemy slider from the king.
    Snipers are found with the board cleared, then a single occupied square
    between king and sniper is a blocker."""
    enemy_rq = (bb[R + 6] | bb[Q + 6]) if side == WHITE else (bb[R] | bb[Q])
    enemy_bq = (bb[B + 6] | bb[Q + 6]) if side == WHITE else (bb[B] | bb[Q])
    snipers = (rook_attacks(ksq, ZERO) & enemy_rq)         | (bishop_attacks(ksq, ZERO) & enemy_bq)
    sniper_occ = bb[OCC_A] ^ snipers
    blockers = ZERO
    while snipers:
        sq = lsb(snipers)
        snipers &= snipers - ONE
        between = BETWEEN[ksq, sq] & sniper_occ
        if between and not (between & (between - ONE)):
            blockers |= between
    return count_bits(blockers)

# The twelve imbalance piece counts are each below 16, so they pack into one
# integer as 4-bit fields. This keeps the term allocation-free on the hot path.
@njit(cache=False, fastmath=True, error_model='numpy')
def _count_of(packed, side, pt):
    return int64((packed >> uint64(side * 24 + pt * 4)) & uint64(0xF))


@njit(cache=False, fastmath=True, error_model='numpy')
def _clamp15(v):
    return 15 if v > 15 else v


@njit(cache=False, fastmath=True, error_model='numpy')
def _pack_side(bb, base, shift):
    bishops = count_bits(bb[base + B])
    packed = uint64(1 if bishops > 1 else 0) << uint64(shift)
    packed |= uint64(_clamp15(count_bits(bb[base + P]))) << uint64(shift + 4)
    packed |= uint64(_clamp15(count_bits(bb[base + N]))) << uint64(shift + 8)
    packed |= uint64(_clamp15(bishops)) << uint64(shift + 12)
    packed |= uint64(_clamp15(count_bits(bb[base + R]))) << uint64(shift + 16)
    packed |= uint64(_clamp15(count_bits(bb[base + Q]))) << uint64(shift + 20)
    return packed


@njit(cache=False, fastmath=True, error_model='numpy')
def _pack_counts(bb):
    return _pack_side(bb, 0, 0) | _pack_side(bb, 6, 24)


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
def _imbalance(bb, phase):
    """Piece-pair polynomial imbalance, from white's point of view. Subsumes
    the bishop pair via the index 0 pseudo-piece."""
    packed = _pack_counts(bb)
    mg = c_div(_imbalance_side(packed, WHITE, 0) - _imbalance_side(packed, BLACK, 0), 16)
    eg = c_div(_imbalance_side(packed, WHITE, 1) - _imbalance_side(packed, BLACK, 1), 16)
    return _taper(mg, eg, phase)


@njit(cache=False, fastmath=True, error_model='numpy')
def _passed_count(bb):
    """Passed pawns of both colours."""
    white = bb[P]
    black = bb[P + 6]
    total = 0
    pieces = white
    while pieces:
        sq = lsb(pieces)
        pieces &= pieces - ONE
        if not (WHITE_PASSED[sq] & black):
            total += 1
    pieces = black
    while pieces:
        sq = lsb(pieces)
        pieces &= pieces - ONE
        if not (BLACK_PASSED[sq] & white):
            total += 1
    return total


@njit(cache=False, fastmath=True, error_model='numpy')
def _initiative(bb, phase, score):
    """How winnable the position is, as opposed to how good it looks.

    Pulls sterile positions toward a draw - no passed pawns, pawns on one flank
    only, kings not outflanking - and rewards real winning chances.

    The correction is clamped against the tapered score rather than against a
    separate middlegame/endgame pair, because every term here tapers into one
    integer. That is near-exact at both ends of the taper, and the endgame end
    is where this term does its work.

    Square 0 is a8, so rank indices run 8 down to 1."""
    w_king = lsb(bb[K])
    b_king = lsb(bb[K + 6])
    df = FILE_OF[w_king] - FILE_OF[b_king]
    dr = RANK_OF[w_king] - RANK_OF[b_king]
    outflanking = (df if df >= 0 else -df) - (dr if dr >= 0 else -dr)

    pawns = bb[P] | bb[P + 6]
    both_flanks = 1 if (pawns & QUEEN_SIDE) and (pawns & KING_SIDE) else 0
    passed = _passed_count(bb)
    infiltration = 1 if RANK_OF[w_king] <= 3 or RANK_OF[b_king] >= 4 else 0
    pieces = (bb[N] | bb[B] | bb[R] | bb[Q]
              | bb[N + 6] | bb[B + 6] | bb[R + 6] | bb[Q + 6])
    no_pieces = 1 if pieces == ZERO else 0
    almost_unwinnable = 1 if passed == 0 and outflanking < 0 \
        and both_flanks == 0 else 0

    complexity = (9 * passed
                  + 11 * count_bits(pawns)
                  + 9 * outflanking
                  + 21 * both_flanks
                  + 24 * infiltration
                  + 51 * no_pieces
                  - 43 * almost_unwinnable
                  - 110)

    magnitude = score if score >= 0 else -score
    sign = 1 if score > 0 else (-1 if score < 0 else 0)

    u = complexity + 50
    if u > 0:
        u = 0
    if u < -magnitude:
        u = -magnitude
    v = complexity
    if v < -magnitude:
        v = -magnitude
    return _taper(sign * u, sign * v, phase)


# Material-only fallback, used instead of the hand-crafted evaluation when the
# network is shipped. This is what keeps _hand_crafted out of the compiled
# build entirely: with USE_NNUE a compile-time constant, numba never reaches it,
# and the ~27 s that the full evaluation costs to compile disappears from init.
# It matters because the incremental accumulator pushed init to 42.2 s against a
# 90 s platform budget at 1.8x - too little headroom to be safe.
#
# It only ever runs below NNUE_MIN_PIECES, where insufficient_material and
# endgame_probe already answer nearly everything (KPK, KBNvK, KRvK, KQvK). This
# is the last-resort arm for the handful of tiny positions they decline.
_MAT = np.array([100, 320, 330, 500, 900, 0], dtype=np.int64)


@njit(cache=False, fastmath=True, error_model='numpy')
def _material_only(bb, st):
    score = 0
    for piece in range(6):
        score += _MAT[piece] * count_bits(bb[piece])
        score -= _MAT[piece] * count_bits(bb[piece + 6])
    return score if st[SIDE] == WHITE else -score


@njit(cache=False, fastmath=True, error_model='numpy')
def _specialised(bb, st):
    """(handled, score) for the endgames that own their own evaluation."""
    if insufficient_material(bb):
        return True, 0
    handled, exact = endgame_probe(bb, st)
    if handled:
        return True, exact if st[SIDE] == WHITE else -exact
    return False, 0


@njit(cache=False, fastmath=True, error_model='numpy')
def evaluate(bb, st):
    """Full evaluation from the side to move's point of view.

    Rebuilds the accumulator from the board. The search uses evaluate_cached
    instead, which reads the accumulator the move loop already maintains; this
    entry point is for callers that have no accumulator - tests, the gate, and
    the root before the search starts."""
    if USE_ENDGAMES:
        handled, score = _specialised(bb, st)
        if handled:
            return score
    # After the specialised endgames, never before: below six pieces this
    # function returns a mating drive rather than a positional score, and the
    # network was never trained on that scale. It would talk the search out of
    # winning KBNvK exactly the way delta pruning and correction history did.
    if USE_NNUE and nnue_applies(bb):
        centipawns = nnue_eval(
            bb, st[SIDE], NET_FT_W, NET_FT_B, NET_OUT_W, NET_OUT_B,
            NET_L1, NET_QA, NET_QB, NET_SCALE, NET_TABLE, NET_BUCKETS)
        return centipawns * NET_UNITS // 100
    if USE_NNUE:
        return _material_only(bb, st)
    return _hand_crafted(bb, st)


@njit(cache=False, fastmath=True, error_model='numpy')
def evaluate_cached(bb, st, acc_row):
    """Evaluation using an accumulator the caller already maintains.

    Identical to evaluate() except that the network path reads acc_row rather
    than rebuilding it from 32 pieces. test_accumulator.py asserts the two agree
    at every node of a move-tree walk, so the only difference is cost."""
    if USE_ENDGAMES:
        handled, score = _specialised(bb, st)
        if handled:
            return score
    if USE_NNUE and nnue_applies(bb):
        centipawns = btc_nnue.propagate(
            acc_row, st[SIDE], NET_OUT_W, NET_OUT_B,
            NET_L1, NET_QA, NET_QB, NET_SCALE,
            btc_nnue._out_bucket(bb, NET_OUT_BUCKETS))
        return centipawns * NET_UNITS // 100
    if USE_NNUE:
        return _material_only(bb, st)
    return _hand_crafted(bb, st)


@njit(cache=False, fastmath=True, error_model='numpy')
def _hand_crafted(bb, st):
    """The hand-crafted positional evaluation, side-to-move relative."""
    phase = game_phase(bb)

    # int64() rather than the bare WHITE/BLACK constants: numba specialises
    # on integer literals, so passing the constants compiled every function
    # reached from here twice, once per colour, doubling the eval's share of
    # the init budget. See docs/PROGRESS.md.
    side_w = int64(WHITE)
    side_b = int64(BLACK)

    white_pawn_att, white_pawn_double = _pawn_attacks_of(bb, side_w)
    black_pawn_att, black_pawn_double = _pawn_attacks_of(bb, side_b)
    white_span = _pawn_attack_span(bb, side_w)
    black_span = _pawn_attack_span(bb, side_b)
    white_blockers = _king_blockers(bb, side_w)
    black_blockers = _king_blockers(bb, side_b)
    white_area = _mobility_area(bb, side_w, black_pawn_att, white_blockers)
    black_area = _mobility_area(bb, side_b, white_pawn_att, black_blockers)

    # accumulate first: the passed-pawn path bonus in _side_score needs both
    # sides' full attack sets, and _accumulate depends only on the board
    (w_knight, w_bishop, w_rook, w_queen, w_king, white_attacks,
     w_double) = _accumulate(bb, side_w)
    (b_knight, b_bishop, b_rook, b_queen, b_king, black_attacks,
     b_double) = _accumulate(bb, side_b)

    white_score = _side_score(
        bb, side_w, phase, white_pawn_att, black_pawn_att, black_span,
        white_area, white_blockers, white_attacks, black_attacks)
    black_score = _side_score(
        bb, side_b, phase, black_pawn_att, white_pawn_att, white_span,
        black_area, black_blockers, black_attacks, white_attacks)

    # shelter first: its mg total feeds back into king danger, as in BTC
    w_sh_mg, w_sh_eg = _shelter_storm(bb, side_w, lsb(bb[K]))
    b_sh_mg, b_sh_eg = _shelter_storm(bb, side_b, lsb(bb[K + 6]))
    white_score += _taper(w_sh_mg, w_sh_eg, phase)
    black_score += _taper(b_sh_mg, b_sh_eg, phase)

    white_score += _king_safety(
        bb, side_w, phase, white_attacks, w_double, w_king, w_queen, w_knight,
        white_pawn_double, black_attacks, b_double, b_rook, b_queen, b_bishop,
        b_knight, black_pawn_att, w_sh_mg)
    black_score += _king_safety(
        bb, side_b, phase, black_attacks, b_double, b_king, b_queen, b_knight,
        black_pawn_double, white_attacks, w_double, w_rook, w_queen, w_bishop,
        w_knight, white_pawn_att, b_sh_mg)

    if USE_THREATS:
        white_score += _threats(bb, side_w, phase, w_knight, w_bishop, w_rook,
                                w_king, white_attacks, w_double,
                                black_attacks, black_pawn_att, b_double,
                                b_queen, white_area)
        black_score += _threats(bb, side_b, phase, b_knight, b_bishop, b_rook,
                                b_king, black_attacks, b_double,
                                white_attacks, white_pawn_att, w_double,
                                w_queen, black_area)

    score = white_score - black_score
    # net across both colours, then interpolate once: BTC's
    # computePawnStructure accumulates netMg/netEg over every pawn of both
    # sides and calls interpolate a single time
    w_ps_mg, w_ps_eg = _pawn_structure(bb, side_w, black_pawn_att)
    b_ps_mg, b_ps_eg = _pawn_structure(bb, side_b, white_pawn_att)
    score += _taper(w_ps_mg - b_ps_mg, w_ps_eg - b_ps_eg, phase)
    score += _imbalance(bb, phase)

    if USE_SPACE:
        # middlegame term: taper it away toward the endgame
        space = _space(bb, side_w, phase, black_pawn_att, black_attacks,
                       white_pawn_double, black_pawn_double) \
            - _space(bb, side_b, phase, white_pawn_att, white_attacks,
                     white_pawn_double, black_pawn_double)
        score += _taper(space, 0, phase)

    # tempo joins the white-relative total before scaling, as in BTC, where it
    # is added at the top of evaluate() and the scale factor applies to it too
    score += TEMPO if st[SIDE] == WHITE else -TEMPO

    # Applied to the positional total, before the scale factor.
    if USE_INITIATIVE:
        score += _initiative(bb, phase, score)

    if USE_SCALE:
        sf = endgame_scale(bb, st, score)
        if sf != 64:
            eg_weight = OPENING_PHASE - phase
            if eg_weight < 0:
                eg_weight = 0
            num = OPENING_PHASE * 64 - eg_weight * (64 - sf)
            score = c_div(score * num, OPENING_PHASE * 64)

    return score if st[SIDE] == WHITE else -score
