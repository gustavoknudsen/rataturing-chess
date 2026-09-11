"""Bitboard core ported from BetterThanCris v2.8 (C++).

Covers board state, magic attack tables, pseudo-legal move generation,
copy-make with legality test, Zobrist hashing and perft. Square mapping
a8=0..h1=63 and piece indices P N B R Q K p n b r q k = 0..11 match the
C engine exactly.

Numba 0.67 constraints this module is built around (docs/DESIGN_DECISIONS.md):
module-global arrays are read-only inside njit, so constant tables are globals
and all mutable state is passed as arguments; bitboards are strict np.uint64.

State layout:
  bb: uint64[16]  0..11 piece bitboards, 12/13/14 white/black/all occupancy,
                  15 zobrist hash
  st: int64[6]    0 side, 1 en passant square (64 none), 2 castle rights,
                  3 fifty counter, 4 game halfmove number, 5 spare
"""

import numpy as np
from numba import int64, njit, uint64

U64 = np.uint64

P, N, B, R, Q, K, p, n, b, r, q, k = range(12)
WHITE, BLACK, BOTH = 0, 1, 2
NO_SQ = 64

OCC_W, OCC_B, OCC_A, HASH = 12, 13, 14, 15
SIDE, EP, CASTLE, FIFTY, MOVENUM = 0, 1, 2, 3, 4

WK_CASTLE, WQ_CASTLE, BK_CASTLE, BQ_CASTLE = 1, 2, 4, 8

a8, b8, c8, d8, e8, f8, g8, h8 = range(8)
a1, b1, c1, d1, e1, f1, g1, h1 = range(56, 64)

MAX_PLY = 128

ONE = U64(1)
ZERO = U64(0)

CAP_FLAG = 1 << 20
DBL_FLAG = 1 << 21
EP_FLAG = 1 << 22
CASTLE_FLAG = 1 << 23

NOT_A_FILE = U64(18374403900871474942)
NOT_H_FILE = U64(9187201950435737471)
NOT_AB_FILE = U64(18229723555195321596)
NOT_GH_FILE = U64(4557430888798830399)

CASTLING_RIGHTS = np.array([
     7, 15, 15, 15,  3, 15, 15, 11,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    13, 15, 15, 15, 12, 15, 15, 14,
], dtype=np.int64)

BISHOP_BITS = np.array([
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6,
], dtype=np.int64)

ROOK_BITS = np.array([
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12,
], dtype=np.int64)

ROOK_MAGICS = np.array([
    0x8a80104000800020, 0x140002000100040, 0x2801880a0017001, 0x100081001000420,
    0x200020010080420, 0x3001c0002010008, 0x8480008002000100, 0x2080088004402900,
    0x800098204000, 0x2024401000200040, 0x100802000801000, 0x120800800801000,
    0x208808088000400, 0x2802200800400, 0x2200800100020080, 0x801000060821100,
    0x80044006422000, 0x100808020004000, 0x12108a0010204200, 0x140848010000802,
    0x481828014002800, 0x8094004002004100, 0x4010040010010802, 0x20008806104,
    0x100400080208000, 0x2040002120081000, 0x21200680100081, 0x20100080080080,
    0x2000a00200410, 0x20080800400, 0x80088400100102, 0x80004600042881,
    0x4040008040800020, 0x440003000200801, 0x4200011004500, 0x188020010100100,
    0x14800401802800, 0x2080040080800200, 0x124080204001001, 0x200046502000484,
    0x480400080088020, 0x1000422010034000, 0x30200100110040, 0x100021010009,
    0x2002080100110004, 0x202008004008002, 0x20020004010100, 0x2048440040820001,
    0x101002200408200, 0x40802000401080, 0x4008142004410100, 0x2060820c0120200,
    0x1001004080100, 0x20c020080040080, 0x2935610830022400, 0x44440041009200,
    0x280001040802101, 0x2100190040002085, 0x80c0084100102001, 0x4024081001000421,
    0x20030a0244872, 0x12001008414402, 0x2006104900a0804, 0x1004081002402,
], dtype=np.uint64)

BISHOP_MAGICS = np.array([
    0x40040844404084, 0x2004208a004208, 0x10190041080202, 0x108060845042010,
    0x581104180800210, 0x2112080446200010, 0x1080820820060210, 0x3c0808410220200,
    0x4050404440404, 0x21001420088, 0x24d0080801082102, 0x1020a0a020400,
    0x40308200402, 0x4011002100800, 0x401484104104005, 0x801010402020200,
    0x400210c3880100, 0x404022024108200, 0x810018200204102, 0x4002801a02003,
    0x85040820080400, 0x810102c808880400, 0xe900410884800, 0x8002020480840102,
    0x220200865090201, 0x2010100a02021202, 0x152048408022401, 0x20080002081110,
    0x4001001021004000, 0x800040400a011002, 0xe4004081011002, 0x1c004001012080,
    0x8004200962a00220, 0x8422100208500202, 0x2000402200300c08, 0x8646020080080080,
    0x80020a0200100808, 0x2010004880111000, 0x623000a080011400, 0x42008c0340209202,
    0x209188240001000, 0x400408a884001800, 0x110400a6080400, 0x1840060a44020800,
    0x90080104000041, 0x201011000808101, 0x1a2208080504f080, 0x8012020600211212,
    0x500861011240000, 0x180806108200800, 0x4000020e01040044, 0x300000261044000a,
    0x802241102020002, 0x20906061210001, 0x5a84841004010310, 0x4010801011c04,
    0xa010109502200, 0x4a02012000, 0x500201010098b028, 0x8040002811040900,
    0x28000010020204, 0x6000020202d0240, 0x8918844842082200, 0x4010011029020020,
], dtype=np.uint64)

ROOK_SHIFTS = (64 - ROOK_BITS).astype(np.uint64)
BISHOP_SHIFTS = (64 - BISHOP_BITS).astype(np.uint64)

DEBRUIJN64 = U64(0x03F79D71B4CB0A89)
INDEX64 = np.array([
     0, 47,  1, 56, 48, 27,  2, 60,
    57, 49, 41, 37, 28, 16,  3, 61,
    54, 58, 35, 52, 50, 42, 21, 44,
    38, 32, 29, 23, 17, 11,  4, 62,
    46, 55, 26, 59, 40, 36, 15, 53,
    34, 51, 20, 43, 31, 22, 10, 45,
    25, 39, 14, 33, 19, 30,  9, 24,
    13, 18,  8, 12,  7,  6,  5, 63,
], dtype=np.int64)


def _xorshift32_stream(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        yield state


def _init_zobrist():
    """Bit-identical to the C engine's initRandomKeys (seed 1804289383)."""
    rng = _xorshift32_stream(1804289383)

    def random64():
        parts = [next(rng) & 0xFFFF for _ in range(4)]
        return parts[0] | (parts[1] << 16) | (parts[2] << 32) | (parts[3] << 48)

    piece_keys = np.zeros((12, 64), dtype=np.uint64)
    for piece in range(12):
        for sq in range(64):
            piece_keys[piece, sq] = random64()
    ep_keys = np.array([random64() for _ in range(64)], dtype=np.uint64)
    castle_keys = np.array([random64() for _ in range(16)], dtype=np.uint64)
    side_key = U64(random64())
    return piece_keys, ep_keys, castle_keys, side_key


PIECE_KEYS, EP_KEYS, CASTLE_KEYS, SIDE_KEY = _init_zobrist()


def _pawn_masks(pb, naf, nhf):
    w = bl = 0
    if pb & nhf:
        w |= pb >> 7
        bl |= pb << 9
    if pb & naf:
        w |= pb >> 9
        bl |= pb << 7
    return w, bl


def _knight_mask(pb, naf, nhf, nabf, nghf):
    a = 0
    if pb & naf:
        a |= (pb >> 17) | (pb << 15)
    if pb & nabf:
        a |= (pb >> 10) | (pb << 6)
    if pb & nhf:
        a |= (pb >> 15) | (pb << 17)
    if pb & nghf:
        a |= (pb >> 6) | (pb << 10)
    return a


def _king_mask(pb, naf, nhf):
    a = (pb >> 8) | (pb << 8)
    if pb & nhf:
        a |= (pb >> 7) | (pb << 1) | (pb << 9)
    if pb & naf:
        a |= (pb >> 9) | (pb >> 1) | (pb << 7)
    return a


def _build_leapers():
    pawn = np.zeros((2, 64), dtype=np.uint64)
    knight = np.zeros(64, dtype=np.uint64)
    king = np.zeros(64, dtype=np.uint64)
    mask = (1 << 64) - 1
    naf, nhf = int(NOT_A_FILE), int(NOT_H_FILE)
    nabf, nghf = int(NOT_AB_FILE), int(NOT_GH_FILE)
    for sq in range(64):
        pb = 1 << sq
        w, bl = _pawn_masks(pb, naf, nhf)
        pawn[WHITE, sq] = w & mask
        pawn[BLACK, sq] = bl & mask
        knight[sq] = _knight_mask(pb, naf, nhf, nabf, nghf) & mask
        king[sq] = _king_mask(pb, naf, nhf) & mask
    return pawn, knight, king


PAWN_ATTACKS, KNIGHT_ATTACKS, KING_ATTACKS = _build_leapers()


@njit(cache=False, error_model='numpy')
def _mask_bishop(sq):
    att = ZERO
    tr, tf = sq // 8, sq % 8
    for i in range(1, 7):
        if tr + i <= 6 and tf + i <= 6:
            att |= ONE << uint64((tr + i) * 8 + tf + i)
        if tr - i >= 1 and tf + i <= 6:
            att |= ONE << uint64((tr - i) * 8 + tf + i)
        if tr + i <= 6 and tf - i >= 1:
            att |= ONE << uint64((tr + i) * 8 + tf - i)
        if tr - i >= 1 and tf - i >= 1:
            att |= ONE << uint64((tr - i) * 8 + tf - i)
    return att


@njit(cache=False, error_model='numpy')
def _mask_rook(sq):
    att = ZERO
    tr, tf = sq // 8, sq % 8
    for rr in range(tr + 1, 7):
        att |= ONE << uint64(rr * 8 + tf)
    for rr in range(tr - 1, 0, -1):
        att |= ONE << uint64(rr * 8 + tf)
    for ff in range(tf - 1, 0, -1):
        att |= ONE << uint64(tr * 8 + ff)
    for ff in range(tf + 1, 7):
        att |= ONE << uint64(tr * 8 + ff)
    return att


@njit(cache=False, error_model='numpy')
def _slider_otf(sq, block, diagonal):
    att = ZERO
    tr, tf = sq // 8, sq % 8
    dirs = ((1, 1), (-1, 1), (1, -1), (-1, -1)) if diagonal else \
           ((1, 0), (-1, 0), (0, 1), (0, -1))
    for dr, df in dirs:
        rr, ff = tr + dr, tf + df
        while 0 <= rr <= 7 and 0 <= ff <= 7:
            bit = ONE << uint64(rr * 8 + ff)
            att |= bit
            if bit & block:
                break
            rr += dr
            ff += df
    return att


@njit(cache=False, error_model='numpy')
def _set_occupancy(index, bits_in_mask, mask):
    occ = ZERO
    for count in range(bits_in_mask):
        sq = INDEX64[int(((mask ^ (mask - ONE)) * DEBRUIJN64) >> uint64(58))]
        mask ^= ONE << uint64(sq)
        if index & (1 << count):
            occ |= ONE << uint64(sq)
    return occ


@njit(cache=False, error_model='numpy')
def _fill_sliders(bmasks, rmasks, batt, ratt):
    for sq in range(64):
        bmasks[sq] = _mask_bishop(sq)
        rmasks[sq] = _mask_rook(sq)
        for index in range(1 << BISHOP_BITS[sq]):
            occ = _set_occupancy(index, BISHOP_BITS[sq], bmasks[sq])
            magic_index = int((occ * BISHOP_MAGICS[sq]) >> BISHOP_SHIFTS[sq])
            batt[sq, magic_index] = _slider_otf(sq, occ, True)
        for index in range(1 << ROOK_BITS[sq]):
            occ = _set_occupancy(index, ROOK_BITS[sq], rmasks[sq])
            magic_index = int((occ * ROOK_MAGICS[sq]) >> ROOK_SHIFTS[sq])
            ratt[sq, magic_index] = _slider_otf(sq, occ, False)


BISHOP_MASKS = np.zeros(64, dtype=np.uint64)
ROOK_MASKS = np.zeros(64, dtype=np.uint64)
BISHOP_ATTACKS = np.zeros((64, 512), dtype=np.uint64)
ROOK_ATTACKS = np.zeros((64, 4096), dtype=np.uint64)

_fill_sliders(BISHOP_MASKS, ROOK_MASKS, BISHOP_ATTACKS, ROOK_ATTACKS)


@njit(cache=False, fastmath=True, error_model='numpy')
def lsb(bbv):
    """Index of the least significant set bit.

    Written as a popcount of the trailing-zero mask rather than a De Bruijn
    multiply: LLVM recognises this form as `cttz` and emits a single `tzcnt`,
    where the table lookup stays six instructions. Called from 78 sites.

    The `& 63` matters - this yields 64 for an input of 0, and the mask keeps
    a malformed position from indexing out of bounds."""
    x = (bbv - ONE) & ~bbv
    cnt = 0
    while x:
        cnt += 1
        x &= x - ONE
    return cnt & 63


@njit(cache=False, fastmath=True, error_model='numpy')
def count_bits(bbv):
    cnt = 0
    while bbv:
        cnt += 1
        bbv &= bbv - ONE
    return cnt


@njit(cache=False, fastmath=True, error_model='numpy')
def bishop_attacks(sq, occ):
    occ &= BISHOP_MASKS[sq]
    occ *= BISHOP_MAGICS[sq]
    return BISHOP_ATTACKS[sq, int(occ >> BISHOP_SHIFTS[sq])]


@njit(cache=False, fastmath=True, error_model='numpy')
def rook_attacks(sq, occ):
    occ &= ROOK_MASKS[sq]
    occ *= ROOK_MAGICS[sq]
    return ROOK_ATTACKS[sq, int(occ >> ROOK_SHIFTS[sq])]


@njit(cache=False, fastmath=True, error_model='numpy')
def queen_attacks(sq, occ):
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ)


@njit(cache=False, fastmath=True, error_model='numpy')
def is_under_attack(bb, sq, attacking_side):
    base = 0 if attacking_side == WHITE else 6
    if PAWN_ATTACKS[attacking_side ^ 1, sq] & bb[base + P]:
        return 1
    if KNIGHT_ATTACKS[sq] & bb[base + N]:
        return 1
    if KING_ATTACKS[sq] & bb[base + K]:
        return 1
    occ = bb[OCC_A]
    if bishop_attacks(sq, occ) & (bb[base + B] | bb[base + Q]):
        return 1
    if rook_attacks(sq, occ) & (bb[base + R] | bb[base + Q]):
        return 1
    return 0


@njit(cache=False, fastmath=True, error_model='numpy')
def get_source(mv):
    return mv & 0x3F


@njit(cache=False, fastmath=True, error_model='numpy')
def get_target(mv):
    return (mv & 0xFC0) >> 6


@njit(cache=False, fastmath=True, error_model='numpy')
def get_piece(mv):
    return (mv & 0xF000) >> 12


@njit(cache=False, fastmath=True, error_model='numpy')
def get_promoted(mv):
    return (mv & 0xF0000) >> 16


@njit(cache=False, fastmath=True, error_model='numpy')
def get_capture(mv):
    return mv & CAP_FLAG


@njit(cache=False, fastmath=True, error_model='numpy')
def get_double(mv):
    return mv & DBL_FLAG


@njit(cache=False, fastmath=True, error_model='numpy')
def get_enpassant(mv):
    return mv & EP_FLAG


@njit(cache=False, fastmath=True, error_model='numpy')
def get_castle_flag(mv):
    return mv & CASTLE_FLAG


SEE_VALUE = np.array([100, 305, 333, 563, 950, 32000] * 2, dtype=np.int64)


@njit(cache=False, fastmath=True, error_model='numpy')
def attackers_to(bb, sq, occ):
    """All pieces of either colour attacking sq, using occ for slider blockers."""
    return ((PAWN_ATTACKS[BLACK, sq] & bb[P])
            | (PAWN_ATTACKS[WHITE, sq] & bb[p])
            | (KNIGHT_ATTACKS[sq] & (bb[N] | bb[n]))
            | (KING_ATTACKS[sq] & (bb[K] | bb[k]))
            | (bishop_attacks(sq, occ) & (bb[B] | bb[b] | bb[Q] | bb[q]))
            | (rook_attacks(sq, occ) & (bb[R] | bb[r] | bb[Q] | bb[q])))


@njit(cache=False, fastmath=True, error_model='numpy')
def _least_valuable_attacker(bb, side_attackers, stm):
    """Cheapest attacker piece index in side_attackers, or -1 for king only."""
    base = 0 if stm == WHITE else 6
    for ptype in range(P, K):
        if side_attackers & bb[base + ptype]:
            return base + ptype
    return -1


@njit(cache=False, fastmath=True, error_model='numpy')
def _see_recompute_xrays(bb, attackers, to, occ, ptype):
    """Add sliders revealed behind the piece just removed from occ."""
    if ptype == P or ptype == B:
        attackers |= bishop_attacks(to, occ) & (bb[B] | bb[b] | bb[Q] | bb[q])
    elif ptype == R:
        attackers |= rook_attacks(to, occ) & (bb[R] | bb[r] | bb[Q] | bb[q])
    elif ptype == Q:
        attackers |= (bishop_attacks(to, occ) & (bb[B] | bb[b] | bb[Q] | bb[q])) \
            | (rook_attacks(to, occ) & (bb[R] | bb[r] | bb[Q] | bb[q]))
    return attackers


@njit(cache=False, fastmath=True, error_model='numpy')
def see_ge(bb, st, mv, threshold):
    """Static exchange evaluation, null-window swap.
    Returns 1 if the exchange on the target square is worth >= threshold.
    En passant, castling and promotions are approximated as SEE = 0, matching
    the C engine (conservative enough for bad-capture filtering)."""
    if (mv & EP_FLAG) or (mv & CASTLE_FLAG) or ((mv & 0xF0000) >> 16):
        return 1 if 0 >= threshold else 0

    src = mv & 0x3F
    to = (mv & 0xFC0) >> 6
    attacker = (mv & 0xF000) >> 12
    side = st[SIDE]

    captured = -1
    start_enemy = p if side == WHITE else P
    to_bit = ONE << uint64(to)
    for bp in range(start_enemy, start_enemy + 6):
        if bb[bp] & to_bit:
            captured = bp
            break

    gain = SEE_VALUE[captured] if captured >= 0 else 0
    swap = gain - threshold
    if swap < 0:
        return 0
    swap = SEE_VALUE[attacker] - swap
    if swap <= 0:
        return 1

    occ = bb[OCC_A] ^ (ONE << uint64(src)) ^ to_bit
    attackers = attackers_to(bb, to, occ)
    stm = side
    res = 1

    while True:
        stm ^= 1
        attackers &= occ
        my_pieces = bb[OCC_W] if stm == WHITE else bb[OCC_B]
        side_attackers = attackers & my_pieces
        if not side_attackers:
            break
        res ^= 1
        piece = _least_valuable_attacker(bb, side_attackers, stm)
        if piece < 0:
            # king capture is illegal while the opponent still attacks the square
            return (res ^ 1) if (attackers & ~my_pieces) else res
        ptype = piece % 6
        swap = SEE_VALUE[ptype] - swap
        if swap < res:
            break
        piece_bb = side_attackers & bb[piece]
        occ ^= piece_bb & (ZERO - piece_bb)
        attackers = _see_recompute_xrays(bb, attackers, to, occ, ptype)

    return res


@njit(cache=False, fastmath=True, error_model='numpy')
def _add_promotions(ml, cnt, src, tgt, pawn, base, cap):
    for pt in (Q, R, N, B):
        ml[cnt] = src | (tgt << 6) | (pawn << 12) | ((base + pt) << 16) | cap
        cnt += 1
    return cnt


@njit(cache=False, fastmath=True, error_model='numpy')
def _gen_pawn_pushes(bb, ml, cnt, side):
    occ_all = bb[OCC_A]
    if side == WHITE:
        pawn, base, fwd, promo_lo, dbl_lo = P, 0, -8, 8, 48
    else:
        pawn, base, fwd, promo_lo, dbl_lo = p, 6, 8, 48, 8
    bbv = bb[pawn]
    while bbv:
        src = lsb(bbv)
        bbv &= bbv - ONE
        tgt = src + fwd
        if tgt < 0 or tgt > 63 or (occ_all & (ONE << uint64(tgt))):
            continue
        if promo_lo <= src <= promo_lo + 7:
            cnt = _add_promotions(ml, cnt, src, tgt, pawn, base, 0)
        else:
            ml[cnt] = src | (tgt << 6) | (pawn << 12)
            cnt += 1
            dbl = tgt + fwd
            if dbl_lo <= src <= dbl_lo + 7 and not (occ_all & (ONE << uint64(dbl))):
                ml[cnt] = src | (dbl << 6) | (pawn << 12) | DBL_FLAG
                cnt += 1
    return cnt


@njit(cache=False, fastmath=True, error_model='numpy')
def _gen_pawn_captures(bb, st, ml, cnt, side):
    ep = st[EP]
    if side == WHITE:
        pawn, base, promo_lo, opp_occ = P, 0, 8, bb[OCC_B]
    else:
        pawn, base, promo_lo, opp_occ = p, 6, 48, bb[OCC_W]
    bbv = bb[pawn]
    while bbv:
        src = lsb(bbv)
        bbv &= bbv - ONE
        atts = PAWN_ATTACKS[side, src] & opp_occ
        while atts:
            tgt = lsb(atts)
            atts &= atts - ONE
            if promo_lo <= src <= promo_lo + 7:
                cnt = _add_promotions(ml, cnt, src, tgt, pawn, base, CAP_FLAG)
            else:
                ml[cnt] = src | (tgt << 6) | (pawn << 12) | CAP_FLAG
                cnt += 1
        if ep != NO_SQ and (PAWN_ATTACKS[side, src] & (ONE << uint64(ep))):
            ml[cnt] = src | (ep << 6) | (pawn << 12) | CAP_FLAG | EP_FLAG
            cnt += 1
    return cnt


@njit(cache=False, fastmath=True, error_model='numpy')
def _gen_castling(bb, st, ml, cnt, side):
    """Castling moves. The attack tests take int64() squares rather than the
    square constants: numba specialises on integer literals, and the literal
    forms compiled six extra copies of is_under_attack and of the magic
    attack helpers it calls."""
    castle = st[CASTLE]
    occ_all = bb[OCC_A]
    if side == WHITE:
        if castle & WK_CASTLE:
            path = (ONE << uint64(f1)) | (ONE << uint64(g1))
            if not (occ_all & path) and not is_under_attack(bb, int64(e1), int64(BLACK)) \
                    and not is_under_attack(bb, int64(f1), int64(BLACK)):
                ml[cnt] = e1 | (g1 << 6) | (K << 12) | CASTLE_FLAG
                cnt += 1
        if castle & WQ_CASTLE:
            path = (ONE << uint64(d1)) | (ONE << uint64(c1)) | (ONE << uint64(b1))
            if not (occ_all & path) and not is_under_attack(bb, int64(e1), int64(BLACK)) \
                    and not is_under_attack(bb, int64(d1), int64(BLACK)):
                ml[cnt] = e1 | (c1 << 6) | (K << 12) | CASTLE_FLAG
                cnt += 1
        return cnt
    if castle & BK_CASTLE:
        path = (ONE << uint64(f8)) | (ONE << uint64(g8))
        if not (occ_all & path) and not is_under_attack(bb, int64(e8), int64(WHITE)) \
                and not is_under_attack(bb, int64(f8), int64(WHITE)):
            ml[cnt] = e8 | (g8 << 6) | (k << 12) | CASTLE_FLAG
            cnt += 1
    if castle & BQ_CASTLE:
        path = (ONE << uint64(d8)) | (ONE << uint64(c8)) | (ONE << uint64(b8))
        if not (occ_all & path) and not is_under_attack(bb, int64(e8), int64(WHITE)) \
                and not is_under_attack(bb, int64(d8), int64(WHITE)):
            ml[cnt] = e8 | (c8 << 6) | (k << 12) | CASTLE_FLAG
            cnt += 1
    return cnt


@njit(cache=False, fastmath=True, error_model='numpy')
def _piece_attacks(ptype, sq, occ):
    if ptype == N:
        return KNIGHT_ATTACKS[sq]
    if ptype == B:
        return bishop_attacks(sq, occ)
    if ptype == R:
        return rook_attacks(sq, occ)
    if ptype == Q:
        return queen_attacks(sq, occ)
    return KING_ATTACKS[sq]


@njit(cache=False, fastmath=True, error_model='numpy')
def _gen_piece_moves(bb, ml, cnt, side, captures_only):
    occ_all = bb[OCC_A]
    my_occ = bb[OCC_W] if side == WHITE else bb[OCC_B]
    opp_occ = bb[OCC_B] if side == WHITE else bb[OCC_W]
    base = 0 if side == WHITE else 6
    allowed = opp_occ if captures_only else ~my_occ
    for ptype in range(N, K + 1):
        piece = base + ptype
        bbv = bb[piece]
        while bbv:
            src = lsb(bbv)
            bbv &= bbv - ONE
            atts = _piece_attacks(ptype, src, occ_all) & allowed
            while atts:
                tgt = lsb(atts)
                atts &= atts - ONE
                cap = CAP_FLAG if (opp_occ & (ONE << uint64(tgt))) else 0
                ml[cnt] = src | (tgt << 6) | (piece << 12) | cap
                cnt += 1
    return cnt


@njit(cache=False, fastmath=True, error_model='numpy')
def generate_moves(bb, st, ml):
    """Pseudo-legal move generation. Fills ml (int32[256]), returns count."""
    side = st[SIDE]
    cnt = _gen_pawn_pushes(bb, ml, 0, side)
    cnt = _gen_pawn_captures(bb, st, ml, cnt, side)
    cnt = _gen_castling(bb, st, ml, cnt, side)
    return _gen_piece_moves(bb, ml, cnt, side, False)


@njit(cache=False, fastmath=True, error_model='numpy')
def generate_captures(bb, st, ml):
    """Captures, capture-promotions and en passant only (quiescence contract:
    quiet promotions and castling are intentionally skipped)."""
    side = st[SIDE]
    cnt = _gen_pawn_captures(bb, st, ml, 0, side)
    return _gen_piece_moves(bb, ml, cnt, side, True)


@njit(cache=False, fastmath=True, error_model='numpy')
def _remove_captured(bb, side, tgt):
    start_piece = p if side == WHITE else P
    tgt_bit = ONE << uint64(tgt)
    for piece in range(start_piece, start_piece + 6):
        if bb[piece] & tgt_bit:
            bb[piece] ^= tgt_bit
            bb[HASH] ^= PIECE_KEYS[piece, tgt]
            return


@njit(cache=False, fastmath=True, error_model='numpy')
def _move_castle_rook(bb, tgt):
    if tgt == g1:
        rook, src, dst = R, h1, f1
    elif tgt == c1:
        rook, src, dst = R, a1, d1
    elif tgt == g8:
        rook, src, dst = r, h8, f8
    else:
        rook, src, dst = r, a8, d8
    bb[rook] ^= ONE << uint64(src)
    bb[rook] |= ONE << uint64(dst)
    bb[HASH] ^= PIECE_KEYS[rook, src]
    bb[HASH] ^= PIECE_KEYS[rook, dst]


@njit(cache=False, fastmath=True, error_model='numpy')
def _rebuild_occupancies(bb):
    occ_w = ZERO
    for i in range(P, K + 1):
        occ_w |= bb[i]
    occ_b = ZERO
    for i in range(p, k + 1):
        occ_b |= bb[i]
    bb[OCC_W] = occ_w
    bb[OCC_B] = occ_b
    bb[OCC_A] = occ_w | occ_b


@njit(cache=False, fastmath=True, error_model='numpy')
def _restore(bb, st, undo_bb, undo_st, ply):
    for i in range(16):
        bb[i] = undo_bb[ply, i]
    for i in range(6):
        st[i] = undo_st[ply, i]


@njit(cache=False, fastmath=True, error_model='numpy')
def make_move(bb, st, undo_bb, undo_st, ply, mv):
    """Copy-make: snapshot to undo[ply], apply, verify own king is safe.
    Returns 1 with the move made, or restores and returns 0 if illegal.
    Deviations from the C engine: no eval
    accumulator rebuild; en passant square recorded only when an enemy pawn
    can pseudo-legally capture it."""
    for i in range(16):
        undo_bb[ply, i] = bb[i]
    for i in range(6):
        undo_st[ply, i] = st[i]

    src = mv & 0x3F
    tgt = (mv & 0xFC0) >> 6
    piece = (mv & 0xF000) >> 12
    promoted = (mv & 0xF0000) >> 16
    side = st[SIDE]

    bb[piece] ^= ONE << uint64(src)
    bb[piece] |= ONE << uint64(tgt)
    bb[HASH] ^= PIECE_KEYS[piece, src]
    bb[HASH] ^= PIECE_KEYS[piece, tgt]

    st[FIFTY] = 0 if (piece == P or piece == p or (mv & CAP_FLAG)) else st[FIFTY] + 1
    st[MOVENUM] += 1

    if st[EP] != NO_SQ:
        bb[HASH] ^= EP_KEYS[st[EP]]
    st[EP] = NO_SQ

    if (mv & CAP_FLAG) and not (mv & EP_FLAG):
        _remove_captured(bb, side, tgt)

    if promoted:
        pawn = P if side == WHITE else p
        bb[pawn] ^= ONE << uint64(tgt)
        bb[HASH] ^= PIECE_KEYS[pawn, tgt]
        bb[promoted] |= ONE << uint64(tgt)
        bb[HASH] ^= PIECE_KEYS[promoted, tgt]

    if mv & EP_FLAG:
        cap_sq = tgt + 8 if side == WHITE else tgt - 8
        opp_pawn = p if side == WHITE else P
        bb[opp_pawn] ^= ONE << uint64(cap_sq)
        bb[HASH] ^= PIECE_KEYS[opp_pawn, cap_sq]

    if mv & DBL_FLAG:
        ep_sq = tgt + 8 if side == WHITE else tgt - 8
        opp_pawn = p if side == WHITE else P
        if PAWN_ATTACKS[side, ep_sq] & bb[opp_pawn]:
            st[EP] = ep_sq
            bb[HASH] ^= EP_KEYS[ep_sq]

    if mv & CASTLE_FLAG:
        _move_castle_rook(bb, tgt)

    bb[HASH] ^= CASTLE_KEYS[st[CASTLE]]
    st[CASTLE] &= CASTLING_RIGHTS[src]
    st[CASTLE] &= CASTLING_RIGHTS[tgt]
    bb[HASH] ^= CASTLE_KEYS[st[CASTLE]]

    _rebuild_occupancies(bb)
    st[SIDE] = side ^ 1
    bb[HASH] ^= SIDE_KEY

    king_sq = lsb(bb[k]) if st[SIDE] == WHITE else lsb(bb[K])
    if is_under_attack(bb, king_sq, st[SIDE]):
        _restore(bb, st, undo_bb, undo_st, ply)
        return 0
    return 1


@njit(cache=False, fastmath=True, error_model='numpy')
def unmake(bb, st, undo_bb, undo_st, ply):
    _restore(bb, st, undo_bb, undo_st, ply)


@njit(cache=False, fastmath=True, error_model='numpy')
def generate_hash_key(bb, st):
    """From-scratch Zobrist key, used by FEN parse and debug assertions."""
    key = ZERO
    for piece in range(12):
        bbv = bb[piece]
        while bbv:
            sq = lsb(bbv)
            bbv &= bbv - ONE
            key ^= PIECE_KEYS[piece, sq]
    if st[EP] != NO_SQ:
        key ^= EP_KEYS[st[EP]]
    key ^= CASTLE_KEYS[st[CASTLE]]
    if st[SIDE] == BLACK:
        key ^= SIDE_KEY
    return key


@njit(cache=False, fastmath=True, error_model='numpy')
def perft(bb, st, undo_bb, undo_st, mls, depth, ply):
    if depth == 0:
        return 1
    cnt = generate_moves(bb, st, mls[ply])
    nodes = 0
    for i in range(cnt):
        if make_move(bb, st, undo_bb, undo_st, ply, mls[ply, i]) == 0:
            continue
        nodes += perft(bb, st, undo_bb, undo_st, mls, depth - 1, ply + 1)
        unmake(bb, st, undo_bb, undo_st, ply)
    return nodes


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

CHAR_TO_PIECE = {c: i for i, c in enumerate("PNBRQKpnbrqk")}
PIECE_TO_CHAR = "PNBRQKpnbrqk"


def new_board():
    bb = np.zeros(16, dtype=np.uint64)
    st = np.zeros(6, dtype=np.int64)
    st[EP] = NO_SQ
    return bb, st


def new_stacks():
    undo_bb = np.zeros((MAX_PLY, 16), dtype=np.uint64)
    undo_st = np.zeros((MAX_PLY, 6), dtype=np.int64)
    mls = np.zeros((MAX_PLY, 256), dtype=np.int32)
    return undo_bb, undo_st, mls


def sq_name(sq):
    return "abcdefgh"[sq % 8] + str(8 - sq // 8)


def sq_index(name):
    return (8 - int(name[1])) * 8 + (ord(name[0]) - ord("a"))


def _parse_pieces(board_field, bb):
    sq = 0
    for ch in board_field:
        if ch == "/":
            continue
        if ch.isdigit():
            sq += int(ch)
        else:
            bb[CHAR_TO_PIECE[ch]] |= ONE << U64(sq)
            sq += 1


def _parse_castling(castling_field):
    flags = {"K": WK_CASTLE, "Q": WQ_CASTLE, "k": BK_CASTLE, "q": BQ_CASTLE}
    castle = 0
    for ch in castling_field:
        castle |= flags.get(ch, 0)
    return castle


def parse_fen(fen, bb, st):
    """FEN into (bb, st), with the same EP normalisation as make_move."""
    bb[:] = 0
    st[:] = 0
    fields = fen.split()
    _parse_pieces(fields[0], bb)
    side = WHITE if fields[1] == "w" else BLACK
    st[SIDE] = side
    st[CASTLE] = _parse_castling(fields[2] if len(fields) > 2 else "-")

    st[EP] = NO_SQ
    ep_field = fields[3] if len(fields) > 3 else "-"
    if ep_field != "-":
        ep_sq = sq_index(ep_field)
        my_pawn = P if side == WHITE else p
        if PAWN_ATTACKS[side ^ 1, ep_sq] & bb[my_pawn]:
            st[EP] = ep_sq

    st[FIFTY] = int(fields[4]) if len(fields) > 4 else 0
    fullmove = int(fields[5]) if len(fields) > 5 else 1
    st[MOVENUM] = 2 * (fullmove - 1) + side

    _rebuild_occupancies(bb)
    bb[HASH] = generate_hash_key(bb, st)


def _piece_on(bb, sq):
    bit = ONE << U64(sq)
    for i in range(12):
        if bb[i] & bit:
            return i
    return -1


def _board_field(bb):
    rows = []
    for rank in range(8):
        row = ""
        empty = 0
        for file in range(8):
            piece = _piece_on(bb, rank * 8 + file)
            if piece < 0:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            row += PIECE_TO_CHAR[piece]
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)


def to_fen(bb, st):
    stm = "w" if st[SIDE] == WHITE else "b"
    castle = ""
    for ch, flag in (("K", WK_CASTLE), ("Q", WQ_CASTLE), ("k", BK_CASTLE), ("q", BQ_CASTLE)):
        if st[CASTLE] & flag:
            castle += ch
    castle = castle or "-"
    ep = "-" if st[EP] == NO_SQ else sq_name(int(st[EP]))
    fullmove = int(st[MOVENUM]) // 2 + 1
    return f"{_board_field(bb)} {stm} {castle} {ep} {int(st[FIFTY])} {fullmove}"


def move_to_uci(mv):
    mv = int(mv)
    s = sq_name(mv & 0x3F) + sq_name((mv & 0xFC0) >> 6)
    promoted = (mv & 0xF0000) >> 16
    if promoted:
        s += "pnbrqk"[promoted % 6]
    return s


def legal_moves(bb, st):
    """Python-side legal move list of packed ints. Not for the hot path."""
    undo_bb, undo_st, mls = new_stacks()
    cnt = generate_moves(bb, st, mls[0])
    out = []
    for i in range(cnt):
        # pass the int32 straight through rather than int(): a Python int
        # types as int64 and compiled a second copy of make_move, which is
        # one of the largest jitted functions in the engine
        if make_move(bb, st, undo_bb, undo_st, 0, mls[0, i]):
            unmake(bb, st, undo_bb, undo_st, 0)
            out.append(int(mls[0, i]))
    return out


def print_board(bb, st):
    for rank in range(8):
        line = f"{8 - rank}  "
        for file in range(8):
            piece = _piece_on(bb, rank * 8 + file)
            line += ("." if piece < 0 else PIECE_TO_CHAR[piece]) + " "
        print(line)
    print("   a b c d e f g h")
    print(f"   side {'wb'[int(st[SIDE])]} castle {int(st[CASTLE]):04b} "
          f"ep {'-' if st[EP] == NO_SQ else sq_name(int(st[EP]))} "
          f"fifty {int(st[FIFTY])} hash {int(bb[HASH]):016x}")


def warmup(compile_perft=False):
    """Compile the jitted functions the agent uses, with production argument
    types. On the platform this runs at import, inside the 90 s init budget,
    so it deliberately does NOT compile perft: that is a test-only function
    and compiling it would spend init budget the agent never gets back.
    Tests pass compile_perft=True."""
    import numba
    assert not numba.config.DISABLE_JIT, "numba JIT is disabled"
    bb, st = new_board()
    undo_bb, undo_st, mls = new_stacks()
    parse_fen(START_FEN, bb, st)
    legal = 0
    count = generate_moves(bb, st, mls[0])
    for i in range(count):
        if make_move(bb, st, undo_bb, undo_st, 0, mls[0, i]):
            unmake(bb, st, undo_bb, undo_st, 0)
            legal += 1
    assert legal == 20, f"warmup legal moves = {legal}, expected 20"
    assert generate_captures(bb, st, mls[0]) == 0
    assert count_bits(bb[OCC_A]) == 32
    assert generate_hash_key(bb, st) == bb[HASH]
    assert see_ge(bb, st, mls[0, 0], 0) in (0, 1)
    if compile_perft:
        total = perft(bb, st, undo_bb, undo_st, mls, 2, 0)
        assert total == 400, f"warmup perft(2) = {total}, expected 400"
