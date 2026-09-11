"""Precomputed evaluation masks.

Generated from their definitions rather than transcribed from the C source,
then property-checked in test_eval.py against those same definitions stated
independently. Square mapping is a8=0..h1=63, so rank index 0 is rank 8 and
"forward" for white is a decreasing square index.
"""

import numpy as np

U64 = np.uint64

WHITE, BLACK = 0, 1


def _bit(sq):
    return 1 << sq


def _sq(rank, file):
    return rank * 8 + file


def _rank_of(sq):
    return sq // 8


def _file_of(sq):
    return sq % 8


def _build_file_rank():
    files = np.zeros(8, dtype=np.uint64)
    ranks = np.zeros(8, dtype=np.uint64)
    for f in range(8):
        mask = 0
        for r in range(8):
            mask |= _bit(_sq(r, f))
        files[f] = mask
    for r in range(8):
        mask = 0
        for f in range(8):
            mask |= _bit(_sq(r, f))
        ranks[r] = mask
    return files, ranks


FILE_MASK, RANK_MASK = _build_file_rank()


def _build_adjacent_files():
    adj = np.zeros(8, dtype=np.uint64)
    for f in range(8):
        mask = 0
        if f > 0:
            mask |= int(FILE_MASK[f - 1])
        if f < 7:
            mask |= int(FILE_MASK[f + 1])
        adj[f] = mask
    return adj


ADJACENT_FILES = _build_adjacent_files()


def _build_passed_masks():
    """Squares that must be free of enemy pawns for a pawn on sq to be passed:
    the pawn's file and both adjacent files, strictly ahead of the pawn."""
    white = np.zeros(64, dtype=np.uint64)
    black = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        r, f = _rank_of(sq), _file_of(sq)
        wm = bm = 0
        for ff in (f - 1, f, f + 1):
            if not 0 <= ff <= 7:
                continue
            for rr in range(0, r):
                wm |= _bit(_sq(rr, ff))
            for rr in range(r + 1, 8):
                bm |= _bit(_sq(rr, ff))
        white[sq] = wm
        black[sq] = bm
    return white, black


WHITE_PASSED, BLACK_PASSED = _build_passed_masks()


def _build_forward_file():
    """The pawn's own file, strictly ahead (used for doubled and opposed)."""
    white = np.zeros(64, dtype=np.uint64)
    black = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        r, f = _rank_of(sq), _file_of(sq)
        wm = bm = 0
        for rr in range(0, r):
            wm |= _bit(_sq(rr, f))
        for rr in range(r + 1, 8):
            bm |= _bit(_sq(rr, f))
        white[sq] = wm
        black[sq] = bm
    return white, black


WHITE_FORWARD_FILE, BLACK_FORWARD_FILE = _build_forward_file()


def _build_isolated():
    iso = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        iso[sq] = ADJACENT_FILES[_file_of(sq)]
    return iso


ISOLATED = _build_isolated()


def _build_phalanx_support():
    """Phalanx: pawns beside each other on the same rank. Support: friendly
    pawns that defend the square (one rank behind, adjacent files)."""
    phalanx = np.zeros(64, dtype=np.uint64)
    w_support = np.zeros(64, dtype=np.uint64)
    b_support = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        r, f = _rank_of(sq), _file_of(sq)
        pm = wm = bm = 0
        for ff in (f - 1, f + 1):
            if not 0 <= ff <= 7:
                continue
            pm |= _bit(_sq(r, ff))
            if r + 1 <= 7:
                wm |= _bit(_sq(r + 1, ff))
            if r - 1 >= 0:
                bm |= _bit(_sq(r - 1, ff))
        phalanx[sq] = pm
        w_support[sq] = wm
        b_support[sq] = bm
    return phalanx, w_support, b_support


PHALANX, WHITE_SUPPORT, BLACK_SUPPORT = _build_phalanx_support()


def _build_between_and_line():
    """between[a][b]: squares strictly between two aligned squares.
    line[a][b]: the full rank/file/diagonal through both. Empty if not
    aligned. Used for pin detection and pinned-piece move restriction."""
    between = np.zeros((64, 64), dtype=np.uint64)
    line = np.zeros((64, 64), dtype=np.uint64)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1))
    for a in range(64):
        ra, fa = _rank_of(a), _file_of(a)
        for dr, df in directions:
            ray = []
            rr, ff = ra + dr, fa + df
            while 0 <= rr <= 7 and 0 <= ff <= 7:
                ray.append(_sq(rr, ff))
                rr += dr
                ff += df
            acc = 0
            for target in ray:
                between[a, target] = acc
                acc |= _bit(target)
            full = 0
            rr, ff = ra - dr, fa - df
            while 0 <= rr <= 7 and 0 <= ff <= 7:
                full |= _bit(_sq(rr, ff))
                rr -= dr
                ff -= df
            for target in ray:
                line[a, target] = full | acc | _bit(a)
    return between, line


BETWEEN, LINE = _build_between_and_line()


def _build_king_zone():
    """King ring: the king's own squares plus the three squares two ranks
    ahead, with the king pulled off the edge files so the ring stays 3 wide."""
    white = np.zeros(64, dtype=np.uint64)
    black = np.zeros(64, dtype=np.uint64)
    for sq in range(64):
        r, f = _rank_of(sq), _file_of(sq)
        cf = min(max(f, 1), 6)
        wm = bm = 0
        for rr in range(r - 1, r + 2):
            for ff in range(cf - 1, cf + 2):
                if 0 <= rr <= 7 and 0 <= ff <= 7:
                    wm |= _bit(_sq(rr, ff))
                    bm |= _bit(_sq(rr, ff))
        for ff in range(cf - 1, cf + 2):
            if r - 2 >= 0:
                wm |= _bit(_sq(r - 2, ff))
            if r + 2 <= 7:
                bm |= _bit(_sq(r + 2, ff))
        white[sq] = wm
        black[sq] = bm
    return white, black


WHITE_KING_ZONE, BLACK_KING_ZONE = _build_king_zone()


def _build_king_flank():
    """File groups defining the king's flank, BTC's kingFlankMask.

    Files e, f and g were wrong: e mapped to d-g instead of c-f, and f and g
    to f-h instead of e-h. Those are the two commonest king files, castled
    and uncastled, and the mask drives both the pawnless-flank penalty and the
    flank attack and defence counts in king danger."""
    groups = {
        0: (0, 2), 1: (0, 3), 2: (0, 3), 3: (2, 5),
        4: (2, 5), 5: (4, 7), 6: (4, 7), 7: (5, 7),
    }
    flank = np.zeros(8, dtype=np.uint64)
    for f, (lo, hi) in groups.items():
        mask = 0
        for ff in range(lo, hi + 1):
            mask |= int(FILE_MASK[ff])
        flank[f] = mask
    return flank


KING_FLANK = _build_king_flank()


def _build_camp():
    """Own half plus the middle rank, per side (space and king safety)."""
    camp = np.zeros(2, dtype=np.uint64)
    white_mask = black_mask = 0
    for r in range(3, 8):
        white_mask |= int(RANK_MASK[r])
    for r in range(0, 5):
        black_mask |= int(RANK_MASK[r])
    camp[WHITE] = white_mask
    camp[BLACK] = black_mask
    return camp


CAMP = _build_camp()


def _build_forward_ranks():
    """All squares strictly ahead of a square's rank, per side."""
    forward = np.zeros((2, 64), dtype=np.uint64)
    for sq in range(64):
        r = _rank_of(sq)
        wm = bm = 0
        for rr in range(0, r):
            wm |= int(RANK_MASK[rr])
        for rr in range(r + 1, 8):
            bm |= int(RANK_MASK[rr])
        forward[WHITE, sq] = wm
        forward[BLACK, sq] = bm
    return forward


FORWARD_RANKS = _build_forward_ranks()

CENTER_FILES = U64(int(FILE_MASK[2]) | int(FILE_MASK[3])
                   | int(FILE_MASK[4]) | int(FILE_MASK[5]))

# outposts sit on the 4th to 6th rank from the owner's side
OUTPOST_RANKS_WHITE = U64(int(RANK_MASK[2]) | int(RANK_MASK[3]) | int(RANK_MASK[4]))
OUTPOST_RANKS_BLACK = U64(int(RANK_MASK[3]) | int(RANK_MASK[4]) | int(RANK_MASK[5]))

RANK_OF = np.array([_rank_of(sq) for sq in range(64)], dtype=np.int64)
FILE_OF = np.array([_file_of(sq) for sq in range(64)], dtype=np.int64)

# relative rank: 0 on the owner's back rank, 7 on the promotion rank
RELATIVE_RANK = np.zeros((2, 64), dtype=np.int64)
for _s in range(64):
    RELATIVE_RANK[WHITE, _s] = 7 - _rank_of(_s)
    RELATIVE_RANK[BLACK, _s] = _rank_of(_s)
