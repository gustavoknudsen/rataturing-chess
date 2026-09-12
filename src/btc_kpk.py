"""KPK bitbase: exact win/draw for king and pawn against king.

The C engine generates the same table with initKPK. Generation is retrograde
analysis over every legal KPK position, iterated until nothing changes, which
is fast enough to run at import once jitted (the alternative, shipping the
table as data, would be allowed by the rules but is far less readable).

Positions are normalised so the strong side is white with its pawn on files
a-d, giving 64 x 24 x 64 x 2 = 196,608 entries. Square mapping is the engine's
a8=0..h1=63, so white's pawn advances toward lower indices.
"""

import os

import numpy as np
from numba import njit, uint64

# Jit-only helpers. Suppressing the Python-callable wrappers cuts compile
# time, but calling one of these from Python then crashes the process, so a
# function gets this only once every call site is known to be jitted.
_NOWRAP = {"no_cpython_wrapper": True, "no_cfunc_wrapper": True}


from btc_core import KING_ATTACKS, PAWN_ATTACKS, WHITE

DRAW = 0
WIN = 1
UNKNOWN = 2

# index = ((white_king * 24 + pawn_index) * 64 + black_king) * 2 + side_to_move
KPK_SIZE = 64 * 24 * 64 * 2


@njit(cache=False, **_NOWRAP)
def _pawn_index(sq):
    """Pawn squares are files a-d, ranks 2-7: 4 files x 6 ranks = 24."""
    rank = sq // 8
    file = sq % 8
    return (rank - 1) * 4 + file


@njit(cache=False, **_NOWRAP)
def _kpk_index(wk, pawn, bk, stm):
    return ((wk * 24 + _pawn_index(pawn)) * 64 + bk) * 2 + stm


@njit(cache=False, **_NOWRAP)
def _legal_layout(wk, pawn, bk):
    """Kings apart, no piece overlapping another."""
    if wk == bk or wk == pawn or bk == pawn:
        return False
    if KING_ATTACKS[wk] & (uint64(1) << uint64(bk)):
        return False
    return True


@njit(cache=False, **_NOWRAP)
def _classify_immediate(wk, pawn, bk, stm):
    """Terminal results that need no lookahead. Only the promotion win is
    seeded here; stalemate and pawn capture fall out of the move loops in
    _resolve, which count legal moves rather than inferring them."""
    if stm == 0 and pawn // 8 == 1:
        promo = pawn - 8
        if promo != wk and promo != bk:
            attacked = (KING_ATTACKS[bk] & (uint64(1) << uint64(promo))) != uint64(0)
            defended = (KING_ATTACKS[wk] & (uint64(1) << uint64(promo))) != uint64(0)
            if not attacked or defended:
                return WIN
    return UNKNOWN


@njit(cache=False)
def _step(table):
    """One retrograde pass. Returns the number of entries newly resolved."""
    changed = 0
    for wk in range(64):
        for pawn_sq in range(8, 56):
            if pawn_sq % 8 > 3:
                continue
            for bk in range(64):
                if not _legal_layout(wk, pawn_sq, bk):
                    continue
                for stm in range(2):
                    idx = _kpk_index(wk, pawn_sq, bk, stm)
                    if table[idx] != UNKNOWN:
                        continue
                    result = _resolve(table, wk, pawn_sq, bk, stm)
                    if result != UNKNOWN:
                        table[idx] = result
                        changed += 1
    return changed


@njit(cache=False, **_NOWRAP)
def _resolve(table, wk, pawn, bk, stm):
    """White to move wins if any move wins; black to move draws if any move
    draws. Unresolved children leave the position unknown for this pass."""
    if stm == 0:
        saw_unknown = False
        legal = 0
        for target in range(64):
            if not (KING_ATTACKS[wk] & (uint64(1) << uint64(target))):
                continue
            if target == pawn or target == bk:
                continue
            if KING_ATTACKS[bk] & (uint64(1) << uint64(target)):
                continue
            legal += 1
            child = table[_kpk_index(target, pawn, bk, 1)]
            if child == WIN:
                return WIN
            if child == UNKNOWN:
                saw_unknown = True
        push = pawn - 8
        if push >= 8 and push != wk and push != bk:
            legal += 1
            child = table[_kpk_index(wk, push, bk, 1)]
            if child == WIN:
                return WIN
            if child == UNKNOWN:
                saw_unknown = True
            if pawn // 8 == 6:
                double = pawn - 16
                if double >= 8 and double != wk and double != bk:
                    legal += 1
                    child = table[_kpk_index(wk, double, bk, 1)]
                    if child == WIN:
                        return WIN
                    if child == UNKNOWN:
                        saw_unknown = True
        if legal == 0:
            return DRAW      # white stalemated
        return UNKNOWN if saw_unknown else DRAW

    saw_unknown = False
    legal = 0
    for target in range(64):
        if not (KING_ATTACKS[bk] & (uint64(1) << uint64(target))):
            continue
        if target == wk:
            continue
        if KING_ATTACKS[wk] & (uint64(1) << uint64(target)):
            continue
        if target == pawn:
            # the black king may take the pawn only if it is undefended
            if KING_ATTACKS[wk] & (uint64(1) << uint64(pawn)):
                continue
            return DRAW     # king and king
        if PAWN_ATTACKS[WHITE, pawn] & (uint64(1) << uint64(target)):
            continue
        legal += 1
        child = table[_kpk_index(wk, pawn, target, 0)]
        if child == DRAW:
            return DRAW
        if child == UNKNOWN:
            saw_unknown = True
    if legal == 0:
        return DRAW          # black stalemated: no legal king move
    return UNKNOWN if saw_unknown else WIN


@njit(cache=False)
def _seed(table):
    for wk in range(64):
        for pawn_sq in range(8, 56):
            if pawn_sq % 8 > 3:
                continue
            for bk in range(64):
                if not _legal_layout(wk, pawn_sq, bk):
                    continue
                for stm in range(2):
                    result = _classify_immediate(wk, pawn_sq, bk, stm)
                    if result != UNKNOWN:
                        table[_kpk_index(wk, pawn_sq, bk, stm)] = result


def build():
    """Retrograde analysis to a fixed point. Only used to regenerate the
    embedded table: jitting the solver costs ~14 s locally, roughly 26 s of
    the platform's 90 s init budget, which is far too much for one endgame."""
    table = np.full(KPK_SIZE, UNKNOWN, dtype=np.uint8)
    _seed(table)
    for _ in range(64):
        if _step(table) == 0:
            break
    # anything still unresolved is a draw: no forced win was ever proved
    table[table == UNKNOWN] = DRAW
    return table


def _load():
    """Unpack the embedded bitbase. Costs about a millisecond, against ~26 s
    of platform budget to solve it at startup."""
    import base64
    import zlib

    from btc_kpk_data import ENTRIES, KPK_B64, WINS

    raw = zlib.decompress(base64.b64decode(KPK_B64))
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    table = bits[:ENTRIES].astype(np.uint8)
    assert table.size == KPK_SIZE, "embedded KPK table is the wrong size"
    assert int(table.sum()) == WINS, "embedded KPK table failed its checksum"
    return table


if os.environ.get("BTC_KPK_REGEN") == "1":
    KPK_TABLE = build()
else:
    KPK_TABLE = _load()


@njit(cache=False, fastmath=True, **_NOWRAP)
def kpk_win(wk, pawn, bk, stm):
    """Raw lookup. Inputs must already be normalised: white to promote, pawn
    on files a-d. Callers should use probe(), which does the normalisation."""
    return KPK_TABLE[_kpk_index(wk, pawn, bk, stm)] == WIN


@njit(cache=False, fastmath=True)
def probe(strong_king, pawn, weak_king, strong_is_white, strong_to_move):
    """Is KP vs K won for the pawn's side? Handles both normalisations the
    table omits: mirroring files e-h onto a-d, and flipping ranks when the
    strong side is black. Squares are the engine's a8=0..h1=63."""
    wk = strong_king
    ps = pawn
    bk = weak_king
    if not strong_is_white:
        wk ^= 56
        ps ^= 56
        bk ^= 56
    if ps % 8 > 3:
        wk ^= 7
        ps ^= 7
        bk ^= 7
    return KPK_TABLE[_kpk_index(wk, ps, bk, 0 if strong_to_move else 1)] == WIN
