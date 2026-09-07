"""Search ported from BTC search.cpp, phase 2 subset.

Implements iterative deepening with aspiration windows, PVS, a bucketed
transposition table, killer and history move ordering, null-move pruning,
reverse futility pruning, mate distance pruning, the improving heuristic,
repetition and fifty-move draw detection, in-check extension and a
captures-only quiescence search. LMR, SEE, continuation history, singular
extensions and ProbCut land in later phases per docs/PLAN.md.

Feature flags (environment, read at import so numba freezes them into the
compiled code): BTC_MINIMAL=1 disables TT, null move, RFP and mate distance
pruning; used by the reference differential in run_tests.py.

C semantics note: divisions in ported formulas use c_div (truncation toward
zero) because C and Python round negative quotients differently.
"""

import os
import time

import numpy as np
from numba import int64, njit, objmode, uint64

from btc_core import (
    EP, EP_KEYS, FIFTY, HASH, K, MAX_PLY, N, NO_SQ, ONE, P, Q, SIDE,
    SIDE_KEY, WHITE, ZERO, count_bits, generate_captures, generate_moves,
    is_under_attack, k, legal_moves, lsb, make_move, p, see_ge, unmake,
)
from btc_eval import evaluate as full_evaluate
from btc_psqt import MIRROR, PIECE_TABLES


def _flag(name):
    """Feature toggle, frozen into the compiled code at import."""
    return os.environ.get(name, "1") == "1"


MINIMAL = os.environ.get("BTC_MINIMAL") == "1"
USE_TT = not MINIMAL and _flag("BTC_TT")
USE_NULL = not MINIMAL and _flag("BTC_NULL")
USE_RFP = not MINIMAL and _flag("BTC_RFP")
USE_MDP = not MINIMAL and _flag("BTC_MDP")
USE_SEE = not MINIMAL and _flag("BTC_SEE")
USE_LMR = not MINIMAL and _flag("BTC_LMR")
USE_LMP = not MINIMAL and _flag("BTC_LMP")
USE_FUTILITY = not MINIMAL and _flag("BTC_FUTILITY")
USE_RAZOR = not MINIMAL and _flag("BTC_RAZOR")
USE_IIR = not MINIMAL and _flag("BTC_IIR")
USE_CONT_HIST = not MINIMAL and _flag("BTC_CONTHIST")
USE_FULL_EVAL = _flag("BTC_EVAL")

INFINITY = 50000
MATE_VALUE = 49000
MATE_SCORE = 48000
MAX_SEARCH_PLY = 64
NO_HASH_ENTRY = 100000
EVAL_NONE = 100000

HASH_EXACT, HASH_ALPHA, HASH_BETA = 0, 1, 2

TT_ENTRIES = 1 << 24
TT_BUCKETS = TT_ENTRIES // 4
SCORE_BIAS = 1 << 17

SCORE_TT_BEST = 10_000_000
SCORE_GOOD_CAPTURE = 1_000_000
SCORE_KILLER_1 = 800_000
SCORE_KILLER_2 = 700_000
SCORE_COUNTER = 600_000
SCORE_BAD_CAPTURE = -1_000_000

HIST_MAX = 8192

# LMR: Ethereal-style base reduction table, indexed [depth][move index].
# Precomputed at import because log() per node is wasted work.
LMR_MAX_DEPTH = 64
LMR_MAX_MOVES = 64
_LMR = np.zeros((LMR_MAX_DEPTH, LMR_MAX_MOVES), dtype=np.int64)
for _d in range(1, LMR_MAX_DEPTH):
    for _m in range(1, LMR_MAX_MOVES):
        _LMR[_d, _m] = int(0.7844 + np.log(_d) * np.log(_m) / 2.4696)
LMR_TABLE = _LMR

SC_NODES, SC_STOP, SC_TT_GEN = 0, 1, 2

TEMPO = 28
OPENING_PHASE = 15196
ENDGAME_PHASE = 1841

MATERIAL_MG = np.array([126, 781, 825, 1276, 2538, 10000], dtype=np.int64)
MATERIAL_EG = np.array([208, 854, 915, 1380, 2682, 10000], dtype=np.int64)

MVV_LVA = np.array([
    [105, 205, 305, 405, 505, 605] * 2,
    [104, 204, 304, 404, 504, 604] * 2,
    [103, 203, 303, 403, 503, 603] * 2,
    [102, 202, 302, 402, 502, 602] * 2,
    [101, 201, 301, 401, 501, 601] * 2,
    [100, 200, 300, 400, 500, 600] * 2,
] * 2, dtype=np.int64)


@njit(cache=False, fastmath=True)
def c_div(a, b):
    """C integer division: truncation toward zero. Requires b > 0."""
    q = a // b
    if q < 0 and q * b != a:
        q += 1
    return q


@njit(cache=False, fastmath=True)
def interpolate(opening_value, endgame_value, stage_score):
    return c_div(opening_value * stage_score
                 + endgame_value * (OPENING_PHASE - stage_score), OPENING_PHASE)


@njit(cache=False, fastmath=True)
def game_stage_score(bb):
    score = 0
    for pc in range(N, Q + 1):
        score += (count_bits(bb[pc]) + count_bits(bb[pc + 6])) * MATERIAL_MG[pc]
    return score


@njit(cache=False, fastmath=True)
def _material_term(pc, stage, stage_score):
    if stage == 2:
        return interpolate(MATERIAL_MG[pc], MATERIAL_EG[pc], stage_score)
    return MATERIAL_MG[pc] if stage == 0 else MATERIAL_EG[pc]


@njit(cache=False, fastmath=True)
def _pst_term(pc, sq, stage, stage_score):
    if stage == 2:
        return interpolate(PIECE_TABLES[0, pc, sq], PIECE_TABLES[1, pc, sq],
                           stage_score)
    return PIECE_TABLES[stage, pc, sq]


@njit(cache=False, fastmath=True)
def _evaluate_psqt(bb, st):
    """Material + piece-square tables + tempo. Kept as the BTC_EVAL=0 fallback
    so the full evaluation can be A/B tested against it."""
    stage_score = game_stage_score(bb)
    if stage_score > OPENING_PHASE:
        stage = 0
    elif stage_score < ENDGAME_PHASE:
        stage = 1
    else:
        stage = 2

    score = TEMPO if st[SIDE] == WHITE else -TEMPO
    for pc in range(P, K + 1):
        mat = _material_term(pc, stage, stage_score)
        bbv = bb[pc]
        while bbv:
            sq = lsb(bbv)
            bbv &= bbv - ONE
            score += mat + _pst_term(pc, sq, stage, stage_score)
        bbv = bb[pc + 6]
        while bbv:
            sq = lsb(bbv)
            bbv &= bbv - ONE
            score -= mat + _pst_term(pc, MIRROR[sq], stage, stage_score)
    return score if st[SIDE] == WHITE else -score


@njit(cache=False, fastmath=True)
def evaluate(bb, st):
    if USE_FULL_EVAL:
        return full_evaluate(bb, st)
    return _evaluate_psqt(bb, st)


@njit(cache=False, fastmath=True)
def _tt_pack(move, score, depth, flag, age):
    v = uint64(move & 0xFFFFFF)
    v |= uint64(score + SCORE_BIAS) << uint64(24)
    v |= uint64(depth) << uint64(42)
    v |= uint64(flag) << uint64(49)
    v |= uint64(age & 0xFF) << uint64(51)
    return v


# int64() casts are load-bearing: numba's int(uint64) stays uint64, and a
# later int64/uint64 branch unification silently promotes to float64

@njit(cache=False, fastmath=True)
def _tt_move(data):
    return int64(data & uint64(0xFFFFFF))


@njit(cache=False, fastmath=True)
def _tt_score(data):
    return int64((data >> uint64(24)) & uint64(0x3FFFF)) - SCORE_BIAS


@njit(cache=False, fastmath=True)
def _tt_depth(data):
    return int64((data >> uint64(42)) & uint64(0x7F))


@njit(cache=False, fastmath=True)
def _tt_flag(data):
    return int64((data >> uint64(49)) & uint64(0x3))


@njit(cache=False, fastmath=True)
def _tt_age(data):
    return int64((data >> uint64(51)) & uint64(0xFF))


@njit(cache=False, fastmath=True)
def tt_probe(bb, alpha, beta, depth, ply, tt_key, tt_data):
    """Returns (cutoff score or NO_HASH_ENTRY, tt move or 0)."""
    if not USE_TT:
        return NO_HASH_ENTRY, 0
    key = bb[HASH]
    base = int64(key & uint64(tt_key.shape[0] // 4 - 1)) * 4
    for i in range(4):
        if tt_key[base + i] != key:
            continue
        data = tt_data[base + i]
        move = _tt_move(data)
        score = _tt_score(data)
        if score < -MATE_SCORE:
            score += ply
        if score > MATE_SCORE:
            score -= ply
        if _tt_depth(data) >= depth:
            flag = _tt_flag(data)
            if flag == HASH_EXACT:
                return score, move
            if flag == HASH_ALPHA and score <= alpha:
                return alpha, move
            if flag == HASH_BETA and score >= beta:
                return beta, move
        return NO_HASH_ENTRY, move
    return NO_HASH_ENTRY, 0


@njit(cache=False, fastmath=True)
def tt_record(bb, score, depth, flag, move, ply, tt_key, tt_data, gen):
    if not USE_TT:
        return
    key = bb[HASH]
    base = int64(key & uint64(tt_key.shape[0] // 4 - 1)) * 4
    slot = -1
    replace = base
    replace_value = 1 << 30
    for i in range(4):
        idx = base + i
        if tt_key[idx] == key or tt_key[idx] == ZERO:
            slot = idx
            break
        value = _tt_depth(tt_data[idx]) - 8 * ((gen - _tt_age(tt_data[idx])) & 0xFF)
        if value < replace_value:
            replace_value = value
            replace = idx
    if slot < 0:
        slot = replace
    # depth-preferred: keep a deeper same-key entry unless exact or close;
    # still refresh its move and age (search.cpp recordHash)
    if tt_key[slot] == key and flag != HASH_EXACT and depth < _tt_depth(tt_data[slot]) - 3:
        old = tt_data[slot]
        kept_move = move if move else _tt_move(old)
        tt_data[slot] = _tt_pack(kept_move, _tt_score(old), _tt_depth(old),
                                 _tt_flag(old), gen)
        return
    if score < -MATE_SCORE:
        score -= ply
    if score > MATE_SCORE:
        score += ply
    tt_key[slot] = key
    tt_data[slot] = _tt_pack(move, score, depth, flag, gen)


@njit(cache=False, fastmath=True)
def _update_history(hist, piece, tgt, bonus):
    if bonus > HIST_MAX:
        bonus = HIST_MAX
    if bonus < -HIST_MAX:
        bonus = -HIST_MAX
    e = int(hist[piece, tgt])
    e += bonus - c_div(e * abs(bonus), HIST_MAX)
    hist[piece, tgt] = e


@njit(cache=False, fastmath=True)
def _history_bonus(depth):
    bonus = 16 * depth * depth + 32 * depth - 16
    if bonus > 1200:
        bonus = 1200
    if bonus < 0:
        bonus = 0
    return bonus


@njit(cache=False, fastmath=True)
def _captured_piece(bb, st, mv):
    if mv & (1 << 22):
        return p if st[SIDE] == WHITE else P
    tgt = (mv & 0xFC0) >> 6
    start = p if st[SIDE] == WHITE else P
    tgt_bit = ONE << uint64(tgt)
    for piece in range(start, start + 6):
        if bb[piece] & tgt_bit:
            return piece
    return start


@njit(cache=False, fastmath=True)
def _capture_score(bb, st, mv, cap_hist):
    """Good captures (SEE >= 0) sort above quiets, bad captures below them."""
    captured = _captured_piece(bb, st, mv)
    attacker = (mv & 0xF000) >> 12
    tgt = (mv & 0xFC0) >> 6
    base = MVV_LVA[attacker, captured] * 100 + int64(cap_hist[attacker, tgt, captured])
    if not USE_SEE:
        return SCORE_GOOD_CAPTURE + base
    if see_ge(bb, st, mv, 0):
        return SCORE_GOOD_CAPTURE + base
    return SCORE_BAD_CAPTURE + base


@njit(cache=False, fastmath=True)
def _cont_hist_score(cont_hist, played, ply, mv):
    """1-ply and 2-ply continuation history for a quiet move."""
    if not USE_CONT_HIST:
        return 0
    piece = (mv & 0xF000) >> 12
    tgt = (mv & 0xFC0) >> 6
    total = 0
    if ply >= 1 and played[ply] != 0:
        prev = played[ply]
        total += int64(cont_hist[0, (prev & 0xF000) >> 12, (prev & 0xFC0) >> 6,
                                 piece, tgt])
    if ply >= 2 and played[ply - 1] != 0:
        prev2 = played[ply - 1]
        total += int64(cont_hist[1, (prev2 & 0xF000) >> 12, (prev2 & 0xFC0) >> 6,
                                 piece, tgt])
    return total


@njit(cache=False, fastmath=True)
def _counter_move(counters, played, ply):
    if ply < 1 or played[ply] == 0:
        return 0
    prev = played[ply]
    return int64(counters[(prev & 0xF000) >> 12, (prev & 0xFC0) >> 6])


@njit(cache=False, fastmath=True)
def _score_move(bb, st, mv, ply, tt_move, killers, main_hist, cap_hist,
                cont_hist, counters, played):
    if tt_move != 0 and mv == tt_move:
        return SCORE_TT_BEST
    if mv & (1 << 20):
        return _capture_score(bb, st, mv, cap_hist)
    if killers[0, ply] == mv:
        return SCORE_KILLER_1
    if killers[1, ply] == mv:
        return SCORE_KILLER_2
    if _counter_move(counters, played, ply) == mv:
        return SCORE_COUNTER
    return int64(main_hist[(mv & 0xF000) >> 12, (mv & 0xFC0) >> 6]) \
        + _cont_hist_score(cont_hist, played, ply, mv)


@njit(cache=False, fastmath=True)
def _insertion_sort(ml, scores, cnt):
    for i in range(1, cnt):
        cur_score = scores[i]
        cur_move = ml[i]
        j = i - 1
        while j >= 0 and scores[j] < cur_score:
            scores[j + 1] = scores[j]
            ml[j + 1] = ml[j]
            j -= 1
        scores[j + 1] = cur_score
        ml[j + 1] = cur_move


@njit(cache=False, fastmath=True)
def _sort_moves(bb, st, ml, scores, cnt, ply, tt_move, killers, main_hist,
                cap_hist, cont_hist, counters, played):
    for i in range(cnt):
        scores[i] = _score_move(bb, st, ml[i], ply, tt_move, killers,
                                main_hist, cap_hist, cont_hist, counters,
                                played)
    _insertion_sort(ml, scores, cnt)


@njit(cache=False, fastmath=True)
def _sort_captures(bb, st, ml, scores, cnt, cap_hist):
    for i in range(cnt):
        scores[i] = _capture_score(bb, st, ml[i], cap_hist)
    _insertion_sort(ml, scores, cnt)


@njit(cache=False, fastmath=True)
def _check_time(sc, fc):
    if (sc[SC_NODES] & 2047) == 0:
        with objmode(now="f8"):
            now = time.perf_counter()
        if now >= fc[0]:
            sc[SC_STOP] = 1


@njit(cache=False, fastmath=True)
def _in_check(bb, st):
    side = st[SIDE]
    king_sq = lsb(bb[K]) if side == WHITE else lsb(bb[k])
    return is_under_attack(bb, king_sq, side ^ 1)


@njit(cache=False, fastmath=True)
def _is_repetition(bb, rep, rep_idx):
    key = bb[HASH]
    for i in range(rep_idx):
        if rep[i] == key:
            return 1
    return 0


@njit(cache=False, fastmath=True)
def qsearch(alpha, beta, bb, st, undo_bb, undo_st, mls, scores, cap_hist,
            sc, fc, ply):
    _check_time(sc, fc)
    sc[SC_NODES] += 1
    if ply > MAX_SEARCH_PLY - 1:
        return evaluate(bb, st)

    ev = evaluate(bb, st)
    if ev >= beta:
        return beta
    if ev > alpha:
        alpha = ev

    cnt = generate_captures(bb, st, mls[ply])
    _sort_captures(bb, st, mls[ply], scores[ply], cnt, cap_hist)

    for i in range(cnt):
        mv = mls[ply, i]
        # SEE pruning: losing captures cannot raise the stand-pat
        if USE_SEE and not see_ge(bb, st, mv, 0):
            continue
        if make_move(bb, st, undo_bb, undo_st, ply, mv) == 0:
            continue
        score = -qsearch(-beta, -alpha, bb, st, undo_bb, undo_st, mls,
                         scores, cap_hist, sc, fc, ply + 1)
        unmake(bb, st, undo_bb, undo_st, ply)
        if sc[SC_STOP]:
            return 0
        if score > alpha:
            alpha = score
            if score >= beta:
                return beta
    return alpha


@njit(cache=False, fastmath=True)
def _make_null(bb, st, rep, rep_idx):
    """Apply a null move (side flip, EP cleared). Non-recursive so the
    recursive call stays inside negamax: numba 0.67 rejects mutual recursion."""
    saved_ep = st[EP]
    saved_hash = bb[HASH]
    if saved_ep != NO_SQ:
        bb[HASH] ^= EP_KEYS[saved_ep]
    st[EP] = NO_SQ
    st[SIDE] ^= 1
    bb[HASH] ^= SIDE_KEY
    rep[rep_idx] = saved_hash
    return saved_ep, saved_hash


@njit(cache=False, fastmath=True)
def _unmake_null(bb, st, saved_ep, saved_hash):
    st[SIDE] ^= 1
    st[EP] = saved_ep
    bb[HASH] = saved_hash


@njit(cache=False, fastmath=True)
def _update_cont_hist(cont_hist, played, ply, mv, bonus):
    """1-ply full bonus, 2-ply at 3/4 (Stockfish 780/1040 weighting)."""
    if not USE_CONT_HIST:
        return
    piece = (mv & 0xF000) >> 12
    tgt = (mv & 0xFC0) >> 6
    if ply >= 1 and played[ply] != 0:
        prev = played[ply]
        _update_cont_entry(cont_hist, 0, (prev & 0xF000) >> 12,
                           (prev & 0xFC0) >> 6, piece, tgt, bonus)
    if ply >= 2 and played[ply - 1] != 0:
        prev2 = played[ply - 1]
        _update_cont_entry(cont_hist, 1, (prev2 & 0xF000) >> 12,
                           (prev2 & 0xFC0) >> 6, piece, tgt, c_div(bonus * 3, 4))


@njit(cache=False, fastmath=True)
def _update_cont_entry(cont_hist, table, prev_piece, prev_tgt, piece, tgt, bonus):
    if bonus > HIST_MAX:
        bonus = HIST_MAX
    if bonus < -HIST_MAX:
        bonus = -HIST_MAX
    e = int64(cont_hist[table, prev_piece, prev_tgt, piece, tgt])
    e += bonus - c_div(e * abs(bonus), HIST_MAX)
    cont_hist[table, prev_piece, prev_tgt, piece, tgt] = e


@njit(cache=False, fastmath=True)
def _update_capture_history(bb, st, mv, cap_hist, captures, capture_cnt, bonus):
    captured = _captured_piece(bb, st, mv)
    _update_cap_entry(cap_hist, (mv & 0xF000) >> 12, (mv & 0xFC0) >> 6,
                      captured, bonus)
    for i in range(capture_cnt - 1):
        bad = captures[i]
        bad_cap = _captured_piece(bb, st, bad)
        _update_cap_entry(cap_hist, (bad & 0xF000) >> 12, (bad & 0xFC0) >> 6,
                          bad_cap, -bonus)


@njit(cache=False, fastmath=True)
def _update_cap_entry(cap_hist, piece, tgt, captured, bonus):
    if bonus > HIST_MAX:
        bonus = HIST_MAX
    if bonus < -HIST_MAX:
        bonus = -HIST_MAX
    e = int64(cap_hist[piece, tgt, captured])
    e += bonus - c_div(e * abs(bonus), HIST_MAX)
    cap_hist[piece, tgt, captured] = e


@njit(cache=False, fastmath=True)
def _quiet_cutoff_update(mv, depth, ply, killers, main_hist, cont_hist,
                         counters, played, quiets, quiet_cnt):
    if killers[0, ply] != mv:
        killers[1, ply] = killers[0, ply]
        killers[0, ply] = mv
    if ply >= 1 and played[ply] != 0:
        prev = played[ply]
        counters[(prev & 0xF000) >> 12, (prev & 0xFC0) >> 6] = mv
    bonus = _history_bonus(depth)
    _update_history(main_hist, (mv & 0xF000) >> 12, (mv & 0xFC0) >> 6, bonus)
    _update_cont_hist(cont_hist, played, ply, mv, bonus)
    for i in range(quiet_cnt - 1):
        bad = quiets[i]
        _update_history(main_hist, (bad & 0xF000) >> 12, (bad & 0xFC0) >> 6,
                        -bonus)
        _update_cont_hist(cont_hist, played, ply, bad, -bonus)


@njit(cache=False, fastmath=True)
def _beta_cutoff_update(bb, st, mv, depth, ply, killers, main_hist, cap_hist,
                        cont_hist, counters, played, quiets, quiet_cnt,
                        captures, capture_cnt):
    bonus = _history_bonus(depth)
    if mv & (1 << 20):
        _update_capture_history(bb, st, mv, cap_hist, captures, capture_cnt,
                                bonus)
        return
    _quiet_cutoff_update(mv, depth, ply, killers, main_hist, cont_hist,
                         counters, played, quiets, quiet_cnt)


@njit(cache=False, fastmath=True)
def _improving(static_evals, ev, ply, in_check):
    static_evals[ply] = EVAL_NONE if in_check else ev
    if in_check:
        return 0
    if ply >= 2 and static_evals[ply - 2] != EVAL_NONE:
        return 1 if ev > static_evals[ply - 2] else 0
    if ply >= 4 and static_evals[ply - 4] != EVAL_NONE:
        return 1 if ev > static_evals[ply - 4] else 0
    return 0


# _node_prologue outcome codes
NODE_CONTINUE, NODE_RETURN, NODE_QSEARCH = 0, 1, 2


@njit(cache=False, fastmath=True)
def _razor(alpha, beta, depth, ev, bb, st, undo_bb, undo_st, mls, scores,
           cap_hist, sc, fc, ply):
    """Strelka-style razoring. Returns (should_return, value)."""
    score = ev + 192
    if score >= beta:
        return False, 0
    if depth == 1:
        new_score = qsearch(alpha, beta, bb, st, undo_bb, undo_st, mls,
                            scores, cap_hist, sc, fc, ply)
        return True, new_score if new_score > score else score
    score += 269
    if score < beta and depth <= 2:
        new_score = qsearch(alpha, beta, bb, st, undo_bb, undo_st, mls,
                            scores, cap_hist, sc, fc, ply)
        if new_score < beta:
            return True, new_score if new_score > score else score
    return False, 0


@njit(cache=False, fastmath=True)
def _prologue_head(alpha, beta, depth, ply, rep_idx, pv_node, bb, st, rep,
                   tt_key, tt_data, sc, fc):
    """Draw rules, mate distance pruning, TT probe and leaf dispatch.
    Returns (code, value, alpha, beta, tt_move)."""
    if ply and (_is_repetition(bb, rep, rep_idx) or st[FIFTY] >= 100):
        return NODE_RETURN, 0, alpha, beta, 0

    if USE_MDP and ply:
        if alpha < -MATE_VALUE + ply:
            alpha = -MATE_VALUE + ply
        if beta > MATE_VALUE - ply - 1:
            beta = MATE_VALUE - ply - 1
        if alpha >= beta:
            return NODE_RETURN, alpha, alpha, beta, 0

    score, tt_move = tt_probe(bb, alpha, beta, depth, ply, tt_key, tt_data)
    if ply and not pv_node and score != NO_HASH_ENTRY:
        return NODE_RETURN, score, alpha, beta, tt_move

    _check_time(sc, fc)

    if depth == 0:
        return NODE_QSEARCH, 0, alpha, beta, tt_move
    if ply > MAX_SEARCH_PLY - 1:
        return NODE_RETURN, evaluate(bb, st), alpha, beta, tt_move
    return NODE_CONTINUE, 0, alpha, beta, tt_move


@njit(cache=False, fastmath=True)
def _node_prologue(alpha, beta, depth, ply, rep_idx, pv_node, bb, st, undo_bb,
                   undo_st, mls, scores, cap_hist, rep, static_evals, tt_key,
                   tt_data, sc, fc):
    """Every non-recursive early exit of negamax, in BTC's order. Returns
    (code, value, alpha, beta, depth, tt_move, in_check, ev, improving)."""
    code, value, alpha, beta, tt_move = _prologue_head(
        alpha, beta, depth, ply, rep_idx, pv_node, bb, st, rep, tt_key,
        tt_data, sc, fc)
    if code != NODE_CONTINUE:
        return code, value, alpha, beta, depth, tt_move, 0, 0, 0

    sc[SC_NODES] += 1

    in_check = _in_check(bb, st)
    if in_check:
        depth += 1

    ev = evaluate(bb, st)
    improving = _improving(static_evals, ev, ply, in_check)
    quiet_node = not pv_node and not in_check

    if USE_RFP and depth < 3 and quiet_node and abs(beta) < MATE_SCORE:
        margin = 168 * (depth - improving)
        if ev - margin >= beta:
            return (NODE_RETURN, ev - margin, alpha, beta, depth, tt_move,
                    in_check, ev, improving)

    if USE_RAZOR and quiet_node and depth <= 3:
        done, value = _razor(alpha, beta, depth, ev, bb, st, undo_bb, undo_st,
                             mls, scores, cap_hist, sc, fc, ply)
        if done:
            return (NODE_RETURN, value, alpha, beta, depth, tt_move,
                    in_check, ev, improving)

    # internal iterative reductions: no TT move at depth means this iteration
    # is cheap and only exists to populate the TT for the next one
    if USE_IIR and ply > 0 and depth >= 6 and tt_move == 0:
        depth -= 1

    return (NODE_CONTINUE, 0, alpha, beta, depth, tt_move, in_check, ev,
            improving)


@njit(cache=False, fastmath=True)
def _skip_quiet(mv, depth, moves_searched, improving, ev, alpha, pv_node,
                in_check, opp_in_check):
    """Late move pruning and frontier futility on quiet moves."""
    if moves_searched == 0 or pv_node or in_check or opp_in_check:
        return False
    if (mv & (1 << 20)) or ((mv & 0xF0000) >> 16):
        return False
    if abs(alpha) >= MATE_SCORE:
        return False
    if USE_LMP and depth <= 8 \
            and moves_searched >= c_div(3 + depth * depth, 2 - improving):
        return True
    if USE_FUTILITY and depth <= 6 and ev + 184 * depth <= alpha:
        return True
    return False


@njit(cache=False, fastmath=True)
def _lmr_reduction(mv, depth, moves_searched, improving, in_check,
                   opp_in_check, pv_node, main_hist, cont_hist, played, ply):
    """Reduction for this move, or -1 when LMR does not apply."""
    if not USE_LMR or moves_searched < 5 or depth < 2 or in_check or pv_node:
        return -1
    if (mv & (1 << 20)) or ((mv & 0xF0000) >> 16):
        return 2 if opp_in_check else 3
    d = depth if depth < LMR_MAX_DEPTH else LMR_MAX_DEPTH - 1
    m = moves_searched if moves_searched < LMR_MAX_MOVES else LMR_MAX_MOVES - 1
    reduction = LMR_TABLE[d, m]
    hist = int64(main_hist[(mv & 0xF000) >> 12, (mv & 0xFC0) >> 6]) \
        + _cont_hist_score(cont_hist, played, ply, mv)
    reduction -= c_div(hist, 4096)
    if not improving:
        reduction += 1
    if opp_in_check:
        reduction -= 1
    return reduction if reduction > 0 else 0


@njit(cache=False, fastmath=True)
def negamax(alpha, beta, depth, ply, rep_idx, bb, st, undo_bb, undo_st, mls,
            scores, killers, main_hist, cap_hist, cont_hist, counters, played,
            static_evals, pv_table, pv_len, rep, tt_key, tt_data, sc, fc):
    pv_len[ply] = ply
    pv_node = beta - alpha > 1

    code, value, alpha, beta, depth, tt_move, in_check, ev, improving = \
        _node_prologue(alpha, beta, depth, ply, rep_idx, pv_node, bb, st,
                       undo_bb, undo_st, mls, scores, cap_hist, rep,
                       static_evals, tt_key, tt_data, sc, fc)
    if code == NODE_RETURN:
        return value
    if code == NODE_QSEARCH:
        return qsearch(alpha, beta, bb, st, undo_bb, undo_st, mls, scores,
                       cap_hist, sc, fc, ply)

    if USE_NULL and depth >= 3 and not in_check and ply:
        saved_ep, saved_hash = _make_null(bb, st, rep, rep_idx)
        played[ply + 1] = 0
        score = -negamax(-beta, -beta + 1, depth - 1 - 2, ply + 1,
                         rep_idx + 1, bb, st, undo_bb, undo_st, mls, scores,
                         killers, main_hist, cap_hist, cont_hist, counters,
                         played, static_evals, pv_table, pv_len, rep, tt_key,
                         tt_data, sc, fc)
        _unmake_null(bb, st, saved_ep, saved_hash)
        if sc[SC_STOP]:
            return 0
        if score >= beta:
            return beta

    cnt = generate_moves(bb, st, mls[ply])
    _sort_moves(bb, st, mls[ply], scores[ply], cnt, ply, tt_move, killers,
                main_hist, cap_hist, cont_hist, counters, played)

    hash_flag = HASH_ALPHA
    best_move = 0
    legal_count = 0
    moves_searched = 0
    quiets = mls[MAX_SEARCH_PLY + ply]
    quiet_cnt = 0
    captures = scores[MAX_SEARCH_PLY + ply]
    capture_cnt = 0

    for i in range(cnt):
        mv = mls[ply, i]
        rep[rep_idx] = bb[HASH]
        if make_move(bb, st, undo_bb, undo_st, ply, mv) == 0:
            continue
        legal_count += 1
        opp_in_check = _in_check(bb, st)

        if _skip_quiet(mv, depth, moves_searched, improving, ev, alpha,
                       pv_node, in_check, opp_in_check):
            unmake(bb, st, undo_bb, undo_st, ply)
            continue

        played[ply + 1] = mv

        if moves_searched == 0:
            score = -negamax(-beta, -alpha, depth - 1, ply + 1, rep_idx + 1,
                             bb, st, undo_bb, undo_st, mls, scores, killers,
                             main_hist, cap_hist, cont_hist, counters, played,
                             static_evals, pv_table, pv_len, rep, tt_key,
                             tt_data, sc, fc)
        else:
            reduction = _lmr_reduction(mv, depth, moves_searched, improving,
                                       in_check, opp_in_check, pv_node,
                                       main_hist, cont_hist, played, ply)
            if reduction >= 0:
                reduced = depth - 1 - reduction
                if reduced < 1:
                    reduced = 1
                score = -negamax(-alpha - 1, -alpha, reduced, ply + 1,
                                 rep_idx + 1, bb, st, undo_bb, undo_st, mls,
                                 scores, killers, main_hist, cap_hist,
                                 cont_hist, counters, played, static_evals,
                                 pv_table, pv_len, rep, tt_key, tt_data, sc, fc)
            else:
                score = alpha + 1
            if score > alpha:
                score = -negamax(-alpha - 1, -alpha, depth - 1, ply + 1,
                                 rep_idx + 1, bb, st, undo_bb, undo_st, mls,
                                 scores, killers, main_hist, cap_hist,
                                 cont_hist, counters, played, static_evals,
                                 pv_table, pv_len, rep, tt_key, tt_data, sc, fc)
                if alpha < score < beta:
                    score = -negamax(-beta, -alpha, depth - 1, ply + 1,
                                     rep_idx + 1, bb, st, undo_bb, undo_st,
                                     mls, scores, killers, main_hist,
                                     cap_hist, cont_hist, counters, played,
                                     static_evals, pv_table, pv_len, rep,
                                     tt_key, tt_data, sc, fc)

        unmake(bb, st, undo_bb, undo_st, ply)
        if sc[SC_STOP]:
            return 0

        if mv & (1 << 20):
            if capture_cnt < 256:
                captures[capture_cnt] = mv
                capture_cnt += 1
        elif quiet_cnt < 256:
            quiets[quiet_cnt] = mv
            quiet_cnt += 1
        moves_searched += 1

        if score > alpha:
            hash_flag = HASH_EXACT
            best_move = mv
            alpha = score
            pv_table[ply, ply] = mv
            for next_ply in range(ply + 1, pv_len[ply + 1]):
                pv_table[ply, next_ply] = pv_table[ply + 1, next_ply]
            pv_len[ply] = pv_len[ply + 1]
            if score >= beta:
                tt_record(bb, beta, depth, HASH_BETA, mv, ply, tt_key,
                          tt_data, int(sc[SC_TT_GEN]))
                _beta_cutoff_update(bb, st, mv, depth, ply, killers,
                                    main_hist, cap_hist, cont_hist, counters,
                                    played, quiets, quiet_cnt, captures,
                                    capture_cnt)
                return beta

    if legal_count == 0:
        return -MATE_VALUE + ply if in_check else 0

    tt_record(bb, alpha, depth, hash_flag, best_move, ply, tt_key, tt_data,
              int(sc[SC_TT_GEN]))
    return alpha


class SearchState:
    """Preallocated search memory. History tables persist across moves within
    a game like BTC's; the platform restarts the process per game.
    Move-list rows 0..63 serve the search plies, rows 64..127 hold each ply's
    tried-quiets for the history malus."""

    def __init__(self, tt_entries=TT_ENTRIES):
        assert tt_entries >= 8 and tt_entries & (tt_entries - 1) == 0, \
            "tt_entries must be a power of two (bucket mask derives from it)"
        self.undo_bb = np.zeros((MAX_PLY, 16), dtype=np.uint64)
        self.undo_st = np.zeros((MAX_PLY, 6), dtype=np.int64)
        self.mls = np.zeros((MAX_PLY, 256), dtype=np.int32)
        self.scores = np.zeros((MAX_PLY, 256), dtype=np.int64)
        self.killers = np.zeros((2, MAX_SEARCH_PLY + 1), dtype=np.int32)
        self.main_hist = np.zeros((12, 64), dtype=np.int16)
        self.cap_hist = np.zeros((12, 64, 12), dtype=np.int16)
        # [table][prev piece][prev to][piece][to]; table 0 is 1-ply, 1 is 2-ply
        self.cont_hist = np.zeros((2, 12, 64, 12, 64), dtype=np.int16)
        self.counters = np.zeros((12, 64), dtype=np.int32)
        self.played = np.zeros(MAX_SEARCH_PLY + 8, dtype=np.int32)
        self.static_evals = np.zeros(MAX_SEARCH_PLY + 8, dtype=np.int64)
        self.pv_table = np.zeros((MAX_SEARCH_PLY + 1, MAX_SEARCH_PLY + 1),
                                 dtype=np.int32)
        self.pv_len = np.zeros(MAX_SEARCH_PLY + 1, dtype=np.int64)
        self.rep = np.zeros(1024 + MAX_SEARCH_PLY + 8, dtype=np.uint64)
        self.tt_key = np.zeros(tt_entries, dtype=np.uint64)
        self.tt_data = np.zeros(tt_entries, dtype=np.uint64)
        self.sc = np.zeros(8, dtype=np.int64)
        self.fc = np.zeros(2, dtype=np.float64)


def _root_negamax(state, bb, st, alpha, beta, depth, rep_base):
    return negamax(alpha, beta, depth, 0, rep_base, bb, st, state.undo_bb,
                   state.undo_st, state.mls, state.scores, state.killers,
                   state.main_hist, state.cap_hist, state.cont_hist,
                   state.counters, state.played, state.static_evals,
                   state.pv_table, state.pv_len, state.rep, state.tt_key,
                   state.tt_data, state.sc, state.fc)


def _is_slower_mate(best_score, new_score):
    """Mate scores already encode distance (-MATE_VALUE + ply), so a shorter
    mate scores higher. A deeper iteration can still return a longer mate
    after a TT cutoff; never trade a proven mate for a slower one."""
    return best_score > MATE_SCORE and MATE_SCORE < new_score < best_score


def _adjusted_soft(soft_ms, stable_count, score_drop):
    adjusted = soft_ms
    if stable_count >= 5:
        adjusted = adjusted * 70 // 100
    if score_drop:
        adjusted = adjusted * 130 // 100
    return adjusted


def _aspiration_search(state, bb, st, prev_score, depth, rep_base):
    delta = 77
    # A window centred on a mate score lets TT entries for longer mates cut
    # off inside it, so the root can drift from mate in 3 to mate in 7.
    # Search mate scores with a full window instead.
    if depth >= 4 and abs(prev_score) < MATE_SCORE:
        alpha, beta = prev_score - delta, prev_score + delta
    else:
        alpha, beta = -INFINITY, INFINITY
    while True:
        score = _root_negamax(state, bb, st, alpha, beta, depth, rep_base)
        if state.sc[SC_STOP]:
            return score
        if alpha < score < beta:
            return score
        if score <= alpha:
            alpha = max(score - delta, -INFINITY)
        else:
            beta = min(score + delta, INFINITY)
        delta *= 2
        if delta > 1230:
            alpha, beta = -INFINITY, INFINITY


def search_position(state, bb, st, rep_keys, rep_count, soft_ms, hard_ms,
                    max_depth=MAX_SEARCH_PLY):
    """Iterative deepening driver. Returns (best_move, score, depth, nodes).
    rep_keys[0:rep_count] is the game history including the current position.
    Partial iterations are discarded (BTC behaviour): the move comes from the
    last completed depth, or any legal move if depth 1 was cut short."""
    started = time.perf_counter()
    state.sc[SC_NODES] = 0
    state.sc[SC_STOP] = 0
    state.sc[SC_TT_GEN] += 1
    state.killers[:] = 0
    state.pv_table[:] = 0
    state.pv_len[:] = 0
    state.static_evals[:] = 0
    state.played[:] = 0
    state.fc[0] = started + hard_ms / 1000.0

    rep_base = min(int(rep_count), 1024)
    state.rep[:rep_base] = rep_keys[rep_count - rep_base:rep_count]

    legal = legal_moves(bb, st)
    if not legal:
        return 0, 0, 0, 0
    if len(legal) == 1:
        return legal[0], 0, 0, 0

    best_move, best_score, completed = legal[0], 0, 0
    prev_best, prev_score, stable_count, score_drop = 0, 0, 0, False

    for depth in range(1, max_depth + 1):
        if depth > 1 and soft_ms > 0:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if elapsed_ms * 2 > _adjusted_soft(soft_ms, stable_count, score_drop):
                break
        score = _aspiration_search(state, bb, st, prev_score, depth, rep_base)
        if state.sc[SC_STOP]:
            break
        if state.pv_len[0] > 0 and not _is_slower_mate(best_score, score):
            best_move = int(state.pv_table[0, 0])
            best_score = score
            completed = depth
        cur_best = int(state.pv_table[0, 0])
        score_drop = depth > 1 and score < prev_score - 50
        stable_count = stable_count + 1 if depth > 1 and cur_best == prev_best else 0
        prev_best, prev_score = cur_best, score

    return best_move, best_score, completed, int(state.sc[SC_NODES])
