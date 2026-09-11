"""The search, ported from BTC and extended.

Iterative deepening with aspiration windows, PVS and a bucketed
transposition table. Move ordering from killers, counter moves, main and
continuation history, and SEE for captures. Pruning and reductions: null
move with an adaptive reduction, reverse futility, razoring, futility,
late move pruning, late move reductions, SEE pruning, internal iterative
reductions and ProbCut. Extensions: in-check, and the singular family
with multicut and a negative extension. Correction history adjusts the
static evaluation. Draws by repetition and the fifty-move rule are
detected here; quiescence is captures-only, with evasions when in check.

One function, deliberately. numba 0.67 compiles self-recursion but not
mutual recursion, so any helper that calls back into negamax fails to
compile and the whole search has to live in one place.

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
    B, EP, EP_KEYS, FIFTY, HASH, K, MAX_PLY, N, NO_SQ, OCC_A, ONE, P, Q, R,
    SIDE, SIDE_KEY, WHITE, ZERO, count_bits, generate_captures,
    generate_moves, is_under_attack, k, legal_moves, lsb, make_move, p,
    see_ge, unmake,
)
from btc_endgame import SPECIALISED_MAX_PIECES
from btc_eval import evaluate as full_evaluate
from btc_eval import evaluate_cached as full_evaluate_cached
from btc_eval import (NET_BUCKETS, NET_FT_B, NET_FT_W, NET_L1,
                      NET_TABLE, USE_NNUE)
from btc_nnue import refresh as refresh_acc
from btc_nnue import update as acc_update
from btc_psqt import MIRROR, PIECE_TABLES


def _flag(name):
    """Feature toggle, frozen into the compiled code at import."""
    return os.environ.get(name, "1") == "1"


def _tune(name, default):
    """Tunable search constant, frozen into the compiled code at import.

    Every one of these came from BTC, where they were tuned at C node counts.
    We search roughly 4-5 plies shallower, and the depth-gated heuristics cover
    a much larger fraction of a shallower tree, so the values are very unlikely
    to be optimal here. tune.py sweeps them against the arena.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return type(default)(raw)


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
# Transposition probe inside quiescence. Neither BTC nor the port had this.
# Every cutoff here also skips a full evaluation, which is worth more to us
# than to a C engine: evaluation is about a third of our runtime.
USE_QTT = not MINIMAL and _flag("BTC_QTT")
USE_QS_DELTA = not MINIMAL and _flag("BTC_QS_DELTA_PRUNE")
# Evasions in quiescence. Default OFF pending an SPRT; _flag() defaults ON, so
# this is an explicit test. A node reaching depth 0 in check never gets the
# check extension - _prologue_head returns NODE_QSEARCH before _node_prologue
# computes in_check - so quiescence is the only place the position can be
# handled correctly.
USE_QS_EVASION = not MINIMAL and _flag("BTC_QS_EVASION")
# Draw-rule corrections, Four draw-rule corrections. Default ON: shipped
# in batch 2 (24W 52D 13L over 89 games, 56.2%).
USE_DRAW_FIX = not MINIMAL and _flag("BTC_DRAW_FIX")
# A second occurrence against game history is not a draw -
# on its own flag, and OFF even when the rest of the group is on.
#
# It is correct by the rules and it measured *negative*: over 72 games it raised
# the threefold share 6.4 points and dropped the win share 6.2 against the same
# build without it. Scoring the second occurrence as a draw is wrong, but it
# also makes the engine refuse to repeat, and that refusal is worth more than
# the correctness. Engines that score repetitions correctly pair it with a
# contempt term so a drawn line is worth slightly less than zero; without that,
# fixing this alone only teaches the engine that walking toward a threefold is
# safe. Do not enable it until there is a draw-score offset to go with it.
USE_DRAW_HIST = not MINIMAL and os.environ.get("BTC_DRAW_HIST") == "1"
USE_NULL_ZUGZWANG_GUARD = not MINIMAL and _flag("BTC_NULL_ZUGZWANG")
# Static-eval gate on the null move (only try to prove a fail-high when the
# evaluation already says we are above beta). Default OFF: it breaks KBN vs K
# conversion (threefold after 10 moves; with it off, mate in 20). The direction
# is worth understanding - the gate only ever *restricts* null move, and
# restricting it makes conversion worse, because KBN vs K is converted by
# reaching depth and null-move cutoffs are what buy that depth. No measured
# benefit here to weigh against a lost half point, so it stays off.
USE_NULL_EVAL_GATE = not MINIMAL and os.environ.get("BTC_NULL_EVAL_GATE") == "1"
# Adaptive null-move reduction, R = 3 + depth/3 + min((eval-beta)/200, 3),
# against BTC's flat R = 2.
#
# **ON, and the story is a lesson about the harness rather than the feature.**
# At a 3 s clock it measured -33 elo [-68, +1] over 400 games and was switched
# off. That was an artefact: a 3 s *clock* is 45 ms per *move* (the budget
# divides by MTG), which searches about depth 6, and at depth 6 this formula
# gives R = 5, so `depth - 1 - R` = 0 and the null search collapses straight
# into quiescence - it proves nothing and prunes on nothing. Re-run at 30 s,
# far closer to the tournament control: +27 elo [-22, +76] over 197 games.
# Two positive results and a mechanism explaining the negative one.
USE_NULL_ADAPTIVE = not MINIMAL and _flag("BTC_NULL_ADAPTIVE")
# Cut-node awareness. Default OFF until measured in games: the 300-game A/B of
# this session's other search work came back at -29 elo despite a 31.6% node
# reduction, so nothing here ships on node counts again.
USE_CUTNODE_LMR = not MINIMAL and os.environ.get("BTC_CUTNODE_LMR") == "1"
USE_NULL_CUTNODE = not MINIMAL and os.environ.get("BTC_NULL_CUTNODE") == "1"
# Verification search behind a null-move fail-high. Default OFF pending an
# SPRT. Pair it with a raised BTC_NULL_BASE_R: verification is what makes a
# larger reduction affordable, and alone it only adds searches.
USE_NULL_VERIFY = not MINIMAL and os.environ.get("BTC_NULL_VERIFY") == "1"
# Correction-history planes beyond the original three. Default OFF pending an
# SPRT, and separately flagged so a grouped result can be bisected.
USE_CORR_MINOR = not MINIMAL and os.environ.get("BTC_CORR_MINOR") == "1"
USE_CORR_CONT = not MINIMAL and _flag("BTC_CORR_CONT")
# ProbCut early return off a wide transposition-table lower bound. Default
# ON: shipped in batch 2. The full search half is BTC_PROBCUT_FULL, which
# is off - it never pays for itself at our depth (+0.7% nodes at best).
USE_PROBCUT = not MINIMAL and _flag("BTC_PROBCUT")
# Singular extension family. SING_NEGATIVE is default ON (shipped in
# batch 2); SING_DOUBLE is off and unmeasured (+12.5% nodes on screen).
USE_SING_NEGATIVE = not MINIMAL and _flag("BTC_SING_NEGATIVE")
USE_SING_DOUBLE = not MINIMAL and os.environ.get("BTC_SING_DOUBLE") == "1"
# Batch 3. All default OFF pending an SPRT.
USE_DRAW_JITTER = not MINIMAL and os.environ.get("BTC_DRAW_JITTER") == "1"
USE_IMPROVING_BETA = not MINIMAL and os.environ.get("BTC_IMPROVING_BETA") == "1"
USE_OPP_WORSENING = not MINIMAL and os.environ.get("BTC_OPP_WORSENING") == "1"
USE_IIR_ALLNODE = not MINIMAL and os.environ.get("BTC_IIR_ALLNODE") == "1"
# Correction magnitude feeds the futility margin and the LMR reduction.
USE_CORR_MARGIN = not MINIMAL and os.environ.get("BTC_CORR_MARGIN") == "1"
# Stop extending once the root says the game is already decided.
USE_SEEK_MATE = not MINIMAL and os.environ.get("BTC_SEEK_MATE") == "1"
# Skip the singular verification when a piece is shuffling.
USE_SHUFFLE_GUARD = not MINIMAL and os.environ.get("BTC_SHUFFLE_GUARD") == "1"
USE_LMR_TTPV = not MINIMAL and os.environ.get("BTC_LMR_TTPV") == "1"
USE_LMR_TTCAP = not MINIMAL and os.environ.get("BTC_LMR_TTCAP") == "1"
USE_PROBCUT_FULL = not MINIMAL and os.environ.get("BTC_PROBCUT_FULL") == "1"
# Capture ordering by victim value and capture history, with no
# least-valuable-attacker term. Once SEE decides
# good from bad the attacker is noise.
USE_CAP_SCORE_V2 = not MINIMAL and _flag("BTC_CAP_SCORE_V2")
USE_CAP_SEE_DYN = not MINIMAL and os.environ.get("BTC_CAP_SEE_DYN") == "1"
# Give the null-move child at least one ply of real search. Without it
# the reduction can exceed the depth remaining and the child is entered
# at a negative depth, where five separate margins invert their sign:
# RFP's `RFP_MARGIN * (depth - improving)` becomes a BONUS, futility's
# does too, and LMP, _history_bonus and _update_correction are all
# quadratic or depth-scaled and treat depth -5 like depth +5. It also
# ends the _tt_pack corruption at its source: a negative depth sets
# every bit above 42, so the entry reads back as depth 127 and becomes
# permanently unreplaceable. Null move is the ONLY producer of negative
# depths - every other call site is floored or gated.
#
# Measured over searchbench at depth 9: corrupt entries 953 -> 0,
# occupancy 133,928 -> 166,023, KBN vs K mate 18 -> mate 15.
# Preferred over BTC_DEPTH_CLAMP, which fixes the same family
# downstream but costs nine moves of conversion (mate 18 -> 27).
USE_NULL_CD_FLOOR = not MINIMAL and _flag("BTC_NULL_CD_FLOOR")
USE_TT_KEEP_MOVE = not MINIMAL and os.environ.get("BTC_TT_KEEP_MOVE") == "1"
USE_ASP_V2 = not MINIMAL and os.environ.get("BTC_ASP_V2") == "1"
USE_EVASION_ORDER = not MINIMAL and os.environ.get("BTC_EVASION_ORDER") == "1"
# Feed LMP and the LMR table index the count of every legal
# move considered - rather than our count of moves actually searched,
# which stalls whenever futility or SEE pruning fires and made both
# pruners systematically more conservative than intended.
#
# Note the _see_prunable half is inert by construction: that function
# reads its counter only as `== 0`, and the two counters are zero on
# exactly the same move (the first legal one, which is never pruned).
USE_MOVECOUNT_FIX = not MINIMAL and _flag("BTC_MOVECOUNT_FIX")
# Start reducing at move 3 instead of 5. Standard practice is to reduce
# from the second move, and at a cut node that cuts off in two or three
# moves an onset of 5 reduces nothing at all. Gated on material_scale: ungated this costs KBN vs K
# nine moves (mate 15 -> 26), because below SPECIALISED_MAX_PIECES the
# score is a clamped mating drive rather than centipawns and reducing
# discards the only lines that make progress. Same unit-scale hazard
# that delta pruning and fail-soft quiescence hit.
USE_LMR_ONSET = not MINIMAL and _flag("BTC_LMR_ONSET")
# Treat a negative search depth as quiescence rather than searching it full
# width. Default OFF only so a grouped SPRT can be bisected; this is a
# correctness fix, not a heuristic.
USE_DEPTH_CLAMP = not MINIMAL and os.environ.get("BTC_DEPTH_CLAMP") == "1"
# Same negative-depth pathology: floor the depth feeding the history
# bonus. Separate flag so a grouped result stays bisectable.
USE_HIST_DEPTH_FLOOR = not MINIMAL and os.environ.get("BTC_HIST_DEPTH_FLOOR") == "1"
# Correction history. Default ON, and matched: +113 =85 -102 over 300 games,
# 51.8%, +13 elo [-27, +52]. It cuts the static
# evaluation's mean absolute prediction error over the benchmark from 562.9
# to 458.9, an 18.5% improvement.
#
# The old note here said it costs 20.9% more nodes. That is stale: measured
# at depth 11, shipping is 899,621 nodes against 982,968 with BTC_CORRHIST=0,
# so turning it OFF now costs 9.3%. The continuation plane changed the
# economics. Do not spend a match slot re-testing this.
USE_CORR_HIST = not MINIMAL and _flag("BTC_CORRHIST")
# History pruning of quiets. Default OFF: measured at nothing, and the reason
# is structural rather than a bad threshold. It is redundant with LMP as
# configured here - at depth <= 4 LMP already admits only (3 + depth^2)/2
# quiets, which is fewer than history pruning would reach. Margin 1024*depth
# pruned 3 nodes out of 943,275; scanning 512/256/128/64 moved node counts
# non-monotonically (938,967 / 975,001 / 939,584 / 1,042,289), which is the
# signature of a heuristic doing nothing but perturbing move order. Worth
# revisiting only if LMP is retuned to admit more moves.
USE_HIST_PRUNE = not MINIMAL and os.environ.get("BTC_HISTPRUNE") == "1"
# Disable the centipawn-margin pruning rules (RFP, razoring, futility) where
# the evaluation is a mating drive score rather than material.
#
# **Default OFF: principled, and measurably worse.** The unit mismatch is real -
# it is the same one that broke KBN vs K with delta pruning and KR vs K with
# correction history - but unlike those two it breaks nothing here, and guarding
# it costs depth: KBN vs K converts in 18 moves unguarded against 22 guarded.
# The reason the theory does not bite is that the specialised drive score is
# monotonic in closeness to mate, so an RFP cutoff on it prunes nodes that are
# genuinely hopeless for the defender rather than misjudging material. Depth is
# worth more than the tidier invariant. Kept behind a flag because the argument
# is close and the constants it interacts with are being retuned.
USE_EVAL_SCALE_GUARD = not MINIMAL and os.environ.get("BTC_EVAL_SCALE_GUARD") == "1"
CORR_DIAG = os.environ.get("BTC_CORR_DIAG") == "1"
# Depth-scaled SEE pruning in the main move loop. BTC uses SEE for ordering
# and for quiescence pruning but never here.
#
# Default OFF: it works, but it is exactly free. At thresholds 20/10 it cuts
# nodes to depth 11 by 7.2% (630,690 against 679,560) and the wall time is
# unchanged, because see_ge costs what the pruned nodes saved. Two reasons
# specific to this engine: our SEE is a swap-off loop in numba, expensive
# relative to a node, and _sort_moves already runs see_ge on every capture to
# split good from bad, so pruning recomputes it. Reusing that result would
# make capture pruning nearly free and is the way to revisit this.
# Pushing harder backfires: 5/5 at depth<=12 raised nodes to 782,359.
# Read the move and its score through row views hoisted out of the move
# loops rather than indexing the 2-D arrays per move.
USE_ROWVIEW = os.environ.get("BTC_ROWVIEW", "1") == "1"
# Unrolled plane scan in _captured_piece.
USE_CAP_UNROLL = os.environ.get("BTC_CAP_UNROLL", "1") == "1"
# Defer the accumulator refresh past the pruning tests, so a move that is
# pruned never pays for one.
USE_LAZY_ACC = os.environ.get("BTC_LAZY_ACC", "1") == "1"
USE_SEE_PRUNE = not MINIMAL and _flag("BTC_SEEPRUNE")
# Singular extensions default OFF. On its own it is fine, and it measured
# +20 elo over 120 games (interval -42 to +84, so no evidence of gain). But
# combined with RFP_MARGIN=130 it breaks KBN vs K conversion: both prune or
# extend more aggressively and together they cut the mating line
# (test_convert.py, threefold repetition after 17 moves). RFP=130 has real
# evidence behind it (+67, interval excluding zero), singular did not at
# the time. Singular was later re-enabled and is ON by default below;
# BTC_SINGULAR=0 disables it.
# Certified at +52 elo [+1, +105] over 176 games, 57.4%, as a same-binary
# A/B. Default ON via _flag; BTC_SINGULAR=0 disables. The material_scale
# gate inside the block is what makes it safe - see there.
USE_SINGULAR = not MINIMAL and _flag("BTC_SINGULAR")
# Multi-cut off the back of the singular verification. Requires singular,
# because it reuses that search.
USE_MULTICUT = USE_SINGULAR \
    and os.environ.get("BTC_MULTICUT", "1") == "1"
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
# any score below this is a bad capture and nothing else: the tier base
# spans about [-8k, +69k], and the next tier down is quiet-move history
SCORE_BAD_CAPTURE_LIMIT = -500_000

HIST_MAX = 8192

# Correction history. One combined table: plane 0 is keyed on the pawn
# structure, planes 1 and 2 on each colour's non-pawn configuration. Combining
# them keeps the recursion to a single extra parameter.
CORR_SIZE = 1 << 14
CORR_MASK = CORR_SIZE - 1
CORR_LIMIT = 1024
CORR_NORM = 128
CORR_WEIGHT_PAWN = _tune("BTC_CORR_W_PAWN", 6)
CORR_WEIGHT_NONPAWN = _tune("BTC_CORR_W_NONPAWN", 5)
# Untuned. Deliberately below the pawn weight: two extra planes at the
# existing weights would nearly double the total correction the evaluation
# can take, which is a larger change than the planes themselves.
CORR_WEIGHT_MINOR = _tune("BTC_CORR_W_MINOR", 4)
CORR_WEIGHT_CONT = _tune("BTC_CORR_W_CONT", 4)
# 0 pawn, 1 white non-pawn, 2 black non-pawn, 3 minor, 4 continuation.
CORR_PLANES = 5

# Tunable search constants. Defaults are BTC's values; see _tune above for why
# they are unlikely to be optimal at our node counts.
LMR_BASE = _tune("BTC_LMR_BASE", 0.7844)
LMR_DIV = _tune("BTC_LMR_DIV", 2.4696)
LMR_HIST_DIV = max(1, _tune("BTC_LMR_HIST_DIV", 4096))
LMR_MIN_MOVES = _tune("BTC_LMR_MIN_MOVES", 5)
# Onset used only while the evaluation is in centipawns. Below the
# specialised-endgame threshold the score is a clamped mating drive and
# reducing there loses the short mate.
LMR_MIN_MOVES_MAT = max(1, _tune("BTC_LMR_MIN_MOVES_MAT", 3))
LMR_MIN_DEPTH = _tune("BTC_LMR_MIN_DEPTH", 2)
LMR_CUTNODE = _tune("BTC_LMR_CUTNODE", 2)
# Flat reduction for a capture or promotion. The stronger alternative is
# to reduce captures through the table and then subtract a statscore
# built from the victim value and capture history.
LMR_CAPTURE = _tune("BTC_LMR_CAPTURE", 3)
LMR_CAPTURE_CHECK = _tune("BTC_LMR_CAPTURE_CHECK", 2)
# One ply, not the two-and-a-bit used by engines that search far deeper:
# a whole ply is already a large share of a depth-14 search.
LMR_TTPV = _tune("BTC_LMR_TTPV_R", 1)
LMR_TTCAP = _tune("BTC_LMR_TTCAP_R", 1)
LMP_BASE = _tune("BTC_LMP_BASE", 3)
LMP_MAX_DEPTH = _tune("BTC_LMP_MAX_DEPTH", 8)
FUTILITY_MARGIN = _tune("BTC_FUT_MARGIN", 184)
FUTILITY_MAX_DEPTH = _tune("BTC_FUT_MAX_DEPTH", 6)
# Given back from the futility margin when the opponent's evaluation is
# falling too. Untuned.
OPP_WORSENING_MARGIN = _tune("BTC_OPP_WORSENING_MARGIN", 24)
# Continuation-history planes: plies 1..CONT_PLANES back. 2 is the
# shipped behaviour; the extra planes are pruned at compile time.
# Clamped at 2: plane 1 is read and written unconditionally, so a lower
# value is an out-of-bounds store with bounds checking off.
CONT_PLANES = max(2, _tune("BTC_CONT_PLANES", 2))
# Divisors on the correction magnitude. Untuned. The magnitude is a
# centipawn quantity bounded by the correction weights, so it lands in
# the low hundreds at most.
CORR_MARGIN_DIV = max(1, _tune("BTC_CORR_MARGIN_DIV", 2))
CORR_LMR_DIV = max(1, _tune("BTC_CORR_LMR_DIV", 40))
# Root depth and score beyond which singular extensions stop. Untuned.
SEEK_MATE_DEPTH = _tune("BTC_SEEK_MATE_DEPTH", 16)
SEEK_MATE_SCORE = _tune("BTC_SEEK_MATE_SCORE", 2000)
# A shuffle needs a real fifty-move count behind it, distance from the
# last null move, and enough plies to be a pattern rather than an
# accident.
SHUFFLE_MIN_FIFTY = _tune("BTC_SHUFFLE_MIN_FIFTY", 10)
SHUFFLE_MIN_FROM_NULL = _tune("BTC_SHUFFLE_MIN_FROM_NULL", 6)
SHUFFLE_MIN_PLY = _tune("BTC_SHUFFLE_MIN_PLY", 20)
# Margin and depth cap were swept jointly: the optimal margin falls as the cap
# rises, so tuning either alone searches the wrong axis. Cap 5 / margin 90 is
# the best point that still converts KBN vs K - margin 90 fails conversion at
# cap 3, and the failure boundary is threshold-like rather than a trend, so
# node counts are the diagnostic here and conversion is the verdict.
RFP_MARGIN = _tune("BTC_RFP_MARGIN", 90)
RFP_MAX_DEPTH = _tune("BTC_RFP_MAX_DEPTH", 5)
NULL_REDUCTION = _tune("BTC_NULL_R", 2)
NULL_BASE_R = _tune("BTC_NULL_BASE_R", 3)
NULL_DEPTH_DIV = max(1, _tune("BTC_NULL_DEPTH_DIV", 3))
NULL_EVAL_DIV = max(1, _tune("BTC_NULL_EVAL_DIV", 200))
NULL_EVAL_MAX = _tune("BTC_NULL_EVAL_MAX", 3)
NULL_MIN_DEPTH = _tune("BTC_NULL_MIN_DEPTH", 3)
# Below this the null search is cheap enough that verifying it costs more
# than the cutoff is worth.
NULL_VERIFY_MIN_DEPTH = _tune("BTC_NULL_VERIFY_MIN_DEPTH", 16)
ASPIRATION_DELTA = _tune("BTC_ASP_DELTA", 77)
# Window growth per failed re-search, as a fraction of 128.
ASPIRATION_GROWTH = max(1, _tune("BTC_ASP_GROWTH", 47))
SINGULAR_MIN_DEPTH = _tune("BTC_SINGULAR_MIN_DEPTH", 8)
# How far above beta a stored bound must sit before it alone ends the node,
# and how much shallower that entry may be. Untuned here. The returned bound is
# beta + this, so the node is skipped unless that stays clear of MATE_SCORE:
# a value inside the mate range is read as a proven mate everywhere downstream.
# Clamped at zero: a negative margin would both weaken the mate-range
# guard below and return a bound under beta as if it were a fail-high.
PROBCUT_MARGIN = max(0, _tune("BTC_PROBCUT_MARGIN", 428))
PROBCUT_MIN_DEPTH = _tune("BTC_PROBCUT_MIN_DEPTH", 5)
PROBCUT_DEPTH_SLACK = _tune("BTC_PROBCUT_SLACK", 4)
# The search half. The usual formulation searches captures at depth - 4
# and requires depth > 3; at our depths that leaves a verification of 1
# to 2 plies, which is why the qsearch filter matters as much as the
# margin.
PROBCUT_FULL_MIN_DEPTH = _tune("BTC_PROBCUT_FULL_MIN_DEPTH", 5)
PROBCUT_FULL_REDUCTION = max(1, _tune("BTC_PROBCUT_FULL_R", 4))
# How far below singular_beta the verification must fail for a second ply.
SINGULAR_DOUBLE_MARGIN = _tune("BTC_SING_DOUBLE_MARGIN", 80)
# Reduction applied to a transposition move the verification could not prove
# singular. The stronger evidence - a stored score already at or above beta -
# earns the deeper cut.
SINGULAR_NEG_TT = _tune("BTC_SING_NEG_TT", 3)
SINGULAR_NEG_CUT = _tune("BTC_SING_NEG_CUT", 2)
QS_DELTA_MARGIN = _tune("BTC_QS_DELTA", 200)
HIST_PRUNE_MAX_DEPTH = _tune("BTC_HISTP_MAX_DEPTH", 4)
HIST_PRUNE_MARGIN = _tune("BTC_HISTP_MARGIN", 1024)
SEE_PRUNE_MAX_DEPTH = _tune("BTC_SEEP_MAX_DEPTH", 8)
# Adaptive re-search depth after a failed-high reduction.
LMR_DEEPER = _tune("BTC_LMR_DEEPER", 50)
LMR_SHALLOWER = _tune("BTC_LMR_SHALLOWER", 8)
SEE_PRUNE_QUIET = _tune("BTC_SEEP_QUIET", 80)
SEE_PRUNE_CAPTURE = _tune("BTC_SEEP_CAPTURE", 30)

# LMR base reduction table, indexed [depth][move index].
# Precomputed at import because log() per node is wasted work.
LMR_MAX_DEPTH = 64
LMR_MAX_MOVES = 64
_LMR = np.zeros((LMR_MAX_DEPTH, LMR_MAX_MOVES), dtype=np.int64)
for _d in range(1, LMR_MAX_DEPTH):
    for _m in range(1, LMR_MAX_MOVES):
        _LMR[_d, _m] = int(LMR_BASE + np.log(_d) * np.log(_m) / LMR_DIV)
LMR_TABLE = _LMR

SC_NODES, SC_STOP, SC_TT_GEN = 0, 1, 2
# Diagnostic only (BTC_CORR_DIAG): absolute static-eval prediction error
# summed over correction-update sites, and the number of those sites.
SC_CORR_ERR, SC_CORR_N = 3, 4
# Where the pre-root game history ends in `rep`. Constant for a whole search,
# so it rides here rather than becoming another negamax parameter.
SC_REP_BASE = 5
# How far this node's evaluation was corrected, as an absolute value.
# Written by _node_prologue, read into a local by negamax straight after -
# a recursive call overwrites it at a deeper ply.
SC_CORR_ADJ = 7
# Size the array from the indices. numba indexes without bounds
# checking, so an index past the end is a silent bad read or write
# rather than an error.
# Root depth and score from the last completed iteration, for seekMate.
SC_ROOT_DEPTH, SC_ROOT_SCORE = 8, 9
# The pv bit of the entry the prologue probed. Read into a local right
# after _node_prologue returns and before any recursive call, like
# SC_CORR_ADJ: a deeper ply overwrites the slot.
SC_TT_PV = 10
SC_COUNT = 11
# Ply below which null move is disabled, while a verification search runs.
# 0 means no verification is in progress. Rides in `sc` for the same reason:
# an extra array argument to negamax costs NRT traffic on every call.
SC_NMP_MIN = 6

TEMPO = 28
OPENING_PHASE = 15196
ENDGAME_PHASE = 1841

MATERIAL_MG = np.array([126, 781, 825, 1276, 2538, 10000], dtype=np.int64)
MATERIAL_EG = np.array([208, 854, 915, 1380, 2682, 10000], dtype=np.int64)

# Capture ordering by the value of what is taken, with no attacker term.
CAP_VALUE = np.array(list(MATERIAL_MG) * 2, dtype=np.int64)
# 7 * value spans roughly 880 (pawn) to 17800 (queen) against a capture
# history bounded at HIST_MAX, so history can move a capture within a
# piece class and a little beyond it, but cannot invert queen against pawn.
CAP_VALUE_MULT = _tune("BTC_CAP_VALUE_MULT", 7)
CAP_SEE_DIV = max(1, _tune("BTC_CAP_SEE_DIV", 18))

MVV_LVA = np.array([
    [105, 205, 305, 405, 505, 605] * 2,
    [104, 204, 304, 404, 504, 604] * 2,
    [103, 203, 303, 403, 503, 603] * 2,
    [102, 202, 302, 402, 502, 602] * 2,
    [101, 201, 301, 401, 501, 601] * 2,
    [100, 200, 300, 400, 500, 600] * 2,
] * 2, dtype=np.int64)


@njit(cache=False, fastmath=True, error_model='numpy')
def c_div(a, b):
    """C integer division: truncation toward zero. Requires b > 0."""
    q = a // b
    if q < 0 and q * b != a:
        q += 1
    return q


@njit(cache=False, fastmath=True, error_model='numpy')
def interpolate(opening_value, endgame_value, stage_score):
    return c_div(opening_value * stage_score
                 + endgame_value * (OPENING_PHASE - stage_score), OPENING_PHASE)


@njit(cache=False, fastmath=True, error_model='numpy')
def game_stage_score(bb):
    score = 0
    for pc in range(N, Q + 1):
        score += (count_bits(bb[pc]) + count_bits(bb[pc + 6])) * MATERIAL_MG[pc]
    return score


@njit(cache=False, fastmath=True, error_model='numpy')
def _material_term(pc, stage, stage_score):
    if stage == 2:
        return interpolate(MATERIAL_MG[pc], MATERIAL_EG[pc], stage_score)
    return MATERIAL_MG[pc] if stage == 0 else MATERIAL_EG[pc]


@njit(cache=False, fastmath=True, error_model='numpy')
def _pst_term(pc, sq, stage, stage_score):
    if stage == 2:
        return interpolate(PIECE_TABLES[0, pc, sq], PIECE_TABLES[1, pc, sq],
                           stage_score)
    return PIECE_TABLES[stage, pc, sq]


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
def evaluate(bb, st):
    if USE_FULL_EVAL:
        return full_evaluate(bb, st)
    return _evaluate_psqt(bb, st)


# Diagnostic: force every evaluation to rebuild the accumulator from the board
# instead of using the maintained one. Results must be *identical* - same score,
# same best move, same node count - because the two differ only in how the
# accumulator was produced. Any divergence means the search is indexing the
# accumulator stack wrongly, which test_accumulator.py cannot see because it
# walks the tree itself rather than through negamax.
ACC_REFRESH = os.environ.get("BTC_ACC_REFRESH") == "1"


@njit(cache=False, fastmath=True, error_model='numpy')
def evaluate_cached(bb, st, acc_row):
    """Evaluation reading the accumulator the move loop already maintains."""
    if not USE_FULL_EVAL:
        return _evaluate_psqt(bb, st)
    if ACC_REFRESH:
        return full_evaluate(bb, st)
    return full_evaluate_cached(bb, st, acc_row)


@njit(cache=False, fastmath=True, error_model='numpy')
def _tt_pack(move, score, depth, flag, age, pv):
    v = uint64(move & 0xFFFFFF)
    v |= uint64(score + SCORE_BIAS) << uint64(24)
    v |= uint64(depth) << uint64(42)
    v |= uint64(flag) << uint64(49)
    v |= uint64(age & 0xFF) << uint64(51)
    if USE_LMR_TTPV and pv:
        v |= uint64(1) << uint64(59)
    return v


# int64() casts are load-bearing: numba's int(uint64) stays uint64, and a
# later int64/uint64 branch unification silently promotes to float64

@njit(cache=False, fastmath=True, error_model='numpy')
def _tt_move(data):
    return int64(data & uint64(0xFFFFFF))


@njit(cache=False, fastmath=True, error_model='numpy')
def _tt_score(data):
    return int64((data >> uint64(24)) & uint64(0x3FFFF)) - SCORE_BIAS


@njit(cache=False, fastmath=True, error_model='numpy')
def _tt_depth(data):
    return int64((data >> uint64(42)) & uint64(0x7F))


@njit(cache=False, fastmath=True, error_model='numpy')
def _tt_flag(data):
    return int64((data >> uint64(49)) & uint64(0x3))


@njit(cache=False, fastmath=True, error_model='numpy')
def _tt_pv(data):
    return int64((data >> uint64(59)) & uint64(1))


@njit(cache=False, fastmath=True, error_model='numpy')
def _tt_age(data):
    return int64((data >> uint64(51)) & uint64(0xFF))


@njit(cache=False, fastmath=True, error_model='numpy')
def tt_probe(bb, alpha, beta, depth, ply, tt_key, tt_data, fifty):
    """Returns (cutoff score or NO_HASH_ENTRY, tt move or 0, pv bit)."""
    if not USE_TT:
        return NO_HASH_ENTRY, 0, int64(0)
    key = bb[HASH]
    base = int64(key & uint64(tt_key.shape[0] // 4 - 1)) * 4
    for i in range(4):
        if tt_key[base + i] != key:
            continue
        data = tt_data[base + i]
        move = _tt_move(data)
        pv = _tt_pv(data)
        score = _tt_score(data)
        if score < -MATE_SCORE:
            score += ply
        if score > MATE_SCORE:
            score -= ply
        # A stored mate that cannot arrive before the fifty-move draw is not a
        # mate. "Mate in 20" retrieved at a fifty count of 85 is unreachable,
        # and returning it makes the search chase a win it cannot have. The
        # entry's move is still useful for ordering, so only the cutoff is
        # withheld and the node searches instead.
        if USE_DRAW_FIX and score > MATE_SCORE \
                and MATE_VALUE - score > 100 - fifty:
            return NO_HASH_ENTRY, move, pv
        if USE_DRAW_FIX and score < -MATE_SCORE \
                and MATE_VALUE + score > 100 - fifty:
            return NO_HASH_ENTRY, move, pv
        if _tt_depth(data) >= depth:
            flag = _tt_flag(data)
            if flag == HASH_EXACT:
                return score, move, pv
            # Fail-soft: hand back the stored score, not the window edge.
            # A node that proved +300 against a beta of +50 is recorded as
            # +50 under fail-hard, so every later probe of that position
            # inherits a bound 250 centipawns worse than what was proven.
            if flag == HASH_ALPHA and score <= alpha:
                return score, move, pv
            if flag == HASH_BETA and score >= beta:
                return score, move, pv
        return NO_HASH_ENTRY, move, pv
    return NO_HASH_ENTRY, 0, int64(0)


@njit(cache=False, fastmath=True, error_model='numpy')
def tt_peek(bb, ply, tt_key, tt_data):
    """Raw entry for this position: (hit, depth, flag, ply-adjusted score).
    Singular extensions need the stored bound even when it was too shallow to
    produce a cutoff, which tt_probe does not expose."""
    key = bb[HASH]
    base = int64(key & uint64(tt_key.shape[0] // 4 - 1)) * 4
    for i in range(4):
        if tt_key[base + i] != key:
            continue
        data = tt_data[base + i]
        score = _tt_score(data)
        if score < -MATE_SCORE:
            score += ply
        if score > MATE_SCORE:
            score -= ply
        return 1, _tt_depth(data), _tt_flag(data), score
    return 0, 0, 0, 0


@njit(cache=False, fastmath=True, error_model='numpy')
def tt_record(bb, score, depth, flag, move, ply, tt_key, tt_data, gen, pv):
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
    if USE_LMR_TTPV and tt_key[slot] == key and _tt_pv(tt_data[slot]):
        # Sticky: once a position has been seen on the
        # principal variation the bit survives every later rewrite of the
        # same entry, including quiescence's, which has no pv notion of its
        # own and always passes 0.
        pv = int64(1)
    # depth-preferred: keep a deeper same-key entry unless exact or close;
    # still refresh its move and age (search.cpp recordHash)
    if tt_key[slot] == key and flag != HASH_EXACT and depth < _tt_depth(tt_data[slot]) - 3:
        old = tt_data[slot]
        kept_move = move if move else _tt_move(old)
        tt_data[slot] = _tt_pack(kept_move, _tt_score(old), _tt_depth(old),
                                 _tt_flag(old), gen, pv)
        return
    if score < -MATE_SCORE:
        score -= ply
    if score > MATE_SCORE:
        score += ply
    if USE_DEPTH_CLAMP and depth < 0:
        # Discard. _tt_pack shifts depth into an unsigned field, so a
        # negative value sets every bit above 42 and the entry reads back
        # as depth 127 with a corrupt flag, which a depth-preferred policy
        # can never replace.
        #
        # This must return BEFORE tt_key[slot] is written. Stamping the key
        # and then skipping the data leaves the slot claiming this position
        # while holding the previous occupant's entry, and on an empty slot
        # that reads back as flag 0 - HASH_EXACT - with score -SCORE_BIAS,
        # which tt_probe will cut on. Worse than the corrupt entry it
        # replaces, which was at least inert.
        #
        # Storing at depth 0 instead is also wrong: a depth-0 entry is one
        # quiescence can cut on, which flattens the mating drive and loses
        # KBN vs K.
        #
        # Not an airtight guarantee: _node_prologue adds a ply when in
        # check, after the gate in _prologue_head has already tested the
        # incoming depth, so a node entered at depth -1 while in check
        # still records at depth 0. That is unchanged from the flag being
        # off, so it is not a regression, but the rule is not absolute.
        return
    if USE_TT_KEEP_MOVE and move == 0 and tt_key[slot] == key:
        # Same position, and this store has no move to offer. Keeping the
        # one already there costs nothing and preserves the ordering and
        # singular-extension guidance a deeper search paid for.
        move = _tt_move(tt_data[slot])
    tt_key[slot] = key
    tt_data[slot] = _tt_pack(move, score, depth, flag, gen, pv)


@njit(cache=False, fastmath=True, error_model='numpy')
def _update_history(hist, piece, tgt, bonus):
    if bonus > HIST_MAX:
        bonus = HIST_MAX
    if bonus < -HIST_MAX:
        bonus = -HIST_MAX
    e = int(hist[piece, tgt])
    e += bonus - c_div(e * abs(bonus), HIST_MAX)
    hist[piece, tgt] = e


@njit(cache=False, fastmath=True, error_model='numpy')
def _history_bonus(depth):
    if USE_HIST_DEPTH_FLOOR and depth < 0:
        # The formula is quadratic, so a negative depth earns a bonus as
        # if it were positive: depth -5 pays the same as depth +3. The
        # null-move reduction reaches negative depths, so this credits a
        # search below the horizon as though it were three plies deep.
        depth = int64(0)
    bonus = 16 * depth * depth + 32 * depth - 16
    if bonus > 1200:
        bonus = 1200
    if bonus < 0:
        bonus = 0
    return bonus


@njit(cache=False, fastmath=True, error_model='numpy')
def _pawn_corr_index(bb):
    key = bb[P] * uint64(0x9E3779B97F4A7C15) + bb[P + 6] * uint64(0xC2B2AE3D27D4EB4F)
    return int64((key >> uint64(32)) & uint64(CORR_MASK))


@njit(cache=False, fastmath=True, error_model='numpy')
def _white_corr_index(bb):
    key = (bb[N] * uint64(0x9E3779B97F4A7C15) + bb[B] * uint64(0xC2B2AE3D27D4EB4F)
           + bb[R] * uint64(0x165667B19E3779F9) + bb[Q] * uint64(0xD6E8FEB86659FD93)
           + bb[K] * uint64(0xA0761D6478BD642F))
    return int64((key >> uint64(32)) & uint64(CORR_MASK))


@njit(cache=False, fastmath=True, error_model='numpy')
def _black_corr_index(bb):
    key = (bb[N + 6] * uint64(0x9E3779B97F4A7C15)
           + bb[B + 6] * uint64(0xC2B2AE3D27D4EB4F)
           + bb[R + 6] * uint64(0x165667B19E3779F9)
           + bb[Q + 6] * uint64(0xD6E8FEB86659FD93)
           + bb[K + 6] * uint64(0xA0761D6478BD642F))
    return int64((key >> uint64(32)) & uint64(CORR_MASK))


@njit(cache=False, fastmath=True, error_model='numpy')
def _minor_corr_index(bb):
    key = (bb[N] * uint64(0x9E3779B97F4A7C15)
           + bb[B] * uint64(0xC2B2AE3D27D4EB4F)
           + bb[N + 6] * uint64(0x165667B19E3779F9)
           + bb[B + 6] * uint64(0xD6E8FEB86659FD93))
    return int64((key >> uint64(32)) & uint64(CORR_MASK))


@njit(cache=False, fastmath=True, error_model='numpy')
def _cont_corr_index(played, ply):
    """Index for the moves 2 and 4 plies back, or -1 when neither exists.

    played[ply] is the move that reached this node, so 2 and 4 plies back are
    played[ply - 1] and played[ply - 3]."""
    key = uint64(0)
    seen = False
    if ply >= 2 and played[ply - 1] != 0:
        prev = played[ply - 1]
        slot = uint64((((prev & 0xF000) >> 12) << 6) | ((prev & 0xFC0) >> 6))
        key += (slot + uint64(1)) * uint64(0x9E3779B97F4A7C15)
        seen = True
    if ply >= 4 and played[ply - 3] != 0:
        prev = played[ply - 3]
        slot = uint64((((prev & 0xF000) >> 12) << 6) | ((prev & 0xFC0) >> 6))
        key += (slot + uint64(1)) * uint64(0xC2B2AE3D27D4EB4F)
        seen = True
    if not seen:
        # No history to key on. Index 0 would alias every such node together.
        return int64(-1)
    return int64((key >> uint64(32)) & uint64(CORR_MASK))


@njit(cache=False, fastmath=True, error_model='numpy')
def _cont_corr_delta(corr, played, ply, side):
    """Continuation-plane contribution, already divided by CORR_NORM.

    Applied only on the main-search path: qsearch has no `played` and giving
    it one would add an array argument to the function that runs at every
    leaf."""
    if not USE_CORR_CONT:
        return int64(0)
    idx = _cont_corr_index(played, ply)
    if idx < 0:
        return int64(0)
    return c_div(CORR_WEIGHT_CONT * int64(corr[4, idx, side]), CORR_NORM)


@njit(cache=False, fastmath=True, error_model='numpy')
def _corrected_eval(bb, st, corr, ev):
    """Static evaluation nudged by what past searches at this pawn structure
    and piece configuration actually returned.

    Disabled below the specialised-endgame threshold, and that guard is not
    optional: there `evaluate()` returns a mating drive score rather than a
    material-scale value, so correction history learns an "error" against a
    quantity that is not an evaluation and then corrupts the mating gradient.
    Without it KR vs K stops converting - threefold after 9 moves. This is the
    same hazard that delta pruning hit, and the general rule is that **anything
    which adds a margin to, or learns from, the static evaluation must be off
    where the evaluation is not denominated in material**.

    Separate colour indices rather than one function taking a colour: numba
    specialises on integer literals, so a colour argument would compile two
    copies of everything below it."""
    if not USE_CORR_HIST or count_bits(bb[OCC_A]) <= SPECIALISED_MAX_PIECES:
        return ev
    side = int64(st[SIDE])
    pawn = int64(corr[0, _pawn_corr_index(bb), side])
    white = int64(corr[1, _white_corr_index(bb), side])
    black = int64(corr[2, _black_corr_index(bb), side])
    total = CORR_WEIGHT_PAWN * pawn + CORR_WEIGHT_NONPAWN * (white + black)
    if USE_CORR_MINOR:
        total += CORR_WEIGHT_MINOR * int64(corr[3, _minor_corr_index(bb), side])
    value = ev + c_div(total, CORR_NORM)
    if value > MATE_SCORE:
        return MATE_SCORE
    if value < -MATE_SCORE:
        return -MATE_SCORE
    return value


@njit(cache=False, fastmath=True, error_model='numpy')
def _corr_gravity(old, bonus):
    """History gravity: the closer an entry is to the bound, the less a new
    observation moves it, so one outlier cannot dominate the evaluation."""
    if bonus > CORR_LIMIT:
        bonus = CORR_LIMIT
    elif bonus < -CORR_LIMIT:
        bonus = -CORR_LIMIT
    return old + bonus - c_div(old * abs(bonus), CORR_LIMIT)


@njit(cache=False, fastmath=True, error_model='numpy')
def _update_correction(bb, st, corr, corrected, best_score, depth,
                       has_best_move, sc, played, ply):
    """Fold this node's static-eval error into the correction tables. The
    bonus is the signed gap between what the search found and what the
    corrected static evaluation predicted, scaled by depth."""
    if CORR_DIAG:
        gap = best_score - corrected
        sc[SC_CORR_ERR] += gap if gap >= 0 else -gap
        sc[SC_CORR_N] += 1
    if not USE_CORR_HIST or count_bits(bb[OCC_A]) <= SPECIALISED_MAX_PIECES:
        return
    weight = 16 if has_best_move else 24
    bonus = c_div((best_score - corrected) * depth * weight, 128)
    cap = CORR_LIMIT // 4
    if bonus > cap:
        bonus = cap
    elif bonus < -cap:
        bonus = -cap
    side = int64(st[SIDE])
    pawn_i = _pawn_corr_index(bb)
    corr[0, pawn_i, side] = _corr_gravity(int64(corr[0, pawn_i, side]), bonus)
    white_i = _white_corr_index(bb)
    corr[1, white_i, side] = _corr_gravity(int64(corr[1, white_i, side]), bonus)
    black_i = _black_corr_index(bb)
    corr[2, black_i, side] = _corr_gravity(int64(corr[2, black_i, side]), bonus)
    if USE_CORR_MINOR:
        minor_i = _minor_corr_index(bb)
        corr[3, minor_i, side] = _corr_gravity(
            int64(corr[3, minor_i, side]), bonus)
    if USE_CORR_CONT:
        cont_i = _cont_corr_index(played, ply)
        if cont_i >= 0:
            corr[4, cont_i, side] = _corr_gravity(
                int64(corr[4, cont_i, side]), bonus)


@njit(cache=False, fastmath=True, error_model='numpy')
def _captured_piece(bb, st, mv):
    if mv & (1 << 22):
        return p if st[SIDE] == WHITE else P
    tgt = (mv & 0xFC0) >> 6
    start = p if st[SIDE] == WHITE else P
    tgt_bit = ONE << uint64(tgt)
    if USE_CAP_UNROLL:
        if bb[start] & tgt_bit:
            return start
        if bb[start + 1] & tgt_bit:
            return start + 1
        if bb[start + 2] & tgt_bit:
            return start + 2
        if bb[start + 3] & tgt_bit:
            return start + 3
        if bb[start + 4] & tgt_bit:
            return start + 4
        if bb[start + 5] & tgt_bit:
            return start + 5
        return start
    for piece in range(start, start + 6):
        if bb[piece] & tgt_bit:
            return piece
    return start


@njit(cache=False, fastmath=True, error_model='numpy')
def _capture_score(bb, st, mv, cap_hist):
    """Good captures (SEE >= 0) sort above quiets, bad captures below them."""
    if USE_EVASION_ORDER and not (mv & (1 << 20)):
        # Reached only from _sort_captures on an evasion list, where
        # generate_moves produced quiets too. Scoring a quiet as a phantom
        # pawn capture is noise, and the see_ge it pays for is the most
        # frequently executed SEE in the engine. A quiet promotion is still
        # worth trying early; everything else sorts between the good and bad
        # capture bands. The descending-order break in quiescence is disabled
        # while in check, so nothing depends on these staying above
        # SCORE_BAD_CAPTURE_LIMIT.
        if (mv & 0xF0000) >> 16:
            return SCORE_GOOD_CAPTURE
        return int64(0)
    captured = _captured_piece(bb, st, mv)
    attacker = (mv & 0xF000) >> 12
    tgt = (mv & 0xFC0) >> 6
    hist = int64(cap_hist[attacker, tgt, captured])
    if USE_CAP_SCORE_V2:
        base = CAP_VALUE[captured] * CAP_VALUE_MULT + hist
    else:
        base = MVV_LVA[attacker, captured] * 100 + hist
    if not USE_SEE:
        return SCORE_GOOD_CAPTURE + base
    threshold = int64(0)
    if USE_CAP_SEE_DYN:
        # The threshold must divide the whole ordering score, not the history
        # alone: `see_ge(m, -score / 18)` with `score = 7 * value(captured)
        # + capture history`. Using the history by itself lets a negative
        # history push the threshold POSITIVE, and see_ge then short-circuits
        # every promotion and en-passant capture to "bad" and rejects a free
        # pawn before running the exchange loop. Here the good/bad bit gates
        # the quiescence break and _see_prunable, so both drop the move
        # outright - a soundness bug rather than an ordering wobble.
        #
        # Clamped at zero as well, so no tuning of CAP_VALUE_MULT or
        # CAP_SEE_DIV can bring the positive branch back.
        threshold = -c_div(CAP_VALUE[captured] * CAP_VALUE_MULT + hist,
                           CAP_SEE_DIV)
        if threshold > 0:
            threshold = int64(0)
    if see_ge(bb, st, mv, threshold):
        return SCORE_GOOD_CAPTURE + base
    return SCORE_BAD_CAPTURE + base


@njit(cache=False, fastmath=True, error_model='numpy')
def _cont_hist_score(cont_hist, played, ply, mv):
    """Continuation history for a quiet move, over CONT_PLANES planes."""
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
    if CONT_PLANES >= 3 and ply >= 3 and played[ply - 2] != 0:
        p3 = played[ply - 2]
        total += int64(cont_hist[2, (p3 & 0xF000) >> 12, (p3 & 0xFC0) >> 6,
                                 piece, tgt])
    if CONT_PLANES >= 4 and ply >= 4 and played[ply - 3] != 0:
        p4 = played[ply - 3]
        total += int64(cont_hist[3, (p4 & 0xF000) >> 12, (p4 & 0xFC0) >> 6,
                                 piece, tgt])
    return total


@njit(cache=False, fastmath=True, error_model='numpy')
def _counter_move(counters, played, ply):
    if ply < 1 or played[ply] == 0:
        return 0
    prev = played[ply]
    return int64(counters[(prev & 0xF000) >> 12, (prev & 0xFC0) >> 6])


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
def _sort_moves(bb, st, ml, scores, cnt, ply, tt_move, killers, main_hist,
                cap_hist, cont_hist, counters, played):
    """Score and order a move list.

    **The per-move body is inlined and its loop-invariants hoisted, and that is
    worth +48% NPS with byte-identical node counts** (355,858 nodes either way
    at depth 9; 241,794 -> 358,040 NPS). It reads worse than calling
    `_score_move` per move, which is why it carries this comment.

    The cause is a numba codegen cliff, not cache or branch prediction. A
    *guarded* history lookup inside a per-move helper - `if ply >= 1 and
    played[ply] != 0:` wrapped around a read of `counters` or `cont_hist` -
    measured **25.7 ns** per quiet move against 0.7 ns for the same read
    unguarded. Two of those per move made `_score_move` cost more than an
    entire average node: 4,586 ns at a 37-move node, against 967 ns here.
    `inline='always'` on the helpers does not fix it, and neither does
    NUMBA_SLP_VECTORIZE.

    Every hoisted quantity - the killers, the counter move, and the two
    continuation-history rows - is invariant across the move list, so this is
    pure redundant work removal and cannot change what the search explores.
    The node count is the test: it must not move.

    **The general rule: never put a conditional table lookup inside a helper
    called once per move.** Hoist the guard and the row view out of the loop."""
    k0 = killers[0, ply]
    k1 = killers[1, ply]
    counter = int64(0)
    ch0 = cont_hist[0, 0, 0]
    ch1 = cont_hist[1, 0, 0]
    use0 = False
    use1 = False
    if ply >= 1 and played[ply] != 0:
        prev = played[ply]
        pp = (prev & 0xF000) >> 12
        pt = (prev & 0xFC0) >> 6
        counter = int64(counters[pp, pt])
        if USE_CONT_HIST:
            ch0 = cont_hist[0, pp, pt]
            use0 = True
    if USE_CONT_HIST and ply >= 2 and played[ply - 1] != 0:
        prev2 = played[ply - 1]
        ch1 = cont_hist[1, (prev2 & 0xF000) >> 12, (prev2 & 0xFC0) >> 6]
        use1 = True
    # Placeholder rows first: with CONT_PLANES == 2 these planes do not
    # exist, and numba indexes without bounds checking.
    ch2 = ch0
    ch3 = ch0
    use2 = False
    use3 = False
    if CONT_PLANES >= 3 and USE_CONT_HIST and ply >= 3 and played[ply - 2] != 0:
        p3 = played[ply - 2]
        ch2 = cont_hist[2, (p3 & 0xF000) >> 12, (p3 & 0xFC0) >> 6]
        use2 = True
    if CONT_PLANES >= 4 and USE_CONT_HIST and ply >= 4 and played[ply - 3] != 0:
        p4 = played[ply - 3]
        ch3 = cont_hist[3, (p4 & 0xF000) >> 12, (p4 & 0xFC0) >> 6]
        use3 = True
    for i in range(cnt):
        mv = ml[i]
        if tt_move != 0 and mv == tt_move:
            scores[i] = SCORE_TT_BEST
        elif mv & (1 << 20):
            scores[i] = _capture_score(bb, st, mv, cap_hist)
        elif k0 == mv:
            scores[i] = SCORE_KILLER_1
        elif k1 == mv:
            scores[i] = SCORE_KILLER_2
        elif counter == mv:
            scores[i] = SCORE_COUNTER
        else:
            piece = (mv & 0xF000) >> 12
            tgt = (mv & 0xFC0) >> 6
            total = int64(main_hist[piece, tgt])
            if use0:
                total += int64(ch0[piece, tgt])
            if use1:
                total += int64(ch1[piece, tgt])
            if CONT_PLANES >= 3 and use2:
                total += int64(ch2[piece, tgt])
            if CONT_PLANES >= 4 and use3:
                total += int64(ch3[piece, tgt])
            scores[i] = total
    _insertion_sort(ml, scores, cnt)


@njit(cache=False, fastmath=True, error_model='numpy')
def _sort_captures(bb, st, ml, scores, cnt, cap_hist):
    for i in range(cnt):
        scores[i] = _capture_score(bb, st, ml[i], cap_hist)
    _insertion_sort(ml, scores, cnt)


@njit(cache=False, fastmath=True, error_model='numpy')
def _check_time(sc, fc):
    if (sc[SC_NODES] & 2047) == 0:
        with objmode(now="f8"):
            now = time.perf_counter()
        if now >= fc[0]:
            sc[SC_STOP] = 1


@njit(cache=False, fastmath=True, error_model='numpy')
def _in_check(bb, st):
    side = st[SIDE]
    king_sq = lsb(bb[K]) if side == WHITE else lsb(bb[k])
    return is_under_attack(bb, king_sq, side ^ 1)


@njit(cache=False, fastmath=True, error_model='numpy')
def _has_legal_move(bb, st, undo_bb, undo_st, mls, ply):
    """True if the side to move has any legal move.

    Only ever called when the fifty-move counter has reached 100, which is rare
    enough that generating and trying moves costs nothing measurable. `mls[ply]`
    is safe to use: the node regenerates into it before its own move loop."""
    cnt = generate_moves(bb, st, mls[ply])
    for i in range(cnt):
        if make_move(bb, st, undo_bb, undo_st, ply, mls[ply, i]) != 0:
            unmake(bb, st, undo_bb, undo_st, ply)
            return True
    return False


@njit(cache=False, fastmath=True, error_model='numpy')
def _is_repetition(bb, rep, rep_idx, rep_base, null_floor):
    """Draw by repetition, with the two boundaries the naive scan ignores.

    **The null-move boundary.** `_make_null` writes the pre-null hash into
    `rep[rep_idx]` and recurses one deeper, so a scan that starts at 0 compares
    positions inside a null subtree against real game history. After a null the
    side to move has flipped with no move played, so such a match is not a
    repetition - no legal sequence produces it - and treating it as one returns
    a false 0 from a null-window search that is trying to prove `score >= beta`.
    When beta is at or below zero that false draw becomes a false fail-high.
    The scan therefore starts at `null_floor`, the ply after the most recent
    null move.

    **The history boundary.** `rep[0:rep_base]` is the game before the root and
    `rep[rep_base:rep_idx]` is the current search tree. A match inside the tree
    is a draw - the standard, sound heuristic, since a position repeatable once
    inside the tree can normally be repeated again. A match against history is
    only the *second* occurrence, and a claim needs three, so it is not a draw.
    Counting history matches distinguishes them: two there plus this one is the
    third and is a draw.

    Counting rather than precomputing a per-history flag array is deliberate.
    An extra array argument to negamax costs NRT reference-count traffic on
    every call, which is the single largest speed lever in this engine."""
    if not USE_DRAW_FIX:
        key = bb[HASH]
        for i in range(rep_idx):
            if rep[i] == key:
                return 1
        return 0
    key = bb[HASH]
    history_hits = 0
    for i in range(null_floor, rep_idx):
        if rep[i] == key:
            # Inside the search tree, or the whole-game rule is off: draw now.
            if i >= rep_base or not USE_DRAW_HIST:
                return 1
            history_hits += 1
    return 1 if history_hits >= 2 else 0


@njit(cache=False, fastmath=True, error_model='numpy')
def _delta_prunable(bb, st, mv, ev, alpha):
    """Delta pruning. Winning the captured piece outright, plus a margin for
    the positional swing a capture can carry, still falls short of alpha, so
    the capture cannot matter. Neither BTC nor the port had this; quiescence
    is most of the tree, so it is the cheapest place to remove nodes.

    Not applied when the side to move is in check, because quiescence here
    does not generate evasions - it stand-pats instead, and pruning on top of
    that would compound the existing approximation."""
    gain = MATERIAL_MG[_captured_piece(bb, st, mv) % 6]
    if (mv & 0xF0000) >> 16:
        gain += MATERIAL_MG[4] - MATERIAL_MG[0]
    return ev + gain + QS_DELTA_MARGIN <= alpha


@njit(cache=False, fastmath=True, error_model='numpy')
def qsearch(acc, alpha, beta, bb, st, undo_bb, undo_st, mls, scores, cap_hist,
            corr, sc, fc, ply, tt_key, tt_data):
    _check_time(sc, fc)
    sc[SC_NODES] += 1
    if ply > MAX_SEARCH_PLY - 1:
        return _corrected_eval(bb, st, corr, evaluate_cached(bb, st, acc[ply]))

    # Probe before evaluating: a cutoff here saves the evaluation as well as
    # the subtree. Quiescence stores at depth 0, so these entries can never
    # satisfy a main-search node, which always probes with depth >= 1.
    if USE_QTT:
        hit, _, _ = tt_probe(bb, alpha, beta, 0, ply, tt_key, tt_data,
                             st[FIFTY])
        if hit != NO_HASH_ENTRY:
            return hit

    original_alpha = alpha
    ev = _corrected_eval(bb, st, corr, evaluate_cached(bb, st, acc[ply]))
    # Fail-soft, but only where the evaluation is denominated in material.
    #
    # Below SPECIALISED_MAX_PIECES the evaluation is a mating drive clamped to
    # MATE_SCORE - 1 (btc_endgame.py:127), so a great many distinct positions
    # score identically. Fail-hard returned the window edge and the search
    # never saw that; propagating the true stand-pat instead hands the search a
    # flat score across the whole conversion and it shuffles. Measured: full
    # fail-soft here takes 320,684 nodes but loses KBN vs K to a threefold at
    # move 14, where the gated version keeps the node win and converts.
    #
    # This is the same unit-scale rule delta pruning and correction history
    # already follow, and the third bug that rule has caught.
    soft = count_bits(bb[OCC_A]) > SPECIALISED_MAX_PIECES

    # In check there is no stand pat. Standing pat asserts that passing is a
    # lower bound on the position, which is not legal reasoning when the king
    # is attacked, and captures are not the only reply that matters. The score
    # starts at -INFINITY and every evasion is searched.
    in_check = USE_QS_EVASION and _in_check(bb, st)
    # Fifty-move draw. Unreachable while quiescence generated captures only -
    # a capture resets the counter - but an evasion is a quiet move, so with
    # USE_QS_EVASION the count can cross 100 here. Checkmate takes precedence
    # and is settled below, once evasions have been generated.
    if USE_DRAW_FIX and st[FIFTY] >= 100 and not in_check:
        return _draw_score(bb)
    if in_check:
        best_value = -INFINITY
        cnt = generate_moves(bb, st, mls[ply])
    else:
        best_value = ev
        if ev >= beta:
            out = ev if soft else beta
            if USE_QTT:
                tt_record(bb, out, int64(0), int64(HASH_BETA), 0, ply, tt_key,
                          tt_data, int64(sc[SC_TT_GEN]), int64(0))
            return out
        if ev > alpha:
            alpha = ev
        cnt = generate_captures(bb, st, mls[ply])
    _sort_captures(bb, st, mls[ply], scores[ply], cnt, cap_hist)

    # Delta pruning compares an evaluation against a material gain, so it is
    # only valid while the evaluation is denominated in material. Below the
    # specialised-endgame threshold it is not: probe() returns a mating drive
    # score, and insufficient_material() returns a flat 0. Pruning there cost
    # KBN vs K outright - the search stopped seeing that capturing the last
    # minor turns a lost position into a drawn one, because material value
    # does not express a change of endgame class.
    can_delta = USE_QS_DELTA and not in_check and abs(alpha) < MATE_SCORE \
        and count_bits(bb[OCC_A]) > SPECIALISED_MAX_PIECES
    qrow = mls[ply]
    qsrow = scores[ply]
    played = 0
    for i in range(cnt):
        mv = qrow[i] if USE_ROWVIEW else mls[ply, i]
        # Losing captures cannot raise the stand-pat. The verdict is already
        # in the score: _capture_score ran see_ge during the sort to separate
        # good captures from bad. Calling see_ge again here paid for the most
        # frequently executed SEE in the engine twice. Good captures all score
        # above +500k and bad ones below -500k, so once the descending list
        # reaches a bad one every remaining move is bad too.
        qs_score = qsrow[i] if USE_ROWVIEW else scores[ply, i]
        # The descending-order break is a capture-list property. An evasion
        # list is not sorted that way and a losing-looking evasion may be the
        # only legal move, so in check every move is searched.
        if USE_SEE and not in_check and qs_score < SCORE_BAD_CAPTURE_LIMIT:
            break
        if can_delta and _delta_prunable(bb, st, mv, ev, alpha):
            continue
        if make_move(bb, st, undo_bb, undo_st, ply, mv) == 0:
            continue
        played += 1
        if USE_NNUE:
            acc_update(acc, ply, ply + 1, undo_bb[ply], bb, NET_FT_W,
                       NET_FT_B, NET_L1, NET_TABLE, NET_BUCKETS)
        score = -qsearch(acc, -beta, -alpha, bb, st, undo_bb, undo_st, mls,
                         scores, cap_hist, corr, sc, fc, ply + 1, tt_key,
                         tt_data)
        unmake(bb, st, undo_bb, undo_st, ply)
        if sc[SC_STOP]:
            return 0
        if score > best_value:
            best_value = score
        if score > alpha:
            alpha = score
            if score >= beta:
                out = score if soft else beta
                if USE_QTT:
                    tt_record(bb, out, int64(0), int64(HASH_BETA), mv, ply,
                              tt_key, tt_data, int64(sc[SC_TT_GEN]), int64(0))
                return out
    # No legal evasion existed, so the side to move is mated here. Without
    # this the horizon reports a stand-pat score for a checkmate.
    if in_check and played == 0:
        return -MATE_VALUE + ply
    if USE_DRAW_FIX and st[FIFTY] >= 100:
        return _draw_score(bb)
    final = best_value if soft else alpha
    if USE_QTT:
        flag = int64(HASH_EXACT) if alpha > original_alpha else int64(HASH_ALPHA)
        tt_record(bb, final, int64(0), int64(flag), 0, ply, tt_key,
                  tt_data, int64(sc[SC_TT_GEN]), int64(0))
    return final


@njit(cache=False, fastmath=True, error_model='numpy')
def _has_non_pawn_material(bb, st):
    """Zugzwang guard for the null move. With only king and pawns, "pass" is
    not a lower bound on the best move: the whole point of a zugzwang is that
    every real move is worse than passing, so a null search fails high and
    prunes a lost pawn ending as if it were winning. Neither BTC nor the port
    had this guard; king-and-pawn endings are exactly where it costs games."""
    if st[SIDE] == WHITE:
        return (bb[N] | bb[B] | bb[R] | bb[Q]) != ZERO
    return (bb[N + 6] | bb[B + 6] | bb[R + 6] | bb[Q + 6]) != ZERO


@njit(cache=False, fastmath=True, error_model='numpy')
def _null_reduction(depth, ev, beta):
    """Adaptive null-move reduction. BTC uses a flat R=2, which is very
    conservative by current practice: the deeper the remaining search and the
    further the static evaluation already sits above beta, the more confident
    the null-move cutoff is and the harder it can be reduced."""
    if not USE_NULL_ADAPTIVE:
        reduction = NULL_REDUCTION
    else:
        reduction = NULL_BASE_R + c_div(depth, NULL_DEPTH_DIV)
        gain = c_div(ev - beta, NULL_EVAL_DIV)
        if gain > NULL_EVAL_MAX:
            gain = NULL_EVAL_MAX
        if gain > 0:
            reduction += gain
    if USE_NULL_CD_FLOOR and reduction > depth - 2:
        # Leave the child `depth - 1 - reduction >= 1`. Past that point the
        # null search is a quiescence stand-pat of the position after
        # passing, which is a margin-free version of exactly what
        # RFP_MAX_DEPTH exists to cap - and it runs several plies deeper.
        reduction = depth - 2
        if reduction < 1:
            reduction = 1
    return reduction


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
def _unmake_null(bb, st, saved_ep, saved_hash):
    st[SIDE] ^= 1
    st[EP] = saved_ep
    bb[HASH] = saved_hash


@njit(cache=False, fastmath=True, error_model='numpy')
def _update_cont_hist(cont_hist, played, ply, mv, bonus):
    """Full bonus at 1 ply, three quarters at 2, half beyond."""
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
    if CONT_PLANES >= 3 and ply >= 3 and played[ply - 2] != 0:
        p3 = played[ply - 2]
        _update_cont_entry(cont_hist, 2, (p3 & 0xF000) >> 12,
                           (p3 & 0xFC0) >> 6, piece, tgt,
                           c_div(bonus, 2))
    if CONT_PLANES >= 4 and ply >= 4 and played[ply - 3] != 0:
        p4 = played[ply - 3]
        _update_cont_entry(cont_hist, 3, (p4 & 0xF000) >> 12,
                           (p4 & 0xFC0) >> 6, piece, tgt,
                           c_div(bonus, 2))


@njit(cache=False, fastmath=True, error_model='numpy')
def _update_cont_entry(cont_hist, table, prev_piece, prev_tgt, piece, tgt, bonus):
    if bonus > HIST_MAX:
        bonus = HIST_MAX
    if bonus < -HIST_MAX:
        bonus = -HIST_MAX
    e = int64(cont_hist[table, prev_piece, prev_tgt, piece, tgt])
    e += bonus - c_div(e * abs(bonus), HIST_MAX)
    cont_hist[table, prev_piece, prev_tgt, piece, tgt] = e


@njit(cache=False, fastmath=True, error_model='numpy')
def _update_capture_history(bb, st, mv, cap_hist, captures, capture_cnt, bonus):
    captured = _captured_piece(bb, st, mv)
    _update_cap_entry(cap_hist, (mv & 0xF000) >> 12, (mv & 0xFC0) >> 6,
                      captured, bonus)
    for i in range(capture_cnt - 1):
        bad = captures[i]
        bad_cap = _captured_piece(bb, st, bad)
        _update_cap_entry(cap_hist, (bad & 0xF000) >> 12, (bad & 0xFC0) >> 6,
                          bad_cap, -bonus)


@njit(cache=False, fastmath=True, error_model='numpy')
def _update_cap_entry(cap_hist, piece, tgt, captured, bonus):
    if bonus > HIST_MAX:
        bonus = HIST_MAX
    if bonus < -HIST_MAX:
        bonus = -HIST_MAX
    e = int64(cap_hist[piece, tgt, captured])
    e += bonus - c_div(e * abs(bonus), HIST_MAX)
    cap_hist[piece, tgt, captured] = e


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
def _is_shuffling(mv, st, played, ply, plies_from_null):
    """True when this move walks a piece back and forth.

    `mv` returns to the square the move two plies ago left, and that move
    returned to the square the move four plies ago left. played[ply] is
    the move that reached this node, so those are played[ply - 1] and
    played[ply - 3]."""
    if not USE_SHUFFLE_GUARD:
        return False
    if mv & (1 << 20):
        return False
    if st[FIFTY] < SHUFFLE_MIN_FIFTY:
        return False
    if plies_from_null < SHUFFLE_MIN_FROM_NULL or ply < SHUFFLE_MIN_PLY:
        return False
    prev2 = played[ply - 1]
    prev4 = played[ply - 3]
    if prev2 == 0 or prev4 == 0:
        return False
    return (mv & 0x3F) == ((prev2 & 0xFC0) >> 6) \
        and (prev2 & 0x3F) == ((prev4 & 0xFC0) >> 6)


@njit(cache=False, fastmath=True, error_model='numpy')
def _draw_score(bb):
    """A draw is worth zero, but scoring every draw as the same zero
    leaves the search unable to order two equally drawn lines. One point
    either way breaks the tie and averages to zero over nodes, which is
    what makes this a tie-break and not a contempt term."""
    if not USE_DRAW_JITTER:
        return int64(0)
    # Bit 1 of the position key: uniform, uncorrelated with ply parity -
    # a ply-correlated jitter would be contempt, not a tie-break - and
    # stable per position, so a reduction and its re-search agree.
    return int64(-1) + int64(bb[HASH] & uint64(2))


@njit(cache=False, fastmath=True, error_model='numpy')
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


@njit(cache=False, fastmath=True, error_model='numpy')
def _razor(acc, alpha, beta, depth, ev, bb, st, undo_bb, undo_st, mls, scores,
           cap_hist, corr, sc, fc, ply, tt_key, tt_data):
    """Razoring. Returns (should_return, value)."""
    score = ev + 192
    if score >= beta:
        return False, 0
    if depth == 1:
        new_score = qsearch(acc, alpha, beta, bb, st, undo_bb, undo_st, mls,
                            scores, cap_hist, corr, sc, fc, ply, tt_key,
                            tt_data)
        return True, new_score if new_score > score else score
    score += 269
    if score < beta and depth <= 2:
        new_score = qsearch(acc, alpha, beta, bb, st, undo_bb, undo_st, mls,
                            scores, cap_hist, corr, sc, fc, ply, tt_key,
                            tt_data)
        if new_score < beta:
            return True, new_score if new_score > score else score
    return False, 0


@njit(cache=False, fastmath=True, error_model='numpy')
def _prologue_head(acc, alpha, beta, depth, ply, rep_idx, pv_node, bb, st, rep,
                   tt_key, tt_data, sc, fc, excluded, null_floor):
    """Draw rules, mate distance pruning, TT probe and leaf dispatch.
    Returns (code, value, alpha, beta, tt_move).

    During a singular verification (excluded != 0) the TT cutoff is skipped:
    the stored score was produced by a search that included the very move we
    are now forbidding."""
    if USE_LMR_TTPV:
        # Cleared up front so the early exits below cannot leave a sibling's
        # bit visible to the caller.
        sc[SC_TT_PV] = 0
    if ply and _is_repetition(bb, rep, rep_idx, int64(sc[SC_REP_BASE]),
                              null_floor):
        return NODE_RETURN, _draw_score(bb), alpha, beta, 0
    # With the fix on, the fifty-move test moves to _node_prologue, where a
    # move list exists to tell a draw from a checkmate delivered on the
    # hundredth half-move. Depth-0 nodes go to quiescence, which carries its
    # own test.
    if ply and not USE_DRAW_FIX and st[FIFTY] >= 100:
        return NODE_RETURN, _draw_score(bb), alpha, beta, 0

    if USE_MDP and ply:
        if alpha < -MATE_VALUE + ply:
            alpha = -MATE_VALUE + ply
        if beta > MATE_VALUE - ply - 1:
            beta = MATE_VALUE - ply - 1
        if alpha >= beta:
            return NODE_RETURN, alpha, alpha, beta, 0

    score, tt_move, tt_pv_bit = tt_probe(bb, alpha, beta, depth, ply,
                                         tt_key, tt_data, st[FIFTY])
    if USE_LMR_TTPV:
        sc[SC_TT_PV] = tt_pv_bit
    if excluded == 0 and ply and not pv_node and score != NO_HASH_ENTRY:
        return NODE_RETURN, score, alpha, beta, tt_move

    _check_time(sc, fc)

    if depth == 0 or (USE_DEPTH_CLAMP and depth < 0
                      and count_bits(bb[OCC_A]) > SPECIALISED_MAX_PIECES):
        # The null-move reduction can exceed the depth remaining, so this is
        # reached with depth < 0. Searching such a node full width also hands
        # tt_record a negative depth, which packs into the depth field as 127
        # and makes the slot permanently unreplaceable.
        #
        # Not below the specialised-endgame threshold, though. There the
        # evaluation is a mating drive clamped to MATE_SCORE - 1, so many
        # distinct positions score identically and a quiescence stand-pat is
        # flat across the whole conversion - the same hazard that fail-soft
        # quiescence and delta pruning hit. Measured: clamping there loses
        # KBN vs K to a threefold after 19 moves, against mate in 20 without.
        return NODE_QSEARCH, 0, alpha, beta, tt_move
    if ply > MAX_SEARCH_PLY - 1:
        return NODE_RETURN, evaluate_cached(bb, st, acc[ply]), alpha, beta,             tt_move
    return NODE_CONTINUE, 0, alpha, beta, tt_move


@njit(cache=False, fastmath=True, error_model='numpy')
def _node_prologue(acc, alpha, beta, depth, ply, rep_idx, pv_node, bb, st, undo_bb,
                   undo_st, mls, scores, cap_hist, corr, rep, static_evals,
                   tt_key, tt_data, sc, fc, excluded, null_floor, played):
    """Every non-recursive early exit of negamax, in BTC's order. Returns
    (code, value, alpha, beta, depth, tt_move, in_check, ev, improving).

    A singular verification (excluded != 0) must reach the move loop, so RFP,
    razoring and IIR are all skipped in that case."""
    code, value, alpha, beta, tt_move = _prologue_head(acc, 
        alpha, beta, depth, ply, rep_idx, pv_node, bb, st, rep, tt_key,
        tt_data, sc, fc, excluded, null_floor)
    if code != NODE_CONTINUE:
        return code, value, alpha, beta, depth, tt_move, 0, 0, 0, True

    sc[SC_NODES] += 1

    in_check = _in_check(bb, st)

    # Checkmate on the hundredth half-move is mate, not a draw: the game ends
    # before a fifty-move claim can be made. Everything else at 100 is a draw.
    if USE_DRAW_FIX and ply and st[FIFTY] >= 100:
        if not in_check or _has_legal_move(bb, st, undo_bb, undo_st, mls, ply):
            return (NODE_RETURN, _draw_score(bb), alpha, beta, depth,
                    tt_move, 0, 0, 0, True)

    if in_check:
        depth += 1

    # In check the static evaluation is never read: _improving stores
    # EVAL_NONE, RFP and razoring are gated on quiet_node, and _skip_quiet
    # returns early. Evaluating anyway spent a full evaluation - a third of
    # runtime - on every check node.
    if in_check:
        raw = int64(0)
        ev = int64(0)
    else:
        raw = evaluate_cached(bb, st, acc[ply])
        ev = _corrected_eval(bb, st, corr, raw)
    if USE_CORR_HIST and USE_CORR_CONT and not in_check \
            and count_bits(bb[OCC_A]) > SPECIALISED_MAX_PIECES:
        # Same gate as _corrected_eval: below the threshold the evaluation is
        # a mating drive, and correction learns against a quantity that is not
        # an evaluation.
        ev += _cont_corr_delta(corr, played, ply, int64(st[SIDE]))
        if ev > MATE_SCORE:
            ev = MATE_SCORE
        elif ev < -MATE_SCORE:
            ev = -MATE_SCORE
    if USE_CORR_MARGIN:
        sc[SC_CORR_ADJ] = int64(0)
    if USE_CORR_MARGIN and not in_check:
        # How far past searches moved this evaluation. A large value means
        # the raw evaluation has been wrong here before.
        gap = ev - raw
        sc[SC_CORR_ADJ] = gap if gap >= 0 else -gap
    improving = _improving(static_evals, ev, ply, in_check)

    # RFP, razoring and futility all compare the evaluation against a margin in
    # centipawns, which is only meaningful while the evaluation is denominated
    # in material. Below the specialised-endgame threshold it is not: probe()
    # returns a mating drive score and insufficient_material() returns a flat 0.
    # This is the same hazard that broke KBN vs K with delta pruning and KR vs K
    # with correction history; these three inherit it from BTC and were the last
    # unguarded cases.
    material_scale = count_bits(bb[OCC_A]) > SPECIALISED_MAX_PIECES
    quiet_node = not pv_node and not in_check and \
        (material_scale or not USE_EVAL_SCALE_GUARD)

    if USE_RFP and excluded == 0 and depth < RFP_MAX_DEPTH and quiet_node \
            and abs(beta) < MATE_SCORE:
        margin = RFP_MARGIN * (depth - improving)
        if ev - margin >= beta:
            return (NODE_RETURN, ev - margin, alpha, beta, depth, tt_move,
                    in_check, ev, improving, material_scale)

    # depth <= 2, not 3: at depth 3 every path through _razor returns
    # (False, 0), so the call is a guaranteed no-op
    if USE_RAZOR and excluded == 0 and quiet_node and depth <= 2:
        done, value = _razor(acc, alpha, beta, depth, ev, bb, st, undo_bb, undo_st,
                             mls, scores, cap_hist, corr, sc, fc, ply,
                             tt_key, tt_data)
        if done:
            return (NODE_RETURN, value, alpha, beta, depth, tt_move,
                    in_check, ev, improving, material_scale)

    return (NODE_CONTINUE, 0, alpha, beta, depth, tt_move, in_check, ev,
            improving, material_scale)


@njit(cache=False, fastmath=True, error_model='numpy')
def _see_prunable(move_score, mv, depth, moves_searched, pv_node, in_check,
                  alpha, tt_move, excluded):
    """Prune losing captures at shallow depth.

    Uses the SEE that move ordering already computed: _capture_score calls
    see_ge once per capture to sort good above quiets and bad below them, and
    that verdict is recoverable from the score. Calling see_ge again here was
    exactly self-defeating - it cut nodes by 7.2% and cost the same in time,
    because our SEE is a swap-off loop in numba rather than a few C
    instructions. Reading the score instead makes the test free.
    """
    if not USE_SEE_PRUNE or excluded != 0 or moves_searched == 0:
        return False
    if pv_node or in_check or mv == tt_move:
        return False
    if (mv & 0xF0000) >> 16:            # never prune a promotion
        return False
    if depth > SEE_PRUNE_MAX_DEPTH or abs(alpha) >= MATE_SCORE:
        return False
    return move_score < SCORE_BAD_CAPTURE_LIMIT


@njit(cache=False, fastmath=True, error_model='numpy')
def _skip_quiet(move_score, mv, depth, moves_searched, improving, ev, alpha,
                pv_node, in_check, opp_in_check, material_scale,
                opp_worse, corr_adj):
    """Late move pruning, frontier futility and history pruning on quiets.

    History pruning is free here for the same reason SEE pruning is: for a
    quiet move the ordering score already *is* main history plus continuation
    history, computed once during the sort. Killers and counter moves score
    far above any real history value, so this test never reaches them."""
    if moves_searched == 0 or pv_node or in_check or opp_in_check:
        return False
    if (mv & (1 << 20)) or ((mv & 0xF0000) >> 16):
        return False
    if abs(alpha) >= MATE_SCORE:
        return False
    if USE_LMP and depth <= LMP_MAX_DEPTH \
            and moves_searched >= c_div(LMP_BASE + depth * depth, 2 - improving):
        return True
    # futility compares ev against a centipawn margin, so it needs the
    # evaluation to be on the material scale; LMP above counts moves and does
    # not, so it stays unguarded
    if USE_FUTILITY and material_scale and depth <= FUTILITY_MAX_DEPTH:
        margin = FUTILITY_MARGIN * depth
        if USE_CORR_MARGIN:
            # The evaluation has needed correcting here, so trust it less
            # and prune less: a bigger margin is harder to satisfy.
            margin += c_div(corr_adj, CORR_MARGIN_DIV)
        if USE_OPP_WORSENING and opp_worse:
            # The opponent's evaluation is falling too, so this position
            # is less likely to be saved by something unsearched.
            margin -= OPP_WORSENING_MARGIN
        if ev + margin <= alpha:
            return True
    if USE_HIST_PRUNE and depth <= HIST_PRUNE_MAX_DEPTH \
            and move_score < -HIST_PRUNE_MARGIN * depth:
        return True
    return False


@njit(cache=False, fastmath=True, error_model='numpy')
def _lmr_reduction(mv, depth, moves_searched, improving, in_check,
                   opp_in_check, pv_node, cut_node, main_hist, cont_hist,
                   played, ply, corr_adj, tt_pv, tt_capture, material_scale):
    """Reduction for this move, or -1 when LMR does not apply.

    A cut node is one the search expects to fail high. Guessing wrong there is
    cheap, because the node was going to be cut off anyway, so reductions can
    be more aggressive. Neither BTC nor the port had the concept at all."""
    onset = LMR_MIN_MOVES
    if USE_LMR_ONSET and material_scale:
        onset = LMR_MIN_MOVES_MAT
    if not USE_LMR or moves_searched < onset or depth < LMR_MIN_DEPTH \
            or in_check or pv_node:
        return -1
    if (mv & (1 << 20)) or ((mv & 0xF0000) >> 16):
        return LMR_CAPTURE_CHECK if opp_in_check else LMR_CAPTURE
    d = depth if depth < LMR_MAX_DEPTH else LMR_MAX_DEPTH - 1
    m = moves_searched if moves_searched < LMR_MAX_MOVES else LMR_MAX_MOVES - 1
    reduction = LMR_TABLE[d, m]
    hist = int64(main_hist[(mv & 0xF000) >> 12, (mv & 0xFC0) >> 6]) \
        + _cont_hist_score(cont_hist, played, ply, mv)
    reduction -= c_div(hist, LMR_HIST_DIV)
    if USE_CORR_MARGIN:
        # Same reasoning as the futility margin: an evaluation that has
        # needed correcting here is a poor basis for reducing hard.
        reduction -= c_div(corr_adj, CORR_LMR_DIV)
    if USE_LMR_TTPV and tt_pv:
        # This position is on the principal variation, or was on one earlier
        # in the search. Its late moves are more likely to matter than a
        # plain non-PV node's, so reduce them less. Only quiets reach here.
        reduction -= LMR_TTPV
    if USE_LMR_TTCAP and tt_capture:
        # The stored best move here is a capture. A quiet alternative is
        # correspondingly less likely to be the move, so reduce it harder.
        reduction += LMR_TTCAP
    if USE_CUTNODE_LMR and cut_node:
        reduction += LMR_CUTNODE
    if not improving:
        reduction += 1
    if opp_in_check:
        reduction -= 1
    return reduction if reduction > 0 else 0


@njit(cache=False, fastmath=True, error_model='numpy')
def negamax(acc, alpha, beta, depth, ply, rep_idx, bb, st, undo_bb, undo_st, mls,
            scores, killers, main_hist, cap_hist, corr, cont_hist,
            counters, played, static_evals, pv_table, pv_len, rep, tt_key,
            tt_data, sc, fc, excluded, cut_node, null_floor):
    """excluded != 0 means this is a singular verification: the same position
    is searched with that one move forbidden, so the node must not take a TT
    cutoff, prune with RFP/razoring/null move/LMP/futility, write the TT, or
    trigger another singular check."""
    # int64(0), not the literal 0: numba specialises on integer literals, so
    # a literal here compiled a second copy of this whole function for the
    # singular-verification call site, which passes a real move.
    no_excl = int64(0)

    pv_len[ply] = ply
    pv_node = beta - alpha > 1

    code, value, alpha, beta, depth, tt_move, in_check, ev, improving, \
        material_scale = \
        _node_prologue(acc, alpha, beta, depth, ply, rep_idx, pv_node, bb, st,
                       undo_bb, undo_st, mls, scores, cap_hist, corr, rep,
                       static_evals, tt_key, tt_data, sc, fc, excluded,
                       null_floor, played)
    corr_adj = int64(sc[SC_CORR_ADJ])
    # Read now, before any recursive call overwrites the slot. A pv node is
    # ttPv by definition even when the entry is missing.
    tt_pv = int64(1) if (pv_node or sc[SC_TT_PV]) else int64(0)
    tt_capture = int64(1) if (tt_move and (tt_move & (1 << 20))) else int64(0)
    if code == NODE_RETURN:
        return value
    if code == NODE_QSEARCH:
        return qsearch(acc, alpha, beta, bb, st, undo_bb, undo_st, mls, scores,
                       cap_hist, corr, sc, fc, ply, tt_key, tt_data)

    null_ok = USE_NULL and excluded == 0 and depth >= NULL_MIN_DEPTH \
        and not in_check and ply
    if null_ok and USE_NULL_ZUGZWANG_GUARD:
        null_ok = _has_non_pawn_material(bb, st)
    if null_ok and USE_NULL_CUTNODE:
        # Only try to prove a fail-high at a node we already expect to
        # fail high.
        null_ok = cut_node != 0
    if null_ok and USE_NULL_VERIFY:
        # A verification search disables null move down to the ply it started
        # from, so the re-search cannot lean on the move it is checking.
        null_ok = ply >= sc[SC_NMP_MIN]
    if null_ok and USE_NULL_EVAL_GATE:
        # No point passing when the static evaluation already says we are
        # below beta: the null search is being asked to prove something the
        # evaluation contradicts, and it almost never does.
        null_ok = ev >= beta
    if null_ok:
        saved_ep, saved_hash = _make_null(bb, st, rep, rep_idx)
        if USE_NNUE:
            # A null move changes only the side to move, so the
            # accumulator carries over unchanged; propagate()
            # takes the side separately and swaps perspectives.
            acc[ply + 1] = acc[ply]
        played[ply + 1] = 0
        reduction = _null_reduction(depth, ev, beta)
        score = -negamax(acc, -beta, -beta + 1, depth - 1 - reduction, ply + 1,
                         rep_idx + 1, bb, st, undo_bb, undo_st, mls, scores,
                         killers, main_hist, cap_hist, corr, cont_hist, counters,
                         played, static_evals, pv_table, pv_len, rep, tt_key,
                         tt_data, sc, fc, no_excl, 1 - cut_node,
                         rep_idx + 1)
        _unmake_null(bb, st, saved_ep, saved_hash)
        if sc[SC_STOP]:
            return 0
        if score >= beta:
            if not USE_NULL_VERIFY:
                return score
            # A null move must never be allowed to prove a mate: the side to
            # move was handed a free tempo, so the mate may not exist.
            if abs(score) < MATE_SCORE:
                verify_depth = depth - 1 - reduction
                if sc[SC_NMP_MIN] or depth < NULL_VERIFY_MIN_DEPTH:
                    return score
                if verify_depth < 1:
                    # Below this the verification is a quiescence search of a
                    # node whose evaluation already beats beta, so it confirms
                    # almost anything. Take the cutoff unverified instead.
                    return score
                sc[SC_NMP_MIN] = ply + c_div(3 * verify_depth, 4)
                verify = negamax(acc, beta - 1, beta, verify_depth, ply,
                                 rep_idx, bb, st, undo_bb, undo_st, mls,
                                 scores, killers, main_hist, cap_hist, corr,
                                 cont_hist, counters, played, static_evals,
                                 pv_table, pv_len, rep, tt_key, tt_data, sc,
                                 fc, no_excl, int64(0), null_floor)
                sc[SC_NMP_MIN] = 0
                # The verifier reuses this ply's PV slots; reset so its scratch
                # line is not propagated upward.
                pv_len[ply] = ply
                if sc[SC_STOP]:
                    return 0
                if verify >= beta:
                    return score

    if USE_IMPROVING_BETA and not in_check and ev >= beta:
        # A node whose evaluation already beats beta is improving by any
        # useful definition, whatever the score two plies ago was.
        improving = 1

    # Internal iterative reductions: no TT move at this depth means the
    # iteration is cheap and mainly exists to populate the TT for the next one.
    # This must come after the null-move search, as in BTC: applying it in the
    # prologue shrank the null search by a ply on exactly the nodes that have
    # no TT guidance, which is where null move earns the most.
    iir_ok = USE_IIR and excluded == 0 and ply > 0 and depth >= 6 \
        and tt_move == 0
    if iir_ok and USE_IIR_ALLNODE:
        # At an all-node every move is searched anyway, so the iteration
        # is not mainly there to populate the table for the next one.
        iir_ok = pv_node or cut_node != 0
    if iir_ok:
        depth -= 1

    # ProbCut, transposition half. A stored lower bound this far above beta,
    # at a depth close to this one, decides the node without searching it.
    # material_scale gates it for the same reason it gates RFP and futility:
    # the margin is in centipawns and the specialised endgame score is not.
    if USE_PROBCUT and excluded == 0 and ply > 0 and not in_check \
            and not pv_node and material_scale \
            and depth >= PROBCUT_MIN_DEPTH \
            and abs(beta) < MATE_SCORE - PROBCUT_MARGIN:
        probcut_beta = beta + PROBCUT_MARGIN
        pc_hit, pc_d, pc_f, pc_s = tt_peek(bb, ply, tt_key, tt_data)
        # The depth floor is not decoration. Quiescence stores at depth 0,
        # and those entries are a raw stand-pat rather than a search; the
        # main search never sees them only because it probes at depth >= 1.
        # This block probes at depth - SLACK, so it is the one place that
        # invariant can be crossed by tuning the two constants apart.
        #
        # The flag test is an equality on purpose. A bitmask - `pc_f & 2`,
        # or the usual `bound & LOWER` - would also admit flag 3, which is
        # not a bound at all but the signature of an entry written at a
        # negative depth: _tt_pack shifts that into an unsigned field, so
        # it reads back as depth 127 and passes any depth gate, over a
        # score that may be a fail-low. Those entries exist whenever
        # BTC_DEPTH_CLAMP is off, which is the shipped default.
        floor = depth - PROBCUT_DEPTH_SLACK
        if floor < 1:
            floor = 1
        if pc_hit and pc_f == HASH_BETA and pc_d >= floor \
                and pc_s >= probcut_beta and abs(pc_s) < MATE_SCORE:
            return probcut_beta

    # ProbCut, search half. A capture that still beats beta by a wide margin
    # when searched at reduced depth says the node fails high, so it can be
    # pruned without a full-depth search.
    #
    # Runs before generate_moves for the same reason the singular
    # verification does: it borrows the shared mls[ply] row, and the real
    # move loop regenerates and re-sorts that row afterwards.
    if USE_PROBCUT_FULL and excluded == 0 and ply > 0 and not in_check \
            and not pv_node and material_scale \
            and depth >= PROBCUT_FULL_MIN_DEPTH \
            and abs(beta) < MATE_SCORE - PROBCUT_MARGIN:
        pc_beta = beta + PROBCUT_MARGIN
        pc_depth = depth - PROBCUT_FULL_REDUCTION
        if pc_depth < 1:
            pc_depth = 1
        # Nothing to learn when the table already holds a score below the
        # target at a depth at least as good as the one we would search at:
        # the captures would only confirm what is stored.
        pcf_hit, pcf_d, pcf_f, pcf_s = tt_peek(bb, ply, tt_key, tt_data)
        pc_skip = pcf_hit and pcf_d >= pc_depth and pcf_s < pc_beta \
            and abs(pcf_s) < MATE_SCORE
        if not pc_skip:
            pc_cnt = generate_captures(bb, st, mls[ply])
            _sort_captures(bb, st, mls[ply], scores[ply], pc_cnt, cap_hist)
            for pc_i in range(pc_cnt):
                pc_mv = mls[ply, pc_i]
                # Descending order, so once a losing capture appears every
                # remaining move is losing too - the same list property
                # quiescence relies on.
                if scores[ply, pc_i] < SCORE_BAD_CAPTURE_LIMIT:
                    break
                # The capture must be able to reach pc_beta on material
                # alone, otherwise the reduced search is asked to prove
                # something the exchange cannot deliver.
                if not see_ge(bb, st, pc_mv, pc_beta - ev):
                    continue
                rep[rep_idx] = bb[HASH]
                if make_move(bb, st, undo_bb, undo_st, ply, pc_mv) == 0:
                    continue
                if USE_NNUE:
                    acc_update(acc, ply, ply + 1, undo_bb[ply], bb, NET_FT_W,
                               NET_FT_B, NET_L1, NET_TABLE, NET_BUCKETS)
                played[ply + 1] = pc_mv
                # Quiescence first: it is far cheaper than the reduced
                # search and rejects most candidates outright.
                pc_score = -qsearch(acc, -pc_beta, -pc_beta + 1, bb, st,
                                    undo_bb, undo_st, mls, scores, cap_hist,
                                    corr, sc, fc, ply + 1, tt_key, tt_data)
                if pc_score >= pc_beta:
                    pc_score = -negamax(acc, -pc_beta, -pc_beta + 1, pc_depth,
                                        ply + 1, rep_idx + 1, bb, st, undo_bb,
                                        undo_st, mls, scores, killers,
                                        main_hist, cap_hist, corr, cont_hist,
                                        counters, played, static_evals,
                                        pv_table, pv_len, rep, tt_key,
                                        tt_data, sc, fc, no_excl,
                                        1 - cut_node, null_floor)
                unmake(bb, st, undo_bb, undo_st, ply)
                if sc[SC_STOP]:
                    return 0
                if pc_score >= pc_beta:
                    # pc_depth + 1 is what was actually proven: the reduced
                    # search plus the move that led to it.
                    tt_record(bb, pc_score, pc_depth + 1, int64(HASH_BETA),
                              pc_mv, ply, tt_key, tt_data,
                              int64(sc[SC_TT_GEN]), tt_pv)
                    return pc_score

    # Singular extension: if the TT move is much better than every alternative
    # at reduced depth, it is worth an extra ply. Verified by searching this
    # same position with that move excluded, against a window just below the
    # stored score; failing low there means nothing else comes close.
    #
    # Runs before this ply's moves are generated. The C verifies after
    # generating, but its move list is a stack local, while ours is the shared
    # mls[ply] row: a verification at the same ply regenerates and re-sorts
    # that row, so the parent would then iterate a list it did not sort. Same
    # moves, different order. Generating afterwards avoids it.
    singular = 0
    # material_scale gates the whole block: below SPECIALISED_MAX_PIECES the
    # score is a clamped mating drive, so singular_beta is not a margin below
    # the stored score and the verification decides nothing. Ungated, this
    # loses KBN vs K to a threefold - the same signature as the quiescence
    # fail-soft and adaptive re-search bugs.
    seek_mate = False
    if USE_SEEK_MATE and sc[SC_ROOT_DEPTH] >= SEEK_MATE_DEPTH:
        root_score = sc[SC_ROOT_SCORE]
        if root_score < 0:
            root_score = -root_score
        seek_mate = root_score > SEEK_MATE_SCORE
    if USE_SINGULAR and material_scale \
            and excluded == 0 and ply > 0 \
            and depth >= SINGULAR_MIN_DEPTH and tt_move != 0 \
            and not _is_shuffling(tt_move, st, played, ply,
                                  rep_idx - null_floor):
        tt_hit, tt_d, tt_f, tt_s = tt_peek(bb, ply, tt_key, tt_data)
        if tt_hit and tt_d >= depth - 3 and abs(tt_s) < MATE_SCORE \
                and (tt_f == HASH_BETA or tt_f == HASH_EXACT):
            singular_beta = tt_s - 2 * depth
            sing_d = c_div(depth - 1, 2)
            sing_score = negamax(acc, singular_beta - 1, singular_beta, sing_d, ply,
                                 rep_idx, bb, st, undo_bb, undo_st, mls,
                                 scores, killers, main_hist, cap_hist,
                                 corr, cont_hist, counters, played, static_evals,
                                 pv_table, pv_len, rep, tt_key, tt_data,
                                 sc, fc, tt_move, cut_node, null_floor)
            # the verifier reuses this ply's PV slots; reset so its scratch
            # line is not propagated upward
            pv_len[ply] = ply
            if sc[SC_STOP]:
                return 0
            if sing_score < singular_beta:
                singular = 1
                if USE_SING_DOUBLE and not pv_node \
                        and sing_score < singular_beta - SINGULAR_DOUBLE_MARGIN:
                    # Failed low by a wide margin: nothing else is close.
                    singular = 2
            elif USE_MULTICUT and sing_score >= beta \
                    and abs(sing_score) < MATE_SCORE:
                # Not singular, and something other than the TT move already
                # fails high at reduced depth, so the node itself fails high.
                return sing_score
            elif USE_SING_NEGATIVE and not pv_node and tt_s >= beta:
                # Neither singular nor enough for multicut, and the stored
                # score says this node is expected to fail high anyway. Search
                # the TT move shallower and let the alternatives have the depth.
                singular = -SINGULAR_NEG_TT
            elif USE_SING_NEGATIVE and not pv_node and cut_node != 0:
                singular = -SINGULAR_NEG_CUT
            if seek_mate and singular > 0:
                # Already winning by a wide margin at high depth: stop
                # spending plies proving it. Only the extensions are
                # dropped - multicut and the negative extension both make
                # the tree smaller and are exactly what is wanted here.
                singular = 0

    cnt = generate_moves(bb, st, mls[ply])
    _sort_moves(bb, st, mls[ply], scores[ply], cnt, ply, tt_move, killers,
                main_hist, cap_hist, cont_hist, counters, played)

    # Loop-invariant, so computed once. played[ply] == 0 means this node
    # was reached by a null move, where the child evaluates the same
    # position from the other side and the comparison is only noise.
    opp_worse = False
    if USE_OPP_WORSENING and ply >= 1 and not in_check \
            and played[ply] != 0 \
            and static_evals[ply] != EVAL_NONE \
            and static_evals[ply - 1] != EVAL_NONE:
        opp_worse = static_evals[ply] > -static_evals[ply - 1]

    hash_flag = int64(HASH_ALPHA)
    best_move = 0
    best_score = -INFINITY
    legal_count = 0
    moves_searched = 0
    quiets = mls[MAX_SEARCH_PLY + ply]
    quiet_cnt = 0
    captures = scores[MAX_SEARCH_PLY + ply]
    capture_cnt = 0

    row = mls[ply]
    srow = scores[ply]
    for i in range(cnt):
        mv = row[i] if USE_ROWVIEW else mls[ply, i]
        if mv == excluded:
            continue
        mv_score = srow[i] if USE_ROWVIEW else scores[ply, i]
        # SEE pruning is decided on the position before the move, but applied
        # after make_move has confirmed the move is legal: pruning earlier
        # would leave legal_count short and could report a false stalemate.
        # The counter must include every legal move considered; ours counted
        # only those actually searched, so the pruners lagged.
        see_count = legal_count if USE_MOVECOUNT_FIX else moves_searched
        prune_see = _see_prunable(mv_score, mv, depth, see_count,
                                  pv_node, in_check, alpha, tt_move, excluded)

        rep[rep_idx] = bb[HASH]
        if make_move(bb, st, undo_bb, undo_st, ply, mv) == 0:
            continue
        if USE_NNUE and not USE_LAZY_ACC:
            acc_update(acc, ply, ply + 1, undo_bb[ply], bb, NET_FT_W,
                       NET_FT_B, NET_L1, NET_TABLE, NET_BUCKETS)
        legal_count += 1
        opp_in_check = _in_check(bb, st)

        move_count = legal_count - 1 if USE_MOVECOUNT_FIX else moves_searched
        skip = _skip_quiet(mv_score, mv, depth, move_count,
                           improving, ev, alpha, pv_node, in_check,
                           opp_in_check, material_scale, opp_worse,
                           corr_adj)
        if (prune_see and not opp_in_check) or (excluded == 0 and skip):
            unmake(bb, st, undo_bb, undo_st, ply)
            continue
        # Below the pruning tests on purpose: a pruned move never reads this
        # accumulator, and nothing between make_move and here reads it either.
        if USE_NNUE and USE_LAZY_ACC:
            acc_update(acc, ply, ply + 1, undo_bb[ply], bb, NET_FT_W,
                       NET_FT_B, NET_L1, NET_TABLE, NET_BUCKETS)

        played[ply + 1] = mv
        new_depth = depth - 1 + (singular if mv == tt_move else 0)

        if moves_searched == 0:
            score = -negamax(acc, -beta, -alpha, new_depth, ply + 1, rep_idx + 1,
                             bb, st, undo_bb, undo_st, mls, scores, killers,
                             main_hist, cap_hist, corr, cont_hist, counters, played,
                             static_evals, pv_table, pv_len, rep, tt_key,
                             tt_data, sc, fc, no_excl,
                             int64(0) if pv_node else 1 - cut_node,
                             null_floor)
        else:
            reduction = _lmr_reduction(mv, depth, move_count, improving,
                                       in_check, opp_in_check, pv_node,
                                       cut_node, main_hist, cont_hist, played,
                                       ply, corr_adj, tt_pv, tt_capture,
                                       material_scale)
            reduced = int64(-1)
            if reduction >= 0:
                reduced = new_depth - reduction
                if reduced < 1:
                    reduced = int64(1)
                score = -negamax(acc, -alpha - 1, -alpha, reduced, ply + 1,
                                 rep_idx + 1, bb, st, undo_bb, undo_st, mls,
                                 scores, killers, main_hist, cap_hist,
                                 corr, cont_hist, counters, played, static_evals,
                                 pv_table, pv_len, rep, tt_key, tt_data,
                                 sc, fc, no_excl, int64(1), null_floor)
            else:
                score = alpha + 1
            if score > alpha:
                # How far above best_score the reduced search came back says how
                # wrong the reduction was. Far above: the move deserves a ply
                # more than new_depth. Barely above: it can afford a ply less.
                #
                # reduced == -1 means LMR never applied, and that path must
                # re-search at new_depth unconditionally - applying the
                # verify_depth > reduced guard there would skip the search and
                # leave score at the sentinel alpha + 1.
                #
                # Both margins are centipawns compared against best_score, so
                # they inherit the unit-scale rule. Below
                # SPECIALISED_MAX_PIECES the score is a mating drive clamped to
                # MATE_SCORE - 1, which makes `score < best_score + 8` true on
                # essentially every move: the re-search then loses a ply on
                # every move of the conversion. Measured: ungated, this loses
                # KBN vs K to a threefold at move 14, the same signature the
                # quiescence fail-soft bug produced.
                verify_depth = new_depth
                if reduced >= 0 and material_scale:
                    if reduced < new_depth and score > best_score + LMR_DEEPER:
                        verify_depth += 1
                    if score < best_score + LMR_SHALLOWER:
                        verify_depth -= 1
                if reduced < 0 or verify_depth > reduced:
                    score = -negamax(acc, -alpha - 1, -alpha, verify_depth,
                                     ply + 1, rep_idx + 1, bb, st, undo_bb,
                                     undo_st, mls, scores, killers, main_hist,
                                     cap_hist, corr, cont_hist, counters,
                                     played, static_evals, pv_table, pv_len,
                                     rep, tt_key, tt_data, sc, fc, no_excl,
                                     1 - cut_node, null_floor)
                if alpha < score < beta:
                    score = -negamax(acc, -beta, -alpha, new_depth, ply + 1,
                                     rep_idx + 1, bb, st, undo_bb, undo_st,
                                     mls, scores, killers, main_hist,
                                     cap_hist, corr, cont_hist, counters, played,
                                     static_evals, pv_table, pv_len, rep,
                                     tt_key, tt_data, sc, fc, no_excl,
                                     int64(0), null_floor)

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
        if score > best_score:
            best_score = score

        if score > alpha:
            hash_flag = int64(HASH_EXACT)
            best_move = mv
            alpha = score
            pv_table[ply, ply] = mv
            for next_ply in range(ply + 1, pv_len[ply + 1]):
                pv_table[ply, next_ply] = pv_table[ply + 1, next_ply]
            pv_len[ply] = pv_len[ply + 1]
            if score >= beta:
                if excluded == 0:
                    tt_record(bb, best_score, depth, int64(HASH_BETA), mv, ply,
                              tt_key, tt_data, int64(sc[SC_TT_GEN]), tt_pv)
                _beta_cutoff_update(bb, st, mv, depth, ply, killers,
                                    main_hist, cap_hist, cont_hist, counters,
                                    played, quiets, quiet_cnt, captures,
                                    capture_cnt)
                if excluded == 0 and not in_check and not (mv & (1 << 20))                         and best_score > ev:
                    _update_correction(bb, st, corr, ev, best_score, depth,
                                       1, sc, played, ply)
                return best_score

    if legal_count == 0:
        # inside a verification, "no legal moves" only means every move was
        # the excluded one; that is not mate
        if excluded != 0:
            return alpha
        return -MATE_VALUE + ply if in_check else _draw_score(bb)

    if excluded == 0:
        tt_record(bb, best_score, depth, int64(hash_flag), best_move, ply,
                  tt_key, tt_data, int64(sc[SC_TT_GEN]), tt_pv)
    if excluded == 0 and not in_check and best_score > -INFINITY             and not (best_move != 0 and (best_move & (1 << 20)))             and (best_score > ev) == (best_move != 0):
        _update_correction(bb, st, corr, ev, best_score, depth,
                           1 if best_move != 0 else 0, sc, played, ply)
    # best_score is set whenever a legal move was searched, and the first
    # legal move always is: _see_prunable and _skip_quiet both return False
    # while moves_searched == 0. The guard covers the degenerate case only.
    return best_score if best_score > -INFINITY else alpha


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
        # int32, not int64: the ordering band runs from SCORE_TT_BEST
        # at 10,000,000 down to about -1,008,000, so the headroom is
        # three orders of magnitude, and the insertion sort moves half
        # the bytes. The row doubles as the tried-capture list, which
        # holds int32 move words.
        self.scores = np.zeros((MAX_PLY, 256), dtype=np.int32)
        self.killers = np.zeros((2, MAX_SEARCH_PLY + 1), dtype=np.int32)
        self.main_hist = np.zeros((12, 64), dtype=np.int16)
        self.cap_hist = np.zeros((12, 64, 12), dtype=np.int16)
        # [table][prev piece][prev to][piece][to]; table 0 is 1-ply, 1 is 2-ply
        self.cont_hist = np.zeros((CONT_PLANES, 12, 64, 12, 64),
                                  dtype=np.int16)
        self.counters = np.zeros((12, 64), dtype=np.int32)
        # [plane][key][side to move]; plane 0 pawn, 1 white pieces, 2 black
        self.corr = np.zeros((CORR_PLANES, CORR_SIZE, 2), dtype=np.int16)
        self.played = np.zeros(MAX_SEARCH_PLY + 8, dtype=np.int32)
        self.static_evals = np.zeros(MAX_SEARCH_PLY + 8, dtype=np.int64)
        self.pv_table = np.zeros((MAX_SEARCH_PLY + 1, MAX_SEARCH_PLY + 1),
                                 dtype=np.int32)
        self.pv_len = np.zeros(MAX_SEARCH_PLY + 1, dtype=np.int64)
        self.rep = np.zeros(1024 + MAX_SEARCH_PLY + 8, dtype=np.uint64)
        self.tt_key = np.zeros(tt_entries, dtype=np.uint64)
        self.tt_data = np.zeros(tt_entries, dtype=np.uint64)
        # One accumulator per ply, parallel to undo_bb. Copy-make
        # means unmake needs no work here: the parent's row is
        # still intact, so returning up the tree is free.
        self.acc = np.zeros((MAX_PLY + 1, 2, max(NET_L1, 1)),
                            dtype=np.int16)
        self.sc = np.zeros(SC_COUNT, dtype=np.int64)
        self.fc = np.zeros(2, dtype=np.float64)


def _root_negamax(state, bb, st, alpha, beta, depth, rep_base):
    return negamax(state.acc, alpha, beta, depth, 0, rep_base, bb, st, state.undo_bb,
                   state.undo_st, state.mls, state.scores, state.killers,
                   state.main_hist, state.cap_hist, state.corr,
                   state.cont_hist, state.counters, state.played,
                   state.static_evals,
                   state.pv_table, state.pv_len, state.rep, state.tt_key,
                   state.tt_data, state.sc, state.fc, 0, int64(0),
                   int64(0))


def _is_slower_mate(best_score, new_score):
    """Mate scores already encode distance (-MATE_VALUE + ply), so a shorter
    mate scores higher. A deeper iteration can still return a longer mate
    after a TT cutoff; never trade a proven mate for a slower one."""
    return best_score > MATE_SCORE and MATE_SCORE < new_score < best_score


# How much the next iteration is predicted to cost, as a multiple of the time
# already spent. The loop starts another iteration only while
# elapsed * ITER_RATIO <= adjusted_soft, so 2.0 means "stop once half the soft
# budget is gone" - the textbook assumption that the next ply costs as much as
# every ply before it.
#
# Measured on this engine, that assumption is wrong. Cumulative time per extra
# ply, three positions, depths 10-20, is a median of **1.46**, not 2.00, so the
# next iteration costs about 0.46x the time already spent. At 2.0 we stop at
# 0.50x soft where 1.46 would allow 0.68x - about 37% more thinking per move.
#
# Safer than it looks: soft only gates whether to *start* another iteration.
# hard_ms is what aborts one in flight, and that is what stands between us and
# a flag, so this does not trade against time-loss risk. The spread is wide
# (0.90 to 4.32 across positions), which is exactly what hard_ms absorbs.
ITER_RATIO = _tune("BTC_ITER_RATIO", 2.0)


def _adjusted_soft(soft_ms, stable_count, score_drop):
    adjusted = soft_ms
    if stable_count >= 5:
        adjusted = adjusted * 70 // 100
    if score_drop:
        adjusted = adjusted * 130 // 100
    return adjusted


def _aspiration_search(state, bb, st, prev_score, depth, rep_base):
    delta = ASPIRATION_DELTA
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
            if USE_ASP_V2:
                # A fail low says the truth is below the window, so the old
                # beta is far too optimistic. Pulling it halfway to alpha
                # keeps the re-search narrow where it matters.
                beta = (alpha + beta) // 2
            alpha = max(score - delta, -INFINITY)
        else:
            beta = min(score + delta, INFINITY)
        if USE_ASP_V2:
            delta += max(1, delta * ASPIRATION_GROWTH // 128)
        else:
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
    state.sc[SC_NMP_MIN] = 0
    state.sc[SC_CORR_ADJ] = 0
    state.sc[SC_ROOT_DEPTH] = 0
    state.sc[SC_ROOT_SCORE] = 0
    state.sc[SC_TT_GEN] += 1
    state.killers[:] = 0
    state.pv_table[:] = 0
    state.pv_len[:] = 0
    state.static_evals[:] = 0
    state.played[:] = 0
    state.fc[0] = started + hard_ms / 1000.0

    # Seed ply 0 from the board. Every deeper ply is derived from its parent by
    # btc_nnue.update, so this is the only place the accumulator is ever built
    # from scratch during a search.
    if USE_NNUE:
        refresh_acc(bb, NET_FT_W, NET_FT_B, NET_L1, state.acc[0], NET_TABLE,
                    NET_BUCKETS)

    # Ancestors only: the current position is written into the array by the
    # root's own move loop. Including it here duplicated one entry, which
    # inverts the parity of everything older and would break the
    # every-other-slot walk in _is_repetition.
    history = max(int(rep_count) - 1, 0)
    rep_base = min(history, 1024)
    state.rep[:rep_base] = rep_keys[history - rep_base:history]
    # Where history ends and the search tree begins. _is_repetition needs it to
    # tell a second occurrence against the real game from a repetition inside
    # the tree, and it rides in sc rather than becoming a negamax parameter.
    state.sc[SC_REP_BASE] = rep_base

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
            if elapsed_ms * ITER_RATIO > _adjusted_soft(soft_ms, stable_count,
                                                        score_drop):
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
        # Read during the next iteration: a statement about the position,
        # not about the iteration in progress.
        state.sc[SC_ROOT_DEPTH] = depth
        state.sc[SC_ROOT_SCORE] = score

    return best_move, best_score, completed, int(state.sc[SC_NODES])
