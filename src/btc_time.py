"""Time management ported from BTC uci.cpp parseGo (sudden death path).

Returns (soft_ms, hard_ms): soft is the iterative-deepening target, hard is
the wall the search never crosses. OVERHEAD_MS covers FEN parsing, tracker
work, the python-chess safety net and referee measurement slack; calibrate it
from the platform validation log."""

import os

# Measured on this machine: our own search does NOT overshoot - wall time
# came in within 9 ms of the authorised budget at every clock level. So
# whatever the platform charges beyond that is referee slack, and a rated
# floor of ~200 ms implies a real per-move cost near 420 ms. Tunable so
# that can be tested; default unchanged until it is.
OVERHEAD_MS = int(os.environ.get("BTC_MOVE_OVERHEAD", "420"))

# Assumed moves remaining, the divisor for the whole scheme.
#
# 24 rather than BTC's 30. At 30 the engine plays whole games on about three
# quarters of the clock it is entitled to; 24 recovers most of that and still
# finishes an 80-move game with 10 s in hand. Verified over 50 games at the
# real control with zero losses on time.
#
# Re-check for flags before going lower: 21 leaves only 0.6 s at move 100 if
# the search spends its full soft budget every move.
# CHANGED 2026-09-11: 24 -> 40, together with OVERHEAD_MS 120 -> 420.
# The old pair was calibrated against a modelled per-move cost of 120 ms. The
# real cost is about 420 ms, so every move leaked ~300 ms the model did not
# know about: the base clock was gone by move 57 and the engine then played
# 80 ms moves for the rest of the game. Measured: our search does NOT
# overshoot its budget (within 9 ms at every clock level), so the leak is
# platform side and the reserve had to be told about it.
#
# Same total time, redistributed: mean spend over 90 moves is 392 ms against
# the old 411 ms, but moves 40-90 get 261 ms instead of 80 ms.
MTG = int(os.environ.get("BTC_MTG", "40"))


def budget(time_left_ms, move_number, increment_ms=500):
    time_left = time_left_ms + increment_ms * (MTG - 1) - OVERHEAD_MS * (2 + MTG)
    # The reserve above is a constant 17.6 s at the shipped values. Against
    # the 120 s clock it was tuned for it is a sensible cushion, but at any
    # shorter control, or at any control whose increment does not refill it,
    # it exceeds the whole clock and the old floor of 1 made every budget
    # below derive from 1 ms: 0 ms soft and the 10 ms hard floor while
    # seconds remained, for the rest of the game. It was also non-monotonic,
    # handing out 10 ms at a 13 s clock and 130 ms at 900 ms.
    #
    # Flooring at half the clock instead keeps a starved budget proportional
    # to what is actually left. At increment 500 this cannot bind at any
    # clock value, so the tournament control is bit-identical; both
    # test_time_budget and an independent audit confirmed that over the full
    # range rather than trusting the algebra.
    #
    # Be clear about what this does NOT do. Below about 450 ms of increment
    # the floor overrides the reserve rather than protecting it, and hands
    # out time the platform still charges 420 ms per move for. In a naive
    # drain model that reaches a zero clock 15 to 25 moves earlier than the
    # old formula did. The old behaviour there was 10 ms moves for the rest
    # of the game, so this is a different failure rather than a strictly
    # safer one. At any control under 450 ms increment, set BTC_MTG and
    # BTC_MOVE_OVERHEAD from finals_day/tc_tune.py as well; the floor alone
    # is not a fix for a short control.
    floor = time_left_ms // 2
    if floor < 1:
        floor = 1
    if time_left < floor:
        time_left = floor

    opt_scale = min((0.9 + move_number / 120.0) / MTG,
                    0.9 * time_left_ms / time_left)
    optimal = opt_scale * time_left

    hard = min((time_left_ms - OVERHEAD_MS) / 4.0, 2.5 * optimal)
    hard = max(hard, optimal)
    soft = optimal

    if time_left_ms < 1500:
        emergency = max(time_left_ms / 2.0 + increment_ms - OVERHEAD_MS, 10.0)
        soft = hard = emergency

    hard = max(min(hard, time_left_ms - OVERHEAD_MS), 10.0)
    # The emergency branch above sets soft = hard, and the clamp then lowers
    # hard alone, so soft could end up above the wall it is supposed to sit
    # under: at a 500 ms clock this returned soft 330 against hard 80. hard is
    # enforced and soft is not, so the search merely started an iteration it
    # could not finish, but nothing downstream should have to know that.
    # Untouched above 1500 ms, where the emergency branch never runs.
    if soft > hard:
        soft = hard
    return int(soft), int(hard)
