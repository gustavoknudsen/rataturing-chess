"""Time management ported from BTC uci.cpp parseGo (sudden death path).

Returns (soft_ms, hard_ms): soft is the iterative-deepening target, hard is
the wall the search never crosses. OVERHEAD_MS covers FEN parsing, tracker
work, the python-chess safety net and referee measurement slack; calibrate it
from the platform validation log."""

import os

# Per-move reserve. The platform charges 1 to 2 ms per move (measured from
# the clock lines of the finals-day logs) and the search itself never
# overshoots its budget by more than 9 ms, so the shipped panel value is 100.
# The 420 default here is the pre-finals figure, kept so an unset environment
# reproduces the qualification build.
OVERHEAD_MS = int(os.environ.get("BTC_MOVE_OVERHEAD", "420"))

# Assumed moves remaining, the divisor for the whole scheme. 40 is the
# qualification value; the finals build sets 28 from agent.py, which spends
# about 30 percent more thinking time in a normal game and still ends a
# 150-move game with clock in hand (simulated with this function).
MTG = int(os.environ.get("BTC_MTG", "40"))


def budget(time_left_ms, move_number, increment_ms=500):
    time_left = time_left_ms + increment_ms * (MTG - 1) - OVERHEAD_MS * (2 + MTG)
    # Floor at half the clock: at short controls the constant reserve above
    # would exceed the clock and every budget would derive from 1 ms.
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
    # The clamp above lowers hard alone; soft must never sit above it.
    if soft > hard:
        soft = hard
    return int(soft), int(hard)
