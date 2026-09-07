"""Time management ported from BTC uci.cpp parseGo (sudden death path).

Returns (soft_ms, hard_ms): soft is the iterative-deepening target, hard is
the wall the search never crosses. OVERHEAD_MS covers FEN parsing, tracker
work, the python-chess safety net and referee measurement slack; calibrate it
from the platform validation log."""

OVERHEAD_MS = 120
MTG = 30


def budget(time_left_ms, move_number, increment_ms=500):
    time_left = time_left_ms + increment_ms * (MTG - 1) - OVERHEAD_MS * (2 + MTG)
    if time_left < 1:
        time_left = 1

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
    return int(soft), int(hard)
