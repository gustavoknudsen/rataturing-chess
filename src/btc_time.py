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
