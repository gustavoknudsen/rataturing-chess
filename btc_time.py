"""Time management ported from BTC uci.cpp parseGo (sudden death path).

Returns (soft_ms, hard_ms): soft is the iterative-deepening target, hard is
the wall the search never crosses. OVERHEAD_MS covers FEN parsing, tracker
work, the python-chess safety net and referee measurement slack; calibrate it
from the platform validation log."""

import os

OVERHEAD_MS = 120

# Assumed moves remaining, the divisor for the whole scheme.
#
# 30 is BTC's value and is measurably too conservative here: simulating a
# 60-move game at 120s+0.5s leaves 12.6 s unused when the search spends its
# full soft budget, and 28.9 s when the stable-move rule in _adjusted_soft
# takes its 30% off, which is most moves. The engine plays whole games at
# about three quarters of the clock it is entitled to. 24 recovers most of
# that and still finishes an 80-move game with 10 s in hand.
#
# Confirmed at the real control: 50 games of MTG=24 against MTG=30 at
# 120s+0.5s scored 56.0%, +42 elo [-34, +118], likelihood of superiority 86%,
# and critically **zero losses on time** and no fifty-move or repetition
# artefacts. The elo interval spans zero, so the strength claim is unproven;
# what the match establishes is that spending the extra clock is safe, and
# more time at the same node rate is the most reliable gain in computer chess.
# Sweep further with BTC_MTG, but re-check for flags before shipping a lower
# value: 21 leaves only 0.6 s at move 100 if the search spends its full soft
# budget every move.
MTG = int(os.environ.get("BTC_MTG", "24"))


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
