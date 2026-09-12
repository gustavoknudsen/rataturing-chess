"""Pick MTG and the overhead model for a time control other than 120 + 0.5.

    .venv/Scripts/python.exe tools/tc_tune.py

The distinction this tool exists to respect: OVERHEAD_MS is two different
things wearing one name. There is the real per-move cost the platform charges
us, which the finals-day logs measured at 1 to 2 ms, and there
is the engine's model of that cost, which is what OVERHEAD_MS actually feeds.
Tuning can only move the model. So the simulation always charges the measured
real cost per move and lets the model vary, which is the only honest way to
ask whether a different model plays a better game. REAL_OVERHEAD_MS below
is that measured figure; the engine's model is OVERHEAD_MS in btc_time.

What this tool can and cannot tell you matters. Whether a configuration runs
out of clock over a 100 move game is exact, and losing on time is a lost game,
so that verdict is worth trusting. Which of two configurations that both
survive plays better chess is not something a simulation can answer: earlier
versions scored by total thinking time and by evenness of allocation, and the
two objectives recommended opposite extremes of the grid. Only an SPRT at the
new control can rank survivors.

So the output is deliberately narrow. For each control it says whether the
shipped settings flag, and if they do, the smallest change away from them that
does not. Smallest, not best, because a minimal change keeps every hour of
tuning that went into the shipped pair and fixes only what is broken.

btc_time.budget reads MTG and OVERHEAD_MS as globals at call time, so this
sets them on the module rather than duplicating the formula. There is one
implementation of the budget and this is testing it, not a copy of it.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import btc_time

# What the platform charges per move, measured from the clock lines of the
# finals-day match logs.
REAL_OVERHEAD_MS = 2

CONTROLS = [
    ("120 + 0.5  (tournament)", 120000, 500),
    ("120 + 0.1", 120000, 100),
    ("60 + 0.5", 60000, 500),
    ("60 + 0.2", 60000, 200),
    ("30 + 0.5", 30000, 500),
    ("30 + 0.2", 30000, 200),
    ("30 + 0.1", 30000, 100),
    ("15 + 0.1", 15000, 100),
    ("10 + 0.1", 10000, 100),
    ("180 + 2", 180000, 2000),
    ("300 + 0", 300000, 0),
    ("600 + 0", 600000, 0),
]

MTG_GRID = (12, 16, 20, 24, 30, 36, 40, 48, 56)
MODEL_GRID = (120, 200, 280, 360, 420, 500, 600)
GAME_MOVES = 100

SHIPPED_MTG = btc_time.MTG
SHIPPED_OVERHEAD = btc_time.OVERHEAD_MS


def simulate(base_ms, inc_ms, mtg, model_overhead):
    """Play a whole game against the budget function.

    Returns (total thinking ms, balance score) or None if the clock ran out,
    which is a loss and disqualifies the configuration.
    """
    saved_mtg, saved_overhead = btc_time.MTG, btc_time.OVERHEAD_MS
    btc_time.MTG, btc_time.OVERHEAD_MS = mtg, model_overhead
    try:
        clock = float(base_ms)
        total = 0.0
        balance = 0.0
        for move in range(1, GAME_MOVES + 1):
            soft, _ = btc_time.budget(int(clock), move, inc_ms)
            cost = soft + REAL_OVERHEAD_MS
            if cost >= clock:
                return None
            clock -= cost
            clock += inc_ms
            total += soft
            # Sum of logs, so the score is driven by the geometric mean rather
            # than the arithmetic one. Total thinking time alone rewards
            # spending the clock on the opening and starving every move after
            # it, which is how an earlier version of this tool came to
            # recommend MTG 12. A starved move costs far more than a generous
            # one gains, and the log captures that where a sum does not.
            balance += math.log(max(soft, 1.0))
        return total, balance
    finally:
        btc_time.MTG, btc_time.OVERHEAD_MS = saved_mtg, saved_overhead


def _distance(mtg, model):
    """How far a setting sits from the shipped pair, scaled so the two axes
    are comparable rather than dominated by the larger raw number."""
    return (abs(mtg - SHIPPED_MTG) / float(SHIPPED_MTG)
            + abs(model - SHIPPED_OVERHEAD) / float(SHIPPED_OVERHEAD))


def nearest_surviving(base_ms, inc_ms):
    """The smallest departure from the shipped settings that does not flag."""
    best = None
    for mtg in MTG_GRID:
        for model in MODEL_GRID:
            if simulate(base_ms, inc_ms, mtg, model) is None:
                continue
            distance = _distance(mtg, model)
            if best is None or distance < best[0]:
                best = (distance, mtg, model)
    return best


def main():
    print("recommended settings per time control")
    print("real per move cost charged in simulation: {} ms".format(
        REAL_OVERHEAD_MS))
    print("shipped values: MTG {}  OVERHEAD_MS {}".format(
        btc_time.MTG, btc_time.OVERHEAD_MS))
    print("")

    header = "{:<24} {:>10} {:>22}".format(
        "control", "shipped", "smallest safe change")
    print(header)
    print("-" * len(header))

    for label, base, inc in CONTROLS:
        if simulate(base, inc, SHIPPED_MTG, SHIPPED_OVERHEAD) is not None:
            print("{:<24} {:>10} {:>22}".format(label, "survives", "none needed"))
            continue
        best = nearest_surviving(base, inc)
        if best is None:
            print("{:<24} {:>10} {:>22}".format(
                label, "FLAGS", "nothing survives"))
            continue
        _, mtg, model = best
        print("{:<24} {:>10} {:>22}".format(
            label, "FLAGS", "MTG {}  OVERHEAD {}".format(mtg, model)))

    print("")
    print("'nothing survives' means the real per move cost alone exceeds the")
    print("clock over 100 moves, so no setting of ours rescues that control.")
    print("REAL_OVERHEAD_MS is the finals-day measurement; re-measure it from")
    print("the engine's clock lines if the platform changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
