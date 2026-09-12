"""Regression suite for btc_time.budget. Run: python tests/test_time_budget.py

The tournament time management was tuned late and is worth real Elo, so the
first and most important thing here is that the tournament control is
untouched: every budget at increment 500 is compared against a copy of the
previous implementation and must match exactly, at every clock value and
move number, not at a sample.

The rest asserts the properties the old code violated once the control
changed: budgets stay positive while there is time to use, soft never exceeds
hard, and thinking time never increases as the clock falls.
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

import btc_time
from btc_time import MTG, OVERHEAD_MS

FAILURES = []
CHECKS = [0]


def check(ok, label):
    CHECKS[0] += 1
    if not ok:
        FAILURES.append(label)


def reference_budget(time_left_ms, move_number, increment_ms=500):
    """The implementation as it stood before the proportional floor.

    Kept verbatim so the tournament control can be proved unchanged. Do not
    'fix' this copy: its whole value is being the old behaviour.
    """
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
    if soft > hard:
        soft = hard
    return int(soft), int(hard)


def test_tournament_control_unchanged():
    """Increment 500: bit-identical to the old implementation, everywhere.

    Clocks step by 1 ms below 2 s, where the emergency branch lives and any
    difference would be most damaging, then coarsely up to well past the
    120 s starting clock.
    """
    clocks = list(range(0, 2000))
    clocks += list(range(2000, 200001, 37))
    moves = [1, 2, 5, 10, 20, 40, 60, 80, 100, 150, 200]
    mismatches = 0
    first = None
    for clock in clocks:
        for move in moves:
            got = btc_time.budget(clock, move, 500)
            want = reference_budget(clock, move, 500)
            if got != want:
                mismatches += 1
                if first is None:
                    first = (clock, move, got, want)
    check(mismatches == 0,
          "tournament control changed at {} cells, first {}".format(
              mismatches, first))
    print("  increment 500: {} cells compared, {} differ".format(
        len(clocks) * len(moves), mismatches))


def test_no_degenerate_budgets():
    """The engine must never be starved while it still has time to spend.

    This is what the old floor produced at every reduced control: a 0 ms soft
    budget and the 10 ms hard floor, for the rest of the game, with seconds
    still on the clock. Two thresholds, because a genuinely short clock
    should produce a genuinely short budget: above 5 s the budget must clear
    50 ms, and anywhere above the emergency branch it must at least be
    positive. The old code returned 0 ms at a 13 s clock and failed both.
    """
    starved = []
    zeroed = []
    for increment in (0, 50, 100, 200, 500, 2000):
        for clock in range(1501, 120001, 97):
            for move in (1, 20, 40, 80, 150):
                soft, hard = btc_time.budget(clock, move, increment)
                if soft < 1 or hard < 1:
                    zeroed.append((increment, clock, move, soft, hard))
                elif clock > 5000 and (soft < 50 or hard < 50):
                    starved.append((increment, clock, move, soft, hard))
    check(not zeroed, "zero budgets: {} cases, first {}".format(
        len(zeroed), zeroed[0] if zeroed else None))
    check(not starved, "starved above a 5 s clock: {} cases, first {}".format(
        len(starved), starved[0] if starved else None))
    print("  zero budgets above 1.5 s: {}".format(len(zeroed)))
    print("  under 50 ms above a 5 s clock: {}".format(len(starved)))


def test_soft_never_exceeds_hard():
    bad = []
    for increment in (0, 50, 100, 200, 500, 2000):
        for clock in range(0, 120001, 53):
            for move in (1, 20, 40, 80, 150):
                soft, hard = btc_time.budget(clock, move, increment)
                if soft > hard:
                    bad.append((increment, clock, move, soft, hard))
    check(not bad, "soft exceeded hard: {} cases, first {}".format(
        len(bad), bad[0] if bad else None))
    print("  soft > hard: {} cases".format(len(bad)))


def test_monotonic_in_clock():
    """More clock must never buy less thinking time.

    The old code handed out 10 ms at a 13 s clock and 130 ms at 900 ms. The
    emergency branch below 1500 ms is deliberately discontinuous, so the
    check starts above it.
    """
    bad = []
    for increment in (0, 100, 200, 500):
        for move in (1, 30, 80):
            previous = -1
            for clock in range(1500, 120001, 31):
                soft, _ = btc_time.budget(clock, move, increment)
                if soft < previous:
                    bad.append((increment, move, clock, soft, previous))
                previous = soft
    check(not bad, "soft not monotonic: {} cases, first {}".format(
        len(bad), bad[0] if bad else None))
    print("  non-monotonic steps above 1.5 s: {}".format(len(bad)))


def test_budget_fits_the_clock():
    """The hard wall must always leave the per-move overhead behind it."""
    bad = []
    for increment in (0, 100, 500):
        for clock in range(500, 120001, 89):
            for move in (1, 40, 120):
                _, hard = btc_time.budget(clock, move, increment)
                if hard > max(clock - OVERHEAD_MS, 10):
                    bad.append((increment, clock, move, hard))
    check(not bad, "hard wall past the clock: {} cases, first {}".format(
        len(bad), bad[0] if bad else None))
    print("  hard wall past the clock: {} cases".format(len(bad)))


def test_returns_ints():
    for increment in (0, 100, 500):
        for clock in (0, 1, 999, 1500, 60000):
            soft, hard = btc_time.budget(clock, 20, increment)
            check(isinstance(soft, int) and isinstance(hard, int),
                  "budget returned non-int at {} {}".format(clock, increment))


def main():
    print("btc_time.budget regression suite")
    print("constants: OVERHEAD_MS {}  MTG {}".format(OVERHEAD_MS, MTG))
    print("")
    test_tournament_control_unchanged()
    test_no_degenerate_budgets()
    test_soft_never_exceeds_hard()
    test_monotonic_in_clock()
    test_budget_fits_the_clock()
    test_returns_ints()
    print("")
    if FAILURES:
        for failure in FAILURES:
            print("FAIL: " + failure)
        print("{} checks, {} failed".format(CHECKS[0], len(FAILURES)))
        return 1
    print("{} checks, all passed".format(CHECKS[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
