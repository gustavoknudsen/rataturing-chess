"""Check what btc_time.budget does at time controls other than 120 + 0.5.

The tuned constants assume the tournament control. If the final changes it,
the question is not which values are optimal but whether the function still
returns something sane, so this sweeps the plausible controls and simulates a
whole game at each, flagging degenerate budgets rather than scoring them.

    .venv/Scripts/python.exe finals_day/tc_sweep.py

A budget is called degenerate when the engine is handed under 50 ms to think
while more than 2 s remains on its clock. That is not a weak move, it is a
depth 1 move with time to spare, and it would happen on every move for the
rest of the game.
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import btc_time

# (label, base ms, increment ms)
CONTROLS = [
    ("120 + 0.5  (tournament)", 120000, 500),
    ("120 + 0.1", 120000, 100),
    ("60 + 0.5", 60000, 500),
    ("60 + 0.2", 60000, 200),
    ("30 + 0.5", 30000, 500),
    ("30 + 0.2", 30000, 200),
    ("30 + 0.1", 30000, 100),
    ("10 + 0.1", 10000, 100),
    ("10 + 0", 10000, 0),
    ("5 + 0.05", 5000, 50),
    ("180 + 2", 180000, 2000),
    ("600 + 0  (classical)", 600000, 0),
]

DEGENERATE_MS = 50
SPARE_MS = 2000
MAX_PLIES = 300


def simulate(base_ms, inc_ms):
    """Play a game against the budget function, spending the soft budget.

    Returns the move number where budgets first go degenerate, the number of
    moves affected, and the move number where the clock would run out.
    """
    clock = float(base_ms)
    first_bad = 0
    bad_moves = 0
    flagged = 0
    flat_out = 0
    for move in range(1, MAX_PLIES + 1):
        if clock <= 0:
            flat_out = move
            break
        soft, hard = btc_time.budget(int(clock), move, inc_ms)
        if soft > hard:
            flagged += 1
        if soft < DEGENERATE_MS and clock > SPARE_MS:
            bad_moves += 1
            if not first_bad:
                first_bad = move
        # Spend the soft budget, which is what the search targets, plus the
        # real per-move cost the constants are calibrated against.
        clock -= soft + btc_time.OVERHEAD_MS
        clock += inc_ms
    return first_bad, bad_moves, flagged, flat_out


def main():
    print("btc_time constants: OVERHEAD_MS {}  MTG {}".format(
        btc_time.OVERHEAD_MS, btc_time.MTG))
    print("reserve term OVERHEAD_MS * (2 + MTG) = {} ms".format(
        btc_time.OVERHEAD_MS * (2 + btc_time.MTG)))
    print("")

    header = "{:<24} {:>10} {:>10} {:>10} {:>9} {:>8}".format(
        "control", "soft mv20", "hard mv20", "first bad", "bad moves", "soft>hard")
    print(header)
    print("-" * len(header))

    for label, base, inc in CONTROLS:
        soft, hard = btc_time.budget(base, 20, inc)
        first_bad, bad_moves, flagged, flat_out = simulate(base, inc)
        note = str(first_bad) if first_bad else "-"
        if flat_out:
            note += " (flag mv {})".format(flat_out)
        print("{:<24} {:>10} {:>10} {:>10} {:>9} {:>8}".format(
            label, soft, hard, note, bad_moves, flagged))

    print("")
    print("budget at a falling clock, increment 100 ms, move 30")
    print("{:>12} {:>8} {:>8}".format("clock ms", "soft", "hard"))
    for clock in (60000, 30000, 20000, 15000, 14000, 13000, 10000,
                  5000, 3000, 1500, 900, 400):
        soft, hard = btc_time.budget(clock, 30, 100)
        print("{:>12} {:>8} {:>8}".format(clock, soft, hard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
