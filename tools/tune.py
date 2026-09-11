"""Sweep search constants against the arena. Run:

    python tune.py [games_per_candidate] [base_ms] [workers]

Every constant came from BTC, where it was tuned at C node counts. We search
roughly 4-5 plies shallower, and the depth-gated heuristics therefore cover a
much larger fraction of our tree than of BTC's.

**What this can and cannot show.** The 95% interval on a match of n games is
about +-100/sqrt(n) percent, so 400 games resolves roughly 35 elo. Individual
search constants are usually worth 5-20. This sweep therefore **cannot confirm
a good setting**; it can only reject a clearly bad one. Treat every result as
a filter that produces a shortlist, and confirm anything being shipped with a
separate, longer, single-worker run.

That asymmetry is the whole point after 2026-09-08: a bundle of changes that
cut nodes 31.6% measured -29 elo, so the job of this script is to catch that
kind of loss early, not to chase small gains it cannot see.

Each candidate runs as a parallel match (arena_par.py), and candidates run one
at a time so they do not compete for cores with each other.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Round 3, after the null-move regression. Two groups:
#
#   - the null-move parameters, because the adaptive reduction is the prime
#     suspect for the -29 elo and a milder setting may still beat flat R=2
#   - the reduction and pruning constants most exposed to our shallower tree
#
# LMR_HIST_DIV is here because BTC's 4096 was fitted against a continuation
# history term that is structurally always zero in the C (BTC_UPSTREAM_ISSUES
# item 6); ours reads the populated slots, so the divisor is very likely wrong.
CANDIDATES = [
    ("BTC_NULL_BASE_R", "2"),
    ("BTC_NULL_DEPTH_DIV", "6"),
    ("BTC_NULL_EVAL_MAX", "1"),
    ("BTC_LMR_HIST_DIV", "12288"),
    ("BTC_LMR_MIN_MOVES", "3"),
    ("BTC_LMR_DIV", "2.80"),
    ("BTC_RFP_MAX_DEPTH", "5"),
    ("BTC_QS_DELTA", "400"),
]


def run_match(name, value, games, base_ms, workers):
    """Returns (label, percent, detail). percent is None on failure."""
    label = f"{name}={value}"
    out = subprocess.run(
        [sys.executable, "arena_par.py", f".|{name}={value}", ".", str(games),
         str(base_ms), "50", str(workers)],
        capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        return label, None, out.stderr.strip()[-200:]
    for line in out.stdout.splitlines():
        if line.startswith("score "):
            return label, float(line.split()[1].rstrip("%")), line.strip()
    return label, None, out.stdout.strip()[-200:]


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    base_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    margin = 100.0 / (games ** 0.5)
    print(f"{len(CANDIDATES)} candidates, {games} games each at "
          f"{base_ms/1000:.0f}s+0.05s, {workers} workers per candidate")
    print(f"95% interval is about +-{margin:.1f}%, so this resolves roughly "
          f"{margin * 7:.0f} elo at best: a filter, not a verdict\n", flush=True)

    results = []
    for name, value in CANDIDATES:
        label, percent, detail = run_match(name, value, games, base_ms, workers)
        if percent is None:
            print(f"{label:28s} FAILED {detail}", flush=True)
            continue
        print(f"{label:28s} {detail}", flush=True)
        results.append((percent, label))

    print("\nranked (challenger = the changed setting):")
    for percent, label in sorted(results, reverse=True):
        edge = percent - 50.0
        if abs(edge) < margin:
            verdict = "within noise"
        elif edge > 0:
            verdict = "shortlist: confirm with a longer run"
        else:
            verdict = "worse, reject"
        print(f"  {percent:5.1f}%  {label:28s} {verdict}")


if __name__ == "__main__":
    main()
