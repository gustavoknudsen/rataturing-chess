"""Sweep search constants against the arena. Run:

    python tune.py [games_per_candidate] [workers] [base_ms]

Every constant came from BTC, where it was tuned at C node counts. We search
roughly 4-5 plies shallower, and the depth-gated heuristics (LMP, futility,
razoring) therefore cover a much larger fraction of our tree than of BTC's, so
the prior is that our pruning and reductions are too aggressive.

What this can and cannot show. The 95% interval on a match of n games is about
+-100/sqrt(n) percent, so 24 games resolves nothing below ~140 elo and even 200
games only resolves ~45 elo. Search constants are usually worth 5-20 elo, which
needs thousands of games. Treat every result here as a coarse filter: it rejects
settings that are clearly bad and shortlists ones worth a longer confirmation
run. Do not ship a change on this evidence alone.

Matches run in parallel because the platform's one-core limit applies to the
agent, not to this machine. Note the tradeoff: concurrent games share cores, so
time management is measured less faithfully. Confirmation runs for anything
being shipped should use one worker.
"""

import concurrent.futures
import os
import subprocess
import sys

# Round 1 (11 candidates, 120 games each) showed the shallow search wants MORE
# aggressive pruning, not less: RFP margin 130 scored 59.6% (+67 elo, interval
# excluding zero) while 220 scored 44.6%, and less LMR reduction scored 43.8%.
# Round 2 pushes that direction and tests the combination.
CANDIDATES = [
    ("BTC_RFP_MARGIN", "130"),
    ("BTC_RFP_MARGIN", "100"),
    ("BTC_RFP_MARGIN", "150"),
    ("BTC_NULL_R", "3"),
    ("BTC_FUT_MAX_DEPTH", "4"),
    ("BTC_LMR_DIV", "2.20"),
    ("BTC_RFP_MAX_DEPTH", "4"),
]

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_match(name, value, games, base_ms):
    """Returns (label, percent, summary) or (label, None, error)."""
    label = f"{name}={value}"
    out = subprocess.run(
        [sys.executable, "arena_ab.py", f".|{name}={value}", ".", str(games),
         str(base_ms), "50"],
        capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        return label, None, out.stderr.strip()[-200:]
    score_line = None
    for line in out.stdout.splitlines():
        if line.startswith("score "):
            score_line = line.strip()
    if score_line is None:
        return label, None, out.stdout.strip()[-200:]
    percent = float(score_line.split()[1].rstrip("%"))
    return label, percent, score_line


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    base_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 3000

    margin = 100.0 / (games ** 0.5)
    print(f"{len(CANDIDATES)} candidates, {games} games each at "
          f"{base_ms/1000:.0f}s+0.05s, {workers} parallel workers")
    print(f"95% interval is about +-{margin:.1f}%, so this resolves roughly "
          f"{margin * 7:.0f} elo at best: a filter, not a verdict\n", flush=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_match, name, value, games, base_ms)
                   for name, value in CANDIDATES]
        for future in concurrent.futures.as_completed(futures):
            label, percent, detail = future.result()
            if percent is None:
                print(f"{label:28s} FAILED {detail}", flush=True)
                continue
            print(f"{label:28s} {detail}", flush=True)
            results.append((percent, label))

    print("\nranked:")
    for percent, label in sorted(results, reverse=True):
        edge = percent - 50.0
        if abs(edge) < margin:
            verdict = "within noise"
        elif edge > 0:
            verdict = "shortlist: confirm with a long single-worker run"
        else:
            verdict = "worse"
        print(f"  {percent:5.1f}%  {label:28s} {verdict}")


if __name__ == "__main__":
    main()
