"""
Drive the tree to closure, unattended.

Expanding needs candidates, and candidates grow the tree, which needs more
candidates. This alternates the two until the frontier is empty, so the process
converges instead of chasing a moving target by hand.

    python build_tree.py --threshold 3e-3 --max-move 16 --workers 5

Every stage is resumable: candidates.tsv is append-only and skips what it has,
so killing this at any point loses nothing and re-running continues.
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import data, engine  # noqa: E402


import argparse
import os
import subprocess
import sys
import time

PY = sys.executable


def run(cmd):
    print(f"  $ {' '.join(cmd[1:])}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    for line in out.splitlines():
        if line.strip():
            print(f"    {line}", flush=True)
    return out


def count(path):
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", default="3e-3")
    ap.add_argument("--max-move", default="16")
    ap.add_argument("--workers", default="5")
    ap.add_argument("--engine", default=engine("stockfish.exe"))
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--tree", default=data("wide.tsv"))
    a = ap.parse_args()

    for rnd in range(1, a.rounds + 1):
        print(f"\n=== round {rnd} ===", flush=True)
        out = run([PY, "expand_wide.py", a.tree,
                   "--candidates", "candidates.tsv",
                   "--threshold", a.threshold, "--max-move", a.max_move])
        need = count("need_candidates.txt")
        tree = count(a.tree)
        print(f"  tree {tree:,} positions, frontier {need:,}", flush=True)
        if need == 0:
            print("  converged: every opponent node has candidates", flush=True)
            break

        # Order the frontier by reach probability so a cut-short run has still
        # done the positions we are most likely to reach.
        prob = {}
        with open(a.tree, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2:
                    prob[p[0]] = float(p[1])
        frontier = [l.strip() for l in
                    open("need_candidates.txt", encoding="utf-8") if l.strip()]
        frontier.sort(key=lambda x: -prob.get(x, 0.0))
        with open("need_candidates_ordered.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(frontier) + "\n")

        t0 = time.time()
        run([PY, "candidates.py", "need_candidates_ordered.txt", "candidates.tsv",
             "--engine", a.engine, "--multipv", "8", "--depth", "14",
             "--workers", a.workers, "--hash-mb", "32"])
        print(f"  round took {(time.time()-t0)/60:.1f} min, "
              f"candidates now {count('candidates.tsv'):,}", flush=True)

    print("\n=== done ===", flush=True)
    print(f"  tree: {count(a.tree):,} positions", flush=True)
    print(f"  candidates: {count('candidates.tsv'):,}", flush=True)


if __name__ == "__main__":
    main()
