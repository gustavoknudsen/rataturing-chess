"""
Decide whether -- and where -- to threshold the Lichess evals.

The question is not "is a deep eval better than a shallow one" (within the DB,
depth barely moves the answer: 78.3% agreement with the position's most-searched
eval at depth 12 versus 77.4% at depth 30). The question is whether a DB eval is
a fair substitute for the depth-22 Stockfish label we would otherwise compute
ourselves.

So this compares the DB's move against OUR label on the positions where we have
both, bucketed by the DB entry's depth and knodes. A bucket that agrees with our
own labelling as often as our labelling agrees with itself is a bucket we can
take for free; one that does not is CPU we should spend.

    python pick_threshold.py
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import data  # noqa: E402


import argparse
from collections import defaultdict


def load_ours(path):
    """fen -> (uci, score, depth); deeper wins on repeats."""
    out = {}
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        return out
    with f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            try:
                depth = int(p[3])
            except ValueError:
                continue
            prev = out.get(p[0])
            if prev is None or depth > prev[2]:
                out[p[0]] = (p[1], p[2], depth)
    return out


def load_db(path):
    """fen -> (uci, cp, depth, knodes)."""
    out = {}
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        return out
    with f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            try:
                out[p[0]] = (p[1], p[2], int(p[3]), int(p[4]))
            except ValueError:
                continue
    return out


def table(rows, keyfn, buckets, label):
    agg = defaultdict(lambda: [0, 0])
    for fen, (ours, db) in rows.items():
        k = keyfn(db)
        b = next((b for b in buckets if k >= b), buckets[-1])
        a = agg[b]
        a[1] += 1
        a[0] += (ours[0] == db[0])
    print(f"\n  agreement with our depth-22 label, by {label}")
    print(f"  {'bucket':>14} {'agree':>8} {'n':>8}  {'cumulative n':>13}")
    cum = 0
    for b in buckets:
        s, t = agg[b]
        cum += t
        if t == 0:
            continue
        print(f"  {b:>14,} {s/t:>7.1%} {t:>8,} {cum:>13,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=data("labels.tsv"))
    ap.add_argument("--db", default=data("lichess_labels.tsv"))
    a = ap.parse_args()

    ours = load_ours(a.ours)
    db = load_db(a.db)
    both = {f: (ours[f], db[f]) for f in ours if f in db}
    print(f"  our labels: {len(ours):,}   DB entries: {len(db):,}   "
          f"overlap: {len(both):,}")
    if not both:
        print("  no overlap yet -- run the scan and some labelling first")
        return

    agree = sum(1 for o, d in both.values() if o[0] == d[0])
    print(f"  overall agreement: {agree/len(both):.1%}")
    table(both, lambda d: d[2], [40, 35, 30, 26, 24, 22, 20, 18, 16, 14, 12, 0],
          "DB depth")
    table(both, lambda d: d[3],
          [500_000, 200_000, 100_000, 50_000, 20_000, 10_000, 5_000, 1_000, 0],
          "DB knodes")

    # What a threshold would actually cost in coverage across the whole DB pull.
    print("\n  coverage cost of a knodes threshold, over the full DB pull")
    print(f"  {'min knodes':>12} {'kept':>10} {'share':>8}")
    for t in (0, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000):
        n = sum(1 for v in db.values() if v[3] >= t)
        print(f"  {t:>12,} {n:>10,} {n/max(1,len(db)):>7.1%}")


if __name__ == "__main__":
    main()
