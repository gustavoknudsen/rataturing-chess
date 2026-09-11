"""
Build progress: what is running, how fast, how much longer.

    python progress.py            one snapshot (samples rates for 20s)
    python progress.py --watch    refresh until you Ctrl-C
    python progress.py --quick    no rate sample, counts only

Everything is measured against the current wide tree (wide.tsv), which is the
set of positions the book must answer. Rates are sampled live because
throughput swings with whatever SPRT or training run is sharing the CPU.
"""

import argparse
import os
import time

import chess.polyglot as pg

TREE = "wide.tsv"


def count(path):
    if not os.path.exists(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def fmt_eta(seconds):
    if seconds is None or seconds <= 0 or seconds != seconds:
        return "-"
    if seconds > 86400 * 2:
        return f"{seconds/86400:.1f}d"
    if seconds > 3600:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/60:.0f}m"


def bar(frac):
    n = max(0, min(40, int(frac * 40)))
    return "#" * n + "." * (40 - n)


def snapshot(sample):
    tree = count(TREE)
    jobs = [
        # name, file being written, target, note
        ("candidates", "candidates.tsv", tree,
         "opponent replies -> tree width + a depth-14 move"),
        ("deep labels", "labels.tsv", tree,
         "depth-22 moves; pure quality upgrade"),
        ("starts @ d30", "labels_d30.tsv", 198,
         "the 308 starts, played in every game"),
        ("cerebellum mine", "mined.tsv", 500_000,
         "free engine positions for the hedge"),
    ]
    before = {n: count(f) for n, f, _, _ in jobs}
    if sample > 0:
        time.sleep(sample)

    for name, path, target, note in jobs:
        done = count(path)
        rate = (done - before[name]) / sample if sample else None
        frac = done / target if target else 0
        left = max(0, target - done)
        state = "idle" if (rate is not None and rate <= 0) else \
                ("running" if rate else "?")
        print(f"  {name:<16}[{bar(frac)}] {frac:5.1%}  {state}")
        print(f"  {'':<16} {done:>8,} / {target:,}   {note}")
        if rate:
            print(f"  {'':<16} {rate:>8.2f}/s   eta {fmt_eta(left/rate)}")
        print()

    # The tree still growing means the candidate loop has not converged.
    front = count("need_candidates.txt")
    print(f"  tree {tree:,} positions (ceiling ~72,700), "
          f"frontier {front:,} {'-- CONVERGED' if front == 0 else ''}")

    for book in ("rataturing.bin", "rataturing_hedge.bin"):
        if not os.path.exists(book):
            continue
        sz = os.path.getsize(book)
        try:
            with pg.open_reader(book) as r:
                pass
            ok = "ok"
        except Exception:
            ok = "UNREADABLE"
        print(f"  {book:<22} {sz//16:>8,} entries  {sz/1e6:5.2f} MB  {ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sample", type=int, default=20)
    a = ap.parse_args()
    secs = 0 if a.quick else a.sample

    if not a.watch:
        snapshot(secs)
        return
    try:
        while True:
            print("\x1b[2J\x1b[H", end="")
            print(f"  {time.strftime('%H:%M:%S')}\n")
            snapshot(max(secs, 5))
            print("\n  Ctrl-C to stop watching (jobs keep running)")
    except KeyboardInterrupt:
        print("\n  stopped watching; jobs continue")


if __name__ == "__main__":
    main()
