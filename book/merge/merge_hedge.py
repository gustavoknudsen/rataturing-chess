"""
Merge the three hedge trees into one position list.

Each covers a different failure, and none of them subsumes the others:

  mined.tsv         every position Cerebellum answers to about move 6. Broad
                    and shallow -- insurance against an unfamiliar start.
  mined_deep.tsv    Cerebellum's own move plus moves two or more books endorse,
                    followed to move 20. Narrow and deep -- this is what
                    actually plays a run of book moves.
  hedge_positions.tsv  the book-vote expansion, including the 51 forced gap
                    lines for openings absent from the ladder pool.

Priority for the output ordering is shallowest-first, because a position nearer
the start is reached more often and, on a Zobrist collision, is the one worth
keeping.

    python merge_hedge.py hedge_all.tsv
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import data  # noqa: E402


import argparse
import os

import chess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    # Ordered worst-to-best by measured value per byte, so a later file wins the
    # shallowest-first tiebreak. Coverage of the 308 known curated starts:
    # exhaustive 13, book-vote 32, deep 129, and mined_k4 targets the move 9-13
    # band where all three fall to near zero.
    ap.add_argument("--inputs", nargs="+",
                    default=[data("mined.tsv"), data("hedge_positions.tsv"),
                             "mined_deep.tsv", "mined_k4.tsv"])
    ap.add_argument("--max-move", type=int, default=20)
    a = ap.parse_args()

    if a.max_move > 20:
        raise SystemExit("max-move above 20 is outside the permitted scope")

    best = {}
    per_file = {}
    for path in a.inputs:
        if not os.path.exists(path):
            print(f"  skip (missing): {path}")
            continue
        n = new = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                fen = p[0]
                if fen.count("/") != 7:
                    continue
                n += 1
                try:
                    board = chess.Board(fen)
                except ValueError:
                    continue
                if board.fullmove_number > a.max_move:
                    continue
                # Rank by how early the position is: a shallower position is
                # reached more often, whatever tree it came from.
                score = 1.0 / (board.fullmove_number + 1)
                if fen not in best:
                    new += 1
                if score > best.get(fen, 0.0):
                    best[fen] = score
        per_file[path] = (n, new)
        print(f"  {path:<24} {n:>8,} rows, {new:>8,} new")

    rows = sorted(best.items(), key=lambda kv: -kv[1])
    with open(a.out, "w", encoding="utf-8") as f:
        for fen, s in rows:
            f.write(f"{fen}\t{s:.6f}\n")
    print(f"\n  {len(rows):,} unique positions -> {a.out}")
    print(f"  at 16 B/entry that is {len(rows)*16/1e6:.2f} MB if all are answered")


if __name__ == "__main__":
    main()
