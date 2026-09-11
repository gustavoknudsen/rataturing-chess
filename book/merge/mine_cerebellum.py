"""
Mine the opening tree Cerebellum actually covers.

A Polyglot file is keyed by Zobrist hash and stores no positions, so its contents
cannot be enumerated directly. But walking forward from the standard start and
probing every legal move recovers exactly the subtree it answers -- with real
positions, so the move cap and legality checks still apply.

This is the cheapest coverage available anywhere in the build: every position it
finds comes with an engine-derived move at zero Stockfish cost. It is used for
the knockout hedge, where we have no reply data and breadth is the whole point.

    python mine_cerebellum.py mined.tsv --max-move 16 --limit 500000

Writes as it goes (fen<TAB>ply), so an interrupted run keeps everything found so
far, and re-running resumes from the deepest complete ply.
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import book, data  # noqa: E402


import argparse
import os
import time

import chess
import chess.polyglot as pg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--book", default=book("Cerebellum3Merge.bin"))
    ap.add_argument("--max-move", type=int, default=16)
    ap.add_argument("--limit", type=int, default=500_000)
    ap.add_argument("--max-ply", type=int, default=24)
    a = ap.parse_args()

    reader = pg.open_reader(a.book)

    def answered(board):
        try:
            reader.find(board)
            return True
        except (IndexError, KeyError, ValueError):
            return False

    seen = set()
    start = chess.Board()
    seen.add(start.fen())
    frontier = [start]
    out = open(a.out, "w", encoding="utf-8")
    out.write(f"{start.fen()}\t0\n")
    t0 = time.time()

    for ply in range(1, a.max_ply + 1):
        nxt = []
        for board in frontier:
            # push/pop rather than copying the board: the copy dominates the
            # cost at this scale and the probe itself is an mmap read.
            for mv in list(board.legal_moves):
                board.push(mv)
                try:
                    if board.fullmove_number <= a.max_move:
                        fen = board.fen()
                        if fen not in seen and answered(board):
                            seen.add(fen)
                            nxt.append(board.copy(stack=False))
                            out.write(f"{fen}\t{ply}\n")
                finally:
                    board.pop()
            if len(seen) >= a.limit:
                break
        out.flush()
        print(f"  ply {ply:2d}: +{len(nxt):,}  total {len(seen):,}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        frontier = nxt
        if not nxt or len(seen) >= a.limit:
            break

    out.close()
    reader.close()
    print(f"\n  {len(seen):,} positions -> {a.out} "
          f"({len(seen)*16/1e6:.2f} MB as Polyglot entries)")


if __name__ == "__main__":
    main()
