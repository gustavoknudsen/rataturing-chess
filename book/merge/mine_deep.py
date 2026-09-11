"""
Mine Cerebellum along plausible lines only -- narrow and deep.

The exhaustive mine (mine_cerebellum.py) doubles every ply, so it drowns in
breadth at about move 6 and never reaches the move cap. Depth has to come from
branching less, not from raising the cap.

Here a move is expanded only if it is plausible:
  * it is the move Cerebellum itself plays in that position, or
  * at least --min-books of the other source books endorse it,
and Cerebellum must answer the resulting position either way.

That is the same criterion that let the book-vote hedge reach move 16 on 35k
positions while the exhaustive mine needed 227k to reach move 6.

    python mine_deep.py mined_deep.tsv --max-move 20 --min-books 2

Writes as it goes, so an interrupted run keeps what it found.
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import book, data  # noqa: E402


import argparse
import glob
import heapq
import os
import time

import chess
import chess.polyglot as pg


def find_books(patterns, exclude="Cerebellum"):
    out = []
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                if os.path.getsize(p) % 16 or os.path.getsize(p) < 1000:
                    continue
            except OSError:
                continue
            if exclude in os.path.basename(p):
                continue
            out.append(p)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cerebellum", default=book("Cerebellum3Merge.bin"))
    ap.add_argument("--books", nargs="*", default=[
        r"../c bot/lichess-bot-master/engines/*.bin",
        r"../python bot/books/*.bin",
    ])
    ap.add_argument("--max-move", type=int, default=20)
    ap.add_argument("--min-books", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=0,
                    help="cap moves expanded per node, best-endorsed first; 0 "
                         "means no cap. Lower k trades width for depth, which "
                         "is what reaches the move 9-13 cut points.")
    ap.add_argument("--limit", type=int, default=400_000)
    a = ap.parse_args()

    if a.max_move > 20:
        raise SystemExit("max-move above 20 is outside the permitted scope")

    cere = pg.open_reader(a.cerebellum)
    others = [pg.open_reader(p) for p in find_books(a.books)]
    print(f"  Cerebellum + {len(others)} voting books, "
          f"<= move {a.max_move}", flush=True)

    def cere_move(board):
        try:
            return cere.find(board).move
        except (IndexError, KeyError, ValueError):
            return None

    def answered(board):
        try:
            cere.find(board)
            return True
        except (IndexError, KeyError, ValueError):
            return False

    def plausible(board):
        """Cerebellum's own move, plus any move enough other books endorse."""
        keep = []
        best = cere_move(board)
        if best is not None:
            keep.append(best)
        votes = {}
        for r in others:
            try:
                for e in r.find_all(board):
                    votes[e.move] = votes.get(e.move, 0) + 1
            except (IndexError, KeyError, ValueError):
                continue
        for mv, v in sorted(votes.items(), key=lambda kv: -kv[1]):
            if v >= a.min_books and mv not in keep:
                keep.append(mv)
        return keep[:a.top_k] if a.top_k else keep

    start = chess.Board()
    seen = {start.fen()}
    frontier = [start]
    out = open(a.out, "w", encoding="utf-8")
    out.write(f"{start.fen()}\t0\n")
    t0 = time.time()

    for ply in range(1, 40):
        nxt = []
        for board in frontier:
            for mv in plausible(board):
                if mv not in board.legal_moves:
                    continue
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
    cere.close()
    for r in others:
        r.close()
    print(f"\n  {len(seen):,} positions -> {a.out} "
          f"({len(seen)*16/1e6:.2f} MB as Polyglot entries)")


if __name__ == "__main__":
    main()
