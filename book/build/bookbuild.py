"""Build a one-move-per-position Polyglot book for a fixed set of start
positions.

    python bookbuild.py starts.txt out.bin [--replies replies.tsv]
                        [--max-move 20] [--budget-mb 20]

`starts.txt`  one FEN per line.
`replies.tsv` optional, `fen<TAB>uci<TAB>count`  -  the opponent replies actually
              observed. With it, expansion follows reach probability. Without
              it, every legal reply is expanded, which is far more expensive and
              mostly wasted.

Sources are consulted in priority order: the first book that answers a position
wins. Cerebellum is engine-derived and already one move per position, so it goes
first; the rest are game-statistics books and only fill gaps.

Output is a standard Polyglot .bin  -  16-byte big-endian records sorted by key  - 
so `chess.polyglot.open_reader` in the base image reads it with no custom code.
"""
import argparse
import os
import struct
import sys
from collections import defaultdict

import chess
import chess.polyglot as pg

# Priority order. Cerebellum first: it is the only engine-derived source here,
# and it is already one move per position (1.01 moves/position over 11.1M).
DEFAULT_SOURCES = [
    "Cerebellum3Merge.bin",
    "codekiddy.bin",
    "rodent.bin",
    "KomodoVariety.bin",
    "Book.bin",
    "final-book.bin",
    "DCbook_large.bin",
    "komodo.bin",
    "Titans.bin",
    "gm2600.bin",
]

ENTRY = struct.Struct(">QHHI")


def load(paths):
    readers = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  skip (missing): {p}", file=sys.stderr)
            continue
        if os.path.getsize(p) % 16:
            print(f"  skip (not a multiple of 16 bytes, corrupt): {p}",
                  file=sys.stderr)
            continue
        readers.append((os.path.basename(p), pg.open_reader(p)))
    return readers


def best_move(readers, board):
    """First source that answers wins. Returns (move, source) or (None, None)."""
    for name, r in readers:
        e = list(r.find_all(board))
        if e:
            return max(e, key=lambda x: x.weight).move, name
    return None, None


def read_replies(path):
    """fen -> [(uci, count), ...] sorted by count descending."""
    out = defaultdict(list)
    if not path:
        return out
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            out[parts[0]].append((parts[1], int(parts[2])))
    for k in out:
        out[k].sort(key=lambda x: -x[1])
    return out


def build(starts, readers, replies, max_move, budget_entries):
    """Breadth-first by reach probability. Returns {key: move}."""
    book = {}
    seen = set()
    # (probability, board)  -  a simple priority frontier, highest reach first
    frontier = [(1.0, chess.Board(f)) for f in starts]
    stats = defaultdict(int)
    while frontier and len(book) < budget_entries:
        frontier.sort(key=lambda x: -x[0])
        prob, board = frontier.pop(0)
        if board.fullmove_number > max_move:
            continue
        key = pg.zobrist_hash(board)
        if key in seen:
            continue
        seen.add(key)
        mv, src = best_move(readers, board)
        if mv is None:
            stats["gap"] += 1
            continue
        book[key] = mv
        stats[src] += 1
        nb = board.copy()
        nb.push(mv)
        if nb.fullmove_number > max_move:
            continue
        obs = replies.get(nb.fen())
        if obs:
            total = sum(c for _, c in obs)
            for uci, c in obs:
                try:
                    m = chess.Move.from_uci(uci)
                except ValueError:
                    continue
                if m not in nb.legal_moves:
                    continue
                child = nb.copy()
                child.push(m)
                frontier.append((prob * c / total, child))
        else:
            # no observed data: follow what the books think is playable
            cand = []
            for name, r in readers:
                cand += [x.move for x in r.find_all(nb)]
            if not cand:
                cand = list(nb.legal_moves)[:3]
            uniq = list(dict.fromkeys(cand))[:3]
            for m in uniq:
                child = nb.copy()
                child.push(m)
                frontier.append((prob / len(uniq), child))
    return book, stats


def write_polyglot(book, path):
    """Sorted by key, weight 1, learn 0. python-chess bisects, so order matters."""
    with open(path, "wb") as f:
        for key in sorted(book):
            mv = book[key]
            promo = 0 if mv.promotion is None else mv.promotion - 1
            enc = (mv.to_square & 63) | ((mv.from_square & 63) << 6) \
                | (promo << 12)
            f.write(ENTRY.pack(key, enc, 1, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("starts")
    ap.add_argument("out")
    ap.add_argument("--replies", default=None)
    ap.add_argument("--max-move", type=int, default=20)
    ap.add_argument("--budget-mb", type=float, default=20.0)
    ap.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES)
    a = ap.parse_args()

    starts = [l.strip() for l in open(a.starts) if l.strip()]
    readers = load(a.sources)
    if not readers:
        raise SystemExit("no readable source books")
    replies = read_replies(a.replies)
    budget = int(a.budget_mb * 1e6 // 16)
    print(f"  {len(starts)} starts, {len(readers)} sources, "
          f"budget {budget:,} entries")

    book, stats = build(starts, readers, replies, a.max_move, budget)
    write_polyglot(book, a.out)

    for name, r in readers:
        r.close()
    print(f"  wrote {len(book):,} positions -> {os.path.getsize(a.out)/1e6:.2f} MB")
    for k in sorted(stats, key=lambda x: -stats[x]):
        print(f"    {k:24s} {stats[k]:,}")

    # verify it round-trips through the reader the agent will use
    with pg.open_reader(a.out) as r:
        b = chess.Board(starts[0])
        e = list(r.find_all(b))
        print(f"  verify: first start position -> "
              f"{e[0].move.uci() if e else 'NOT COVERED'}")


if __name__ == "__main__":
    main()
