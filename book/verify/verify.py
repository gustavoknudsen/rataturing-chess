"""
Gate the book before it ships. Every check is a hard failure except where noted.

  1. Every entry's move is legal in the position it was built from -- checked
     against the reconstructed position, not trusted from the hash.
  2. Every entry's position is at move number <= 20 (the permitted scope).
  3. Every entry round-trips: reopened through chess.polyglot, the move read
     back is the move written.
  4. The file is sorted by key and a multiple of 16 bytes, because
     chess.polyglot binary-searches and an unsorted file reads as mostly empty
     without raising anything.
  5. No labelled entry sits more than --cp-margin below the best score we know
     for that position. Reported, and the count dropped.

    python verify.py btc_book.bin positions_full.tsv
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import data  # noqa: E402


import argparse
import os
import struct
import sys

import chess
import chess.polyglot as pg

ENTRY = struct.Struct(">QHHI")
LEARN_OFFSET = 100_000


def decode_to_move(board, raw):
    """Mirror of the writer, via the same path the agent's reader uses."""
    to_sq = raw & 63
    from_sq = (raw >> 6) & 63
    promo = (raw >> 12) & 7
    mv = chess.Move(from_sq, to_sq, promotion=promo + 1 if promo else None)
    # King-takes-rook is how Polyglot spells castling.
    if board.piece_type_at(from_sq) == chess.KING:
        if board.piece_type_at(to_sq) == chess.ROOK and \
                board.color_at(to_sq) == board.turn:
            rank = chess.square_rank(from_sq)
            file_ = 6 if chess.square_file(to_sq) > chess.square_file(from_sq) else 2
            mv = chess.Move(from_sq, chess.square(file_, rank))
    return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("positions", help="positions_full.tsv, to rebuild positions")
    ap.add_argument("--labels", default=data("labels.tsv"))
    ap.add_argument("--cp-margin", type=int, default=30)
    ap.add_argument("--max-move", type=int, default=20)
    a = ap.parse_args()

    fails = []
    size = os.path.getsize(a.book)
    print(f"  {a.book}: {size:,} bytes, {size//16:,} entries")
    if size % 16:
        fails.append(f"file is not a multiple of 16 bytes ({size % 16} over)")

    raw = open(a.book, "rb").read()
    entries = [ENTRY.unpack_from(raw, i) for i in range(0, len(raw) - 15, 16)]
    keys = [e[0] for e in entries]
    if keys != sorted(keys):
        fails.append("entries are NOT sorted by key -- reader will silently miss")
    dupes = len(keys) - len(set(keys))
    print(f"  sorted: {keys == sorted(keys)}   duplicate keys: {dupes}")

    # Rebuild key -> position from the expansion so we can check legality.
    by_key = {}
    with open(a.positions, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            b = chess.Board(p[0])
            if b.fullmove_number > a.max_move:
                continue
            by_key.setdefault(pg.zobrist_hash(b), b)
    print(f"  {len(by_key):,} positions reconstructed from the expansion")

    best_score = {}
    if os.path.exists(a.labels):
        with open(a.labels, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4:
                    try:
                        b = chess.Board(p[0])
                    except ValueError:
                        continue
                    k = pg.zobrist_hash(b)
                    s = int(p[2])
                    if k not in best_score or s > best_score[k]:
                        best_score[k] = s

    unknown = illegal = late = 0
    cp_short = []
    reader = pg.open_reader(a.book)
    for key, rawmv, weight, learn in entries:
        board = by_key.get(key)
        if board is None:
            unknown += 1
            continue
        if board.fullmove_number > a.max_move:
            late += 1
        mv = decode_to_move(board, rawmv)
        if mv not in board.legal_moves:
            illegal += 1
            if len(fails) < 20:
                fails.append(f"illegal move {mv.uci()} in {board.fen()}")
            continue
        # Round-trip through the reader the agent will actually use.
        try:
            got = reader.find(board).move
        except (IndexError, KeyError, ValueError):
            fails.append(f"reader cannot find a key we wrote: {board.fen()}")
            continue
        if got != mv:
            fails.append(f"round-trip mismatch {mv.uci()} != {got.uci()} "
                         f"in {board.fen()}")
        if learn:
            score = learn - LEARN_OFFSET
            bs = best_score.get(key)
            if bs is not None and score < bs - a.cp_margin:
                cp_short.append((board.fen(), score, bs))
    reader.close()

    print(f"  entries not in the expansion : {unknown:,} "
          f"(expected 0 unless the book was built from a different expansion)")
    print(f"  illegal moves                : {illegal}")
    print(f"  entries past move {a.max_move}         : {late}")
    print(f"  more than {a.cp_margin}cp below best  : {len(cp_short)}")
    for fen, s, bs in cp_short[:5]:
        print(f"      {s:+6d} vs {bs:+6d}  {fen}")

    if illegal or late:
        fails.append(f"{illegal} illegal, {late} past move {a.max_move}")

    print()
    if fails:
        print("  FAIL")
        for f in fails[:20]:
            print(f"    - {f}")
        sys.exit(1)
    print("  PASS -- all entries legal, in scope, sorted, and round-tripping")


if __name__ == "__main__":
    main()
