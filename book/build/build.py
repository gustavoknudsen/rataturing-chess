"""
Write the Polyglot book.

Sources, in priority order, and nothing else:
  1. our Stockfish labels  (labels.tsv)
  2. Lichess cloud evals   (lichess_labels.tsv -- engine analysis, 11.3% of this
                            tree, about twice Cerebellum's reach)
  3. Cerebellum            (engine-derived, 5.9%, and only 0.64% of it unique
                            once Lichess is in)

The game-frequency books are deliberately excluded: they are built from human
game counts, disagree with engine analysis 22-48% of the time, and were the
whole quality risk in the earlier build.

    python build.py positions_full.tsv rataturing.bin

Output is a standard Polyglot .bin: 16-byte big-endian records, sorted by key,
so `chess.polyglot.open_reader` reads it with no custom code.
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
from collections import Counter

import chess
import chess.polyglot as pg

ENTRY = struct.Struct(">QHHI")
LEARN_OFFSET = 100_000          # so a signed centipawn score fits in uint32


def encode_move(board, move):
    """Polyglot move encoding: to | from<<6 | promo<<12.

    Castling is stored king-takes-rook (e1h1, not e1g1). python-chess reads both
    forms, but the king-takes-rook form is what the format specifies and what
    any other reader will expect, so write that.
    """
    from_sq, to_sq = move.from_square, move.to_square
    if board.is_castling(move):
        rook = board.rook_at_castling(move) if hasattr(board, "rook_at_castling") \
            else None
        if rook is None:
            # King moved two files; the rook is the corner one on that side.
            rank = chess.square_rank(from_sq)
            file_ = 7 if chess.square_file(to_sq) > chess.square_file(from_sq) else 0
            rook = chess.square(file_, rank)
        to_sq = rook
    promo = 0 if move.promotion is None else move.promotion - 1
    return (to_sq & 63) | ((from_sq & 63) << 6) | (promo << 12)


def load_labels(path):
    """fen -> (uci, score, depth). On a repeat, the deeper search wins."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            fen, uci = p[0], p[1]
            try:
                score, depth = int(p[2]), int(p[3])
            except ValueError:
                continue
            prev = out.get(fen)
            if prev is None or depth > prev[2]:
                out[fen] = (uci, score, depth)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positions", help="positions_full.tsv from expand.py")
    ap.add_argument("out")
    ap.add_argument("--labels", nargs="+",
                    default=[data("labels.tsv"), data("labels_d30.tsv"), data("labels_gap.tsv")],
                    help="one or more label files; on a repeated FEN the "
                         "deeper search wins, so a deepening pass goes in its "
                         "own file")
    ap.add_argument("--lichess", default=data("lichess_labels.tsv"))
    ap.add_argument("--candidates", default=data("candidates.tsv"))
    ap.add_argument("--cerebellum", default=data("Cerebellum3Merge.bin"))
    ap.add_argument("--max-move", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the top N positions by reach probability")
    a = ap.parse_args()

    # label.py resumes by skipping FENs already present, regardless of depth,
    # so a deeper re-label must be written to a separate file and merged here.
    labels = {}
    for path in a.labels:
        for fen, val in load_labels(path).items():
            prev = labels.get(fen)
            if prev is None or val[2] > prev[2]:
                labels[fen] = val
    # Polyglot keys by Zobrist hash, which carries no move number, so a label
    # we already searched can belong to a position whose FEN string differs only
    # by its counters. Keying labels by FEN alone silently drops those (measured:
    # 117 tree positions) and falls through to a weaker source -- one of them
    # wrote a -302 Lichess move where our own label scored +613. Keep the deeper
    # label when two FENs collapse to the same key.
    labels_z = {}
    for fen, val in labels.items():
        try:
            k = pg.zobrist_hash(chess.Board(fen))
        except ValueError:
            continue
        prev = labels_z.get(k)
        if prev is None or val[2] > prev[2]:
            labels_z[k] = val

    lichess = {}
    if a.lichess and os.path.exists(a.lichess):
        with open(a.lichess, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    try:
                        lichess[p[0]] = (p[1], int(p[2]))
                    except ValueError:
                        continue
    # Top move from the shallow MultiPV pass, used only where nothing deeper
    # exists. It is a real move, just searched to depth 14 rather than 22.
    cands = {}
    if a.candidates and os.path.exists(a.candidates):
        with open(a.candidates, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[1]:
                    # Keep the whole ranked list: the top move is the fallback,
                    # and membership is used as a sanity gate on Lichess below.
                    cands[p[0]] = p[1].split(",")
    print(f"  {len(labels):,} of our labels, {len(lichess):,} Lichess evals, "
          f"{len(cands):,} candidate lists")

    cere = None
    if a.cerebellum and os.path.exists(a.cerebellum):
        cere = pg.open_reader(a.cerebellum)

    rows = []
    with open(a.positions, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                rows.append((p[0], float(p[1])))
    rows.sort(key=lambda r: -r[1])
    if a.limit:
        rows = rows[:a.limit]
    print(f"  {len(rows):,} positions in the expansion")

    # key -> (encoded_move, score, prob, fullmove). Zobrist ignores the move
    # counters, so the same position at two move numbers collides on purpose;
    # keep whichever we reach more often.
    book = {}
    src = Counter()
    for fen, prob in rows:
        board = chess.Board(fen)
        if board.fullmove_number > a.max_move:
            continue
        lab = labels.get(fen) or labels_z.get(pg.zobrist_hash(board))
        if lab:
            uci, score, _ = lab
            source = "labels"
        elif fen in lichess:
            # Lichess cloud evals: engine analysis, and 2x Cerebellum's coverage
            # of this tree (11.3% vs 5.9%), so it sits ahead of it. No depth or
            # knodes threshold is applied -- both were measured to correlate
            # negatively with agreement against our own labels, because they
            # track position difficulty rather than eval quality.
            uci, score = lichess[fen]
            source = "lichess"
            # Sanity gate. A deeper search essentially never leaves our depth-14
            # top-8: measured, the depth-22 move was inside it in 631/631 cases.
            # So a Lichess move that is NOT in that list is not deeper insight,
            # it is an anomaly (stale, or a mis-scored line) -- and this is the
            # bucket that produced a 915cp blunder in verify. Fall back to our
            # own top move, which we searched ourselves.
            ranked = cands.get(fen)
            if ranked and uci not in ranked:
                uci, score, source = ranked[0], None, "candidates-d14"
        else:
            uci = score = source = None
            if cere is not None:
                try:
                    uci, source = cere.find(board).move.uci(), "cerebellum"
                except (IndexError, KeyError, ValueError):
                    pass
            if uci is None and fen in cands:
                # Last resort: the top move from the shallow MultiPV pass that
                # built the tree. Depth 14 picks the same move as depth 22 73.2%
                # of the time, and our engine reaches roughly depth 8-14 in its
                # 3.5s, so this is no worse than searching and saves the clock.
                # Ranked last because everything above it is a deeper search.
                uci, source = cands[fen][0], "candidates-d14"
            if uci is None:
                src["gap"] += 1
                continue

        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            src["bad-uci"] += 1
            continue
        if move not in board.legal_moves:
            src["illegal"] += 1
            continue

        key = pg.zobrist_hash(board)
        prev = book.get(key)
        if prev is not None and prev[2] >= prob:
            continue
        book[key] = (encode_move(board, move), score, prob, board.fullmove_number)
        src[source] += 1

    if cere is not None:
        cere.close()

    with open(a.out, "wb") as f:
        for key in sorted(book):
            enc, score, _, _ = book[key]
            # weight 1 throughout (one move per position, nothing to weigh).
            # learn carries the labelled score so a runtime sanity check is
            # possible; 0 means "no score", i.e. it came from Cerebellum.
            learn = 0 if score is None else max(0, min(0xFFFFFFFF,
                                                       score + LEARN_OFFSET))
            f.write(ENTRY.pack(key, enc, 1, learn))

    size = os.path.getsize(a.out)
    print(f"  wrote {len(book):,} entries -> {a.out} ({size/1e6:.2f} MB)")
    for k in sorted(src, key=lambda x: -src[x]):
        print(f"    {k:14s} {src[k]:,}")


if __name__ == "__main__":
    main()
