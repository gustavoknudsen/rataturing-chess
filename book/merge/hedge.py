"""
The playoff hedge: a general book from the standard starting position.

The 308 ladder starts are the pool we have reply data for. A knockout may draw
a different or wider pool, and for a position we have never been given we have
no observed replies at all. This expands from the standard start instead, so
there is something in book either way.

Two differences from the main expansion:

  * No reply data exists here, so the opponent's candidate moves come from the
    source books -- a move must appear in at least two of them to be expanded,
    which filters out one book's idiosyncrasies.
  * Opening frequencies at large are nowhere near the 97.9% top-3 concentration
    we measure against ladder opponents, so coverage decays fast with depth.
    Stop at move 16; past that the entries stop being reached.

The ladder pool covers 34 families and misses a lot of mainstream theory, so the
gap lines below are forced into the tree rather than left to whichever move the
source books happen to rank first.

    python hedge.py hedge_positions.tsv --max-move 16
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import data  # noqa: E402


import argparse
import glob
import heapq
import os

import chess
import chess.polyglot as pg

# Mainstream openings absent from the 308-position ladder pool. Forced into the
# expansion so the hedge reaches them even when the books rank them second.
GAP_LINES = {
    "Ruy Lopez Berlin":      "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6",
    "Ruy Lopez Main":        "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "Scandinavian":          "e2e4 d7d5",
    "Alekhine":              "e2e4 g8f6",
    "Sicilian Rossolimo":    "e2e4 c7c5 g1f3 b8c6 f1b5",
    "Sicilian Moscow":       "e2e4 c7c5 g1f3 d7d6 f1b5",
    "Sicilian Alapin":       "e2e4 c7c5 c2c3",
    "Caro-Kann Panov":       "e2e4 c7c6 d2d4 d7d5 e4d5 c6d5 c2c4",
    "Benoni":                "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6",
    "Benko/Volga":           "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5",
    "KID Saemisch":          "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f3",
    "KID Fianchetto":        "d2d4 g8f6 c2c4 g7g6 g1f3 f8g7 g2g3",
    "KID Four Pawns":        "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f4",
    "Dutch Leningrad":       "d2d4 f7f5 g2g3 g8f6 f1g2 g7g6",
    "Dutch Classical":       "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6",
    "English Reversed Sic":  "c2c4 e7e5",
    "Anglo-Indian":          "c2c4 g8f6",
    "QGD Tarrasch":          "d2d4 d7d5 c2c4 e7e6 b1c3 c7c5",
    "QGD Ragozin":           "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 g1f3 f8b4",
    "Slav Exchange":         "d2d4 d7d5 c2c4 c7c6 c4d5 c6d5",
    "Trompowsky":            "d2d4 g8f6 c1g5",
    "Torre":                 "d2d4 g8f6 g1f3 e7e6 c1g5",
    "Colle":                 "d2d4 d7d5 g1f3 g8f6 e2e3",
    "Bogo-Indian":           "d2d4 g8f6 c2c4 e7e6 g1f3 f8b4",
    "Old Indian":            "d2d4 g8f6 c2c4 d7d6",
    "Vienna":                "e2e4 e7e5 b1c3",
    "Philidor":              "e2e4 e7e5 g1f3 d7d6",
    "Modern":                "e2e4 g7g6",
    "King's Gambit":         "e2e4 e7e5 f2f4",
    # Second pass. An audit of 32 mainstream openings against the built book
    # found only 11 answered: expanding on moves that appear in two or more
    # source books tracks the main lines and never turns into these, so they
    # have to be forced the same way the list above is.
    "Ruy Lopez Marshall":    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 e8g8 c2c3 d7d5",
    "Evans Gambit":          "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4",
    "Ponziani":              "e2e4 e7e5 g1f3 b8c6 c2c3",
    "Sicilian Grand Prix":   "e2e4 c7c5 b1c3 b8c6 f2f4",
    "Smith-Morra Gambit":    "e2e4 c7c5 d2d4 c5d4 c2c3",
    "French Exchange":       "e2e4 e7e6 d2d4 d7d5 e4d5 e6d5",
    "Caro-Kann Two Knights": "e2e4 c7c6 b1c3 d7d5 g1f3",
    "Caro-Kann Fantasy":     "e2e4 c7c6 d2d4 d7d5 f2f3",
    "Pirc Austrian":         "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f2f4",
    "Nimzowitsch Defence":   "e2e4 b8c6",
    "Grunfeld Exchange":     "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3",
    "Queen's Indian Petro":  "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 a2a3",
    "KID Averbakh":          "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 c1g5",
    "Semi-Tarrasch":         "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 g1f3 c7c5",
    "Budapest Gambit":       "d2d4 g8f6 c2c4 e7e5",
    "Albin Counter-Gambit":  "d2d4 d7d5 c2c4 e7e5",
    "Chigorin Defence":      "d2d4 d7d5 c2c4 b8c6",
    "Dutch Staunton":        "d2d4 f7f5 e2e4",
    "English Four Knights":  "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6",
    "King's Indian Attack":  "g1f3 d7d5 g2g3",
    "Bird's Opening":        "f2f4",
    "Larsen 1.b3":           "b2b3",
}


def find_books(patterns):
    out = []
    for pat in patterns:
        for p in glob.glob(pat):
            if os.path.getsize(p) % 16 or os.path.getsize(p) < 1000:
                continue          # corrupt, or the short-lines set
            if "Cerebellum" in os.path.basename(p):
                continue          # priority source, consulted separately
            out.append(p)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cerebellum", default=data("Cerebellum3Merge.bin"))
    ap.add_argument("--books", nargs="*", default=[
        r"../c bot/lichess-bot-master/engines/*.bin",
        r"../python bot/books/*.bin",
    ])
    ap.add_argument("--max-move", type=int, default=16)
    ap.add_argument("--top-replies", type=int, default=3)
    ap.add_argument("--min-books", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=1e-5)
    a = ap.parse_args()

    paths = find_books(a.books)
    readers = [pg.open_reader(p) for p in paths]
    print(f"  {len(readers)} source books for opponent candidates")
    cere = pg.open_reader(a.cerebellum) if os.path.exists(a.cerebellum) else None

    def candidates(board):
        """Moves appearing in >= min_books, most-endorsed first."""
        votes = {}
        for r in readers:
            seen = set()
            try:
                for e in r.find_all(board):
                    if e.move in seen:
                        continue
                    seen.add(e.move)
                    votes[e.move] = votes.get(e.move, 0) + 1
            except (IndexError, KeyError, ValueError):
                continue
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        return [m for m, v in ranked if v >= a.min_books][:a.top_replies]

    def our_move(board):
        if cere is not None:
            try:
                return cere.find(board).move
            except (IndexError, KeyError, ValueError):
                pass
        c = candidates(board)
        return c[0] if c else None

    best = {}
    seen = set()
    heap = []
    seq = 0

    def push(board, prob, colour):
        nonlocal seq
        heapq.heappush(heap, (-prob, seq, board.fen(), colour))
        seq += 1

    # Forced gap lines first, at full probability, so they are always present.
    forced = 0
    for name, line in GAP_LINES.items():
        board = chess.Board()
        ok = True
        for uci in line.split():
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                print(f"  ! {name}: illegal {uci}")
                ok = False
                break
            board.push(mv)
        if not ok:
            continue
        forced += 1
        for colour in (chess.WHITE, chess.BLACK):
            push(board, 1.0, colour)
    print(f"  {forced}/{len(GAP_LINES)} gap lines seeded")

    for colour in (chess.WHITE, chess.BLACK):
        push(chess.Board(), 1.0, colour)

    while heap:
        negp, _, fen, colour = heapq.heappop(heap)
        prob = -negp
        if prob < a.threshold:
            continue
        if (fen, colour) in seen:
            continue
        seen.add((fen, colour))
        board = chess.Board(fen)
        if board.fullmove_number > a.max_move:
            continue
        best[fen] = max(best.get(fen, 0.0), prob)

        if board.turn == colour:
            mv = our_move(board)
            kids = [(mv, 1.0)] if mv else []
        else:
            cand = candidates(board)
            kids = [(m, 1.0 / len(cand)) for m in cand] if cand else []

        for mv, share in kids:
            if mv not in board.legal_moves:
                continue
            nb = board.copy(stack=False)
            nb.push(mv)
            if nb.fullmove_number > a.max_move:
                continue
            push(nb, prob * share, colour)

    for r in readers:
        r.close()
    if cere is not None:
        cere.close()

    rows = sorted(best.items(), key=lambda kv: -kv[1])
    with open(a.out, "w", encoding="utf-8") as f:
        for fen, p in rows:
            f.write(f"{fen}\t{p:.9g}\t{p:.9g}\n")
    txt = os.path.splitext(a.out)[0] + ".txt"
    with open(txt, "w", encoding="utf-8") as f:
        for fen, _ in rows:
            f.write(fen + "\n")
    print(f"  {len(rows):,} positions (<= move {a.max_move}) -> {a.out}")


if __name__ == "__main__":
    main()
