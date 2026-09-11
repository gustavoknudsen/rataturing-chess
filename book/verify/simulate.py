"""
Acceptance test: how many moves does the book actually play, and what clock does
that buy?

Replays games against the observed reply distribution. From each start, weighted
by how often that start was given, we play our book move and the opponent answers
by sampling their observed replies by count. A game leaves book the first time we
are asked a position the book does not answer.

Both colours are simulated from every start, because either can be handed the
same position.

    python simulate.py btc_book.bin

Reports expected book moves per game, the share of games with none, the tail
shares, and the clock that buys at --seconds-per-move.
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import data  # noqa: E402


import argparse
import csv
import os
import random
from collections import Counter, defaultdict

import chess
import chess.polyglot as pg

MAX_BOOK_MOVE = 20


def load_starts(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(";")]
            games = 1
            for p in parts[1:]:
                if p.startswith("games="):
                    games = int(p.split("=", 1)[1])
            out.append((parts[0], games))
    return out


def load_replies(path):
    out = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 3:
                continue
            out[row[0]].append((row[1], int(row[2])))
    return out


def probe(reader, board):
    """Exactly what the agent will do: scope guard, then legality guard."""
    if board.fullmove_number > MAX_BOOK_MOVE:
        return None
    try:
        entry = reader.find(board)
    except (IndexError, KeyError, ValueError):
        return None
    return entry.move if entry.move in board.legal_moves else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("--starts", default=data("start_fens.txt"))
    ap.add_argument("--replies", default="out/continuations_all.csv")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--seconds-per-move", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--off-tree", choices=("stop", "engine"), default="stop",
                    help="what the opponent does once the observed games run "
                         "out: 'stop' ends the game (conservative, the headline "
                         "number); 'engine' has them play our book's move for "
                         "their side, which is the optimistic bound")
    a = ap.parse_args()

    starts = load_starts(a.starts)
    replies = load_replies(a.replies)
    rng = random.Random(a.seed)
    reader = pg.open_reader(a.book)

    weights = [g for _, g in starts]
    counts = []
    left_at = Counter()
    why = Counter()
    for _ in range(a.games):
        fen, _ = rng.choices(starts, weights=weights, k=1)[0]
        our_colour = rng.choice([chess.WHITE, chess.BLACK])
        board = chess.Board(fen)
        n = 0
        while True:
            if board.is_game_over():
                break
            if board.turn == our_colour:
                mv = probe(reader, board)
                if mv is None:
                    left_at[board.fullmove_number] += 1
                    why["our book had no move"] += 1
                    break
                n += 1
            else:
                obs = replies.get(board.fen())
                if not obs:
                    # Opponent off the observed tree. Conservatively we stop:
                    # inventing their reply from the same book that built the
                    # tree would make the test agree with itself.
                    if a.off_tree == "engine":
                        mv = probe(reader, board)
                        if mv is not None:
                            board.push(mv)
                            continue
                    left_at[board.fullmove_number] += 1
                    why["opponent left the observed tree"] += 1
                    break
                ucis = [u for u, _ in obs]
                cnts = [c for _, c in obs]
                mv = chess.Move.from_uci(rng.choices(ucis, weights=cnts, k=1)[0])
                if mv not in board.legal_moves:
                    break
            board.push(mv)
        counts.append(n)

    reader.close()
    g = len(counts)
    mean = sum(counts) / g
    print(f"  {os.path.basename(a.book)}  ({g:,} simulated games)")
    print(f"  expected book moves per game : {mean:.2f}")
    print(f"  games with no book move      : {sum(1 for c in counts if c == 0)/g:.1%}")
    for k in (3, 5, 8, 12):
        print(f"  games with >= {k:<2d} book moves   : "
              f"{sum(1 for c in counts if c >= k)/g:.1%}")
    print(f"  clock saved @ {a.seconds_per_move}s/move  : "
          f"{mean*a.seconds_per_move:.1f}s")
    print("  why the book ended:")
    for k, v in why.most_common():
        print(f"      {k:32s} {v/g:6.1%}")
    med = sorted(left_at.elements())
    if med:
        print(f"  median move when book ends   : {med[len(med)//2]}")


if __name__ == "__main__":
    main()
