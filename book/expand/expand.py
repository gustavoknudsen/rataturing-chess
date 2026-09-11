"""
Enumerate the positions the book must answer, with their reach probability.

Two things this does that a naive expansion does not:

1. **Both parities.** Colour is not fixed by the start position -- in the scraped
   ladder, teams played both sides of the same start. So every node in the tree
   is a position we might be asked to move in, not every other one.

2. **Follows our move, not the crowd's.** With --labels, at nodes where it is our
   turn we play the move Stockfish chose, because that is what the agent will
   actually play. Bootstrapping on the most-played reply instead makes the game
   leave the tree at our first book move and everything below it is never
   reached. Run this once without --labels, label the output, then re-run with
   --labels and label the new frontier, until it stops growing.

Opponent replies come from the observed games. Where a line runs off the observed
tree, --cerebellum lets a strong engine book stand in for the opponent, which is
the best available guess at what an engine opponent plays.

    python expand.py positions.tsv --threshold 1e-4
    python expand.py positions.tsv --threshold 1e-4 --labels labels.tsv

Writes `positions.tsv` (fen<TAB>prob) and `positions.txt` (fen only), both
sorted by reach probability descending -- so labelling that stops early has
still done the positions that matter most.
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import data  # noqa: E402


import argparse
import csv
import heapq
import os
import sys
from collections import defaultdict

import chess
import chess.polyglot as pg

MAX_MOVE = 20


def load_starts(path):
    """start_fens.txt: 'FEN ; opening ; games=N ; white_score=X'."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(";")]
            fen = parts[0]
            games = 1
            for p in parts[1:]:
                if p.startswith("games="):
                    games = int(p.split("=", 1)[1])
            out.append((fen, games))
    return out


def load_replies(path):
    """fen -> [(uci, count), ...] descending by count."""
    out = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 3:
                continue
            out[row[0]].append((row[1], int(row[2])))
    for k in out:
        out[k].sort(key=lambda x: -x[1])
    return out


def load_labels(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="positions.tsv to write")
    ap.add_argument("--starts", default=data("start_fens.txt"))
    ap.add_argument("--replies", default="out/continuations_all.csv")
    ap.add_argument("--labels", default=None,
                    help="labels.tsv; enables follow-our-move expansion")
    ap.add_argument("--cerebellum", default=data("Cerebellum3Merge.bin"),
                    help="stand-in opponent off the observed tree; '' to disable")
    ap.add_argument("--threshold", type=float, default=1e-4)
    ap.add_argument("--max-move", type=int, default=MAX_MOVE)
    a = ap.parse_args()

    starts = load_starts(a.starts)
    replies = load_replies(a.replies)
    labels = load_labels(a.labels)
    total_games = sum(g for _, g in starts)
    print(f"  {len(starts)} starts ({total_games:,} games), "
          f"{len(replies):,} positions with observed replies, "
          f"{len(labels):,} labels")

    cere = None
    if a.cerebellum and os.path.exists(a.cerebellum):
        cere = pg.open_reader(a.cerebellum)
        print(f"  opponent stand-in: {a.cerebellum}")

    # Two probabilities, doing two different jobs:
    #   cond   - probability of reaching this node GIVEN this start. Pruning uses
    #            it, so every start is covered to the same depth. We are handed
    #            exactly one start per game and they are near-equally likely, so
    #            a rare start deserves the same book depth as a common one.
    #   glob   - cond * the start's share of games. Ordering uses it, so if
    #            labelling is cut short it has done the highest-traffic nodes.
    # (-glob, tiebreak, fen, our_colour, cond)
    # Both colours are expanded from every start. Colour is not fixed by the
    # start position, and which side we are decides where we play our own book
    # move and where the opponent gets to branch -- so the two expansions reach
    # genuinely different positions. Deduplicating on the FEN alone would let
    # whichever colour arrived first swallow the other's subtree.
    heap = []
    seq = 0
    for fen, games in starts:
        share = games / total_games
        for colour in (chess.WHITE, chess.BLACK):
            heap.append((-share, seq, fen, colour, 1.0))
            seq += 1
    heapq.heapify(heap)

    best = {}          # fen -> max global reach probability, over both colours
    bcond = {}         # fen -> max conditional probability, for threshold sweeps
    seen = {}          # (fen, our_colour) -> best prob this subtree was expanded at
    pushed = 0
    off_tree = 0

    while heap:
        negg, _, fen, colour, cond = heapq.heappop(heap)
        glob = -negg
        if cond < a.threshold:
            continue
        board = chess.Board(fen)
        if board.fullmove_number > a.max_move:
            continue
        if seen.get((fen, colour), 0.0) >= cond:
            continue
        seen[(fen, colour)] = cond
        # Record every node: whichever colour we are, we may be asked to move
        # here, so it needs a book entry.
        if glob > best.get(fen, 0.0):
            best[fen] = glob
        if cond > bcond.get(fen, 0.0):
            bcond[fen] = cond

        # Whose turn it is comes from the position itself -- no ply bookkeeping.
        our_turn = board.turn == colour
        obs = replies.get(fen)

        def stand_in():
            """What a strong engine plays, when the observed games run out.

            Used for the opponent as well as for us. Once our own coverage is
            good, the opponent stepping off the observed tree becomes the main
            reason a game leaves book, and the best available prediction of an
            engine opponent's move is the move an engine picks. Our own label
            comes first because it is a deeper search than Cerebellum's entry;
            Cerebellum backs it up where we have not labelled.
            """
            lab = labels.get(fen)
            if lab:
                return [(lab, 1.0)]
            if cere is None:
                return []
            try:
                return [(cere.find(board).move.uci(), 1.0)]
            except (IndexError, KeyError, ValueError):
                return []

        if our_turn:
            # We play exactly one move, so this is a single child at full
            # probability -- never a distribution. Branching here would model a
            # player who plays every move at once, which is what floods the tree
            # with single-game lines and buries the deep nodes we actually reach.
            if labels.get(fen):
                children = [(labels[fen], 1.0)]
            elif obs:
                children = [(obs[0][0], 1.0)]      # bootstrap: most-played
            else:
                children = stand_in()
                off_tree += bool(children)
        else:
            # The opponent is the only source of branching.
            if obs:
                tot = sum(c for _, c in obs)
                children = [(u, c / tot) for u, c in obs]
            else:
                children = stand_in()
                off_tree += bool(children)

        for uci, share in children:
            ccond = cond * share
            if ccond < a.threshold:
                continue
            try:
                mv = chess.Move.from_uci(uci)
            except ValueError:
                continue
            if mv not in board.legal_moves:
                continue
            nb = board.copy(stack=False)
            nb.push(mv)
            if nb.fullmove_number > a.max_move:
                continue
            heapq.heappush(heap, (-glob * share, seq, nb.fen(), colour, ccond))
            seq += 1
            pushed += 1

        if len(best) % 20000 == 0:
            print(f"    {len(best):,} positions, heap {len(heap):,}",
                  file=sys.stderr)

    if cere is not None:
        cere.close()

    rows = sorted(best.items(), key=lambda kv: -kv[1])
    with open(a.out, "w", encoding="utf-8") as f:
        for fen, p in rows:
            f.write(f"{fen}\t{p:.9g}\t{bcond[fen]:.9g}\n")
    txt = os.path.splitext(a.out)[0] + ".txt"
    with open(txt, "w", encoding="utf-8") as f:
        for fen, _ in rows:
            f.write(fen + "\n")

    print(f"  {len(rows):,} positions (threshold {a.threshold:g}, "
          f"<= move {a.max_move})")
    print(f"  {pushed:,} edges pushed, {off_tree:,} nodes used the stand-in "
          f"opponent")
    print(f"  wrote {a.out} and {txt}")


if __name__ == "__main__":
    main()
