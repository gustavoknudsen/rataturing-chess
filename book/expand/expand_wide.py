"""
Width-first expansion: a bushy, shallow tree instead of a deep thread.

Why this replaces the depth-first expansion
-------------------------------------------
Held-out games say the opponent's actual move is in our observed-reply set for
1 reply 84.4% of the time, 2 replies 48.4%, 3 replies 24.2%, 4 replies 10.8%,
5 replies 4.0%. A book threaded to move 20 is therefore reached by well under
1% of games. Depth past three or four opponent replies is close to worthless.

Width cannot come from the game data: 90% of positions in it were seen in
exactly one game, so there is no second reply to add. It has to come from the
engine -- the top-N candidate moves at each opponent node.

The opponent model
------------------
`--rank-probs` is P(opponent plays the engine's rank-k move), measured by
sampling real opponent decisions from held-out games and asking Stockfish for
its top 10. Reach probability is the product of these along a line, and pruning
on it is what makes the tree bushy near the root and narrow deep -- the shape
falls out of the model rather than being imposed by a depth cap. `--max-move`
is only a backstop, and the rules cap it at 20 regardless.

Candidate moves come from `--candidates`, a TSV of `fen<TAB>uci,uci,uci,...`
produced by a cheap shallow MultiPV pass. Where a position has observed replies
those are merged in first: real evidence outranks the engine's guess.

    python expand_wide.py wide_positions.tsv --candidates cands.tsv
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
from collections import defaultdict

import chess
import chess.polyglot as pg

# P(the opponent plays Stockfish's rank-k move), measured by probing 400 real
# opponent decisions from held-out games at MultiPV 10, depth 18:
#
#   top-1 44.5%   top-2 61.5%   top-3 68.5%   top-4 74.5%   top-5 78.8%
#   top-6 81.2%   top-8 86.5%   top-10 90.0%   beyond top-10 10.0%
#
# Top-1 at 44.5% independently matches the 46.4% top-1 share measured at
# positions where we have 50+ games, so the model is not an artefact of either
# method. It is also the whole argument for width: following only the engine's
# first choice for the opponent is wrong more than half the time.
#
# Values past rank 5 are within noise of each other at n=400, so they are
# smoothed monotone; pruning kills those branches quickly either way. The ~10%
# beyond rank 10 is mass we accept losing.
# P(opponent plays the rank-k move) under the merged ordering this file actually
# uses: observed replies by descending count, then engine candidates. Measured
# over 12,033 real replies at tree nodes seen in 50+ games -- the 50-game floor
# matters, because positions seen once contribute a single reply at rank 1 and
# inflate it to 0.677. Rank 1 lands at 0.464, independently matching the 46.4%
# measured earlier by a different method.
#
# The previous vector [0.445, 0.170, 0.070, 0.060, 0.043, 0.030, 0.025, 0.020]
# came from engine-rank MultiPV probing and underweighted ranks 2-3 badly
# (0.240 against a measured 0.329), so the threshold pruned the opponent's most
# likely replies. Correcting it and extending to 16 moved the book from 3.59 to
# 3.64 expected book moves per game.
DEFAULT_RANK_PROBS = [0.4637, 0.2118, 0.1170, 0.0647, 0.0400, 0.0303,
                      0.0214, 0.0156, 0.0113, 0.0081, 0.0056, 0.0039,
                      0.0027, 0.0018, 0.0010, 0.0005]


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
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 3:
                out[row[0]].append((row[1], int(row[2])))
    for k in out:
        out[k].sort(key=lambda x: -x[1])
    return out


def load_candidates(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1]:
                out[p[0]] = p[1].split(",")
    return out


def load_labels(path):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out[p[0]] = p[1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--starts", default=data("start_fens.txt"))
    ap.add_argument("--replies", default="out/continuations_all.csv")
    ap.add_argument("--candidates", default=data("candidates.tsv"))
    ap.add_argument("--labels", nargs="+",
                    default=[data("labels.tsv"), data("labels_d30.tsv"), data("labels_gap.tsv")])
    ap.add_argument("--lichess", default=data("lichess_labels.tsv"))
    ap.add_argument("--cerebellum", default=data("Cerebellum3Merge.bin"))
    ap.add_argument("--rank-probs", default=",".join(str(x) for x in
                                                     DEFAULT_RANK_PROBS))
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--max-move", type=int, default=16)
    a = ap.parse_args()

    if a.max_move > 20:
        raise SystemExit("max-move above 20 is outside the permitted scope")

    rank_probs = [float(x) for x in a.rank_probs.split(",") if x]
    starts = load_starts(a.starts)
    replies = load_replies(a.replies)
    cands = load_candidates(a.candidates)
    # The tree must branch on the move the BOOK will actually play, or we
    # prepare replies to a move we never make. build.py resolves our move as
    # labels -> lichess -> cerebellum -> observed, so mirror that order exactly.
    # (This defaulted to labels.tsv alone, 3,290 of 34,985 labels, so most of
    # our nodes silently followed Cerebellum instead of us.)
    labels = {}
    for _p in a.labels:
        labels.update(load_labels(_p))
    lichess_mv = load_labels(a.lichess) if a.lichess else {}
    total_games = sum(g for _, g in starts)
    print(f"  {len(starts)} starts, {len(cands):,} positions with candidates, "
          f"{len(labels):,} labels, {len(lichess_mv):,} lichess")
    print(f"  opponent model: {rank_probs}")

    cere = pg.open_reader(a.cerebellum) if os.path.exists(a.cerebellum) else None

    heap = []
    seq = 0
    for fen, games in starts:
        share = games / total_games
        for colour in (chess.WHITE, chess.BLACK):
            heap.append((-share, seq, fen, colour, 1.0))
            seq += 1
    heapq.heapify(heap)

    best, bcond, seen = {}, {}, {}
    need_cands = set()
    pushed = 0

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
        if glob > best.get(fen, 0.0):
            best[fen] = glob
        if cond > bcond.get(fen, 0.0):
            bcond[fen] = cond

        if board.turn == colour:
            # Our move: exactly one child. We play one move, not a distribution.
            uci = labels.get(fen) or lichess_mv.get(fen)
            if not uci and cere is not None:
                try:
                    uci = cere.find(board).move.uci()
                except (IndexError, KeyError, ValueError):
                    uci = None
            if not uci:
                obs = replies.get(fen)
                uci = obs[0][0] if obs else None
            kids = [(uci, 1.0)] if uci else []
        else:
            # Opponent: branch over candidates, weighted by the measured model.
            obs = [u for u, _ in replies.get(fen, [])]
            cand = cands.get(fen)
            if cand is None:
                need_cands.add(fen)
            merged = list(dict.fromkeys(obs + (cand or [])))[:len(rank_probs)]
            kids = [(u, rank_probs[i]) for i, u in enumerate(merged)]

        for uci, share in kids:
            if not uci:
                continue
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
    if need_cands:
        with open("need_candidates.txt", "w", encoding="utf-8") as f:
            for fen in need_cands:
                f.write(fen + "\n")

    print(f"  {len(rows):,} positions (<= move {a.max_move}, "
          f"threshold {a.threshold:g}), {pushed:,} edges")
    print(f"  {len(need_cands):,} positions still need a MultiPV candidate list "
          f"-> need_candidates.txt")
    print(f"  wrote {a.out} and {txt}")


if __name__ == "__main__":
    main()
