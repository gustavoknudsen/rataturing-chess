"""Turn the Lichess evaluations database into a shuffled NNUE training set.

    set BTC_EVAL_DB=D:\\chess_nnue\\lichess_db_eval.jsonl.zst
    python nnue_data.py <out_dir> [target_positions]

Source: https://database.lichess.org/lichess_db_eval.jsonl.zst - 394.7M positions
evaluated by Stockfish, **CC0 1.0 (public domain)**. Competition rules allow this
explicitly ("Training data unrestricted, including engine-annotated positions");
what is banned is shipping someone else's *network*, which we are not doing.

Output, as .npy in <out_dir>:
  feats.npy  uint16 (N, 32)  feature indices (colour*6 + piece)*64 + square,
                             square 0 = a8 (the engine's own orientation),
                             padded with 65535
  counts.npy uint8  (N,)
  stm.npy    uint8  (N,)     0 white to move, 1 black
  score.npy  int16  (N,)     centipawns **from the side to move's point of view**

Three things here are load-bearing and each is a silent-failure class:

1. **The score is converted to side-to-move relative.** Lichess `cp` is
   White-relative - established empirically twice, by correlating the sign against
   material-decided positions (80.4% White-relative against 54.6% for
   side-to-move, i.e. chance). The network output is side-to-move relative
   because the accumulators are concatenated stm-first. Skip the conversion and
   the net learns to predict |eval| and calls every position equal.
2. **The rows are shuffled on disk.** The database is derived from analysed
   games, so consecutive records are consecutive positions of the same game with
   near-identical evaluations; without shuffling a 16384 batch has a tiny
   effective sample size. Measured at +17.29 +-8.50 elo in tcheran's log for the
   shuffle alone.
3. **One row per FEN**, taking the deepest evaluation and its first PV, as the
   database's own guidance recommends. Emitting every eval would upweight
   heavily-analysed positions, which are already over-represented.
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import json
import os
import subprocess
import sys

import numpy as np

from btc_core import is_under_attack, lsb
from numba import njit, uint64

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
MAX_PIECES = 32
PAD = 65535
MAX_EVAL = 10000          # bullet's binpack filter; beyond this the sigmoid is flat
MIN_PIECES = 4
PIECE_INDEX = {c: i for i, c in enumerate("PNBRQK")}
PIECE_INDEX.update({c: i + 6 for i, c in enumerate("pnbrqk")})


def parse_board(fen_board, feats, board):
    """Fill feature indices and a square->piece map. Returns count, or -1."""
    square = 0
    count = 0
    board[:] = -1
    for ch in fen_board:
        if ch == "/":
            continue
        if ch.isdigit():
            square += ord(ch) - 48
            continue
        piece = PIECE_INDEX.get(ch)
        if piece is None or square > 63 or count >= MAX_PIECES:
            return -1
        feats[count] = piece * 64 + square
        board[square] = piece
        count += 1
        square += 1
    return count if square == 64 else -1


def uci_square(text):
    """'e2' -> index with square 0 = a8, matching the engine's orientation."""
    return (8 - (ord(text[1]) - 48)) * 8 + (ord(text[0]) - 97)


def is_tactical(line, board):
    """True if the best move is a capture, promotion or en passant.

    The static evaluation of such a position is about to be invalidated by a
    material swing, which is exactly what makes it a bad training target. This
    is the single most valuable filter available from this data format."""
    if not line:
        return True
    move = line.split(" ", 1)[0]
    if len(move) == 5:                       # promotion
        return True
    if len(move) != 4:
        return True
    src = uci_square(move[0:2])
    dst = uci_square(move[2:4])
    if not (0 <= src <= 63 and 0 <= dst <= 63):
        return True
    if board[dst] != -1:                     # capture
        return True
    piece = board[src]
    if piece in (0, 6) and (src % 8) != (dst % 8):
        return True                          # pawn changing file onto an empty square: en passant
    return False


@njit(cache=False)
def _in_check_batch(feats, counts, stm, keep):
    """Drop positions where the side to move is in check.

    Rebuilds the bitboards from the stored feature indices rather than keeping a
    second copy of the board, then reuses the engine's own attack detection so
    the filter agrees exactly with how the engine sees the position."""
    bb = np.zeros(16, dtype=np.uint64)
    for row in range(feats.shape[0]):
        if not keep[row]:
            continue
        bb[:] = uint64(0)
        for i in range(counts[row]):
            index = feats[row, i]
            bb[index // 64] |= uint64(1) << uint64(index % 64)
        for piece in range(6):
            bb[12] |= bb[piece]
            bb[13] |= bb[piece + 6]
        bb[14] = bb[12] | bb[13]
        side = stm[row]
        king = bb[5] if side == 0 else bb[11]
        if king == uint64(0):
            keep[row] = False
            continue
        if is_under_attack(bb, lsb(king), 1 - side):
            keep[row] = False


def best_eval(record):
    """(cp, line) for the deepest evaluation, or None. Mates are dropped: a
    mate score is not on the centipawn scale and teaching the net to emit
    +-30000 for them distorts everything else."""
    best_depth = -1
    best = None
    for entry in record.get("evals", ()):
        depth = entry.get("depth", 0)
        if depth <= best_depth:
            continue
        pvs = entry.get("pvs") or ()
        if not pvs:
            continue
        cp = pvs[0].get("cp")
        if cp is None:
            continue
        best_depth = depth
        best = (cp, pvs[0].get("line", ""))
    return best


@njit(cache=False)
def _compact(feats, counts, stm, score, keep):
    """Move the kept rows down in place. Returns the surviving count.

    `a[keep]` would be one line, but it allocates a second copy of every array,
    and feats alone is 64 bytes per position - 5 GB at 80M positions. **Peak
    memory, not disk or time, is what caps the dataset size**, and a boolean
    mask doubles it. Compaction in place is one forward pass, no allocation.
    njit because this is 80M iterations; in Python it would take hours."""
    out = 0
    for i in range(keep.shape[0]):
        if not keep[i]:
            continue
        if out != i:
            for k in range(feats.shape[1]):
                feats[out, k] = feats[i, k]
            counts[out] = counts[i]
            stm[out] = stm[i]
            score[out] = score[i]
        out += 1
    return out


@njit(cache=False)
def _shuffle(feats, counts, stm, score, count, seed):
    """Fisher-Yates over the rows, in place, for the same reason as _compact.

    Shuffling matters more than it looks: the database is derived from analysed
    games, so neighbouring records are consecutive positions of one game with
    near-identical evaluations, and an unshuffled 16384 batch has a tiny
    effective sample size. Measured at +17.29 +-8.50 elo in tcheran's log."""
    np.random.seed(seed)
    for i in range(count - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        if j == i:
            continue
        for k in range(feats.shape[1]):
            tmp = feats[i, k]
            feats[i, k] = feats[j, k]
            feats[j, k] = tmp
        c = counts[i]
        counts[i] = counts[j]
        counts[j] = c
        s = stm[i]
        stm[i] = stm[j]
        stm[j] = s
        v = score[i]
        score[i] = score[j]
        score[j] = v


def main():
    out_dir = sys.argv[1]
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000_000
    os.makedirs(out_dir, exist_ok=True)

    feats = np.full((target, MAX_PIECES), PAD, dtype=np.uint16)
    counts = np.zeros(target, dtype=np.uint8)
    stm = np.zeros(target, dtype=np.uint8)
    score = np.zeros(target, dtype=np.int16)

    local = os.environ.get("BTC_EVAL_DB")
    if local and os.path.exists(local):
        curl = None
        zstd = subprocess.Popen(["zstd", "-dc", local], stdout=subprocess.PIPE)
    else:
        curl = subprocess.Popen(["curl", "-sL", URL], stdout=subprocess.PIPE)
        zstd = subprocess.Popen(["zstd", "-dc"], stdin=curl.stdout,
                                stdout=subprocess.PIPE)
        curl.stdout.close()

    row = np.empty(MAX_PIECES, dtype=np.uint16)
    board = np.empty(64, dtype=np.int8)
    kept = seen = 0
    dropped = {"nocp": 0, "eval": 0, "pieces": 0, "book": 0, "tactical": 0}
    try:
        for raw in zstd.stdout:
            seen += 1
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            found = best_eval(record)
            if found is None:
                dropped["nocp"] += 1
                continue
            cp, line = found
            if cp > MAX_EVAL or cp < -MAX_EVAL:
                dropped["eval"] += 1
                continue
            parts = record.get("fen", "").split(" ")
            if len(parts) < 3:
                continue
            row[:] = PAD
            n = parse_board(parts[0], row, board)
            if n < MIN_PIECES:
                dropped["pieces"] += 1
                continue
            # Book proxy: the database has no fullmove counter, and analysed-game
            # data massively over-represents the first few moves. A full board
            # with every castling right is a near-book position whose evaluation
            # is arbitrary.
            if n == 32 and parts[2] == "KQkq":
                dropped["book"] += 1
                continue
            if is_tactical(line, board):
                dropped["tactical"] += 1
                continue
            white_to_move = parts[1] == "w"
            feats[kept] = row
            counts[kept] = n
            stm[kept] = 0 if white_to_move else 1
            # Lichess cp is White-relative; the network is side-to-move relative.
            score[kept] = cp if white_to_move else -cp
            kept += 1
            if kept >= target:
                break
            if kept % 2_000_000 == 0:
                print(f"kept {kept:,} of {seen:,} seen", flush=True)
    finally:
        for proc in (zstd, curl):
            if proc is None:
                continue
            try:
                proc.kill()
            except OSError:
                pass

    feats, counts = feats[:kept], counts[:kept]
    stm, score = stm[:kept], score[:kept]

    print(f"read {seen:,} lines, kept {kept:,}", flush=True)
    print(f"dropped: {dropped}", flush=True)

    keep = np.ones(kept, dtype=np.bool_)
    _in_check_batch(feats, counts, stm, keep)
    print(f"in check / no king: {kept - int(keep.sum()):,}", flush=True)

    total = _compact(feats, counts, stm, score, keep)
    feats, counts = feats[:total], counts[:total]
    stm, score = stm[:total], score[:total]
    _shuffle(feats, counts, stm, score, total, 20260908)

    print(f"writing {total:,} shuffled positions", flush=True)
    np.save(os.path.join(out_dir, "feats.npy"), feats)
    np.save(os.path.join(out_dir, "counts.npy"), counts)
    np.save(os.path.join(out_dir, "stm.npy"), stm)
    np.save(os.path.join(out_dir, "score.npy"), score)
    print("done", flush=True)


if __name__ == "__main__":
    main()
