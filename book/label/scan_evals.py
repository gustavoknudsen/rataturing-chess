"""
Harvest Lichess cloud evals for the positions we care about.

One streaming pass over lichess_db_eval.jsonl.zst, matching on the first four
FEN fields -- the eval DB stores no halfmove/fullmove counters, so a position is
identified by board/turn/castling/ep alone.

Two products, because the DB serves two jobs with different quality bars:

  --out-labels     fen<TAB>uci<TAB>cp<TAB>depth   for OUR move. Only accepted at
                   >= --min-depth-move, since below our own labelling depth we
                   would rather spend the CPU and get a better move.
  --out-cands      fen<TAB>uci,uci,...            for OPPONENT candidates. A much
                   lower bar: we are only deciding which replies are plausible
                   enough to branch on, and PV width matters more than depth.

The FEN written out is our original (with counters) so downstream stages line up.

    zstd is used via a pipe, so nothing is ever decompressed to disk.

    python scan_evals.py D:\\lichess\\lichess_db_eval.jsonl.zst \\
        --positions positions_full.tsv hedge_positions.tsv start_fens.txt
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import data  # noqa: E402


import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import chess


def key4(fen):
    return " ".join(fen.split()[:4])


def load_position_files(paths):
    """fen4 -> [original fen, ...]. Accepts .tsv, .txt and start_fens.txt."""
    out = defaultdict(list)
    for p in paths:
        if not os.path.exists(p):
            print(f"  skip (missing): {p}", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fen = line.split("\t")[0].split(";")[0].strip()
                if fen.count("/") != 7:
                    continue
                out[key4(fen)].append(fen)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--positions", nargs="+", required=True)
    ap.add_argument("--out-labels", default=data("lichess_labels.tsv"))
    ap.add_argument("--out-cands", default=data("lichess_candidates.tsv"))
    ap.add_argument("--min-depth-move", type=int, default=22)
    ap.add_argument("--min-depth-cand", type=int, default=12)
    ap.add_argument("--zstd", default="zstd")
    a = ap.parse_args()

    want = load_position_files(a.positions)
    print(f"  looking for {len(want):,} distinct positions "
          f"({sum(len(v) for v in want.values()):,} FEN spellings)")

    proc = subprocess.Popen([a.zstd, "-dc", a.db], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    depth_hist = Counter()
    seen = 0
    hits = {}

    for raw in proc.stdout:
        seen += 1
        if seen % 20_000_000 == 0:
            print(f"    {seen:,} records, {len(hits):,} matched", file=sys.stderr)
        # Fast reject without parsing JSON: the fen is the first field.
        if not raw.startswith(b'{"fen":"'):
            continue
        j = raw.find(b'"', 8)
        if j < 0:
            continue
        k = raw[8:j].decode("ascii", "ignore")
        if k not in want or k in hits:
            continue
        try:
            obj = json.loads(raw)
        except Exception:                                    # noqa: BLE001
            continue
        best = None
        for e in obj.get("evals", []):
            d = e.get("depth")
            if isinstance(d, int) and (best is None or d > best.get("depth", -1)):
                best = e
        if not best:
            continue
        hits[k] = best
        depth_hist[best.get("depth", 0)] += 1

    proc.stdout.close()
    proc.wait()
    print(f"  scanned {seen:,} records, matched {len(hits):,} of {len(want):,} "
          f"({len(hits)/max(1,len(want)):.1%})")

    # ---- coverage by depth threshold: the number that picks the threshold ----
    print(f"\n  {'min depth':>10} {'positions':>10} {'of our set':>11}")
    for t in (0, 10, 14, 18, 20, 22, 25, 28, 30, 35, 40):
        n = sum(v for d, v in depth_hist.items() if d >= t)
        print(f"  {t:>10} {n:>10,} {n/max(1,len(want)):>10.1%}")

    n_lab = n_cand = 0
    with open(a.out_labels, "w", encoding="utf-8") as fl, \
            open(a.out_cands, "w", encoding="utf-8") as fc:
        for k, e in hits.items():
            depth = e.get("depth", 0)
            pvs = e.get("pvs") or []
            for fen in want[k]:
                board = chess.Board(fen)
                moves, scores = [], []
                for pv in pvs:
                    line = (pv.get("line") or "").split()
                    if not line:
                        continue
                    try:
                        mv = chess.Move.from_uci(line[0])
                    except ValueError:
                        continue
                    if mv not in board.legal_moves:
                        continue          # eval belongs to a different position
                    moves.append(line[0])
                    cp = pv.get("cp")
                    if cp is None and pv.get("mate") is not None:
                        m = int(pv["mate"])
                        cp = (100000 - abs(m) * 100) * (1 if m > 0 else -1)
                    scores.append(cp if cp is not None else 0)
                if not moves:
                    continue
                # Both files carry depth and knodes, and the default thresholds
                # are 0, on purpose. Depth turned out to carry almost no signal
                # about which move you get -- 78.3% agreement with the same
                # position's most-searched eval at depth 12 versus 77.4% at
                # depth 30 -- so cutting on it loses coverage and buys no
                # quality. Choose the threshold downstream, on the data.
                knodes = e.get("knodes") or 0
                if depth >= a.min_depth_move:
                    fl.write(f"{fen}\t{moves[0]}\t{scores[0]}\t{depth}\t{knodes}\n")
                    n_lab += 1
                if depth >= a.min_depth_cand:
                    fc.write(f"{fen}\t{','.join(moves)}\t{depth}\t{knodes}\n")
                    n_cand += 1

    print(f"\n  wrote {n_lab:,} moves (depth >= {a.min_depth_move}) -> "
          f"{a.out_labels}")
    print(f"  wrote {n_cand:,} candidate lists (depth >= {a.min_depth_cand}) -> "
          f"{a.out_cands}")


if __name__ == "__main__":
    main()
