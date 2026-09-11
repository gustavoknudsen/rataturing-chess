"""
Export the raw, un-opinionated tables: start positions and observed continuations.

No thresholds, no shrinkage, no recommendations -- just what the 15k games did,
so the numbers can be re-analysed however you like.

Outputs into ./out :
    start_fens.txt        the 308 book-exit FENs, one per line (side to move in
                          field 2), each with its opening name after a ';'
    start_fens_plain.txt  the same FENs, bare, one per line
    continuations_exit.csv    (fen, reply_uci, count, ...) from the start FENs only
    continuations_all.csv     (fen, reply_uci, count, ...) for every position
                              reached within BOOK_DEPTH plies of the book exit

    python export_book.py
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import data  # noqa: E402


import csv
from collections import defaultdict
from pathlib import Path

import chess

from analyse_openings import BOOK_DEPTH, parse_pgn, score_for

OUT = Path(data("scrape"))
PGN_DIR = OUT / "pgn"


def main():
    index = OUT / "games_index.csv"
    if not index.exists():
        raise SystemExit(
            "missing %s\n"
            "This stage reads the scrape output. Run scrape/chessathon_scrape.py first,\n"
            "or place an existing games_index.csv in book/data/scrape/."
            % index)
    index = {r["game_id"]: r for r in
             csv.DictReader((OUT / "games_index.csv").open(encoding="utf-8"))}
    print(f"reading {len(index)} games...")

    # fen -> uci -> [count, points for the side to move, white points]
    table = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    starts = {}          # fen -> {opening, games, w, d, l}
    skipped = 0

    for n, (gid, meta) in enumerate(index.items(), 1):
        tags, moves = parse_pgn((PGN_DIR / f"{gid}.pgn").read_text(encoding="utf-8"))
        result = tags.get("Result", "*")
        if result not in ("1-0", "0-1", "1/2-1/2") or not moves:
            skipped += 1
            continue

        board = chess.Board(tags["FEN"])
        start_fen = board.fen()
        s = starts.setdefault(start_fen, {"opening": meta["opening"], "games": 0,
                                          "w": 0, "d": 0, "l": 0})
        s["games"] += 1
        s["w" if result == "1-0" else "l" if result == "0-1" else "d"] += 1

        for mv in moves[:BOOK_DEPTH]:
            cell = table[board.fen()][mv.uci()]
            cell[0] += 1
            cell[1] += score_for(result, board.turn)
            cell[2] += score_for(result, chess.WHITE)
            board.push(mv)

        if n % 5000 == 0:
            print(f"  {n}/{len(index)}")

    print(f"{len(starts)} start positions, {len(table)} positions total, "
          f"{skipped} games skipped (no moves played)")

    # ---- start FENs -----------------------------------------------------
    ordered = sorted(starts.items(), key=lambda kv: (kv[1]["opening"],
                                                     -kv[1]["games"]))
    with (OUT / "start_fens.txt").open("w", encoding="utf-8") as f:
        for fen, s in ordered:
            f.write(f"{fen} ; {s['opening']} ; games={s['games']} ; "
                    f"white_score={(s['w'] + 0.5*s['d'])/s['games']:.4f}\n")
    with (OUT / "start_fens_plain.txt").open("w", encoding="utf-8") as f:
        for fen, _ in ordered:
            f.write(fen + "\n")

    # ---- continuations --------------------------------------------------
    def dump(path, fens):
        rows = 0
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fen", "reply_uci", "count", "score_side_to_move",
                        "white_score"])
            for fen in fens:
                for uci, (cnt, stm_pts, w_pts) in sorted(
                        table[fen].items(), key=lambda kv: -kv[1][0]):
                    w.writerow([fen, uci, cnt, round(stm_pts / cnt, 4),
                                round(w_pts / cnt, 4)])
                    rows += 1
        print(f"  {path.name}: {rows} rows")

    dump(OUT / "continuations_exit.csv", [f for f, _ in ordered])
    dump(OUT / "continuations_all.csv", table.keys())


if __name__ == "__main__":
    main()
