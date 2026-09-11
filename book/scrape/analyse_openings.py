"""
Turn the scraped Chessathon PGNs into an opening book keyed by position.

The tournament assigns each game a book line and hands the engines a position to
play from; that position is the PGN's [FEN] tag and is all the book leaves behind.
So everything here is keyed by position, never by move order -- transpositions
collapse onto the same key, which is what an engine actually probes.

Two products:

  1. Exit-position census. Every distinct book-exit position, grouped under its
     opening name, with how often it was played and how it scored. This is
     "Sicilian Closed" split into its 34 actual starting positions.

  2. Post-book table. Every position reached AFTER the book ends, with the moves
     played from it, their frequency and their score for the side to move --
     i.e. the prep that starts where the tournament book stops.

Outputs into ./out :
    exit_positions.csv   one row per book-exit position
    opening_families.csv one row per opening name
    book.json            post-book position -> {move: stats}, for your engine
    book_report.txt      readable summary

    python analyse_openings.py
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import data  # noqa: E402


import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chess

OUT = Path(data("scrape"))
PGN_DIR = OUT / "pgn"

# How far past the book exit to keep aggregating. Beyond ~20 plies almost every
# position is unique, so it stops earning its size.
BOOK_DEPTH = 20
# A position needs this many games before its move stats mean anything.
MIN_GAMES = 4
# Pseudo-games of shrinkage toward the position mean. Roughly: a move must be
# played ~PRIOR_W times before its own score dominates the recommendation.
PRIOR_W = 10

TAG_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')
COMMENT_RE = re.compile(r"\{[^}]*\}")
NUM_RE = re.compile(r"\d+\.(\.\.)?")
RESULT_RE = re.compile(r"(1-0|0-1|1/2-1/2|\*)$")


def parse_pgn(text: str):
    """Return (tags, [chess.Move...]) replayed from the PGN's own [FEN]."""
    tags = dict(TAG_RE.findall(text))
    # Clock comments contain ']' too, so cut at the end of the last tag pair.
    last = None
    for last in TAG_RE.finditer(text):
        pass
    body = text[last.end():] if last else text
    body = COMMENT_RE.sub(" ", body)
    body = RESULT_RE.sub(" ", body.strip())
    body = NUM_RE.sub(" ", body)

    board = chess.Board(tags["FEN"])
    moves = []
    for tok in body.split():
        if tok in ("*", "1-0", "0-1", "1/2-1/2"):
            break
        try:
            moves.append(board.push_san(tok))
        except ValueError:
            # Truncated or malformed tail: keep what replayed cleanly.
            break
    return tags, moves


def score_for(result: str, colour: bool) -> float:
    """Points for `colour` (1 win, 0.5 draw, 0 loss)."""
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if colour == chess.WHITE else 0.0
    return 0.0 if colour == chess.WHITE else 1.0


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

    # ---- exit positions -------------------------------------------------
    exit_stats = defaultdict(lambda: {"games": 0, "w": 0, "d": 0, "l": 0,
                                      "opening": "", "ply": 0})
    # ---- post-book position table --------------------------------------
    # epd -> move_uci -> [games, score_for_side_to_move]
    table = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    pos_meta = {}
    skipped = 0

    for n, (gid, meta) in enumerate(index.items(), 1):
        text = (PGN_DIR / f"{gid}.pgn").read_text(encoding="utf-8")
        try:
            tags, moves = parse_pgn(text)
        except Exception as e:                                   # noqa: BLE001
            print(f"  ! {gid}: {e}", file=sys.stderr)
            skipped += 1
            continue
        result = tags.get("Result", "*")
        if result not in ("1-0", "0-1", "1/2-1/2") or not moves:
            skipped += 1
            continue

        board = chess.Board(tags["FEN"])
        exit_epd = board.epd()
        e = exit_stats[exit_epd]
        e["games"] += 1
        e["opening"] = meta["opening"]
        e["ply"] = 2 * (board.fullmove_number - 1) + (0 if board.turn else 1)
        e["w" if result == "1-0" else "l" if result == "0-1" else "d"] += 1

        for i, mv in enumerate(moves[:BOOK_DEPTH]):
            epd = board.epd()
            if epd not in pos_meta:
                pos_meta[epd] = {"opening": meta["opening"], "exit_ply": e["ply"],
                                 "depth_past_exit": i, "turn":
                                 "w" if board.turn == chess.WHITE else "b"}
            cell = table[epd][mv.uci()]
            cell[0] += 1
            cell[1] += score_for(result, board.turn)
            board.push(mv)

        if n % 2500 == 0:
            print(f"  {n}/{len(index)}  {len(table)} positions")

    print(f"parsed {len(index)-skipped} games, skipped {skipped}")
    print(f"{len(exit_stats)} exit positions, {len(table)} post-book positions")

    # ---- exit_positions.csv --------------------------------------------
    rows = []
    for epd, s in exit_stats.items():
        moves = table.get(epd, {})
        best = max(moves.items(), key=lambda kv: kv[1][0]) if moves else None
        rows.append({
            "opening": s["opening"],
            "exit_ply": s["ply"],
            "games": s["games"],
            "white_score": round((s["w"] + 0.5 * s["d"]) / s["games"], 4),
            "wins_white": s["w"], "draws": s["d"], "wins_black": s["l"],
            "side_to_move": epd.split()[1],
            "distinct_replies": len(moves),
            "top_reply": best[0] if best else "",
            "top_reply_games": best[1][0] if best else 0,
            "epd": epd,
        })
    rows.sort(key=lambda r: (r["opening"], -r["games"]))
    with (OUT / "exit_positions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- opening_families.csv ------------------------------------------
    fam = defaultdict(lambda: {"games": 0, "positions": 0, "w": 0.0})
    for r in rows:
        d = fam[r["opening"]]
        d["games"] += r["games"]
        d["positions"] += 1
        d["w"] += r["white_score"] * r["games"]
    fam_rows = [{"opening": k, "games": v["games"], "distinct_positions":
                 v["positions"], "white_score": round(v["w"] / v["games"], 4)}
                for k, v in fam.items()]
    fam_rows.sort(key=lambda r: -r["games"])
    with (OUT / "opening_families.csv").open("w", newline="",
                                             encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fam_rows[0]))
        w.writeheader()
        w.writerows(fam_rows)

    # ---- book.json ------------------------------------------------------
    book = {}
    for epd, moves in table.items():
        total = sum(c[0] for c in moves.values())
        if total < MIN_GAMES:
            continue
        # Raw score is treacherous at these sample sizes -- a move played 3 times
        # scoring 0.67 is noise, not an improvement. Shrink each move toward the
        # position's own mean so frequency has to earn the recommendation.
        pos_mean = sum(c[1] for c in moves.values()) / total
        entries = []
        for uci, (cnt, pts) in moves.items():
            if cnt < 2:
                continue
            entries.append({
                "move": uci, "games": cnt,
                "score": round(pts / cnt, 4),
                "adj_score": round((pts + PRIOR_W * pos_mean) / (cnt + PRIOR_W), 4),
            })
        if not entries:
            continue
        # Best = shrunk score, frequency breaking ties; that is what to play.
        entries.sort(key=lambda e: (-e["adj_score"], -e["games"]))
        book[epd] = {"games": total, **pos_meta[epd], "moves": entries}
    (OUT / "book.json").write_text(json.dumps(book, indent=1), encoding="utf-8")

    # ---- report ---------------------------------------------------------
    lines = []
    lines.append(f"Chessathon opening book -- {sum(r['games'] for r in rows)} games")
    lines.append(f"{len(rows)} book-exit positions across {len(fam_rows)} openings")
    lines.append(f"{len(book)} post-book positions with >={MIN_GAMES} games\n")
    lines.append(f"{'opening':32}{'games':>7}{'pos':>6}{'W score':>9}")
    for r in fam_rows:
        lines.append(f"{r['opening'][:31]:32}{r['games']:>7}"
                     f"{r['distinct_positions']:>6}{r['white_score']:>9.3f}")
    lines.append("\n\nBook-exit positions, by opening\n")
    for r in rows:
        lines.append(f"{r['opening'][:30]:31} ply {r['exit_ply']:>2} "
                     f"{r['games']:>5} games  W {r['white_score']:.3f}  "
                     f"{r['side_to_move']} to move  best {r['top_reply']:>6} "
                     f"({r['top_reply_games']})")
        lines.append(f"    {r['epd']}")
    (OUT / "book_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nwrote exit_positions.csv, opening_families.csv, book.json, "
          "book_report.txt")
    print(f"\n{'opening':32}{'games':>7}{'pos':>6}{'W score':>9}")
    for r in fam_rows[:12]:
        print(f"{r['opening'][:31]:32}{r['games']:>7}"
              f"{r['distinct_positions']:>6}{r['white_score']:>9.3f}")


if __name__ == "__main__":
    main()
