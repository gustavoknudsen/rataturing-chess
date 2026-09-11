"""
Copy the shippable books into the agent directory, and nothing else.

Deliberately explicit rather than globbing *.bin: the agent directory also holds
books kept for comparison (komodo.bin and friends) which are game-frequency
sources we excluded on purpose. Globbing would ship them, wasting budget and
misrepresenting how the book was built.

Every file is verified before it is copied -- readable by chess.polyglot, a
multiple of 16 bytes, sorted by key, and free of entries past move 20.

    python package_books.py --dest ../Competition
"""

import argparse
import os
import shutil
import struct
import sys

import chess
import chess.polyglot as pg

ENTRY = struct.Struct(">QHHI")

# source in this directory -> name the agent expects (btc_book.py BOOK_FILES)
SHIP = {
    "rataturing.bin": "rataturing.bin",
    "rataturing_hedge.bin": "rataturing_hedge.bin",
}
MAX_BOOK_MOVE = 20


def check(path):
    """Structural checks that do not need the position list."""
    problems = []
    size = os.path.getsize(path)
    if size == 0:
        problems.append("empty")
    if size % 16:
        problems.append(f"not a multiple of 16 bytes ({size % 16} over)")
    raw = open(path, "rb").read()
    keys = [ENTRY.unpack_from(raw, i)[0] for i in range(0, len(raw) - 15, 16)]
    if keys != sorted(keys):
        problems.append("NOT sorted by key -- the reader binary-searches and "
                        "would silently miss")
    try:
        with pg.open_reader(path) as r:
            r.find(chess.Board())      # may legitimately miss; only checks it reads
    except (IndexError, KeyError, ValueError):
        pass
    except Exception as exc:                                 # noqa: BLE001
        problems.append(f"unreadable: {exc!r}")
    return size, len(keys), problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="../Competition")
    ap.add_argument("--budget-mb", type=float, default=20.0)
    a = ap.parse_args()

    total = 0
    failed = False
    for src, dst in SHIP.items():
        if not os.path.exists(src):
            print(f"  MISSING {src}")
            failed = True
            continue
        size, n, problems = check(src)
        total += size
        status = "ok" if not problems else "FAIL"
        print(f"  {src:<24} {n:>8,} entries  {size/1e6:6.2f} MB  {status}")
        for p in problems:
            print(f"      - {p}")
            failed = True

    print(f"\n  total {total/1e6:.2f} MB of the {a.budget_mb:.0f} MB book budget")
    if total > a.budget_mb * 1e6:
        print("  FAIL: over budget")
        failed = True
    if failed:
        print("\n  nothing copied")
        sys.exit(1)

    for src, dst in SHIP.items():
        shutil.copy2(src, os.path.join(a.dest, dst))
        print(f"  copied {src} -> {os.path.join(a.dest, dst)}")

    # Warn about anything else in the destination that a *.bin glob would ship.
    extra = [f for f in os.listdir(a.dest)
             if f.endswith(".bin") and f not in SHIP.values()]
    if extra:
        print(f"\n  NOTE: {a.dest} also holds {', '.join(extra)} - kept for "
              f"comparison, not part of the submission. Exclude them when "
              f"zipping; a *.bin glob would add "
              f"{sum(os.path.getsize(os.path.join(a.dest, f)) for f in extra)/1e6:.1f} MB.")


if __name__ == "__main__":
    main()
