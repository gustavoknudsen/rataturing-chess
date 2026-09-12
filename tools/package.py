"""Build and verify submission.zip. Run: python package.py

Ships only the runtime modules, at the zip root, as the platform requires.
Tests, docs, benchmarks and the venv are deliberately excluded. The built zip
is then extracted to a scratch directory and played out of, so a missing file
fails here instead of costing one of the ten daily uploads.
"""

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

# This script lives in tools/. The engine and its data files live in src/,
# and every shipped file goes into the zip FLAT (arcname is the bare
# filename) because the platform does `import agent` at the zip root.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = os.path.join(REPO, "src")
ZIP_PATH = os.path.join(REPO, "submission.zip")
MAX_UNZIPPED_BYTES = 50_000_000

SHIP = [
    "agent.py",
    "btc_core.py",
    "btc_endgame.py",
    "btc_eval.py",
    "btc_evalmasks.py",
    "btc_game.py",
    "btc_kpk.py",
    "btc_kpk_data.py",
    "btc_psqt.py",
    "btc_scale.py",
    "btc_book.py",
    "btc_nnue.py",
    "btc_nrt.py",
    # Shipped but never imported at BTC_THREADS=1, which is the shipped
    # configuration. It rides along so that raising the thread count on
    # finals day is a panel change rather than a file copy under time
    # pressure. About 7 KB against a 50 MB cap.
    "btc_parallel.py",
    "btc_search.py",
    "btc_time.py",
]

# Data files: shipped when present, skipped when not, never parsed as source.
# net.npz is the self-trained network. Shipping it is what turns the NNUE
# evaluation on, because btc_eval only enables it when the file is actually
# there - so a build with no net.npz is exactly the hand-crafted engine.
# Both books ship. rataturing.bin has no entry for the standard start - it
# is built outward from the tournament's CURATED opening positions, not from
# move 1 - which is why an earlier note here wrongly called it zobrist
# mismatched. Verified: it answers all 8 published curated samples and the
# real round-107 rated position, 9 for 9.
#
# btc_book.py gates every lookup at MAX_BOOK_MOVE = 20. Never raise it: a
# Zobrist key carries no move number, so a position stored at move 5 would
# otherwise return a hit at move 34, and that runtime gate is the only thing
# keeping a shipped table on the legal side of "an engine in another shape".
DATA = ["net.npz", "rataturing.bin", "rataturing_hedge.bin"]

# llvmlite is a hard dependency of numba - its metadata declares
# llvmlite<0.50,>=0.49.0dev0 - so it is present wherever numba is, and
# numba is what the whole engine is built on. btc_nrt imports it.
PREINSTALLED = {"chess", "numpy", "numba", "torch", "onnxruntime",
                "llvmlite"}

SMOKE = """
import time
t = time.perf_counter()
import agent
init = time.perf_counter() - t
import chess
board = chess.Board()
for _ in range(12):
    uci = agent.get_move(board.fen(), 120000)
    mv = chess.Move.from_uci(uci)
    assert mv in board.legal_moves, "illegal move %s in %s" % (uci, board.fen())
    board.push(mv)
    if board.is_game_over():
        break
    board.push(sorted(board.legal_moves, key=lambda m: m.uci())[0])
print("SMOKE_OK init=%.1f plies=%d" % (init, board.ply()))
"""


def _local_module_names():
    return {name[:-3] for name in SHIP}


def check_imports():
    """Every import in a shipped file must be stdlib, preinstalled, or shipped."""
    ok = True
    local = _local_module_names()
    for name in SHIP:
        tree = ast.parse(open(os.path.join(ROOT, name), encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for root in roots:
                if root in local or root in PREINSTALLED:
                    continue
                if root in sys.stdlib_module_names:
                    continue
                print(f"ERROR {name} imports unavailable module '{root}'")
                ok = False
    return ok


def shipped_data():
    return [name for name in DATA if os.path.exists(os.path.join(ROOT, name))]


def build():
    total = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in SHIP + shipped_data():
            path = os.path.join(ROOT, name)
            if not os.path.exists(path):
                print(f"ERROR missing {name}")
                return None
            total += os.path.getsize(path)
            zf.write(path, arcname=name)
    return total


def smoke():
    workdir = tempfile.mkdtemp(prefix="btc_smoke_")
    try:
        with zipfile.ZipFile(ZIP_PATH) as zf:
            names = zf.namelist()
            if "agent.py" not in names:
                print(f"ERROR agent.py not at zip root: {names}")
                return False
            if any("/" in n for n in names):
                print(f"ERROR zip has subdirectories: {names}")
                return False
            zf.extractall(workdir)
        out = subprocess.run([sys.executable, "-c", SMOKE], capture_output=True,
                             text=True, cwd=workdir, timeout=600)
        print(out.stdout.strip())
        if out.returncode:
            print(out.stderr[-1500:])
        return out.returncode == 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    if not check_imports():
        sys.exit(1)
    total = build()
    if total is None:
        sys.exit(1)
    zipped = os.path.getsize(ZIP_PATH)
    data = shipped_data()
    print(f"files: {len(SHIP) + len(data)} -> {', '.join(SHIP + data)}")
    # State this explicitly: which evaluation the zip actually plays is decided
    # by whether net.npz is in it, and that is far too important to infer from
    # a file list.
    print("evaluation: NNUE (net.npz shipped)" if data
          else "evaluation: hand-crafted (no net.npz)")
    print(f"unzipped {total} bytes ({total / MAX_UNZIPPED_BYTES:.2%} of cap), "
          f"zipped {zipped} bytes")
    if total > MAX_UNZIPPED_BYTES:
        print("ERROR over the 50 MB unzipped cap")
        sys.exit(1)
    if not smoke():
        print("ERROR smoke test failed")
        sys.exit(1)
    print(f"OK {ZIP_PATH}")


if __name__ == "__main__":
    main()
