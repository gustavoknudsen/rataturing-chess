"""Gate on the staged submission. Run: python finals_day/test_staged.py

The claim being tested is narrow and the whole point of the build: import
returns fast enough for any init budget, a legal move comes back immediately
afterwards, and the real engine takes over once it has compiled.

Runs against finals_day/staging/staged/, so build it first with stage.py.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGED = os.path.join(HERE, "staging", "staged")

DRIVER = r'''
import sys, time
sys.path.insert(0, STAGED_DIR)
t0 = time.perf_counter()
import agent
IMPORT = time.perf_counter() - t0
print("IMPORT %.3f" % IMPORT)

import chess
board = chess.Board()
# First move must come back immediately, long before the engine is compiled.
t = time.perf_counter()
mv = agent.get_move(board.fen(), 120000)
print("FIRST %.3f %s pure=%s" % (time.perf_counter() - t, mv,
                                 agent._engine is None))
board.push_uci(mv)

# Play until the real engine takes over, or give up after a generous wall.
deadline = time.perf_counter() + 180
plies = 1
while agent._engine is None and time.perf_counter() < deadline:
    mv = agent.get_move(board.fen(), 120000)
    m = chess.Move.from_uci(mv)
    if m not in board.legal_moves:
        print("ILLEGAL %s" % mv)
        break
    board.push(m)
    plies += 1
    if board.is_game_over():
        board = chess.Board()
print("UPGRADED %s after %d plies at %.1fs"
      % (agent._engine is not None, plies, time.perf_counter() - t0))

# And a move from the real engine, to prove the handover works.
mv = agent.get_move(board.fen(), 120000)
print("AFTER %s legal=%s" % (mv, chess.Move.from_uci(mv) in board.legal_moves))
'''.replace("STAGED_DIR", repr(STAGED))

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    print("  %-34s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if ok:
        PASS += 1
    else:
        FAIL += 1


def main():
    if not os.path.isdir(STAGED):
        raise SystemExit("build it first: python finals_day/stage.py")
    result = subprocess.run([sys.executable, "-c", DRIVER],
                            capture_output=True, text=True, timeout=600)
    out = result.stdout
    print(out.strip() + "\n")
    lines = {l.split()[0]: l.split() for l in out.splitlines() if l.split()}

    imp = float(lines["IMPORT"][1]) if "IMPORT" in lines else 999
    check("import is under 5 s", imp < 5.0, "%.3fs" % imp)
    check("import is under 1 s (any budget)", imp < 1.0, "%.3fs" % imp)

    first = lines.get("FIRST")
    check("first move returned", first is not None, first[2] if first else "")
    check("first move came from the fallback",
          bool(first) and first[3] == "pure=True", first[3] if first else "")

    up = lines.get("UPGRADED")
    check("real engine took over", bool(up) and up[1] == "True",
          " ".join(up[2:]) if up else "never")

    after = lines.get("AFTER")
    check("move after handover is legal",
          bool(after) and after[2] == "legal=True", after[1] if after else "")
    check("no illegal move at any stage", "ILLEGAL" not in out)

    print("\n%d passed, %d failed" % (PASS, FAIL))
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
