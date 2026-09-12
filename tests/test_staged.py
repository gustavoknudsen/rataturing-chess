"""Gate on the staged submission. Run: python tests/test_staged.py

The claim being tested is narrow and the whole point of the build: import
returns fast enough for any init budget, a legal move comes back immediately
afterwards, and the real engine takes over once it has compiled.

Runs against staged/staging/staged/, so build it first with staged/build.py.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGED = os.path.join(os.path.dirname(HERE), "staged", "staging", "staged")

DRIVER = r'''
import sys, time
sys.path.insert(0, STAGED_DIR)
t0 = time.perf_counter()
import agent
IMPORT = time.perf_counter() - t0
print("IMPORT %.3f" % IMPORT)

import chess, os
board = chess.Board(os.environ.get("TEST_FEN") or chess.STARTING_FEN)
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


def run(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    result = subprocess.run([sys.executable, "-c", DRIVER],
                            capture_output=True, text=True, timeout=600,
                            env=env)
    out = result.stdout
    print(out.strip() + "\n")
    return out, {l.split()[0]: l.split() for l in out.splitlines() if l.split()}


def main():
    if not os.path.isdir(STAGED):
        raise SystemExit("build it first: python staged/build.py")

    # 1. As shipped: import waits for the compile up to INIT_WAIT_S, so it
    # must return inside the budget whether or not the engine finished.
    print("-- shipped configuration (import waits up to 26 s)")
    out, lines = run({})
    imp = float(lines["IMPORT"][1]) if "IMPORT" in lines else 999
    check("import is under 27 s", imp < 27.0, "%.3fs" % imp)
    first = lines.get("FIRST")
    check("first move returned", first is not None, first[2] if first else "")
    up = lines.get("UPGRADED")
    check("real engine took over", bool(up) and up[1] == "True",
          " ".join(up[2:]) if up else "never")
    after = lines.get("AFTER")
    check("move after handover is legal",
          bool(after) and after[2] == "legal=True", after[1] if after else "")
    check("no illegal move at any stage", "ILLEGAL" not in out)

    # 2. Deadline forced to half a second: proves the wait really is bounded
    # and that the fallback answers while the compile is still running.
    print("-- forced deadline (import waits 0.5 s)")
    out, lines = run({"BTC_STAGE_INIT_WAIT_S": "0.5"})
    imp = float(lines["IMPORT"][1]) if "IMPORT" in lines else 999
    check("import is under 2 s with a 0.5 s deadline", imp < 2.0, "%.3fs" % imp)
    first = lines.get("FIRST")
    # In book, so the move is the book move either way; what matters is
    # that a move came back and the engine arrived afterwards.
    check("first move returned after the deadline", first is not None,
          " ".join(first[1:]) if first else "")
    up = lines.get("UPGRADED")
    check("real engine took over after the deadline",
          bool(up) and up[1] == "True", " ".join(up[2:]) if up else "never")
    check("no illegal move at any stage", "ILLEGAL" not in out)

    # 3. Out of book with the compile unfinished: the move must wait for the
    # real engine rather than come from the fallback. This is the smoke game
    # 2 situation from the platform log.
    OUT_OF_BOOK = "rnbqkb1r/1p3ppp/p2ppn2/8/4PP2/1NN5/PPP3PP/R1BQKB1R b KQkq - 0 7"
    # A 10 s deadline leaves less compile than the 30 s slice, which is the
    # platform's situation (22 s window, 38 to 40 s compile).
    print("-- out of book, forced 10 s deadline: wait for the engine")
    out, lines = run({"BTC_STAGE_INIT_WAIT_S": "10", "TEST_FEN": OUT_OF_BOOK})
    first = lines.get("FIRST")
    check("first move came from the real engine",
          bool(first) and first[3] == "pure=False", first[3] if first else "")
    check("first move waited for the compile",
          bool(first) and float(first[1]) > 5.0, first[1] if first else "")
    check("no illegal move at any stage", "ILLEGAL" not in out)

    # 4. Same, but the donation slice is capped at 2 s: the fallback must
    # answer, legally, without waiting for the engine.
    print("-- out of book, 2 s donation cap: fallback answers")
    out, lines = run({"BTC_STAGE_INIT_WAIT_S": "0.5", "TEST_FEN": OUT_OF_BOOK,
                      "BTC_STAGE_DONATE_MAX_S": "2"})
    first = lines.get("FIRST")
    check("first move came from the fallback",
          bool(first) and first[3] == "pure=True", first[3] if first else "")
    check("fallback answered inside 8 s",
          bool(first) and float(first[1]) < 8.0, first[1] if first else "")
    check("no illegal move at any stage", "ILLEGAL" not in out)

    print("\n%d passed, %d failed" % (PASS, FAIL))
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
