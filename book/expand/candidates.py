"""
Generate opponent candidate moves: a cheap shallow MultiPV pass.

This is what gives the book its width. It is deliberately shallow -- we are not
choosing a move to play here, only deciding which opponent replies are plausible
enough to branch on, and the ordering of the top few moves is stable well before
the depth we would need to trust one of them. Our own move still gets the deep
single-PV search in label.py.

    python candidates.py need_candidates.txt candidates.tsv \
        --engine data/engines/stockfish.exe --multipv 6 --depth 14 --workers 5

Output is `fen<TAB>uci,uci,uci,...` best-first, appended as it goes. Re-running
skips positions already present, so it is resumable like label.py.
"""

# Stage scripts live one level below book/, where paths.py is. Put it on
# the path so data files resolve to book/data/ from any directory.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from paths import engine  # noqa: E402


import argparse
import os
import queue
import subprocess
import sys
import threading


def worker(engine, jobs, out, lock, depth, multipv, hash_mb, done_counter):
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1)

    def send(s):
        p.stdin.write(s + "\n")
        p.stdin.flush()

    def wait(token):
        lines = []
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            lines.append(line.rstrip("\n"))
            if line.startswith(token):
                return lines

    send("uci")
    wait("uciok")
    send(f"setoption name Hash value {hash_mb}")
    send("setoption name Threads value 1")
    send(f"setoption name MultiPV value {multipv}")
    send("isready")
    wait("readyok")

    while True:
        try:
            fen = jobs.get_nowait()
        except queue.Empty:
            break
        send("ucinewgame")
        send("isready")
        wait("readyok")
        send(f"position fen {fen}")
        send(f"go depth {depth}")
        lines = wait("bestmove")
        # Keep the last line reported for each multipv slot: that is the
        # deepest iteration's opinion.
        pv = {}
        for line in lines:
            if line.startswith("info ") and " multipv " in line and " pv " in line:
                tok = line.split()
                try:
                    rank = int(tok[tok.index("multipv") + 1])
                    pv[rank] = tok[tok.index("pv") + 1]
                except (ValueError, IndexError):
                    pass
        moves = [pv[r] for r in sorted(pv)]
        if moves:
            with lock:
                out.write(f"{fen}\t{','.join(moves)}\n")
                out.flush()
                done_counter[0] += 1
    send("quit")
    try:
        p.wait(timeout=10)
    except Exception:                                        # noqa: BLE001
        p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positions")
    ap.add_argument("out")
    ap.add_argument("--engine", default=engine("stockfish.exe"))
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--multipv", type=int, default=6)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--hash-mb", type=int, default=32)
    a = ap.parse_args()

    want = [l.strip() for l in open(a.positions, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            for line in f:
                parts = line.split("\t")
                if parts:
                    done.add(parts[0])
    todo = [f for f in want if f not in done]
    print(f"  {len(want):,} positions, {len(done):,} already done, "
          f"{len(todo):,} to go")
    if not todo:
        return

    jobs = queue.Queue()
    for f in todo:
        jobs.put(f)
    out = open(a.out, "a", encoding="utf-8")
    lock = threading.Lock()
    counter = [0]
    ts = []
    for _ in range(a.workers):
        t = threading.Thread(target=worker, args=(
            a.engine, jobs, out, lock, a.depth, a.multipv, a.hash_mb, counter),
            daemon=True)
        t.start()
        ts.append(t)
    try:
        while any(t.is_alive() for t in ts):
            for t in ts:
                t.join(timeout=1)
            print(f"\r  {counter[0]:,}/{len(todo):,}", end="", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n  stopped; re-run to resume", file=sys.stderr)
    out.close()
    print()


if __name__ == "__main__":
    main()
