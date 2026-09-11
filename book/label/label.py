"""Label book positions with Stockfish. Resumable, parallel, crash-safe.

    python label.py positions.txt labels.tsv --engine stockfish.exe
                    --movetime 2000 --workers 8

`positions.txt`  one FEN per line (from bookbuild.py's gap list, or the full
                 reach-probability expansion).
`labels.tsv`     fen<TAB>bestmove_uci<TAB>score_cp<TAB>depth, appended as it
                 goes. Re-running skips anything already in the file, so you can
                 stop and restart freely.

Use --movetime for a fixed wall-clock budget per position, or --depth for a
fixed depth. Depth is more consistent across machines; movetime is more
predictable in total runtime. For an opening book, depth 30 or 2000 ms are both
far beyond what the agent reaches in-game.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading


def uci_worker(engine, jobs, out, lock, movetime, depth, hash_mb, threads):
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1)

    def send(s):
        p.stdin.write(s + "\n")
        p.stdin.flush()

    def wait_for(token):
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            yield line.rstrip("\n")
            if line.startswith(token):
                return

    send("uci")
    for _ in wait_for("uciok"):
        pass
    send(f"setoption name Hash value {hash_mb}")
    send(f"setoption name Threads value {threads}")
    send("isready")
    for _ in wait_for("readyok"):
        pass

    while True:
        try:
            fen = jobs.get_nowait()
        except queue.Empty:
            break
        send("ucinewgame")
        send("isready")
        for _ in wait_for("readyok"):
            pass
        send(f"position fen {fen}")
        send(f"go movetime {movetime}" if depth is None else f"go depth {depth}")
        score = None
        seen_depth = 0
        best = None
        for line in wait_for("bestmove"):
            if line.startswith("info ") and " pv " in line:
                tok = line.split()
                try:
                    if "depth" in tok:
                        seen_depth = int(tok[tok.index("depth") + 1])
                    if "score" in tok:
                        i = tok.index("score")
                        if tok[i + 1] == "cp":
                            score = int(tok[i + 2])
                        elif tok[i + 1] == "mate":
                            m = int(tok[i + 2])
                            score = 100000 - abs(m) * 100
                            if m < 0:
                                score = -score
                except (ValueError, IndexError):
                    pass
            elif line.startswith("bestmove"):
                best = line.split()[1]
        if best and best != "(none)":
            with lock:
                out.write(f"{fen}\t{best}\t{score if score is not None else 0}"
                          f"\t{seen_depth}\n")
                out.flush()
        jobs.task_done()
    send("quit")
    p.wait(timeout=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positions")
    ap.add_argument("labels")
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--movetime", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--hash-mb", type=int, default=256)
    ap.add_argument("--engine-threads", type=int, default=1)
    a = ap.parse_args()

    want = [l.strip() for l in open(a.positions) if l.strip()]
    done = set()
    if os.path.exists(a.labels):
        with open(a.labels) as f:
            for line in f:
                parts = line.split("\t")
                if parts:
                    done.add(parts[0])
    todo = [f for f in want if f not in done]
    print(f"  {len(want):,} positions, {len(done):,} already labelled, "
          f"{len(todo):,} to go")
    if not todo:
        return

    per = a.movetime / 1000 if a.depth is None else 2.0
    print(f"  estimate: {len(todo)*per/3600/a.workers:.1f} h on {a.workers} workers")

    jobs = queue.Queue()
    for f in todo:
        jobs.put(f)
    out = open(a.labels, "a")
    lock = threading.Lock()
    ts = []
    for _ in range(a.workers):
        t = threading.Thread(target=uci_worker, args=(
            a.engine, jobs, out, lock, a.movetime, a.depth, a.hash_mb,
            a.engine_threads), daemon=True)
        t.start()
        ts.append(t)
    try:
        while any(t.is_alive() for t in ts):
            for t in ts:
                t.join(timeout=1)
            left = jobs.qsize()
            print(f"\r  {len(todo)-left:,}/{len(todo):,}", end="", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n  stopped; re-run to resume", file=sys.stderr)
    out.close()
    print()


if __name__ == "__main__":
    main()
