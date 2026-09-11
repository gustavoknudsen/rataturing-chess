"""UCI protocol gate. Run: python tests/test_uci.py [path-to-exe]

With no argument it drives uci/rataturing_uci.py from source. Given a path it
drives that executable instead, which is how a built release is checked.

Startup compiles the engine, so a run takes a few minutes.

MIDGAME is past the book's move-20 bound, so the book can never answer it and
a search always runs. Positions inside the book are answered instantly with no
info lines, which looks exactly like a broken search if the test does not
account for it.
"""

import os
import subprocess
import sys
import threading
import time

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
ADAPTER = os.path.join(REPO, "uci", "rataturing_uci.py")
MIDGAME = "2r3k1/1q3pp1/p2p1n1p/1p1Pp3/4P3/1P2BP2/P1Q3PP/2R3K1 w - - 0 25"
MATE = "3k4/R7/1R6/8/8/8/8/4K3 w - - 0 1"

PASS, FAIL = 0, 0
LINES = []
LOCK = threading.Lock()
PROC = None
STARTED = 0.0


def check(name, ok, detail=""):
    global PASS, FAIL
    print("%-44s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if ok:
        PASS += 1
    else:
        FAIL += 1


def send(*commands):
    for text in commands:
        PROC.stdin.write(text + "\n")
    PROC.stdin.flush()


def wait_for(prefix, timeout):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        with LOCK:
            for stamp, line in LINES:
                if line.startswith(prefix):
                    return stamp, line
        time.sleep(0.01)
    return None, None


def collect(prefix):
    with LOCK:
        return [line for _, line in LINES if line.startswith(prefix)]


def reset():
    with LOCK:
        del LINES[:]


def depths():
    return [int(line.split()[2]) for line in collect("info depth")]


def test_handshake():
    """uci must answer before the engine has compiled, or a GUI waits a
    minute just to list the engine."""
    send("uci")
    stamp, line = wait_for("uciok", 10)
    check("uciok arrives", line is not None)
    check("uciok is immediate", stamp is not None and stamp - STARTED < 3.0,
          "%.2fs" % (stamp - STARTED) if stamp else "timeout")
    names = collect("id name")
    check("id name present",
          bool(names) and names[0].startswith("id name Rataturing"),
          names[0] if names else "")
    check("options declared", len(collect("option name")) == 3,
          "%d options" % len(collect("option name")))


def test_ready():
    send("setoption name Hash value 64", "setoption name Move Overhead value 10",
         "isready")
    stamp, line = wait_for("readyok", 400)
    check("readyok arrives", line is not None,
          "%.1fs" % (stamp - STARTED) if stamp else "timeout")
    return line is not None


def test_timed_search():
    reset()
    send("ucinewgame", "position fen " + MIDGAME)
    sent = time.perf_counter()
    send("go movetime 3000")
    stamp, best = wait_for("bestmove", 60)
    check("bestmove after movetime", best is not None, best or "timeout")
    check("movetime is respected", stamp is not None and stamp - sent < 6.0,
          "%.2fs" % (stamp - sent) if stamp else "timeout")
    infos = collect("info depth")
    check("info lines emitted", len(infos) >= 3, "%d lines" % len(infos))
    check("info carries a pv", any(" pv " in line for line in infos))
    check("info carries nps and time",
          bool(infos) and "nps " in infos[-1] and "time " in infos[-1])
    seen = depths()
    check("depths increase", seen == sorted(seen) and len(set(seen)) > 2,
          str(seen[:14]))


def test_depth_limit():
    """The limit is applied by refusing to start depth N+1, not by stopping
    after depth N. Stopping returns depth N-1's move, because the driver tests
    the stop flag before recording the new best move."""
    reset()
    send("position fen " + MIDGAME, "go depth 8")
    stamp, best = wait_for("bestmove", 180)
    seen = depths()
    check("go depth stops at the asked depth", bool(seen) and max(seen) == 8,
          "reached %s" % (seen or "none"))
    check("bestmove after go depth", best is not None, best or "timeout")


def test_infinite_and_stop():
    reset()
    send("position fen " + MIDGAME, "go infinite")
    time.sleep(5.0)
    stamp, best = wait_for("bestmove", 0.1)
    check("go infinite keeps searching", best is None, best or "")
    sent = time.perf_counter()
    send("stop")
    stamp, best = wait_for("bestmove", 15)
    check("stop returns a bestmove", best is not None, best or "timeout")
    check("stop is prompt", stamp is not None and 0 <= stamp - sent < 2.0,
          "%.2fs" % (stamp - sent) if stamp else "timeout")


def test_mate_score():
    """A mate reported as cp 47999 is indistinguishable from a huge advantage
    in every GUI."""
    reset()
    send("ucinewgame", "position fen " + MATE, "go depth 10")
    stamp, best = wait_for("bestmove", 180)
    infos = collect("info depth")
    check("mate is scored as mate",
          any("score mate" in line for line in infos),
          infos[-1][:66] if infos else "no info")
    check("mate position returns a move", best is not None, best or "timeout")


def test_book():
    reset()
    send("ucinewgame", "position startpos")
    sent = time.perf_counter()
    send("go movetime 4000")
    stamp, best = wait_for("bestmove", 40)
    elapsed = stamp - sent if stamp else 99
    check("book answers the start position",
          best is not None and elapsed < 1.0,
          "%s in %.2fs" % (best, elapsed) if best else "timeout")
    check("book move is announced", bool(collect("info string book move")))

    reset()
    send("setoption name OwnBook value false", "position startpos",
         "go movetime 4000")
    stamp, best = wait_for("bestmove", 40)
    check("OwnBook false forces a search",
          best is not None and len(collect("info depth")) >= 3,
          "%s, %d info lines" % (best, len(collect("info depth"))))
    send("setoption name OwnBook value true")


def test_hash_change():
    reset()
    send("setoption name Hash value 128", "ucinewgame",
         "position fen " + MIDGAME, "go movetime 2000")
    stamp, best = wait_for("bestmove", 40)
    check("search works after a hash change", best is not None,
          best or "timeout")


def main():
    global PROC, STARTED
    command = sys.argv[1:] or [sys.executable, "-u", ADAPTER]
    print("driving: %s\n" % command[-1])
    PROC = subprocess.Popen(command, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def reader():
        for line in PROC.stdout:
            with LOCK:
                LINES.append((time.perf_counter(), line.rstrip()))

    threading.Thread(target=reader, daemon=True).start()
    STARTED = time.perf_counter()

    test_handshake()
    if not test_ready():
        PROC.kill()
        raise SystemExit("engine never became ready")
    for test in (test_timed_search, test_depth_limit, test_infinite_and_stop,
                 test_mate_score, test_book, test_hash_change):
        test()

    send("quit")
    PROC.wait(timeout=20)
    print("\n%d passed, %d failed" % (PASS, FAIL))
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
