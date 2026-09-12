"""Measure what the agent actually costs in memory, stage by stage.

The platform gives 2 GB. If that is cut on finals day the only question worth
answering is which knob to turn and how far, so this reports the real curve
instead of an estimate: peak working set and peak commit for a full agent
import plus a short game, at a range of transposition table sizes.

Run the whole sweep:

    .venv/Scripts/python.exe finals_day/mem_probe.py

One size only, printing the per-stage breakdown:

    .venv/Scripts/python.exe finals_day/mem_probe.py --child 4194304

Peak, not current, is the number that matters: an allocation that is freed
again still has to fit at the moment it exists.
"""

import ctypes
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
MB = 1024.0 * 1024.0

# 1 << 24 is the shipped size, 256 MiB of table. Each step halves it.
SIZES = [1 << 24, 1 << 23, 1 << 22, 1 << 21, 1 << 20, 1 << 18]


class _MemCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
# argtypes are not optional here. Without them ctypes passes the process
# handle as a 32 bit int, the call fails on x64, and every reading is zero.
_KERNEL32.GetCurrentProcess.restype = ctypes.c_void_p
_KERNEL32.GetCurrentProcess.argtypes = []
_KERNEL32.K32GetProcessMemoryInfo.restype = ctypes.c_int
_KERNEL32.K32GetProcessMemoryInfo.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(_MemCounters), ctypes.c_uint32]


def _mem():
    """Working set and commit for this process, current and peak, in MiB."""
    counters = _MemCounters()
    counters.cb = ctypes.sizeof(_MemCounters)
    handle = _KERNEL32.GetCurrentProcess()
    ok = _KERNEL32.K32GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return {
        "rss": counters.WorkingSetSize / MB,
        "peak_rss": counters.PeakWorkingSetSize / MB,
        "commit": counters.PagefileUsage / MB,
        "peak_commit": counters.PeakPagefileUsage / MB,
    }


def _child(tt_entries):
    """Import the agent at a given table size and play, reporting each stage."""
    stages = []

    def mark(name):
        row = _mem()
        row["stage"] = name
        stages.append(row)

    mark("interpreter")

    import numpy  # noqa: F401
    mark("numpy")

    import numba  # noqa: F401
    mark("numba")

    sys.path.insert(0, SRC)
    if tt_entries:
        os.environ["BTC_TT_ENTRIES"] = str(tt_entries)

    started = time.perf_counter()
    import agent
    import_s = time.perf_counter() - started
    mark("agent imported")

    import chess
    board = chess.Board()
    for _ in range(12):
        if board.is_game_over():
            break
        uci = agent.get_move(board.fen(), 60000)
        board.push(chess.Move.from_uci(uci))
    mark("12 plies played")

    print("PROBE " + json.dumps({
        "tt_entries": tt_entries,
        "import_s": round(import_s, 2),
        "stages": stages,
    }))


def _run_one(tt_entries):
    """Run the probe in a fresh process so nothing carries over between sizes."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", str(tt_entries)],
        capture_output=True, text=True, env=env, timeout=900)
    for line in proc.stdout.splitlines():
        if line.startswith("PROBE "):
            return json.loads(line[6:])
    sys.stdout.write(proc.stdout[-2000:])
    sys.stderr.write(proc.stderr[-2000:])
    return None


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        _child(int(sys.argv[2]))
        return 0

    print("agent memory by transposition table size")
    print("")
    header = "{:>12}  {:>9}  {:>9}  {:>9}  {:>8}".format(
        "tt entries", "tt MiB", "peak rss", "peak commit", "import s")
    print(header)
    print("-" * len(header))

    results = []
    for size in SIZES:
        row = _run_one(size)
        if row is None:
            print("{:>12}  probe failed".format(size))
            continue
        peak = max(s["peak_rss"] for s in row["stages"])
        commit = max(s["peak_commit"] for s in row["stages"])
        print("{:>12}  {:>9.1f}  {:>9.1f}  {:>9.1f}  {:>8.1f}".format(
            size, size * 16.0 / MB, peak, commit, row["import_s"]))
        results.append((size, peak, commit))
        sys.stdout.flush()

    if results:
        print("")
        print("stage breakdown at the smallest size tested")
        last = _run_one(SIZES[-1])
        if last is not None:
            for stage in last["stages"]:
                print("  {:<18} rss {:>7.1f}  commit {:>7.1f}".format(
                    stage["stage"], stage["rss"], stage["commit"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
