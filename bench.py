"""Benchmark harness: cold import time, perft NPS and memory. Run: python bench.py

Cold import is measured in a fresh subprocess because numba compilation state
survives within a process. The platform recompiles every game, so cold import
plus warmup must stay well under the 90 s init budget (local ceiling 65 s).
"""

import subprocess
import sys
import time

COLD_SNIPPET = (
    "import time; t = time.perf_counter(); "
    "import btc_core; btc_core.warmup(); "
    "print(f'{time.perf_counter() - t:.2f}')"
)


def measure_cold_import():
    out = subprocess.run([sys.executable, "-c", COLD_SNIPPET],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip().splitlines()[-1])


def measure_perft(fen, depth, expected):
    import btc_core as core
    bb, st = core.new_board()
    undo_bb, undo_st, mls = core.new_stacks()
    core.parse_fen(fen, bb, st)
    t0 = time.perf_counter()
    nodes = core.perft(bb, st, undo_bb, undo_st, mls, depth, 0)
    dt = time.perf_counter() - t0
    assert nodes == expected, f"perft mismatch: {nodes} != {expected}"
    return nodes, dt


def rss_mb():
    import ctypes
    import ctypes.wintypes as wt

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    kernel32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ok = kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                          ctypes.byref(pmc), pmc.cb)
    if not ok:
        raise OSError("K32GetProcessMemoryInfo failed")
    return pmc.WorkingSetSize / (1024 * 1024)


def main():
    cold = measure_cold_import()

    import btc_core as core
    core.warmup()
    kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    n1, t1 = measure_perft(core.START_FEN, 5, 4865609)
    n2, t2 = measure_perft(kiwipete, 4, 4085603)
    nps = (n1 + n2) / (t1 + t2)

    try:
        mem = f"{rss_mb():.0f} MB"
    except Exception:
        mem = "n/a"

    print(f"cold import + warmup: {cold:.2f}s (ceiling 65s)")
    print(f"perft(5) startpos:    {n1} nodes in {t1:.2f}s")
    print(f"perft(4) kiwipete:    {n2} nodes in {t2:.2f}s")
    print(f"perft NPS:            {nps:,.0f}")
    print(f"working set:          {mem}")


if __name__ == "__main__":
    main()
