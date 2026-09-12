"""Tests for the Lazy SMP driver. Run: python tests/test_parallel.py

Two things are being checked, and the first matters more.

Safety: threads must refuse to start while the non-atomic reference counting
patch is active, because that combination is silent memory corruption rather
than a crash. The refusal has to be a fallback to one thread, not an
exception, since an exception during a live game loses it.

Function: a helper thread must actually run and actually share the table, and
the move the engine plays must still be the main thread's.

BTC_NRT is read when btc_nrt is imported, so each case runs in its own
process. Table size is passed to ParallelSearch explicitly: BTC_TT_ENTRIES is
read by agent.py alone, and these tests construct ParallelSearch directly, so
setting that variable here would silently allocate the full 256 MB table
twice in one process.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")

# 16 MB of table rather than 256 MB. Behaviour here does not depend on
# table size, and the machine usually has a training job on it.
TEST_TT = 1 << 20

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MIDDLE = "r1bqkb1r/pp1n1ppp/2p1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 0 7"


def _case_refuses_with_nrt():
    """BTC_NRT on: four threads requested, one thread allowed."""
    import btc_nrt  # noqa: F401
    import btc_parallel

    assert btc_nrt._APPLIED, "expected the refcount patch to be active"
    allowed = btc_parallel.threads_allowed(4)
    print("RESULT allowed={}".format(allowed))
    assert allowed == 1, "threads were allowed while BTC_NRT was active"

    par = btc_parallel.ParallelSearch(4, TEST_TT)
    assert par.threads == 1
    assert par.helpers == []
    print("RESULT degraded_ok=1")


def _search(par, fen, soft, hard):
    import btc_core as core
    import btc_parallel
    import numpy as np

    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    keys = np.zeros(8, dtype=np.uint64)
    keys[0] = bb[core.HASH]
    return btc_parallel.search_position(par, bb, st, keys, 1, soft, hard)


def _case_threads_run():
    """BTC_NRT off: helpers start, share the table, and add nodes."""
    import btc_core as core  # noqa: F401
    import btc_nrt
    import btc_parallel

    assert not btc_nrt._APPLIED, "expected the refcount patch to be inactive"

    one = btc_parallel.ParallelSearch(1, TEST_TT)
    assert one.threads == 1

    two = btc_parallel.ParallelSearch(2, TEST_TT)
    assert two.threads == 2, "two threads were not allowed"
    helper = two.helpers[0]
    assert helper.tt_key is two.main.tt_key, "helper did not share tt_key"
    assert helper.tt_data is two.main.tt_data, "helper did not share tt_data"
    assert helper.undo_bb is not two.main.undo_bb, "helper shared undo_bb"
    assert helper.acc is not two.main.acc, "helper shared the accumulator"
    print("RESULT helper_mb={:.2f}".format(two.helper_bytes() / (1024 * 1024)))

    # Compile before measuring anything. The first search in a process pays
    # the numba compile with its deadline already running, so it returns
    # having searched almost nothing: an uncompiled single threaded run
    # reported 2 nodes, which would make the comparison below pass without
    # any helper ever starting.
    _search(one, MIDDLE, 200, 400)

    for fen in (START, MIDDLE):
        solo = _search(one, fen, 600, 1200)
        pair = _search(two, fen, 600, 1200)
        assert solo[0] != 0, "single threaded search returned no move"
        assert pair[0] != 0, "parallel search returned no move"
        # The helper's nodes are summed in, so the parallel figure has to be
        # larger. If it is not, the thread never ran.
        assert pair[3] > solo[3], (
            "parallel nodes {} not above single {} - helper did not run"
            .format(pair[3], solo[3]))
        print("RESULT fen_ok nodes_one={} nodes_two={}".format(solo[3], pair[3]))

    # Repeat on one position to shake out a race that only shows on reuse of
    # the same states and the same table.
    for _ in range(4):
        again = _search(two, MIDDLE, 300, 600)
        assert again[0] != 0, "parallel search returned no move on repeat"
    print("RESULT repeat_ok=1")


def _case_legal_moves():
    """Whatever the parallel driver returns must be a legal move."""
    import chess
    import btc_core as core
    import btc_parallel

    par = btc_parallel.ParallelSearch(2, TEST_TT)
    assert par.threads == 2
    for fen in (START, MIDDLE):
        move = _search(par, fen, 400, 800)[0]
        uci = core.move_to_uci(move)
        board = chess.Board(fen)
        assert chess.Move.from_uci(uci) in board.legal_moves, (
            "illegal move {} in {}".format(uci, fen))
        print("RESULT legal {} {}".format(fen.split()[0][:12], uci))


CASES = {
    "refuses": _case_refuses_with_nrt,
    "threads": _case_threads_run,
    "legal": _case_legal_moves,
}


def _run_child(name, nrt):
    env = dict(os.environ)
    env["BTC_NRT"] = nrt
    env["BTC_TT_ENTRIES"] = "1048576"
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", name],
        capture_output=True, text=True, env=env, timeout=1800)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            print("    " + line[7:])
    if proc.returncode != 0:
        print(proc.stdout[-1500:])
        print(proc.stderr[-1500:])
    return proc.returncode == 0


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        sys.path.insert(0, SRC)
        CASES[sys.argv[2]]()
        return 0

    print("parallel search suite")
    plan = [
        ("refuses threads while BTC_NRT is active", "refuses", "1"),
        ("helpers run and share the table", "threads", "0"),
        ("parallel result is always legal", "legal", "0"),
    ]
    failed = 0
    for label, name, nrt in plan:
        print("  " + label)
        if not _run_child(name, nrt):
            print("  FAIL: " + label)
            failed += 1
    print("")
    if failed:
        print("{} of {} cases failed".format(failed, len(plan)))
        return 1
    print("all {} cases passed".format(len(plan)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
