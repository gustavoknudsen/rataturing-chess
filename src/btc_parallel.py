"""Lazy SMP: N searches of the same position sharing one transposition table.

Off unless BTC_THREADS is set above 1, and the platform gives one core, so
this exists for the case where that changes. Nothing here is reachable at
BTC_THREADS=1, which is the shipped configuration.

The mechanism is the shared table and nothing else. Helper threads search the
same root at skewed depths, and every useful bound, best move and exact score
they prove lands in the table where the main thread reads it. The main thread
alone decides the move, so a helper can never make the engine play something
it did not choose.

Safety, which matters more here than speed:

btc_nrt rewrites numba's reference counting to be non-atomic. That is worth 32
percent single-threaded and it is memory corruption the instant a second
thread runs compiled code, because every negamax call increfs and decrefs its
array arguments. This module refuses to start threads while that patch is
active, and the refusal is silent degradation to one thread rather than an
exception: losing the parallel speedup is survivable, and an exception during
a live game is not.

So a parallel build must run with BTC_NRT=0, and gives back 32 percent to do
it. Parallel has to beat that before it is worth taking, which is a question
for measurement, not for this docstring.
"""

import os
import threading

import numpy as np

import btc_nrt
import btc_search as search
from btc_search import SC_ROOT_DEPTH, SC_ROOT_SCORE, SC_STOP, SC_TT_GEN

THREADS = int(os.environ.get("BTC_THREADS", "1"))

# Helpers share the table, so their own allocation only has to be a legal
# shape: it is replaced immediately after construction. SearchState requires a
# power of two of at least 8, so 8 is the cheapest stub that is accepted.
_STUB_TT = 8


def threads_allowed(requested):
    """How many threads may actually run, given the reference counting patch."""
    if requested <= 1:
        return 1
    if getattr(btc_nrt, "_APPLIED", False):
        print("parallel: refusing threads because BTC_NRT is active, "
              "set BTC_NRT=0 to enable them", flush=True)
        return 1
    return requested


class ParallelSearch:
    """One SearchState per thread, all sharing the main thread's table.

    Each thread also needs its own board: negamax makes and unmakes moves in
    place, so sharing bb or st across threads would have them trample each
    other's position rather than merely each other's history tables.
    """

    def __init__(self, requested=THREADS, tt_entries=None):
        self.threads = threads_allowed(requested)
        # Whether a second thread may run at all, which is a different
        # question from how many search threads were asked for: pondering
        # wants one background thread beside a single-threaded search, and it
        # is governed by the same reference counting rule.
        self.threads_possible = not getattr(btc_nrt, "_APPLIED", False)
        self.main = search.SearchState(tt_entries or search.TT_ENTRIES)
        self.helpers = []
        for _ in range(self.threads - 1):
            helper = search.SearchState(_STUB_TT)
            helper.tt_key = self.main.tt_key
            helper.tt_data = self.main.tt_data
            self.helpers.append(helper)

    def helper_bytes(self):
        """Memory one helper costs, excluding the table it shares."""
        total = 0
        for name in ("undo_bb", "undo_st", "mls", "scores", "killers",
                     "main_hist", "cap_hist", "cont_hist", "counters", "corr",
                     "played", "static_evals", "pv_table", "pv_len", "rep",
                     "acc", "sc", "fc"):
            total += getattr(self.helpers[0], name).nbytes
        return total


def _prep_helper(state, generation, rep_keys, rep_base, history, deadline):
    """Give a helper the same starting knowledge the main thread gets.

    Mirrors the setup half of search_position. The accumulator refresh is the
    part that cannot be skipped: every deeper ply is derived from its parent,
    so ply 0 is the only place it is ever built from the board.
    """
    state.sc[search.SC_NODES] = 0
    state.sc[SC_STOP] = 0
    state.sc[search.SC_NMP_MIN] = 0
    state.sc[search.SC_CORR_ADJ] = 0
    state.sc[SC_ROOT_DEPTH] = 0
    state.sc[SC_ROOT_SCORE] = 0
    state.sc[search.SC_BEST_NODES] = 0
    state.sc[SC_TT_GEN] = generation
    state.killers[:] = 0
    state.pv_table[:] = 0
    state.pv_len[:] = 0
    state.static_evals[:] = 0
    state.played[:] = 0
    state.fc[0] = deadline
    state.rep[:rep_base] = rep_keys[history - rep_base:history]
    state.sc[search.SC_REP_BASE] = rep_base


def _helper_loop(state, bb, st, rep_base, max_depth, skew):
    """Deepen until stopped, filling the shared table.

    No soft-time check: a helper has no reason to stop early, since its only
    product is table entries and it is not deciding anything. It runs to the
    hard deadline or until the main thread stops it.

    The skew is what makes a helper worth having. Threads all searching the
    identical depth in the identical order would mostly duplicate work; an
    offset start means they reach different depths at different moments and
    prove different bounds. Skews repeat once there are more than four
    helpers, which only matters above five cores.
    """
    if search.USE_NNUE:
        search.refresh_acc(bb, search.NET_FT_W, search.NET_FT_B, search.NET_L1,
                           state.acc[0], search.NET_TABLE, search.NET_BUCKETS)
    prev_score = 0
    depth = 1 + skew
    while depth <= max_depth:
        if state.sc[SC_STOP]:
            return
        score = search._aspiration_search(state, bb, st, prev_score, depth,
                                          rep_base)
        if state.sc[SC_STOP]:
            return
        prev_score = score
        state.sc[SC_ROOT_DEPTH] = depth
        state.sc[SC_ROOT_SCORE] = score
        depth += 1


def _stop(state):
    """Stop a helper. The deadline is moved as well as the flag because the
    flag is only read at unwind points, while the deadline is what the node
    level time check consults."""
    state.fc[0] = 0.0
    state.sc[SC_STOP] = 1


def search_position(par, bb, st, rep_keys, rep_count, soft_ms, hard_ms,
                    max_depth=search.MAX_SEARCH_PLY):
    """Parallel front end with the same contract as search.search_position.

    Returns (best_move, score, depth, nodes) from the main thread, with nodes
    summed across every thread so the figure still means work done.
    """
    if par.threads <= 1:
        return search.search_position(par.main, bb, st, rep_keys, rep_count,
                                      soft_ms, hard_ms, max_depth)

    history = max(int(rep_count) - 1, 0)
    rep_base = min(history, 1024)
    # search_position increments the main state's generation, so helpers are
    # given the value it is about to hold rather than the one it holds now.
    generation = int(par.main.sc[SC_TT_GEN]) + 1
    deadline = search.time.perf_counter() + hard_ms / 1000.0

    workers = []
    for index, helper in enumerate(par.helpers):
        _prep_helper(helper, generation, rep_keys, rep_base, history, deadline)
        worker = threading.Thread(
            target=_helper_loop,
            args=(helper, bb.copy(), st.copy(), rep_base, max_depth,
                  1 + index % 4),
            daemon=True)
        worker.start()
        workers.append(worker)

    try:
        result = search.search_position(par.main, bb, st, rep_keys, rep_count,
                                        soft_ms, hard_ms, max_depth)
    finally:
        for helper in par.helpers:
            _stop(helper)
        stuck = False
        for worker in workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                stuck = True
        if stuck:
            # A helper that outlived its stop signal must never be handed its
            # SearchState again: the next search would reset that state and
            # start a second thread on it, and two threads sharing one acc,
            # undo stack and board is the corruption this module exists to
            # avoid. Degrade permanently and keep playing.
            print("parallel: a helper did not stop, dropping to one thread",
                  flush=True)
            par.threads = 1
            par.helpers = []

    move, score, depth, nodes = result
    for helper in par.helpers:
        nodes += int(helper.sc[search.SC_NODES])
    return move, score, depth, nodes


class Ponderer:
    """Keep searching the position we expect next, into the shared table.

    Off unless BTC_PONDER=1, and worthless under the tournament rules, which
    say both agents run on the same machine and take the core in turns: there
    is no CPU to think with while the opponent thinks. It exists for the case
    where that condition is the one that changes, because then it is worth
    far more than parallel search is. Pondering roughly doubles thinking time,
    against the 32 percent that turning BTC_NRT off costs.

    Deliberately not UCI-style ponderhit. There is no predicted-move
    bookkeeping and no committing to a line: the thread simply searches the
    predicted position and everything it proves lands in the shared table. A
    correct prediction means the next real search starts with a warm table; a
    wrong one means some table entries for a position that did not occur,
    which is what a transposition table is already full of. Nothing downstream
    has to know whether the guess was right, which removes the entire class of
    bugs that ponderhit handling usually brings.
    """

    def __init__(self, par):
        self.par = par
        self.state = None
        self.thread = None
        if par.threads_possible:
            self.state = search.SearchState(_STUB_TT)
            self.state.tt_key = par.main.tt_key
            self.state.tt_data = par.main.tt_data

    def start(self, bb, st, rep_keys, rep_count, generation):
        """Begin pondering a position. Safe to call when disabled."""
        if self.state is None or self.thread is not None:
            return
        history = max(int(rep_count) - 1, 0)
        rep_base = min(history, 1024)
        # No deadline: the thread runs until stop() is called, which happens
        # before the next real search. A deadline would only make it stop
        # early and waste the opponent's clock.
        _prep_helper(self.state, generation, rep_keys, rep_base, history,
                     search.time.perf_counter() + 3600.0)
        self.thread = threading.Thread(
            target=_helper_loop,
            args=(self.state, bb, st, rep_base, search.MAX_SEARCH_PLY, 0),
            daemon=True)
        self.thread.start()

    def stop(self):
        """Stop pondering. Must complete before any real search begins."""
        if self.thread is None:
            return
        _stop(self.state)
        self.thread.join(timeout=1.0)
        alive = self.thread.is_alive()
        self.thread = None
        if alive:
            # Same rule as a stuck helper: never hand a live thread's state
            # back to a second thread. Give up pondering for the rest of the
            # game rather than risk two threads on one accumulator.
            print("parallel: ponder thread did not stop, pondering off",
                  flush=True)
            self.state = None
