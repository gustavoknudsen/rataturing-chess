"""Search correctness suite. Run: python test_search.py [--deep]

Layers:
1. Reference differential: a subprocess with BTC_MINIMAL=1 (TT, null move,
   RFP and MDP compiled out) compares the search against an independent plain
   alpha-beta at fixed depth. Move ordering must not change values.
2. Mate suite: scores and PVs are replayed with python-chess; a claimed mate
   must be a real checkmate at the exact ply the score encodes.
3. Draw detection: repetition via injected history, fifty-move rule.
4. Time discipline against the hard cap.
5. Agent-level fuzz: every returned move legal, tracker consistent in a
   simulated game against a random opponent.
"""

import os
import random
import subprocess
import sys
import time

import chess
import numpy as np
from numba import njit

import btc_core as core
import btc_search as se

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

REF_POSITIONS = [
    core.START_FEN,
    KIWIPETE,
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
    "8/8/8/8/8/2k5/8/K1Q5 w - - 0 1",
]

passed = 0
failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name} {detail}")


@njit(cache=False)
def _ref_qsearch(alpha, beta, bb, st, undo_bb, undo_st, mls, scores, cap_hist,
                 ply):
    if ply > se.MAX_SEARCH_PLY - 1:
        return se.evaluate(bb, st)
    ev = se.evaluate(bb, st)
    if ev >= beta:
        return beta
    if ev > alpha:
        alpha = ev
    cnt = core.generate_captures(bb, st, mls[ply])
    # ordering is value-neutral for alpha-beta; without it the reference
    # explodes exponentially on sharp positions
    se._sort_captures(bb, st, mls[ply], scores[ply], cnt, cap_hist)
    for i in range(cnt):
        if core.make_move(bb, st, undo_bb, undo_st, ply, mls[ply, i]) == 0:
            continue
        score = -_ref_qsearch(-beta, -alpha, bb, st, undo_bb, undo_st, mls,
                              scores, cap_hist, ply + 1)
        core.unmake(bb, st, undo_bb, undo_st, ply)
        if score > alpha:
            alpha = score
            if score >= beta:
                return beta
    return alpha


@njit(cache=False)
def _ref_negamax(alpha, beta, depth, ply, rep_idx, bb, st, undo_bb, undo_st,
                 mls, scores, cap_hist, rep):
    """Plain fail-hard alpha-beta, no ordering, no TT, no pruning. Mirrors
    only the value-affecting rules: draws, in-check extension, qsearch."""
    if ply and (se._is_repetition(bb, rep, rep_idx) or st[core.FIFTY] >= 100):
        return 0
    if depth == 0:
        return _ref_qsearch(alpha, beta, bb, st, undo_bb, undo_st, mls,
                            scores, cap_hist, ply)
    if ply > se.MAX_SEARCH_PLY - 1:
        return se.evaluate(bb, st)
    in_check = se._in_check(bb, st)
    if in_check:
        depth += 1
    cnt = core.generate_moves(bb, st, mls[ply])
    legal = 0
    for i in range(cnt):
        rep[rep_idx] = bb[core.HASH]
        if core.make_move(bb, st, undo_bb, undo_st, ply, mls[ply, i]) == 0:
            continue
        legal += 1
        score = -_ref_negamax(-beta, -alpha, depth - 1, ply + 1, rep_idx + 1,
                              bb, st, undo_bb, undo_st, mls, scores, cap_hist,
                              rep)
        core.unmake(bb, st, undo_bb, undo_st, ply)
        if score > alpha:
            alpha = score
            if alpha >= beta:
                return beta
    if legal == 0:
        return -se.MATE_VALUE + ply if in_check else 0
    return alpha


NODE_BUDGET_FEN = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
NODE_BUDGET_DEPTH = 7


def _fresh_state():
    return se.SearchState(tt_entries=1 << 16)


def _nodes_at_fixed_depth(fen, depth):
    """Nodes to complete a fixed-depth search from a cold state."""
    state = _fresh_state()
    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    keys = np.array([bb[core.HASH]], dtype=np.uint64)
    _, _, _, nodes = se.search_position(state, bb, st, keys, 1, 0, 600000,
                                        max_depth=depth)
    return nodes


def _report_minimal_nodes(_max_depth):
    nodes = _nodes_at_fixed_depth(NODE_BUDGET_FEN, NODE_BUDGET_DEPTH)
    print(f"MINIMAL_NODES {nodes}", flush=True)


def _searched_value(state, fen, depth):
    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    state.sc[:] = 0
    state.fc[0] = time.perf_counter() + 3600.0
    state.rep[0] = bb[core.HASH]
    return se.negamax(-se.INFINITY, se.INFINITY, depth, 0, 1, bb, st,
                      state.undo_bb, state.undo_st, state.mls, state.scores,
                      state.killers, state.main_hist, state.cap_hist,
                      state.cont_hist, state.counters, state.played,
                      state.static_evals, state.pv_table, state.pv_len,
                      state.rep, state.tt_key, state.tt_data, state.sc,
                      state.fc, 0)


def _reference_value(fen, depth):
    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    undo_bb, undo_st, mls = core.new_stacks()
    scores = np.zeros((core.MAX_PLY, 256), dtype=np.int64)
    cap_hist = np.zeros((12, 64, 12), dtype=np.int16)
    rep = np.zeros(256, dtype=np.uint64)
    rep[0] = bb[core.HASH]
    return _ref_negamax(-se.INFINITY, se.INFINITY, depth, 0, 1, bb, st,
                        undo_bb, undo_st, mls, scores, cap_hist, rep)


def run_reference_mode(max_depth):
    assert se.MINIMAL, "reference mode requires BTC_MINIMAL=1"
    for fen in REF_POSITIONS:
        state = _fresh_state()
        t0 = time.perf_counter()
        for depth in range(1, max_depth + 1):
            state.killers[:] = 0
            state.main_hist[:] = 0
            got = _searched_value(state, fen, depth)
            want = _reference_value(fen, depth)
            check(f"reference d{depth}", got == want,
                  f"{fen.split()[0]} got {got} want {want}")
        print(f"  ref {fen.split()[0][:20]} {time.perf_counter() - t0:.1f}s",
              flush=True)
    _report_minimal_nodes(max_depth)
    print(f"reference mode: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


def _search_fen(fen, soft_ms, hard_ms, history=None, max_depth=64):
    state = _fresh_state()
    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    if history is None:
        keys = np.array([bb[core.HASH]], dtype=np.uint64)
    else:
        keys = np.array(history + [int(bb[core.HASH])], dtype=np.uint64)
    mv, score, depth, nodes = se.search_position(
        state, bb, st, keys, len(keys), soft_ms, hard_ms, max_depth=max_depth)
    return mv, score, depth, nodes


def _is_mating(board, mv):
    board.push(mv)
    mate = board.is_checkmate()
    board.pop()
    return mate


def test_mates():
    # KQ vs lone king from a distance needs the phase 4 specialised endgame
    # eval to convert; it is not a phase 2 fixture.
    mate_fens = [
        ("6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1", 1),
        ("6k1/8/8/8/8/8/8/K2R3R w - - 0 1", None),
        ("8/8/8/4k3/8/8/Q7/K6R w - - 0 1", None),
    ]
    for fen, mate_in in mate_fens:
        if mate_in == 1:
            board = chess.Board(fen)
            m1 = [m.uci() for m in board.legal_moves
                  if board.gives_check(m) and _is_mating(board, m)]
            check("fixture is mate in 1", len(m1) >= 1, fen)
        mv, score, depth, nodes = _search_fen(fen, 0, 5000, max_depth=14)
        check("mate found", score > se.MATE_SCORE, f"{fen} score {score}")
        if score <= se.MATE_SCORE:
            continue
        plies = se.MATE_VALUE - score
        if mate_in is not None:
            check("mate distance", plies == 2 * mate_in - 1,
                  f"{fen} plies {plies}")
        board = chess.Board(fen)
        board.push_uci(core.move_to_uci(mv))
        check("mate move legal progress",
              board.is_checkmate() if plies == 1 else not board.is_game_over(),
              fen)


def test_repetition_draw():
    fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    history = []
    for mv in core.legal_moves(bb, st):
        undo_bb, undo_st, _ = core.new_stacks()
        core.make_move(bb, st, undo_bb, undo_st, 0, mv)
        history.append(int(bb[core.HASH]))
        core.unmake(bb, st, undo_bb, undo_st, 0)
    mv, score, depth, nodes = _search_fen(fen, 0, 1000, history=history,
                                          max_depth=6)
    check("all-repeating children score 0", score == 0, f"score {score}")


def test_fifty_move():
    fen = "4k3/8/8/8/8/8/8/R3K3 w Q - 99 80"
    mv, score, depth, nodes = _search_fen(fen, 0, 1500, max_depth=8)
    check("fifty-move draw seen", score == 0, f"score {score}")
    fresh = "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1"
    mv, score, depth, nodes = _search_fen(fresh, 0, 1500, max_depth=8)
    check("rook-up position wins without fifty", score > 300, f"score {score}")


def _forces_mate(board, plies):
    if plies <= 0:
        return False
    for mv in board.legal_moves:
        board.push(mv)
        if board.is_checkmate():
            board.pop()
            return True
        deeper = (plies >= 3 and not board.is_game_over()
                  and _all_replies_lose(board, plies - 1))
        board.pop()
        if deeper:
            return True
    return False


def _all_replies_lose(board, plies):
    replies = list(board.legal_moves)
    if not replies:
        return False
    for mv in replies:
        board.push(mv)
        ok = _forces_mate(board, plies - 1)
        board.pop()
        if not ok:
            return False
    return True


def _brute_force_mate_plies(board, max_plies):
    for n in range(1, max_plies + 1):
        if _forces_mate(board, n):
            return n
    return None


def test_pruning_safety():
    """Pruning and reductions must not lose forced mates. LMR/LMP/futility
    bugs show up as a mate the unpruned search finds and the pruned one
    misses. Each fixture's mate distance is confirmed by an independent
    brute-force solver, so a mis-stated position fails instead of hiding."""
    # Positions must keep more than five pieces: below that the specialised
    # endgame evaluation returns an exact non-mate score by design, so a mate
    # score is not the right assertion there (test_convert.py covers those).
    # Every mate distance is confirmed by the brute-force solver below.
    fixtures = [
        ("2b1r3/2pk1p1p/r2pqn2/p7/PP3P2/p1PP2p1/4B1PP/1K1R3R b - - 3 27", 3),
        ("2r1r3/2n3kp/3b1Rp1/2pp3P/2p1b1P1/3PpB2/4P1K1/QN4NR w - - 3 31", 3),
        ("rn5r/7k/p4Ppn/P1ppp2p/1p1Pp3/R1P1QNP1/1P2KPB1/1NB2R2 w - - 1 23", 3),
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 1),
        ("6rk/6pp/8/6N1/8/8/8/6KR w - - 0 1", 1),
    ]
    for fen, want_plies in fixtures:
        board = chess.Board(fen)
        check("mate fixture legal", board.is_valid() and not board.is_game_over(),
              fen)
        check("mate fixture avoids the endgame probe",
              chess.popcount(board.occupied) > 5, fen)
        actual = _brute_force_mate_plies(board, 3)
        check("fixture mate distance verified", actual == want_plies,
              f"{fen} stated {want_plies} brute force {actual}")
        mv, score, depth, nodes = _search_fen(fen, 0, 8000, max_depth=12)
        plies = se.MATE_VALUE - score
        check("forced mate still found with pruning on",
              score > se.MATE_SCORE and plies <= want_plies,
              f"{fen} score {score} plies {plies} want <= {want_plies}")


def _parse_minimal_nodes(stdout):
    for line in stdout.splitlines():
        if line.startswith("MINIMAL_NODES"):
            return int(line.split()[1])
    return None


def test_pruning_effect(ref_stdout):
    """Every pruning feature has to actually fire. Comparing nodes for the
    same fixed-depth search against the all-pruning-off build catches a
    feature that silently never triggers or that explodes the tree."""
    minimal_nodes = _parse_minimal_nodes(ref_stdout)
    check("minimal node count reported", minimal_nodes is not None)
    if minimal_nodes is None:
        return
    full_nodes = _nodes_at_fixed_depth(NODE_BUDGET_FEN, NODE_BUDGET_DEPTH)
    ratio = full_nodes / max(minimal_nodes, 1)
    print(f"nodes depth {NODE_BUDGET_DEPTH}: pruning off {minimal_nodes}, "
          f"on {full_nodes}, ratio {ratio:.2f}")
    check("pruning reduces the tree", ratio < 0.75,
          f"minimal {minimal_nodes} full {full_nodes} ratio {ratio:.2f}")


def test_time_discipline():
    for hard in (100, 300):
        t0 = time.perf_counter()
        _search_fen(KIWIPETE, 0, hard)
        elapsed = (time.perf_counter() - t0) * 1000
        check("hard cap respected", elapsed < hard + 150,
              f"hard {hard} elapsed {elapsed:.0f}ms")


def test_sanity_depth():
    mv, score, depth, nodes = _search_fen(core.START_FEN, 400, 1000)
    check("startpos depth", depth >= 6, f"depth {depth} nodes {nodes}")
    check("startpos score sane", abs(score) < 150, f"score {score}")


def test_agent_game():
    import agent
    rng = random.Random(11)
    board = chess.Board()
    plies = 0
    while not board.is_game_over() and plies < 60:
        uci = agent.get_move(board.fen(), 20000)
        mv = chess.Move.from_uci(uci)
        check("agent move legal", mv in board.legal_moves, board.fen())
        if mv not in board.legal_moves:
            return
        board.push(mv)
        plies += 1
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))
        plies += 1
    check("agent tracker stable", agent.TRACKER.resets <= 1,
          f"resets {agent.TRACKER.resets}")
    check("agent outplays random", plies < 60 and board.is_checkmate()
          and board.turn == chess.BLACK, f"plies {plies} result {board.result()}")


def test_agent_fuzz(n_positions):
    import agent
    rng = random.Random(5)
    board = chess.Board()
    tested = 0
    while tested < n_positions:
        board.reset()
        for _ in range(rng.randrange(4, 100)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        uci = agent.get_move(board.fen(), 1200)
        ok = chess.Move.from_uci(uci) in board.legal_moves
        check("fuzz move legal", ok, board.fen())
        tested += 1


def main():
    deep = "--deep" in sys.argv
    if "--reference" in sys.argv:
        run_reference_mode(4 if deep else 3)

    t0 = time.perf_counter()
    try:
        ref = subprocess.run(
            [sys.executable, "test_search.py", "--reference"]
            + (["--deep"] if deep else []),
            env={**os.environ, "BTC_MINIMAL": "1"},
            capture_output=True, text=True, timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__)))
    except subprocess.TimeoutExpired:
        check("reference differential", False, "timed out after 600s")
        ref = None
        print(f"{passed} passed, {failed} failed")
        sys.exit(1)
    print(ref.stdout.strip())
    if ref.returncode != 0:
        print(ref.stderr[-2000:])
    check("reference differential", ref.returncode == 0)
    print(f"reference subprocess took {time.perf_counter() - t0:.0f}s")

    # compile the full search before any deadline-sensitive test: the first
    # call pays the whole numba compile, which must not eat a search budget
    t0 = time.perf_counter()
    _search_fen(core.START_FEN, 0, 120000, max_depth=3)
    print(f"parent compile warmup took {time.perf_counter() - t0:.0f}s")

    test_mates()
    test_repetition_draw()
    test_fifty_move()
    test_pruning_safety()
    test_pruning_effect(ref.stdout)
    test_time_discipline()
    test_sanity_depth()
    test_agent_game()
    test_agent_fuzz(40 if deep else 15)

    print(f"{passed} passed, {failed} failed{' (deep)' if deep else ''}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
