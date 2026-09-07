"""Correctness suite for the bitboard core. Run: python run_tests.py [--deep]

Validates perft against canonical node counts, cross-checks move generation
and FEN handling against python-chess, and exercises the game tracker with a
simulated referee. Perft must be bit-exact or nothing else proceeds.
"""

import random
import sys
import time

import chess
import numpy as np

import btc_core as core
from btc_game import GameTracker

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

PERFT_SUITE = [
    (core.START_FEN, [20, 400, 8902, 197281, 4865609, 119060324]),
    (KIWIPETE, [48, 2039, 97862, 4085603, 193690690]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     [14, 191, 2812, 43238, 674624, 11030083]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     [6, 264, 9467, 422333, 15833292]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     [44, 1486, 62379, 2103487, 89941194]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [46, 2079, 89890, 3894594, 164075551]),
]

FAST_MAX_NODES = 5_000_000

passed = 0
failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name} {detail}")


def run_perft(fen, depth):
    bb, st = core.new_board()
    undo_bb, undo_st, mls = core.new_stacks()
    core.parse_fen(fen, bb, st)
    return core.perft(bb, st, undo_bb, undo_st, mls, depth, 0)


def test_bit_primitives():
    rng = random.Random(1)
    for _ in range(2000):
        v = rng.getrandbits(64)
        if v == 0:
            continue
        u = np.uint64(v)
        check("lsb", core.lsb(u) == (v & -v).bit_length() - 1, hex(v))
        check("popcount", core.count_bits(u) == bin(v).count("1"), hex(v))


def test_perft_suite(deep):
    for fen, counts in PERFT_SUITE:
        for depth, expected in enumerate(counts, start=1):
            if not deep and expected > FAST_MAX_NODES:
                continue
            got = run_perft(fen, depth)
            check(f"perft({depth}) {fen.split()[0]}", got == expected,
                  f"got {got} expected {expected}")


def test_movegen_differential():
    rng = random.Random(42)
    bb, st = core.new_board()
    for game in range(40):
        board = chess.Board()
        for _ in range(rng.randrange(10, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            core.parse_fen(board.fen(), bb, st)
            ours = sorted(core.move_to_uci(m) for m in core.legal_moves(bb, st))
            theirs = sorted(m.uci() for m in board.legal_moves)
            check("movegen", ours == theirs, board.fen())
            if ours != theirs:
                return


def test_hash_and_state_restoration():
    rng = random.Random(7)
    bb, st = core.new_board()
    undo_bb, undo_st, mls = core.new_stacks()
    core.parse_fen(KIWIPETE, bb, st)
    for _ in range(600):
        moves = core.legal_moves(bb, st)
        if not moves:
            core.parse_fen(KIWIPETE, bb, st)
            continue
        before_bb, before_st = bb.copy(), st.copy()
        mv = rng.choice(moves)
        core.make_move(bb, st, undo_bb, undo_st, 0, mv)
        check("incremental hash",
              core.generate_hash_key(bb, st) == bb[core.HASH],
              core.move_to_uci(mv))
        occ = bb[core.OCC_W] | bb[core.OCC_B]
        check("occupancy", occ == bb[core.OCC_A])
        core.unmake(bb, st, undo_bb, undo_st, 0)
        check("unmake bb", np.array_equal(bb, before_bb))
        check("unmake st", np.array_equal(st, before_st))
        core.make_move(bb, st, undo_bb, undo_st, 0, mv)


def test_fen_round_trip():
    rng = random.Random(3)
    bb, st = core.new_board()
    fens = [f for f, _ in PERFT_SUITE]
    board = chess.Board()
    for _ in range(60):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
        fens.append(board.fen())
    for fen in fens:
        core.parse_fen(fen, bb, st)
        ref = chess.Board(fen)
        ours = core.to_fen(bb, st).split()
        theirs = ref.fen().split()
        check("fen board", ours[0] == theirs[0], fen)
        check("fen side/castle", ours[1:3] == theirs[1:3], fen)
        check("fen counters", ours[4:6] == theirs[4:6], fen)


def test_tracker():
    rng = random.Random(9)
    for game in range(20):
        board = chess.Board()
        tracker = GameTracker()
        plies = 0
        while not board.is_game_over() and plies < 120:
            tracker.update(board.fen())
            our_moves = core.legal_moves(tracker.bb, tracker.st)
            uci_choice = rng.choice(list(board.legal_moves)).uci()
            packed = next(m for m in our_moves if core.move_to_uci(m) == uci_choice)
            tracker.push_our_move(packed)
            board.push_uci(uci_choice)
            plies += 1
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
            plies += 1
        check("tracker no resets", tracker.resets == 0, f"game {game}")
        keys, count = tracker.history()
        check("tracker key count", count >= plies - 1, f"{count} vs {plies}")


def test_tracker_repetition():
    fen = "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
    tracker = GameTracker()
    board = chess.Board(fen)
    shuffle = ["e1d1", "e8d8", "d1e1", "d8e8"] * 2
    for i, uci in enumerate(shuffle):
        if board.turn == chess.WHITE:
            tracker.update(board.fen())
            packed = next(m for m in core.legal_moves(tracker.bb, tracker.st)
                          if core.move_to_uci(m) == uci)
            tracker.push_our_move(packed)
        board.push_uci(uci)
    tracker.update(board.fen())
    keys, count = tracker.history()
    start_key = keys[0]
    occurrences = sum(1 for i in range(count) if keys[i] == start_key)
    check("tracker sees repetition", occurrences >= 3, f"occurrences {occurrences}")
    check("tracker no resets in shuffle", tracker.resets == 0)


def main():
    deep = "--deep" in sys.argv
    t0 = time.perf_counter()
    core.warmup(compile_perft=True)
    print(f"warmup {time.perf_counter() - t0:.1f}s")
    test_bit_primitives()
    test_perft_suite(deep)
    test_movegen_differential()
    test_hash_and_state_restoration()
    test_fen_round_trip()
    test_tracker()
    test_tracker_repetition()
    print(f"{passed} passed, {failed} failed{' (deep)' if deep else ''}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
