"""Evaluation mask and term tests. Run: python test_eval.py

Masks are generated from definitions in btc_evalmasks.py, so these tests
restate each definition independently (mostly via python-chess square helpers)
and compare. A mask that agrees with a restatement of its own definition is
worth far more than a mask transcribed from C and eyeballed.
"""

import sys

import chess
import numpy as np

import btc_evalmasks as em

passed = 0
failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name} {detail}")


def _bits(mask):
    """Our square indices (a8=0) present in a bitboard."""
    value = int(mask)
    return {i for i in range(64) if value & (1 << i)}


def _to_chess_sq(sq):
    """Our a8=0 indexing to python-chess a1=0 indexing."""
    return chess.square(sq % 8, 7 - sq // 8)


def _from_chess_sq(sq):
    return (7 - chess.square_rank(sq)) * 8 + chess.square_file(sq)


def test_indexing_matches_python_chess():
    for sq in range(64):
        name = "abcdefgh"[sq % 8] + str(8 - sq // 8)
        check("square name", chess.square_name(_to_chess_sq(sq)) == name,
              f"{sq} -> {name}")
        check("round trip", _from_chess_sq(_to_chess_sq(sq)) == sq, str(sq))


def test_file_rank_masks():
    for f in range(8):
        want = {sq for sq in range(64) if sq % 8 == f}
        check("file mask", _bits(em.FILE_MASK[f]) == want, f"file {f}")
    for r in range(8):
        want = {sq for sq in range(64) if sq // 8 == r}
        check("rank mask", _bits(em.RANK_MASK[r]) == want, f"rank {r}")


def test_passed_masks():
    """A white pawn on sq is passed iff no enemy pawn stands on the mask.
    Restated: files f-1..f+1, ranks strictly closer to the promotion rank."""
    for sq in range(64):
        f, r = sq % 8, sq // 8
        want_white = {s for s in range(64)
                      if abs(s % 8 - f) <= 1 and s // 8 < r}
        want_black = {s for s in range(64)
                      if abs(s % 8 - f) <= 1 and s // 8 > r}
        check("white passed mask", _bits(em.WHITE_PASSED[sq]) == want_white,
              f"sq {sq}")
        check("black passed mask", _bits(em.BLACK_PASSED[sq]) == want_black,
              f"sq {sq}")


def test_passed_mask_semantics():
    """End-to-end: a pawn is passed exactly when the mask is clear of enemy
    pawns, cross-checked against a direct scan of the board."""
    cases = [
        ("8/8/8/3P4/8/8/8/8 w - - 0 1", True),
        ("8/2p5/8/3P4/8/8/8/8 w - - 0 1", False),
        ("8/8/2p5/3P4/8/8/8/8 w - - 0 1", False),
        ("8/p7/8/3P4/8/8/8/8 w - - 0 1", True),
        ("8/8/8/3P4/3p4/8/8/8 w - - 0 1", True),
    ]
    for fen, want in cases:
        board = chess.Board(fen)
        pawn = list(board.pieces(chess.PAWN, chess.WHITE))[0]
        our_sq = _from_chess_sq(pawn)
        enemy = 0
        for s in board.pieces(chess.PAWN, chess.BLACK):
            enemy |= 1 << _from_chess_sq(s)
        is_passed = (int(em.WHITE_PASSED[our_sq]) & enemy) == 0
        check("passed semantics", is_passed == want, f"{fen} got {is_passed}")


def test_between_and_line():
    for a in range(64):
        for b in range(64):
            between = _bits(em.BETWEEN[a, b])
            line = _bits(em.LINE[a, b])
            ra, fa, rb, fb = a // 8, a % 8, b // 8, b % 8
            aligned = (ra == rb or fa == fb
                       or abs(ra - rb) == abs(fa - fb)) and a != b
            if not aligned:
                check("between empty when unaligned", not between, f"{a},{b}")
                check("line empty when unaligned", not line, f"{a},{b}")
                continue
            steps = max(abs(ra - rb), abs(fa - fb))
            dr = (rb - ra) // steps
            df = (fb - fa) // steps
            want = {(ra + dr * i) * 8 + (fa + df * i) for i in range(1, steps)}
            check("between squares", between == want, f"{a},{b}")
            check("line contains both ends", a in line and b in line, f"{a},{b}")
            check("line contains between", want <= line, f"{a},{b}")


def test_relative_rank():
    for sq in range(64):
        check("white relative rank",
              em.RELATIVE_RANK[em.WHITE, sq] == 7 - sq // 8, str(sq))
        check("black relative rank",
              em.RELATIVE_RANK[em.BLACK, sq] == sq // 8, str(sq))
    # a white pawn on its start rank is relative rank 1
    start = _from_chess_sq(chess.E2)
    check("white e2 relative rank", em.RELATIVE_RANK[em.WHITE, start] == 1)
    black_start = _from_chess_sq(chess.E7)
    check("black e7 relative rank", em.RELATIVE_RANK[em.BLACK, black_start] == 1)


def test_king_zone():
    """The ring must contain the king's square and its neighbours, and stay
    three files wide even with the king on an edge file."""
    for sq in range(64):
        zone = _bits(em.WHITE_KING_ZONE[sq])
        chess_sq = _to_chess_sq(sq)
        neighbours = {_from_chess_sq(s)
                      for s in chess.SquareSet(chess.BB_KING_ATTACKS[chess_sq])}
        check("king zone covers king moves", neighbours <= zone, f"sq {sq}")
        files = {s % 8 for s in zone}
        check("king zone at least three files wide", len(files) >= 3,
              f"sq {sq} files {sorted(files)}")


def test_support_and_phalanx():
    for sq in range(64):
        f, r = sq % 8, sq // 8
        want_phalanx = {s for s in range(64)
                        if s // 8 == r and abs(s % 8 - f) == 1}
        check("phalanx", _bits(em.PHALANX[sq]) == want_phalanx, str(sq))
        want_wsupport = {s for s in range(64)
                         if s // 8 == r + 1 and abs(s % 8 - f) == 1}
        check("white support", _bits(em.WHITE_SUPPORT[sq]) == want_wsupport,
              str(sq))


def test_isolated_and_adjacent():
    for sq in range(64):
        f = sq % 8
        want = {s for s in range(64) if abs(s % 8 - f) == 1}
        check("isolated mask", _bits(em.ISOLATED[sq]) == want, str(sq))
    for f in range(8):
        want = {s for s in range(64) if abs(s % 8 - f) == 1}
        check("adjacent files", _bits(em.ADJACENT_FILES[f]) == want, str(f))


def _mirror_fen(fen):
    """Flip the board vertically and swap colours, including castling rights,
    en passant square and side to move."""
    board = chess.Board(fen)
    return board.mirror().fen()


SYMMETRY_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    "r1bqkb1r/pp3ppp/2np4/1N1Pp3/8/8/PPP2PPP/R1BQKB1R b KQkq - 0 8",
    "8/8/8/4k3/8/8/Q7/K6R w - - 0 1",
    "2r3k1/R7/8/1R6/8/8/P4KPP/8 w - - 0 40",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
]


def test_eval_symmetry():
    """Mirroring a position swaps the side to move as well, so the tempo term
    follows it and cancels: the score from the side to move's point of view
    must be identical, not merely close. Any difference is a colour-indexing
    bug, the exact class BTC v2.4 had to fix repeatedly."""
    import btc_core as core
    import btc_eval as ev

    bb, st = core.new_board()
    bb2, st2 = core.new_board()
    for fen in SYMMETRY_FENS:
        core.parse_fen(fen, bb, st)
        original = int(ev.evaluate(bb, st))
        core.parse_fen(_mirror_fen(fen), bb2, st2)
        mirrored = int(ev.evaluate(bb2, st2))
        check("eval symmetry", original == mirrored,
              f"{fen} stm {original} mirrored {mirrored} "
              f"diff {original - mirrored}")


def test_eval_sanity():
    import btc_core as core
    import btc_eval as ev

    bb, st = core.new_board()

    core.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                   bb, st)
    start = int(ev.evaluate(bb, st))
    check("startpos near equal", abs(start - ev.TEMPO) < 80, f"score {start}")

    core.parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1",
                   bb, st)
    rook_down = int(ev.evaluate(bb, st))
    check("a rook down is clearly worse", rook_down < start - 800,
          f"start {start} rook_down {rook_down}")

    core.parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1", bb, st)
    bare = int(ev.evaluate(bb, st))
    check("bare kings near zero", abs(bare - ev.TEMPO) < 60, f"score {bare}")

    core.parse_fen("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1", bb, st)
    queen_up = int(ev.evaluate(bb, st))
    check("a queen up is winning", queen_up > 2000, f"score {queen_up}")


def test_eval_determinism():
    import btc_core as core
    import btc_eval as ev

    bb, st = core.new_board()
    for fen in SYMMETRY_FENS:
        core.parse_fen(fen, bb, st)
        first = int(ev.evaluate(bb, st))
        for _ in range(3):
            check("eval deterministic", int(ev.evaluate(bb, st)) == first, fen)


def main():
    test_indexing_matches_python_chess()
    test_file_rank_masks()
    test_passed_masks()
    test_passed_mask_semantics()
    test_between_and_line()
    test_relative_rank()
    test_king_zone()
    test_support_and_phalanx()
    test_isolated_and_adjacent()
    test_eval_symmetry()
    test_eval_sanity()
    test_eval_determinism()
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()


