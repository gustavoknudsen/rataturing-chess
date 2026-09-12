"""Chess960 safety net. Run: python tests/test_frc_safety.py

The engine does not play Chess960 and is not going to before the final. What
it must do is survive one: parse the position, generate only legal moves, and
return a legal move from get_move. Losing the right to castle is a handicap.
Crashing, or playing an illegal move, is a forfeited game.

Castling rights in a Chess960 FEN are file letters (HAha) rather than KQkq.
Those parse to no rights at all, so the engine simply never castles. That is
the degraded-but-legal behaviour this file pins down, so that if someone later
teaches it real Chess960 castling, the safety properties are already asserted.

Everything runs in one process: the numba compile is the expensive part and
there is no reason to pay it per position.
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "src"))

import chess

import btc_nrt  # noqa: F401
import btc_core as core

FAILURES = []
CHECKS = [0]

# Real Chess960 starting positions, in X-FEN form, plus two midgame positions
# reached from irregular starts. 518 is the standard array, included so the
# standard case is covered by the same assertions.
POSITIONS = [
    ("frc 518 standard",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1"),
    ("frc 0 bbqnnrkr",
     "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w HFhf - 0 1"),
    ("frc 959 rkrnnqbb",
     "rkrnnqbb/pppppppp/8/8/8/8/PPPPPPPP/RKRNNQBB w CAca - 0 1"),
    ("frc rook on b",
     "nrbbnkqr/pppppppp/8/8/8/8/PPPPPPPP/NRBBNKQR w HBhb - 0 1"),
    ("frc king on b",
     "qrkrnbbn/pppppppp/8/8/8/8/PPPPPPPP/QRKRNBBN w DBdb - 0 1"),
    ("frc midgame",
     "1rkr1bbn/pp1p1ppp/2n1p3/q1p5/3P4/2N1P3/PPP2PPP/1RKRQBBN w DBdb - 0 6"),
    ("frc no rights",
     "1rkr1bbn/pp1p1ppp/2n1p3/q1p5/3P4/2N1P3/PPP2PPP/1RKRQBBN w - - 0 6"),
]


def check(ok, label):
    CHECKS[0] += 1
    if not ok:
        FAILURES.append(label)


def our_moves(fen):
    """Every move our generator produces, as UCI. Raises if parsing fails."""
    import numpy as np
    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    ml = np.zeros(256, dtype=np.int32)
    cnt = core.generate_moves(bb, st, ml)
    return [core.move_to_uci(int(ml[i])) for i in range(cnt)]


def test_parsing_and_legality():
    """Parse, generate, and prove every move we make is legal in Chess960."""
    for name, fen in POSITIONS:
        try:
            mine = our_moves(fen)
        except Exception as exc:
            check(False, "{}: parsing or generation raised {!r}".format(
                name, exc))
            continue

        board = chess.Board(fen, chess960=True)
        legal = set(m.uci() for m in board.legal_moves)
        illegal = [uci for uci in mine if uci not in legal]

        check(bool(mine), "{}: generated no moves at all".format(name))
        check(not illegal, "{}: generated illegal moves {}".format(
            name, illegal[:4]))
        print("  {:<20} {:>3} moves, {:>3} legal in python-chess, "
              "{} illegal".format(name, len(mine), len(legal), len(illegal)))


def test_never_castles():
    """X-FEN rights must read as no rights, so no castling move appears.

    This is the assertion that documents the limitation. If real Chess960
    castling is ever implemented, this test is the one that should fail, and
    whoever changes it should replace it with a correctness test rather than
    deleting it.
    """
    import numpy as np
    for name, fen in POSITIONS:
        bb, st = core.new_board()
        try:
            core.parse_fen(fen, bb, st)
        except Exception:
            continue
        ml = np.zeros(256, dtype=np.int32)
        cnt = core.generate_moves(bb, st, ml)
        castles = [core.move_to_uci(int(ml[i])) for i in range(cnt)
                   if int(ml[i]) & core.CASTLE_FLAG]
        check(not castles, "{}: generated castling {} from X-FEN rights".format(
            name, castles))


def test_agent_returns_legal():
    """get_move must return a legal move on a Chess960 position."""
    import agent
    for name, fen in POSITIONS:
        try:
            uci = agent.get_move(fen, 3000)
        except Exception as exc:
            check(False, "{}: get_move raised {!r}".format(name, exc))
            continue
        board = chess.Board(fen, chess960=True)
        legal = set(m.uci() for m in board.legal_moves)
        check(uci in legal, "{}: get_move returned {} which is not legal"
              .format(name, uci))
        print("  {:<20} get_move -> {}".format(name, uci))


def main():
    print("Chess960 safety net")
    print("")
    print("move generation")
    test_parsing_and_legality()
    print("")
    print("castling is declined rather than faked")
    test_never_castles()
    print("")
    print("agent entry point")
    test_agent_returns_legal()
    print("")
    if FAILURES:
        for failure in FAILURES:
            print("FAIL: " + failure)
        print("{} checks, {} failed".format(CHECKS[0], len(FAILURES)))
        return 1
    print("{} checks, all passed".format(CHECKS[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
