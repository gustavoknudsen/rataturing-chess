"""KPK bitbase tests. Run: python test_kpk.py

Checked against textbook king-and-pawn theory rather than against another
implementation, since a wrong bitbase is worse than none: it would make the
engine confidently misjudge the most common pawn ending.
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import sys
import time

import chess

import btc_kpk as kpk

passed = 0
failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name} {detail}")


def sq(name):
    """python-chess square name to our a8=0 indexing."""
    s = getattr(chess, name.upper())
    return (7 - chess.square_rank(s)) * 8 + chess.square_file(s)


def win(wk, pawn, bk, white_to_move):
    """Goes through probe() so file mirroring and rank flipping are exercised;
    the raw table only stores files a-d with white as the strong side."""
    return bool(kpk.probe(sq(wk), sq(pawn), sq(bk), True, white_to_move))


def win_black(bk_strong, pawn, wk_weak, black_to_move):
    """Same position with colours reversed, to exercise the rank flip."""
    return bool(kpk.probe(sq(bk_strong), sq(pawn), sq(wk_weak), False,
                          black_to_move))


def test_known_positions():
    """Textbook cases. Opposition and key squares decide these."""
    cases = [
        # King on the sixth directly in front of a non-rook pawn queens
        # regardless of who moves: 1...Kd8 2.Kf7 and the pawn walks.
        ("Ke6 Pe5 vs Ke8, white", "e6", "e5", "e8", True, True),
        ("Ke6 Pe5 vs Ke8, black", "e6", "e5", "e8", True, False),
        # King on the sixth ahead of the pawn is winning regardless.
        ("Kc6 Pc5 vs Kc8, white", "c6", "c5", "c8", True, True),
        # Pawn far advanced, defending king cut off: won.
        ("Kb6 Pb5 vs Kd7, white", "b6", "b5", "d7", True, True),
        # Defending king in front of the pawn holds the draw.
        ("Kd5 Pd4 vs Kd7, white", "d5", "d4", "d7", False, True),
        # Rook pawn with the defending king in the corner is the classic draw.
        ("Ka6 Pa5 vs Ka8, white", "a6", "a5", "a8", False, True),
        ("Ka7 Pa6 vs Ka8 blocked", "a7", "a6", "a8", False, False),
    ]
    for name, wk, pawn, bk, expect_win, white_to_move in cases:
        got = win(wk, pawn, bk, white_to_move)
        check(f"kpk: {name}", got == expect_win,
              f"got {'win' if got else 'draw'}")


def test_colour_symmetry():
    """A position and its colour-and-rank mirror must give the same verdict.
    e6/e5/e8 for white mirrors to e3/e4/e1 for black."""
    cases = [
        (("e6", "e5", "e8", True), ("e3", "e4", "e1", True)),
        (("c6", "c5", "c8", True), ("c3", "c4", "c1", True)),
        (("a6", "a5", "a8", False), ("a3", "a4", "a1", False)),
        (("b6", "b5", "d7", True), ("b3", "b4", "d2", True)),
    ]
    for (wk, pawn, bk, stm), (bk2, pawn2, wk2, stm2) in cases:
        white_view = win(wk, pawn, bk, stm)
        black_view = win_black(bk2, pawn2, wk2, stm2)
        check(f"kpk: colour symmetry {wk}/{pawn}/{bk}",
              white_view == black_view,
              f"white {white_view} black {black_view}")


def test_structural():
    """Properties that must hold across the whole table."""
    # a pawn on the seventh with the king escorting it is winning
    check("kpk: escorted pawn on the seventh wins",
          win("b6", "b7", "d7", True))
    # any position where black simply takes an undefended pawn is drawn
    check("kpk: undefended pawn captured is a draw",
          not win("a1", "d5", "d6", False))
    # far more positions should be wins than the trivial handful
    total_win = int((kpk.KPK_TABLE == kpk.WIN).sum())
    check("kpk: table has a plausible number of wins",
          20000 < total_win < 150000, f"{total_win} wins")
    check("kpk: table fully resolved",
          int((kpk.KPK_TABLE == kpk.UNKNOWN).sum()) == 0)


def main():
    t0 = time.perf_counter()
    # the table is built at import; report the cost since it lands in the
    # platform's init budget
    print(f"kpk table ready, {kpk.KPK_TABLE.size} entries "
          f"({time.perf_counter() - t0:.2f}s to reach the tests)", flush=True)
    test_known_positions()
    test_colour_symmetry()
    test_structural()
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
