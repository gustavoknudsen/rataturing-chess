"""Draw-rule correctness: repetition boundaries, fifty-move, mate precedence.

    BTC_DRAW_FIX=1 python test_draw.py

These fixes are invisible to `searchbench.py`: it seeds history with random keys
that never match and never approaches a fifty-move count, so it reports
identical node counts either way. That makes it useless as evidence here, and
these targeted tests are the evidence instead.

Every test states the rule it is checking, so a failure says what is wrong with
the engine rather than what is wrong with the test.
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import os
import sys

import numpy as np
import chess

import btc_nrt  # noqa: F401
import btc_core as core
import btc_search as se

FIX = se.USE_DRAW_FIX
PASS = 0
FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}: got {got!r}, want {want!r}")


def board_at(fen):
    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    return bb, st


def test_repetition_unit():
    """_is_repetition, called directly, against the two boundaries."""
    print("repetition semantics (items 1 and 2)")
    bb, st = board_at(chess.STARTING_FEN)
    key = bb[core.HASH]
    other = np.uint64(0xDEADBEEF12345678)

    # rep_base marks where pre-root game history ends.
    # A match inside the search tree is a draw: the position can normally be
    # repeated again from there.
    rep = np.array([other, other, key], dtype=np.uint64)
    check("tree match is a draw",
          se._is_repetition(bb, rep, 3, 2, 0), 1)

    # A single match in game history is only the SECOND occurrence overall, so
    # by the rules it is not a draw. That is gated on BTC_DRAW_HIST, which
    # measured negative on its own and is off by default - see the flag comment
    # in btc_search: without a contempt term it only teaches the engine that
    # walking toward a threefold is safe.
    rep = np.array([key, other, other], dtype=np.uint64)
    got = se._is_repetition(bb, rep, 3, 3, 0)
    check("one history match: draw unless BTC_DRAW_HIST",
          got, 0 if se.USE_DRAW_HIST else 1)

    # Twice in history plus now is the third occurrence: a draw.
    rep = np.array([key, other, key], dtype=np.uint64)
    check("two history matches is a draw",
          se._is_repetition(bb, rep, 3, 3, 0), 1)

    # Inside a null-move subtree nothing before the null may match: the side to
    # move flipped without a move being played, so no legal sequence produced
    # the position and it is not a repetition.
    rep = np.array([key, other, other], dtype=np.uint64)
    got = se._is_repetition(bb, rep, 3, 0, 2)
    check("pre-null match is ignored inside a null subtree" if FIX
          else "pre-null match (unfixed build draws)", got, 0 if FIX else 1)

    # The floor must not hide a genuine match at or after it.
    rep = np.array([other, other, key], dtype=np.uint64)
    check("match at the floor still counts",
          se._is_repetition(bb, rep, 3, 0, 2), 1)

    # No match anywhere.
    rep = np.array([other, other, other], dtype=np.uint64)
    check("no match is not a draw", se._is_repetition(bb, rep, 3, 0, 0), 0)


def search(fen, depth, history=()):
    """(score, move) at fixed depth, with an optional pre-root history."""
    state = se.SearchState(tt_entries=1 << 16)
    bb, st = board_at(fen)
    keys = np.zeros(len(history) + 8, dtype=np.uint64)
    for i, key in enumerate(history):
        keys[i] = key
    keys[len(history)] = bb[core.HASH]
    mv, score, _, _ = se.search_position(
        state, bb, st, keys, len(history) + 1, soft_ms=0, hard_ms=600000,
        max_depth=depth)
    return int(score), core.move_to_uci(mv)


def test_mate_beats_fifty():
    """A mate delivered on the hundredth half-move is mate, not a draw."""
    print("fifty-move versus checkmate (item 4)")
    fen = "7k/R7/8/8/8/8/8/1R5K w - - 99 60"
    board = chess.Board(fen)
    mate = chess.Move.from_uci("b1b8")
    check("the test position really is mate in 1",
          _is_mate_in_one(board, mate), True)
    score, move = search(fen, 4)
    check("engine finds the mate", move, "b1b8")
    check("and scores it as mate, not a draw", score > se.MATE_SCORE, True)


def _is_mate_in_one(board, move):
    probe = board.copy()
    probe.push(move)
    return probe.is_checkmate()


def test_fifty_is_still_a_draw():
    """Everything at 100 that is not mate remains a draw."""
    print("fifty-move still draws when it should (items 3 and 4)")
    # Two bare kings and a rook, clock already at 99: after any quiet move the
    # counter reaches 100 and the position is drawn, not won.
    fen = "8/8/4k3/8/8/4K3/8/7R w - - 99 60"
    score, _ = search(fen, 6)
    check("a rook up at fifty-move is scored a draw, not a win",
          abs(score) < 40, True)


def test_history_repetition_not_drawn():
    """A position seen once before the root must not score as a draw."""
    print("second occurrence against game history (item 2)")
    # White is a rook up and winning. Feed a history containing this very
    # position once. Under the bug that single match scores 0 at every node
    # that revisits it, so the engine's score collapses toward a draw.
    fen = "8/8/4k3/8/8/4K3/8/7R w - - 4 40"
    bb, _ = board_at(fen)
    score, _ = search(fen, 8, history=(bb[core.HASH],))
    check("still evaluated as winning with one prior occurrence",
          score > 200, True)


def test_tt_mate_downgrade():
    """A stored mate that cannot arrive before the fifty-move draw is not a mate.

    Item 5 had no test until now. tt_probe is exercised directly: the same entry
    is read at three fifty counts, and only the one with too few half-moves left
    may refuse the cutoff."""
    print("TT mate downgrade near the fifty-move limit (item 5)")
    bb, st = board_at(chess.STARTING_FEN)
    tt_key = np.zeros(1 << 12, dtype=np.uint64)
    tt_data = np.zeros(1 << 12, dtype=np.int64)

    mate_in = 30
    score = se.MATE_VALUE - mate_in
    se.tt_record(bb, score, 20, se.HASH_EXACT, 0, 0, tt_key, tt_data, 1, 0,
                 se.TT_NO_EVAL)

    low, _, _, _ = se.tt_probe(bb, -se.INFINITY, se.INFINITY, 1, 0, tt_key,
                               tt_data, 0)
    check("fifty=0, mate returned", low, score)

    # 100 - 60 = 40 half-moves left, more than the 30 the mate needs.
    ok, _, _, _ = se.tt_probe(bb, -se.INFINITY, se.INFINITY, 1, 0, tt_key,
                              tt_data, 60)
    check("fifty=60, still reachable so still returned", ok, score)

    # 100 - 80 = 20 half-moves left, fewer than the 30 needed.
    high, _, _, _ = se.tt_probe(bb, -se.INFINITY, se.INFINITY, 1, 0, tt_key,
                                tt_data, 80)
    check("fifty=80, cutoff withheld" if FIX else "fifty=80, unfixed returns it",
          high, se.NO_HASH_ENTRY if FIX else score)


def test_no_crash_paths():
    """The new branches must not break ordinary searches."""
    print("regression: ordinary positions still search")
    for fen in (chess.STARTING_FEN,
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"):
        score, move = search(fen, 7)
        legal = chess.Move.from_uci(move) in chess.Board(fen).legal_moves
        check(f"{fen.split()[0][:22]}... returns a legal move", legal, True)


def main():
    print(f"BTC_DRAW_FIX={'1' if FIX else '0'}  "
          f"BTC_DRAW_HIST={'1' if se.USE_DRAW_HIST else '0'}  "
          f"BTC_QS_EVASION={'1' if se.USE_QS_EVASION else '0'}\n")
    test_repetition_unit()
    test_mate_beats_fifty()
    test_fifty_is_still_a_draw()
    test_history_repetition_not_drawn()
    test_tt_mate_downgrade()
    test_no_crash_paths()
    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
