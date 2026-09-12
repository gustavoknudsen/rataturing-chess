"""get_move must never raise. Run: python tests/test_hostile_fen.py

Every FEN here either crashed the agent before this suite existed, or is a
position python-chess accepts and our own parser has to survive. A raised
exception out of get_move is a lost game, and so is a forfeit-shaped "0000"
when legal moves exist, so the bar is: do not raise, and return a move that is
actually legal whenever any parser can read the position.

The existing fuzz suite cannot find these: test_robust builds chess.Board(fen)
to pick its expected move, so every FEN it tries is valid by construction.
"""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import chess

import agent

PASS, FAIL = 0, 0

# Rejected by python-chess. The agent used to raise on all of these, because
# _fallback_move called chess.Board(fen) outside any handler.
REJECTED = [
    ("three-check", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 +0+0"),
    ("crazyhouse", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"),
    ("crazyhouse pocket",
     "r3k2r/8/8/8/8/8/8/R3K2R[QPqp] w KQkq - 0 1"),
    ("pocket and bad ep",
     "r3k2r/8/8/8/8/8/8/R3K2R[QPqp] w KQkq z9 0 1"),
    ("empty", ""),
    ("whitespace", "   "),
    ("truncated", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"),
    ("bad piece char", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBXR w KQkq - 0 1"),
    ("too few ranks", "rnbqkbnr/pppppppp/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("clock not a number", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - x 1"),
    ("extra fields", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 9 9 9"),
    ("nonsense", "not a fen at all"),
]

# Accepted by python-chess but illegal or unusual. These must not crash the
# engine's own parser, move generator, NNUE accumulator or endgame probes.
ACCEPTED = [
    ("no black king", "8/8/8/8/8/8/4P3/4K3 w - - 0 1"),
    ("three white kings", "4k3/8/8/8/8/8/8/RK2KRKR w - - 0 1"),
    ("pawns on back rank", "4k3/8/8/8/8/8/8/P3K3 w - - 0 1"),
    ("side to move wins", "4k3/8/8/8/8/8/8/R3K2R b KQ - 0 1"),
    ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
    ("checkmate", "7k/5Q1K/8/8/8/8/8/8 b - - 0 1"),
    ("only king", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
    ("huge move number", "4k3/8/8/8/8/8/8/4K3 w - - 99 9999"),
    ("frc-style castling", "4k3/8/8/8/8/8/8/RK6 w KQ - 0 1"),
    ("x-fen rights", "1rk1r3/8/8/8/8/8/8/1RK1R3 w BEbe - 0 1"),
]

CLOCKS = [120000, 1000, 100, 1, 0, -5]


def check(name, ok, detail=""):
    global PASS, FAIL
    print("  %-24s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if ok:
        PASS += 1
    else:
        FAIL += 1


def probe(label, fen, clock):
    """get_move must not raise, and must not answer "0000" when a legal move
    exists.

    Shape alone is too weak a bar. "0000" is four characters and a forfeit, and
    an earlier version of this suite passed a case where the engine returned it
    for a position with 30 legal moves.
    """
    try:
        uci = agent.get_move(fen, clock)
    except Exception as exc:
        check(label, False, "RAISED %r" % (exc,))
        return None
    if not (isinstance(uci, str) and len(uci) in (4, 5)):
        check(label, False, "returned %r" % (uci,))
        return None
    try:
        board = chess.Board(agent._normalise_fen(fen))
        legal = list(board.legal_moves)
    except Exception:
        # Nothing can read this position, so any shaped answer is acceptable.
        check(label, True, uci)
        return uci
    if not legal:
        check(label, True, uci)
        return uci
    ok = uci != "0000" and chess.Move.from_uci(uci) in board.legal_moves
    check(label, ok, uci if ok else "%s, but %d legal moves exist"
          % (uci, len(legal)))
    return uci


def test_rejected():
    print("FENs python-chess rejects:")
    for name, fen in REJECTED:
        probe(name, fen, 1000)


def test_accepted():
    print("\nFENs python-chess accepts but are illegal or unusual:")
    for name, fen in ACCEPTED:
        probe(name, fen, 1000)


def test_clocks():
    print("\nStandard position at hostile clock values:")
    start = chess.Board().fen()
    for clock in CLOCKS:
        probe("clock %d" % clock, start, clock)


def test_legality_preserved():
    """The hardening must not have cost us legality on ordinary positions."""
    print("\nOrdinary positions still return a legal move:")
    for name, fen in [
        ("start", chess.Board().fen()),
        ("midgame",
         "2r3k1/1q3pp1/p2p1n1p/1p1Pp3/4P3/1P2BP2/P1Q3PP/2R3K1 w - - 0 25"),
        ("endgame", "8/5k2/8/8/8/3K4/4P3/8 w - - 0 60"),
    ]:
        uci = probe(name, fen, 2000)
        if uci is None:
            continue
        board = chess.Board(fen)
        legal = chess.Move.from_uci(uci) in board.legal_moves
        check(name + " legal", legal, "" if legal else uci)


def main():
    test_rejected()
    test_accepted()
    test_clocks()
    test_legality_preserved()
    print("\n%d passed, %d failed" % (PASS, FAIL))
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
