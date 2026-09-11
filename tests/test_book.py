"""Opening book: the rule boundary, re-entry, and every failure path.

    python test_book.py [book.bin[,second.bin]]

Defaults to our own shipped books, which also runs the acceptance vectors.
Pass any other Polyglot file to check the reader against a foreign book's
format rather than against itself.

The test that matters most is the move-number gate. The rules permit a shipped
table to answer "a position whose move number is 20 or lower" and call anything
later "a stored search", so the boundary is checked from both sides on a
position that is definitely in the book - which isolates the gate from whether
the book happens to contain deep lines.
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import os
import sys

import chess
import chess.polyglot

import btc_book

PASS = 0
FAIL = 0

# Fixed vectors for our own book. The last one is the format check that catches
# the classic Polyglot mistake: castling is stored as king-takes-own-rook and
# must read back as e1g1, not e1h1.
VECTORS = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e7e5"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "g1f3"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "b8c6"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f1c4"),
    ("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
     "e1g1"),
]


def check(label, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}: got {got!r}, want {want!r}")


def at_move(board, number):
    """The same position, presented as move `number`."""
    clone = board.copy()
    clone.fullmove_number = number
    return clone


def test_gate():
    print("move-number gate (rules: 'move number is 20 or lower')")
    board = chess.Board()
    check("move 1 answered", btc_book.probe(at_move(board, 1)) is not None, True)
    check("move 20 answered", btc_book.probe(at_move(board, 20)) is not None,
          True)
    check("move 21 refused", btc_book.probe(at_move(board, 21)), None)
    check("move 40 refused", btc_book.probe(at_move(board, 40)), None)
    check("in_window(20)", btc_book.in_window(at_move(board, 20)), True)
    check("in_window(21)", btc_book.in_window(at_move(board, 21)), False)
    check("MAX_BOOK_MOVE is the rule", btc_book.MAX_BOOK_MOVE, 20)


def _first_covered_position():
    """A position the FIRST reader actually holds.

    rataturing.bin is built outward from the tournament's curated opening
    positions, not from move 1, so it has no entry for the standard start.
    Walking from the start with whatever book answers finds a position the
    first reader covers, which is what the top-move property needs.
    """
    board = chess.Board()
    for _ in range(40):
        try:
            btc_book._READERS[0].find(board)
            return board
        except IndexError:
            pass
        uci = btc_book.probe(board)
        if uci is None:
            return None
        board.push(chess.Move.from_uci(uci))
    return None


def test_top_move():
    print("always the top move, no variety")
    board = _first_covered_position()
    if board is None:
        check("found a position the first book covers", False, True)
        return
    first = btc_book.probe(board)
    check("repeatable across probes", [btc_book.probe(board) for _ in range(5)],
          [first] * 5)
    best = max(btc_book._READERS[0].find_all(board), key=lambda e: e.weight)
    check("agrees with the highest weight", first, best.move.uci())


def test_legality():
    print("every returned move is legal")
    board = chess.Board()
    plies = 0
    while plies < 40:
        uci = btc_book.probe(board)
        if uci is None:
            break
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            check(f"legal at ply {plies}", uci, "a legal move")
            return
        board.push(move)
        plies += 1
    check("walked a book line with no illegal move", True, True)
    print(f"       (followed {plies} plies of top book moves)")


def test_reentry():
    print("re-entry after leaving the book")
    board = chess.Board()
    board.push_san("a3")
    board.push_san("a6")
    # Which lines leave book depends on book content, so find the exit
    # rather than asserting a fixed position is outside it.
    for _ in range(60):
        uci = btc_book.probe(board)
        if uci is None:
            break
        board.push(chess.Move.from_uci(uci))
    check("a line eventually leaves the book", btc_book.probe(board), None)

    trans = chess.Board()
    for san in ("Nf3", "d5", "d4", "Nf6", "c4"):
        trans.push_san(san)
    direct = chess.Board()
    for san in ("d4", "d5", "c4", "Nf6", "Nf3"):
        direct.push_san(san)
    check("the two move orders transpose",
          trans.board_fen() == direct.board_fen(), True)
    check("same polyglot key",
          chess.polyglot.zobrist_hash(trans)
          == chess.polyglot.zobrist_hash(direct), True)
    if btc_book.probe(direct) is not None:
        check("book answers the transposition after a miss",
              btc_book.probe(trans), btc_book.probe(direct))
    else:
        print("       (no entry for that line, transposition untested)")


def test_vectors(names):
    if not any("hedge" in n or "rataturing" in n for n in names):
        print("acceptance vectors: skipped, not our book")
        return
    print("acceptance vectors")
    for fen, want in VECTORS:
        check(f"{fen.split()[0][:24]}... -> {want}",
              btc_book.probe(chess.Board(fen)), want)


def test_failure_paths():
    print("failure paths (a broken book must degrade, never crash)")
    saved = os.environ.get("BTC_BOOK")

    os.environ["BTC_BOOK"] = "does_not_exist.bin"
    check("missing file", btc_book.load(), "no book")
    check("probes to None", btc_book.probe(chess.Board()), None)

    os.environ["BTC_BOOK"] = "agent.py"      # not a multiple of 16 bytes
    check("non-book file rejected", "rejected" in btc_book.load(), True)
    check("probes to None", btc_book.probe(chess.Board()), None)

    # btc_book._path() resolves names relative to its own module, so the
    # fixture has to be written where it will actually be looked for.
    truncated = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src",
        "truncated_book.bin")
    try:
        with open(truncated, "wb") as handle:
            handle.write(b"\x00" * 33)       # 33 is not a multiple of 16
        os.environ["BTC_BOOK"] = truncated
        check("truncated file rejected", "rejected" in btc_book.load(), True)
        check("probes to None", btc_book.probe(chess.Board()), None)
    finally:
        if os.path.exists(truncated):
            os.remove(truncated)

    if saved is None:
        os.environ.pop("BTC_BOOK", None)
    else:
        os.environ["BTC_BOOK"] = saved


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else "rataturing.bin,rataturing_hedge.bin"
    os.environ["BTC_BOOK"] = spec
    print(btc_book.load())
    if not btc_book._READERS:
        raise SystemExit(f"cannot open {spec}")

    names = [n.strip() for n in spec.split(",")]
    test_gate()
    test_top_move()
    test_legality()
    test_reentry()
    test_vectors(names)
    test_failure_paths()

    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
