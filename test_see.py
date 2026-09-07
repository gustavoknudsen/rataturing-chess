"""SEE correctness. Run: python test_see.py [--deep]

see_ge is validated two ways:
1. Hand fixtures with known exchange outcomes.
2. Differential against an independent brute-force swap simulator built on
   python-chess: repeatedly play the least valuable attacker on the target
   square and take the negamax of the material sequence. Both must agree on
   the sign against many thresholds over random positions.
"""

import random
import sys

import chess

import btc_core as core

VALUES = {chess.PAWN: 100, chess.KNIGHT: 305, chess.BISHOP: 333,
          chess.ROOK: 563, chess.QUEEN: 950, chess.KING: 32000}

passed = 0
failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print(f"FAIL {name} {detail}")


def _swap_value(board, target, side):
    """Material swing of the optimal exchange sequence on `target` for `side`,
    computed independently of the engine. Each side may stand pat."""
    attackers = board.attackers(side, target)
    if not attackers:
        return 0
    victim = board.piece_at(target)
    if victim is None:
        return 0
    best_from = min(attackers, key=lambda s: VALUES[board.piece_type_at(s)])
    piece_type = board.piece_type_at(best_from)
    if piece_type == chess.KING and board.attackers(not side, target):
        return 0
    captured_value = VALUES[victim.piece_type]
    board_copy = board.copy(stack=False)
    board_copy.remove_piece_at(target)
    board_copy.set_piece_at(target, chess.Piece(piece_type, side))
    board_copy.remove_piece_at(best_from)
    gain = captured_value - _swap_value(board_copy, target, not side)
    return max(0, gain)


def _reference_see(board, move):
    """Exchange value of `move` per the independent simulator."""
    victim = board.piece_at(move.to_square)
    captured_value = VALUES[victim.piece_type] if victim else 0
    piece_type = board.piece_type_at(move.from_square)
    after = board.copy(stack=False)
    after.remove_piece_at(move.to_square)
    after.set_piece_at(move.to_square, chess.Piece(piece_type, board.turn))
    after.remove_piece_at(move.from_square)
    return captured_value - _swap_value(after, move.to_square, not board.turn)


def _engine_see(fen, uci, threshold):
    bb, st = core.new_board()
    core.parse_fen(fen, bb, st)
    packed = None
    for mv in core.legal_moves(bb, st):
        if core.move_to_uci(mv) == uci:
            packed = mv
            break
    assert packed is not None, f"{uci} not legal in {fen}"
    return core.see_ge(bb, st, packed, threshold)


def test_fixtures():
    """Hand cases with the exchange value stated explicitly. Each expected
    value is also confirmed by the independent simulator, so a mis-set-up
    position fails loudly instead of encoding a wrong assumption."""
    cases = [
        # free pawn: nothing defends d5
        ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 100),
        # pawn defended by pawn, rook initiates: 100 - 563
        ("4k3/8/2p5/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", -463),
        # queen takes pawn defended by pawn: 100 - 950
        ("4k3/8/2p5/3p4/8/8/3Q4/4K3 w - - 0 1", "d2d5", -850),
        # equal queen trade, black king recaptures: 950 - 950
        ("8/8/4k3/3q4/8/8/3Q4/4K3 w - - 0 1", "d2d5", 0),
        # undefended queen is simply won
        ("4k3/8/8/3q4/8/8/3Q4/3RK3 w - - 0 1", "d2d5", 950),
    ]
    for fen, uci, want_value in cases:
        board = chess.Board(fen)
        mv = chess.Move.from_uci(uci)
        check("fixture is legal", mv in board.legal_moves, f"{fen} {uci}")
        ref = _reference_see(board, mv)
        check("fixture value matches simulator", ref == want_value,
              f"{fen} {uci} stated {want_value} simulator {ref}")
        for threshold in (want_value - 1, want_value, want_value + 1):
            got = bool(_engine_see(fen, uci, threshold))
            check("see fixture", got == (want_value >= threshold),
                  f"{fen} {uci} thr {threshold} got {got} value {want_value}")


def test_differential(n_positions):
    rng = random.Random(17)
    board = chess.Board()
    tested = 0
    while tested < n_positions:
        board.reset()
        for _ in range(rng.randrange(4, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        fen = board.fen()
        for mv in board.legal_moves:
            if not board.is_capture(mv) or board.is_en_passant(mv) or mv.promotion:
                continue
            want = _reference_see(board, mv)
            for threshold in (-900, -400, -100, 0, 1, 100, 400):
                got = bool(_engine_see(fen, mv.uci(), threshold))
                check("see differential", got == (want >= threshold),
                      f"{fen} {mv.uci()} thr {threshold} engine {got} "
                      f"ref_value {want}")
        tested += 1


def main():
    deep = "--deep" in sys.argv
    test_fixtures()
    test_differential(120 if deep else 40)
    print(f"{passed} passed, {failed} failed{' (deep)' if deep else ''}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
