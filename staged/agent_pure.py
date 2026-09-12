"""Fallback agent: python-chess only. No numba, no numpy, no network.

In the finals build it ships beside the real engine and answers only while the
engine is still compiling. On its own it is the last resort for constraints
the real engine cannot meet at all:

    init budget <= 10 s    the numba compile takes about 40 s on the platform
    memory cap <= 256 MB   numba's own runtime is ~320 MB before our first array
    numpy removed          every bitboard in the engine is np.uint64

It is not competitive with the real engine and is not meant to be. It is
competitive with the other teams' engines that also stopped working, and it
imports in under a second holding about 30 MB.

Deliberately one file with no imports beyond the standard library and chess, so
it also answers "one file only" and any source-size limit.

Design notes, because the shortcuts are not obvious:

Evaluation is material plus piece-square tables, tapered between a middlegame
and an endgame table by remaining material. That is the cheapest evaluation
that still knows a knight belongs in the centre and a king does not, until the
pawns come off.

Move ordering matters more here than anywhere else, because the tree is tiny
and every cutoff is a large fraction of it. Transposition move first, then
captures by MVV-LVA, then killers.

Quiescence is captures-only. Without it the engine hangs pieces at the horizon,
which at depth 4 is every move.
"""

import time

import chess
import chess.polyglot

# Centipawns. King is scored by the tables alone; a missing king cannot happen
# in a position the referee sends.
VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
         chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# Non-pawn material at the start, used to taper. Two of each minor and rook,
# one queen, per side.
PHASE_MAX = 2 * (2 * 320 + 2 * 330 + 2 * 500 + 900)

_PAWN_MG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0]
_KNIGHT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50]
_BISHOP = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20]
_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0]
_QUEEN = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20]
_KING_MG = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20]
_KING_EG = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50]
_PAWN_EG = [
     0,  0,  0,  0,  0,  0,  0,  0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    15, 15, 15, 15, 15, 15, 15, 15,
     5,  5,  5,  5,  5,  5,  5,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0]

# The tables above are written from white's point of view with a8 first, which
# is the order they are printed in, so index by (square ^ 56) for white and by
# square for black.
MG = {chess.PAWN: _PAWN_MG, chess.KNIGHT: _KNIGHT, chess.BISHOP: _BISHOP,
      chess.ROOK: _ROOK, chess.QUEEN: _QUEEN, chess.KING: _KING_MG}
EG = {chess.PAWN: _PAWN_EG, chess.KNIGHT: _KNIGHT, chess.BISHOP: _BISHOP,
      chess.ROOK: _ROOK, chess.QUEEN: _QUEEN, chess.KING: _KING_EG}

MATE = 30000
MAX_PLY = 64
OVERHEAD_MS = 50

_tt = {}
_killers = [[None, None] for _ in range(MAX_PLY + 1)]
_deadline = 0.0
_nodes = 0


class _Timeout(Exception):
    """Raised inside the search to unwind to the last completed depth."""


def _phase(board):
    total = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        count = len(board.pieces(piece_type, chess.WHITE)) \
            + len(board.pieces(piece_type, chess.BLACK))
        total += count * VALUE[piece_type]
    return min(total, PHASE_MAX)


def evaluate(board):
    """Centipawns from the side to move's point of view."""
    phase = _phase(board)
    mg = eg = 0
    for square, piece in board.piece_map().items():
        value = VALUE[piece.piece_type]
        index = square ^ 56 if piece.color == chess.WHITE else square
        sign = 1 if piece.color == chess.WHITE else -1
        mg += sign * (value + MG[piece.piece_type][index])
        eg += sign * (value + EG[piece.piece_type][index])
    score = (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX
    return score if board.turn == chess.WHITE else -score


def _capture_score(board, move):
    """MVV-LVA. Most valuable victim first, cheapest attacker as the tiebreak."""
    victim = board.piece_type_at(move.to_square)
    if victim is None:
        victim = chess.PAWN          # en passant
    attacker = board.piece_type_at(move.from_square) or chess.PAWN
    return VALUE[victim] * 16 - VALUE[attacker]


def _ordered(board, ply, tt_move):
    """Transposition move, then captures by MVV-LVA, then killers, then rest."""
    captures, quiets = [], []
    tt_legal = False
    for move in board.legal_moves:
        if move == tt_move:
            tt_legal = True
            continue
        if board.is_capture(move) or move.promotion:
            captures.append(move)
        else:
            quiets.append(move)
    captures.sort(key=lambda m: _capture_score(board, m), reverse=True)
    killers = [m for m in _killers[ply] if m in quiets]
    for move in killers:
        quiets.remove(move)
    # Only if it is actually legal here. Zobrist keys collide, and a stored
    # move from the colliding position would be pushed and raise.
    head = [tt_move] if tt_legal else []
    return head + captures + killers + quiets


def _check_time():
    global _nodes
    _nodes += 1
    # The clock is read every 1024 nodes. Reading it every node costs more than
    # the search at this speed.
    if not _nodes & 1023 and time.perf_counter() > _deadline:
        raise _Timeout()


def _quiesce(board, alpha, beta):
    _check_time()
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    for move in sorted((m for m in board.legal_moves
                        if board.is_capture(m) or m.promotion),
                       key=lambda m: _capture_score(board, m), reverse=True):
        board.push(move)
        score = -_quiesce(board, -beta, -alpha)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def _negamax(board, depth, alpha, beta, ply):
    _check_time()
    if board.is_repetition(2) or board.is_fifty_moves():
        return 0
    key = chess.polyglot.zobrist_hash(board)
    entry = _tt.get(key)
    tt_move = None
    if entry is not None:
        stored_depth, stored_score, stored_flag, tt_move = entry
        if stored_depth >= depth and ply > 0:
            if stored_flag == 0:
                return stored_score
            if stored_flag == 1 and stored_score >= beta:
                return stored_score
            if stored_flag == 2 and stored_score <= alpha:
                return stored_score
    if depth <= 0:
        return _quiesce(board, alpha, beta)

    best_move = None
    best = -MATE * 2
    flag = 2
    for move in _ordered(board, ply, tt_move):
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        if score > best:
            best, best_move = score, move
        if score > alpha:
            alpha, flag = score, 0
        if alpha >= beta:
            if not board.is_capture(move) and _killers[ply][0] != move:
                _killers[ply] = [move, _killers[ply][0]]
            flag = 1
            break

    if best_move is None:
        # No legal moves: mate is scored by distance so a shorter mate wins.
        return -MATE + ply if board.is_check() else 0
    _tt[key] = (depth, best, flag, best_move)
    return best


def _budget(time_left_ms):
    """A thirtieth of the clock, never more than a quarter of it."""
    usable = max(time_left_ms - OVERHEAD_MS, 10)
    return min(usable / 30.0, usable / 4.0) / 1000.0


def search(board, seconds):
    """Iterative deepening. Returns the best move found before time ran out."""
    global _deadline, _nodes
    started = time.perf_counter()
    _deadline = started + seconds
    _nodes = 0
    legal = list(board.legal_moves)
    if not legal:
        return None
    best = legal[0]
    # _Timeout unwinds through _negamax and _quiesce without reaching their
    # board.pop() calls, leaving the board several moves deep. Every legality
    # test afterwards would then be against the wrong position: measured, this
    # returned e1e2 from the starting position. Record the depth and unwind to
    # it by hand.
    root_depth = len(board.move_stack)
    for depth in range(1, MAX_PLY):
        try:
            _negamax(board, depth, -MATE * 2, MATE * 2, 0)
        except _Timeout:
            while len(board.move_stack) > root_depth:
                board.pop()
            break
        entry = _tt.get(chess.polyglot.zobrist_hash(board))
        # Only if it is legal here: the table is never cleared between moves,
        # and a Zobrist collision would otherwise hand back a move from the
        # colliding position as the answer.
        if entry is not None and entry[3] is not None                 and entry[3] in board.legal_moves:
            best = entry[3]
        # Each depth costs several times the last, so a depth that used more
        # than half the budget means the next cannot finish. Starting it only
        # throws away the remainder.
        if time.perf_counter() - started > seconds * 0.5:
            break
    return best


def get_move(fen, time_left_ms):
    """Entry point. Must never raise: an exception loses the game outright."""
    try:
        board = chess.Board(" ".join(fen.split()[:6]))
    except Exception:
        return "0000"
    try:
        move = search(board, _budget(time_left_ms))
        if move is not None and move in board.legal_moves:
            return move.uci()
    except Exception as exc:
        print(f"pure: search failed {exc!r}", flush=True)
    legal = list(board.legal_moves)
    return legal[0].uci() if legal else "0000"


print("init: pure-python fallback agent ready", flush=True)
