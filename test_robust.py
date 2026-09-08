"""Robustness sweep. Run: python test_robust.py [games] [positions]

A crash, an illegal move or a flag loses a whole game, which outweighs any
plausible evaluation gain, so this hunts failure modes rather than strength:

1. Adversarial and awkward positions through get_move: stalemates, bare kings,
   promotion races, en passant pins, positions with one legal move, positions
   already drawn, very low clocks.
2. Fuzz: random legal positions with random clocks, every reply checked.
3. Full games against a random mover with tiny clocks, which is where time
   management and the tracker break if they are going to.
4. Tracker integrity across a game, including the desync-and-recover path.
"""

import os
import random
import subprocess
import sys
import time

import chess

STARVED = """
import chess
import agent
board = chess.Board()
uci = agent.get_move(board.fen(), 120000)
assert chess.Move.from_uci(uci) in board.legal_moves, "illegal move"
print("STARVED_OK")
"""

AWKWARD = [
    # one legal move
    ("single legal reply", "7k/8/8/8/8/8/5Q2/6RK b - - 0 1"),
    # stalemate is one move away for the opponent
    ("near stalemate", "7k/5Q2/8/8/8/8/8/6RK w - - 0 1"),
    # bare kings plus one pawn, promotion imminent
    ("promotion race", "8/6P1/8/8/8/8/1p6/K6k w - - 0 1"),
    # en passant available and the capturer is pinned
    ("en passant pin", "8/8/8/K1pP3q/8/8/8/7k w - c6 0 2"),
    # both sides can promote
    ("mutual promotion", "8/P6k/8/8/8/8/p6K/8 w - - 0 1"),
    # fifty move counter nearly expired
    ("fifty move edge", "8/8/4k3/8/8/4K3/8/7R w - - 99 120"),
    # castling available both sides
    ("castling rights", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"),
    # maximum-ish material
    ("crowded", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    # king on the edge, many checks available
    ("checks everywhere", "7k/8/8/8/8/8/6QQ/K7 w - - 0 1"),
    # underpromotion matters
    ("underpromotion", "8/5P1k/8/8/8/8/8/K6R w - - 0 1"),
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


def _legal(agent, fen, clock):
    """get_move must return a legal move for this position, whatever happens."""
    board = chess.Board(fen)
    if board.is_game_over():
        return True, "position already over, skipped"
    try:
        uci = agent.get_move(fen, clock)
    except Exception as exc:
        return False, f"raised {type(exc).__name__}: {exc}"
    try:
        mv = chess.Move.from_uci(uci)
    except ValueError:
        return False, f"malformed {uci!r}"
    if mv not in board.legal_moves:
        return False, f"illegal {uci}"
    if len(uci) > 5:
        return False, f"oversized reply {uci!r}"
    return True, uci


def test_awkward(agent):
    for name, fen in AWKWARD:
        for clock in (120000, 2000, 300, 50):
            ok, detail = _legal(agent, fen, clock)
            check(f"awkward: {name} @ {clock}ms", ok, f"{fen} -> {detail}")


def test_fuzz(agent, positions, seed=23):
    rng = random.Random(seed)
    board = chess.Board()
    tested = 0
    while tested < positions:
        board.reset()
        for _ in range(rng.randrange(2, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if board.is_game_over():
                break
        if board.is_game_over():
            continue
        clock = rng.choice([120000, 30000, 5000, 1000, 200, 60])
        ok, detail = _legal(agent, board.fen(), clock)
        check("fuzz", ok, f"{board.fen()} @ {clock}ms -> {detail}")
        tested += 1


def test_low_clock_games(agent, games, seed=5):
    """Whole games on a tiny clock: the flag and tracker failure surface."""
    rng = random.Random(seed)
    for game in range(games):
        board = chess.Board()
        clock = 3000.0
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < 300:
            started = time.perf_counter()
            ok, detail = _legal(agent, board.fen(), int(max(clock, 1)))
            spent = (time.perf_counter() - started) * 1000.0
            if not ok:
                check("low clock game", False, f"{board.fen()} -> {detail}")
                break
            clock -= spent
            clock += 50
            if clock <= 0:
                check("low clock game flagged", False,
                      f"game {game} ply {plies} overspent")
                break
            board.push_uci(detail)
            plies += 1
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
            plies += 1
        else:
            check("low clock game completed", True)
            continue
        if ok and clock > 0:
            check("low clock game completed", True)


def test_tracker_recovers(agent):
    """The tracker must survive being handed an unrelated position mid-game,
    which is what a desync looks like from inside get_move."""
    board = chess.Board()
    for _ in range(6):
        ok, uci = _legal(agent, board.fen(), 60000)
        check("tracker: normal move", ok, uci)
        if not ok:
            return
        board.push_uci(uci)
        board.push(list(board.legal_moves)[0])
    jumped = "r1bqk2r/pp1pppbp/2n2np1/2p5/2P5/2N1PNP1/PP1P1PBP/R1BQK2R b KQkq - 0 6"
    ok, uci = _legal(agent, jumped, 60000)
    check("tracker: recovers from an unrelated position", ok, uci)
    ok, uci = _legal(agent, board.fen(), 60000)
    check("tracker: recovers back to the original game", ok, uci)


def test_starved_warmup():
    """A warmup that cannot finish must degrade, never raise.

    search_position fixes its stop time before its first njit call, and that
    call is where the whole numba compile happens, so any deadline shorter
    than the compile makes the warmup return depth 0. That is exactly what
    crashed the platform upload on 2026-09-08: an exception during import is
    an immediate loss. It passed locally because this machine compiles inside
    a budget the platform's slower core blew. Runs in a subprocess because the
    condition only exists before anything is compiled.
    """
    env = dict(os.environ, BTC_WARMUP_HARD_MS="1")
    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, "-c", STARVED], capture_output=True,
                         text=True, cwd=here, env=env, timeout=1800)
    check("starved warmup degrades instead of crashing",
          out.returncode == 0 and "STARVED_OK" in out.stdout,
          out.stderr[-400:])


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    positions = int(sys.argv[2]) if len(sys.argv) > 2 else 120

    test_starved_warmup()
    print(f"starved warmup done ({passed} passed)", flush=True)

    t0 = time.perf_counter()
    import agent
    print(f"agent import {time.perf_counter() - t0:.0f}s", flush=True)

    test_awkward(agent)
    print(f"awkward done ({passed} passed)", flush=True)
    test_tracker_recovers(agent)
    print(f"tracker done ({passed} passed)", flush=True)
    test_fuzz(agent, positions)
    print(f"fuzz done ({passed} passed)", flush=True)
    test_low_clock_games(agent, games)

    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
