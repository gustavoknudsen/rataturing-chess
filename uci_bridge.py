"""UCI adapter for the Chessathon agent. Local testing only, never shipped.

The platform speaks get_move(fen, time_left_ms); match runners and GUIs speak
UCI. This wraps one in the other so the port can be played against external
engines, in particular the original C++ BTC. That reference matters because
every other measurement in this project is the port against itself, and a
defect the port shares with its own A/B opponent is invisible to those tests.

Moves are produced by calling agent.get_move, not by reaching into the search,
so a match here exercises the same path the platform uses: tracker diffing,
time budgeting and the python-chess legality net.

fd 1 is redirected to stderr before the agent is imported, exactly as the
platform runner does, so the agent's own prints cannot corrupt the protocol.
UCI replies go to a saved duplicate of the original fd 1.
"""

import functools
import os
import sys

import chess

_REAL_OUT = os.dup(1)
os.dup2(2, 1)

ENGINE = None
BUDGET = None
LAST = {"score": 0, "depth": 0, "nodes": 0}


def _say(line):
    os.write(_REAL_OUT, (line + "\n").encode("ascii"))


def _instrument(eng):
    """Record each search's depth and score so the bridge can emit info lines.
    get_move returns only a move string, but depth is the measurement that
    separates 'searches less' from 'plays worse at the same depth'."""
    original = eng.search.search_position

    def wrapper(*args, **kwargs):
        packed, score, depth, nodes = original(*args, **kwargs)
        LAST["score"], LAST["depth"], LAST["nodes"] = score, depth, nodes
        return packed, score, depth, nodes

    eng.search.search_position = wrapper


def _load():
    """Import the agent on demand. This pays the whole numba compile, so it
    happens under isready rather than at startup, where runners time out."""
    global ENGINE, BUDGET
    if ENGINE is None:
        import agent
        ENGINE = agent
        BUDGET = agent.btc_time.budget
        _instrument(agent)
    return ENGINE


def _new_game():
    """Reset to the state a freshly started platform process would have.

    The platform starts one process per game; here one process plays a whole
    match so the compile is paid once. Everything that carries between moves
    within a game must therefore be cleared between games, or game N+1 starts
    with game N's transposition table and history."""
    eng = _load()
    print("ucinewgame: state reset", file=sys.stderr)
    eng.TRACKER = eng.btc_game.GameTracker()
    state = eng.STATE
    for arr in (state.main_hist, state.cap_hist, state.cont_hist,
                state.counters, state.killers, state.tt_key, state.tt_data,
                state.sc, state.fc, state.played, state.static_evals,
                state.pv_len):
        arr[:] = 0


def _parse_position(tokens):
    if "fen" in tokens:
        start = tokens.index("fen") + 1
        end = tokens.index("moves") if "moves" in tokens else len(tokens)
        board = chess.Board(" ".join(tokens[start:end]))
    else:
        board = chess.Board()
    if "moves" in tokens:
        for uci in tokens[tokens.index("moves") + 1:]:
            board.push_uci(uci)
    return board


def _parse_go(tokens):
    params = {}
    for i, token in enumerate(tokens):
        if token in ("wtime", "btime", "winc", "binc", "movetime") \
                and i + 1 < len(tokens):
            params[token] = int(tokens[i + 1])
    return params


def _fixed_budget(ms):
    """Fixed time per move. The agent reads its budget only through
    btc_time.budget, so replacing it is enough to force st= mode."""
    return lambda time_left_ms, move_number, increment_ms=0: (ms, ms)


def _clock(board, params):
    if board.turn == chess.WHITE:
        return params.get("wtime", 60000), params.get("winc", 0)
    return params.get("btime", 60000), params.get("binc", 0)


def _pick(board, tokens):
    eng = _load()
    params = _parse_go(tokens)
    original = BUDGET
    if "movetime" in params:
        eng.btc_time.budget = _fixed_budget(params["movetime"])
        time_left = params["movetime"] * 4
    else:
        time_left, increment = _clock(board, params)
        # get_move takes no increment; the agent's default assumes the
        # platform's 0.5 s. Bind whatever this match actually uses.
        eng.btc_time.budget = functools.partial(original,
                                                increment_ms=increment)
    try:
        return eng.get_move(board.fen(), int(time_left))
    finally:
        eng.btc_time.budget = original


def main():
    board = chess.Board()
    for raw in sys.stdin:
        tokens = raw.split()
        if not tokens:
            continue
        cmd = tokens[0]
        if cmd == "uci":
            _say("id name BTC-Python")
            _say("id author Gustavo Knudsen")
            _say("uciok")
        elif cmd == "isready":
            _load()
            _say("readyok")
        elif cmd == "ucinewgame":
            _new_game()
        elif cmd == "position":
            board = _parse_position(tokens)
        elif cmd == "go":
            move = _pick(board, tokens)
            _say("info depth %d score cp %d nodes %d"
                 % (LAST["depth"], LAST["score"], LAST["nodes"]))
            _say("bestmove " + move)
        elif cmd == "quit":
            return


if __name__ == "__main__":
    main()
