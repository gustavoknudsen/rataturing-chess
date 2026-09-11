"""A/B match between two agent directories. Run:

    python arena_ab.py <challenger_dir> <champion_dir> [games] [base_ms]

Each agent runs in its own process, as on the platform, so module state and
numba compilation are isolated and the init cost is paid once per agent per
match rather than once per game. Colours alternate and openings are fixed by
game index, so a score change is a change you made. Prints the score with a
95% interval; a result whose interval spans 50% has not proven anything.
"""

import json
import math
import os
import subprocess
import sys
import time

import chess

import openings as opening_book

# Mirrors the platform runner: fd 1 is pointed at stderr before the agent is
# imported, so anything the agent prints cannot corrupt the protocol.
WORKER = r"""
import json, sys, os
protocol = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
sys.path.insert(0, sys.argv[1])
import agent
protocol.write(json.dumps({"ready": True}) + "\n")
protocol.flush()
for line in sys.stdin:
    req = json.loads(line)
    mv = agent.get_move(req["fen"], req["time_left_ms"])
    protocol.write(json.dumps({"move": mv}) + "\n")
    protocol.flush()
"""


def parse_spec(spec):
    """"dir" or "dir|VAR=VAL,VAR=VAL" -> (directory, env overrides).
    The env form drives the feature toggles, so any single heuristic can be
    A/B tested against the same build without a second checkout."""
    if "|" not in spec:
        return spec, {}
    directory, assignments = spec.split("|", 1)
    env = {}
    for pair in assignments.split(","):
        if not pair:
            continue
        key, _, value = pair.partition("=")
        env[key.strip()] = value.strip()
    return directory, env


class Engine:
    def __init__(self, spec, init_budget=None):
        if init_budget is None:
            # Machine load inflates numba compile time; the packaging gate
            # measures the real 90 s constraint on an idle box instead.
            init_budget = float(os.environ.get("BTC_INIT_BUDGET", "90"))
        directory, env_overrides = parse_spec(spec)
        label = os.path.basename(os.path.abspath(directory))
        if env_overrides:
            label += "[" + ",".join(f"{k}={v}" for k, v in env_overrides.items()) + "]"
        self.name = label
        self.proc = subprocess.Popen(
            [sys.executable, "-c", WORKER, os.path.abspath(directory)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            env={**os.environ, **env_overrides})
        started = time.perf_counter()
        line = self.proc.stdout.readline()
        self.init_s = time.perf_counter() - started
        if not line or not json.loads(line).get("ready"):
            raise RuntimeError(f"{self.name} failed to start")
        if self.init_s > init_budget:
            raise RuntimeError(f"{self.name} init {self.init_s:.1f}s over budget")

    def move(self, fen, time_left_ms):
        self.proc.stdin.write(json.dumps(
            {"fen": fen, "time_left_ms": int(time_left_ms)}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name} died")
        return json.loads(line)["move"]

    def stop(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


SIM_OVERHEAD_MS = float(os.environ.get("BTC_SIM_OVERHEAD", "0"))


def play(white, black, start_fen, base_ms, increment_ms):
    board = chess.Board(start_fen)
    clocks = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    players = {chess.WHITE: white, chess.BLACK: black}
    while True:
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome.winner is None:
                return 0.5, outcome.termination.name
            return (1.0 if outcome.winner == chess.WHITE else 0.0), \
                outcome.termination.name
        if board.ply() >= 600:
            return 0.5, "PLY_CAP"
        side = board.turn
        started = time.perf_counter()
        try:
            uci = players[side].move(board.fen(), clocks[side])
        except RuntimeError:
            return (0.0 if side == chess.WHITE else 1.0), "CRASH"
        # The platform charges more per move than wall time alone: FEN
        # handling, referee measurement slack, scheduling. Locally that
        # cost is near zero, so without this the arena cannot reproduce
        # the regime where the increment is eaten and the clock pins at
        # its floor. Both arms pay it equally. Default 0 = old behaviour.
        clocks[side] -= ((time.perf_counter() - started) * 1000.0
                         + SIM_OVERHEAD_MS)
        if clocks[side] < 0:
            return (0.0 if side == chess.WHITE else 1.0), "FLAG"
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return (0.0 if side == chess.WHITE else 1.0), "MALFORMED"
        if mv not in board.legal_moves:
            return (0.0 if side == chess.WHITE else 1.0), "ILLEGAL"
        board.push(mv)
        clocks[side] += increment_ms


def _interval(score, games):
    if games == 0:
        return 0.0
    mean = score / games
    var = max(mean * (1 - mean), 1e-9)
    return 1.96 * math.sqrt(var / games)


def _elo(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / p - 1)


def _elo_to_score(elo):
    return 1.0 / (1.0 + 10 ** (-elo / 400.0))


def sprt_llr(wins, draws, losses, elo0=0.0, elo1=20.0):
    """Log-likelihood ratio for H1 (elo1) against H0 (elo0), trinomial model.

    This is what lets a match stop the moment it is decided instead of always
    playing a fixed number of games, and it is the *disciplined* version of
    stopping early. Two results this session show why the undisciplined version
    is dangerous: RFP_MARGIN=168 read **+70 elo at 25 games** and finished at
    **-3 over 200**; L1=512 read +63 at 50 and finished at +9. Watching a
    running score and stopping when it looks good is exactly how those become
    shipped mistakes.

    SPRT differs because the stopping rule is fixed in advance and the error
    rates are controlled: the boundaries below give roughly 5% false-accept and
    5% false-reject regardless of when the crossing happens. It also stops early
    on *bad* changes, which is where most of the time is actually saved.

    Score-model LLR: each game's contribution is the log ratio of its
    probability under the two hypotheses. Draw probability is estimated from the
    sample rather than assumed, so heavily drawish pairings - which ours are,
    with a shared opening book - are handled correctly.

    **The hypothesis pair is [0, 20] elo, not the conventional [0, 5].** How fast SPRT
    decides depends on how far apart H0 and H1 are: at elo1 = 8 the two
    hypotheses are 0.5000 against 0.5115, so even the +200 elo NNUE result
    accumulates an LLR of only 1.47 over 75 games and would not stop. The
    convention
    uses tight bounds because it is chasing 1-2 elo regressions over tens of
    thousands of games. We are looking for 20-40 elo changes with hours left, so
    the bounds are set to the effect size worth shipping."""
    games = wins + draws + losses
    if games == 0 or wins == 0 or losses == 0:
        return 0.0
    win_rate = wins / games
    draw_rate = draws / games
    loss_rate = losses / games
    score = win_rate + 0.5 * draw_rate
    variance = (win_rate * (1 - score) ** 2
                + draw_rate * (0.5 - score) ** 2
                + loss_rate * score ** 2)
    if variance <= 0:
        return 0.0
    s0, s1 = _elo_to_score(elo0), _elo_to_score(elo1)
    return games * (s1 - s0) * (2 * score - s0 - s1) / (2 * variance)


def sprt_bounds(alpha=0.05, beta=0.05):
    """(lower, upper) log-likelihood bounds. Cross low: reject; high: accept."""
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def main():
    challenger_dir = sys.argv[1]
    champion_dir = sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    base_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    increment_ms = int(sys.argv[5]) if len(sys.argv) > 5 else 100

    challenger = Engine(challenger_dir)
    champion = Engine(champion_dir)
    print(f"init: {challenger.name} {challenger.init_s:.1f}s, "
          f"{champion.name} {champion.init_s:.1f}s", flush=True)

    # each opening is played once with each colour, so a pair cancels any
    # opening bias; needing more pairs than openings is flagged because the
    # 95% interval assumes independent games
    book = opening_book.load()
    pairs = (games + 1) // 2
    if pairs > len(book):
        print(f"warning: {pairs} game pairs but only {len(book)} openings, "
              f"games repeat and the interval is optimistic")

    score = 0.0
    wins = draws = losses = 0
    try:
        for game in range(games):
            opening = book[(game // 2) % len(book)]
            challenger_white = game % 2 == 0
            if challenger_white:
                result, term = play(challenger, champion, opening, base_ms,
                                    increment_ms)
                ours = result
            else:
                result, term = play(champion, challenger, opening, base_ms,
                                    increment_ms)
                ours = 1.0 - result
            score += ours
            wins += ours == 1.0
            draws += ours == 0.5
            losses += ours == 0.0
            print(f"game {game + 1}/{games}: {ours} [{term}] "
                  f"running {score}/{game + 1}", flush=True)
    finally:
        challenger.stop()
        champion.stop()

    played = wins + draws + losses
    pct = score / played if played else 0.0
    margin = _interval(score, played)
    print(f"\n{challenger.name} vs {champion.name}: "
          f"+{wins} ={draws} -{losses}")
    print(f"score {pct:.1%} +- {margin:.1%}  "
          f"elo {_elo(pct):+.0f} [{_elo(max(pct - margin, 1e-4)):+.0f}, "
          f"{_elo(min(pct + margin, 1 - 1e-4)):+.0f}]")


if __name__ == "__main__":
    main()
