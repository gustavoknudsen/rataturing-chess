"""Fixed-depth move accuracy: mean centipawn loss under deep analysis.

    python evalsuite.py build <suite.json> [positions] [sf_depth]
    python evalsuite.py run <suite.json> <label>=<net.npz>[:units] ... [depth]

Why this exists. `nnue_rank.py` scores a net on held-out rows, which measures
how well it predicts an evaluation. It cannot see what the net does *inside the
search*, and that gap is not theoretical: a net 4.2% better on validation loss
scored 49.7% in games, and a search change with 31.6% fewer nodes lost 29 elo.
Nodes and time-to-depth are equally blind - they price efficiency, never
accuracy.

This measures the thing both of those miss. Every net searches the *same*
positions to the *same fixed depth*, so speed is removed from the comparison by
construction and what is left is the quality of the move chosen. The metric is
mean centipawn loss, measured by analysing every position offline to a far
greater depth than the engine itself reaches.

Read it as a ranking instrument, not a verdict. It resolves accuracy at a fixed
depth; it says nothing about the speed a wider net gives up, and only an SPRT
prices the two against each other.

**Calibrate before trusting.** Run the shipped 512 net against k2_1024 first.
Those two scored 49.7% in a real match while validation loss called the 1024
net 4.2% better, so a metric that ranks them level carries information
validation loss does not, and a metric that repeats the 4.2% verdict should be
discarded rather than argued with.

Regret is measured symmetrically. Both the best move and ours are scored
from the position *after* they are played, at the same depth, so neither gets
an extra ply. The root choice is used only for the agreement
percentage. Child evaluations are cached in the suite file keyed by position
and move, so the second and third net cost almost nothing.

Per-net units matter. `NET_UNITS` rescales every search margin and belongs to
one specific net; running a net at another net's units changes the search
rather than the evaluation. Pass it after a colon, or the run refuses.
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import json
import os
import random
import subprocess
import sys

ENGINE_DEFAULT = (r"C:\Users\Gustavo\Documents\Chess"
              r"\BanksiaGui-0.58-win64\bsg-engines\stockfish_15.1_x64_bmi2.exe")
ENGINE_PATH = os.environ.get("BTC_ANALYSER", ENGINE_DEFAULT)

# A mate is not a centipawn score, but regret has to be a number. Scoring mates
# on a scale above any material advantage keeps "missed a forced mate" ranked
# worse than "dropped a queen", which is the ordering we want, while the cap in
# the summary stops one such position from owning the mean.
MATE_CP = 20000
REGRET_CAP = 200

# Positions are drawn from our own games, so the distribution is the one the
# engine actually meets. Anything before this ply is opening book or close to
# it and carries no information about the evaluation.
MIN_PLY = 20


class UciEngine:
    """A UCI engine held open over many analyses."""

    def __init__(self, path=ENGINE_PATH, threads=1, hash_mb=256):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._ready()

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token):
        lines = []
        for line in self.proc.stdout:
            line = line.strip()
            lines.append(line)
            if line.startswith(token):
                return lines
        raise RuntimeError("analysis engine exited early")

    def _ready(self):
        self._send("isready")
        self._read_until("readyok")

    def analyse(self, fen, moves, depth):
        """Returns (best_move_uci, score_cp) from the side to move's view.

        The hash is cleared first so a position's score never depends on what
        was analysed before it - the suite has to give the same numbers on a
        re-run or the cache is not a cache."""
        self._send("ucinewgame")
        self._ready()
        position = f"position fen {fen}"
        if moves:
            position += " moves " + " ".join(moves)
        self._send(position)
        self._send(f"go depth {depth}")
        return _parse_analysis(self._read_until("bestmove"))

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def _parse_analysis(lines):
    """Last scored info line plus the bestmove that followed it."""
    score = None
    best = None
    for line in lines:
        if line.startswith("info ") and " score " in line:
            score = _parse_score(line.split())
        elif line.startswith("bestmove"):
            parts = line.split()
            best = parts[1] if len(parts) > 1 else None
    if score is None:
        raise RuntimeError("no score in engine output")
    return best, score


def _parse_score(tokens):
    index = tokens.index("score")
    kind, value = tokens[index + 1], int(tokens[index + 2])
    if kind == "mate":
        sign = 1 if value >= 0 else -1
        return sign * (MATE_CP - min(abs(value), 100))
    return value


def out_bucket(pieces, buckets):
    """Matches btc_nnue._out_bucket and nnue_train.out_bucket_of."""
    index = (pieces - 1) * buckets // 32
    return max(0, min(buckets - 1, index))


def collect_positions(paths, limit, seed=20260910):
    """Quiet, non-terminal positions from our own games, spread over buckets.

    Stratified on the 16-bucket grid because that is the finest one in play;
    the 8-bucket grid is a coarsening of it, so one sample serves both. Without
    stratification a sample of our games is nearly all middlegame and the
    piece-count buckets under test are represented by a handful of rows each."""
    import chess
    import chess.pgn

    seen = set()
    by_bucket = {}
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                _harvest_game(game, seen, by_bucket)

    rng = random.Random(seed)
    for rows in by_bucket.values():
        rng.shuffle(rows)
    return _round_robin(by_bucket, limit)


def _harvest_game(game, seen, by_bucket):
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        if board.ply() < MIN_PLY or board.is_check() or board.is_game_over():
            continue
        pieces = len(board.piece_map())
        if pieces < 8:
            continue
        fen = board.fen()
        key = fen.rsplit(" ", 2)[0]
        if key in seen:
            continue
        seen.add(key)
        by_bucket.setdefault(out_bucket(pieces, 16), []).append((fen, pieces))


def _round_robin(by_bucket, limit):
    """One position per bucket per pass, so thin buckets are not crowded out."""
    picked = []
    order = sorted(by_bucket)
    depth = 0
    while len(picked) < limit:
        added = 0
        for bucket in order:
            rows = by_bucket[bucket]
            if depth < len(rows) and len(picked) < limit:
                picked.append(rows[depth])
                added += 1
        if not added:
            break
        depth += 1
    return picked


def build(suite_path, count, sf_depth, pgn_paths):
    rows = collect_positions(pgn_paths, count)
    print(f"collected {len(rows)} positions from {len(pgn_paths)} pgn files")
    engine = UciEngine()
    suite = {"sf_depth": sf_depth, "positions": [], "child": {}}
    try:
        for index, (fen, pieces) in enumerate(rows):
            _label(engine, suite, fen, pieces, sf_depth)
            if (index + 1) % 20 == 0:
                print(f"  labelled {index + 1}/{len(rows)}", flush=True)
                _write_json(suite_path, suite)
    finally:
        engine.close()
    _write_json(suite_path, suite)
    _report_build(suite)


def _label(engine, suite, fen, pieces, sf_depth):
    best, score = engine.analyse(fen, [], sf_depth)
    if best is None or best == "(none)":
        return
    suite["positions"].append({"fen": fen, "pieces": pieces,
                               "best": best, "root": score})
    _child_score(engine, suite, fen, best, sf_depth)


def _child_score(engine, suite, fen, move, sf_depth):
    """Score of `fen` after `move`, from the mover's view, cached."""
    key = fen + "|" + move
    cached = suite["child"].get(key)
    if cached is not None:
        return cached
    _, score = engine.analyse(fen, [move], sf_depth)
    value = -score
    suite["child"][key] = value
    return value


def _write_json(path, suite):
    with open(path, "w", encoding="ascii") as handle:
        json.dump(suite, handle)


def _read_json(path):
    with open(path, encoding="ascii") as handle:
        return json.load(handle)


def _report_build(suite):
    counts = {}
    for row in suite["positions"]:
        counts[out_bucket(row["pieces"], 16)] = \
            counts.get(out_bucket(row["pieces"], 16), 0) + 1
    print(f"\nsuite: {len(suite['positions'])} positions, "
          f"analysis depth {suite['sf_depth']}")
    print("bucket16 counts: " + ", ".join(
        f"{b}:{counts[b]}" for b in sorted(counts)))


def worker(suite_path, depth):
    """Search every position to a fixed depth and print the move chosen.

    Runs as a subprocess because btc_eval binds the network at import time from
    BTC_NET, so one process can hold exactly one net."""
    import numpy as np

    import btc_nrt  # noqa: F401  - must precede any njit compile
    import btc_core as core
    import btc_search as search

    suite = _read_json(suite_path)
    state = search.SearchState()
    bb, st = core.new_board()
    keys = np.zeros(8, dtype=np.uint64)
    core.parse_fen(suite["positions"][0]["fen"], bb, st)
    keys[0] = bb[core.HASH]
    search.search_position(state, bb, st, keys, 1, soft_ms=0,
                           hard_ms=3_600_000, max_depth=3)

    for row in suite["positions"]:
        _clear(state)
        core.parse_fen(row["fen"], bb, st)
        keys[0] = bb[core.HASH]
        packed, score, reached, nodes = search.search_position(
            state, bb, st, keys, 1, soft_ms=0, hard_ms=3_600_000,
            max_depth=depth)
        print(json.dumps({"fen": row["fen"], "move": core.move_to_uci(packed),
                          "score": int(score), "depth": int(reached),
                          "nodes": int(nodes)}), flush=True)


def _clear(state):
    state.tt_key[:] = 0
    state.tt_data[:] = 0
    state.main_hist[:] = 0
    state.cap_hist[:] = 0
    state.cont_hist[:] = 0
    state.counters[:] = 0


def _parse_spec(spec):
    """label=path[:units] -> (label, path, units or None)."""
    label, _, rest = spec.partition("=")
    if not rest:
        raise SystemExit(f"bad net spec {spec!r}, want label=path[:units]")
    path, colon, units = rest.rpartition(":")
    if not colon or not units.isdigit():
        return label, rest, None
    return label, path, int(units)


def _run_net(suite_path, spec, depth):
    label, path, units = _parse_spec(spec)
    if units is None:
        raise SystemExit(
            f"{label}: no units. NET_UNITS belongs to one net and rescales "
            f"every search margin - derive it with nnue_gate.py and pass "
            f"{label}={path}:<units>.")
    # btc_nnue.find_net falls back to the local net.npz when BTC_NET names a
    # file that does not exist. That is right for the engine and fatal here: a
    # mistyped path silently scores the same net twice and the run looks like a
    # dead heat. Fail instead.
    full = os.path.abspath(path)
    if not os.path.exists(full):
        raise SystemExit(
            f"{label}: no net at {full!r}. A Git Bash path like /d/... is not "
            f"a Windows path - use D:/... instead.")
    env = dict(os.environ)
    env["BTC_NET"] = full
    env["BTC_NET_UNITS"] = str(units)
    print(f"[{label}] {path} units {units}, depth {depth}", flush=True)
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "_worker", suite_path,
         str(depth)], env=env, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{label} failed:\n{out.stderr[-2000:]}")
    return label, [json.loads(line) for line in out.stdout.splitlines()
                   if line.startswith("{")]


def run(suite_path, specs, depth):
    suite = _read_json(suite_path)
    results = [_run_net(suite_path, spec, depth) for spec in specs]
    _fill_cache(suite_path, suite, results)
    scored = [(label, _score(suite, moves)) for label, moves in results]
    _report(scored)


def _fill_cache(suite_path, suite, results):
    """One analysis per new (position, move) pair, shared across nets."""
    index = {row["fen"]: row for row in suite["positions"]}
    wanted = {(item["fen"], item["move"]) for _, moves in results
              for item in moves if item["fen"] in index}
    missing = [pair for pair in sorted(wanted)
               if pair[0] + "|" + pair[1] not in suite["child"]]
    if not missing:
        return
    print(f"analysis: {len(missing)} new (position, move) pairs to score",
          flush=True)
    engine = UciEngine()
    try:
        for count, (fen, move) in enumerate(missing, 1):
            _child_score(engine, suite, fen, move, suite["sf_depth"])
            if count % 20 == 0:
                print(f"  scored {count}/{len(missing)}", flush=True)
    finally:
        engine.close()
    _write_json(suite_path, suite)


def _score(suite, moves):
    """Per-position regret in centipawns, plus the bucket it belongs to."""
    index = {row["fen"]: row for row in suite["positions"]}
    rows = []
    for item in moves:
        row = index.get(item["fen"])
        if row is None:
            continue
        ours = suite["child"].get(row["fen"] + "|" + item["move"])
        theirs = suite["child"].get(row["fen"] + "|" + row["best"])
        if ours is None or theirs is None:
            continue
        rows.append({"fen": row["fen"], "regret": max(0, theirs - ours),
                     "agree": item["move"] == row["best"],
                     "pieces": row["pieces"], "nodes": item["nodes"]})
    return rows


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _summary(rows):
    """Capped regret is the headline; the raw mean is deliberately not shown.

    A missed forced mate enters regret at MATE_CP, and one of them moves the
    uncapped mean by hundreds of centipawns - so that number measures whether
    the sample happened to contain a mate rather than how well the net plays.
    Counting the positions at or over the cap keeps that information without
    letting it swamp everything else."""
    regrets = [r["regret"] for r in rows]
    capped = [min(r, REGRET_CAP) for r in regrets]
    return {"n": len(rows), "capped": _mean(capped),
            "bad": sum(1 for r in regrets if r >= REGRET_CAP),
            "agree": 100.0 * _mean([1.0 if r["agree"] else 0.0 for r in rows]),
            "nodes": _mean([r["nodes"] for r in rows])}


def _report(scored):
    print("\n" + "=" * 74)
    print(f"{'net':<14}{'n':>6}{'agree%':>9}{'regret':>10}"
          f"{'blunders':>10}{'nodes':>12}")
    print("-" * 74)
    for label, rows in scored:
        s = _summary(rows)
        print(f"{label:<14}{s['n']:>6}{s['agree']:>9.1f}{s['capped']:>10.1f}"
              f"{s['bad']:>10}{s['nodes']:>12,.0f}")
    _report_buckets(scored)
    _report_pairs(scored)
    print(f"\nRegret is the mean centipawn loss, capped at {REGRET_CAP}; blunders "
          f"counts the positions at or over that cap. Lower is better for "
          f"both. Fixed depth, so this prices accuracy only - the speed a "
          f"wider net gives up is not in these numbers, and only an SPRT"
          f" puts the two on one scale.")


def _report_buckets(scored, buckets=8):
    print(f"\ncapped regret by output bucket ({buckets} buckets)")
    labels = [label for label, _ in scored]
    print("bucket  pieces   " + "".join(f"{l:>12}" for l in labels))
    per = [{} for _ in scored]
    for column, (_, rows) in enumerate(scored):
        for row in rows:
            bucket = out_bucket(row["pieces"], buckets)
            per[column].setdefault(bucket, []).append(
                min(row["regret"], REGRET_CAP))
    for bucket in sorted({b for table in per for b in table}):
        span = _bucket_span(bucket, buckets)
        cells = "".join(f"{_mean(table.get(bucket, [])):>12.1f}"
                        for table in per)
        count = len(per[0].get(bucket, []))
        print(f"{bucket:>6}  {span:<7}  {cells}   (n={count})")


def _bucket_span(bucket, buckets):
    lo = (bucket * 32) // buckets + 1
    hi = ((bucket + 1) * 32) // buckets
    return f"{lo}-{hi}"


def _report_pairs(scored):
    """Paired differences. Every net saw the same positions, so pairing them
    removes position difficulty from the noise and resolves far more than two
    independent means would."""
    if len(scored) < 2:
        return
    print("\npaired difference in capped regret, vs the first net")
    base_label, base_rows = scored[0]
    base = {r["fen"]: min(r["regret"], REGRET_CAP) for r in base_rows}
    for label, rows in scored[1:]:
        # Paired on the position, not on list order: a net whose move missed
        # the analysis cache is dropped from its own list, and pairing by
        # index would then silently compare different positions.
        diffs = [min(r["regret"], REGRET_CAP) - base[r["fen"]]
                 for r in rows if r["fen"] in base]
        pairs = len(diffs)
        mean = _mean(diffs)
        error = _stderr(diffs)
        verdict = "better" if mean < 0 else "worse"
        print(f"  {label} vs {base_label}: {mean:+.1f} +- {1.96 * error:.1f} "
              f"cp ({verdict}, n={pairs})")


def _stderr(values):
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return (variance / len(values)) ** 0.5


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    mode = sys.argv[1]
    if mode == "build":
        suite_path = sys.argv[2]
        count = int(sys.argv[3]) if len(sys.argv) > 3 else 400
        sf_depth = int(sys.argv[4]) if len(sys.argv) > 4 else 18
        pgns = sys.argv[5:] or _default_pgns()
        build(suite_path, count, sf_depth, pgns)
    elif mode == "run":
        suite_path = sys.argv[2]
        specs = [a for a in sys.argv[3:] if "=" in a]
        tail = [a for a in sys.argv[3:] if "=" not in a]
        run(suite_path, specs, int(tail[0]) if tail else 12)
    elif mode == "_worker":
        worker(sys.argv[2], int(sys.argv[3]))
    else:
        raise SystemExit(f"unknown mode {mode!r}")


def _default_pgns():
    root = os.environ.get("BTC_PGN_DIR", ".")
    found = [os.path.join(root, name) for name in sorted(os.listdir(root))
             if name.endswith(".pgn")]
    if not found:
        raise SystemExit(f"no .pgn files in {root}; pass paths explicitly")
    return found


if __name__ == "__main__":
    main()
