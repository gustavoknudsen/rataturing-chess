"""Parallel A/B match. Same semantics as arena_ab.py, many games at once. Run:

    python arena_par.py <challenger_spec> <champion_spec> [games] [base_ms]
                        [increment_ms] [workers]

Why this exists: a 300-game match at 3 s + 0.05 s takes about 90 minutes
sequentially, and a 300-game match resolves only about 40 elo. With a
submission lock two days out, the number of hypotheses that can be tested is
set by match throughput and nothing else.

Only one engine thinks at a time within a game, so a sequential match leaves
nine of ten cores idle. This runs `workers` independent engine pairs, each
playing a disjoint slice of the game indices. Game i uses exactly the opening
and colour it would have used sequentially, so results stay comparable with
arena_ab.py.

Each pair pays the numba compile once at startup, so keep `workers` well below
the core count: the compiles happen simultaneously and are the heaviest moment
of the run. Four is a good default on a ten-core machine.

Caveat worth remembering: concurrent games share cores, so time management is
measured less faithfully than in a sequential match. Load hits both arms of a
pair equally, so an A/B comparison stays fair, but a *final* confirmation of
anything time-management-related should use arena_ab.py with one worker.
"""

import concurrent.futures
import sys
import threading

import openings as opening_book
from arena_ab import Engine, _elo, _interval, play, sprt_llr, sprt_bounds

_print_lock = threading.Lock()

# Engine.__init__ enforces the platform's 90 s init budget, and every pair pays
# a full numba compile at startup. Starting them all at once is self-defeating:
# with two matches running, fourteen simultaneous compiles pushed init to
# 109.6 s and the match aborted. Compiles are serialised so each engine sees a
# roughly idle machine; the games themselves still run fully in parallel.
_compile_lock = threading.Lock()


def run_slice(spec_a, spec_b, indices, book, base_ms, increment_ms, progress):
    """Play the given game indices on one dedicated pair of engines."""
    with _compile_lock:
        challenger = Engine(spec_a)
        champion = Engine(spec_b)
    results = []
    try:
        for game in indices:
            if progress[3]:
                break
            opening = book[(game // 2) % len(book)]
            if game % 2 == 0:
                result, term = play(challenger, champion, opening, base_ms,
                                    increment_ms)
                ours = result
            else:
                result, term = play(champion, challenger, opening, base_ms,
                                    increment_ms)
                ours = 1.0 - result
            results.append((game, ours, term))
            with _print_lock:
                progress[0] += ours
                progress[1] += 1
                progress[4 + (0 if ours == 1.0 else
                              (1 if ours == 0.5 else 2))] += 1
                wins, draws, losses = progress[4], progress[5], progress[6]
                llr = sprt_llr(wins, draws, losses)
                low, high = sprt_bounds()
                note = ""
                if llr >= high:
                    progress[3] = 1
                    note = "  SPRT accept"
                elif llr <= low:
                    progress[3] = 1
                    note = "  SPRT reject"
                print(f"game {progress[1]}/{progress[2]}: {ours} [{term}] "
                      f"running {progress[0]}/{progress[1]} "
                      f"LLR {llr:+.2f}{note}", flush=True)
    finally:
        challenger.stop()
        champion.stop()
    return results


def main():
    spec_a = sys.argv[1]
    spec_b = sys.argv[2]
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    base_ms = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    increment_ms = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 4

    book = opening_book.load()
    pairs = (games + 1) // 2
    if pairs > len(book):
        print(f"warning: {pairs} game pairs but only {len(book)} openings, "
              f"games repeat and the interval is optimistic")

    # Deal game indices round robin, keeping each colour pair (2i, 2i+1) on the
    # same worker so an opening is always played both ways under the same load.
    slices = [[] for _ in range(workers)]
    for pair_index in range(pairs):
        target = slices[pair_index % workers]
        target.append(2 * pair_index)
        if 2 * pair_index + 1 < games:
            target.append(2 * pair_index + 1)

    # [score, played, target, stop_flag, wins, draws, losses]
    progress = [0.0, 0, games, 0, 0, 0, 0]
    low, high = sprt_bounds()
    print(f"{games} games at {base_ms/1000:.1f}s+{increment_ms/1000:.2f}s "
          f"across {workers} engine pairs", flush=True)
    # SPRT stops the match as soon as the evidence is decisive either way, with
    # the stopping rule fixed in advance so it is not the optional-stopping trap
    # that made RFP_MARGIN=168 look like +70 elo at 25 games and finish at -3.
    # `games` is now a cap rather than a target: unclear results still run the
    # full distance, which is exactly what those two needed.
    print(f"SPRT H0 0 elo vs H1 20 elo, stop outside [{low:.2f}, {high:.2f}], "
          f"cap {games} games", flush=True)

    score = 0.0
    wins = draws = losses = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_slice, spec_a, spec_b, chunk, book,
                               base_ms, increment_ms, progress)
                   for chunk in slices if chunk]
        for future in concurrent.futures.as_completed(futures):
            for _, ours, _ in future.result():
                score += ours
                wins += ours == 1.0
                draws += ours == 0.5
                losses += ours == 0.0

    played = wins + draws + losses
    pct = score / played if played else 0.0
    margin = _interval(score, played)
    llr = sprt_llr(wins, draws, losses)
    if llr >= high:
        verdict = "SPRT ACCEPT - challenger is better"
    elif llr <= low:
        verdict = "SPRT REJECT - challenger is not better"
    else:
        verdict = "SPRT inconclusive at the game cap - read the interval"
    print(f"\n{spec_a} vs {spec_b}: +{wins} ={draws} -{losses}")
    print(f"score {pct:.1%} +- {margin:.1%}  "
          f"elo {_elo(pct):+.0f} [{_elo(max(pct - margin, 1e-4)):+.0f}, "
          f"{_elo(min(pct + margin, 1 - 1e-4)):+.0f}]")
    print(f"LLR {llr:+.2f}  {verdict}")


if __name__ == "__main__":
    main()
