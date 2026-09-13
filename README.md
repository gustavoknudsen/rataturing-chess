# Rataturing

A chess engine for the AI Chessathon 2026, written in Python and compiled with numba. **5th of 334 entrants** in the qualification Swiss, then **5th to 8th** in the London finals knockout, where it lost the quarter-final 1.5 to 2.5 to the eventual winner. Rated afterwards at about **3390 CCRL Blitz** against engines with published ratings.

Two evaluations exist. **Rataturing NNUE** uses a neural network trained from scratch for this entry, and is what competed. **Rataturing Classic** uses a hand-crafted evaluation and runs whenever no network is present.

## Contents

- [Competition result](#competition-result)
- [Play against it](#play-against-it)
- [Competition constraints](#competition-constraints)
- [How it works](#how-it-works)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Performance](#performance)
- [Strength estimate](#strength-estimate)
- [Development method](#development-method)
- [Further reading](#further-reading)

## Competition result

| stage | format | result |
|---|---|---|
| Qualification Swiss, 11 September | 13 rounds, 334 entrants, locked builds | **5th**, 10.0 / 13, 8 wins 4 draws 1 loss, rating 2853 |
| London finals knockout, 12 September | single elimination, four games per round | **5th to 8th**, lost the quarter-final 1.5 to 2.5 to AlphaFish, who won the event |

Half a point separated this entry from first place in the Swiss, and only the Buchholz tiebreak from third and fourth. The final began with a surprise constraint: the init budget was cut from 90 s to 30 s at 10:30, with builds locked at 14:00. How that was answered is in [`docs/FINALS_DAY.md`](docs/FINALS_DAY.md). Leaderboards: [Swiss](https://aichessathon.com/leaderboard?stage=finalset), [knockout](https://aichessathon.com/leaderboard?stage=knockout).

Every entrant ran on the same fixed platform under the same limits, listed in [Competition constraints](#competition-constraints). Most of the engine's design follows from them.

## Play against it

Releases include a UCI executable for Arena, Cute Chess and any other standard GUI. Point the GUI at the executable and keep the folder intact.

| release | engine | contents |
|---|---|---|
| **1.0** | the build that played the qualification Swiss | NNUE and Classic executables |
| **1.1** | the build that played the London final | NNUE executable, plus the submission zip exactly as uploaded |

The engine compiles itself with numba when it starts, which takes about a minute. That happens once per session rather than once per game, so only the first game waits. [`uci/README.md`](uci/README.md) explains what was tried to shorten it and why none of it worked.

Build them yourself with `python uci/build.py`. The build takes the engine source out of `submission.zip` rather than out of `src/`, so the released executable is verifiably the engine that competed, with nothing added but a protocol adapter.

## Competition constraints

| constraint | value |
|---|---|
| Time control | 120 s + 0.5 s per move, per side, wall time |
| Init budget | 90 s in the qualifier, **30 s in the final**; no output in that window is a loss |
| Submission size | 50 MB unzipped |
| Hardware | one core, 2 GB, no network, scratch space wiped between games |
| Entry point | `agent.py` at the zip root, exposing `get_move(fen, time_left_ms)` |
| Process model | one process per game, frozen whenever it is not our move |

**Banned:** third-party engines and any wrapper, port or translation of one; published or pretrained networks; native binaries; obfuscated agents; tables that answer a middlegame position.

**Allowed:** your own prior work, self-trained networks, unrestricted training data, and a shipped table that answers the opening or the endgame, where the opening is a position whose move number is 20 or lower.

Rataturing is built on BetterThanCris, a C engine by the same author, since extended with C++ files and utilities, which the rules permit: "Your moves come from code you wrote." The network is trained from scratch. The opening books are gated at move 20 in `src/btc_book.py`, checked against the referee's own FEN. Full summary in [`docs/RULES.md`](docs/RULES.md).

## How it works

**Search.** Bitboard move generation with magic sliders, make/unmake, iterative deepening with aspiration windows, and a negamax with transposition table, null move, late move reductions, late move pruning, futility and reverse futility, SEE pruning, singular extensions with multicut, ProbCut, internal iterative reductions, and main, capture, continuation and correction histories. numba cannot compile mutual recursion, so the whole main search is one function.

**Evaluation.** A king-bucketed NNUE: 32 king buckets by 768 features into a 512-wide layer, 8 output buckets keyed on piece count, quantised to int16. The accumulator is updated incrementally through make and unmake. Trained from scratch on public engine-labelled data; the training pipeline and notebook are in `training/`.

**Opening books.** Three Polyglot files: one built on finals day from the positions the tournament had actually used, one built before the qualifier from the published curated starts, and a hedge for the standard start. All three are gap-filled from public data and labelled by this engine.

**The final's staged wrapper.** The engine's numba compile takes about 40 s on the platform against a 30 s budget. `staged/agent_staged.py` starts the compile on a thread, spends 26 s of the init window waiting for it, and pays the rest from the clock on move one, where the book usually answers. It is what shipped for the knockout.

**Time management.** Budget per move from the clock and an assumed number of moves remaining; both constants were recalibrated from the platform's own clock lines on finals day.

## Repository structure

    src/          the engine. These 16 files, the network, the attack tables
                  and the books are exactly what ships. The zip is flat because
                  the platform does `import agent` at its root.
    staged/       the wrapper that met the 30 s init budget, and its build
    tests/        correctness suite, about 20,000 assertions
    tools/        match arena, benchmarks, packaging, tuning, clock simulator
    uci/          UCI adapter and the release build
    training/     NNUE data pipeline and the notebook that trained the network
    book/         opening book pipeline: scrape, expand, label, merge, build
    docs/         rules, design decisions, the finals-day record, research notes

The network (`src/net.npz`, 24 MB) and the three opening books (`src/*.bin`, 24 MB) are included, so a clone runs the engine that actually competed. They sit beside the engine rather than in a data directory because `btc_nnue.find_net()` and `btc_book._path()` both resolve relative to their own module, and the submission zip is flat.

## Getting started

    pip install -r requirements.txt

Build the submission:

    python tools/package.py           the flat engine zip, played from a temp dir before it is accepted
    python staged/build.py --zip      the finals layout on top of it

The first checks that every import in a shipped file exists on the platform, writes `submission.zip`, extracts it and plays a short game from it. The second renames the engine to `agent_real.py`, adds the wrapper and the fallback, and refuses to continue unless every engine file hashes equal to the archive.

Run the tests:

    python tests/run_tests.py         core correctness, perft
    python tests/test_search.py       search against a reference implementation
    python tests/test_accumulator.py  incremental NNUE accumulator equals a full refresh
    python tests/test_book.py         opening books and their move-20 gate
    python tests/test_staged.py       the staged wrapper: deadline, handover, fallback
    python tests/test_uci.py          UCI protocol

`tests/test_eval.py` checks hand-crafted evaluation terms and expects the network disabled: run it with `BTC_NNUE=0`. `tests/test_nnue.py` needs the training data set and is not runnable from a clone.

Measure a change:

    tools/searchbench.py      nodes to a fixed depth, the screening metric
    tools/arena_par.py        A/B match with a sequential probability ratio test
    tools/arena_ab.py         single worker, for anything clock-related
    tools/tc_tune.py          the clock budget simulated over whole games

Reproducing the training or book pipelines needs `pip install -r requirements-dev.txt`. Do not install torch into the engine's own environment: it costs import time and resident memory.

## Performance

About 470,000 nodes per second at depth 9 on the development machine and about 450,000 on the platform's core, from the match logs. Treat local figures as a reading of one machine on one day: the same command on the same build varies by more than a third with load and core placement, and only figures measured back to back under the same conditions compare.

Node counts at a fixed depth do not have that problem. `python tools/searchbench.py 9` prints a total that is deterministic across runs and machines, so it identifies the search exactly. For the shipped engine it reads **268059**, and every change that is meant to be pure speed has to leave it untouched. It is the cheapest gate in the project and the one that caught the most.

Three speed changes verified that way on finals day did not ship for lack of match time and are in `src/btc_search.py` behind flags that default off: the static evaluation cached in the transposition table, the accumulator built lazily in the child, and pick-best-and-shift move selection. Each has its measurements beside its flag, and the shipped path is unchanged when they are off.

The network is not the bottleneck. Disabling it lowers throughput, because the NNUE accumulator is updated incrementally while the hand-crafted evaluation recomputes pawn structure, king safety and mobility at every leaf.

## Strength estimate

Rataturing NNUE 1.1, the finals build, was rated against engines with published CCRL Blitz ratings, under CCRL Blitz conditions: 2 min + 1 s, one thread, 128 MB hash, no books, no tablebases, neutral 8-ply openings played with both colours. An adaptive gauntlet picked the opponent nearest the running estimate in blocks of eight games, 427 games in all.

**About 3390 CCRL Blitz, plus or minus 40.**

| opponent | rating | games | score | performance |
|---|---|---|---|---|
| Stockfish 10 | 3509 | 64 | 36.7% | 3414 |
| Komodo 13.02 | 3461 | 64 | 52.3% | 3477 |
| Pedantic 2.1 | 3400 | 64 | 52.3% | 3416 |
| Nalwald 19 | 3340 | 64 | 54.7% | 3373 |
| Maelstrom 3.3 | 3314 | 64 | 57.8% | 3369 |
| StockNemo 5.7 | 3269 | 54 | 63.0% | 3361 |
| Gull 3 | 3159 | 24 | 60.4% | 3232 |

Inverse-variance weighted over every opponent, including three weaker ones decided early: 3393, standard error 12, reduced chi-square 2.28, overdispersion-inflated interval plus or minus 37. The estimate inherits the opponents' own rating errors and was measured on a laptop with four games at once, so read it as a placement on the CCRL scale rather than a CCRL entry. The C engine it was ported from is listed at 2933.

The gauntlet tool is its own repository, [chess-elo-gauntlet](https://github.com/gustavoknudsen/chess-elo-gauntlet).

## Development method

The search carries many features behind environment flags, defaulting off. numba folds a module-level constant at compile time, so a disabled feature costs nothing and both arms of a match can be built from one directory, differing only in environment.

1. Screen the change for node cost against the current build. This rejects anything the engine cannot afford before any match time is spent.
2. Implement it behind a flag, default off.
3. Review it against the failure modes this codebase actually has: silent out-of-bounds indexing, shared per-ply scratch state, sign errors at negative depth, constants that are mathematically inert.
4. Run the test suite.
5. Run an A/B match, and read the confidence interval rather than the headline.

## Further reading

- [`docs/FINALS_DAY.md`](docs/FINALS_DAY.md) - the 30 s constraint, the staged wrapper, and everything that changed on the day
- [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) - why the engine is built this way, with the measurements behind each choice
- [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) - how the engine, the network and the book were made and tested
- [`docs/RULES.md`](docs/RULES.md) - the competition rules, verified against the published documentation
- [`docs/research/`](docs/research/) - the port audit, the search and speed research, and what was learned training the networks
- [`book/README.md`](book/README.md) - the opening book pipeline, including how to obtain its inputs and re-run it
- [`training/`](training/) - the NNUE data pipeline and the notebook that trained the shipped network
