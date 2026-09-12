# Rataturing

A chess engine for the AI Chessathon, written in Python and compiled with numba. **Finished 5th of 334 entrants** in the qualification Swiss and reached the London final.

Two evaluations exist. **Rataturing NNUE** uses a neural network trained from scratch for this entry, and is what ships. **Rataturing Classic** uses a hand-crafted evaluation and runs whenever no network is present.

## Contents

- [Competition result](#competition-result)
- [Play against it](#play-against-it)
- [Competition constraints](#competition-constraints)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Performance](#performance)
- [Development method](#development-method)
- [Further reading](#further-reading)

## Competition result

| | |
|---|---|
| Placing | **5th of 334** |
| Score | 10.0 / 13 |
| Record | 8 wins, 4 draws, 1 loss |
| Rating | 2853 |
| Tiebreak | Buchholz 113.5 |
| Outcome | qualified for the London final, a knockout among the top 50 |

Qualification was a 13-round Swiss played over locked builds, so every entrant submitted once and the same binary played all thirteen games. Half a point separated this entry from first place, and only the Buchholz tiebreak separated it from third and fourth. [Leaderboard](https://aichessathon.com/leaderboard?stage=finalset).

Every entrant ran on the same fixed platform under the same limits, listed in [Competition constraints](#competition-constraints) below. Most of this engine's design follows from them.

## Play against it

Releases include a UCI executable for Arena, Cute Chess and any other standard GUI, in both evaluations. Point the GUI at the executable and keep the folder intact.

The engine compiles itself with numba when it starts, which takes about a minute. That happens once per session rather than once per game, so only the first game waits. [`uci/README.md`](uci/README.md) explains what was tried to shorten it and why none of it worked.

Build them yourself with `python uci/build.py`. The build takes the engine source out of `submission.zip` rather than out of `src/`, so the released executable is verifiably the engine that competed, with nothing added but a protocol adapter.

## Competition constraints

Most of the engine's design follows from these.

| constraint | value |
|---|---|
| Time control | 120 s + 0.5 s per move, per side, wall time |
| Import budget | 90 s before the clock starts; no output in that window is a loss |
| Submission size | 50 MB unzipped |
| Hardware | one core |
| Entry point | `agent.py` at the zip root, exposing `get_move(fen, time_left_ms)` |

**Banned:** third-party engines and any wrapper, port or translation of one; published or pretrained networks; native binaries; obfuscated agents; tables that answer a middlegame position.

**Allowed:** your own prior work, self-trained networks, unrestricted training data, and a shipped table that answers the opening or the endgame, where the opening is a position whose move number is 20 or lower.

Rataturing is built on BetterThanCris, a C engine by the same author, since extended with C++ files and utilities, which the rules permit: "Your moves come from code you wrote." The network is trained from scratch. The opening book is gated at move 20 in `src/btc_book.py`, checked against the referee's own FEN. Full summary in [`docs/RULES.md`](docs/RULES.md).

## Repository structure

    src/          the engine. These 15 files, plus the network and books, are
                  exactly what ships. The zip is flat because the platform
                  does `import agent` at its root.
    tests/        correctness suite, about 20,000 assertions
    tools/        match arena, benchmarks, packaging, tuning
    uci/          UCI adapter and the release build
    training/     NNUE data pipeline and the notebook that trained the network
    book/         opening book pipeline: scrape, expand, label, merge, build
    docs/         competition rules and design decisions

The network (`src/net.npz`, 24 MB) and the two opening books (`src/*.bin`, 16 MB) are included, so a clone runs the engine that actually competed. They sit beside the engine rather than in a data directory because `btc_nnue.find_net()` and `btc_book._path()` both resolve relative to their own module, and the submission zip is flat.

## Getting started

    pip install -r requirements.txt

Build the submission:

    python tools/package.py

This checks that every import in a shipped file exists on the platform, writes `submission.zip`, extracts it to a temporary directory and plays a short game from it. It refuses to produce a zip that does not run.

Run the tests:

    python tests/run_tests.py       core correctness
    python tests/test_see.py        static exchange evaluation
    python tests/test_convert.py    endgame conversion
    python tests/test_draw.py       draw rules
    python tests/test_book.py       opening book and its move-20 gate
    python tests/test_uci.py        UCI protocol, 22 checks

`tests/test_eval.py` checks hand-crafted evaluation terms and expects the network disabled: run it with `BTC_NNUE=0`.

Measure a change:

    tools/searchbench.py      nodes to a fixed depth, the screening metric
    tools/arena_par.py        A/B match with a sequential probability ratio test
    tools/arena_ab.py         single worker, for anything clock-related

Reproducing the training or book pipelines needs `pip install -r requirements-dev.txt`. Do not install torch into the engine's own environment: it costs import time and resident memory the 90 s budget cannot spare.

## Performance

About 650,000 nodes per second at depth 11 on the development machine, idle:

    python tools/searchbench.py 11 3

Treat that as a reading of this machine on that day, not a property of the engine. The match hardware is a single EPYC 9V74 core and is slower. More importantly, the same command on the same build varies by well over a third with machine load: repeated runs here have given anywhere from 490,000 to 650,000. Only compare figures measured back to back under the same conditions.

Node counts at a fixed depth do not have that problem. They are deterministic, reproduce exactly across runs and machines, and are what the screening step in the development method below actually uses.

The network is not the bottleneck. Disabling it *lowers* throughput, to about 437,000 nodes per second, because the NNUE accumulator is updated incrementally through make and unmake while the hand-crafted evaluation recomputes pawn structure, king safety and mobility at every leaf. The size of that gap moves with load; its direction has held in every measurement.

## Development method

The search carries many features behind environment flags, defaulting off. numba folds a module-level constant at compile time, so a disabled feature costs nothing and both arms of a match can be built from one directory, differing only in environment.

1. Screen the change for node cost against the current build. This rejects anything the engine cannot afford before any match time is spent.
2. Implement it behind a flag, default off.
3. Review it against the failure modes this codebase actually has: silent out-of-bounds indexing, shared per-ply scratch state, sign errors at negative depth, constants that are mathematically inert.
4. Run the test suite.
5. Run an A/B match, and read the confidence interval rather than the headline.

Two habits earned their place.

**Read the margin, not the verdict.** `test_convert` passes a KBN versus K conversion at anything under 50 moves, so it reported success identically for mate in 15, mate in 18 and mate in 26. Two real regressions were invisible at the gate's own threshold, and were caught only by reading the printed mate distance.

**A trend inside a match is usually the opening set.** Openings are assigned in order from a fixed list, so the first and second halves of a match are different positions, not the same position measured twice. One change read 58 percent over 43 games and 52.6 percent over 345.

## Further reading

- [`docs/RULES.md`](docs/RULES.md) - the competition rules, verified against the published documentation
- [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md) - why the engine is built this way, with the measurements behind each choice
- [`book/README.md`](book/README.md) - the opening book pipeline, including how to obtain its inputs and re-run it
- [`training/`](training/) - the NNUE data pipeline and the notebook that trained the shipped network
