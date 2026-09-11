# Rataturing

A chess engine for the AI Chessathon, written in Python and compiled with numba. Two evaluations exist: Rataturing NNUE, which uses a neural network trained from scratch for this entry and is what ships, and Rataturing, which uses a hand-crafted evaluation and runs whenever no network is present.

The engine is built on BetterThanCris (BTC), a C++ engine by the same author. The competition bans third-party engines and any wrapper, port or translation of one, and explicitly permits your own work: "Your moves come from code you wrote." BTC is the author's own engine, so building on it is within the rules. The network is trained from scratch, which is also required, since published or pretrained networks are banned. See `docs/RULES.md`.

## Layout

    src/          the engine. These 15 files, plus the network and books, are
                  exactly what ships. The submission zip is flat because the
                  platform does `import agent` at its root.
    tests/        correctness suite, about 20,000 assertions
    tools/        match arena, benchmarks, packaging, tuning, UCI bridge
    training/     NNUE data pipeline and the notebook that trained the net
    docs/         design decisions and a summary of the competition rules

The network (`src/net.npz`) and the opening books (`src/*.bin`) are not in the repository. They are large, and they live beside the engine because `btc_nnue.find_net()` and `btc_book._path()` both resolve relative to their own module.

## Build the submission

    .venv/Scripts/python.exe tools/package.py

This validates that every import in a shipped file is available on the platform, writes `submission.zip`, extracts it to a temporary directory, and plays a short game from it. It refuses to produce a zip that does not run.

## Run the tests

    .venv/Scripts/python.exe tests/run_tests.py      core correctness
    .venv/Scripts/python.exe tests/test_see.py       static exchange evaluation
    .venv/Scripts/python.exe tests/test_convert.py   endgame conversion
    .venv/Scripts/python.exe tests/test_draw.py      draw rules
    .venv/Scripts/python.exe tests/test_book.py      opening book and its gate

`tests/test_eval.py` checks hand-crafted evaluation terms and expects the network disabled: run it with `BTC_NNUE=0`.

## Measure a change

    tools/searchbench.py      nodes to a fixed depth, the screening metric
    tools/arena_par.py        A/B match with sequential probability ratio test
    tools/arena_ab.py         single-worker match, for anything clock-related

## Performance

About 627,000 nodes per second at depth 11 on the development machine, idle. Two things to know before comparing that against anything else.

The match machine is a single EPYC 9V74 core and is slower, so expect roughly 1.5x to 2x less there. Separately, the same benchmark on the same build varies by 1.6x between an idle machine and a loaded one, so only compare numbers measured back to back under the same conditions. Node counts at fixed depth are deterministic and do transfer; nodes per second, init time and reachable depth do not.

The network is not the speed bottleneck. Disabling it drops the engine to about 381,000 nodes per second, because the NNUE accumulator is updated incrementally through make/unmake while the hand-crafted evaluation recomputes pawn structure, king safety and mobility at every leaf.

## How the engine was developed

The search carries many features behind environment flags, defaulting off. numba folds a module-level constant at compile time, so a disabled feature costs nothing and both arms of a match can be built from one directory, differing only in environment. That is the whole method:

1. Screen the change for node cost against the current build. This rejects anything the engine cannot afford before any match time is spent.
2. Implement it behind a flag, default off.
3. Review it against the failure modes this codebase actually has: silent out-of-bounds indexing, shared per-ply scratch state, sign errors at negative depth, constants that are mathematically inert.
4. Run the test suite.
5. Run an A/B match, and read the confidence interval rather than the headline.

Two habits earned their place and are worth repeating:

**Read the margin, not the verdict.** `test_convert` passes a KBN versus K conversion at anything under 50 moves, so it reported success identically for mate in 15, mate in 18 and mate in 26. Two real regressions were invisible at the gate's own threshold and were caught only by reading the printed mate distance.

**A trend inside a match is usually the opening set.** Openings are assigned in order from a fixed list, so the first and second halves of a match are different positions, not the same position measured twice. One change read 58 percent over 43 games and 52.6 percent over 345.

## Constraints worth knowing

- 90 seconds to import before the clock starts. No output in that window is a loss. Measured cold init is about 56 seconds.
- 50 MB unzipped. The current submission is 41.9 MB, most of it the network and the two books.
- One core, 120 seconds per side plus 0.5 seconds per move.
- No third-party engines, no published or pretrained networks, no native binaries. Training data is unrestricted.
