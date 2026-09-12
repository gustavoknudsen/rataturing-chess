# UCI release

Turns the competition engine into a UCI executable that runs in Arena, Cute Chess, Banksia and any other standard GUI.

The engine is not modified. `rataturing_uci.py` imports it and drives it through `get_move`, the same entry point the Chessathon platform used, so a game played in a GUI runs the code that played the tournament.

`build.py` takes the engine source out of `submission.zip` rather than out of `src/`, then hashes every file it is about to ship against that archive and refuses to continue on any mismatch. That check is the reason for reading the zip at all, and `--cache` is the one flag that turns it off, because it rewrites the njit decorators.

## Startup takes about a minute

The engine is Python compiled by numba at import. That compile is the entire startup cost and it cannot be avoided, so the release is built around hiding it rather than removing it.

The engine answers `uci` in under a second, so a GUI registers and configures it with no wait. Compilation starts on a background thread at the same moment, and `isready` waits for it, which is what `isready` is for. The cost is paid once per process, and a GUI keeps one process for a whole session, so only the first game of a session waits.

Three ways to make it shorter were measured and rejected. A lower LLVM optimisation level changes nothing, because the time goes to numba's own type inference on a 2500 line function rather than to LLVM. A portable pre-built cache halves the engine's speed, because it requires a generic CPU target. Caching on disk works but saves only a quarter, since seventeen functions read large module-level arrays that numba bakes in as pointers and cannot serialise, and those seventeen are the expensive ones. `build.py --cache` enables it anyway if you want it.

## The two builds

Both contain identical code. They differ only in whether `net.npz` is present, because presence of that file is what the engine keys its evaluation on.

| build | evaluation | size | speed |
|---|---|---|---|
| NNUE | trained network | 101 MB | about 690,000 nodes per second |
| Classic | hand-crafted | 88 MB | about 170,000 nodes per second |

NNUE is the stronger build and the one that competed. Classic is slower despite doing less work per node, because the network accumulator updates incrementally through make and unmake while the hand-crafted evaluation recomputes pawn structure, king safety and mobility at every leaf. It is included because it plays noticeably differently and needs no network file.

Speeds are the best of three six second searches from one middlegame position on the development machine. Treat them as a reading of that machine on that day, not a property of the engine. Node counts at a fixed depth are the reproducible measurement.

## Building

Needs `submission.zip` in the repository root and PyInstaller.

    python tools/package.py
    pip install pyinstaller
    python uci/build.py

That writes both archives to `release/`. Each build ends by running its own executable through a UCI handshake and a real search, and fails if no legal move comes back.

    python uci/build.py --variant nnue     one build only
    python uci/build.py --cache            enable numba's on-disk cache
    python uci/build.py --skip-smoke       skip the post-build check

## Running from the repository

No build required. The adapter finds the engine in `../src`.

    .venv/Scripts/python.exe uci/rataturing_uci.py

## Options

| option | default | meaning |
|---|---|---|
| `Hash` | 256 | transposition table size in MB |
| `OwnBook` | true | play the opening book |
| `Move Overhead` | 50 | time reserved per move in ms |

`Move Overhead` was 420 in the qualification build and 100 in the finals build, after the platform's per-move charge was measured at 1 to 2 ms. The engine itself never overshoots its budget by more than 9 ms. A local GUI has no such charge, so the adapter defaults to 50.

The book covers the first 20 moves. It is suppressed for `go infinite` and `go depth`, because answering an analysis request from a book returns a move with no evaluation and no line.

## What is supported

`uci`, `isready`, `ucinewgame`, `position` from `startpos` or a FEN with a move list, `setoption`, `go` with `wtime`, `btime`, `winc`, `binc`, `movetime`, `depth` and `infinite`, `stop`, and `quit`.

After every completed depth the engine reports depth, score, nodes, nps, time and the principal variation. Mate is reported as `mate N`, not as a large centipawn number.

Not supported: pondering, `MultiPV`, and more than one thread. None are declared, so no GUI will ask for them.

## Why no engine changes were needed

Three properties of the search made the adapter purely additive.

The iterative deepening driver in `btc_search.search_position` is plain Python, so wrapping `_aspiration_search` is enough to report after every depth.

The search polls a stop flag, `state.sc[SC_STOP]`, at seven points. Writing it from another thread ends the search in about 20 ms, which is what makes `stop` and `go infinite` work.

Data files resolve relative to their own module rather than the working directory, so the engine folder can sit anywhere.

The engine ships as ordinary `.py` files in `engine/`, never bundled into the executable, because numba locates its cache through each function's source path and needs those to be real files.

Only the search thread touches numba arrays. The main thread reads stdin and formats strings, and its one write into engine memory is the stop flag, a plain store that takes no reference count. That is what keeps the non-atomic reference counting in `btc_nrt.py` correct here.

## Files

| file | what it is |
|---|---|
| `rataturing_uci.py` | the adapter |
| `build.py` | builds both releases from `submission.zip` |
| [`../tests/test_uci.py`](../tests/test_uci.py) | the protocol gate, 22 checks |

`test_uci.py` drives the adapter from source, or a built executable if given its path:

    python tests/test_uci.py
    python tests/test_uci.py release/staging/nnue/dist/Rataturing-NNUE/Rataturing-NNUE.exe

Build output goes to `release/`, which is not tracked.
