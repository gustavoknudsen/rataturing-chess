# The London final: a 30 second init budget

The final was played on 12 September 2026 as a knockout among the top of
the qualification Swiss. At 10:30 the organisers announced the
surprise constraint: the init budget dropped from 90 s to 30 s. Builds could
be changed until 14:00 and then played the knockout as they stood.

Rataturing was seeded 5th and finished 5th to 8th, losing the quarter-final
1.5 to 2.5 to AlphaFish, who went on to win the event.

This document records what the constraint meant, how it was answered, and
what changed in the engine that day. Every number was measured, on the
platform where stated and otherwise on the development machine.

## What the constraint meant

The engine is Python compiled by numba at import, and the platform wipes its
scratch space between games, so every game recompiles the whole engine.
That compile took 38 to 40 s of CPU on the platform's core. A 30 s budget
could not be met by compiling faster: even the trims described below leave
the compile near 30 s on their hardware.

The starter harness, which mirrors the platform, settles three facts:

1. Init is wall time from process spawn to the runner's ready line, printed
   immediately after `import agent`.
2. The process is frozen with SIGSTOP the moment it is ready and after every
   move, and resumed only for its own `get_move`. Background work only
   progresses while the engine's own clock is running.
3. Move time charged to the clock is the wall time of `get_move`.

So the init window is the only free CPU in the game. Every second of compile
that lands inside it costs nothing; every second after it comes off the
120 s clock.

## The answer: a staged wrapper

`staged/agent_staged.py` ships as `agent.py`, with the real engine renamed to
`agent_real.py` beside it and a python-chess fallback as `agent_pure.py`.

- The real engine's import starts on a thread at module top.
- Import blocks until the engine is ready or 26 s have passed, measured from
  the real process start where the platform can report it. The wait releases
  the GIL, so the compile thread owns the core for the whole window.
- If the compile is not finished, move one donates one large slice of clock
  to it, up to 30 s. If the position is in the opening book the book answers,
  so that move costs clock but not quality. Out of book, the move waits for
  the real engine and lets it search with its normal budget; the fallback
  only plays if the compile is still not done after the slice.
- From then on the real engine plays unchanged.

Measured on the platform: ready at 26.4 s of 30, with a 0.4 s overhead
between process start and the module's first line; compile finished 12 to
16 s into move one. The whole cost of the constraint was one donated slice
per game and, on a few occasions, a fallback move.

`staged/build.py` builds the layout from `submission.zip` and refuses to
continue unless every engine file hashes equal to the archive.
`tests/test_staged.py` checks the shipped configuration, a forced short
deadline, the out-of-book wait, and the fallback path.

## Things that did not work, so nobody retries them

- A shipped numba cache. The search function reads large module-level arrays
  that numba bakes in as pointers and cannot serialise, so the expensive
  function recompiles regardless, and the read-only filesystem plus wiped
  scratch defeat the loader anyway.
- A lower LLVM optimisation level. Compile time is unchanged within noise,
  because the cost is numba's own type inference and lowering on a 2500 line
  function, and nps drops 12 percent.
- Disabling features to shrink the compile. `BTC_MINIMAL` saves 28 percent
  and costs strength for the whole game.

## What changed in the engine that day

Every change was gated on the fixed-depth node fingerprint
(`tools/searchbench.py 9`), the full test suite, and a match against the
qualification build.

**Compile trims, identical machine code, 25 percent off the compile.**
numba's Python-callable wrappers are no longer generated for the 170 jitted
functions only ever called from other jitted code; eight functions that
were being compiled two to four times because call sites passed different
integer types now compile once; and the magic slider tables ship as
`attack_tables.npz` instead of being compiled and built at import.

**Deferred move sort.** The transposition-table move is searched before the
remaining moves are sorted, and the sort only runs if it fails to cut off.
This cannot be fingerprint-identical: the sort reads the history tables, and
the first child's subtree writes them, so the remaining moves are ordered
against fresher history. Measured at depth 12: 13 percent fewer nodes at
higher speed. Deferring move generation as well hit a numba codegen cliff
and cost 47 percent, so generation stays eager.

**Zobrist keys.** The C engine generated its 64-bit keys from four draws of
a 32-bit xorshift, a linear map of its state. The 849 keys had rank 32 over
GF(2), so every position hash carried 32 bits and distinct positions
collided at the 32-bit birthday rate. The keys now come from splitmix64 and
have full rank. Found while implementing the static-evaluation cache below,
whose node counts moved only through those collisions.

**Time management.** The platform charges 1 to 2 ms per move, measured from
the clock lines of every match log, while the engine reserved 420 ms per
move and was finishing lost games with a minute unused. The reserve is now
100 ms and the moves-to-go divisor 28 instead of 40, chosen from a sweep of
the engine's own budget function over 60, 100 and 150 move games at the
observed spend: never flags, 30 percent more thinking in a normal game, and
a 150-move game still ends with clock in hand. Eight games at the real
control confirmed no time loss.

**Opening book.** A third book, built from the positions the tournament had
actually used, sits first in the loader; the qualification books remain as
gap fill. It answered every known finals start.

Match evidence for the batch: 55.4 percent over 84 games against the
qualification build at 10 s + 0.04 s with identical time settings on both
arms, no crashes, no time losses.

## Verified but not shipped

Four speed changes were implemented and proved tree-identical by fingerprint
on the day but did not ship. Three live in `src/btc_search.py` behind flags
that default off, with the shipped path unchanged when they are off:

| flag | change | why it stayed off |
|---|---|---|
| `BTC_TT_EVAL` | raw static evaluation cached in the transposition table key's low 16 bits, so a known position skips the network forward pass | with the lazy accumulator and the new book, 47.8 percent +- 6.9 over 200 games: not distinguishable from zero, not enough time for a longer match |
| `BTC_CHILD_ACC` | the NNUE accumulator row is built in the child after its table probe, so cutoff children never pay a delta | same match |
| `BTC_PICK_MOVES` | pick-best-and-shift per move tried instead of a full insertion sort, reproducing the stable order exactly | finished in the last twenty minutes; nps unmeasurable on a loaded machine |

The fourth, the child inheriting the parent's in-check result, was kept out
of the source. It needs an extra argument on the search function, and numba
marshals that argument on every call whether or not the feature is on, which
measured 3 percent slower with the flag off. It is recorded here rather than
carried as code.

Measured after the event, pinned to one core, best of four at depth 12:
with every flag on the tree is identical and the speed is 0.97x of the
flags-off engine. No speed gain is demonstrated on this machine, which is
consistent with the match result. They stay off until a longer match says
otherwise.

Two ordering changes were tried and rejected on fixed-depth node counts: a
steeper history malus (fewer nodes at depth 9, more at depth 11) and a
4-back continuation history plane kept out of the pruning paths (13 percent
more nodes).

## Timeline

| time | event |
|---|---|
| 10:30 | constraint announced |
| 11:14 | staged build validated, 24.5 s of 30; first calibration numbers from the platform |
| 11:40 | rebuilt with the out-of-book wait; wait raised to 26 s after two practice games measured the overhead at 0.4 s |
| 13:03 | compile trims, deferred sort and Zobrist fix integrated; match against the qualification build |
| 13:17 | uploaded with the time-management values |
| 13:55 | final upload: the same engine with the new book |
| 14:00 | uploads closed |
