# Engine speed research, 2026-09-09

Status: research note from 9 September 2026, kept as written. Item 3 (pick-best-on-demand) was implemented on finals day behind BTC_PICK_MOVES; see docs/FINALS_DAY.md.

Where the runtime actually goes, what was tried, and what is left. Produced by a
dedicated research pass that built and measured every item against the real engine
rather than a synthetic harness. Item 1 is banked; items 2-5 are open.

## The methodological finding, which is worth more than any single item

**Every microbenchmark in this repo that passes `l1` / `qa` / `qb` / `scale` as runtime
arguments measures a different program than the engine runs.** numba freezes the
engine's module globals (`NET_L1`, `NET_QA`, ...) as compile-time constants and LLVM
then generates completely different code.

Two concrete consequences:

- The old cost table put NNUE at 65% of a node. Measured correctly it is ~22%, and after
  the item-1 fix, ~18%. The earlier hunt for a 2x was aimed at a quarter of the runtime.
- A float64 `propagate` that is genuinely 3.1x faster with runtime arguments (382 ns vs
  1182 ns at L1=512, and bit-exact) is a *regression* with the constants folded:
  **109 ns int64 vs 195 ns float64**. Built and measured end to end: `evaluate_cached`
  went 259 ns -> 305 ns.

Benchmark hot kernels by setting `BTC_NET` before importing `btc_eval` and driving them
from inside an njit loop. A Python-level `perf_counter` loop around an njit function adds
~11 us of dispatch per call, which swamps everything discussed here.
`scratchpad/width_real.py` does it correctly.

## 1. `_sort_moves` hoist - DONE, +70% NPS, byte-identical search

`_sort_moves` called `_score_move` per move, which called `_counter_move` and
`_cont_hist_score`. Each wraps a history-table read in `if ply >= 1 and played[ply] != 0`.
Every quantity in those guards is invariant across the whole move list.

Measured cost per quiet move, real tables, realistic `played`:

| construct | ns |
|---|---|
| killers only | 0.6 |
| + main_hist | 1.0 |
| + counter-move, **guarded** | **25.7** |
| + counter-move, unguarded | 0.7 |
| cont-hist, **guarded** (2 lookups) | **28.6** |
| cont-hist, unguarded | 1.5 |
| full `_score_move` replica | 103.7 |

`_sort_moves` cost 4,586 ns at a 37-move node - more than an entire average node.

Shipped result: searchbench depth 9 **241,794 -> 410,892 NPS**, nodes 355,858 unchanged,
depth and score identical on all 15 positions.

**The mechanism was not identified.** Not cache (`cont_hist` is 2.36 MB, int16, two
1.5 KB rows per node), not branch misprediction (the guards are constant across the
loop), not fixed by `NUMBA_SLP_VECTORIZE=1`, and not fixed by `inline='always'` (applied
to 20 hot helpers: no change). It did not reproduce in a synthetic helper with the same
argument count and branch shape. Treat it as an empirical numba codegen cliff.

**The rule: never put a conditional table lookup inside a helper called once per move.**
Hoist the guard and the row view out of the loop.

## 2. Audit the remaining move loops for the same pattern - OPEN

The per-move helpers `_capture_score` (unconditional read), `_see_prunable` and
`_skip_quiet` (scalar only) were checked and the cliff does not recur.
`_lmr_reduction` has the pattern but runs on too few moves to matter (measured, no gain).

Still unaudited for **general** loop-invariant hoisting, as opposed to the guarded-lookup
cliff specifically: the quiescence move loop (`btc_search.py:743-817`) and the main
negamax move loop. The rule to apply: anything inside `for i in range(cnt)` that does not
depend on `i` gets hoisted. Pass/fail is unambiguous - `searchbench.py 9` must print
355,858 nodes and a lower time.

## 3. Pick-best-on-demand instead of a full insertion sort - OPEN, ~5-10%

`_insertion_sort` on 37 fresh scores costs **420 ns per node**, now roughly 40% of what
is left of `_sort_moves`. At a cut node only 1-3 moves are ever examined. A selection
scan (argmax over the unexamined tail, swap into place) costs `cnt` comparisons per move
actually searched.

**Caveat: this changes tie-breaking order, so node counts move and it needs an A/B
match, not just a bench.** That makes it a much slower item to validate than 1 or 2.
Call sites: `btc_search.py:667` (definition), used at 746 and 753 and inside
`_sort_moves`.

## 4. A `piece_on_square[64]` mailbox beside the bitboards - OPEN, ~3-6%

`_captured_piece` (`btc_search.py:596`) costs **45 ns per capture** because it scans up
to 6 bitboard planes. It runs on every capture during ordering and again inside SEE
(`btc_search.py:611, 795, 957, 962`). A mailbox array maintained in `make_move` /
`unmake` turns it into one array read.

Risk: new state that must stay in sync with the bitboards, which is the classic
corruption bug. Validatable in about an hour - `run_tests.py` perft plus node identity
on searchbench.

## 5. Runtime selection of the propagate kernel at init - OPEN, insurance not speed

The dev laptop is `alderlake` (AVX2, no AVX-512); the platform is an EPYC 9V74, Zen 4.
Cross-compiling the current int64 propagate with `NUMBA_CPU_NAME=znver4` emits
**`vpmullq` on `zmm`** (AVX-512DQ, 8-wide int64 multiply, 20 instructions) where
alderlake emits **60 `vpmuludq` on `ymm`** (4-wide, emulated).

So propagate is *relatively cheaper on the platform than on the laptop*, and any laptop
A/B between integer and float formulations does not transfer. If that kernel is ever
changed, compile both variants at import (~2 s of the ~45 s headroom), time each on a
few thousand random accumulators (~1 ms), and set a module flag.

## Ruled out, with numbers

- **torch batched leaf evaluation.** Single `(1,1024)@(1024,1)`: **5.73 us**;
  `nn.Linear` batch 1: **6.81 us**; `torch.clamp` on 1024 elements: **2.61 us** - 12-30x
  a whole node. Batched `clamp+matmul` at batch 256 is 107 us = **416 ns per position**,
  already worse than the 109 ns propagate before gathering accumulators into a
  contiguous tensor, and before the fact that alpha-beta cannot produce 256 independent
  leaves without destroying the ordering and cutoffs that make it alpha-beta. Plus 2.43 s
  of import against the 90 s budget.
- **numpy from Python.** Full propagate with preallocated `out=` buffers: **5.94 us**;
  `np.dot` alone at n=1024 float32: 1.25 us; `np.clip` alone: 4.45 us. Leaving njit to
  reach it costs ~11 us of dispatch. Against 109 ns.
- **`np.dot` inside njit.** numba routes it through `scipy.linalg.cython_blas`; scipy is
  in neither `.venv` nor the platform image. `ImportError: scipy 0.16+ is required`.
- **int32 blocked partials.** 1687 / 1976 / 1798 ns (block 64 / 128 / 256) against
  977 ns for int64 at L1=256.
- **numba global flags.** `LOOP_VECTORIZE=1`, `ENABLE_AVX=True`, `OPT=3`,
  `LLVM_REFPRUNE_PASS=1` are already set. `NUMBA_SLP_VECTORIZE=1` changed nothing
  measurable and is off by default in numba for miscompilation reasons.
- **`inline='always'`.** Applied to 20 hot helpers: baseline, no change.
- **A fused accumulator update.** `dst[k] = src[k] + w_add[k] - w_sub[k]` in one pass
  measured 550 -> 408 ns synthetically but **266 -> 266 ns in the real engine** - with
  `l1` folded, LLVM already vectorises it (`vpaddw` / `vpsubw` on `ymm`, `zmm` on
  znver4).

## Where a 2x would have to come from

1.70x is banked. Items 2-4 are plausibly worth another 1.15-1.25x combined, which lands
at 1.9-2.1x. Whether that materialises depends entirely on whether the guarded-lookup
cliff exists in one or two more places, which is a two-hour audit with an unambiguous
pass/fail test, not a research project.
