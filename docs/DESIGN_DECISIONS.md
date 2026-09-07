# Design Decisions

Living document. Each entry: options → evidence → decision → reasoning. Updated as we learn.
Measurements from 2026-09-07 on the dev machine (Windows, Python 3.12 venv with the
platform-pinned versions: numba 0.67.0, numpy 2.5.2, python-chess 1.11.2). The match
machine (1× EPYC 9V74 2.60 GHz) is slower than a desktop core — treat local numbers as
optimistic by ~1.5–2×; recalibrate from the first upload's validation log.

## 0. Numba 0.67 ground rules (measured, they shape everything below)

Probe scripts: scratchpad `exp2_numba_recursion.py`, `exp3_numba_state.py`,
`exp4_compile_budget.py`.

| Question | Result |
|---|---|
| Self-recursive `@njit` function | ✅ compiles, nopython (toy negamax 0.77 s cold compile) |
| **Mutual recursion** (f↔g) | ❌ `TypingError: cannot type infer runaway recursion` — **confirmed broken** |
| One-way call DAG (negamax→qsearch, each self-recursive) | BTC needs exactly this, never mutual — verify on the real pair in phase 1, day 1 |
| **Mutating a module-global numpy array inside njit** | ❌ **refused**: `Cannot modify readonly array` — numba freezes globals as read-only constants |
| Reading a module-global numpy array inside njit | ✅ frozen by reference, fine for constant tables (attack tables, PST, masks) |
| Mutating an array passed as an argument | ✅ zero-copy reference semantics, visible outside and across nested njit calls |
| uint64 ⊕ int64 / uint64 ⊕ literal arithmetic | stayed exact in all probes (`(1<<63)|1` round-trips `+1`, shifts, masks) — numba 0.67/numpy 2.5 NEP-50 rules. **Discipline stays**: bitboards uint64, indices int64, explicit `uint64()` casts at every mixing point, because one silent float64 promotion anywhere = wrong perft. Perft is the guard. |
| **`int(x)` on uint64 is NOT a cast** | measured in phase 2: numba types `int(uint64_expr)` as **uint64**, and a later branch/ternary unification of int64 with uint64 silently promotes to **float64** (surfaced as a bizarre TypingError deep in the TT pack). The promotion trap is real but strikes via type *unification*, not arithmetic. Rule: cast with `numba.int64(...)` explicitly wherever a uint64-derived value flows into signed integer context. |
| Recursive fn + typing failure diagnosis | when a big recursive njit function fails to type with nonsense float64 errors, bisect bottom-up: compile leaf helpers standalone with known int args; the poison is usually a helper, not the recursion. Eager signatures do not fix a helper's internal mistyping. |
| `import numba` cost | 0.25 s (+0.10 s numpy) |
| Eager signature compile (small fn) | 0.03 s |

**Architectural consequence** (this corrects an assumption in the project brief): hot-path
*mutable* state cannot live in module globals. Pattern:

- **Constant tables** (attack tables, magics, masks, PST, Zobrist keys): module-level numpy
  arrays, *read* from njit — frozen by reference, free.
- **Mutable state** (bitboards, occupancies, scalars, move stacks, TT, histories,
  repetition list): module-level numpy arrays created once in Python, **passed as
  arguments** into the jitted entry points and threaded through inner calls. Zero-copy;
  the signatures get long — group into a handful of slabs (e.g. one uint64 board-state
  vector, one int64 scalar vector, per-ply 2D slabs) to keep them manageable.

## 1. Board representation: own bitboards vs python-chess

- **Options**: (a) python-chess `Board` in search; (b) own magic-bitboard engine in
  numba-jitted uint64 code, python-chess only at the boundary.
- **Evidence** (`exp1_pychess_speed.py`): python-chess perft(4) startpos = **245 k NPS**
  (includes push/pop); legal movegen alone = 40 µs/position (25 k/s) on Kiwipete. A real
  search does movegen + make + eval per node → order 20–50 k NPS in search shape, less on
  the EPYC. BTC's C incarnation runs tens of millions of NPS; a numba toy self-recursive
  alpha-beta loop over uint64 tables ran **157 M simple-ops/s** (`exp4`) — a real
  movegen/make/eval node costs hundreds of ops, so a realistic target is **1–5 M NPS**,
  i.e. **~50–100× python-chess**. That is roughly 6–7 doublings of the node budget ≈
  +400–600 Elo of search depth. Not close.
- **Decision**: **own bitboards** (option b). python-chess never appears in the search.
- Status: DECIDED (pending only the perft proof that the port is correct).

## 2. Where python-chess still earns its place

- **Decision**: exactly three jobs, all off the hot path:
  1. **Test-time cross-validation**: perft differentials and FEN round-trips against our
     board (dev only, never ships in the hot path).
  2. **Boundary safety net in `get_move`**: after search returns, verify the UCI string is
     in `chess.Board(fen).legal_moves`; on any failure (exception, garbage move, timeout
     fallback) return a legal move from python-chess instead. 40 µs/parse + ~40 µs legality
     is noise once per move, and it converts "illegal move = lost game" into "one bad move".
  3. **Game-state reconstruction** (opponent move diffing, §8): convenience code between
     moves, not inside search.
  We parse FEN with our own parser (we need our internal representation anyway); python-chess
  parses the same FEN in parallel only for the safety net.
- Status: DECIDED.

## 3. Magic tables: generate at import vs ship as data

- **Options**: (a) ship `.npy` tables in the zip (~2.25 MB of slider tables); (b) generate
  at import from hardcoded magic *numbers*; (c) search for magics at import.
- **Evidence**: BTC hardcodes all 128 magic numbers in `attacks.cpp` — the expensive search
  (magic.cpp) is offline tooling even in C. Table *filling* given magics is
  64×(4096+512) ≈ 295 k entries of cheap mask/multiply work — sub-second vectorised numpy
  or one small njit builder (~0.5 s compile). 50 MB cap is not under pressure either way
  (we ship no net), so this is about init time vs zip hygiene, and both fit trivially.
- **Decision**: **(b) generate at import**, magic numbers as Python constants copied
  verbatim from attacks.cpp. Judge-readable (it's obviously source, not a binary blob),
  zero zip weight, and the generator doubles as documentation. If import-time measurement
  ever shows the builder mattering (it shouldn't), flip to shipping `.npy` — decision is
  reversible in 20 minutes.
- Status: DECIDED.

## 4. Make/unmake vs copy-make under numba

- **Options**: (a) make/unmake with incremental undo info; (b) copy-make into per-ply
  preallocated slabs.
- **Evidence**: BTC itself is **copy-make** (copy_make.h) — the C macros snapshot ~30
  globals to stack locals. Post-audit, the snapshot we actually need is small: 12 piece
  bitboards + 3 occupancies + hashKey (16×u64) and 4 scalars (side, ep, castle, fifty) —
  ~132 bytes/ply, copied into `undo_stack[ply]` slices of preallocated 2D arrays. At 64
  max ply that is 8 KB resident, L1-hot. numba compiles small fixed-count copy loops to
  memcpy-grade code. Make/unmake would save ~half the copy but add undo-info bookkeeping
  and diverge from the C code we are trying to port faithfully under time pressure.
- **Decision**: **copy-make**, per-ply slabs, mirroring BTC's semantics exactly (minus the
  eval-accumulator snapshotting, which we drop — see ENGINE_AUDIT make_move.h). Revisit
  only if profiling shows the copy dominating, which at 132 B/node it will not.
- Status: DECIDED.

## 5. Move encoding: packed int32 vs struct-of-arrays

- **Options**: (a) BTC's 24-bit packed int in int32; (b) parallel arrays
  (from[], to[], piece[], …).
- **Evidence**: BTC's entire search/ordering/history code manipulates packed ints
  (`getSource(move)` etc.). Shifts+masks are single-cycle in jitted code; SoA would touch
  more cache lines per move during sort (insertion sort swaps 6 arrays instead of 1+score)
  and makes every ported line read differently from the C original.
- **Decision**: **packed int32, identical bit layout to move.h**. Move lists:
  preallocated `int32[MAX_PLY+1][256]` + per-ply counts; score array alongside for sorting.
- Status: DECIDED.

## 6. Forcing compilation at import: eager signatures vs warm-up search

- **Options**: (a) eager `@njit(signature)` at decoration; (b) lazy `@njit` + warm-up
  calls at import with real argument types; (c) both.
- **Evidence**: eager compiles at decoration (0.03 s small fn) but a signature mismatch
  (e.g. C-contiguity, int width) silently triggers a *second* lazy compile on first real
  call — on the match clock. Warm-up with the real arrays guarantees the compiled
  signature is the one used. Official baseline (`baselines/numba/agent.py`) and starter
  docs use warm-up. numba njit raises on typing failure (no silent object-mode fallback);
  a failed compile at import = failed validation upload, loudly, which is what we want.
- **Decision**: **warm-up calls at import** — a fixed short search (depth ~3–4) from the
  start position through the full entry point, exercising every jitted function with the
  exact production array objects. Plus an import-time assertion that
  `NUMBA_DISABLE_JIT` is not set and each dispatcher has exactly the expected signature
  count. Eager signatures only if we later find a function the warm-up can't reach
  (there should be none — unreached code is dead code).
- **Compile budget tracking**: `bench.py` reports cold-import wall time from a fresh
  process from day one; ceiling 65 s (platform slower + 90 s hard limit). Current
  fixed costs: numba+numpy import 0.35 s, toy search compile 0.77 s. The risk item is
  the merged negamax (one huge function compiles superlinearly) — measured every commit.
- Status: DECIDED (mechanism); budget tracked continuously.

## 7. TT sizing against 2 GB, no torch

- **Evidence**: we never import torch/onnxruntime (saves import time and its allocator
  footprint). Baseline process (python + numpy + numba + our tables) is a few hundred MB;
  attack tables 2.25 MB; histories (with tier-4 dropped) ~9.5 MB int16
  (continuationHistory 12·64·12·64×2 tables ×2 B = 9.4 MB — port as int16 numpy);
  per-ply slabs < 1 MB. The TT is the only big consumer. BTC entry = key u64 + packed
  data u64 = 16 B (repacked, ENGINE_AUDIT tt.h). OOM = lost game, and 2 GB is a container
  cap wherever the allocator spikes, so leave a wide margin; TT hit rates past a few
  hundred MB flatten at blitz node counts anyway.
- **Decision**: **2^24 entries × 16 B = 256 MB** (4-entry buckets ⇒ 2^22 buckets), as two
  parallel numpy arrays allocated once at import. Total steady-state target < 700 MB.
  Measure RSS in bench.py; bump to 2^25 (512 MB) only if measurements show > 1 GB of
  headroom on-platform (validation log / local docker proxy).
- Status: DECIDED (size revisitable on evidence).

## 8. Game-state tracking across calls (repetition/fifty awareness)

Not in the original question list but design-relevant (brief §"Game state tracking").

- **Facts**: fresh process per game; module state survives between our own moves;
  referee auto-claims threefold/fifty (checked *before* asking for a move) counting from
  the game's first FEN; we only see positions on our own turn.
- **Design**: module-level (plain Python) game record. On each `get_move(fen)`:
  1. If no stored state or `fen` is not reachable from our predicted position →
     (re)initialise history from `fen` alone (its halfmove clock seeds the fifty count;
     prior repetitions unknowable — but from game start we see every position, and
     mid-game resets only happen if our own tracking broke).
  2. Else: opponent's move = the unique legal move from our post-move position whose
     resulting position matches the parsed `fen` (compare bitboards + side + castle + EP,
     not raw strings). Append its Zobrist key.
  3. Our own move: appended after search returns.
  The key list feeds the in-search repetition table (search sees game history + search
  path, exactly like BTC's `repetitionTable`/`repetitionIndex`), so the engine scores
  walking into a referee-claimed threefold as 0.00 instead of blundering a won game into
  a shuffle draw.
- Status: DECIDED.

## Open questions (to resolve with measurements in later phases)

- **LSB extraction**: de Bruijn multiply vs `bb & -bb` tricks inside njit — microbench in
  phase 1 (cheapest correct one wins; it's the hottest primitive in the engine).
- **Merged vs split negamax/quiescence**: split (BTC shape) is expected to work; if the
  one-way DAG surprises us, fall back to a merged single function with a `depth<=0 ⇒
  qsearch` mode. Decided by the day-1 recursion proof on the real functions.
- **Eval accumulator strategy** (phase 3): compute-in-evaluate (decided in audit) —
  confirm cost vs BTC's compute-in-make with a node-profile once eval exists.
- **Opening book**: rated games start from curated positions ⇒ move-1 books are useless;
  a small own-games book is a phase-5 luxury, likely DROP.
- **Syzygy 3–4 man** via `chess.syzygy` at the boundary (root probe only, off hot path):
  cheap Elo in pawn races; phase-5 if time allows.
