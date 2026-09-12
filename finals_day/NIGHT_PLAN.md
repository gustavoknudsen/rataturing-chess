# Night plan, 2026-09-11 into finals day

Live document. Updated as results land. The plan may change; the rules at the
bottom may not.

## The hard constraint

Training and match play cannot overlap. One training job puts system commit at
87 percent, and six match workers once drove it to 96 percent and nearly froze
the machine. Every slot tonight is either training or matches, never both.

No opening book workers tonight, by instruction, to protect memory.

## Schedule

    23:43  run 1 (16 x 256) ends, banked at finals_day/nets/b16x256.npz
           32 x 384 starts automatically
    04:00  32 x 384 done, then gate and SPRT automatically
    05:15  branch: base-engine SPRTs with whatever time is left
    08:00  buffer, freeze, package, re-verify

Revised at 00:11: the 32 x 384 run is streaming at 0.200 Mpos/s against the
0.28 the 16 x 256 run managed, because engine benchmarks were competing with
its data workers for CPU. Roughly 3h50 rather than 2h45. Do not run
searchbench or the test suites while training is live unless the result is
needed; the training run is on the critical path and they are not.

A second trained net is unlikely to fit before the deadline. That is the right
sacrifice: one net that might beat the shipped one is worth more than two that
only insure against a size cap that may not be announced. fingerprint and init

## ON THE DAY

Announcement 10:30. Practice every ten minutes 10:40 to 13:50, counting for
nothing. Uploads close 14:00 and the build standing then plays the knockout.

**Before 10:30.** Upload the current build unchanged, first thing, before
touching anything. Record the fingerprint from `tools/searchbench.py 9`; it
must read 259210.

**First 30 minutes after the announcement: read, decide, measure. Do not
type.** Then one change per upload, and keep the last two uploads in reserve.

### If the time control changes

1. **Set `INCREMENT_MS` in `src/agent.py` to the new increment in
   milliseconds.** It is a hardcoded 500 and nothing derives it. Getting this
   wrong mis-budgets every move of every game, silently. This is action one and
   nothing else matters until it is done.
2. **Upload that alone and run a practice round.** Do not change MTG or
   OVERHEAD yet. The engine played six games at 10 + 0.1 locally without
   losing on time, so the simulated warnings are probably too harsh, and one
   practice round replaces all of that guesswork.
3. **Read `implied overhead` from the practice log.** This is the number
   everything else depends on and it has never been measured on the platform.
4. Only now run `finals_day/tc_tune.py`, with `REAL_OVERHEAD_MS` set to what
   you measured. If the row for the new control still says FLAGS, apply the
   panel value it names and upload that as a second change.

The ordering is deliberate. Steps 2 and 3 cost one upload and ten minutes, and
they replace a pessimistic simulation with a measurement. Applying MTG and
OVERHEAD changes first would mean tuning against a number that may be wrong by
a factor of three.

### If the init budget changes

- 40 s or more: no action.
- Under 40 s: `finals_day/stage.py`, then `finals_day/test_staged.py` and
  expect 7 of 7. Upload the staged build.
- Do not reach for `BTC_MINIMAL` or `BTC_ENDGAMES`. Measured tonight:
  `BTC_ENDGAMES=0` saves nothing at all, and `BTC_MINIMAL=1` saves 28 percent
  while costing strength for the whole game rather than the first few moves.

### If memory drops

- 1 GB or more: no action, we use 682 MB.
- 512 MB: `BTC_TT_ENTRIES` 2097152 in PANEL, which measures 459 MB.
- 256 MB: not survivable, the floor is 432 MB of numba runtime. Ship
  `finals_day/agent_pure.py` and accept about 20k nps.

### If cores increase

- Set **both** `BTC_THREADS=N` and `BTC_NRT=0` in PANEL. `BTC_THREADS` alone
  is silently refused, and threads without `BTC_NRT=0` would be memory
  corruption, which is why they are refused.
- Confirm stdout says `init: N search threads`. If it says `refusing threads`,
  `BTC_NRT` was not set.
- Run `tests/test_parallel.py` first, expect 3 of 3.
- This gives back the 32 percent `BTC_NRT` is worth, so two threads may be a
  net loss. Prove it in practice rounds before committing an upload to it.

### If the starting positions change

- Standard start: `rataturing.bin` becomes dead weight but stays harmless, and
  `rataturing_hedge.bin` at 552,078 entries carries the opening.
- Unusual or irregular positions: the network is the exposed component, see
  section 3b. Test `BTC_NNUE=0` rather than assuming the network is better.
- Chess960: `tests/test_frc_safety.py` already passes 28 checks. We play
  legally and never castle.

### If the ply cap changes

No action. Games end well before 200 plies, so a lower cap rarely binds. There
is no contempt term and adding one the morning of the final is not worth the
risk.

### If the 32 x 384 net wins its SPRT and you want to ship it

1. Copy `finals_day/nets/b32x384.npz` over `src/net.npz`.
2. Set `BTC_NET_UNITS` in PANEL to the figure the gate printed. This is not
   optional and it is not cosmetic: `NET_UNITS` rescales raw network output
   into centipawns and feeds every pruning margin, so a wrong value loads,
   compiles, plays legal moves, and gets every margin wrong.
3. **Record a new fingerprint.** `searchbench.py 9` reads 259210 for the
   shipped network only. A different network evaluates differently, so the
   node count changes and 259210 stops being the right answer. Run it, write
   the new number down, and use that as the gate from then on. Do not treat a
   changed fingerprint here as a regression.
4. Repackage with `tools/package.py` and re-measure init from the zip: an
   18 MB network against the shipped 24 MB should not slow init, but measure
   rather than assume.
5. Check the zip is still under the size cap. The staged build was already at
   40.01 MB of 50 MB, so a larger network plus staging is the combination to
   watch.

### Before every upload, without exception

1. `tools/package.py`, which extracts the zip and plays 12 plies from it.
2. `tools/searchbench.py 9`, and record the node count beside the upload.
3. Re-measure init from the extracted zip. Refuse anything over 65 s.

## Training

Both runs use 2.8B rows, which is 87.5M rows per bucket, the same density the
shipped net was trained at. Width is the only variable.

| run | config | size | rows | status |
|---|---|---|---|---|
| 1 | 16 x 256 | 6.0 MB | 1.4B | **done**, banked, insurance only |
| 2 | 32 x 384 | 18.0 MB | 2.8B | **done 03:15**, gated, in SPRT |
| 3 | 32 x 256 | 12.0 MB | 2.8B | not run, no time |

Run 2 finished cleanly: the full cosine schedule completed, 2,800,009,216
positions, train loss 0.016616 and validation 0.016763, and the quantised
worst-case accumulator is 16513 against the int16 limit of 32767, so there is
no saturation risk.

Gate 1 passed at `BTC_NET_UNITS 90`. But the static scores put it below the
shipped network rather than above:

| net | r2_q | known |
|---|---|---|
| shipped 32 x 512 | 0.663 | champion |
| net512, 1 bucket | 0.659 | -123 elo in games |
| **32 x 384** | **0.6338** | under test |
| 16 x 256 | 0.6319 | insurance |

### Verdict: 32 x 384 does not beat the shipped net

    b32x384 at NET_UNITS 90 vs shipped: +50 =129 -67
    score 46.5% +- 6.2%   elo -24 [-68, +19]
    LLR -2.94 at game 246, lower boundary

The hypothesis is refuted. Narrowing from L1 512 to 384 costs about 24 elo
rather than gaining. The interval still touches zero so it is not proven
worse, but it is certainly not better, and the static gate pointed the same
way. **Keep the shipped network.** The reasoning that motivated the run, that
L1 1024 gave better validation and 49.7 percent in games so 512 must be at or
past the optimum, was sound; it simply turns out 512 is close to the peak
rather than past it.

### The result that matters anyway: the size cap is now cheap

This was the scenario rated most likely, and it moves from unsolved to cheap.

Every rung is now measured in games. Nothing here is inferred.

| net | file size | zip unpacked | cost vs shipped |
|---|---|---|---|
| shipped 32 x 512 | 24.0 MB | 40 MB | baseline |
| **32 x 384** | **18.0 MB** | ~34 MB | **-24 elo [-68, +19]** |
| **16 x 256** | **6.0 MB** | ~22 MB | **-64 elo [-140, +6]** |
| net512, 1 bucket | 0.75 MB | ~17 MB | **-123 elo**, do not ship |

Note the 6 MB net cost twice what its gate score predicted: it gates within
0.002 r2_q of the 18 MB net and then loses 40 elo more. That is the third time
tonight the static gate has failed to rank two networks correctly, and it is
the reason every rung above was played rather than inferred. Use the gate to
get `NET_UNITS` and to reject a broken net; do not use it to choose between
working ones.

`NET_UNITS` for each, from its own gate run: 32 x 384 is **90**, 16 x 256 is
**93**. Both must be set in PANEL alongside `BTC_NET`.

Run 2 is not a contingency net. Doubling width to 1024 measured 4.2 percent
better validation and 49.7 percent in games, so L1 512 is at or past the
optimum and a narrower net may be stronger as well as smaller.

Run 3 is skipped if run 2 clearly loses to the shipped net. If 384 is worse
than 512, 256 is worse still, and the slot is worth more spent on search.

No small nets tonight. If the size cap drops, one is trained on the day.

## Base engine

Most of the research backlog turns out to be **already implemented**. Checked
against the code tonight, not against the documents: quiescence TT probe,
delta pruning, quiescence check evasions, cut node awareness, double
extensions, multicut, negative extension refinement and singular extensions
are all in. `SEARCH_RESEARCH.md` tier 3 is described as blocked behind
singular extensions being off; singular extensions are on and have been since
the `singular extensions + kpk` commit. Treat that document as history.

### SPRT results, all against the shipped build at 10 s + 0.04 s

| change | games | score | elo | verdict |
|---|---|---|---|---|
| 32 x 384 net | 246 | 46.5% | **-24** [-68, +19] | reject |
| `BTC_CAP_SEE_DYN` | 269 | 46.7% | **-23** [-65, +18] | reject |
| `BTC_SOFT_CONT` | 364 | 48.2% | **-12** [-48, +23] | reject |
| `BTC_NODE_EFFORT` v2 | 274 | 47.3% | **-19** [-61, +22] | reject |
| 16 x 256 net, 6 MB | 93 | 40.9% | **-64** [-140, +6] | reject |

Every one of these was a genuinely open question, and every one is now closed.
`BTC_SOFT_CONT` in particular had been open since the openings cache fault
invalidated both of its earlier matches; it is not an improvement, it stays
off, and nobody needs to reopen it tomorrow.

All three rejected changes are OFF by default, so the shipped build already
embodies every one of these verdicts. Nothing needs reverting.

### The pattern worth keeping: published refinements do not transfer

Three tree-expanding refinements were tested against the shipped build tonight
and all three came out negative or worse-ordered:

| change | published | measured here |
|---|---|---|
| `BTC_CAP_SEE_DYN` | **+13.97 +-6.76** | **-23 elo [-65, +18]**, SPRT reject |
| killer clearing, ply+2 | +5.77 +-3.92 | +11.5 percent nodes at fixed depth |
| killer clearing, ply+1 | +5.77 +-3.92 | +21 percent nodes at fixed depth |

The common thread is depth. Every one of these widens the tree in exchange for
accuracy, and our search runs four to five plies shallower than the engines
those figures were measured in, so the depth-gated heuristics already cover a
much larger fraction of our tree and the extra nodes are not repaid. The
aspiration window rejection recorded in `SEARCH_RESEARCH.md` failed for the
same reason and said so.

**Conclusion: importing published search refinements is not a reliable source
of elo for this engine.** It is at a local optimum for the depth it reaches.
The remaining backlog items should be assumed neutral-to-negative until
measured, not assumed positive because a figure is attached to them. What has
actually paid this project has been fixing bugs and fixing measurement, both
of which happened again tonight.

Time management is a separate category and is not subject to this argument: it
changes how long the search runs rather than the shape of the tree, which is
why `BTC_SOFT_CONT` is still worth a match.

### Screened and rejected tonight

**Killer clearing on node entry** (published 5.77 +-3.92), behind
`BTC_KILLER_CLEAR`, default off. Clearing ply+1 costs 21 percent more nodes at
fixed depth; ply+2, which is the correct ply since it shares the side to move,
still costs 11.5 percent. This is a pure ordering change, so a bigger tree is
simply a worse one and no match is needed to see it. The flag is left in place
at ply+2 in case a match ever contradicts the node count.

**Lazy evaluation margins.** Not implemented and not attempted. Our own
research records that this class of change "passes short controls and fails
long ones", and 120 + 0.5 is a long control.

## Conditions

### 1. Time control

Largely closed. `budget()` used to return 0 ms soft and the 10 ms hard floor at
any reduced control while seconds remained on the clock, because the reserve
term is a constant 17,640 ms and the floor under it was 1. Fixed with a floor
proportional to the clock, proved inert at increment 500 over 80,872 cells.

### Played, not simulated: the engine survives reduced controls

Before acting on any table below, know that they are pessimistic. Measured with
`arena_ab.py`, which enforces clocks and reports FLAG:

| control | games | time losses |
|---|---|---|
| 30 + 0.5 | 2, both reaching drawn endgames | **none** |
| 10 + 0.1 | 6, including a fifty-move game | **none** |

`tc_tune.py` says "nothing survives" at 10 + 0.1. The engine played six games
there without losing on time. Two reasons the simulation is too harsh: it
assumes the engine spends its whole soft budget every move, which it does not,
and it charges 420 ms of platform overhead per move, which **does not exist in
our local harness**.

That second point is the limit of this evidence and it matters. Locally the
clock only drains by our own thinking time, so these games test that the budget
function is sane, and they cannot test whether the platform's per-move charge
makes a short control unplayable. That question is answered by the `implied
overhead` line in a practice round, not here.

So: treat the flagging predictions as an upper bound on risk. Erring
pessimistic is the safe direction, because acting on a false alarm costs one
panel value and ignoring a real one costs the game. But do not panic-apply
changes at 10:31 on the strength of a simulation when a practice round will
tell you the truth in ten minutes.

### The simulated warning, kept as the upper bound

The shipped MTG 40 / OVERHEAD 420 pair is predicted to **lose on time** at
several ordinary controls, assuming the full 420 ms charge and full soft spend.
Run `finals_day/tc_tune.py`.

| control | shipped | smallest safe change |
|---|---|---|
| 120 + 0.5, 120 + 0.1, 60 + 0.5, 300 + 0, 600 + 0 | survives | none |
| 60 + 0.2 | flags | `BTC_MTG=48` |
| 30 + 0.5 | flags | `BTC_MOVE_OVERHEAD=500` |
| 180 + 2 | flags | `BTC_MOVE_OVERHEAD=500` |
| 30 + 0.2 and shorter | flags | nothing survives |

Both fixes are single panel values. The tool deliberately reports the smallest
change rather than the best one: whether a setting runs out of clock is exact,
but ranking two settings that both survive is not something a simulation can
do, and two different scoring objectives recommended opposite ends of the grid
before this was narrowed.

**The real overhead is now measured automatically.** The bottom rows assume
420 ms, which was inferred from a rated floor and never timed. If it is really
150 ms, those controls are all playable, so the figure decides whether a short
control is survivable at all.

`agent.get_move` now prints a line per move:

    clock: drain 73 ours 392 implied overhead 181

Between two calls the clock moves by our own wall time plus the platform's
overhead minus the increment, and we time our own half, so the overhead falls
out. Verified against an injected 180 ms: recovered 180 to 181 on every move.
Book moves give the cleanest readings, because our own cost is near zero and
the difference is almost entirely overhead.

So in the first practice round, read the implied overhead straight out of the
log. If it is not near 420, re-run `tc_tune.py` with `REAL_OVERHEAD_MS` set to
the measured value before changing any panel setting.

### 2. Init budget

Decision tree, by announced budget. Every number below was measured tonight,
except where marked.

| budget | answer | init | strength cost |
|---|---|---|---|
| 40 s or more | ship unchanged | 31 s | none |
| under 40 s | `agent_staged.py` | 0.10 s | close to none, see below |

Two rows, not three, because the middle options were measured tonight and do
not work. `BTC_ENDGAMES=0` saves **nothing**: 39.1 s against a 39.3 s baseline
in the same window, where the -17 percent in the old notes would have been
6.5 s. `BTC_MINIMAL=1` saves 28 percent, not 42, and it costs strength for the
whole game, so staging beats it at every budget.

**Staging now plays the real book.** `btc_book` is pure python-chess:
`load()` opens polyglot readers and `probe()` takes a board, so it needs no
numba and is available at 0.10 s. The staged agent consults it before the
fallback, which means the moves it plays while compiling are the moves the
real engine would have played, not 20k nps guesses. It then donates the saved
clock to the compile thread, which costs nothing because the move was already
decided.

Measured after the change: import 0.117 s, handover after **11 plies at
40.6 s**, first move returned in 4.002 s which is the donate interval exactly,
confirming the book answered instantly. The book covers moves up to 20 and
handover happens at move 6, so every pre-handover move is inside book range.
Staging's cost is now bounded by how often the opponent leaves book early, not
by the compile time.

Known bounded race: `agent_real`'s import calls `btc_book.load()`, which
clears the reader list, while the main thread may be inside `probe()`. The
probe is wrapped, so the worst case is a single move falling through to the
fallback. Not worth a lock.

Two candidate rescues were tested tonight and both failed. Recording them so
they are not retried at 10:30 under time pressure.

**A shipped numba cache does not work.** `negamax` is uncacheable: numba
reports "Cannot cache compiled function negamax as it uses dynamic globals
(such as ctypes pointers and large global arrays)", which is exactly what the
attack tables, zobrist keys and network weights are. With `cache=True` on all
197 decorators and `NUMBA_CPU_NAME=generic`, a cold compile took 58.4 s and a
warm one still took 23.3 s, because the one function that dominates compile
time is recompiled every run. The cache is 7.8 MB across 93 index files, so
size is not the obstacle; the uncacheable function is. Cross machine
portability was never established either, and cannot be tested here.

**Lowering the optimisation level does not work.** Measured at depth 9, all
three returning the correct 259210 fingerprint:

| NUMBA_OPT | nps | wall |
|---|---|---|
| 3, the default | 440,984 | 42.5 s |
| 1 | 387,201 | 43.1 s |
| 0 | 6,833 | 81.9 s |

Level 1 saves no compile time and costs 12 percent speed. Level 0 is 64 times
slower to run and slower to compile. There is no trade available here.

What remains untried, if a moderate cut needs more than `BTC_ENDGAMES=0`:
making `negamax` cacheable by passing its global tables as arguments. That is
a large refactor and it adds reference counting traffic to the hot path, which
is the cost btc_nrt exists to avoid. Not a finals day change.

### 3. Cores

Highest upside. Order of work: `nogil=True`, then a threaded driver with one
`SearchState` per thread over shared `tt_key` and `tt_data`, aspiration jitter
per thread, first finisher wins.

Enabling threads must force `BTC_NRT=0` automatically. The non-atomic refcount
patch is silent memory corruption the moment a second thread exists. The
interlock belongs in code, not in a document.

Note the trade: `BTC_NRT` is worth 32 percent single-threaded, so parallel has
to beat that before it is worth taking.

### 3b. Starting positions, and a network blind spot

Found while chasing an unrelated report. On kiwipete, a synthetic perft
position, the network evaluates -188 where the hand-crafted evaluation says
+42, and the search drives that to -265. Removing the black h3 pawn makes the
two agree almost exactly, +153 against +138, so the whole disagreement is that
one pawn, which the network prices at about 341 centipawns.

This is not a bug. The network learned from 2.8B positions out of real games,
and a pawn wedged on h3 against an uncastled king does not occur there. It is
out of distribution and the network is unreliable on it.

The consequence for tomorrow is the part worth keeping. If the announced
change is the starting position set, and the new positions are unusual rather
than merely unfamiliar, the network is the component most exposed, and it will
be confidently wrong rather than uncertain. Two responses, in order:

- A curated set of normal-looking positions is fine. Real game structures are
  exactly what the network is good at, whatever the specific opening.
- Genuinely irregular positions, Chess960 above all, are where the network
  degrades. `BTC_NNUE=0` falls back to the hand-crafted evaluation, which is
  weaker on average but does not have distributional blind spots. That is a
  panel value and it is the lever to reach for, and it is worth an SPRT on the
  new position set before trusting either way.

### 4. Chess960

Guardrails done and verified, real castling deliberately not attempted.

`tests/test_frc_safety.py` passes 28 checks over seven positions, including two
Chess960 start arrays, a king on b1 array, an irregular midgame, and the
standard array written in X-FEN form. It asserts three things: every position
parses, every generated move is legal in python-chess Chess960 mode, and
`get_move` returns a legal move. The book answered the standard array even in
X-FEN form.

The engine therefore plays Chess960 legally but never castles: file-letter
rights such as `HAha` parse to no rights at all. That is a real handicap and a
survivable one. Crashing or playing an illegal move is a forfeit; declining to
castle is a worse position.

If it has to be implemented on the day, the eight sites are `CASTLING_RIGHTS`
(btc_core.py:51), `_gen_castling` (:563), `_move_castle_rook` (:666), the
`make_move` castle block (:755), `_parse_castling` (:851), `to_fen` (:913),
`move_to_uci` (:925), and the three `chess.Board` calls in `agent.py`, which
need `chess960=True`. `st[5]` is free and copy-make already preserves it, so
the king file and both rook files fit there without widening any array. Budget
four to six hours and do not start it under time pressure: the safety net above
is what makes not starting it an acceptable outcome.

Note the evaluation caveat in section 3b. Chess960 start arrays are exactly the
out of distribution positions the network is weakest on, so if Chess960 is
announced, test `BTC_NNUE=0` against the network rather than assuming the
network is better.

## Measured facts to stop re-deriving

Memory, peak RSS for a full agent import plus 12 plies:

| tt entries | table | peak rss | peak commit |
|---|---|---|---|
| 1 << 24 (shipped) | 256 MB | 682 MB | 1134 MB |
| 1 << 22 | 64 MB | 491 MB | 943 MB |
| 1 << 21 | 32 MB | 459 MB | 910 MB |
| 1 << 18 | 4 MB | 432 MB | 883 MB |

Scaling is exactly linear. The floor is 432 MB RSS and it is numba, not us.
2 GB and 1 GB need no action, 512 MB needs `BTC_TT_ENTRIES=2097152`, 256 MB is
out of reach. numpy commits 498 MB while using 26 MB, so RSS is the honest
column.

Init timings are not trustworthy while training runs: the same import measured
36 s to 104 s across one sweep.

## Outstanding before any upload

- ~~`finals_day/test_staged.py` re-run against the rebuilt staged build.~~
  **Done 04:10, 7 of 7.** Import 0.077 s, handover after 7 plies at 27.3 s.
  Both are better than the figures recorded earlier in this file, because the
  earlier run was measured while training had the CPU. On an idle machine the
  real engine is ready by move 4.
- **Nothing is committed.** Gustavo writes every commit. Modified: `.gitignore`,
  `README.md`, `src/agent.py`, `src/btc_core.py`, `src/btc_game.py`,
  `src/btc_search.py`, `src/btc_time.py`, `tools/openings.py`,
  `tools/package.py`. New: `src/btc_parallel.py`, `finals_day/`,
  `tests/test_frc_safety.py`, `tests/test_hostile_fen.py`,
  `tests/test_parallel.py`, `tests/test_time_budget.py`,
  `tools/watch_match.ps1`.
- The SPRT logs written by PowerShell `Tee-Object` are UTF-16, not ASCII. They
  are logs rather than source, but do not commit them without converting.

## Rules that do not change

- Gustavo writes every git commit and performs every upload. Never either.
- Every batch: implement, adversarial subagent audit, regression suite.
- Fingerprint `tools/searchbench.py 9` must read 259210. Perft must read 8343.
- One SPRT at a time. Four workers maximum. Never a match beside training.
- Never pipe a match through `tail`; the opening-count warning is on line one.
- ASCII only, no emoji, no em dashes. No external engine names in markdown.
- Never install torch into `.venv`.
