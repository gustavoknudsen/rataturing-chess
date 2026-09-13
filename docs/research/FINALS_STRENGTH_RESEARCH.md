# Strength research, finals day 12:xx

Status: the read-only research pass made at midday on finals day, ranking what could still be built before uploads closed. Section references to a day plan point at what is now docs/FINALS_DAY.md. Items 1 and 2.2 shipped; 2.1, 2.3 and 2.4 are behind flags; see that document.

Read-only pass over the docs, the code and the three platform logs in
finals_day/logs. Nothing was run; every number is measured elsewhere in the
repo (cited) or arithmetic on the code as it stands. Excluded by instruction:
compile trims (C1-C6) and static eval in the TT (6.1), both in progress.
Calibration: 1.7 elo per percent nps, 20 percent time odds = 28 to 39 elo,
and every tree-widening refinement tested so far came out negative.

## 0. Ranking

| # | item | kind | est. elo | agent time | gate | risk |
|---|---|---|---|---|---|---|
| 1 | OVERHEAD_MS 420 -> 100 | time | +10 to +25 | 15 min + 1 practice round | tc_tune with measured overhead, practice log | low (arithmetic below) |
| 2 | TT move searched before move generation | speed, identical tree | +5 to +12 | 45 to 75 min | fingerprint 259210, nps up | low-medium |
| 3 | child inherits in_check; accumulator built after the child's TT probe | speed, identical tree | +2 to +5 | 45 min | fingerprint, nps up | low-medium |
| 4 | partial selection with lazy SEE (extends DAY_PLAN 6.2) | speed, identical tree | +5 to +10 | 60 to 90 min | fingerprint, nps up | medium |
| 5 | king-bucket accumulator cache (finny table) | speed, identical tree | +2 to +4 | 60 to 90 min | fingerprint, test_accumulator | medium |
| 6 | TT 256 MiB -> 32 MiB bench | speed, maybe | 0 to +4 | 15 min bench | fingerprint, nps | low locally, unknown in long games |
| 7 | draw preference for the knockout | policy | unknowable | 30 min | none available | do not ship |

Items 2, 3 and 4 all edit the negamax and qsearch loops. Do them in a
worktree, one at a time, rebased onto whatever 6.1 lands as (6.1 edits
_node_prologue, qsearch and tt_record in the same file). The fingerprint gate
is the cheap part; the merge is the expensive part.

## 1. What the platform logs say about the clock

Three logs, 24 readings of the `implied overhead` line: every value is 0, 1 or
2 ms. Book moves cost us 1 ms of wall time. The platform's per-move charge is
therefore about 2 ms, not 420.

Round 7 (win as black, 71 moves, finals_day/logs/aichessathon-round-7-*.log):

- 55 searched moves, clock 105.99 s at the first search move (after the 14 s
  compile donation), 53.9 s at checkmate. Net drain 0.95 s per move, so the
  search spent about 1.45 s per move including the 0.5 s increment.
- `ours` per move ranged 1.1 to 2.1 s. The soft budget at that clock is
  about 2.6 s (arithmetic in section 2.1), so the engine spends 0.45 to 0.55
  of soft. That is exactly what ITER_RATIO 2.0 plus the 0.7 stable-move cut
  predict: iteration d+1 starts only while elapsed < soft/2 (or 0.35 soft
  once the best move is stable), and costs about 0.46x the time already spent.
- The opponent finished with 18 s, we with 54 s. Depths 16 to 21 in the
  middlegame, 447k nps on the platform (959,562 nodes in 2,145 ms).

## 2. Proposals

### 2.1 OVERHEAD_MS 420 -> 100 (src/btc_time.py:15)

What: the reserve `OVERHEAD_MS * (2 + MTG)` at btc_time.py:40 is 17,640 ms of
clock the budget never sees. With the measured 2 ms the honest model is under
10; 100 keeps a 50x margin over the measurement and a 10x margin over the
sum of everything charged per move (our own 1 ms, referee 2 ms, search
overshoot at most 9 ms, DESIGN_DECISIONS section 11).

Exact arithmetic, opt_scale = (0.9 + move/120) / 40, soft = opt_scale *
(clock + 19,500 - reserve):

| clock, move | soft shipped | soft at 100 | change |
|---|---|---|---|
| 105 s, move 8 | 2583 | 2908 | +12.6% |
| 90 s, move 20 | 2450 | 2808 | +14.6% |
| 60 s, move 40 | 1907 | 2322 | +21.7% |
| 30 s, move 60 | 1115 | 1586 | +42% |

hard follows: min((clock - 100)/4, 2.5 soft). The engine keeps spending
about half of soft, so game-average thinking time rises 15 to 20 percent,
weighted toward the second half of the game, where a knockout against a
stronger opponent is decided.

Flag risk: a move can never spend more than (clock - 100)/4 above 1.5 s, and
below 1.5 s the emergency branch gives clock/2 + 400 capped at clock - 100,
so at a 200 ms clock we spend 100 ms and receive 500. The total per-move
charge outside the search is about 12 ms worst case against a 100 ms reserve.
The only way this flags is a platform charge 50x larger than every reading
in three logs.

Two cautions, stated so they are not rediscovered:

- ITER_RATIO 1.5 was REJECTED (TONIGHT_PLAN.md line 342: 20.5/49, 41.8 percent,
  LLR -1.59, stopped). That was a change to the stopping rule, tested at
  30 s + 0.3 s BEFORE the proportional floor fix, where the 17.6 s reserve
  exceeded half the clock and both arms were starved to 100 to 500 ms moves.
  It is weak evidence against "more time helps" at 120 + 0.5, but it is the
  only game evidence there is, and it is why the estimate above is +10 to +25
  rather than the +25 to +35 the time-odds calibration would give for +18
  percent time.
- finals_day/tc_tune.py hard-codes REAL_OVERHEAD_MS = 420 and will report that
  model 100 FLAGS at 120 + 0.5 (at a 595 ms clock it charges 420 on top of a
  495 ms move). That is the stale assumption, not a real flag. Run it with
  REAL_OVERHEAD_MS = 20 first; it survives with room.

Gate: tests/test_time_budget.py (it imports the constant, so it stays green),
tc_tune.py with REAL_OVERHEAD_MS = 20, then one practice round: no move near
hard, a sane clock trajectory, implied overhead still 1 to 2 ms. Do not touch
MTG in the same upload; it is the divisor and confounds the reading. 60
would give another 2 percent; the margin is worth more.

### 2.2 TT move before move generation (src/btc_search.py:2257-2258)

What: at every interior node the engine generates all moves (2257), scores
every one including a see_ge per capture, and insertion-sorts the list
(1137-1220, 1123), then searches the TT move first because it carries
SCORE_TT_BEST. When the TT move fails high, which is the common outcome at a
cut node, all of that work was for nothing. Search the TT move first without
generating; generate and sort only if it does not cut off, and skip
`mv == tt_move` in the loop the way `mv == excluded` is skipped now.

Why identical: the TT move is always first in the sorted list (unique top
score), _see_prunable and _skip_quiet both return False for the first legal
move, and the only state the loop carries past move one is legal_count,
moves_searched, best_score, alpha, the tried-move lists and the PV, all of
which the staged version writes identically. IIR (tt_move == 0) and singular
(runs before generation) are untouched. The verification search (excluded
!= 0) must skip the stage; IIR and the singular block run before it as now.

The port-time survey rejects full "staged move generation" as a large refactor
for about 6 elo. This is the first stage only, about 40 lines, and unlike the
full refactor it is checkable by the fingerprint.

Cost model: SPEED_RESEARCH measured _sort_moves at 967 ns and the insertion
sort at 420 ns for a 37-move node after the hoist, plus generate_moves at
perft rates (about 200 to 300 ns). So a TT-move cutoff node saves about
1.2 us against a 2.1 us average node. If 10 to 15 percent of all nodes are
interior nodes that cut on the TT move, that is 5 to 8 percent nps; halve it
for pessimism, 3 to 8 percent, 5 to 12 elo.

Guard: a two-line pseudo-legality check before making the TT move (mover bit
set at the source; target not own-occupied unless the castle flag is set).
A 64-bit key match (48 after 6.1) makes a foreign move astronomically
unlikely, but a corrupt move would XOR a piece that is not there and the
subtree would be nonsense until unmake restores it. Gate: fingerprint 259210
with a lower time; tests 8343; test_convert; test_search.

### 2.3 Node-entry redundancies (identical tree)

Two things every node computes that its parent already knew.

a. in_check is computed twice per interior node: the parent computes
`opp_in_check = _in_check(bb, st)` for the move (2307) and the child computes
`in_check = _in_check(bb, st)` again for the same position (1769), and qsearch
computes it a third way when entered at depth 0 (1345, the parent's
opp_in_check is that same value). Add a scalar `in_check_hint` parameter to
negamax and qsearch (scalars cost no NRT traffic, BATCH_PROCESS.md section 1):
opp_in_check from the move loop, 0 after a null move (a legal position with
the mover not in check has the opponent not in check either), the node's own
in_check for the singular and null verifications, -1 where unknown (qsearch
captures, ProbCut, root). is_under_attack is about 30 to 40 ns on roughly
half of all nodes: 1 to 1.5 percent.

b. The accumulator is built for children that never evaluate. acc_update runs
at 2320 (negamax, after pruning) and 1432 (qsearch) before the child probes
the TT, and the child returns on a TT hit, a repetition or mate-distance
pruning without ever reading acc[ply]. Move the build into the child: after
_prologue_head returns NODE_CONTINUE (or before the ply-cap evaluation and the
NODE_QSEARCH dispatch), build acc[ply] from acc[ply-1] and undo_bb[ply-1],
which is exactly the parent's snapshot. The null-move child keeps the copy at
2017 and must not rebuild; it is identifiable by played[ply] == 0 with ply > 0.
qsearch needs an `acc_ready` scalar because it has no `played`: negamax passes
0 unless reached by null, the razoring calls pass 1, qsearch captures pass 0.
A delta update is a 2 KB copy plus two to four weight rows, about 80 to 120
ns; TT cutoffs plus repetitions are 15 to 25 percent of nodes: another 1 to
1.5 percent.

Both are bit-identical by construction and the fingerprint catches either
going wrong (a stale accumulator changes node counts). Do (a) first; drop
(b) if the merge with 6.1, which also moves work around the TT probe, gets
awkward.

### 2.4 Partial selection with lazy SEE (extends DAY_PLAN 6.2)

DAY_PLAN 6.2 proposes pick-best-remaining instead of the full insertion sort,
preserving stability (first maximum, shift not swap). Nobody is implementing
it. It is worth more done together with lazy SEE:

- _capture_score (1024) calls see_ge for every capture at every node, in
  both _sort_moves and _sort_captures, before knowing whether the move is
  ever reached. Score captures optimistically as good (SCORE_GOOD_CAPTURE +
  base); when a capture becomes the current maximum, run see_ge; if it is
  bad, rewrite its score to SCORE_BAD_CAPTURE + base and pick again.
- The emitted sequence equals the stable-sorted sequence: every emitted move
  has its true score, ties keep generation order, and the qsearch break at
  the first bad capture (1424) still holds because a bad true score can only
  be the maximum once every remaining optimistic score has been confirmed.
- In qsearch, test _delta_prunable (1426) on the candidate BEFORE its SEE;
  a delta-pruned capture is dropped without ever paying for see_ge, and
  because alpha at that moment is the alpha the current loop would have had
  when it reached the same move, the tree is unchanged.
- Everything that reads scores[ply, i] after the fact (_see_prunable,
  _skip_quiet, LMR) reads the true score of the emitted move, as now.

Cost model: see_ge is a swap-off loop with attackers_to per iteration, 60 to
120 ns; a middlegame node scores three to six captures and a cut node
reaches one; qsearch nodes are the majority and most reach one or two.
With the sort gone at cut nodes: 3 to 7 percent nps, 5 to 10 elo. 60 to 90
minutes because both sort sites and both loops change; any tie-order slip
shows up as a changed node count.

### 2.5 King-bucket accumulator cache (btc_nnue.py:308-328)

What: with 32 buckets the bucket is the exact canonical king square
(training/nnue_train.py:62 docstring), so EVERY king move changes _persp_key
for its own perspective and update() falls into _refresh_side (321, 327): all
pieces times 512 weight rows, 14x a normal delta in a 28-piece middlegame.
Keep, per perspective and per key (64 = 32 buckets x 2 mirror states), the
last accumulator built for that key and the bitboards it was built from;
on a bucket change, copy the cached accumulator and apply _apply_delta from
the cached bitboards to the current ones (typically two to eight rows).

No new negamax parameter: extend `acc` to (MAX_PLY+1+64, 2, L1) and
`undo_bb` to (MAX_PLY+128, 16) so the cache rides in arrays the search
already passes. A cache entry whose stored OCC_A is zero means "empty": take
the full refresh and store. A freshly zeroed SearchState is therefore valid,
nothing ever needs resetting, and the stored bitboards guarantee correctness.

Value: king moves are 3 to 5 percent of moves in the middlegame (where the
refresh is expensive) and 20 to 40 percent in endgames (where it is cheap);
1 to 2.5 percent nps, 2 to 4 elo. Gate: fingerprint, tests/test_accumulator.py
(it calls update() directly, so its call site changes with the signature).

### 2.6 TT size bench (src/agent.py PANEL, BTC_TT_ENTRIES)

NIGHT_PLAN measured node counts bit-identical from 16 MiB up but never nps.
Every tt_probe into 256 MiB is a DRAM miss; 32 MiB may sit in L3 on the
platform's Zen 4 core. Bench `BTC_TT_ENTRIES=2097152 tools/searchbench.py 9 3`
against shipped; both must print 259210; if nps does not move, drop it.
Hazard the bench cannot see: a game searches about 50 M nodes and the table
persists across moves, so 2 M entries evict within a few moves at depth 18
to 20. Ship only after a practice game shows the usual depths.

## 3. The knockout angle: draw preference

Not recommended, and here is the mechanics in case Gustavo wants it anyway.

- There is no contempt term. _draw_score (1642) takes only bb and returns 0.
  Six call sites (repetition, fifty-move in negamax and qsearch, stalemate)
  plus the eval-side zeros in btc_eval._specialised and btc_endgame.probe,
  which FINALS_DAY.md says must move together or the engine trades into a
  "drawn" KBvK it loses.
- A pro-draw setting would be: store the root side in sc, pass st and sc to
  _draw_score, return +D when st[SIDE] is the root side and -D otherwise, with
  D around 20 to 30 engine units. 30 minutes. The referee auto-claims
  threefold, so a repetition the search plans really does end the game.
- Why not: the expected-score gain for the weaker side is a few elo of match
  score at most; the tiebreak format is unknown, and at a faster control our
  14 to 16 s compile spill is a larger handicap than a draw bias offsets; the
  round 7 checkmate as black does not say we are the weaker side; and there
  is no gate, since self-play cannot measure a policy aimed at one opponent.
- What already exists: scoring one in-tree repetition as a draw when a claim
  needs three makes the engine draw-averse when ahead and draw-seeking when
  behind (TONIGHT_PLAN.md line 60). That is an adaptive bias in the right
  direction, and measured, unlike a constant.

## 4. Already tried, do not redo

Time management: ITER_RATIO 1.5 rejected (41.8 percent at 49 games, 30+0.3);
node effort v1 -6, v2 -19 elo; BTC_SOFT_CONT -12; MTG 24 -> 40 and OVERHEAD
120 -> 420 were a redistribution against a per-move cost that the logs now
show does not exist (section 1).

Search: cut-node LMR plus null-in-cut-nodes 47 percent at 50 games; capture
SEE dynamic margin -23; killer clearing +11.5 / +21 percent nodes; 3 and 4
continuation planes +8 to +11 percent nodes (TONIGHT_PLAN's -1.6 was an
older base); narrower aspiration window more nodes; history pruning inert;
lazy stand-pat in qsearch worse at every margin (fail-soft has no value to
return); lazy eval margins declined; IIR at all nodes +37.8 percent;
improving-beta +8.5; opp-worsening +1.3; draw jitter +1.9; RFP 168 vs 130
flat; SEE pruning via a second see_ge self-defeating; adaptive null R ON
after the 30 s re-measure; SING_DOUBLE, NULL_VERIFY, PROBCUT_FULL,
DEPTH_CLAMP off with recorded reasons; DRAW_HIST negative; anti-draw
contempt rejected on reasoning.

Speed: eval hash changed the node count (rejected); float propagate and int32
blocked partials slower; fused accumulator update already vectorised;
inline='always' nothing; duplicate see_ge in qsearch already CSE'd (-0.7
percent); bounded repetition scan 0.5 percent; NUMBA_OPT no lever; numba cache
impossible; torch and numpy-from-Python 10 to 30x a node; bigger TT worthless;
threads and pondering need cores we do not have; 32x384 and 16x256 nets lose.

Open in the docs and not proposed here: mailbox piece array (SPEED_RESEARCH 4,
smaller since _captured_piece was unrolled, new state in make/unmake);
DAY_PLAN 6.3 and 6.4 (tree-shaping, need a match, the class that has not
transferred).

## 5. Process notes

- Every item except 2.1 is verified by `tools/searchbench.py 9 3` printing
  exactly 259210 with a lower time (min of three, interleaved with the
  baseline), then tests/run_tests.py 8343, test_convert, test_search and
  test_accumulator. A speed item that moves the node count is a bug.
- Record the local compile minimum after each item: compile spill is paid
  from the clock at 1.6 to 1.7x, so +0.5 s is under 1 elo, but write it down.
- 2.1 is one PANEL line (BTC_MOVE_OVERHEAD) or btc_time.py:15. Its gate is a
  practice round, so upload it alone, early, and read the log before
  stacking anything on it.
- Order for one implementation agent with 90 minutes: 2.1 (15 min, then it
  runs on the platform while the rest is built), 2.2, then 2.3a. Items 2.4
  and 2.5 only if a second agent is free; each is a self-contained worktree
  job.
