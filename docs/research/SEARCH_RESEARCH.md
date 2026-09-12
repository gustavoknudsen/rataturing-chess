> **STALE, 2026-09-12.** Checked against the code, not against this file.
> Tier 1 items 2 and 3 (node-fraction soft scaling, quiescence check evasions)
> and all four tier 3 items (double extensions, multicut, negative extension
> refinement, cut-off on singular beta) are **implemented**. Tier 3 is
> described below as blocked behind singular extensions being off; singular
> extensions are on and have been since the `singular extensions + kpk`
> commit. Cut-node awareness, quiescence TT probe and delta pruning are in as
> well. Of the ranked list only `BTC_CAP_SEE_DYN` (tier 2 item 4, published
> 13.97) is implemented-but-untested, and killer clearing (tier 2 item 7) was
> screened on 2026-09-12 and costs 11.5 to 21 percent more nodes at fixed
> depth. Verify against the source before acting on anything here.

# Search and evaluation research, 2026-09-08

Status: research note from 8 September 2026, kept as written. Most of the tier 1 and tier 2 items were implemented in the days after; the tier 3 dependency (singular extensions) was resolved when they shipped. Read it as the reasoning at the time, not as a to-do list.

What the current state of the art does that this engine does not, with published Elo
where it exists, filtered by whether it can plausibly transfer to **our** situation:
a shallow search (depth 11-19, roughly four to five plies behind the C original), a
hand-crafted the reference engine-11-class evaluation, one core, Python under numba, and a hard
90 s compile budget every game.

Everything here is a technique described in public sources and implemented from that
description. None of it is transcribed from another engine's source.

## The primary source, and why

`tcheran`'s CHANGELOG publishes an **SPRT result per patch**, which almost no engine
does. That makes it the only source in this space where individual features can be
ranked by measured elo rather than by folklore. It is used below in preference to the
Chess Programming Wiki, which describes techniques but rarely quantifies them.

Two caveats applied throughout:

- **tcheran is NNUE from version 6 onward.** Its evaluation-side numbers after that
  point tell us nothing. Its *search* numbers still transfer, since search is
  evaluation-agnostic.
- **Its hand-crafted-evaluation era (v4-v5) is far weaker than our evaluation.** It
  gained +41 from piece mobility, +38 from passed pawns, +24 from king safety - all
  terms we already have, in a more developed form. **The evaluation lever for us is
  tuning, not new terms.**

## Confirmed against our own work today

Two of today's changes have published figures that match what we saw, which is a useful
check that the measurements are real:

| change | published | ours |
|---|---|---|
| Null move pruning formula (adaptive R) | **29.37 +-10.50** | part of -21.4% nodes |
| Zugzwang avoidance in null move | **5.70 +-3.94** | part of the same |
| Quiescence futility (our delta pruning) | 4.32 +-3.18 | -12.9% nodes |
| Transposition table in quiescence | **40.16 +-11.74** | -12.3% nodes (session 4) |
| SEE pruning | 25.45 +-9.40 | -22% nodes (session 4) |
| Internal iterative reductions | 9.66 +-5.53 | -24% nodes (session 4) |

## Ranked candidates for the remaining time

Ordered by published elo weighted by how well it transfers to a shallow search, and
divided by risk. Items marked **[BTC too]** are absent from the C engine as well and are
recorded in `BTC_IMPROVEMENTS.md`.

### Tier 1 - highest value, moderate cost

1. **Cut-node awareness.** **[BTC too]** BTC has no concept of it (`grep -ci cutnode`
   returns 0), so neither does the port. A node is a "cut node" when it is expected to
   fail high; passing that expectation down lets the search reduce and prune far more
   confidently there. Two separate published patches:
   - More LMR in cut nodes: **9.87 +-5.55**
   - Null move pruning in cut nodes only: **8.61 +-5.05**

   Cost: one boolean threaded through `negamax`. At a PV node the first child is a PV
   node and the rest are cut nodes; the child of a cut node is an all node, and vice
   versa. This is the single largest structural gap between our search and a current
   one.

2. **Time management: soft-limit scaling by node fraction.** Published as two patches,
   **22.35 +-8.36 STC** (tweaks) and **10.89 +-5.82 STC / 16.01 +-6.94 LTC** (soft time
   scaling). We already scale the soft limit by best-move stability; the missing part is
   scaling by *what fraction of nodes went to the best move* - if one move consumed most
   of the tree, the position is not close and the iteration can stop early.

   Very attractive here: time management is pure Python at the root, so it costs **zero
   compile budget**, which nothing else on this list can claim. STC figures are the
   relevant ones for our 120 s + 0.5 s control.

3. **Check evasions in quiescence.** **[BTC too]** Published **5.16 +-3.69**. BTC's
   quiescence calls `generateCaptures` only (search.cpp 546, 735), so when the side to
   move is in check it stand-pats on a position where the static evaluation is
   meaningless and most legal replies are forced. We reach that case whenever a checking
   move exhausts the last ply of depth. Costs an `_in_check` call per quiescence node,
   which is why many engines limit it to the first ply or two of quiescence.

### Tier 2 - good value, small and self-contained

4. **Capture history as the SEE threshold.** **13.97 +-6.76**, surprisingly high.
   Instead of a fixed SEE cutoff, scale the threshold by the capture's history: a
   capture that has repeatedly worked gets more benefit of the doubt.
5. **Static evaluation cached in the transposition table.** **5.05 +-3.69**. Removes
   re-evaluation across iterative deepening, aspiration re-searches and PVS re-searches.
   Needs spare bits in the TT entry.
6. **LMR reduction on a failed-high re-search.** **7.80 +-4.80**.
7. **Killer clearing for the next ply.** **5.77 +-3.92**. We clear killers once per
   search; BTC does the same (`memset`, search.cpp 124/1170). Clearing ply+1's killers
   on entry to a node keeps them from being stale siblings.
8. **Check-giving quiet moves scored higher in ordering.** **4.97 +-3.54**.
9. **Pinned pieces handled in SEE.** **7.58 +-4.67**. Our SEE is already the expensive
   part of move ordering under numba, so this one has a real speed cost attached.

### Tier 3 - depends on singular extensions, which are currently off

10. Double extensions **9.28 +-5.26**, multicut in the singular search **7.99 +-4.84**,
    negative extension refinement **4.56 +-3.38**, cut-off on singular beta
    **6.97 +-4.40**. **[BTC too]** for all four - BTC has a plain 1-ply singular
    extension and nothing else (search.cpp 836-903).

    All of these are blocked behind re-enabling singular extensions, which is currently
    off because it broke KBN vs K in combination with RFP_MARGIN=130. Worth revisiting
    only if there is time after Tier 1.

### Rejected on our own measurements

- **Aspiration window shrink.** Published at **10.03 +-5.56** for 25->20 and
  **3.40 +-2.70** for 20->15, and our value is **77**, which looks indefensible. It is
  not. Swept on the fixed-depth benchmark, node count rises monotonically as the window
  narrows:

  | delta | 77 | 40 | 25 | 20 | 15 | 10 |
  |---|---|---|---|---|---|---|
  | nodes | **943,275** | 988,086 | 1,013,309 | 1,009,596 | 1,071,701 | 1,094,637 |

  Two reasons it does not transfer. BTC's evaluation is on a unified scale where a pawn
  is **126**, not 100, so 77 is really about 61 centipawns. And our search is four to
  five plies shallower, so the score swings more between iterations and a tight window
  buys nothing but re-searches. **Keep 77.** A good reminder that a published constant
  is only meaningful together with the units and the depth it was tuned at.

- **History pruning of quiets.** Standard, and measured at nothing here: redundant with
  LMP as configured, which at depth <= 4 already admits only `(3 + depth^2)/2` quiets.
  Node counts moved non-monotonically across margins. Details in `PROGRESS.md`.

- **Lazy evaluation.** the reference engine walked it back (commit 7b278aa) because it let a bad
  move persist to high depth; it passes short controls and fails long ones. Our
  evaluation is a third of runtime so the temptation is real, but it trades accuracy for
  speed and we are already the less accurate side of that trade.

- **Staged move generation, low-ply history, pawn history, multicut, upcoming-repetition
  detection.** All under 10 elo published, and BTC's own SPRT measured low-ply and pawn
  history at nothing. Reasons in `BTC_IMPROVEMENTS.md`.

- **Lazy SMP** (**39.62** for 2 threads). We have one core. Not available.

- **Tablebases** (18.97 for 5-man). 3-4 man fits in 50 MB but decides almost no games;
  5-man does not fit.

## On the evaluation

**Corrected after an actual audit.** The first version of this section argued from class
- "ours is the reference engine-11 lineage, therefore stronger than any published hand-crafted
evaluation, so adding terms is not where the elo is." That is reasoning, not research.
Enumerating every term constant in the port against the modern reference's `evaluate.cpp`, and
against Ethereal v11-12 (~3200 CCRL, the strongest HCE ever written), found **two real
gaps**, both detailed in `BTC_IMPROVEMENTS.md` items 17a and 17b:

- **the modern reference's `initiative()` is missing from BTC entirely** (`grep -ci initiative`
  returns 0) - a *fidelity* gap, since our evaluation claims to be an SF11 port and SF11
  runs this on every evaluation. It expresses how *winnable* a position is and pulls
  sterile positions toward a draw.

  **It is not portable here, and the reason matters more than the term.** the reference engine
  carries a packed midgame/endgame score to the end of `evaluate()` and interpolates
  once; `initiative()` reads both halves separately to clamp its correction. BTC tapers
  every term individually into a single integer, so there is nothing for it to read. The
  same missing split is why the endgame scale factor is already an approximation here
  rather than exact. **Not attempted before the lock** - it is an architectural refactor
  of every evaluation term, which is the wrong bet two days out. Written up as
  `BTC_IMPROVEMENTS.md` 17a.
- **Closedness** (Ethereal): knight and rook values scaled by how open the position is,
  from open-file and rammed-pawn counts. Absent from the SF11 lineage entirely. A genuine
  new idea, so its constants would need tuning - materially riskier.

Everything else checks out: the port has the complete SF11 term set. So the claim that
survives is narrower than the one I made - **the evaluation has no *missing category*,
but it does have one missing component of its own reference.**

The rest of the evaluation lever is still tuning:

- **Retuning for our depth.** Published for a full Texel-style retune: tcheran
  **53.17 +-16.76**, RubiChess ~150, Texel ~100 over seven patches, Blunder +60.9. Those
  are hand-set-to-tuned figures so ours would be smaller, but every constant we have was
  fitted at C node counts against a search four to five plies deeper, and the
  depth-gated heuristics cover a much larger fraction of our tree than of BTC's.
- **Material scaling of the evaluation** (**6.02 +-4.02**) - scaling the final score by
  remaining material, distinct from the endgame scale factor we already port.
- **SPSA rather than Texel for search constants.** tcheran's four SPSA runs measured
  **24.36**, **16.55**, **7.87** and **5.58** - i.e. tuning the search constants was
  worth more than most individual features. This is the technique behind the "retune"
  recommendation and it needs no engine changes at all, only machine time.

## What this does not include, and why

**Training a network.** The rules permit it (`any network you ship is one you trained
yourself`), torch and onnxruntime are on the platform, and NNUE is worth +223 elo in
tcheran's own history. It is excluded here purely on schedule: it is a rewrite of the
evaluation, its inference has to run under numba inside our node budget, and it would
have to be trained, validated and integrated in under two days against a submission
lock. The risk of shipping a half-working evaluation outweighs the upside. Recorded so
the decision is explicit rather than forgotten.

## Sources

- tcheran `CHANGELOG.md` - SPRT result per patch, the primary source
- Chess Programming Wiki: Late Move Reductions, Null Move Pruning, Delta Pruning,
  Static Evaluation Correction History, Texel's Tuning Method, Internal Iterative
  Reductions
- the reference engine commit b4d995d (correction history), 7b278aa (lazy evaluation walk-back)
- Viridithas, Stormphrax and Alexandria are the current strongest open-source engines
  and were checked, but all three are NNUE, so only their search ideas are relevant
- BTC v2.8 source, read directly to confirm each **[BTC too]** claim
