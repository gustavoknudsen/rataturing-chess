# Training NNUEs: what was learned, and what to do differently without constraints

Status: what was learned training the shipped network and the finals-day candidates. The measurements are the author's own runs.

Written 2026-09-10, at the end of the Chessathon campaign. The competition forced
a 50 MB file, one CPU core, a 90 s compile budget and a numba engine. **Most of
what follows is not about those constraints** - it is about training, and it
applies directly to a future C++ BTC net with no size limit.

Where a finding is competition-specific it says so.

---

## 1. The single most expensive mistake: an unfinished schedule

Three runs in this project were destroyed by the learning rate never finishing its
decay. It is worth more than every hyperparameter below combined.

| run | schedule completed | val |
|---|---|---|
| 768x32x8 | 40% | 0.018108 |
| 768x32x8 (retry) | 48.6% | 0.018229 |
| 1024x30x8 | 62.4% | not ranked, abandoned |
| **512x32x8** | **100%** | **0.016078** |

A *smaller* net with a completed cosine beat a larger net at half a schedule by
**12%**. The learning rate is still near a quarter of peak when a run is cut off,
so the weights are left mid-flight.

**Rules that follow:**

- Size the row budget to the **time and data actually available**, not to the data
  you wish you had. `BTC_STREAM_ROWS` sets the cosine length, so it decides how far
  the LR decays - it is not just a stopping condition.
- **Know your corpus size before you start.** The 1024 run died because the budget
  was 6.5B and four months of `test80` hold only ~4.05B. Measured: **~1.01B rows per
  month-file**.
- Overrunning degrades gracefully (a schedule stopped at 90-95% has a nearly-decayed
  LR); stopping at 50% is ruinous. **Bias the estimate long, not short.**
- `nnue_train.py` prints a truncation warning when the stream ends early. It was
  added mid-campaign and immediately caught the 1024 run. Keep it, and treat it as
  fatal rather than advisory.

---

## 2. Feature factorization

Train a shared, unbucketed 768-row feature table alongside the bucketed one, and
**fold it into the bucketed weights at export** so inference is unchanged.

Why it matters: with 32 king buckets, a position only trains 1/32 of the feature
transformer. Rare buckets - the ones for unusual king placements - see very little
data and stay near their initialisation. Factorization gives every position a stake
in a shared table, so a rare bucket starts from something sensible and learns a
delta.

- Verified equivalent at export to **2.98e-08**, so folding costs nothing.
- Checkpoints must be **saved folded**, or a resumed run silently loses the shared
  table.
- Costs roughly a third of training throughput (the feature transformer does about
  twice the work). Budget for it in the row count.
- **It is a convergence accelerator**, which means it is worth most exactly when
  data per bucket is thin - i.e. when you have fewer rows, or more buckets.

---

## 3. Weight decay: use zero

AdamW applies decoupled weight decay to **every parameter every step, regardless of
gradient**. A rare bucket that appears in one row in ten thousand is decayed on all
the other steps, so its weights are pulled toward zero faster than the data can
push them anywhere.

Measured: bucket rarity correlated with damage at **+0.588**. Set
`BTC_WEIGHT_DECAY=0`. The regularisation an NNUE needs comes from quantisation and
from the sheer size of the corpus, not from decay.

This interacts with factorization - both target the same weakness - so a run with
factorization on and decay off is strictly the configuration to prefer.

---

## 4. Validation loss lies, in two specific ways

**It is not comparable across runs.** Each run's in-training validation slice is
drawn from a differently sized shuffle buffer, so it is a different set of rows.
Two runs reporting 0.0161 and 0.0159 may be in the opposite order on identical
data. `nnue_rank.py` exists for exactly this: it materialises **one fixed set of
rows** and scores every candidate on it.

**It is blind to speed, and twice pointed the wrong way.**

| net | val | result |
|---|---|---|
| k2_1024 vs shipped 512 | **4.2% better** | **49.7% in games** |
| a search change | 31.6% fewer nodes | **-29 elo** |

A wider net evaluates each position better and gets to look at fewer of them.
Validation loss sees only the first half of that trade. **Only an SPRT decides.**

Two further rules for ranking:

- Score nets **as quantised**, dequantised back from the int16 file the engine will
  actually load, so quantisation error counts against a net instead of hiding. A
  wider layer has more values sharing the same int16 grid.
- Choose a holdout **no candidate trained on**. In this project all the finalists
  used August as their in-training validation month, and the shipped net's corpus
  included August, so August was clean for nobody - ranking moved to April.
  The same net scores **0.016078** on one holdout and **0.017343** on another;
  never compare numbers across holdouts.

---

## 5. Architecture: what the numbers actually said

### Width is expensive, and gets more expensive as the engine gets faster

Measured on the shipped engine, same depth, same positions:

| build | L1=1024 penalty vs L1=512 |
|---|---|
| pre-optimisation (modelled) | 6.9% |
| after a 1.35x speed group | **13.1%** |
| after a further 1.47x | **17.4%** |

The reason is worth internalising: **the NNUE cost per node is roughly fixed in
nanoseconds and roughly linear in L1**, while everything around it got cheaper. So
its *share* of a node grows, and the width penalty grows with it.

**Every speed win pushes the optimal net smaller.** That is a statement about a
*fixed node budget*, and it is the finding least likely to transfer: a C++ engine
at 2-10M nps has a vastly larger budget and its optimum is far wider. Do not carry
"512 is right" across - carry the *method* for pricing it:

    is the width worth it?  ->  does the val gain beat the NPS loss in an SPRT?

At our budget, 4.2% val against 13% NPS came out at 49.7%. That is the exchange
rate; measure your own.

### King buckets versus output buckets

These are not the same kind of parameter and the difference is large.

- **King (input) buckets multiply the whole feature transformer.** 32 buckets at
  L1=512 is 25.2 MB; at L1=768, 37.7 MB; at L1=1024, **50.3 MB, which did not fit
  the competition cap**. They also split the data 32 ways, which is what makes
  factorization necessary.
- **Output buckets are nearly free.** Only one is read per evaluation, so inference
  cost is identical; going 8 -> 16 adds ~16 KB of file. Measured cost of 32 king
  buckets against 4: **1.01x per node**.

So under a size cap, prefer buckets over width. Without a cap, output-bucket count
is the cheapest capacity available and is worth sweeping properly - it was still
untested when this campaign ended.

### The parameter budget

    32 king buckets x 768 x L1 x 2 bytes
    L1=512  -> 25.2 MB      L1=768  -> 37.7 MB      L1=1024 -> 50.3 MB

---

## 6. Data

- Source: Stockfish `test80` binpacks (`linrock/test80-2022` on HuggingFace).
  ~1.01B usable rows per month-file, ~11-13 GB decompressed each.
- **WDL lambda 0.7** throughout - a blend of evaluation and game result. Never
  swept properly; a lambda ramp over training is a known technique and remains
  untested here.
- **Deduplicate.** The corpus used was a dedup pass over five months.
- **Training data is unrestricted under the competition rules**, including
  engine-annotated positions. Only *shipping* a third-party engine's output is
  banned. Offline labelling with a reference engine is explicitly allowed and is
  the obvious way to diagnose an eval - see section 8.

### The filter, and why it is deliberate

`binpack_stream.cpp` drops a position when:

    |score| > 10000                        (decisive, evaluation meaningless)
    the best move is a capture or promotion (static eval about to be invalidated)
    the side to move is in check           (cannot stand pat)

This is **standard practice and matches Stockfish's own trainer**, which is why the
filters were copied verbatim - it keeps nets trained on the same distribution
comparable. The rationale is sound: a static evaluator should be trained on
positions where a static evaluation means something, and quiescence resolves the
rest.

It does mean the net never sees a tactical position. Whether that is a weakness
worth fixing is **an open question, not a known bug** - and the one piece of
evidence gathered here pointed away from it (see section 8).

### One epoch or several?

For this trainer, **one epoch over more unique rows beats several over fewer**,
because each epoch restarts the stream from the beginning of the files:

    1 epoch  x 2.8B rows -> 2.8B presentations, 2.8B unique
    2 epochs x 1.4B rows -> 2.8B presentations, 1.4B unique

Multi-epoch only wins when compute genuinely exceeds unique data. Check which side
of that line you are on before reaching for more epochs.

---

## 7. Quantisation

`QA=255`, `QB=64`, `SCALE=400`, activations clipped to +-1.98, stm-first
accumulator. Worst-case accumulator value is asserted at export against the int16
range - the 512x32x8 net measured 16355 of 32767, comfortably safe.

`NET_UNITS` is **per-net** and rescales every search margin. It is produced by
`nnue_gate.py` and must be re-derived for every net - shipping a net with the
previous net's `NET_UNITS` silently mis-scales RFP, futility, razoring and delta
pruning. The shipped 512x32x8 uses 98.

Untested and worth doing with time: **quantisation-aware training**. Diagnose first
- re-run validation with the quantised weights dequantised back into a float model
and compare to the float number. If the gap is negligible, skip it.

---

## 8. Diagnosing an evaluation, properly

The most useful diagnostic of the whole campaign took ten minutes: take a lost
game, label the positions with a reference engine offline, and compare against the
net's own evaluation.

Example, from one loss:

| move | ours (Black) | Stockfish | verdict |
|---|---|---|---|
| 20 | +1.9 | +0.65 | optimistic, sign right |
| 29 | **+1.7** | **-0.5 (White better)** | **sign wrong, ~2.2 pawns out** |
| 31 | -0.3 | -0.32 | correct |

That immediately separates two failures that look identical from the outside:
move 29 is an **evaluation** error, move 31 is a **search** error (our evaluation
of the position was right to within 0.02 pawns; the move chosen lost 2.2).

**Do this before touching hyperparameters.** It says whether the net or the search
is at fault, and no amount of training fixes a search problem.

Two hypotheses that a fuller version of this diagnostic would separate, both
untested:

1. **Output bucket resolution.** If the error concentrates near a piece-count
   bucket boundary, 8 buckets is too coarse and 16 is the fix - free at inference.
2. **The training filter.** If it concentrates in sharp positions, that is section
   6's filter rather than the architecture, and it is fixable in data.

---

## 9. For a future BTC net with no size constraint

Ordered by expected value.

1. **Re-price width from scratch.** Everything in section 5 about 512 being right
   is a statement about a numba engine at ~0.5M nps. At C++ speeds the optimum is
   much wider. Run the same experiment - two widths, same data, same schedule,
   SPRT - and let it answer.
2. **Keep factorization and weight decay 0.** Neither is a competition artefact;
   both are about bucket sparsity and the AdamW decay rule, which do not change.
3. **Sweep output buckets.** Free at inference, never tested beyond 8, and the
   cheapest capacity available.
4. **Pairwise multiplication instead of SCReLU-then-dot.** Takes the 2xL1
   accumulator to L1 products before the dot, halving the output layer's terms
   while adding a multiplicative nonlinearity. **The only architecture change that
   is speed-positive.** Needs a retrain, never tried here.
5. **Loss shaping**, all untested: exponent 2.5 rather than 2.0; a draw-aware
   win-rate mapping rather than a single sigmoid; a lambda ramp rather than a fixed
   0.7.
6. **More data before more parameters.** Every failure in this project was a
   schedule or a corpus problem, never an architecture one.

---

## 10. Tooling that earned its place

| file | what it is for |
|---|---|
| `nnue_rank.py` | scores every candidate on **one fixed row set**, as quantised. The only comparison that means anything. |
| `nnue_gate.py` | derives `NET_UNITS` per net, plus R2/MAE gates against binpack labels |
| `nnue_train.py` | streaming trainer; prints the truncation warning that catches a dead run |
| `binpack_stream.cpp` | the filter, matched to Stockfish's so distributions stay comparable |
| `test_convert.py` | not a training tool, but the most sensitive instrument in the project - fixed-depth endgame conversion, which catches evaluation-scale bugs that SPRT never will |
