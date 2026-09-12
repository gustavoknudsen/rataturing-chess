# How Rataturing was built

Status: written for the organisers on 11 September 2026, the day the qualification build was locked, and kept as written. The final's changes (a third book, the staged wrapper, the compile trims, the hash-key fix and the time-management recalibration) are in docs/FINALS_DAY.md.

A walkthrough of the engine, the network, the book, and how each was made and tested.

Rules quoted here are from https://aichessathon.com/docs, read 2026-09-11.

---

## Summary

Rataturing is a port of BetterThanCris, a chess engine I wrote in C before this competition. I ported it to Python and compiled it with numba, then extended the search well past what the C engine had.

The network is mine, trained from scratch over five runs. The training data is Stockfish-labelled positions from a public dataset. No Stockfish code and no published network ship.

The opening book is two Polyglot files. The main one is 78.8% my own labelling. Both are gated at move 20 in code.

The submission is 15 Python files, one network and two books. 40.0 MB of the 50 MB cap. No native binaries.

---

## 1. Where the engine came from

I wrote BetterThanCris before this event. It is written in C, extended over time with some C++ files and utilities. For this competition I ported it to Python, because the platform runs Python only.

The port is a translation of my own engine, not of anybody else's. No Stockfish, Lc0 or Maia code is present in any form.

While porting I found and documented 698 lines of bugs in my own C engine. That record is `docs/research/BTC_UPSTREAM_ISSUES.md`.

---

## 2. What the port required

numba forced three structural changes.

numba 0.67 cannot compile mutual recursion. In the C engine `negamax` calls `qsearch` and `qsearch` calls back into `negamax`. That fails to compile in numba, so the whole search lives in one function of about 2500 lines. This is why `src/btc_search.py` looks the way it does.

numba cannot mutate module-level arrays inside compiled code. They are read only. So all mutable state passes as arguments, and extra scalars ride in a single array called `sc`, because every additional array argument costs reference counting on the hottest path.

I replaced numba's atomic reference counting with a non-atomic version in `src/btc_nrt.py`. It is worth about 32% throughput and is only correct because the engine is single threaded. The file says so at the top.

Beyond the port, the search gained late move reductions, SEE pruning, continuation history, correction history, singular extensions with multicut, ProbCut and internal iterative reductions. Each was measured separately before it shipped.

---

## 3. The network

**Architecture.** King-bucketed. 32 king buckets by 768 features, into a 512-wide layer, with 8 output buckets keyed on piece count. 24,576 inputs. Quantised to int16 with QA=255, QB=64, SCALE=400. The file is `src/net.npz`, 25 MB.

**Data.** Stockfish `test80` binpacks, the `linrock/test80-2022` set on HuggingFace. About 1.01 billion usable rows per month file. I used a deduplicated pass over five months.

**Trained from scratch.** Random initialisation.

**Where.** Kaggle notebooks, in `training/notebooks/`. The trainer is `training/nnue_train.py`.

**The data filter.** `tools/binpack_stream.cpp` drops a position if the score is beyond 10000, if the best move is a capture or promotion, or if the side to move is in check.

**Five runs, four failures.** The failure was the same every time: the learning rate schedule never finished decaying, so the weights were left mid flight. A 512-wide net with a completed cosine schedule beat a 1024-wide net at 62% of its schedule by 12% validation loss.

| run | schedule completed | validation |
|---|---|---|
| 768x32x8 | 40% | 0.018108 |
| 768x32x8 retry | 48.6% | 0.018229 |
| 1024x30x8 | 62.4% | abandoned |
| **512x32x8 shipped** | **100%** | **0.016078** |

**Why 512 and not wider.** Two reasons. 32 king buckets at L1=1024 is 50.3 MB, over the cap on its own. And width costs nodes: as the engine got faster, the penalty for L1=1024 over L1=512 grew from 6.9% to 17.4%. I tested a 1024 net against the shipped 512 in a match and it scored 49.7%, so the extra width was not paying for itself.

**How nets were ranked.** `training/nnue_rank.py`. In-training validation loss is not comparable across runs, because each run validates on a different slice of rows. So every candidate is scored on one fixed row set, as quantised, dequantised back from the int16 file the engine actually loads. The holdout month is one no candidate trained on. Validation loss is also blind to speed and twice pointed the wrong way, so the final decision is always a match.

Full detail in `docs/research/NNUE_TRAINING.md`.

---

## 4. The opening book

Two Polyglot books ship. Standard format, read by `chess.polyglot` with no custom code.

| file | entries | my own labelling | merged from other sources |
|---|---|---|---|
| `rataturing.bin` | 475,719 | 375,079 (78.8%) | 100,640 (21.2%) |
| `rataturing_hedge.bin` | 552,078 | 22,286 (4.0%) | 529,792 (96.0%) |

The main book is mine. I took the tournament's curated starting positions, expanded outward across the replies worth covering, and labelled those positions with Stockfish myself.

The hedge book is mostly not mine. It fills gaps from public game data (lichess) and from Cerebellum, an existing Polyglot book. It is consulted only when the main book misses.

Which entries are which is verifiable, not asserted. Every entry carries its label score in Polyglot's `learn` field, offset by +100000. An entry with `learn == 0` came from a merged source and carries no score of mine. The table above was computed from that field, and `book/verify/verify.py` uses the same one.

**The move 20 gate.** `src/btc_book.py` sets `MAX_BOOK_MOVE = 20` at line 21. `in_window()` checks `board.fullmove_number <= MAX_BOOK_MOVE` against the referee's own FEN before any lookup happens. Three independent legality guards sit on top of that, because a hash collision returning an illegal move is an instant loss.

The book cannot answer move 21. `tests/test_book.py` gates that.

The rules place an originality requirement on networks and not on books, so the hedge book's third-party gap fill is within them. The full survey of which books I examined and what I took from each is `book/SOURCES.md`.

---

## 5. How it was tested

**Matches decide.** Self play A/B with a sequential probability ratio test. H0 is 0 elo, H1 is 20 elo, bounds at plus and minus 2.94. The tool is `tools/arena_par.py`.

Openings are assigned in strict order from a 600-position set, `book[(game // 2) % len(book)]`. That changes how results read: a trend inside a match is a different set of openings, not variance. At 150 games the 95% interval is still about plus or minus 70 elo, so I do not call a match before roughly 150 games without a real bound.

**Node counts screen, they do not decide.** `tools/searchbench.py` gives nodes to a fixed depth. It is deterministic and reproduces exactly across runs and machines. It rejects things the engine cannot afford before a three hour match is spent on them. It measures cost, never value.

Deltas do not compose, so everything is measured on the actual shipping build. Three measurements inverted when re-measured on the real baseline instead of standalone.

**Fixed suites, run as gates.**

    tests/run_tests.py     8343 assertions, perft bit exact
    tests/test_see.py      1131
    tests/test_uci.py        22
    tests/test_draw.py       17
    tests/test_book.py       20
    tests/test_kpk.py        15
    tests/test_convert.py    10
    tests/test_mate.py        8

`test_convert.py` is the sensitive one. It measures how many moves a KBN versus K conversion takes. Three features broke it: delta pruning, fail-soft quiescence, and a depth clamp. No match would have found those.

---

## 6. How features were added

The loop, in order. Written up in full in `docs/research/BATCH_PROCESS.md`.

1. Screen the idea's node cost at fixed depth, on the shipping build, before implementing it properly.
2. Implement behind an environment flag, default off. Module-level constants are folded by numba at compile time, so a flag that is off costs nothing.
3. One flag per idea, even inside a batch, so a failed group can still be bisected.
4. Verify the flag is live. Off must reproduce the anchor node count exactly; on must differ. Singular extensions nearly shipped disabled because a default was never flipped, and this step caught it.
5. Audit the diff adversarially.
6. Re-audit the fixes. A fix written under time pressure is unreviewed code going into a shipping build. One of mine put a `tt_record` early return one line after the key write, which was worse than the bug it repaired.
7. Run the gates.
8. Run the match.

Rejections are recorded with their measurements, not just the accepted changes. Correction history came in at +13 elo with an interval of [-27, +52] and needed four independent signals agreeing on the sign before I trusted it. One search change cut nodes by 31.6% and measured minus 29 elo. The log is `docs/research/PROGRESS.md`.

---

## 7. What ships

15 Python files, `net.npz`, and the two `.bin` books. 40.0 MB unzipped against the 50 MB cap. `agent.py` at the root, exposing `get_move(fen, time_left_ms)`.

No native binaries. No third-party engine code. No published network. Nothing obfuscated; the source is commented throughout and explains why it is shaped the way it is.

---

## 8. Checking any of this

    python tools/package.py      builds the zip, prints its size and which evaluation ships
    python tests/run_tests.py    8343 assertions, perft bit exact
    python tests/test_book.py    the move 20 gate
    python tests/test_uci.py     the UCI protocol, 22 checks

    git ls-files | grep -iE "\.(exe|dll|so|dylib)$"     returns nothing

The development record is `docs/research/PROGRESS.md`, 1206 lines, dated by session, with measurements and with the mistakes I made and corrected written down alongside the successes.

---

## 9. Disclosures


**Stockfish is used offline in three places, and ships in none.** As the labeller for my opening book, as the source of the network's training data, and as an analysis reference when diagnosing a lost game.

**The hedge book is mostly not my own work.** It is 4.0% my labelling, with the rest gap-filled from public game data and Cerebellum. It runs only when the main book misses, and it cannot answer past move 20.