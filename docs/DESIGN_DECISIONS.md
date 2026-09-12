# Rataturing: Design Decisions

Why the engine is built the way it is. Each entry states the constraint, the options, and the decision. Measurements are from the development machine (Windows, Python 3.12, numba 0.67.0, numpy 2.5.2, python-chess 1.11.2). The match machine is a single EPYC 9V74 core and is slower, so local numbers are optimistic by roughly 1.5x to 2x.

Rataturing is built on BetterThanCris (BTC), a C engine by the same author, since extended with C++ files and utilities. The rules ban third-party engines and any port or translation of one, and explicitly permit your own work; BTC is the author's own engine. See `RULES.md`.

## 1. Numba is the platform, and it sets the rules

Everything below follows from four properties of numba 0.67, all measured rather than assumed.

**No mutual recursion.** numba compiles self-recursion but cannot infer return types through a cycle of two functions. This is why the entire main search is one function. `negamax` has a cyclomatic complexity in the hundreds and cannot be decomposed: any helper that calls back into `negamax` fails to compile. Full ProbCut had to be inlined for this reason.

**Bounds checking is off.** An index past the end of an array is a silent bad read or write, never an exception. Every array is therefore sized from the same constant that indexes it, and every tunable that reaches an index is clamped at its definition.

**Module-level constants are folded at compile time.** A feature behind `USE_X = os.environ.get("BTC_X") == "1"` costs nothing when off, because the branch is removed entirely. This is what makes it practical to carry many experimental features in one source file and enable them per match.

**Array arguments cost reference counting.** Each array passed to an njit function incurs an incref/decref pair on every call. `negamax` already passes 22 arrays, so adding a 23rd is not free. Extra scalar state rides in a shared `sc` array instead, read into a local immediately because a deeper ply overwrites the slot.

## 2. Board representation: own bitboards

python-chess is too slow in the hot path and its objects cannot cross into njit code. The board is a numpy array of bitboards, passed by reference through the search.

python-chess is still used, deliberately, at the boundary: parsing the FEN the referee sends, the legality guard on book moves, and a final legality check on the move we return. Those are once-per-move costs where correctness matters more than speed, and using a well-tested library there removes a whole class of bug.

## 3. Magic tables: shipped as data

Sliding-piece attack tables were generated during import until finals day, when the init budget dropped to 30 seconds and every second of compile counted. They now ship as `src/attack_tables.npz` (53 KB compressed) and are loaded with `np.load`; the generator remains as the fallback if the file is missing, and a test asserts the loaded tables equal a fresh build.

## 4. Make/unmake, not copy-make

Copy-make allocates a board per node. Under numba the allocation dominates. Make/unmake with an explicit undo stack keeps the search allocation-free once compiled.

## 5. Moves are packed into a single int32

Source, target, piece, promotion and flags in one integer. A struct of arrays would mean several array arguments per call, which by rule 1 above is the expensive kind of change. Packing keeps move lists as plain int32 arrays.

## 6. Compilation is forced at import, then staged for the final

The qualifier allowed 90 seconds before the clock starts, and no output within that window is a loss. Compiling lazily would mean paying for it inside the first move's time budget, so everything compiles during import, driven by a short warm-up search with the real argument types.

The final cut the budget to 30 seconds against a compile of about 40 seconds on the platform's core. The answer was a wrapper that compiles on a thread, spends 26 seconds of the window waiting for it, and pays the remainder from the clock on move one, where the book usually answers. `docs/FINALS_DAY.md` has the mechanics and the measurements. Local init figures vary by a third with load and core placement on the development laptop, so they are only ever compared as the minimum of repeated runs.

## 7. Transposition table sizing

The table is sized to fit comfortably in the platform's memory alongside numpy and numba. torch is never imported: it costs both import time and resident memory, and nothing in the engine needs it.

Entries are packed into a single uint64: move in bits 0 to 23, score 24 to 41, depth 42 to 48, flag 49 to 50, age 51 to 58, and a principal-variation bit at
59. The packing is worth stating because it caused a real bug. A negative depth shifted into the unsigned depth field sets every bit above 42, so the entry reads back as depth 127 and a depth-preferred replacement policy can never evict it. Worse, each failed overwrite refreshed its generation and made it harder to evict. The fix was upstream: guarantee the null-move child at least one ply of real search, so negative depths cannot occur.

## 8. Game state across calls

The referee sends a FEN per move and nothing else, but repetition and the fifty-move rule need history. A tracker reconstructs the game from the FEN sequence and maintains the repetition stack. Book moves push through the same tracker, because a move played without updating it desynchronises repetition detection for the rest of the game.

## 9. Network width: smaller than it looks like it should be

The shipped network is king-bucketed: 32 king buckets by 768 features each (12 piece planes by 64 squares) into L1 = 512, with 8 output buckets keyed on piece count. That is 24,576 inputs. Wider was tried and rejected on measurement, not taste.

| L1 = 1024 against L1 = 512 | |
|---|---|
| validation loss | 4.2 percent better |
| nodes per second | 17.4 percent worse |
| match result | 49.7 percent, no gain |

The reason generalises badly and is worth stating precisely. The network cost per node is roughly fixed in nanoseconds and roughly linear in L1, while the rest of the engine kept getting faster. So the network's share of a node grows, and the width penalty grows with it: the same comparison was 6.9 percent before the speed work and 17.4 percent after. Every search speedup pushes the optimal network smaller at a fixed node budget.

There was also a hard limit. L1 = 1024 is 50.3 MB, over the submission cap.

Do not carry "512 is correct" to another engine. Carry the method: does the validation gain beat the speed loss in a match.

## 10. Opening book

Shipped, contrary to the original plan. The early assumption was that rated games start from curated positions, so a book built from move 1 would never fire, and that was correct as far as it went. The book that ships is built outward from the curated positions themselves.

Verified before shipping: it answers all 8 published curated samples and a real rated game position, it plays 12 consecutive moves from that position, and across 30,000 sampled curated positions it produced no illegal move.

Every lookup is gated at move 20. The rules permit a shipped table to answer "a position whose move number is 20 or lower" and treat a table answering a middlegame position as an engine in another shape. A Zobrist key carries no move number, so a position stored at move 5 would otherwise return a hit at move 34. That runtime gate is the only thing keeping the book on the legal side of the line, and it must not be raised.

## 11. Time management

Budget per move is derived from the clock and an assumed number of moves remaining. Two constants govern it, the per-move reserve and the moves-to-go divisor, and both were recalibrated twice.

The engine's own search does not overshoot: measured against its authorised budget at clock values from 120 seconds down to 250 milliseconds, actual wall time came in within 9 milliseconds at the low end. Whatever the model was missing was on the platform side.

Before the qualifier the reserve was raised from 120 to 420 milliseconds and the divisor from 24 to 40, on the inference that a rated floor near 200 milliseconds implied a real per-move charge near 420. That inference was wrong. On finals day the engine's clock lines measured the platform's charge at 1 to 2 milliseconds on every move, and the engine had been finishing lost games with a minute unused. The reserve is now 100 milliseconds and the divisor 28, chosen from a sweep of the budget function itself over 60, 100 and 150 move games at the observed spend: never flags, about 30 percent more thinking in a normal game, and a 150 move game still ends with clock in hand. The lesson is the one that recurs through this project: measure the platform, do not infer it.

## 12. Hash keys

The C engine generated its 64-bit Zobrist keys from four draws of a 32-bit xorshift. That generator is a linear map of its state, so all 849 keys span a 32-dimensional space over GF(2) and every position hash carried only 32 bits: distinct positions collided at the 32-bit birthday rate, dozens of times per million nodes. It was found on finals day, when a static-evaluation cache keyed on the hash moved node counts that should have been identical. The keys now come from splitmix64 and have full rank; the fixed-depth fingerprint changed by a single node.

## 13. The fingerprint

`tools/searchbench.py 9` prints the total node count of a fixed-depth search over fifteen positions. It is deterministic across runs and machines, so it identifies the search exactly: a change that is meant to be pure speed must leave it untouched, and a change that is meant to alter the tree must change it in the direction claimed at more than one depth. It is the cheapest gate in the project and the one that caught the most.

## Resolved

Questions the early version of this document left open, with their answers.

- **Split negamax and quiescence**: split, as in BTC. The recursion proof held.
- **Eval accumulator**: incremental, updated through make/unmake, with a full refresh only at the root. A test asserts the incremental accumulator is bit-identical to a full refresh at every node.
- **Opening book**: shipped. See section 10.
- **Syzygy tablebases**: dropped. The useful sizes do not fit the cap, and pure-Python probing is too slow for in-search use.
