# Rataturing: Design Decisions

Why the engine is built the way it is. Each entry states the constraint, the options, and the decision. Measurements are from the development machine (Windows, Python 3.12, numba 0.67.0, numpy 2.5.2, python-chess 1.11.2). The match machine is a single EPYC 9V74 core and is slower, so local numbers are optimistic by roughly 1.5x to 2x.

Rataturing is built on BetterThanCris (BTC), a C++ engine by the same author. The rules ban third-party engines and any port or translation of one, and explicitly permit your own work; BTC is the author's own engine. See `RULES.md`.

## 1. Numba is the platform, and it sets the rules

Everything below follows from four properties of numba 0.67, all measured rather than assumed.

**No mutual recursion.** numba compiles self-recursion but cannot infer return types through a cycle of two functions. This is why the entire main search is one function. `negamax` has a cyclomatic complexity in the hundreds and cannot be decomposed: any helper that calls back into `negamax` fails to compile. Full ProbCut had to be inlined for this reason.

**Bounds checking is off.** An index past the end of an array is a silent bad read or write, never an exception. Every array is therefore sized from the same constant that indexes it, and every tunable that reaches an index is clamped at its definition.

**Module-level constants are folded at compile time.** A feature behind `USE_X = os.environ.get("BTC_X") == "1"` costs nothing when off, because the branch is removed entirely. This is what makes it practical to carry many experimental features in one source file and enable them per match.

**Array arguments cost reference counting.** Each array passed to an njit function incurs an incref/decref pair on every call. `negamax` already passes 22 arrays, so adding a 23rd is not free. Extra scalar state rides in a shared `sc` array instead, read into a local immediately because a deeper ply overwrites the slot.

## 2. Board representation: own bitboards

python-chess is too slow in the hot path and its objects cannot cross into njit code. The board is a numpy array of bitboards, passed by reference through the search.

python-chess is still used, deliberately, at the boundary: parsing the FEN the referee sends, the legality guard on book moves, and a final legality check on the move we return. Those are once-per-move costs where correctness matters more than speed, and using a well-tested library there removes a whole class of bug.

## 3. Magic tables: generated at import

Sliding-piece attack tables are built during import rather than shipped as data. Shipping them would cost megabytes against the 50 MB cap and save only a few seconds. Generation is part of the init budget, which is measured and has comfortable headroom.

## 4. Make/unmake, not copy-make

Copy-make allocates a board per node. Under numba the allocation dominates. Make/unmake with an explicit undo stack keeps the search allocation-free once compiled.

## 5. Moves are packed into a single int32

Source, target, piece, promotion and flags in one integer. A struct of arrays would mean several array arguments per call, which by rule 1 above is the expensive kind of change. Packing keeps move lists as plain int32 arrays.

## 6. Compilation is forced at import

The platform allows 90 seconds before the clock starts, and no output within that window is a loss. Compiling lazily would mean paying for it inside the first move's time budget. Everything compiles during import instead.

Measured init on an idle machine is about 56 seconds cold against the 90 second budget. It is load sensitive: measured under a saturated CPU it reached 89 seconds, which is a reason not to benchmark during a match rather than a reason to worry about the platform.

## 7. Transposition table sizing

The table is sized to fit comfortably in the platform's memory alongside numpy and numba. torch is never imported: it costs both import time and resident memory, and nothing in the engine needs it.

Entries are packed into a single uint64: move in bits 0 to 23, score 24 to 41, depth 42 to 48, flag 49 to 50, age 51 to 58, and a principal-variation bit at
59. The packing is worth stating because it caused a real bug. A negative depth shifted into the unsigned depth field sets every bit above 42, so the entry reads back as depth 127 and a depth-preferred replacement policy can never evict it. Worse, each failed overwrite refreshed its generation and made it harder to evict. The fix was upstream: guarantee the null-move child at least one ply of real search, so negative depths cannot occur.

## 8. Game state across calls

The referee sends a FEN per move and nothing else, but repetition and the fifty-move rule need history. A tracker reconstructs the game from the FEN sequence and maintains the repetition stack. Book moves push through the same tracker, because a move played without updating it desynchronises repetition detection for the rest of the game.

## 9. Network width: smaller than it looks like it should be

The shipped network is HalfKAv2-style, 32 king buckets by 11 piece planes into L1 = 512, with 8 output buckets keyed on piece count. Wider was tried and rejected on measurement, not taste.

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

Budget per move is derived from the clock and an assumed number of moves remaining. Two constants govern it and both were recalibrated late.

The engine's own search does not overshoot: measured against its authorised budget at clock values from 120 seconds down to 250 milliseconds, actual wall time came in within 9 milliseconds at the low end. The per-move cost the model was missing is on the platform side.

The original reserve of 120 milliseconds understated the real per-move cost of roughly 420 milliseconds, so every move leaked time the model did not know about. The base clock was exhausted by about move 57, after which the engine played 80 millisecond moves for the rest of the game. Raising the reserve to 420 and the assumed moves remaining from 24 to 40 redistributes the same total time: mean spend across 90 moves is essentially unchanged, but moves 40 to 90 get about 3 times longer to think.

## Resolved

Questions the early version of this document left open, with their answers.

- **Split negamax and quiescence**: split, as in BTC. The recursion proof held.
- **Eval accumulator**: incremental, updated through make/unmake, with a full refresh only at the root. A test asserts the incremental accumulator is bit-identical to a full refresh at every node.
- **Opening book**: shipped. See section 10.
- **Syzygy tablebases**: dropped. The useful sizes do not fit the cap, and pure-Python probing is too slow for in-search use.
