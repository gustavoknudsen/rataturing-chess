# Source books surveyed

Which existing Polyglot books were examined as gap-fill candidates, how large they are, and how often each agrees with an engine-derived book's top move. Only a small fraction of the shipped book comes from these; the main book's moves are our own labelling.

## What the uploaded books actually are

| book | MB | entries | distinct pos | root moves | main line |
|---|---|---|---|---|---|
| Cerebellum3Merge | 178.0 | 11,123,374 | 10,981,751 | 2 | 54 ply |
| codekiddy | 16.5 | 1,030,253 |  -  | 10 | 30 |
| Human | 14.7 | 920,716 |  -  | 20 | 9 |
| komodo | 9.3 | 578,126 |  -  | 2 | 58 |
| DCbook_large | 6.4 | 398,022 |  -  | 6 | 42 |
| Book | 5.8 | 360,125 |  -  | 3 | 28 |
| KomodoVariety | 4.3 | 267,489 |  -  | 17 | 8 |
| final-book | 2.9 | 183,804 |  -  | 11 | 30 |
| rodent | 2.8 | 175,355 |  -  | 10 | 40 |
| Elo2400 | 2.5 | 155,878 |  -  | 13 | 20 |
| Titans | 1.9 | 121,160 |  -  | 14 | 40 |
| varied | 1.5 | 92,229 |  -  | 6 | 27 |
| Performance | 1.5 | 92,954 |  -  | 3 | 29 |
| fruit | 0.5 | 31,480 |  -  | 11 | 38 |
| gm2600 | 0.35 | 21,671 |  -  | 10 | 20 |
| Perfect2023 | 0.05 | 3,127 |  -  | 4 | 16 |
| gavibook | 5.3 | **corrupt** |  -  |  -  |  -  |

Notes:

- **Cerebellum3Merge is byte-identical** (md5 `bb4ef38997cd08ba...`) to the file the `ChrisWhittington/polyglot-books` repo publishes as `book.bin` and credits to Jeroen Noomen. The credit is wrong; it is Cerebellum.
- **Cerebellum is already a best-move book: 1.01 moves per position.** "We only need one move per position" is what it already is. Nothing to strip.
- `gavibook.bin` is not a multiple of 16 bytes. Truncated or corrupt. Discard.
- `Perfect2023.bin` at 3,127 entries is the *short opening lines* set, not a lookup book. Useful as a source of start positions, not of moves.

## Agreement with Cerebellum's top move

Sampled 300 positions at moves 8-15. Coverage percentages are biased (the sample was drawn by walking Cerebellum), so read the agreement column.

| book | covers | agrees |
|---|---|---|
| rodent | 63.7% | **77.5%** |
| Titans | 46.7% | 74.3% |
| gm2600 | 23.3% | 71.4% |
| codekiddy | 65.0% | 70.8% |
| final-book | 59.0% | 68.4% |
| DCbook_large | 52.0% | 64.7% |
| Elo2400 | 31.0% | 63.4% |
| Performance / varied | 40.0% | 63.3% |
| komodo | 53.7% | 60.2% |
| KomodoVariety | 63.7% | 59.2% |
| fruit | 37.0% | 58.6% |
| Book | 59.0% | 57.6% |
| Human | 9.0% | 51.9% |
| Perfect2023 | 19.7% | 40.7% |

Cerebellum is the only engine-derived source in the set  -  it is Stockfish analysis. Everything else is built from game frequency, which is why they disagree with it 22-48% of the time. Treat them as gap-fillers with lower trust, not as equal opinions.

## The binding constraint is coverage, not bytes

Expanding from a start position with our book move and then **every legal** opponent reply:

| our plies | positions/start | in Cerebellum | coverage | MB @16B x 380 |
|---|---|---|---|---|
| 1 | 35 | 6 | 17.1% | 0.04 |
| 2 | 189 | 18 | 9.8% | 0.11 |
| 3 | 469 | 44 | 9.3% | 0.27 |

Three plies deep across 380 start positions is **0.27 MB**. The 24 MB budget is nowhere near binding. What binds is that Cerebellum answers only ~10% of positions once you branch over all legal replies.

That 10% is against *random* legal moves. Real opponents play a small subset, so real coverage will be far higher  -  but by how much is exactly what your PGN reply data answers, and nothing else does.

## bookbuild.py

Merges sources by priority, walks from your start positions in reach-probability order, emits a standard Polyglot `.bin` sorted by key so `chess.polyglot.open_reader` in the base image reads it with no custom code.

```
python bookbuild.py starts.txt out.bin --replies replies.tsv --budget-mb 20
```

`replies.tsv` is `fen<TAB>uci<TAB>count` from your PGNs. Without it the builder guesses at three continuations per node, which wastes most of the budget on lines nobody plays.

Smoke-tested on 40 synthetic starts: 1,252 positions, 97% from Cerebellum, 414 gaps. Those gaps are the positions worth Stockfish time.

## Two runtime guards, both mandatory

```python
if board.fullmove_number <= 20:          # scope: the rules define the
    mv = book_lookup(board)              # opening as move number <= 20
    if mv is not None and mv in board.legal_moves:   # collision guard
        return mv.uci()
```

The first keeps a transposed lookup from answering a middlegame position, which the rules class as a stored search and therefore an engine. The second stops a hash collision from producing an illegal move, which is an instant loss. Leave a comment saying why  -  a judge reads this file.
