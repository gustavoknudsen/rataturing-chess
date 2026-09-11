# Opening book pipeline

Builds the two Polyglot books the engine ships. Both are standard `.bin`, read by `chess.polyglot` with no custom code.

The rules allow a shipped table to answer the opening or the endgame, and define the opening as a position whose move number is 20 or lower. A table that answers a middlegame position is a stored search and counts as an engine. Everything here stays inside that bound, and `btc_book.MAX_BOOK_MOVE` enforces it at runtime. See [`../docs/RULES.md`](../docs/RULES.md).

## The two books

| file | covers |
|---|---|
| `rataturing.bin` | the tournament's curated starting positions, to move 20 |
| `rataturing_hedge.bin` | the standard start, as insurance for openings the main book misses |

The main book is built outward from the curated starts, which is why it has no entry for the standard starting position. Its moves come from our own engine labelling. The hedge fills gaps from public game data and an existing book, and is consulted only when the main book misses.

Entries carry the label score in Polyglot's `learn` field, offset by `+100000`. An entry with `learn == 0` came from a merged source and carries no score of ours. That is what lets `verify/verify.py` separate our labelled moves from gap-fill.

## Inputs you need

None of these are in the repository. Put them where the defaults expect and every stage runs unmodified.

**Stockfish**, for labelling positions. Download a binary from [stockfishchess.org/download](https://stockfishchess.org/download/) and put it at:

    book/data/engines/stockfish.exe

Used by `label/label.py`, `expand/candidates.py` and `expand/build_tree.py`. Any UCI engine works; pass `--engine` to point elsewhere.

**Lichess evaluation database**, an optional second label source. Download `lichess_db_eval.jsonl.zst` from [database.lichess.org](https://database.lichess.org/) and pass its path directly:

    python label/scan_evals.py /path/to/lichess_db_eval.jsonl.zst

It is tens of gigabytes and is streamed, not loaded. `label/pick_threshold.py` decides whether its evaluations are deep enough to trust before any of them are used.

**A source Polyglot book**, for gap-fill where labelling has not reached. Put any `.bin` at:

    book/data/books/

`merge/mine_cerebellum.py` defaults to `Cerebellum3Merge.bin`; pass `--book` for a different one.

**httpx**, only for the scraper:

    pip install httpx

## Running the pipeline

Stages run in this order. Each reads and writes in `data/`, so a stage can be run from any working directory.

**1. Scrape.** Recovers the tournament's curated starting positions from played games.

    python scrape/chessathon_scrape.py          # writes data/scrape/
    python scrape/analyse_openings.py           # games -> opening positions
    python scrape/export_book.py                # start positions + continuations

If you already have `games_index.csv`, drop it in `book/data/scrape/` and skip the first step. The two consumers will tell you if it is missing.

**2. Expand.** Enumerates the positions the book must answer, with the probability of actually reaching each one, and generates the opponent replies worth covering.

    python expand/expand_wide.py --help
    python expand/candidates.py --help
    python expand/build_tree.py --help          # drives to closure, unattended

**3. Label.** Scores those positions. This is the slow stage and the one that bounds the book's size: entries are limited by labelling time, not by disk.

    python label/label.py --help                # resumable, parallel, crash-safe
    python label/scan_evals.py --help           # optional Lichess evals
    python label/pick_threshold.py --help

**4. Merge.** Fills gaps from an existing book and builds the hedge trees.

    python merge/mine_cerebellum.py --help
    python merge/mine_deep.py --help
    python merge/hedge.py --help
    python merge/merge_hedge.py --help

**5. Build.** Writes the Polyglot files and copies them to the engine.

    python build/build.py --help
    python build/package_books.py --help        # copies into src/

**6. Verify.** Run before anything ships.

    python verify/verify.py --help
    python verify/simulate.py --help

`verify.py` is the gate, and every check is a hard failure: entries sorted by key, every move legal in its position, every position inside the move-20 bound, no truncated records. `chess.polyglot` binary-searches the file, so an unsorted book reads as mostly empty and fails silently, which is exactly the kind of defect that reaches a game unnoticed.

`simulate.py` answers the question that decides whether a book is worth shipping: how many moves does it actually play, and what clock does that buy.

## What is tracked

`data/` holds stage inputs and outputs. Most of it is not in the repository: the harvested PGNs, full label tables and source books come to roughly 1.3 GB.

A small reference set is tracked, because it shows what the book was built from:

| file | what it is |
|---|---|
| `start_fens.txt` | the curated starting positions the book must answer |
| `starts_clean.txt`, `starts_deep.txt` | filtered and extended variants of those |
| `candidates16.tsv` | generated opponent replies |
| `conflicts.tsv` | positions where sources disagreed |
| `labels.tsv`, `labels_d30.tsv` | a sample of our own engine labelling |
| `hedge_tolabel.txt`, `round1.txt` | hedge working sets |
| `need_candidates_ordered.txt` | expansion queue order |

`data/engines/` and `data/books/` are ignored entirely and do not appear in the repository; `data/README.txt` records what belongs in them.

## Related

- [`INTEGRATION.md`](INTEGRATION.md) - the runtime interface as built, with coverage measurements and test vectors
- [`SOURCES.md`](SOURCES.md) - the existing books surveyed as gap-fill candidates
- [`../src/btc_book.py`](../src/btc_book.py) - the engine-side probe and its guards
- [`../tests/test_book.py`](../tests/test_book.py) - the gate on those guards
