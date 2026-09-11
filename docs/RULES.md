# AI Chessathon - Competition Rules & Constraints (ground truth)

Kept as the record of the constraints the engine was designed against. Section 5 was re-verified against the published documentation on 2026-09-11; the rest is as gathered on 2026-09-07, from:

- **[L]** https://aichessathon.com/ (landing + FAQ)
- **[T]** https://aichessathon.com/terms
- **[D]** https://aichessathon.com/docs (spec page)
- **[AC]** https://aichessathon.com/docs/agent-contract.md (canonical, per starter repo)
- **[R]** https://aichessathon.com/docs/rules.md (canonical, per starter repo)
- **[S]** github.com/advitrocks9/aichessathon-starter - `AGENTS.md`, `harness/rules.py`, `harness/referee.py`, `harness/runner.py` (the harness mirrors the platform protocol)

`/daily` was not fetched (may block automated access); the Daily Five is a human puzzle side-event and does not affect agent engineering. Note its fair-play rule: it must be solved alone, **without an engine** [T].

The site warns that the docs change; re-fetch [AC] and [R] before the final upload.

---

## 1. Agent API [AC][D][S]

```python
# agent.py, at the ROOT of the zip (not in a folder). Platform does `import agent`.
def get_move(fen: str, time_left_ms: int) -> str: ...
```

- `fen`: standard FEN of the current position. Our colour is the side to move. There is no other input - no move history, no opponent move, no game id.
- `time_left_ms`: our clock **before** this move. The 0.5 s increment lands **after** the move.
- Return: one legal move in UCI (`e2e4`, `e7e8q`). A reply **over 4 KB counts as illegal** [S].
- Process model [AC][S]: the process starts **once per game** and stays alive between our moves; module state survives across our own moves within a game, never across games. The process is **suspended (no CPU) while the opponent thinks** - no pondering, no background work. Two of our games can run concurrently, in separate containers.
- The runner redirects fd 1 to stderr before importing agent, so `print` cannot corrupt the protocol; first 4 KB + last 4 KB of output are kept in a team-visible log with init time, per-move times and clock [S].

## 2. Environment [D][AC][S]

- Python **3.12**. Preinstalled, fixed versions; nothing else installs, and a shipped `requirements.txt` is ignored:
  - torch 2.13.0+cpu
  - numpy 2.5.2
  - python-chess 1.11.2
  - onnxruntime 1.29.0
  - numba 0.67.0
- CPU: **one core** of AMD EPYC 9V74 @ 2.60 GHz, exclusive during our turn. Max 128 processes, but more threads/processes than one core lose time [D][S].
- Memory: **2 GB**. OOM = loss of that game.
- **No network. No GPU.**
- Filesystem: **read-only root**; `/tmp` = **256 MB** scratch, **wiped after each game**; `HOME` and all cache paths point to `/tmp`. Consequence: **numba `cache=True` never hits**, so the engine recompiles from scratch every game, inside the init budget [D][S].
- Our zip is first on `sys.path`: never name a file after a stdlib/installed module (`chess.py`, `types.py`, `random.py`...) [S].

## 3. Time control & failure modes [D][AC][S + referee.py]

- **120 s base + 0.5 s increment per move, per side**, wall time.
- **Init budget: 90 s** before the clock starts (import time; no output within 90 s = loss). Harness grants a further 500 ms watchdog grace (`WATCHDOG_GRACE_MS = 500`) [S].
- **600-ply cap -> automatic draw**; plies are counted by `board.ply()` from move 1, so the curated opening position's own ply count is included [S: referee.py, AGENTS.md].
- Failure modes [D]:
- Illegal or malformed move -> **loss**
- Crash / OOM -> **loss**
- Flag (clock < 0 after a move) -> **loss**, unless the opponent has insufficient material
    to mate -> draw (referee: `has_insufficient_material(not mover)`)
- No output within 90 s init -> **loss**
- Both sides fail -> void, no result
- **The referee claims threefold repetition and fifty-move draws automatically** (`board.is_repetition(3)`, `board.is_fifty_moves()`), checked **before** each move is requested [S: referee.py]. Draws follow FIDE rules [S]. Repetition and fifty-move counts start **from the first FEN of the game** (its halfmove clock is the fifty-move baseline), not from move one [S: AGENTS.md].
- Rated games start from **curated opening positions**, not the standard start; the set is not published. The eight openings in `harness/rules.py` are a sample only [S].

## 4. Submission [D][AC][S]

- Zip archive, **files at the root** (no subdirectory). Required: `agent.py`. Optional: additional `.py` files, model weights, books, tablebases.
- **50 MB unzipped** total cap (`MAX_UNZIPPED_BYTES = 50_000_000` [S]).
- **10 uploads per team per day**; the latest upload that passed validation is the one that plays [S].
- **Upload deadline: 11 September, 11:00** [D][AC].
- Validation on upload: build test + **two smoke games** (one per colour, ~20 plies each [S: SMOKE_PLIES]) at match clock, against a house agent. The platform's validation log is the authority and reports real init time and slowest move [S].

## 5. Banned / allowed [D][AC][R][S]

Verified against aichessathon.com/docs on 2026-09-11. An earlier version of this file paraphrased a stale reading in which any shipped move table looked prohibited; the wording below is the site's.

Banned:
- **Third-party engines**: Stockfish, Lc0, Maia, and any wrapper, port or translation of one. Checked after games are played, not only at upload [S].
- **Native binaries** and compiled extensions (Cython does not work on the platform) [D][S].
- Obfuscated agents - "what you ship must be source a judge can read" [R].
- A published/pretrained network, even fine-tuned or re-exported - "any network you ship is one you trained yourself" [AC]; "Starting from a published chess network is not" [R].
- **Tables that answer a MIDDLEGAME position**: "A table that answers a middlegame position is a stored search and counts as an engine." Opening and endgame tables are explicitly allowed; see Allowed below.
- Network calls, subprocess to external binaries, reading outside agent dir + `/tmp` [S].

Allowed:
- **Our own pre-existing engine**: "Your moves come from code you wrote" [R] - BTC is the author's own engine, so building on it is within the rules. "A model is not required, a classical search is a full entry" [L].
- Model weight files `.onnx`, `.safetensors`, `.pt` (self-trained only) [D].
- **A table you ship and read during a game may answer the opening or the endgame.** Verbatim from the documentation. "The opening is a position whose move number is 20 or lower." The bound applies to the opening; the endgame allowance is separate and is not move-numbered.

  The matching ban is about WHERE the table answers, not where its data came from: "A table that answers a middlegame position is a stored search and counts as an engine." So a book merged from public game data, a third-party book and our own engine labelling is permitted, provided it only answers inside the allowed scope.

  That bound is not self-enforcing. A Polyglot key is a Zobrist hash and carries no move number, so a position stored at move 5 returns a hit at move 34 by transposition. `btc_book.MAX_BOOK_MOVE = 20`, checked against the referee's own FEN before any file is read, is what keeps us inside the rule. Do not raise it.

  `chess.polyglot` and `chess.syzygy` are in the base image. 3-4-man Syzygy fits in 50 MB; 5-man does not [S].
- Training data unrestricted, including engine-annotated positions [L].

## 6. Tournament format & dates [D][R][L]

- **Qualifier ladder**: 4-11 Sep 2026, hourly rated rounds 08:00-22:00. Open worldwide.
- **Submission lock: 11 Sep 11:00.**
- **Qualification Swiss**: 13 rounds, 11 Sep afternoon, locked builds.
- **Final**: 12 Sep, Encode Club, London. 50 seats; at least one UK university student per team required to enter the final Swiss; only UK members occupy London seats; max one seat per UK member / two per team [D][T].
- Tie-breaks: points, Buchholz, head-to-head, **earlier submission** [D].
- Teams: 1-3 people, one team per person [R]. Prizes: GBP 1,000 / 500 / 250 [L].
- Judging: agent design, match performance (Elo), stable execution [L].

## 7. Engineering consequences (derived, for quick reference)

- Everything jitted recompiles **every game** inside 90 s. Compile time is a first-class budget alongside NPS. Target ceiling ~65 s to leave margin (platform core is slower than the dev machine, and numba import + table building also pay into the 90 s).
- Do **not** import torch (RAM + import time; we don't use it). Avoid onnxruntime too.
- No pondering; all computation inside `get_move`. Time measured inside a move is ours alone.
- Keep module state across moves: game history for repetition tracking, TT carryover.
- Return-move safety net must guarantee *some* legal UCI move under every failure path  - an exception that escapes `get_move` is a lost game.
