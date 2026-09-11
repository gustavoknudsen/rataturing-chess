"""
AI Chessathon full scrape.

Phase 1  leaderboard  -> every team uuid            (plain HTTP)
Phase 2  team pages   -> every game uuid + opening  (plain HTTP)
Phase 3  game pages   -> PGN                        (plain HTTP, concurrent)

The site is a Next.js app: every team page ships its complete game table inside
the streamed RSC payload (self.__next_f pushes), so phase 2 needs no browser and
no "show all" click -- the table is server-rendered in full, newest first.

Outputs into ./out :
    all_games.pgn      every finished game, concatenated
    pgn/<uuid>.pgn     one file per game
    games_index.csv    uuid, round, white, black, result, termination, opening
    opening_counts.csv opening, count   (descending)

Resumable: re-running skips games already on disk, reuses teams.json / games.json,
and picks phase 2 back up from out/games.partial.json if it was interrupted.

    pip install httpx
    python chessathon_scrape.py
"""

import asyncio
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

# Requires httpx (async HTTP), which is not part of the engine's runtime
# dependencies: pip install httpx. Only this scraping stage needs it.
import httpx

BASE = "https://aichessathon.com"
OUT = Path("out")
PGN_DIR = OUT / "pgn"

# Be a good citizen. 8 is brisk but not abusive; drop it if you see 429s.
GAME_CONCURRENCY = 8
TEAM_CONCURRENCY = 8
UA = "Mozilla/5.0 (compatible; chessathon-archiver/1.0)"

TEAM_RE = re.compile(r"/team/([0-9a-f-]{36})")
GAME_RE = re.compile(r"/game/([0-9a-f-]{36})")
UUID_RE = re.compile(r"[0-9a-f-]{36}")
# The download link is an inline data URI holding the whole PGN, percent-encoded.
PGN_RE = re.compile(r'href="data:application/x-chess-pgn;charset=utf-8,([^"]+)"')
TAG_RE = re.compile(r'\[(\w+)\s+"([^"]*)"\]')

# Row fields inside the RSC payload, once un-escaped.
ROUND_RE = re.compile(r'"match-round".*?"prefetch":false,"children":\["([^"]*)"', re.S)
COLOUR_RE = re.compile(r'"match-colour","children":\[\[.*?\}\],"([^"]*)"\]', re.S)
OPPONENT_RE = re.compile(r'"team-link".*?"children":"([^"]*)"', re.S)
VERDICT_RE = re.compile(r'"match-verdict","data-verdict":"([^"]*)","children":"([^"]*)"')
OPENING_RE = re.compile(r'"match-opening","children":"([^"]*)"')
PUSH_RE = re.compile(r"self\.__next_f\.push\(\[1,")

_JSON = json.JSONDecoder()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


async def get(client: httpx.AsyncClient, url: str, **kw) -> httpx.Response:
    """GET with three tries and a widening pause -- the site rate-limits bursts."""
    for attempt in range(3):
        try:
            r = await client.get(url, **kw)
            r.raise_for_status()
            return r
        except Exception:                                    # noqa: BLE001
            if attempt == 2:
                raise
            await asyncio.sleep(2 * (attempt + 1))


# ----------------------------------------------------------------- phase 1

async def get_team_ids(client: httpx.AsyncClient) -> list[str]:
    cached = OUT / "teams.json"
    if cached.exists():
        return read_json(cached)

    ids: list[str] = []
    seen = set()
    # Every ladder stage, so teams that dropped off "live" are still caught.
    for stage in ("live", "qualifier", "finalset", "knockout"):
        r = await get(client, f"{BASE}/leaderboard", params={"stage": stage})
        for uuid in TEAM_RE.findall(r.text):
            if uuid not in seen:
                seen.add(uuid)
                ids.append(uuid)

    write_json(cached, ids)
    print(f"phase 1: {len(ids)} teams")
    return ids


# ----------------------------------------------------------------- phase 2

def flight_payload(html: str) -> str:
    """Concatenate the RSC chunks the page streams into self.__next_f."""
    parts = []
    for m in PUSH_RE.finditer(html):
        i = html.index('"', m.end())
        s, _ = _JSON.raw_decode(html, i)   # decodes the JS string literal exactly
        parts.append(s)
    return "".join(parts)


def parse_rows(html: str) -> list[dict]:
    rows = []
    for chunk in flight_payload(html).split('["$","tr","')[1:]:
        game_id = chunk[:36]
        # Standings rows share the <tr> shape; only game rows carry an opening.
        if not UUID_RE.fullmatch(game_id):
            continue
        opening = OPENING_RE.search(chunk)
        if not opening:
            continue
        rnd = ROUND_RE.search(chunk)
        colour = COLOUR_RE.search(chunk)
        opponent = OPPONENT_RE.search(chunk)
        verdict = VERDICT_RE.search(chunk)
        rows.append({
            "game_id": game_id,
            "round": rnd.group(1) if rnd else "",
            "colour": colour.group(1) if colour else "",
            "opponent": opponent.group(1) if opponent else "",
            "result": verdict.group(2) if verdict else "",
            "opening_from_team_page": opening.group(1),
        })
    return rows


async def get_all_games(client: httpx.AsyncClient,
                        team_ids: list[str]) -> dict[str, dict]:
    cached = OUT / "games.json"
    if cached.exists():
        return read_json(cached)

    # Partial checkpoint: written as we go, so an interrupted run resumes here
    # instead of being mistaken for a finished one.
    partial = OUT / "games.partial.json"
    games: dict[str, dict] = {}
    done: set[str] = set()
    if partial.exists():
        state = read_json(partial)
        games, done = state["games"], set(state["done_teams"])
        print(f"phase 2: resuming, {len(done)} teams already scraped")

    todo = [t for t in team_ids if t not in done]
    sem = asyncio.Semaphore(TEAM_CONCURRENCY)
    failed: list[str] = []

    async def one(tid: str):
        async with sem:
            try:
                r = await get(client, f"{BASE}/team/{tid}")
            except Exception as e:                           # noqa: BLE001
                print(f"  ! team {tid}: {e}", file=sys.stderr)
                failed.append(tid)
                return
            for row in parse_rows(r.text):
                games.setdefault(row["game_id"], row)
            done.add(tid)
            n = len(done)
            if n % 20 == 0 or n == len(team_ids):
                print(f"phase 2: {n}/{len(team_ids)} teams, "
                      f"{len(games)} unique games")
                write_json(partial, {"done_teams": sorted(done), "games": games})

    await asyncio.gather(*(one(t) for t in todo))
    write_json(partial, {"done_teams": sorted(done), "games": games})

    if failed:
        print(f"phase 2: {len(failed)} teams failed; re-run to retry them",
              file=sys.stderr)
    else:
        write_json(cached, games)          # only now is the set provably complete
        partial.unlink(missing_ok=True)
    print(f"phase 2: {len(games)} unique games")
    return games


# ----------------------------------------------------------------- phase 3

async def fetch_pgn(client: httpx.AsyncClient, game_id: str) -> dict | None:
    """Return the game's metadata + PGN, or None if it hasn't finished."""
    dest = PGN_DIR / f"{game_id}.pgn"
    if dest.exists():
        pgn = dest.read_text(encoding="utf-8")
    else:
        r = await get(client, f"{BASE}/game/{game_id}")
        m = PGN_RE.search(r.text)
        if not m:
            # No download link = still on the board, or void. Skip it.
            return None
        pgn = unquote(m.group(1))
        dest.write_text(pgn, encoding="utf-8")

    tags = dict(TAG_RE.findall(pgn))
    if tags.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
        return None

    # The opening name lives on the team page, not in the PGN tags.
    return {"game_id": game_id, "pgn": pgn, "tags": tags}


async def phase3(games: dict[str, dict]) -> list[dict]:
    sem = asyncio.Semaphore(GAME_CONCURRENCY)
    done: list[dict] = []
    skipped = 0
    errors = 0

    limits = httpx.Limits(max_connections=GAME_CONCURRENCY)
    async with httpx.AsyncClient(headers={"User-Agent": UA}, limits=limits,
                                 timeout=45, follow_redirects=True) as client:
        async def one(i: int, gid: str):
            nonlocal skipped, errors
            async with sem:
                try:
                    rec = await fetch_pgn(client, gid)
                except Exception as e:                       # noqa: BLE001
                    print(f"  ! game {gid}: {e}", file=sys.stderr)
                    errors += 1
                    return
                if rec is None:
                    skipped += 1
                else:
                    rec["opening"] = games[gid].get(
                        "opening_from_team_page", "") or "Unknown"
                    done.append(rec)
                if i % 250 == 0:
                    print(f"phase 3: {i}/{len(games)}  kept {len(done)}  "
                          f"skipped {skipped}")

        await asyncio.gather(*(one(i, g) for i, g in enumerate(games, 1)))

    print(f"phase 3: {len(done)} finished games, {skipped} unfinished, "
          f"{errors} errors")
    return done


# ----------------------------------------------------------------- output

def round_key(rec: dict):
    """Rounds are '95' in rated play but can be 'QF'/'SF' in the knockout."""
    raw = rec["tags"].get("Round", "")
    m = re.search(r"\d+", raw)
    return (0, int(m.group())) if m else (1, 0)


def write_outputs(records: list[dict]):
    records.sort(key=round_key)

    with (OUT / "all_games.pgn").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r["pgn"].rstrip() + "\n\n")

    with (OUT / "games_index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["game_id", "round", "white", "black", "result",
                    "termination", "date", "opening"])
        for r in records:
            t = r["tags"]
            w.writerow([r["game_id"], t.get("Round", ""), t.get("White", ""),
                        t.get("Black", ""), t.get("Result", ""),
                        t.get("Termination", ""), t.get("Date", ""),
                        r["opening"]])

    counts = Counter(r["opening"] for r in records)
    with (OUT / "opening_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["opening", "count"])
        for name, n in counts.most_common():
            w.writerow([name, n])

    print(f"\nwrote {len(records)} games, {len(counts)} distinct openings")
    for name, n in counts.most_common(15):
        print(f"  {n:6d}  {name}")


async def main():
    OUT.mkdir(exist_ok=True)
    PGN_DIR.mkdir(exist_ok=True)

    limits = httpx.Limits(max_connections=max(GAME_CONCURRENCY, TEAM_CONCURRENCY))
    async with httpx.AsyncClient(headers={"User-Agent": UA}, limits=limits,
                                 timeout=45, follow_redirects=True) as client:
        team_ids = await get_team_ids(client)
        games = await get_all_games(client, team_ids)

    records = await phase3(games)
    write_outputs(records)


if __name__ == "__main__":
    asyncio.run(main())
