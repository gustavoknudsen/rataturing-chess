"""Polyglot opening book, probed at the root only.

The rules permit a shipped table to answer the opening, and define the opening
as "a position whose move number is 20 or lower"; a table answering a
middlegame position counts as an engine. MAX_BOOK_MOVE is that limit, checked
before any file is read.

Books are searched in priority order and the first usable move wins. The
highest-weighted move is always taken, with no randomness. There is no
out-of-book latch, so play that leaves the book and transposes back into it
uses it again. Any failure returns None and the caller searches.
"""

import os

import chess
import chess.polyglot

# A position past this is a middlegame position under the rules and must never
# be looked up. Do not raise it or add a path around it.
MAX_BOOK_MOVE = 20

BOOK_FILES = ("rataturing.bin", "rataturing_hedge.bin")

# The book stores a score in Polyglot's `learn` field, offset, with 0 meaning
# "no score". A move the book itself scores as clearly losing is more likely a
# build error than a recommendation, so it is declined.
LEARN_OFFSET = 100000
MIN_BOOK_SCORE = -150

_READERS = []
_STATUS = "not loaded"


def _names():
    override = os.environ.get("BTC_BOOK")
    if override:
        return tuple(n.strip() for n in override.split(",") if n.strip())
    return BOOK_FILES


def _path(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def load():
    """Open every book present. Returns a status line for the init log."""
    global _STATUS
    del _READERS[:]
    opened = []
    for name in _names():
        path = _path(name)
        if not os.path.exists(path):
            continue
        try:
            size = os.path.getsize(path)
            # Polyglot records are 16 bytes, so any other size is truncated.
            if size == 0 or size % 16 != 0:
                opened.append(f"{name} rejected ({size} B)")
                continue
            _READERS.append(chess.polyglot.open_reader(path))
            opened.append(f"{name} {size // 16} entries")
        except Exception as exc:
            opened.append(f"{name} unavailable ({exc!r})")
    if not _READERS:
        _STATUS = "no book" if not opened else "no book: " + ", ".join(opened)
    else:
        _STATUS = ", ".join(opened) + f", moves <= {MAX_BOOK_MOVE}"
    return _STATUS


def status():
    return _STATUS


def in_window(board):
    """True while the rules permit a table to answer this position."""
    return board.fullmove_number <= MAX_BOOK_MOVE


def _usable(entry, board):
    if entry.move not in board.legal_moves:
        # A hash collision is rare, but an illegal move loses the game.
        print(f"book: illegal move {entry.move.uci()}", flush=True)
        return False
    if entry.learn and entry.learn - LEARN_OFFSET < MIN_BOOK_SCORE:
        return False
    return True


def probe(board):
    """Book move for `board` as UCI, or None to search instead."""
    if not _READERS or not in_window(board):
        return None
    for reader in _READERS:
        try:
            entry = reader.find(board)
        except IndexError:
            continue
        except Exception as exc:
            print(f"book: probe failed {exc!r}", flush=True)
            continue
        if _usable(entry, board):
            return entry.move.uci()
    return None
