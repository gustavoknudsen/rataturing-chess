"""Game state tracking across get_move calls.

The platform sends only a FEN on our turn. The process lives for one game, so
module state persists between our moves. From the position after our last move,
the opponent's reply is recoverable by matching the new FEN against the children
of that position. The resulting key list feeds repetition detection in search
(the referee auto-claims threefold and fifty-move draws, counted from the
game's first FEN).
"""

import numpy as np

import btc_core as core


class GameTracker:
    """Maintains the full position-key history of the current game."""

    def __init__(self):
        self.bb, self.st = core.new_board()
        self._scratch_bb, self._scratch_st = core.new_board()
        self._undo_bb, self._undo_st, self._mls = core.new_stacks()
        self.keys = np.zeros(1024, dtype=np.uint64)
        self.key_count = 0
        self.expected_bb = None
        self.expected_st = None
        self.resets = 0

    def update(self, fen):
        """Parse the incoming FEN and extend history. Returns the opponent's
        move as a packed int, or None (first call, no move made, or a desync
        reset)."""
        core.parse_fen(fen, self.bb, self.st)
        opp_move = None
        unchanged = False
        if self.expected_bb is not None:
            unchanged = self._is_expected()
            if not unchanged:
                opp_move = self._match_opponent_move()
        self.expected_bb = None
        self.expected_st = None
        if unchanged:
            # Asked to move again from the position our own last move
            # produced. push_our_move already appended this key, so appending
            # again would double-count it, and treating it as a desync would
            # throw away the whole repetition history. This is the
            # play-both-sides case. A referee RETRY, which re-sends the
            # position from before our move, still resets: that is a different
            # shape and is not handled here.
            return None
        if opp_move is None:
            if self.key_count > 0:
                self.resets += 1
                print(f"tracker: history reset ({self.resets})", flush=True)
            self.key_count = 0
        self._append_key(self.bb[core.HASH])
        return opp_move

    def _is_expected(self):
        """The incoming position is exactly the one our last move produced."""
        return bool(self.bb[core.HASH] == self.expected_bb[core.HASH]
                    and self._same_position(self.expected_bb,
                                            self.expected_st))

    def push_our_move(self, mv):
        """Record our chosen move; the resulting position joins the history
        and becomes the anchor for diffing the opponent's reply."""
        self._scratch_bb[:] = self.bb
        self._scratch_st[:] = self.st
        ok = core.make_move(self._scratch_bb, self._scratch_st,
                            self._undo_bb, self._undo_st, 0, mv)
        if not ok:
            self.expected_bb = None
            self.expected_st = None
            return
        self.expected_bb = self._scratch_bb.copy()
        self.expected_st = self._scratch_st.copy()
        self._append_key(self.expected_bb[core.HASH])

    def history(self):
        """(keys array, count) for the search's repetition table."""
        return self.keys, self.key_count

    def _append_key(self, key):
        if self.key_count < len(self.keys):
            self.keys[self.key_count] = key
            self.key_count += 1

    def _match_opponent_move(self):
        exp_bb, exp_st = self.expected_bb, self.expected_st
        # update() has already ruled out the unchanged case, so any position
        # reachable here is genuinely a different one.
        for mv in core.legal_moves(exp_bb, exp_st):
            self._scratch_bb[:] = exp_bb
            self._scratch_st[:] = exp_st
            core.make_move(self._scratch_bb, self._scratch_st,
                           self._undo_bb, self._undo_st, 0, mv)
            if self._scratch_bb[core.HASH] == self.bb[core.HASH] \
                    and self._same_position(self._scratch_bb, self._scratch_st):
                return mv
        return None

    def _same_position(self, other_bb, other_st):
        if not np.array_equal(other_bb[:12], self.bb[:12]):
            return False
        for field in (core.SIDE, core.EP, core.CASTLE):
            if other_st[field] != self.st[field]:
                return False
        return True
