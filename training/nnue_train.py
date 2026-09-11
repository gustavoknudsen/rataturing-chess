"""Train and quantise a (768 -> L1)x2 -> 1 NNUE. Run with the CUDA venv:

    .venv_nnue\\Scripts\\python.exe nnue_train.py <data_dir> <out_dir> [L1] [epochs]

Follows the configuration the bullet trainer and nnue-pytorch converge on; every
constant here has a primary source behind it and the reasoning is in
docs/TEST_PLAN.md. The parts that are easy to get quietly wrong, and therefore
carry the most comment, are the perspective ordering, the sign convention, and
the quantisation.

Feature indices are stored by nnue_data.py as

    index = 384 * colour + 64 * piece_type + square,   square 0 = a8

which is exactly the white-perspective formula. The black perspective flips the
board vertically and swaps the colour plane:

    black_index = 384 * (1 - colour) + 64 * piece_type + (square ^ 56)

`square ^ 56` is a vertical flip in either square convention because it inverts
the rank bits, so this transfers unchanged from sources that use square 0 = a1.
What matters is only that training and inference agree, and inference reads the
same table.
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

PAD = 65535

# King buckets. Without them every position shares one 768-feature table, so
# the net has to average over all king placements; bucketing lets it learn
# king-relative structure, which is most of what a hand-crafted evaluation
# spends its king-safety terms on.
#
# Buckets are the cheapest capacity we have. They change neither the number of
# active features (~32) nor the accumulator width, so only the weight table
# grows: measured on the engine, a node costs 1414 ns at 4 buckets and 1430 ns
# at 32. Width is the expensive kind - L1=1024 costs 1.38x per node against
# L1=512. **Prefer buckets over width.**
#
# At 32 this is the standard HalfKAv2_hm feature set: 32 king buckets
# x 11 piece planes x 64 squares = 22,528 features per perspective; ours is
# 32 x 12 x 64 = 24,576, the extra plane being the own king, which a 32-bucket
# index already determines and which therefore acts as a per-bucket bias.
#
# The real limit is data, not speed: buckets split the corpus N ways, so each
# bucket's weights see 1/N of the positions. At 32 buckets, 1.74B positions is
# 54M per bucket against the shipped net's 70M - hence the sweep over 4/16/32
# rather than jumping straight to the largest.
#
# The 50 MB unzipped cap binds at the top: 32 buckets is 25.2 MB at L1=512 and
# 37.7 MB at L1=768, but 50.3 MB at L1=1024, which does not fit.
NUM_BUCKETS = int(os.environ.get("BTC_KING_BUCKETS", "4"))


def _bucket_table(count):
    """King square (0 = a8, perspective orientation) -> bucket.

    With horizontal mirroring the king is always canonicalised onto files a-d,
    so the bucket is simply its canonical file and four buckets cover the board.
    Without mirroring, files e-h would need four more buckets to say the same
    thing, splitting the data across mirror-image positions that are strategically
    identical - which is exactly the waste mirroring exists to remove.

    Entry `square` is read *after* canonicalisation, so only files 0-3 are ever
    looked up; the rest are filled consistently so a stale index cannot silently
    read a wrong bucket.

    Beyond four, rank resolution is added on top of the file, so the counts form
    a hierarchy: 4 splits on file alone, 8 adds the board half, 16 the quarter,
    32 the exact canonical square - which is what the HalfKP-style feature sets
    every strong engine uses amount to. **A bucket refines the previous level
    rather than replacing it**, so the 4-bucket networks already measured stay
    directly comparable to the wider ones.

    This is the cheapest capacity available to us: buckets change neither the
    number of active features (~32) nor the accumulator width, so the per-node
    cost is flat - measured 1414 ns at 4 buckets against 1430 ns at 32, while
    the table grows 3.1 MB to 25.2 MB. Width, by contrast, is paid on every
    node. The limit is the 50 MB unzipped cap: 32 buckets fits at L1=512
    (25.2 MB) and does not at L1=1024."""
    table = np.zeros(64, dtype=np.int32)
    if count == 1:
        return table
    power_of_two = count in (2, 4, 8, 16, 32)
    for square in range(64):
        file_index = square % 8
        if file_index >= 4:
            file_index = 7 - file_index
        rank = square // 8
        if count <= 4:
            table[square] = file_index
        elif power_of_two:
            # count // 4 rank groups spread over 8 ranks:
            # 8 -> rank half, 16 -> rank quarter, 32 -> exact rank
            divisor = 32 // count
            table[square] = (rank // divisor) * 4 + file_index
        else:
            # Any other count: split the 32 canonical squares as evenly as
            # they divide. Some buckets then cover two squares and some one,
            # which costs a little balance but unlocks the counts between the
            # powers of two - and those are what let a given L1 spend the whole
            # 50 MB budget. 24 buckets at L1=1024 is 37.7 MB, where 16 would
            # waste half of it and 32 would not fit.
            #
            # This does not refine the power-of-two levels, so a net built this
            # way is not on the same hierarchy as the 4/8/16/32 ones.
            table[square] = (rank * 4 + file_index) * count // 32
    return table


# WDL blend. The training target is
#
#     lambda * sigmoid(cp / SCALE) + (1 - lambda) * (result + 1) / 2
#
# lambda = 1.0 is pure evaluation, which is all the Lichess evaluations database
# can support because it has no game results. Binpack data carries the outcome
# of the game the position came from, and blending it in is the single
# largest known gain in NNUE training practice - tcheran measured +53.33 +-15.20
# and then a further +27.05 +-9.85 from raising the WDL proportion.
#
# The reason it works: an evaluation says how good a position looks to a search,
# while the result says how often it is actually converted. They differ most in
# exactly the positions that decide games - drawish endings a search scores as
# +1.5, sharp positions a search scores as equal. Defaults to pure eval so that
# eval-only datasets behave as before.
WDL_LAMBDA = float(os.environ.get("BTC_WDL_LAMBDA", "1.0"))

# Output buckets, selected by piece count. One output layer has to express a
# single mapping from accumulator to score across every phase of the game, but
# a pawn-up rook ending and a pawn-up middlegame are not worth the same number
# of centipawns. Eight separate output vectors let the net say so.
#
# **This is free at inference.** The layer is a dot product of 2*L1 terms
# either way; a bucket only changes which weight vector is read. Common practice is
# eight, keyed the same way.
OUT_BUCKETS = int(os.environ.get("BTC_OUT_BUCKETS", "1"))

# Feature factorization. Alongside the bucketed table, train a shared unbucketed
# 768-row table; the effective weight is bucketed + shared. The shared table
# learns what a piece on a square is worth *in general*, so each bucket's rows
# only have to learn the deviation, and a rare bucket's weights start from a
# real prior instead of noise.
#
# This targets the actual binding constraint. At 23.6M parameters and ~7B
# positions we are at roughly 300 positions per parameter, and the entire model
# is the feature transformer - all the capacity sits in the sparsest, least
# frequently trained part of the network. Bucketing splits the corpus N ways;
# factorization gives every position back a stake in a shared table.
#
# **Free at inference.** The shared table is folded into the bucketed one when
# the net is exported, so net.npz, its size and btc_nnue.py are unchanged.
FACTORIZE = os.environ.get("BTC_FACTORIZE", "1") == "1"


def out_bucket_of(piece_count, out_buckets):
    """Piece count (2..32) -> output bucket. Must match btc_nnue._out_bucket.

    Takes a tensor or an int, so the trainer and the tests share one definition
    of the mapping rather than two that can drift apart."""
    if out_buckets <= 1:
        return piece_count * 0
    idx = (piece_count - 1) * out_buckets // 32
    if hasattr(idx, "clamp"):
        return idx.clamp(0, out_buckets - 1)
    return max(0, min(out_buckets - 1, idx))


QA = 255            # feature transformer / accumulator scale
QB = 64             # output layer scale
SCALE = 400         # network float output -> centipawns, and the loss sigmoid
CLIP = 1.98         # weight clip; keeps the int16 accumulator from overflowing
BATCH = 16384
# Peak learning rate. Overridable because a continuation run restarts the
# cosine from a lower peak than a run starting at random weights: the network
# is already near a minimum and a full-height restart would undo it. 1e-4 to
# 3e-4 is the usual choice for a warm restart from 1e-3.
LR = float(os.environ.get("BTC_LR", "1e-3"))
FINAL_LR = 1e-3 * (0.3 ** 5)
# Zero, deliberately. AdamW's decoupled decay multiplies every parameter by
# (1 - lr * wd) on **every step**, whether or not that parameter saw a gradient
# in the batch, and nn.EmbeddingBag holds the whole table as one Parameter. Over
# a 427,000-step run at a mean lr near 5e-4, a row that never receives a gradient
# ends at exp(-427000 * 5e-4 * 0.01) = 12% of where it started.
#
# The rows that rarely receive gradients are exactly the ones king buckets
# create - a rook on a8 with the enemy king in a rare bucket appears in perhaps
# one position in 100,000 - so the decay preferentially destroys the capacity
# the entire parameter budget was spent on, and it does it invisibly because the
# common features stay healthy and validation loss looks plausible.
#
# Measured on the shipped 512x32 net, trained at 0.01: the correlation between
# log(king-bucket frequency) and mean |weight| is **+0.588**, and the rarest
# buckets carry 30% smaller weights than the commonest across a 340x frequency
# range. Common practice is 0.0 for the feature transformer, relying on
# weight clipping alone, which CLIP = 1.98 already provides.
WEIGHT_DECAY = float(os.environ.get("BTC_WEIGHT_DECAY", "0.0"))


def _fixes_line():
    """The two settings that silently decide whether a run is any good.

    Both default to the value we want, so a run can be correct without
    anyone having set them - which is exactly why the log should say so
    rather than leaving it to be inferred from the defaults."""
    return (f"factorize={'on' if FACTORIZE else 'OFF'}, "
            f"weight_decay={WEIGHT_DECAY:g}")
# Path to a best.pt from an earlier run. Training continues from those weights
# instead of random ones, which is how a net that was still improving when its
# position budget ran out gets more data rather than being thrown away.
RESUME = os.environ.get("BTC_RESUME", "")
VAL_FRACTION = 0.01


class Nnue(nn.Module):
    """(768 -> L1) x 2 -> 1 with SCReLU.

    The feature transformer is an EmbeddingBag in sum mode, which *is* the
    accumulator: its weight matrix is (768, L1) row-major, exactly the layout
    the numba inference reads, so one row is one contiguous run of L1 weights."""

    def __init__(self, l1, buckets=1, out_buckets=1):
        super().__init__()
        self.l1 = l1
        self.buckets = buckets
        self.out_buckets = out_buckets
        rows = buckets * 768
        # One extra row: a permanently-zero padding row, so a short piece list
        # sums to the same accumulator as a full one. padding_idx pins it at
        # zero and excludes it from gradients, so weight decay cannot drift it
        # away from zero over 87k steps.
        self.ft = nn.EmbeddingBag(rows + 1, l1, mode="sum", padding_idx=rows)
        self.ft_bias = nn.Parameter(torch.zeros(l1))
        self.out = nn.Linear(2 * l1, out_buckets)
        nn.init.normal_(self.ft.weight, std=0.01)
        # Shared unbucketed table, 768 rows plus a padding row. Starts at zero
        # so the model begins identical to the unfactorized one.
        self.factorize = FACTORIZE and buckets > 1
        if self.factorize:
            self.virtual = nn.EmbeddingBag(769, l1, mode="sum", padding_idx=768)
            nn.init.zeros_(self.virtual.weight)

    def folded_ft(self):
        """Bucketed weights with the shared table folded in.

        Used for export and for checkpoints. Does **not** mutate the model, so
        training continues factorized after a mid-run save - a destructive
        coalesce at the first checkpoint would silently switch factorization off
        for the rest of the run."""
        w = self.ft.weight.detach().clone()
        if self.factorize:
            rows = self.buckets * 768
            w[:rows] += self.virtual.weight.detach()[:768].repeat(self.buckets, 1)
        return w

    def forward(self, white_idx, black_idx, stm, out_bucket=None,
                white_v=None, black_v=None):
        acc_w = self.ft(white_idx) + self.ft_bias
        acc_b = self.ft(black_idx) + self.ft_bias
        if self.factorize and white_v is not None:
            acc_w = acc_w + self.virtual(white_v)
            acc_b = acc_b + self.virtual(black_v)
        # Side to move FIRST. This is what lets the net learn tempo, and it
        # makes the output side-to-move relative, which is what negamax wants.
        # stm is 1.0 when white is to move.
        stm = stm.unsqueeze(1)
        hidden = stm * torch.cat([acc_w, acc_b], dim=1) \
            + (1.0 - stm) * torch.cat([acc_b, acc_w], dim=1)
        hidden = torch.clamp(hidden, 0.0, 1.0) ** 2      # SCReLU
        scores = self.out(hidden)
        if self.out_buckets == 1:
            return scores.squeeze(1)
        # Every bucket's score is computed and one is selected. Wasteful in
        # training and irrelevant there (the layer is tiny next to the
        # embedding); inference reads only the selected vector.
        return scores.gather(1, out_bucket.unsqueeze(1)).squeeze(1)


def _king_square(feats, pad, base):
    """Square of the king whose feature plane starts at `base`.

    The data has exactly one king per side per row - test_nnue and the
    extraction verifier both assert it - so masking to that plane and summing
    recovers the single index without a search."""
    mask = (feats >= base) & (feats < base + 64) & ~pad
    return (feats * mask).sum(dim=1) - base


def build_perspectives(feats, device, table, buckets):
    """(white_idx, black_idx) index tensors from the stored white-perspective
    indices, offset by each perspective's own king bucket.

    Each perspective buckets on *its own* king, the black one after the
    `square ^ 56` flip. That is what keeps the colour-mirror symmetry exact:
    mirroring swaps which king each perspective sees, so the two accumulators
    swap and the output is unchanged."""
    feats = feats.to(device=device, dtype=torch.long)
    pad = feats == PAD
    colour = torch.div(feats, 384, rounding_mode="floor")
    piece = torch.div(feats % 384, 64, rounding_mode="floor")
    square = feats % 64

    if buckets == 1:
        white = feats.clone()
        black = 384 * (1 - colour) + 64 * piece + (square ^ 56)
        white[pad] = 768
        black[pad] = 768
        return white, black, white, black

    white_king = _king_square(feats, pad, 5 * 64)
    black_king = _king_square(feats, pad, 11 * 64) ^ 56
    # Mirror each perspective independently, onto files a-d, keyed on that
    # perspective's own king. `^ 7` inverts the file bits.
    white_flip = ((white_king % 8) >= 4).long().unsqueeze(1) * 7
    black_flip = ((black_king % 8) >= 4).long().unsqueeze(1) * 7
    white_square = square ^ white_flip
    black_square = (square ^ 56) ^ black_flip

    white = 384 * colour + 64 * piece + white_square
    black = 384 * (1 - colour) + 64 * piece + black_square
    white = white + table[white_king].unsqueeze(1) * 768
    black = black + table[black_king].unsqueeze(1) * 768

    # The unbucketed indices are what the shared factorization table reads:
    # the same feature without the king-bucket offset, with its own pad row.
    white_v = 384 * colour + 64 * piece + white_square
    black_v = 384 * (1 - colour) + 64 * piece + black_square
    white_v[pad] = 768
    black_v[pad] = 768

    rows = buckets * 768
    white[pad] = rows
    black[pad] = rows
    return white, black, white_v, black_v


def _step(model, opt, sched, arrays, device, table, train):
    """One batch, from four numpy arrays to a scalar loss.

    Both the memory-mapped path and the streaming path funnel through here, so
    the target construction, the sign conventions and the weight clamp cannot
    drift between a net trained one way and a net trained the other. That
    matters because the two are compared against each other."""
    raw_feats, raw_stm, raw_score, raw_result = arrays
    batch_feats = torch.from_numpy(np.asarray(raw_feats).astype(np.int64))
    batch_stm = torch.from_numpy(
        np.asarray(raw_stm).astype(np.int64)).to(device).float()
    batch_score = torch.from_numpy(
        np.asarray(raw_score).astype(np.float32)).to(device)
    batch_result = None
    if raw_result is not None:
        batch_result = torch.from_numpy(
            np.asarray(raw_result).astype(np.float32)).to(device)
    white, black, white_v, black_v = build_perspectives(
        batch_feats, device, table, model.buckets)
    # stored stm is 0 for white to move; the model wants 1.0 for white
    white_to_move = 1.0 - batch_stm
    target = torch.sigmoid(batch_score / SCALE)
    if batch_result is not None and WDL_LAMBDA < 1.0:
        # result is -1/0/+1 side-to-move relative, the same convention as the
        # score, so it maps to a win probability with (r + 1) / 2.
        outcome = (batch_result + 1.0) * 0.5
        target = WDL_LAMBDA * target + (1.0 - WDL_LAMBDA) * outcome
    bucket = None
    if model.out_buckets > 1:
        # Piece count is the number of non-padding features, so it needs no
        # extra column in the data.
        counts = (batch_feats != PAD).sum(dim=1).to(device)
        bucket = out_bucket_of(counts, model.out_buckets)
    with torch.set_grad_enabled(train):
        pred = torch.sigmoid(model(white, black, white_to_move, bucket,
                                   white_v, black_v))
        loss = ((pred - target) ** 2).mean()
    if train:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        # Hard clamp after every step, not a penalty in the loss. This is the
        # entire reason the int16 accumulator cannot overflow:
        # 505 (bias) + 32 * 505 = 16,665 against a 32,767 ceiling.
        with torch.no_grad():
            params = [model.ft.weight, model.ft_bias,
                      model.out.weight, model.out.bias]
            if model.factorize:
                # The shared table is added to the bucketed one, so the sum is
                # what must stay inside the clip. Half each keeps the sum in
                # range without needing a joint projection.
                params.append(model.virtual.weight)
                for q in (model.ft.weight, model.virtual.weight):
                    q.clamp_(-CLIP * 0.5, CLIP * 0.5)
                for q in (model.ft_bias, model.out.weight, model.out.bias):
                    q.clamp_(-CLIP, CLIP)
            else:
                for q in params:
                    q.clamp_(-CLIP, CLIP)
    with torch.no_grad():
        mae = float((pred - target).abs().sum())
        scalar = float(loss.detach())
    return scalar, mae


class _Progress:
    """Percent, throughput and running loss during an epoch.

    A full pass over 2B positions is hours long; without this a run is silent
    until it ends, so a stall, a thermal throttle or a wrong row count is only
    discovered after the time has already been spent."""

    def __init__(self, total, every):
        self.total = total
        self.every = max(1, every)
        self.started = time.time()

    def update(self, step, count, total_loss):
        """Print a progress line. Returns True when it did, so the caller can
        hang periodic work off the same cadence."""
        if step % self.every:
            return False
        rate = count / max(time.time() - self.started, 1e-9)
        pct = f"{100 * count / self.total:5.1f}%" if self.total else "  ---"
        print(f"    {pct}  {count / 1e6:8.1f}M seen  {rate / 1e6:5.3f} Mpos/s  "
              f"loss {total_loss / max(count, 1):.6f}", flush=True)
        return True


def run_epoch(model, opt, sched, data, device, train, table):
    """One pass over memory-mapped arrays.

    **Contiguous batches, shuffled batch order** - not random indices.

    Indexing a memory-mapped file with 16384 scattered row numbers is fine while
    the file fits in page cache and collapses into random I/O when it does not.
    feats is 64 bytes per position, so 300M positions is 19 GB and random access
    would make training I/O-bound rather than GPU-bound. Reading each batch as
    one contiguous slice keeps the access sequential at any file size, which is
    what makes scaling the dataset possible at all.

    Randomness is not lost: binpack_prep.py shuffles the rows on disk, so a
    contiguous slice is already a random sample of the corpus, and shuffling the
    *order* of batches each epoch stops the model seeing them in a fixed
    sequence. This is what bullet does for the same reason."""
    feats, stm, score, result = data
    total = 0.0
    count = 0
    mae = 0.0
    model.train(train)
    starts = list(range(0, len(score), BATCH))
    if train:
        np.random.default_rng().shuffle(starts)
    progress = _Progress(len(score), len(starts) // 20)
    for step, start in enumerate(starts):
        stop = min(start + BATCH, len(score))
        size = stop - start
        # Partial final batch included. Dropping it silently made the
        # validation loop skip entirely whenever the held-out set was smaller
        # than a batch, and report a loss of exactly 0.
        if size < 2:
            continue
        arrays = (feats[start:stop], stm[start:stop], score[start:stop],
                  None if result is None else result[start:stop])
        loss, batch_mae = _step(model, opt, sched, arrays, device, table, train)
        total += loss * size
        count += size
        mae += batch_mae
        if train:
            progress.update(step, count, total)
    return total / max(count, 1), mae / max(count, 1)


def run_epoch_stream(model, opt, sched, stream, device, train, table,
                     expected=0, checkpoint=None):
    """One pass over binpack files, with nothing materialised on disk.

    `expected` is only used to render a percentage and to space the progress
    reports; the stream itself knows how many positions it will produce only
    once it has produced them.

    `checkpoint` is called at each progress report. A streamed epoch over
    billions of positions runs for hours, and a hosted notebook is killed at a
    fixed wall-clock limit - so an epoch-end-only save means a run that is
    stopped at 99% produces *nothing*. This writes the current weights out
    every time it reports, which costs one 6 MB file write per twentieth of an
    epoch and turns a killed session into a usable net."""
    total = 0.0
    count = 0
    mae = 0.0
    model.train(train)
    progress = _Progress(expected, max(1, expected // BATCH // 20) if expected
                         else 200)
    for step, arrays in enumerate(stream.batches(BATCH)):
        loss, batch_mae = _step(model, opt, sched, arrays, device, table, train)
        size = len(arrays[2])
        total += loss * size
        count += size
        mae += batch_mae
        if train and progress.update(step, count, total) and checkpoint:
            checkpoint()
    return total / max(count, 1), mae / max(count, 1), count


def quantise(model, out_dir, l1, table, name="net.npz"):
    """Export int16 weights, asserting nothing overflows rather than wrapping."""
    rows = model.buckets * 768
    ft_w = model.folded_ft().cpu().numpy()[:rows]
    ft_b = model.ft_bias.detach().cpu().numpy()
    # (out_buckets, 2 * l1) and (out_buckets,), even when there is one bucket,
    # so the engine reads one shape rather than two.
    out_w = model.out.weight.detach().cpu().numpy()
    out_b = model.out.bias.detach().cpu().numpy()

    ft_w_q = np.round(ft_w * QA)
    ft_b_q = np.round(ft_b * QA)
    out_w_q = np.round(out_w * QB)
    out_b_q = np.round(out_b * QA * QB)
    assert out_w_q.shape == (model.out_buckets, 2 * l1), out_w_q.shape

    limit = np.iinfo(np.int16).max
    assert np.abs(ft_w_q).max() <= limit, "feature weights overflow int16"
    assert np.abs(ft_b_q).max() <= limit, "feature bias overflows int16"
    assert np.abs(out_w_q).max() <= limit, "output weights overflow int16"
    worst = abs(ft_b_q).max() + 32 * abs(ft_w_q).max()
    assert worst <= limit, f"accumulator can reach {worst}, over int16"

    np.savez(os.path.join(out_dir, name),
             ft_w=ft_w_q.astype(np.int16), ft_b=ft_b_q.astype(np.int16),
             out_w=out_w_q.astype(np.int16),
             out_b=out_b_q.astype(np.int32), l1=np.int32(l1),
             out_buckets=np.int32(model.out_buckets),
             qa=np.int32(QA), qb=np.int32(QB), scale=np.int32(SCALE),
             buckets=np.int32(model.buckets),
             bucket_table=table.cpu().numpy().astype(np.int32))
    if name == "net.npz":
        print(f"quantised: worst-case accumulator {int(worst)} of {limit}")


def _resume_from_npz(model, path, device):
    """Undo quantise() into the live model.

    The fallback path. A run killed mid-epoch leaves only net_latest.npz, which
    is int16, so the float weights come back rounded to 1/QA - about 0.2% of
    the +-1.98 clip range. That is recoverable; what is gone either way is the
    optimiser state, which AdamW rebuilds within a few hundred steps of a
    multi-billion-position run."""
    data = np.load(path)
    assert int(data["l1"]) == model.l1, "L1 mismatch"
    assert int(data["buckets"]) == model.buckets, "king bucket mismatch"
    rows = model.buckets * 768
    with torch.no_grad():
        model.ft.weight[:rows] = torch.from_numpy(
            data["ft_w"].astype(np.float32) / QA).to(device)
        model.ft_bias.copy_(torch.from_numpy(
            data["ft_b"].astype(np.float32) / QA).to(device))
        out_w = np.atleast_2d(data["out_w"]).astype(np.float32) / QB
        model.out.weight.copy_(torch.from_numpy(out_w).to(device))
        model.out.bias.copy_(torch.from_numpy(
            np.atleast_1d(data["out_b"]).astype(np.float32) / (QA * QB)
        ).to(device))


def _build(l1, device):
    """Model, optimiser and the zeroed padding row, shared by both modes."""
    model = Nnue(l1, NUM_BUCKETS, OUT_BUCKETS).to(device)
    if RESUME:
        if RESUME.endswith(".npz"):
            _resume_from_npz(model, RESUME, device)
        else:
            # strict=False: checkpoints are saved folded and carry no virtual
            # table, while a resumed run may be factorized again (its shared
            # table starts at zero, so the model is identical at step 0).
            missing, unexpected = model.load_state_dict(
                torch.load(RESUME, map_location=device), strict=False)
            unexpected = [k for k in unexpected]
            assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
            for key in missing:
                assert key.startswith("virtual."), f"missing key {key}"

        print(f"resumed from {RESUME}, peak LR {LR:g}")
    # pad row 768 must stay zero and out of the optimiser's way
    with torch.no_grad():
        model.ft.weight[NUM_BUCKETS * 768:].zero_()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=WEIGHT_DECAY)
    return model, opt


def _record(out_dir, l1, table, model, epoch, epochs, tr, va, va_mae, best):
    """Checkpoint on improvement and print the epoch line."""
    flag = ""
    if va < best:
        best = va
        # Folded, always. A checkpoint carrying live virtual weights, resumed
        # into a model built without factorization, silently drops everything
        # the shared table learned - and the loss would look merely mediocre
        # rather than broken.
        state = {k: v for k, v in model.state_dict().items()
                 if not k.startswith("virtual.")}
        state["ft.weight"] = model.folded_ft()
        torch.save(state, os.path.join(out_dir, "best.pt"))
        quantise(model, out_dir, l1, table)
        flag = "  *"
    print(f"epoch {epoch + 1:3d}/{epochs}  train {tr:.6f}  "
          f"val {va:.6f}  val_sig_mae {va_mae:.4f}{flag}", flush=True)
    return best


def _binpacks(path):
    """The binpack files named by `path`, which may be one file or a directory."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    found = sorted(os.path.join(path, f) for f in os.listdir(path)
                   if f.endswith(".binpack"))
    return found


def train_streamed(files, out_dir, l1, epochs):
    """Train directly from binpacks. Nothing but the binpacks touches disk.

    Validation needs its own files rather than a tail slice: the stream shuffles
    within a buffer, not globally, so a tail slice would be a systematically
    later - and therefore different - sample of the corpus."""
    from nnue_stream import BinpackStream

    # .exe on Windows, bare name on the Linux notebooks. The default was
    # the Linux one, so a local run that forgot BTC_STREAM_EXE died at
    # startup with FileNotFoundError.
    default_exe = "tools/binpack_stream"
    if os.name == "nt" and os.path.exists(default_exe + ".exe"):
        default_exe += ".exe"
    exe = os.environ.get("BTC_STREAM_EXE", default_exe)
    val_files = _binpacks(os.environ.get("BTC_VAL_BINPACK", ""))
    if not val_files:
        if len(files) < 2:
            raise SystemExit(
                "streaming needs a validation binpack: set BTC_VAL_BINPACK, or "
                "pass a directory holding more than one .binpack")
        val_files, files = files[-1:], files[:-1]
    # The validation file usually lives in the same directory as the training
    # files, so passing that directory would otherwise train on it and make
    # every validation number meaningless without failing.
    held = {os.path.abspath(f) for f in val_files}
    files = [f for f in files if os.path.abspath(f) not in held]
    if not files:
        raise SystemExit("no training files left after holding out validation")
    # Only used for the progress percentage and the cosine schedule length; the
    # stream cannot know its own length until it has finished.
    expected = int(os.environ.get("BTC_STREAM_ROWS", "0"))
    fixes = _fixes_line()
    val_rows = int(os.environ.get("BTC_STREAM_VAL_ROWS", "4000000"))
    wdl = "eval only" if WDL_LAMBDA >= 1.0 else f"WDL lambda {WDL_LAMBDA}"
    print(f"streaming {len(files)} binpack(s), validating on "
          f"{len(val_files)}, L1={l1}, {NUM_BUCKETS} king buckets, "
          f"{OUT_BUCKETS} output buckets, {epochs} epochs, {wdl}, "
          f"{fixes}")
    for path in files + val_files:
        print(f"  {os.path.getsize(path) / 1e9:7.2f} GB  {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = torch.from_numpy(_bucket_table(NUM_BUCKETS)).to(device)
    model, opt = _build(l1, device)
    steps = max(1, (expected or 500_000_000) // BATCH) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=FINAL_LR)

    best = float("inf")
    for epoch in range(epochs):
        train_stream = BinpackStream(exe, files, seed=epoch,
                                     max_rows=expected)

        def save_latest():
            """Mid-epoch insurance against a hosted session being killed."""
            quantise(model, out_dir, l1, table, "net_latest.npz")

        tr, _, seen = run_epoch_stream(model, opt, sched, train_stream, device,
                                       True, table, expected, save_latest)
        # A stream that ends early is the failure mode with no symptom: the
        # epoch simply finishes, the cosine schedule is left partway down with
        # the learning rate still high, and the net looks merely mediocre. One
        # run ended after the first of two files at 48.6% of its budget and the
        # only sign was a validation loss two points worse than a smaller net.
        if expected and seen < expected * 0.98:
            print(f"  *** WARNING: stream ended at {100 * seen / expected:.1f}%"
                  f" of the {expected:,} row budget ({seen:,} seen). The cosine"
                  f" schedule did NOT complete; this net is undertrained.",
                  flush=True)
        val_stream = BinpackStream(exe, val_files, seed=0,
                                   max_rows=val_rows)
        va, va_mae, _ = run_epoch_stream(model, opt, sched, val_stream, device,
                                         False, table)
        print(f"  epoch {epoch + 1} streamed {seen:,} training positions")
        best = _record(out_dir, l1, table, model, epoch, epochs, tr, va,
                       va_mae, best)


def train_mapped(data_dir, out_dir, l1, epochs):
    """Train from the .npy arrays binpack_prep.py writes."""
    feats = np.load(os.path.join(data_dir, "feats.npy"), mmap_mode="r")
    stm = np.load(os.path.join(data_dir, "stm.npy"), mmap_mode="r")
    score = np.load(os.path.join(data_dir, "score.npy"), mmap_mode="r")
    result_path = os.path.join(data_dir, "result.npy")
    result = np.load(result_path, mmap_mode="r")         if os.path.exists(result_path) else None
    n = len(score)
    # Ablations (bucket count, WDL lambda, L1) only need enough data to rank
    # configurations, not the whole corpus. Capping rows makes each comparison
    # run in a fraction of the time; the winning configuration is then retrained
    # on everything. The data is shuffled on disk, so a prefix is a random
    # sample rather than a biased one.
    cap = int(os.environ.get("BTC_MAX_ROWS", "0"))
    if cap and cap < n:
        n = cap
        feats, stm, score = feats[:n], stm[:n], score[:n]
        if result is not None:
            result = result[:n]
    # The data is already shuffled on disk, so a tail slice is a random sample;
    # taking it explicitly anyway so this does not depend on that holding.
    split = n - int(n * VAL_FRACTION)
    wdl = "eval only" if result is None or WDL_LAMBDA >= 1.0         else f"WDL lambda {WDL_LAMBDA}"
    print(f"{n:,} positions, {n - split:,} held out, L1={l1}, "
          f"{NUM_BUCKETS} king buckets, {OUT_BUCKETS} output buckets, "
          f"{epochs} epochs, {wdl}, {_fixes_line()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = torch.from_numpy(_bucket_table(NUM_BUCKETS)).to(device)
    model, opt = _build(l1, device)
    steps = max(1, (split // BATCH)) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps, eta_min=FINAL_LR)

    train_data = (feats[:split], stm[:split], score[:split],
                  None if result is None else result[:split])
    val_data = (feats[split:], stm[split:], score[split:],
                None if result is None else result[split:])
    best = float("inf")
    for epoch in range(epochs):
        tr, _ = run_epoch(model, opt, sched, train_data, device, True, table)
        va, va_mae = run_epoch(model, opt, sched, val_data, device, False,
                               table)
        best = _record(out_dir, l1, table, model, epoch, epochs, tr, va,
                       va_mae, best)


def main():
    data, out_dir = sys.argv[1], sys.argv[2]
    l1 = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    os.makedirs(out_dir, exist_ok=True)
    files = _binpacks(data)
    if files:
        train_streamed(files, out_dir, l1, epochs)
    else:
        train_mapped(data, out_dir, l1, epochs)


if __name__ == "__main__":
    main()
