"""Shuffle binpack_convert output into the .npy arrays nnue_train.py reads.

    python binpack_prep.py <raw_dir> <out_dir> [max_rows]

`tools/binpack_convert.exe` emits positions in **game order** - consecutive rows
are consecutive plies of the same game with near-identical evaluations. Training
on that directly gives a 16384-position batch an effective sample size of a few
dozen, which is worth real elo (+17.29 +-8.50 in tcheran's log for the shuffle
alone). So the data has to be shuffled, and at this scale that is the hard part.

**Why a two-pass bucket shuffle rather than anything simpler.** The corpus is
tens of GB, far past RAM:

- A whole-file Fisher-Yates is random access over the entire file. On disk that
  is millions of scattered 64-byte reads and writes; it does not finish.
- A windowed shuffle (swap only within a sliding window) is sequential and fast,
  but it only mixes locally. The file's global ordering survives, so the tail -
  which the trainer takes as its validation set - stays a *systematically
  different* sample from the head. That silently biases every validation number
  we would then use to choose a network.

The bucket shuffle avoids both. Pass one assigns every row to a random bucket,
appending sequentially, so each bucket is already a uniform random sample of the
whole corpus. Pass two shuffles each bucket in RAM. The result is globally
shuffled, both passes are sequential, and peak memory is one bucket.
"""

import os
import sys

import numpy as np

# One bucket is loaded into RAM whole and shuffled there, so it sets peak
# memory. At 69 bytes per row, 24M rows is about 1.7 GB - deliberately
# conservative, because an earlier 36M setting combined with memory-mapped
# output writes froze this machine hard (see _npy_writer).
ROWS_PER_BUCKET = int(os.environ.get("BTC_ROWS_PER_BUCKET", "24000000"))
CHUNK = 4_000_000
WRITE_CHUNK = 2_000_000

RECORD = np.dtype([("f", np.uint16, 32), ("c", np.uint8), ("s", np.uint8),
                   ("v", np.int16), ("r", np.int8)])


def _row_count(raw_dir):
    return os.path.getsize(os.path.join(raw_dir, "counts.raw"))


def _open_inputs(raw_dir, total):
    feats = np.memmap(os.path.join(raw_dir, "feats.raw"), dtype=np.uint16,
                      mode="r", shape=(total, 32))
    counts = np.memmap(os.path.join(raw_dir, "counts.raw"), dtype=np.uint8,
                       mode="r", shape=(total,))
    stm = np.memmap(os.path.join(raw_dir, "stm.raw"), dtype=np.uint8,
                    mode="r", shape=(total,))
    score = np.memmap(os.path.join(raw_dir, "score.raw"), dtype=np.int16,
                      mode="r", shape=(total,))
    result = np.memmap(os.path.join(raw_dir, "result.raw"), dtype=np.int8,
                       mode="r", shape=(total,))
    return feats, counts, stm, score, result


def _book_mask(rows, counts):
    """True for near-book positions, which are wildly over-represented here.

    Leela self-play games all start from the same place, so every game donates
    its opening. Measured on a 300k sample of this corpus: **1.35% of all rows
    are the literal starting position** - about 10.8 million identical copies in
    800M - and 12.9% have full material with both sides still uncastled. Those
    carry almost no evaluation signal and they are most of the 10.3% duplicate
    rate, so they distort training by sheer weight of repetition.

    `nnue_data.py` dropped the equivalent 7.9% from the Lichess data using the
    castling field. Binpack has no castling rights, so the test here is
    positional: full material, with both kings and all four rooks still on their
    original squares. Square 0 = a8, so e1 = 60, a1 = 56, h1 = 63, e8 = 4,
    a8 = 0, h8 = 7; feature index is 64 * piece + square."""
    def occupies(piece, square):
        return (rows == (piece * 64 + square)).any(axis=1)

    white_home = (occupies(5, 60) & occupies(3, 56) & occupies(3, 63))
    black_home = (occupies(11, 4) & occupies(9, 0) & occupies(9, 7))
    return (counts == 32) & white_home & black_home


def scatter(raw_dir, tmp_dir, total, n_buckets, rng):
    """Pass one: append each row to a randomly chosen bucket file."""
    feats, counts, stm, score, result = _open_inputs(raw_dir, total)
    handles = [open(os.path.join(tmp_dir, f"b{i}.bin"), "wb")
               for i in range(n_buckets)]
    sizes = np.zeros(n_buckets, dtype=np.int64)
    dropped = [0]
    for start in range(0, total, CHUNK):
        stop = min(start + CHUNK, total)
        block = np.empty(stop - start, dtype=RECORD)
        block["f"] = feats[start:stop]
        block["c"] = counts[start:stop]
        block["s"] = stm[start:stop]
        block["v"] = score[start:stop]
        block["r"] = result[start:stop]
        book = _book_mask(block["f"], block["c"])
        dropped[0] += int(book.sum())
        block = block[~book]
        which = rng.integers(0, n_buckets, size=len(block))
        for b in range(n_buckets):
            part = block[which == b]
            if len(part):
                handles[b].write(part.tobytes())
                sizes[b] += len(part)
        print(f"scattered {stop:,} of {total:,}, book dropped "
              f"{dropped[0]:,}", flush=True)
    for h in handles:
        h.close()
    return sizes


def _npy_writer(path, dtype, shape):
    """Open a .npy for sequential appending, header written up front.

    **Deliberately not open_memmap.** Writing through a 51 GB write-mapped file
    dirties a mapped page per write and Windows must hold every dirty page in
    RAM until writeback; with the flush only at the end, that grows unbounded
    against 15.7 GB of RAM. It froze this machine hard at bucket 15 of 23 -
    Kernel-Power 41, no bugcheck, no minidump, which is the signature of a
    memory-pressure hang rather than a driver fault. Mapped *file* pages do not
    count against the commit limit, so it exhausted RAM without ever raising a
    clean out-of-memory error.

    Ordinary buffered writes go through the normal file cache, which Windows
    flushes on its own schedule and can always drop under pressure."""
    handle = open(path, "wb")
    np.lib.format.write_array_header_2_0(
        handle, {"descr": np.lib.format.dtype_to_descr(dtype),
                 "fortran_order": False, "shape": shape})
    return handle


def gather(tmp_dir, out_dir, sizes, rng):
    """Pass two: shuffle each bucket in RAM and append it to the output."""
    total = int(sizes.sum())
    out = [
        (_npy_writer(os.path.join(out_dir, "feats.npy"), np.dtype(np.uint16),
                     (total, 32)), "f"),
        (_npy_writer(os.path.join(out_dir, "counts.npy"), np.dtype(np.uint8),
                     (total,)), "c"),
        (_npy_writer(os.path.join(out_dir, "stm.npy"), np.dtype(np.uint8),
                     (total,)), "s"),
        (_npy_writer(os.path.join(out_dir, "score.npy"), np.dtype(np.int16),
                     (total,)), "v"),
        (_npy_writer(os.path.join(out_dir, "result.npy"), np.dtype(np.int8),
                     (total,)), "r"),
    ]
    at = 0
    for b, size in enumerate(sizes):
        size = int(size)
        if size == 0:
            continue
        path = os.path.join(tmp_dir, f"b{b}.bin")
        block = np.fromfile(path, dtype=RECORD)
        assert len(block) == size, f"bucket {b}: {len(block)} != {size}"
        rng.shuffle(block)
        # Sub-chunked so the contiguous copy `.tobytes()` makes stays small.
        # A whole field of a 24M-row bucket is another 1.5 GB on top of the
        # bucket itself, and peak memory is what killed the previous run.
        for start in range(0, size, WRITE_CHUNK):
            stop = min(start + WRITE_CHUNK, size)
            for handle, field in out:
                handle.write(
                    np.ascontiguousarray(block[field][start:stop]).tobytes())
        at += size
        del block
        os.remove(path)
        print(f"gathered bucket {b}: {size:,} rows, {at:,} total", flush=True)
    for handle, _ in out:
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
    return total


def main():
    raw_dir, out_dir = sys.argv[1], sys.argv[2]
    total = _row_count(raw_dir)
    if len(sys.argv) > 3:
        total = min(total, int(sys.argv[3]))
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    n_buckets = max(1, -(-total // ROWS_PER_BUCKET))
    print(f"{total:,} rows, {n_buckets} buckets", flush=True)

    rng = np.random.default_rng(20260909)
    sizes = scatter(raw_dir, tmp_dir, total, n_buckets, rng)
    written = gather(tmp_dir, out_dir, sizes, rng)
    os.rmdir(tmp_dir)
    print(f"done: {written:,} shuffled positions with WDL results", flush=True)


if __name__ == "__main__":
    main()
