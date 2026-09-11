"""Training batches read straight out of .binpack files, with no corpus on disk.

    from nnue_stream import BinpackStream
    stream = BinpackStream(exe, ["a.binpack", "b.binpack"])
    for feats, stm, score, result in stream.batches(16384):
        ...

The materialised path (binpack_convert -> binpack_prep -> nnue_train) turns 2B
positions into 128 GB of .raw plus a 128 GB shuffle pass. That is affordable on
a machine with a spare terabyte and several hours, and impossible on a hosted
notebook with ~70 GB of disk in total. Here the only thing on disk is the
binpack, which holds those same 2B positions in 10 GB, so the corpus size stops
being a limit on how much data we can train on.

**How the shuffle works, and why it is enough.** binpack rows arrive in game
order, and consecutive plies of one game have near-identical evaluations, so
training on the raw order would give a 16384-position batch an effective sample
size of a few dozen. The reader therefore fills a large buffer, permutes it, and
serves batches from the permutation. A buffer of 4M positions spans roughly
25,000 games, so a batch drawn from it is effectively a random sample; this is
the same approach bullet uses, and the reason the two-pass external shuffle is
not needed. What it does *not* give is a globally uniform permutation - a
position from the end of the last file can never land in the first batch - which
is why validation must come from separate files rather than from a tail slice.

The reader runs on its own thread and stays a couple of buffers ahead, so the
decode and the host copy overlap GPU work instead of serialising with it.
"""

import os
import queue
import subprocess
import threading

import numpy as np

# Field for field the layout of `struct Record` in tools/binpack_stream.cpp.
# Built from a plain list so numpy packs it, giving the 69 bytes the C++ side
# asserts on; an aligned dtype would silently insert padding and desynchronise
# the whole stream after the first record.
RECORD = np.dtype([("feats", "<u2", 32), ("count", "u1"), ("stm", "u1"),
                   ("score", "<i2"), ("result", "i1")])
assert RECORD.itemsize == 69, RECORD.itemsize

BUFFER_ROWS = int(os.environ.get("BTC_STREAM_BUFFER", "4000000"))
QUEUE_DEPTH = 2


def _drain_stderr(proc, name):
    """Echo the reader's own diagnostics instead of discarding them."""
    try:
        for line in proc.stderr:
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                print(f"    [{name}] {text}", flush=True)
    except (ValueError, OSError):
        pass


class BinpackStream:
    """Shuffled batches from a set of binpack files. One pass = one epoch."""

    def __init__(self, exe, files, buffer_rows=BUFFER_ROWS, seed=None,
                 max_rows=0):
        # Absolute: CreateProcess rejects a relative path containing forward
        # slashes, so "tools/binpack_stream.exe" launches from a shell and not
        # from subprocess.
        self.exe = os.path.abspath(exe)
        self.files = [os.path.abspath(f) for f in files]
        self.buffer_rows = buffer_rows
        self.rng = np.random.default_rng(seed)
        # A cap makes an "epoch" a fixed number of positions rather than a
        # whole pass. With a corpus far larger than the time available that is
        # the only way the cosine schedule can know how long it has to run.
        self.max_rows = max_rows
        self.rows_seen = 0
        for path in self.files + [self.exe]:
            if not os.path.exists(path):
                raise FileNotFoundError(path)

    def _reader(self, proc, out, stop, rows_per_block):
        """Decode fixed-size blocks off the pipe until it ends."""
        want = rows_per_block * RECORD.itemsize
        try:
            while True:
                # readinto on a bytearray, because read() on a pipe returns
                # short reads whenever the writer flushes mid-block and a short
                # block would misalign every record after it.
                buf = bytearray(want)
                got = 0
                while got < want and not stop.is_set():
                    chunk = proc.stdout.read(want - got)
                    if not chunk:
                        break
                    buf[got:got + len(chunk)] = chunk
                    got += len(chunk)
                if got == 0 or stop.is_set():
                    break
                rows = got // RECORD.itemsize
                out.put(np.frombuffer(bytes(buf[:rows * RECORD.itemsize]),
                                      dtype=RECORD))
                if got < want:
                    break
        except (ValueError, OSError):
            # The consumer abandoned the generator and closed the pipe under
            # us. That is a normal early exit, not a failure.
            pass
        finally:
            out.put(None)

    def batches(self, batch_size, drop_last=True):
        """Yield (feats, stm, score, result) numpy arrays, shuffled.

        **One reader per file, interleaved.** A single reader over several
        files drains them in order, so with two months and a budget smaller
        than their total, training would see all of the first month and then
        part of the second - and under a cosine schedule the low-learning-rate
        phase at the end, where the network is refined, would see one month
        only. Reading every file concurrently and mixing a block from each
        makes any prefix of the stream a representative sample of the whole
        corpus.

        Total memory is unchanged: the buffer is split across the readers
        rather than duplicated."""
        per_file = max(batch_size, self.buffer_rows // max(len(self.files), 1))
        out = queue.Queue(maxsize=QUEUE_DEPTH * len(self.files))
        stop = threading.Event()
        procs, threads = [], []
        for path in self.files:
            # stderr is kept, not discarded. The reader prints its
            # "done: kept N of M seen" summary and any failure there, and
            # discarding it once cost a nine-hour training run: the stream
            # ended after the first of two files, the epoch quietly finished
            # at 48.6% of its budget, and the only evidence left was a row
            # count that happened to match one file.
            proc = subprocess.Popen([self.exe, path], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, bufsize=0)
            threading.Thread(target=_drain_stderr,
                             args=(proc, os.path.basename(path)),
                             daemon=True).start()
            thread = threading.Thread(
                target=self._reader, args=(proc, out, stop, per_file),
                daemon=True)
            thread.start()
            procs.append(proc)
            threads.append(thread)
        try:
            live = len(self.files)
            pending = []
            while live or pending:
                while live and len(pending) < live:
                    block = out.get()
                    if block is None:
                        live -= 1
                        continue
                    pending.append(block)
                if not pending:
                    break
                # One block from each live reader, concatenated and permuted
                # together, so a batch mixes files rather than following them.
                merged = np.concatenate(pending) if len(pending) > 1 \
                    else pending[0]
                pending = []
                order = self.rng.permutation(len(merged))
                for start in range(0, len(merged), batch_size):
                    idx = order[start:start + batch_size]
                    if len(idx) < batch_size and drop_last:
                        continue
                    rows = merged[idx]
                    self.rows_seen += len(idx)
                    yield (rows["feats"], rows["stm"], rows["score"],
                           rows["result"])
                    if self.max_rows and self.rows_seen >= self.max_rows:
                        return
        finally:
            # Order matters. Killing the writers first makes each reader's
            # pending read return empty rather than block, and draining the
            # queue unwedges any parked on a full put; only then is it safe to
            # close the pipes.
            stop.set()
            for proc in procs:
                proc.terminate()
            while not out.empty():
                out.get_nowait()
            for thread in threads:
                thread.join(timeout=30)
            for proc in procs:
                proc.stdout.close()
                proc.wait(timeout=30)
