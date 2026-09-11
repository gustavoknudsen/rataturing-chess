"""Rank quantised nets against each other on one identical holdout.

    .venv_nnue/Scripts/python.exe nnue_rank.py <holdout.binpack> a.npz b.npz ...

Each training run already prints a validation loss, but those numbers are not
strictly comparable: the runs use different `BTC_STREAM_BUFFER` sizes, so a 4M
row validation slice is drawn from a differently sized shuffle buffer and is a
different 4M rows. The gap between architectures should dwarf that, but this is
the measurement that picks the net we ship, so it is worth removing the doubt.

Two further differences from the in-training number, both deliberate:

- Every net is scored on **the same rows**, materialised once and reused.
- Nets are scored **as quantised**, dequantised back from the int16 file the
  engine will actually load, so quantisation error counts against a net rather
  than being hidden. A wider layer has more values sharing the same int16 grid.

Architecture is read from each file, so nets with different L1, king bucket and
output bucket counts rank against each other directly.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

# Must precede the nnue_train import: it reads WDL_LAMBDA at module scope.
#
# The default there is 1.0, a pure evaluation target, and inheriting it silently
# would rank nets on an objective none of them was trained for. It is not a
# harmless difference. Measured on the shipped 256x4 net against a 768x32x8 net
# only 30% through training: at lambda 1.0 the gap is 0.07%, at 0.7 it is 2.5%.
# The wider net is barely better at predicting the label evaluation and much
# better at predicting the game result - and the result is the half that
# correlates with winning. Rank on the objective the nets were trained on.
os.environ.setdefault("BTC_WDL_LAMBDA", "0.7")

import nnue_train as nt
from nnue_stream import BinpackStream

BATCH = 16384
ROWS = int(os.environ.get("RANK_ROWS", "2000000"))
EXE = os.environ.get("BTC_STREAM_EXE", "tools/binpack_stream.exe")


def load_quantised(path, device):
    """Rebuild the training model from the int16 file the engine ships."""
    data = np.load(path)
    keys = set(data.files)
    l1 = int(data["l1"])
    # Nets predating king buckets and output buckets have neither key, and
    # predating the bucket table have no table. Defaulting them to 1 and to the
    # all-zero table is exactly what btc_nnue.load does for the same files, so
    # an old net ranks on the same terms the engine would run it on.
    buckets = int(data["buckets"]) if "buckets" in keys else 1
    out_buckets = int(data["out_buckets"]) if "out_buckets" in keys else 1
    model = nt.Nnue(l1, buckets, out_buckets).to(device)
    rows = buckets * 768
    with torch.no_grad():
        model.ft.weight[:rows] = torch.from_numpy(
            data["ft_w"].astype(np.float32) / nt.QA).to(device)
        model.ft.weight[rows:].zero_()
        model.ft_bias.copy_(torch.from_numpy(
            data["ft_b"].astype(np.float32) / nt.QA).to(device))
        model.out.weight.copy_(torch.from_numpy(
            np.atleast_2d(data["out_w"]).astype(np.float32) / nt.QB).to(device))
        model.out.bias.copy_(torch.from_numpy(
            np.atleast_1d(data["out_b"]).astype(np.float32)
            / (nt.QA * nt.QB)).to(device))
    if "bucket_table" in keys:
        table_np = data["bucket_table"].astype(np.int64)
    else:
        table_np = np.zeros(64, dtype=np.int64)
    table = torch.from_numpy(table_np).to(device)
    return model, table, l1, buckets, out_buckets


def materialise(holdout):
    """The identical rows every net is scored on."""
    stream = BinpackStream(EXE, [holdout], seed=0, max_rows=ROWS)
    out = []
    for batch in stream.batches(BATCH):
        out.append(tuple(np.asarray(x).copy() for x in batch))
    return out


def score(model, table, batches, device):
    """Mean loss and mean absolute sigmoid error, weighted by batch size."""
    total = 0.0
    mae_total = 0.0
    seen = 0
    for feats, stm, sc, res in batches:
        loss, mae = nt._step(model, None, None, (feats, stm, sc, res), device,
                             table, False)
        n = len(feats)
        total += loss * n
        mae_total += mae
        seen += n
    return total / max(seen, 1), mae_total / max(seen, 1)


def label(path):
    """Parent directory plus filename.

    Candidates arrive from different runs all named net.npz, so a bare
    basename makes two different nets print identically - and this listing is
    what the final net is chosen from."""
    full = os.path.abspath(path)
    return os.path.join(os.path.basename(os.path.dirname(full)),
                        os.path.basename(full))


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    holdout, nets = sys.argv[1], sys.argv[2:]
    # Defaults to CPU: the GPU is usually busy training the next candidate,
    # and a second CUDA context there slows the run that matters. Set
    # RANK_DEVICE=cuda once the trainer is done.
    want = os.environ.get("RANK_DEVICE", "cpu")
    device = torch.device(want if want != "cuda" or torch.cuda.is_available()
                          else "cpu")
    print(f"device {device}, {ROWS:,} rows from {os.path.basename(holdout)}, "
          f"WDL lambda {nt.WDL_LAMBDA}")
    batches = materialise(holdout)
    rows = sum(len(b[0]) for b in batches)
    print(f"materialised {rows:,} rows in {len(batches)} batches\n")
    results = []
    for path in nets:
        model, table, l1, kb, ob = load_quantised(path, device)
        model.eval()
        loss, mae = score(model, table, batches, device)
        size = os.path.getsize(path)
        results.append((loss, path, l1, kb, ob, size, mae))
        print(f"  {label(path):34s} L1={l1:5d} king={kb:3d} "
              f"out={ob:2d} {size / 1e6:6.1f}MB  val {loss:.6f}  "
              f"mae {mae:.4f}", flush=True)
    print()
    print("ranked best first:")
    for loss, path, l1, kb, ob, size, mae in sorted(results):
        print(f"  {loss:.6f}  {label(path):34s} "
              f"({l1}x{kb}x{ob}, {size / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
