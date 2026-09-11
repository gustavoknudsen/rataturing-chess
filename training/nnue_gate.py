"""Gate 1: does the network out-evaluate the hand-crafted evaluation?

    python nnue_gate.py <net.npz> <data_dir> [n_positions]

Run with the ENGINE venv (.venv), not the training one - it needs btc_eval.

This is the go/no-go before any engine integration. Both evaluators are scored
on the same held-out positions against the same deep labels, and the
network has to win. The reasoning is simple: if it cannot beat the hand-crafted
evaluation at the static-evaluation task, it will not beat it inside a search,
and there is no point spending match time finding that out.

It is a necessary condition, not a sufficient one. The two estimators make
different *kinds* of error - a hand-crafted evaluation's mistakes are smooth and
correlated, which a search can partly wash out - so a win here means "worth a
match", not "will win a match".

Also holds the numpy implementation of quantised inference. The engine's numba
version must agree with it exactly, so any disagreement localises to the engine
integration rather than to the network.
"""

# Engine modules live in src/; this script is run directly, so sys.path[0]
# is this folder. Put src/ on the path before importing them.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "src"))


import os
import sys

# Must precede the btc_eval import: it reads this at module scope, and once
# net.npz sits in the project root - which is exactly what ships - `evaluate()`
# silently becomes the network. That happened, and turned a net-vs-HCE gate
# into a net-vs-net comparison whose "HCE" column was another network's scores.
# The baseline arm of this gate is the hand-crafted evaluation by definition.
os.environ["BTC_NNUE"] = "0"

import numpy as np

import btc_core as core
from btc_eval import evaluate
from btc_nnue import NNUE_MIN_PIECES

PAD = 65535


def load_net(path):
    z = np.load(path)
    table = z["bucket_table"].astype(np.int32) if "bucket_table" in z.files \
        else np.zeros(64, dtype=np.int32)
    buckets = int(z["buckets"]) if "buckets" in z.files else 1
    # (out_buckets, 2 * l1) and (out_buckets,) always; a net saved before
    # output buckets existed stored a flat vector and a scalar.
    out_w = np.atleast_2d(z["out_w"].astype(np.int32))
    out_b = np.atleast_1d(z["out_b"].astype(np.int64))
    return (z["ft_w"].astype(np.int32), z["ft_b"].astype(np.int32),
            out_w, out_b,
            int(z["l1"]), int(z["qa"]), int(z["qb"]), int(z["scale"]),
            table, buckets)


def _king_square(feats, count, base):
    for i in range(count):
        index = int(feats[i])
        if base <= index < base + 64:
            return index - base
    return 0


def net_eval(feats, count, stm, net):
    """Quantised inference, side-to-move relative, in centipawns.

    Deliberately an independent implementation of the same arithmetic as
    btc_nnue.py, written from the feature list rather than from bitboards.
    test_nnue.py asserts the two agree exactly; that only means anything while
    they stay independent, so resist the urge to share code between them.

    int64 throughout the output accumulation. The worst case is
    QA^2 * 1.98*QB * 2*L1 = 4.23e9, which overflows int32 - production C++
    engines use int32 and never hit it in practice, but int64 is free here."""
    ft_w, ft_b, out_w, out_b, l1, qa, qb, scale, table, buckets = net
    acc_w = ft_b.copy()
    acc_b = ft_b.copy()
    white_flip = black_flip = white_offset = black_offset = 0
    if buckets > 1:
        white_king = _king_square(feats, count, 5 * 64)
        black_king = _king_square(feats, count, 11 * 64) ^ 56
        white_flip = 7 if (white_king & 7) >= 4 else 0
        black_flip = 7 if (black_king & 7) >= 4 else 0
        white_offset = 768 * int(table[white_king ^ white_flip])
        black_offset = 768 * int(table[black_king ^ black_flip])
    for i in range(count):
        index = int(feats[i])
        if index == PAD:
            continue
        colour, piece, square = index // 384, (index % 384) // 64, index % 64
        acc_w += ft_w[white_offset + 384 * colour + 64 * piece
                      + (square ^ white_flip)]
        acc_b += ft_w[black_offset + 384 * (1 - colour) + 64 * piece
                      + ((square ^ 56) ^ black_flip)]

    first, second = (acc_w, acc_b) if stm == 0 else (acc_b, acc_w)
    clipped_a = np.clip(first, 0, qa).astype(np.int64)
    clipped_b = np.clip(second, 0, qa).astype(np.int64)
    # Output bucket by piece count, mirroring btc_nnue._out_bucket. `count` is
    # the number of stored features, which is the piece count.
    out_buckets = out_w.shape[0]
    bucket = 0
    if out_buckets > 1:
        bucket = min(out_buckets - 1, max(0, (count - 1) * out_buckets // 32))
    row = out_w[bucket]
    total = int((clipped_a * clipped_a * row[:l1]).sum()
                + (clipped_b * clipped_b * row[l1:]).sum())
    total = total // qa + int(out_b[bucket])
    return total * scale // (qa * qb)


def fit_scale(pred, label):
    """Least-squares scalar mapping this evaluator onto the label scale.

    Necessary for a fair comparison, not a convenience. The hand-crafted
    evaluation is on BTC's unified scale where a pawn is **126**, while the
    labels are standard centipawns - comparing them raw charges the HCE for a
    units mismatch and made its R2 read 0.066. The network has the same freedom
    in the engine, where the output SCALE is an independently tuned parameter
    (tcheran trains at 400 and runs at 321), so giving each evaluator its own
    optimal scale is the honest comparison of *evaluation quality*."""
    denom = float((pred * pred).sum())
    return float((pred * label).sum()) / denom if denom > 0 else 1.0


def report_units(hce_pred, net_pred):
    """Recommend BTC_NET_UNITS: the multiplier putting the network's output on
    the hand-crafted evaluation's scale.

    This is not cosmetic. Every margin in the search - RFP_MARGIN, futility,
    delta pruning, the aspiration window - is a constant in engine units tuned
    against the spread of evaluate(). Swap in an evaluator whose output is half
    as wide and every one of those margins doubles in effective strength
    without a single constant changing. That is a different search, and it
    would be measured as "the network is weak".

    The two fitted scales in fit_scale cannot answer this: least squares
    attenuates a slope by the predictor's own noise, the two evaluators have
    very different noise, so their ratio conflates scale with accuracy. Matching
    the spread of the two output distributions is free of that - it asks only
    how wide each evaluator's scores are, not how right they are.

    Reported on the interquartile range as well as the standard deviation
    because the standard deviation is dominated by the decided positions, which
    are exactly the ones the margins do not care about."""
    net_iqr = float(np.percentile(np.abs(net_pred), 75))
    hce_iqr = float(np.percentile(np.abs(hce_pred), 75))
    sd_ratio = float(hce_pred.std() / max(net_pred.std(), 1e-9))
    iqr_ratio = hce_iqr / max(net_iqr, 1e-9)
    print(f"scale match: sd ratio {sd_ratio:.3f}, |p75| ratio "
          f"{iqr_ratio:.3f}  ->  BTC_NET_UNITS {int(round(iqr_ratio * 100))}")


def metrics(pred, label):
    err = pred - label
    abs_err = np.abs(err)
    ss_res = float((err ** 2).sum())
    ss_tot = float(((label - label.mean()) ** 2).sum())
    # Q-space: the sigmoid compresses the decided positions, so this weights
    # the range that actually decides games rather than the blowouts.
    q_pred = 1.0 / (1.0 + np.exp(-pred / 400.0))
    q_label = 1.0 / (1.0 + np.exp(-label / 400.0))
    q_res = float(((q_pred - q_label) ** 2).sum())
    q_tot = float(((q_label - q_label.mean()) ** 2).sum())
    return {
        "mae": float(abs_err.mean()),
        "r2": 1.0 - ss_res / max(ss_tot, 1e-9),
        "r2_q": 1.0 - q_res / max(q_tot, 1e-9),
        "p68": float(np.percentile(abs_err, 68.27)),
        "p95": float(np.percentile(abs_err, 95.45)),
        "p99": float(np.percentile(abs_err, 99.73)),
    }


def main():
    net_path, data_dir = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    net = load_net(net_path)

    feats = np.load(f"{data_dir}/feats.npy", mmap_mode="r")
    counts = np.load(f"{data_dir}/counts.npy", mmap_mode="r")
    stm = np.load(f"{data_dir}/stm.npy", mmap_mode="r")
    score = np.load(f"{data_dir}/score.npy", mmap_mode="r")

    # the tail of the file is the held-out slice the trainer never saw
    lo = len(score) - n
    feats, counts = np.asarray(feats[lo:]), np.asarray(counts[lo:])
    stm, score = np.asarray(stm[lo:]), np.asarray(score[lo:])

    # Score only the regime the engine actually uses the network in. Below
    # btc_nnue.NNUE_MIN_PIECES the engine keeps the specialised evaluation, and
    # evaluate() there returns a mating drive rather than centipawns - leaving
    # those positions in charges the hand-crafted evaluation thousands of
    # centipawns of "error" for correctly reporting a won ending, which
    # distorts its fitted scale and its outlier percentiles both.
    usable = counts >= NNUE_MIN_PIECES
    dropped = n - int(usable.sum())
    feats, counts = feats[usable], counts[usable]
    stm, score = stm[usable], score[usable]
    n = len(score)

    bb, st = core.new_board()
    net_pred = np.zeros(n, dtype=np.float64)
    hce_pred = np.zeros(n, dtype=np.float64)
    label = score.astype(np.float64)

    for row in range(n):
        net_pred[row] = net_eval(feats[row], counts[row], stm[row], net)
        # rebuild a board from the stored features and ask the engine
        bb[:] = 0
        for i in range(counts[row]):
            index = int(feats[row, i])
            bb[index // 64] |= np.uint64(1) << np.uint64(index % 64)
        for piece in range(6):
            bb[core.OCC_W] |= bb[piece]
            bb[core.OCC_B] |= bb[piece + 6]
        bb[core.OCC_A] = bb[core.OCC_W] | bb[core.OCC_B]
        st[core.SIDE] = stm[row]
        st[core.EP] = core.NO_SQ
        st[core.CASTLE] = 0
        st[core.FIFTY] = 0
        hce_pred[row] = evaluate(bb, st)

    net_scale = fit_scale(net_pred, label)
    hce_scale = fit_scale(hce_pred, label)
    report_units(hce_pred, net_pred)
    net_m = metrics(net_pred * net_scale, label)
    hce_m = metrics(hce_pred * hce_scale, label)
    print(f"{n:,} held-out positions, labels = deep search, "
          f"both scores side-to-move relative")
    print(f"{dropped:,} dropped as under {NNUE_MIN_PIECES} pieces, where the "
          f"engine keeps the specialised evaluation")
    print(f"fitted scale: HCE x{hce_scale:.3f}  net x{net_scale:.3f}  "
          f"(each evaluator gets its own, see fit_scale)\n")
    print(f"{'metric':10s} {'HCE':>12s} {'net':>12s}   better")
    for key in ("mae", "r2", "r2_q", "p68", "p95", "p99"):
        lower_is_better = key in ("mae", "p68", "p95", "p99")
        win = (net_m[key] < hce_m[key]) if lower_is_better \
            else (net_m[key] > hce_m[key])
        print(f"{key:10s} {hce_m[key]:12.4f} {net_m[key]:12.4f}   "
              f"{'net' if win else 'HCE'}")

    passed = net_m["r2_q"] > hce_m["r2_q"] and net_m["p95"] < hce_m["p95"]
    print(f"\nGATE 1: {'PASS' if passed else 'FAIL'} "
          f"(needs Q-space R2 and the 2-sigma error percentile)")


if __name__ == "__main__":
    main()
