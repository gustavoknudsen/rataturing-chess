"""Cut smaller networks out of the shipped one, for a surprise size cap.

    python finals_day/net_reduce.py            build the whole ladder
    python finals_day/net_reduce.py --buckets 4

Why reduce rather than train. The shipped net finished its learning-rate
schedule; four earlier runs did not, and a net stopped at 40% of its cosine
measured 12% worse than a smaller net that finished. An overnight run under
deadline is how you produce a fifth of those. Reducing inherits the completed
schedule for free.

Why it should work. The shipped net was trained with feature factorization: a
shared unbucketed 768-row table is folded into the bucketed weights at export,
so each bucket is approximately that shared table plus a small delta. Averaging
a group of buckets therefore recovers something close to the shared table,
which is exactly what a net with fewer buckets wants in its place.

That is an argument, not a measurement. Rank the output with
training/nnue_rank.py on a fixed holdout before trusting any of it, and derive
BTC_NET_UNITS per net with training/nnue_gate.py: the engine converts the raw
output to centipawns with that constant, and every pruning margin in the search
reads the result. A net with the wrong units loads, compiles and plays legal
moves while every margin is wrong, which is the failure that looks like
nothing at all.
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHIPPED = os.path.join(HERE, os.pardir, "src", "net.npz")
OUT_DIR = os.path.join(HERE, "nets")

# Bucket counts to build by default, smallest file first. 32 is the shipped
# net and is skipped.
LADDER = (1, 2, 4, 8, 16)


def load(path):
    data = dict(np.load(path))
    missing = [k for k in ("ft_w", "ft_b") if k not in data]
    if missing:
        raise SystemExit("not an engine net, missing %s: %s" % (missing, path))
    return data


def bucket_count(data):
    """Buckets implied by the feature transformer, which is (buckets*768, L1)."""
    rows = data["ft_w"].shape[0]
    if rows % 768:
        raise SystemExit("ft_w has %d rows, not a multiple of 768" % rows)
    return rows // 768


def reduce_buckets(data, target):
    """Average groups of king buckets down to `target` of them.

    The feature transformer is (buckets * 768, L1), king bucket b occupying
    rows [768*b, 768*(b+1)). Averaging whole groups keeps that layout, so the
    result loads through the ordinary path with no code change.
    """
    source = bucket_count(data)
    if target > source or source % target:
        raise SystemExit("cannot go from %d buckets to %d: must divide"
                         % (source, target))
    out = dict(data)
    group = source // target
    ft = data["ft_w"]
    rows = []
    for b in range(target):
        # Cast per group, not the whole table: the shipped ft_w is 24 MB of
        # int16 and 100 MB as float64.
        block = ft[768 * group * b: 768 * group * (b + 1)].astype(np.float32)
        # (group, 768, L1) -> mean over the group axis. rint, because .astype
        # on a float truncates and a systematic bias toward zero across every
        # weight is exactly the kind of error that survives every check.
        rows.append(np.rint(block.reshape(group, 768, -1).mean(axis=0)))
    merged = np.concatenate(rows)
    info = np.iinfo(ft.dtype)
    out["ft_w"] = merged.clip(info.min, info.max).astype(ft.dtype)
    out["buckets"] = np.int32(target)
    out["bucket_table"] = _rebucket(data, source, target, group)
    return out


def _rebucket(data, source, target, group):
    """Map each of the 64 king squares to its new bucket index."""
    table = data.get("bucket_table")
    if table is None:
        return np.zeros(64, dtype=np.int32)
    return ((np.asarray(table).astype(np.int32) // group)
            .clip(0, target - 1).astype(np.int32))


def write(out, path):
    np.savez(path, **out)
    return os.path.getsize(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--net", default=SHIPPED, help="source net.npz")
    parser.add_argument("--buckets", type=int, action="append",
                        help="target bucket count; repeatable, default the ladder")
    parser.add_argument("--out", default=OUT_DIR, help="output directory")
    args = parser.parse_args()

    data = load(args.net)
    source = bucket_count(data)
    l1 = data["ft_w"].shape[1]
    print("source: %s" % os.path.normpath(args.net))
    print("        %d buckets, L1 %d, %.2f MB"
          % (source, l1, os.path.getsize(args.net) / 1048576))
    os.makedirs(args.out, exist_ok=True)

    print("\n%-10s %10s  %s" % ("buckets", "MB", "file"))
    for target in sorted(args.buckets or LADDER):
        if target >= source:
            continue
        out = reduce_buckets(data, target)
        name = "net_b%d.npz" % target
        path = os.path.join(args.out, name)
        size = write(out, path)
        print("%-10d %10.2f  %s" % (target, size / 1048576, name))

    print("\nNext, for each one, in this order:")
    print("  1. python training/nnue_gate.py <net> <data_dir> 20000")
    print("     -> the BTC_NET_UNITS value. Without it every search margin is")
    print("        mis-scaled and nothing looks wrong.")
    print("  2. python training/nnue_rank.py   -> rank them on one fixed holdout")
    print("  3. record both numbers in the PANEL comments in src/agent.py")


if __name__ == "__main__":
    main()
