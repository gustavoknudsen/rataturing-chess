"""Check a .binpack decodes to the conventions we train on, before training.

    python verify_binpack.py <binpack> [more.binpack ...]

Every silent disaster this project has had came from data that decoded into
something *plausible* but wrong: the Lichess extraction had a sign convention
inverted and looked fine until the sign-agreement test was run. A net trained on
a board it will never see costs the GPU hours and gives no error, so the cost of
skipping this check is the whole run.

The decisive number is **score/result sign agreement**. If the evaluation and
the game outcome are both side-to-move relative, they agree far more often than
chance in positions with a clear advantage. If one were White-relative instead,
agreement would sit near 50% and nothing else in the file would look wrong.
"""

import os
import sys

import numpy as np

from nnue_stream import BinpackStream

EXE = os.environ.get("BTC_STREAM_EXE", "tools/binpack_stream.exe")
ROWS = int(os.environ.get("BTC_VERIFY_ROWS", "2000000"))
BATCH = 16384


def collect(path, rows):
    stream = BinpackStream(EXE, [path], buffer_rows=min(rows, 500000), seed=0,
                           max_rows=rows)
    parts = [[], [], [], []]
    for feats, stm, score, result in stream.batches(BATCH):
        parts[0].append(feats.astype(np.int32))
        parts[1].append(stm.astype(np.int32))
        parts[2].append(score.astype(np.int32))
        parts[3].append(result.astype(np.int32))
    return [np.concatenate(p) for p in parts]


def check(path, rows):
    feats, stm, score, result = collect(path, rows)
    valid = feats != 65535
    counts = valid.sum(axis=1)
    problems = []

    if feats[valid].max() >= 768 or feats[valid].min() < 0:
        problems.append("feature index outside 0..767")
    kings_w = ((feats >= 5 * 64) & (feats < 6 * 64)).sum(axis=1)
    kings_b = ((feats >= 11 * 64) & (feats < 12 * 64)).sum(axis=1)
    if not ((kings_w == 1).all() and (kings_b == 1).all()):
        problems.append("not exactly one king per side")
    if counts.min() < 4 or counts.max() > 32:
        problems.append("piece count outside 4..32")
    if not set(np.unique(result).tolist()) <= {-1, 0, 1}:
        problems.append("result outside {-1,0,+1}")
    if set(np.unique(stm).tolist()) != {0, 1}:
        problems.append("side to move is not {0,1}")

    # The one that catches a flipped convention. Drawn games are excluded:
    # sign(0) matches nothing and half of all games are drawn, so counting them
    # caps agreement near 65% and makes correct data look inverted.
    clear = (np.abs(score) > 200) & (result != 0)
    agree = float(np.mean(np.sign(score[clear]) == np.sign(result[clear]))) \
        if clear.any() else float("nan")
    if not (agree > 0.75):
        problems.append(f"score/result sign agreement {agree:.3f} - one of "
                        f"them is probably not side-to-move relative")

    # Duplicate rate, which is what the v6-dd suffix claims to have reduced.
    packed = np.ascontiguousarray(feats).view(
        np.dtype((np.void, feats.shape[1] * feats.dtype.itemsize)))
    unique = len(np.unique(packed))

    print(f"{os.path.basename(path)}")
    print(f"  rows sampled        {len(counts):,}")
    print(f"  pieces  mean        {counts.mean():.2f}  "
          f"(min {counts.min()}, max {counts.max()})")
    print(f"  score   mean        {score.mean():+.1f}  "
          f"(min {score.min()}, max {score.max()})")
    print(f"  result  mean        {result.mean():+.4f}  "
          f"(win {np.mean(result > 0):.3f}, draw {np.mean(result == 0):.3f})")
    print(f"  sign agreement      {agree:.3f}  on "
          f"{int(clear.sum()):,} decisive rows with |score| > 200")
    print(f"  unique positions    {100 * unique / len(counts):.2f}%")
    print(f"  verdict             "
          f"{'OK' if not problems else 'REJECT - ' + '; '.join(problems)}")
    return not problems


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    ok = True
    for path in sys.argv[1:]:
        ok &= check(path, ROWS)
        print()
    print("all files usable" if ok else "AT LEAST ONE FILE REJECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
