# NNUE training

The pipeline that produced `src/net.npz`, a king-bucketed network (32 king
buckets by 768 features into a 512-wide layer, 8 output buckets) trained from
scratch on public engine-labelled positions. What was learned doing it is in
[`../docs/research/NNUE_TRAINING.md`](../docs/research/NNUE_TRAINING.md).

Two environments: the engine's own `.venv` (numba, no torch) and a training
venv with torch and CUDA (`.venv_nnue`, from `requirements-dev.txt`). Torch is
never installed into the engine venv.

| step | script | environment |
|---|---|---|
| decode and check a binpack | `verify_binpack.py` | either |
| stream batches straight from binpacks | `nnue_stream.py` (used by the trainer) | training |
| or build a shuffled `.npy` corpus | `binpack_prep.py`, `nnue_data.py` | either |
| train and quantise | `nnue_train.py <data_dir> <out_dir> [L1] [epochs]` | training |
| gate a net against the hand-crafted evaluation | `nnue_gate.py <net.npz> <data_dir>` | engine |
| rank quantised nets on one holdout | `nnue_rank.py` | training |

`nnue_gate.py` also prints `NET_UNITS`, the per-network scale the engine needs
in `src/btc_eval.py`; a net run at another net's units rescales every search
margin. The static gate is used to reject a broken net and to read that
number, never to rank two working nets: three times a net that gated within
0.002 of another lost 40 elo to it in games. Only a match ranks networks.

`notebooks/kaggle_nnue.ipynb` is the notebook that trained the shipped
network on Kaggle. The binpack decoder it streams from is
`tools/binpack_stream.cpp`.
