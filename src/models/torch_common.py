"""Shared PyTorch utilities for the sequence models (LSTM now, Transformer in
Phase D) of the v14 pipeline reproduction (see
``.claude/plans/logical-stirring-sphinx.md`` Phase C/D).

The single most important thing here is ``build_block_sequences``: unlike the
reference hand-off's ``sequence_prep.py`` (which cut sequences on the
**calendar day**, midnight-to-midnight), this repo's sequence unit is the
**forecast issuance block** -- every row sharing one ``data_available_kst_dtm``
value, which is also exactly what ``src/validation/splitter.BlockTimeSeriesSplit``
cuts CV folds on. Building sequences on the same key the CV splitter uses is
what makes the sequence models leakage-safe: a block is a self-contained,
non-overlapping 24-row unit, so a sequence can never straddle a fold boundary
(the calendar-day version could, since a forecast block starts mid-afternoon,
not at midnight -- see the plan's "진행 전 반드시 짚어야 할 것" #3).

None of the functions here import or touch model classes; the LSTM/Transformer
model wrappers import *these*. Kept deliberately small: seed control, a
NaN-safe per-fold feature scaler, and the block->sequence tensor builder.
"""
from __future__ import annotations

import random
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Block/sequence conventions (all blocks in the feature parquets are exactly
# 24 rows -- verified against data/processed/features_*_train.parquet).
BLOCK_COL = "data_available_kst_dtm"
DT_COL = "forecast_kst_dtm"
SEQ_LEN = 24

# cuda-first; every model/tensor in this package moves to this device.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Seed python ``random``, numpy, and torch (CPU + all CUDA devices) so a
    single ``GroupLSTMModel``/``GroupTransformerModel`` seed reproduces the
    same trained weights run-to-run (the models bag over several such seeds).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GroupFeatureScaler:
    """NaN-safe standardization (mean 0 / std 1) fit on **one CV fold's train
    rows only**.

    Fitting on the whole dataset (or on val/test rows) would leak future
    feature statistics into the fold's training, so callers must construct one
    of these per fold and ``fit`` it strictly on that fold's training slice
    (``GroupLSTMModel.fit`` does exactly this). ``transform`` operates on the
    last axis, so it accepts either a flat ``(n_rows, n_features)`` matrix or a
    ``(n_blocks, seq_len, n_features)`` sequence tensor unchanged.

    NaN handling mirrors the reference ``sequence_prep.normalize_features``:
    statistics are computed with ``nanmean``/``nanstd`` (a feature that is
    entirely NaN or constant collapses to mean 0 / std 1), and any NaN
    surviving the transform is replaced with 0 (the standardized mean), so the
    network never sees a NaN.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "GroupFeatureScaler":
        arr = np.asarray(X, dtype=float)
        flat = arr.reshape(-1, arr.shape[-1])
        with warnings.catch_warnings():
            # all-NaN columns raise "Mean of empty slice" / "Degrees of freedom
            # <= 0" here; those columns are handled explicitly just below.
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(flat, axis=0)
            std = np.nanstd(flat, axis=0)
        # all-NaN feature -> nanmean/nanstd are NaN; collapse to mean 0 / std 1.
        self.mean_ = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=0.0) + 1e-6
        self.std_ = np.where(std < 1e-6, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("GroupFeatureScaler.transform() called before fit().")
        arr = np.asarray(X, dtype=float)
        return np.nan_to_num((arr - self.mean_) / self.std_, nan=0.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def build_block_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: Optional[str] = None,
    block_col: str = BLOCK_COL,
    dt_col: str = DT_COL,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Reshape a flat hourly feature frame into per-forecast-block sequences.

    Groups rows by ``block_col`` (each such block == one CV unit), sorts each
    block's rows chronologically by ``dt_col``, and stacks them into a
    ``(n_blocks, seq_len, n_features)`` tensor. Every block must have exactly
    ``seq_len`` rows (true for every block in this repo's feature parquets); a
    block with a different row count raises, since a ragged block would mean
    the block/sequence assumption this whole approach relies on has broken.

    Returns
    -------
    X_seq : (n_blocks, seq_len, n_features) float array of features.
    Y_seq : (n_blocks, seq_len) float array of ``target_col`` values with NaNs
        **preserved** (for loss masking), or ``None`` if ``target_col is None``.
    pos_idx : (n_blocks, seq_len) int array of the ORIGINAL positional row
        index (into ``df`` after a stable 0..n-1 reset) each sequence cell came
        from -- lets ``GroupLSTMModel.predict`` scatter block predictions back
        to exactly the input row order/count.
    block_values : (n_blocks,) the ``block_col`` value of each sequence, in
        ascending (chronological) order.
    """
    df = df.reset_index(drop=True)
    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    pos_list: list[np.ndarray] = []
    block_list: list = []

    for block_val, sub in df.groupby(block_col, sort=True):
        if len(sub) != seq_len:
            raise ValueError(
                f"Forecast block {block_val!r} has {len(sub)} rows, expected exactly "
                f"seq_len={seq_len}; sequence models require complete blocks (a ragged "
                f"block breaks the block==sequence==CV-unit invariant)."
            )
        sub = sub.sort_values(dt_col)
        X_list.append(sub[feature_cols].to_numpy(dtype=float))
        pos_list.append(sub.index.to_numpy())
        block_list.append(block_val)
        if target_col is not None:
            Y_list.append(sub[target_col].to_numpy(dtype=float))

    X_seq = np.stack(X_list, axis=0)
    pos_idx = np.stack(pos_list, axis=0)
    block_values = np.asarray(block_list)
    Y_seq = np.stack(Y_list, axis=0) if target_col is not None else None
    return X_seq, Y_seq, pos_idx, block_values
