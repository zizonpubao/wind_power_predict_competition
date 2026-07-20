"""Leakage-safe time-series cross-validation splitting on forecast blocks.

A "forecast block" is every row sharing the same ``data_available_kst_dtm``
(a single 09:00 issuance's 24-hour block of forecast_kst_dtm values, per
CLAUDE.md section 3). CV splits must cut along these block boundaries, never
across a plain row index or a random shuffle, or future weather information
leaks into training.

`BlockTimeSeriesSplit` implements an expanding-window split: fold k's
validation set is a contiguous later slice of blocks, and its training set is
every strictly-earlier block. By construction, no validation block's
``data_available_kst_dtm`` is ever <= any training block used to predict it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_BLOCK_COL = "data_available_kst_dtm"


def get_forecast_blocks(df: pd.DataFrame) -> pd.Series:
    """Return the sorted, unique `data_available_kst_dtm` values present in df."""
    return pd.Series(df[_BLOCK_COL].unique()).sort_values().reset_index(drop=True)


class BlockTimeSeriesSplit:
    """Expanding-window time-series CV splitter over forecast blocks.

    The sorted forecast blocks are cut into ``n_splits + 1`` contiguous
    chunks. Chunk 0 is never used as a validation fold on its own (it only
    ever serves as training data for fold 1); chunks 1..n_splits are the
    validation fold for folds 1..n_splits respectively, each fold's training
    set being every block strictly before its validation chunk.
    """

    def __init__(self, n_splits: int = 5):
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        self.n_splits = n_splits

    def get_n_splits(self, df: pd.DataFrame | None = None) -> int:
        return self.n_splits

    def split(self, df: pd.DataFrame):
        """Yield (train_idx, val_idx) positional-index arrays into ``df``."""
        blocks = get_forecast_blocks(df)
        n_blocks = len(blocks)
        if n_blocks < self.n_splits + 1:
            raise ValueError(
                f"Not enough forecast blocks ({n_blocks}) for n_splits={self.n_splits}; "
                f"need at least n_splits + 1 = {self.n_splits + 1} distinct "
                f"data_available_kst_dtm values."
            )

        block_values = df[_BLOCK_COL].to_numpy()
        chunks = np.array_split(np.arange(n_blocks), self.n_splits + 1)

        for k in range(1, self.n_splits + 1):
            val_positions = chunks[k]
            train_positions = np.arange(0, val_positions[0])

            train_blocks = blocks.iloc[train_positions].to_numpy()
            val_blocks = blocks.iloc[val_positions].to_numpy()

            train_idx = np.flatnonzero(np.isin(block_values, train_blocks))
            val_idx = np.flatnonzero(np.isin(block_values, val_blocks))
            yield train_idx, val_idx


def assert_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Raise ValueError if train/val forecast blocks overlap or are out of order.

    Checks:
      1. No `data_available_kst_dtm` value appears in both train_df and val_df.
      2. Every train block strictly precedes every val block (max(train) < min(val)).
    """
    train_blocks = set(train_df[_BLOCK_COL].unique())
    val_blocks = set(val_df[_BLOCK_COL].unique())

    overlap = train_blocks & val_blocks
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(
            f"Leakage detected: {len(overlap)} forecast block(s) appear in both "
            f"train and val (e.g. {sample})."
        )

    if train_blocks and val_blocks:
        max_train = max(train_blocks)
        min_val = min(val_blocks)
        if max_train >= min_val:
            raise ValueError(
                f"Leakage detected: max train data_available_kst_dtm ({max_train}) "
                f">= min val data_available_kst_dtm ({min_val}); all train blocks "
                f"must precede all val blocks."
            )
