"""Unit tests for src.validation.splitter (block-aware, leakage-safe CV)."""
import numpy as np
import pandas as pd
import pytest

from src.validation.splitter import (
    BlockTimeSeriesSplit,
    assert_no_leakage,
    get_forecast_blocks,
)

N_BLOCKS = 12
ROWS_PER_BLOCK = 24  # mimics the real 24-hours-per-issuance forecast block shape


def _make_synthetic_df(n_blocks: int = N_BLOCKS, rows_per_block: int = ROWS_PER_BLOCK) -> pd.DataFrame:
    """Build a small synthetic forecast dataframe with several forecast blocks,
    shuffled row order (to make sure the splitter doesn't rely on row order)."""
    base = pd.Timestamp("2024-01-01 13:00:00")
    rows = []
    for b in range(n_blocks):
        data_available = base + pd.Timedelta(days=b)
        for h in range(rows_per_block):
            rows.append(
                {
                    "data_available_kst_dtm": data_available,
                    "forecast_kst_dtm": data_available + pd.Timedelta(hours=h + 12),
                    "feature": b * 100 + h,
                }
            )
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_get_forecast_blocks_sorted_unique():
    df = _make_synthetic_df()
    blocks = get_forecast_blocks(df)
    assert len(blocks) == N_BLOCKS
    assert list(blocks) == sorted(blocks)
    assert blocks.is_unique


def test_number_of_splits():
    df = _make_synthetic_df()
    n_splits = 5
    splitter = BlockTimeSeriesSplit(n_splits=n_splits)
    folds = list(splitter.split(df))
    assert len(folds) == n_splits


def test_folds_are_expanding_and_non_empty():
    df = _make_synthetic_df()
    splitter = BlockTimeSeriesSplit(n_splits=5)
    folds = list(splitter.split(df))

    prev_train_size = 0
    for train_idx, val_idx in folds:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        # no row appears in both
        assert set(train_idx).isdisjoint(set(val_idx))
        # expanding window: training set never shrinks across folds
        assert len(train_idx) >= prev_train_size
        prev_train_size = len(train_idx)


def test_no_leakage_across_all_folds():
    df = _make_synthetic_df()
    splitter = BlockTimeSeriesSplit(n_splits=5)
    for train_idx, val_idx in splitter.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        # must not raise
        assert_no_leakage(train_df, val_df)

        max_train_block = train_df["data_available_kst_dtm"].max()
        min_val_block = val_df["data_available_kst_dtm"].min()
        assert max_train_block < min_val_block


def test_too_few_blocks_raises():
    df = _make_synthetic_df(n_blocks=4)
    splitter = BlockTimeSeriesSplit(n_splits=5)
    with pytest.raises(ValueError):
        list(splitter.split(df))


def test_invalid_n_splits_raises():
    with pytest.raises(ValueError):
        BlockTimeSeriesSplit(n_splits=0)


def test_assert_no_leakage_raises_on_overlapping_blocks():
    df = _make_synthetic_df()
    blocks = get_forecast_blocks(df)
    shared_block = blocks.iloc[3]
    train_df = df[df["data_available_kst_dtm"] <= shared_block]
    val_df = df[df["data_available_kst_dtm"] >= shared_block]
    with pytest.raises(ValueError):
        assert_no_leakage(train_df, val_df)


def test_assert_no_leakage_raises_on_reversed_order():
    df = _make_synthetic_df()
    blocks = get_forecast_blocks(df)
    # intentionally broken split: "train" is actually the later blocks
    late_blocks = blocks.iloc[6:]
    early_blocks = blocks.iloc[:6]
    train_df = df[df["data_available_kst_dtm"].isin(late_blocks)]
    val_df = df[df["data_available_kst_dtm"].isin(early_blocks)]
    with pytest.raises(ValueError):
        assert_no_leakage(train_df, val_df)


def test_assert_no_leakage_passes_on_correct_split():
    df = _make_synthetic_df()
    blocks = get_forecast_blocks(df)
    train_blocks = blocks.iloc[:6]
    val_blocks = blocks.iloc[6:]
    train_df = df[df["data_available_kst_dtm"].isin(train_blocks)]
    val_df = df[df["data_available_kst_dtm"].isin(val_blocks)]
    assert_no_leakage(train_df, val_df)  # should not raise
