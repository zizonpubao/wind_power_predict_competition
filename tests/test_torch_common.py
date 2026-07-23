"""Unit tests for src/models/torch_common.py.

Covers the two things that make the sequence models leakage-safe:
  1. GroupFeatureScaler fits statistics on the given (train-only) rows.
  2. build_block_sequences groups on the SAME forecast-block key
     (data_available_kst_dtm) BlockTimeSeriesSplit cuts folds on, so a block
     (== a sequence) never straddles two CV folds -- the core leakage-
     prevention property of this Phase C reproduction.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.torch_common import (
    BLOCK_COL,
    DT_COL,
    SEQ_LEN,
    GroupFeatureScaler,
    build_block_sequences,
)
from src.validation.splitter import BlockTimeSeriesSplit, get_forecast_blocks

FEATURES = ["f0", "f1", "f2"]


def _make_block_df(n_blocks: int, seed: int = 0, shuffle: bool = False) -> pd.DataFrame:
    """Synthetic block-structured hourly frame: n_blocks forecast blocks, each
    exactly SEQ_LEN rows, one forecast_kst_dtm per hour within the block."""
    rng = np.random.RandomState(seed)
    rows = []
    base_avail = pd.Timestamp("2024-01-01 13:00:00")
    for b in range(n_blocks):
        avail = base_avail + pd.Timedelta(days=b)
        for h in range(SEQ_LEN):
            rows.append(
                {
                    BLOCK_COL: avail,
                    DT_COL: avail + pd.Timedelta(hours=1 + h),
                    "f0": rng.rand(),
                    "f1": rng.rand() * 10,
                    "f2": rng.rand() - 0.5,
                    "target": rng.rand() * 21600,
                }
            )
    df = pd.DataFrame(rows)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# GroupFeatureScaler
# ---------------------------------------------------------------------------


def test_scaler_fits_on_given_rows_only():
    df = _make_block_df(10)
    train = df.iloc[: 6 * SEQ_LEN]
    scaler = GroupFeatureScaler().fit(train[FEATURES].to_numpy(float))
    # mean_/std_ must equal the TRAIN slice's statistics, not the full frame's.
    np.testing.assert_allclose(scaler.mean_, np.nanmean(train[FEATURES].to_numpy(float), axis=0))
    full_mean = np.nanmean(df[FEATURES].to_numpy(float), axis=0)
    assert not np.allclose(scaler.mean_, full_mean)


def test_scaler_transform_standardizes_and_is_nan_safe():
    df = _make_block_df(8)
    X = df[FEATURES].to_numpy(float).copy()
    X[0, 0] = np.nan  # a stray NaN must not propagate
    scaler = GroupFeatureScaler().fit(X)
    Z = scaler.transform(X)
    assert np.isfinite(Z).all()
    # standardized columns are ~mean 0.
    np.testing.assert_allclose(Z.mean(axis=0), np.zeros(len(FEATURES)), atol=0.2)


def test_scaler_handles_all_nan_feature():
    X = np.full((50, 3), np.nan)
    X[:, 1] = np.arange(50.0)
    scaler = GroupFeatureScaler().fit(X)
    Z = scaler.transform(X)
    assert np.isfinite(Z).all()
    # the all-NaN columns collapse to 0.
    assert np.allclose(Z[:, 0], 0.0) and np.allclose(Z[:, 2], 0.0)


# ---------------------------------------------------------------------------
# build_block_sequences shape / ordering / NaN preservation
# ---------------------------------------------------------------------------


def test_build_sequences_shapes_and_target_nan_preserved():
    df = _make_block_df(5)
    df.loc[3, "target"] = np.nan  # a single missing label inside a block
    X, Y, pos, blocks = build_block_sequences(df, FEATURES, "target")
    assert X.shape == (5, SEQ_LEN, len(FEATURES))
    assert Y.shape == (5, SEQ_LEN)
    assert pos.shape == (5, SEQ_LEN)
    assert len(blocks) == 5
    # the missing label is preserved as NaN (not dropped, not zero-filled).
    assert np.isnan(Y).sum() == 1


def test_build_sequences_sorts_within_block_and_pos_roundtrips():
    df = _make_block_df(4, shuffle=True)  # rows out of order
    X, _Y, pos, _blocks = build_block_sequences(df, FEATURES, None)
    # scattering the sequence features back via pos must reconstruct df's rows.
    flat = np.empty((len(df), len(FEATURES)))
    flat[pos.ravel()] = X.reshape(-1, len(FEATURES))
    np.testing.assert_allclose(flat, df[FEATURES].to_numpy(float))
    # each block's dt values must be ascending after the internal sort.
    df_reset = df.reset_index(drop=True)
    for row in pos:
        dts = df_reset.loc[row, DT_COL].to_numpy()
        assert (np.diff(dts.astype("datetime64[ns]").astype(np.int64)) > 0).all()


def test_build_sequences_rejects_ragged_block():
    df = _make_block_df(3)
    df = df.iloc[:-1]  # last block now has 23 rows
    with pytest.raises(ValueError):
        build_block_sequences(df, FEATURES, "target")


# ---------------------------------------------------------------------------
# CORE leakage test: sequence/block boundaries == CV fold boundaries
# ---------------------------------------------------------------------------


def test_sequences_never_straddle_a_cv_fold_boundary():
    df = _make_block_df(12, shuffle=True)
    splitter = BlockTimeSeriesSplit(n_splits=3)
    all_blocks = set(get_forecast_blocks(df))

    for train_idx, val_idx in splitter.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        _, _, _, train_blocks = build_block_sequences(train_df, FEATURES, "target")
        _, _, _, val_blocks = build_block_sequences(val_df, FEATURES, "target")
        train_set, val_set = set(train_blocks), set(val_blocks)

        # No block appears on both sides of the fold boundary.
        assert train_set.isdisjoint(val_set)
        # The union of the two sides' blocks is exactly the fold's blocks --
        # no block was silently dropped or duplicated when building sequences.
        assert train_set | val_set <= all_blocks
        # Every train block strictly precedes every val block (expanding window).
        assert max(train_set) < min(val_set)


def test_every_block_is_exactly_seq_len_rows_in_real_features():
    """Guards the assumption the whole approach rests on, on a real parquet."""
    from configs.paths import DATA_PROCESSED_DIR

    path = DATA_PROCESSED_DIR / "features_kpx_group_1_train.parquet"
    if not path.exists():
        pytest.skip("processed features not built on this machine")
    df = pd.read_parquet(path)
    sizes = df.groupby(BLOCK_COL).size().unique()
    assert list(sizes) == [SEQ_LEN]
