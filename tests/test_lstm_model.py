"""Unit tests for src/models/lstm_model.py (GroupLSTMModel) and the LSTM
harness's fully-missing-block handling.

Kept small/fast: tiny synthetic block data, 1 seed, a couple of epochs. The
focus is correctness properties, not fit quality:
  - predict output aligns to input row order/count exactly,
  - save/load round-trips to identical predictions,
  - the per-fold scaler is fit on train rows only (leakage),
  - partially-missing-label blocks are kept intact (masked), only fully-
    missing blocks are dropped.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.lstm_model import GroupLSTMModel
from src.models.torch_common import BLOCK_COL, DT_COL, SEQ_LEN

FEATURES = ["f0", "f1", "f2"]
CAPACITY = 21_600.0


def _make_block_df(n_blocks: int, seed: int = 0, shuffle: bool = False) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    base_avail = pd.Timestamp("2024-01-01 13:00:00")
    for b in range(n_blocks):
        avail = base_avail + pd.Timedelta(days=b)
        for h in range(SEQ_LEN):
            f0 = rng.rand()
            rows.append(
                {
                    BLOCK_COL: avail,
                    DT_COL: avail + pd.Timedelta(hours=1 + h),
                    "f0": f0,
                    "f1": rng.rand() * 10,
                    "f2": rng.rand() - 0.5,
                    "target": np.clip(f0 * CAPACITY, 0, CAPACITY),
                }
            )
    df = pd.DataFrame(rows)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _tiny_model(**kw) -> GroupLSTMModel:
    return GroupLSTMModel(CAPACITY, FEATURES, n_seeds=1, max_epochs=2, patience=2, **kw)


def test_predict_output_aligns_to_input_rows():
    df = _make_block_df(6)
    model = _tiny_model().fit(df)

    preds = model.predict(df)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(df),)
    assert (preds >= 0.0).all()
    assert (preds <= CAPACITY * 1.01 + 1e-6).all()


def test_predict_is_row_order_equivariant():
    """Reordering the input rows must reorder the outputs the same way -- i.e.
    prediction for a given row doesn't depend on that row's position."""
    df = _make_block_df(6)
    model = _tiny_model().fit(df)

    base = model.predict(df)
    perm = np.random.RandomState(1).permutation(len(df))
    shuffled = model.predict(df.iloc[perm].reset_index(drop=True))
    np.testing.assert_allclose(shuffled, base[perm], rtol=1e-5, atol=1e-4)


def test_save_load_round_trip_identical_predictions(tmp_path):
    df = _make_block_df(6)
    model = _tiny_model().fit(df)
    preds = model.predict(df)

    path = tmp_path / "lstm_model.joblib"
    model.save(path)
    loaded = GroupLSTMModel.load(path)

    np.testing.assert_allclose(loaded.predict(df), preds, rtol=1e-6, atol=1e-6)
    assert loaded.feature_cols == model.feature_cols
    assert len(loaded.state_dicts_) == len(model.state_dicts_)


def test_scaler_fit_on_train_rows_only():
    """After fit, the model's scaler statistics match the training frame's,
    not some larger pool -- the per-fold leakage guarantee."""
    df = _make_block_df(8)
    model = _tiny_model().fit(df)
    np.testing.assert_allclose(
        model.scaler.mean_, np.nanmean(df[FEATURES].to_numpy(float), axis=0)
    )


def test_partially_missing_block_is_kept_and_trains():
    """A block with a few missing labels must NOT be dropped -- it stays a full
    24-row sequence and training still runs (missing hours masked in loss)."""
    df = _make_block_df(5)
    df.loc[3, "target"] = np.nan
    df.loc[27, "target"] = np.nan  # two scattered missing labels in two blocks
    model = _tiny_model().fit(df)  # must not raise (block stays 24 rows)
    preds = model.predict(df)
    assert preds.shape == (len(df),)
    assert np.isfinite(preds).all()


def test_predict_before_fit_raises():
    df = _make_block_df(2)
    model = _tiny_model()
    with pytest.raises(RuntimeError):
        model.predict(df)


def test_apply_mixup_runs():
    """group3's mixup path trains end-to-end without error."""
    df = _make_block_df(5)
    model = _tiny_model(apply_mixup=True).fit(df)
    assert model.predict(df).shape == (len(df),)


# ---------------------------------------------------------------------------
# distributional extension: predict_seed_matrix / predict_quantiles
# ---------------------------------------------------------------------------


def test_predict_seed_matrix_shape_and_predict_is_its_mean():
    df = _make_block_df(6)
    model = GroupLSTMModel(CAPACITY, FEATURES, n_seeds=3, max_epochs=2, patience=2).fit(df)
    seed_mat = model.predict_seed_matrix(df)
    assert seed_mat.shape == (len(df), 3)
    # predict is exactly the row-wise mean of the seed matrix (backward compat).
    np.testing.assert_allclose(model.predict(df), seed_mat.mean(axis=1), rtol=1e-6, atol=1e-6)


def test_predict_quantiles_shape_monotonic_and_median_equals_point():
    from src.features.decision_optimize import QUANTILES

    df = _make_block_df(6)
    model = GroupLSTMModel(CAPACITY, FEATURES, n_seeds=4, max_epochs=2, patience=2).fit(df)
    q = model.predict_quantiles(df)
    assert q.shape == (len(df), len(QUANTILES))
    # non-decreasing across levels per row.
    assert (np.diff(q, axis=1) >= -1e-6).all()
    # median (level 0.5 at index 4) equals the point prediction, up to the 1%
    # capacity-margin clip difference (predict clips to cap*1.01, quantiles to cap).
    median = q[:, QUANTILES.index(0.5)]
    np.testing.assert_allclose(median, np.clip(model.predict(df), 0.0, CAPACITY), rtol=1e-6, atol=1e-4)


def test_predict_quantiles_single_seed_collapses_to_point():
    """With n_seeds=1 the per-row std is 0, so every quantile equals the mean."""
    df = _make_block_df(5)
    model = _tiny_model().fit(df)  # n_seeds=1
    q = model.predict_quantiles(df)
    assert np.allclose(q - q[:, :1], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# harness: fully-missing block dropping vs partial-missing masking
# ---------------------------------------------------------------------------


def test_drop_fully_missing_blocks_only():
    from src.training.train_lstm import _drop_fully_missing_blocks

    df = _make_block_df(4)
    # block 1 (rows 24..47): make ALL labels missing -> should be dropped.
    df.loc[24:47, "target"] = np.nan
    # block 2 (rows 48..71): make ONE label missing -> block kept, masked.
    df.loc[50, "target"] = np.nan

    filtered, n_dropped = _drop_fully_missing_blocks(df)
    assert n_dropped == 1
    assert filtered[BLOCK_COL].nunique() == 3
    # the partially-missing block survives with its one masked NaN intact.
    assert int(filtered["target"].isna().sum()) == 1
