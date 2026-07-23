"""Unit tests for src/models/transformer_model.py (GroupTransformerModel).

Kept small/fast: tiny synthetic block data, 1 seed, a couple of epochs. The
block/sequence leakage-safety is already covered by the Phase C torch_common
tests, so these focus on the Transformer wrapper's own contract, identical to
the LSTM's:
  - predict output aligns to input row order/count exactly,
  - save/load round-trips to identical predictions,
  - the per-fold scaler is fit on train rows only (leakage),
  - partially-missing-label blocks are kept intact (masked),
and Transformer-specific structure:
  - the encoder really has n_layers layers,
  - it exposes NO mixup knob (v14 spec: mixup is LSTM-group3-only).
"""
import numpy as np
import pandas as pd
import pytest

from src.models.transformer_model import GroupTransformerModel, TransformerPointModel
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


def _tiny_model(**kw) -> GroupTransformerModel:
    return GroupTransformerModel(CAPACITY, FEATURES, n_seeds=1, max_epochs=2, patience=2, **kw)


def test_predict_output_aligns_to_input_rows():
    df = _make_block_df(6)
    model = _tiny_model().fit(df)

    preds = model.predict(df)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(df),)
    assert (preds >= 0.0).all()
    assert (preds <= CAPACITY * 1.01 + 1e-6).all()


def test_predict_is_row_order_equivariant():
    """Reordering the input rows must reorder the outputs the same way."""
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

    path = tmp_path / "transformer_model.joblib"
    model.save(path)
    loaded = GroupTransformerModel.load(path)

    np.testing.assert_allclose(loaded.predict(df), preds, rtol=1e-6, atol=1e-6)
    assert loaded.feature_cols == model.feature_cols
    assert len(loaded.state_dicts_) == len(model.state_dicts_)


def test_scaler_fit_on_train_rows_only():
    df = _make_block_df(8)
    model = _tiny_model().fit(df)
    np.testing.assert_allclose(
        model.scaler.mean_, np.nanmean(df[FEATURES].to_numpy(float), axis=0)
    )


def test_partially_missing_block_is_kept_and_trains():
    """A block with a few missing labels stays a full 24-row sequence and
    training still runs (missing hours masked in loss)."""
    df = _make_block_df(5)
    df.loc[3, "target"] = np.nan
    df.loc[27, "target"] = np.nan
    model = _tiny_model().fit(df)  # must not raise
    preds = model.predict(df)
    assert preds.shape == (len(df),)
    assert np.isfinite(preds).all()


def test_predict_before_fit_raises():
    df = _make_block_df(2)
    model = _tiny_model()
    with pytest.raises(RuntimeError):
        model.predict(df)


def test_encoder_has_requested_layer_count():
    """n_layers really controls the number of encoder layers."""
    for n_layers in (1, 3):
        module = TransformerPointModel(len(FEATURES), num_layers=n_layers)
        assert len(module.encoder.layers) == n_layers


def test_no_mixup_knob():
    """v14 spec: the Transformer has no mixup (that is an LSTM-group3-only
    rule). Passing apply_mixup must be rejected, and no such attribute exists."""
    model = _tiny_model()
    assert not hasattr(model, "apply_mixup")
    with pytest.raises(TypeError):
        GroupTransformerModel(CAPACITY, FEATURES, apply_mixup=True)
