"""Unit tests for src/models/lgbm_model.py (GroupLGBMModel wrapper).

Uses a tiny synthetic tabular dataset -- not the real feature parquets (that
end-to-end run is covered by actually executing src/training/train_baseline.py,
not by this suite).
"""
import numpy as np
import pandas as pd
import pytest

from src.models.lgbm_model import GroupLGBMModel

N_ROWS = 300
N_FEATURES = 8


def _make_synthetic_data(n_rows: int = N_ROWS, n_features: int = N_FEATURES, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.rand(n_rows, n_features), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(X.sum(axis=1) * 1000 + rng.rand(n_rows) * 5, name="target")
    return X, y


def test_fit_predict_runs_and_returns_expected_shape():
    X, y = _make_synthetic_data()
    X_train, X_val = X.iloc[:250], X.iloc[250:]
    y_train, y_val = y.iloc[:250], y.iloc[250:]

    model = GroupLGBMModel(capacity_kwh=21_600.0, n_estimators=50)
    fitted = model.fit(X_train, y_train)

    assert fitted is model  # fit returns self
    preds = model.predict(X_val)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_val),)


def test_fit_with_eval_set_uses_early_stopping():
    X, y = _make_synthetic_data()
    X_train, X_val = X.iloc[:250], X.iloc[250:]
    y_train, y_val = y.iloc[:250], y.iloc[250:]

    model = GroupLGBMModel(capacity_kwh=21_600.0, n_estimators=500)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5)

    # best_iteration_ should be recorded (an int) when an eval_set is supplied.
    assert model.best_iteration_ is not None
    assert model.best_iteration_ <= 500


def test_predict_clips_to_capacity_range():
    X, y = _make_synthetic_data()

    # Force absurdly small capacity so every prediction must be clipped down
    # to [0, capacity*1.01], regardless of what the raw model predicts.
    tiny_capacity = 1.0
    model = GroupLGBMModel(capacity_kwh=tiny_capacity, n_estimators=20)
    model.fit(X, y)

    preds = model.predict(X)
    assert (preds >= 0.0).all()
    assert (preds <= tiny_capacity * 1.01 + 1e-9).all()
    # sanity: the raw (unclipped) target scale is far above the tiny capacity,
    # so clipping must actually be engaging, not a no-op.
    assert y.max() > tiny_capacity * 1.01


def test_predict_never_returns_negative_even_with_negative_targets():
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.rand(200, 5), columns=[f"f{i}" for i in range(5)])
    # Targets centered near zero with some negative values, to check the
    # lower clip bound (0) engages even though nothing forces predictions
    # positive by construction.
    y = pd.Series(X.sum(axis=1) * 10 - 30)
    assert (y < 0).any()

    model = GroupLGBMModel(capacity_kwh=21_600.0, n_estimators=30)
    model.fit(X, y)
    preds = model.predict(X)
    assert (preds >= 0.0).all()


def test_capacity_kwh_stored_and_default_params_merged_with_overrides():
    model = GroupLGBMModel(capacity_kwh=21_000.0, num_leaves=7, n_estimators=10)
    assert model.capacity_kwh == pytest.approx(21_000.0)
    assert model.params["num_leaves"] == 7
    assert model.params["n_estimators"] == 10
    # untouched defaults should still be present
    assert model.params["learning_rate"] == pytest.approx(0.03)
