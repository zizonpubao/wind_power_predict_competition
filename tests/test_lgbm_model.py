"""Unit tests for src/models/lgbm_model.py (GroupLGBMModel wrapper).

Uses a tiny synthetic tabular dataset -- not the real feature parquets (that
end-to-end run is covered by actually executing src/training/train_baseline.py,
not by this suite).
"""
import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.lgbm_model import (
    AsymmetricSquaredObjective,
    GroupLGBMModel,
    asymmetric_squared_error_grad_hess,
)

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


# --- Asymmetric squared-error objective (reports/eda/ficr_gap_diagnosis.md #2) ---


def test_asymmetric_grad_hess_alpha_half_matches_standard_squared_error():
    """alpha=0.5 must reduce exactly to plain squared error: grad=r, hess=1
    (r = pred - actual), so it's a continuous degradation of the symmetric
    baseline rather than an on/off switch.
    """
    y_true = np.array([10.0, 5.0, -3.0, 0.0])
    y_pred = np.array([12.0, 4.0, -3.0, 2.0])
    residual = y_pred - y_true

    grad, hess = asymmetric_squared_error_grad_hess(y_true, y_pred, alpha=0.5)

    np.testing.assert_allclose(grad, residual)
    np.testing.assert_allclose(hess, np.ones_like(residual))


def test_asymmetric_grad_hess_penalizes_underprediction_more_when_alpha_gt_half():
    """For alpha > 0.5, an under-prediction residual (pred < actual, r < 0)
    must get a strictly larger-magnitude gradient/hessian than the
    same-magnitude over-prediction residual -- this is the whole point of the
    asymmetry (CLAUDE.md-linked systematic under-prediction bias).
    """
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([90.0, 110.0])  # residuals: -10 (under), +10 (over)
    alpha = 0.8

    grad, hess = asymmetric_squared_error_grad_hess(y_true, y_pred, alpha=alpha)

    grad_under, grad_over = grad
    hess_under, hess_over = hess

    assert abs(grad_under) > abs(grad_over)
    assert hess_under > hess_over
    # exact expected values: coef=alpha for r<0, coef=(1-alpha) for r>=0
    np.testing.assert_allclose(grad, [2 * alpha * -10.0, 2 * (1 - alpha) * 10.0])
    np.testing.assert_allclose(hess, [2 * alpha, 2 * (1 - alpha)])


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_asymmetric_grad_hess_rejects_alpha_outside_open_unit_interval(bad_alpha):
    with pytest.raises(ValueError):
        asymmetric_squared_error_grad_hess(np.array([1.0]), np.array([2.0]), alpha=bad_alpha)


def test_asymmetric_squared_objective_call_matches_pure_function():
    y_true = np.array([5.0, 10.0, 15.0])
    y_pred = np.array([4.0, 12.0, 15.0])
    alpha = 0.65

    obj = AsymmetricSquaredObjective(alpha)
    grad, hess = obj(y_true, y_pred)
    expected_grad, expected_hess = asymmetric_squared_error_grad_hess(y_true, y_pred, alpha)

    np.testing.assert_allclose(grad, expected_grad)
    np.testing.assert_allclose(hess, expected_hess)


def test_asymmetric_squared_objective_rejects_alpha_outside_open_unit_interval():
    with pytest.raises(ValueError):
        AsymmetricSquaredObjective(1.0)


def test_grouplgbmmodel_with_asymmetry_alpha_trains_predicts_and_pickles(tmp_path):
    """asymmetry_alpha!=None should swap in the custom objective, still fit/
    predict normally, and -- since AsymmetricSquaredObjective is a picklable
    module-level class rather than a closure -- survive a joblib round trip
    (needed for experiments/<run_id>/model_<group>.joblib artifacts).
    """
    X, y = _make_synthetic_data()

    model = GroupLGBMModel(capacity_kwh=21_600.0, asymmetry_alpha=0.7, n_estimators=50)
    assert isinstance(model.params["objective"], AsymmetricSquaredObjective)
    assert model.params["objective"].alpha == pytest.approx(0.7)

    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)

    dump_path = tmp_path / "asym_model.joblib"
    joblib.dump(model, dump_path)
    loaded = joblib.load(dump_path)
    np.testing.assert_allclose(loaded.predict(X), preds)


def test_grouplgbmmodel_default_asymmetry_alpha_none_keeps_builtin_objective():
    """Default (asymmetry_alpha=None) must not inject a custom objective at
    all -- unchanged behavior from before this parameter existed (plain
    LightGBM built-in regression objective).
    """
    model = GroupLGBMModel(capacity_kwh=21_600.0, n_estimators=10)
    assert model.asymmetry_alpha is None
    assert "objective" not in model.params


def test_asymmetry_alpha_sets_feature_pre_filter_false_by_default():
    """LightGBM's default feature_pre_filter=True combined with a custom
    fobj objective can crash on small folds/aggressive min_child_samples
    (``LightGBMError: train_data->num_features() > 0``, reproduced directly
    against real feature-table data during development of this asymmetric
    objective -- the crash occurs even at alpha=0.5, confirming it's a custom-
    objective engine quirk, not a bug in this asymmetric loss's math). The
    model must default feature_pre_filter to False whenever asymmetry_alpha
    is set, without overriding an explicit user choice.
    """
    model = GroupLGBMModel(capacity_kwh=21_600.0, asymmetry_alpha=0.7, n_estimators=10)
    assert model.params["feature_pre_filter"] is False

    model_explicit = GroupLGBMModel(
        capacity_kwh=21_600.0, asymmetry_alpha=0.7, n_estimators=10, feature_pre_filter=True
    )
    assert model_explicit.params["feature_pre_filter"] is True

    model_default_objective = GroupLGBMModel(capacity_kwh=21_600.0, n_estimators=10)
    assert "feature_pre_filter" not in model_default_objective.params


def test_higher_asymmetry_alpha_shifts_constant_fit_upward():
    """With a target that has no learnable relationship to the features (so
    every tree's best single-leaf fit is just a constant), a higher alpha
    must push the fitted constant strictly upward relative to a lower alpha
    -- directly demonstrates alpha>0.5 counteracts under-prediction bias
    (reports/eda/ficr_gap_diagnosis.md's motivating finding), isolated from
    any tree-splitting behavior.
    """
    rng = np.random.RandomState(0)
    n = 2000
    X = pd.DataFrame(rng.rand(n, 1), columns=["f0"])
    y = pd.Series(rng.normal(100.0, 20.0, size=n))

    means = {}
    for alpha in (0.5, 0.7, 0.9):
        kwargs = dict(n_estimators=200, num_leaves=2, learning_rate=0.1, min_child_samples=1)
        if alpha != 0.5:
            kwargs["asymmetry_alpha"] = alpha
        model = GroupLGBMModel(capacity_kwh=1_000_000.0, **kwargs)
        model.fit(X, y)
        means[alpha] = float(np.mean(model.predict(X)))

    assert means[0.5] == pytest.approx(y.mean(), abs=1.0)
    assert means[0.7] > means[0.5]
    assert means[0.9] > means[0.7]
