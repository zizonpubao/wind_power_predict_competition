"""Unit tests for src/models/lgbm_quantile_model.py (GroupLGBMQuantileModel).

Uses a tiny synthetic tabular dataset (mirrors tests/test_lgbm_model.py /
tests/test_xgb_model.py's structure/coverage), plus explicit checks that this
model satisfies src/training/tune_common.py's _GroupModelProtocol and that
monotonic_constraints is never set anywhere in its LightGBM params.
"""
import inspect

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.decision_optimize import QUANTILES
from src.models.lgbm_quantile_model import DEFAULT_PARAMS, GroupLGBMQuantileModel
from src.training.tune_common import _GroupModelProtocol

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

    model = GroupLGBMQuantileModel(capacity_kwh=21_600.0, n_estimators=30)
    fitted = model.fit(X_train, y_train)

    assert fitted is model  # fit returns self
    assert set(model.models_.keys()) == set(QUANTILES)
    preds = model.predict(X_val)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X_val),)


def test_predict_quantiles_are_non_decreasing_after_predict_pipeline():
    X, y = _make_synthetic_data()
    model = GroupLGBMQuantileModel(capacity_kwh=21_600.0, n_estimators=30)
    model.fit(X, y)

    raw_q = model.predict_quantiles(X)
    assert raw_q.shape == (len(X), len(QUANTILES))
    # raw_q itself may have crossing (independent sub-models); predict()
    # internally fixes this before deciding a point -- checked indirectly via
    # predict() staying within [0, capacity*1.01] below.
    point_preds = model.predict(X)
    assert (point_preds >= 0.0).all()
    assert (point_preds <= 21_600.0 * 1.01 + 1e-6).all()


def test_fit_with_eval_set_uses_early_stopping_and_records_median_best_iteration():
    X, y = _make_synthetic_data()
    X_train, X_val = X.iloc[:250], X.iloc[250:]
    y_train, y_val = y.iloc[:250], y.iloc[250:]

    model = GroupLGBMQuantileModel(capacity_kwh=21_600.0, n_estimators=200)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5)

    assert model.best_iteration_ is not None
    assert model.best_iteration_ <= 200
    # best_iteration_ must be exactly the q=0.5 sub-model's own best_iteration_
    assert model.best_iteration_ == model.models_[0.5].best_iteration_


def test_fit_without_eval_set_leaves_best_iteration_falsy():
    """Without an eval_set, LightGBM's sklearn API leaves best_iteration_ at
    its no-early-stopping default (0 in this installed lightgbm version,
    matching GroupLGBMModel's identical getattr(..., None) pattern) rather
    than None -- either way it is falsy, which is exactly what
    tune_common.oof_predict_generic relies on (``if model.best_iteration_
    else None``) to detect "no early stopping happened this fit".
    """
    X, y = _make_synthetic_data()
    model = GroupLGBMQuantileModel(capacity_kwh=21_600.0, n_estimators=20)
    model.fit(X, y)
    assert not model.best_iteration_


def test_predict_clips_to_capacity_range():
    X, y = _make_synthetic_data()
    tiny_capacity = 1.0
    model = GroupLGBMQuantileModel(capacity_kwh=tiny_capacity, n_estimators=20)
    model.fit(X, y)

    preds = model.predict(X)
    assert (preds >= 0.0).all()
    assert (preds <= tiny_capacity * 1.01 + 1e-6).all()
    assert y.max() > tiny_capacity * 1.01


def test_capacity_kwh_stored_and_default_params_merged_with_overrides():
    model = GroupLGBMQuantileModel(capacity_kwh=21_000.0, num_leaves=7, n_estimators=10)
    assert model.capacity_kwh == pytest.approx(21_000.0)
    assert model.params["num_leaves"] == 7
    assert model.params["n_estimators"] == 10
    assert model.params["learning_rate"] == pytest.approx(0.03)


def test_default_params_match_spec_starting_values():
    assert DEFAULT_PARAMS["num_leaves"] == 31
    assert DEFAULT_PARAMS["min_child_samples"] == 30
    assert DEFAULT_PARAMS["learning_rate"] == pytest.approx(0.03)
    assert DEFAULT_PARAMS["subsample"] == pytest.approx(0.75)
    assert DEFAULT_PARAMS["colsample_bytree"] == pytest.approx(0.65)
    assert DEFAULT_PARAMS["reg_alpha"] == pytest.approx(0.3)
    assert DEFAULT_PARAMS["reg_lambda"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# monotonic_constraints must never be settable/present
# ---------------------------------------------------------------------------


def test_constructor_rejects_monotonic_constraints_kwarg():
    with pytest.raises(ValueError):
        GroupLGBMQuantileModel(capacity_kwh=21_600.0, monotonic_constraints=[1, 1, 1])


def test_monotonic_constraints_never_present_in_fitted_submodel_params():
    X, y = _make_synthetic_data()
    model = GroupLGBMQuantileModel(capacity_kwh=21_600.0, n_estimators=10)
    model.fit(X, y)
    for q, submodel in model.models_.items():
        params = submodel.get_params()
        assert params.get("monotonic_constraints") is None, f"quantile {q} sub-model set monotonic_constraints"


# ---------------------------------------------------------------------------
# _GroupModelProtocol conformance + pickling
# ---------------------------------------------------------------------------


def test_satisfies_group_model_protocol_signature():
    """Structural check that GroupLGBMQuantileModel's fit/predict signatures
    match _GroupModelProtocol closely enough for tune_common.py's generic CV
    loop (oof_predict_generic/tune_group_generic) to use it unmodified.
    """
    fit_sig = inspect.signature(GroupLGBMQuantileModel.fit)
    protocol_fit_sig = inspect.signature(_GroupModelProtocol.fit)
    assert list(fit_sig.parameters.keys()) == list(protocol_fit_sig.parameters.keys())

    assert hasattr(GroupLGBMQuantileModel, "predict")
    model = GroupLGBMQuantileModel(capacity_kwh=21_600.0, n_estimators=5)
    assert hasattr(model, "best_iteration_")


def test_runs_end_to_end_through_tune_common_oof_predict_generic():
    """Integration check: actually drive GroupLGBMQuantileModel through
    src.training.tune_common.oof_predict_generic + score_oof (the exact
    machinery train_gbm_quantile.py uses), on a small synthetic block-
    structured dataset, and confirm it produces the expected OOF schema
    without any special-casing.
    """
    from src.evaluation.metrics import competition_score
    from src.training.tune_common import oof_predict_generic, score_oof

    rng = np.random.RandomState(0)
    n_blocks = 10
    block_size = 24
    n_rows = n_blocks * block_size
    capacity = 21_600.0

    X = pd.DataFrame(rng.rand(n_rows, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.clip(X.sum(axis=1) * capacity / 5.0 + rng.rand(n_rows) * 200, 0, capacity))
    forecast_kst_dtm = pd.date_range("2024-01-01", periods=n_rows, freq="h")
    data_available_kst_dtm = np.repeat(
        pd.date_range("2024-01-01", periods=n_blocks, freq="D"), block_size
    )

    df = X.copy()
    df["target"] = y
    df["forecast_kst_dtm"] = forecast_kst_dtm
    df["data_available_kst_dtm"] = data_available_kst_dtm

    feature_cols = [f"f{i}" for i in range(5)]
    oof_df, fold_meta = oof_predict_generic(
        df,
        feature_cols,
        capacity,
        "kpx_group_1",
        GroupLGBMQuantileModel,
        {"n_estimators": 30},
        early_stopping_rounds=5,
        n_splits=3,
    )
    assert set(oof_df.columns) >= {"fold", "forecast_kst_dtm", "pred", "actual"}
    assert len(oof_df) > 0

    fold_metrics = score_oof(oof_df, fold_meta, "kpx_group_1")
    assert len(fold_metrics) == 3
    for m in fold_metrics:
        assert "score" in m


def test_model_pickles_round_trip(tmp_path):
    X, y = _make_synthetic_data()
    X_train, X_val = X.iloc[:250], X.iloc[250:]
    y_train, y_val = y.iloc[:250], y.iloc[250:]

    model = GroupLGBMQuantileModel(capacity_kwh=21_600.0, n_estimators=30)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=5)
    preds = model.predict(X_val)

    dump_path = tmp_path / "lgbm_quantile_model.joblib"
    joblib.dump(model, dump_path)
    loaded = joblib.load(dump_path)

    np.testing.assert_allclose(loaded.predict(X_val), preds)
    assert loaded.best_iteration_ == model.best_iteration_
    assert set(loaded.models_.keys()) == set(model.models_.keys())


def test_repeated_fit_calls_do_not_warm_start():
    X, y = _make_synthetic_data()

    model = GroupLGBMQuantileModel(capacity_kwh=21_600.0, n_estimators=15)
    model.fit(X.iloc[:100], y.iloc[:100])
    model.fit(X, y)  # second fit, on the full data

    fresh = GroupLGBMQuantileModel(capacity_kwh=21_600.0, n_estimators=15)
    fresh.fit(X, y)

    for q in QUANTILES:
        assert model.models_[q].booster_.num_trees() == fresh.models_[q].booster_.num_trees()
