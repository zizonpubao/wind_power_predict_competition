"""Light smoke test for src/training/tune_hyperparams.py's Optuna objective.

Does not test tuning quality (that would need many trials + a real held-out
eval) -- just confirms the objective function, wired to the real block-aware
CV splitter (BlockTimeSeriesSplit) and the official competition_score, runs
end-to-end without crashing and returns a finite score for a tiny
(n_trials=2) study on a time-sliced small subset of one group's real feature
table (sliced small purely to keep the test fast).
"""
import json

import numpy as np
import optuna
import pandas as pd
import pytest

from configs.paths import DATA_PROCESSED_DIR, GROUP_CAPACITY_KWH
from src.models.calibration import PredictionCalibrator
from src.training.train_baseline import KPX_GROUPS, _get_feature_cols
from src.training.tune_hyperparams import _evaluate_and_save_calibrators, _make_objective

KPX_GROUP = "kpx_group_1"
N_SPLITS = 3
N_TRIALS = 2


def _small_time_sliced_subset() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / f"features_{KPX_GROUP}_train.parquet"
    if not path.exists():
        pytest.skip(f"processed feature table not found: {path}")

    df = pd.read_parquet(path)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    # Keep only the earliest few forecast blocks (data_available_kst_dtm
    # issuances) so the CV loop -- and each trial's LightGBM fits -- stay
    # fast; BlockTimeSeriesSplit just needs >= n_splits + 1 distinct blocks.
    blocks = sorted(df["data_available_kst_dtm"].unique())[: (N_SPLITS + 1) * 3]
    df = df[df["data_available_kst_dtm"].isin(blocks)].reset_index(drop=True)
    return df


def test_objective_returns_finite_score_without_crashing():
    df = _small_time_sliced_subset()
    feature_cols = _get_feature_cols(df)
    capacity = GROUP_CAPACITY_KWH[KPX_GROUP]

    objective = _make_objective(df, feature_cols, capacity, KPX_GROUP, n_splits=N_SPLITS)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)

    assert len(study.trials) == N_TRIALS
    assert np.isfinite(study.best_value)
    for t in study.trials:
        assert t.value is not None
        assert np.isfinite(t.value)


def _make_consistently_biased_oof(capacity: float, n_per_fold: int = 300, n_folds: int = 5) -> pd.DataFrame:
    """Synthetic OOF frame where every fold shares the SAME systematic
    under-prediction bias -- isotonic calibration fit on any 4 folds
    generalizes well to the 5th, so leave-one-fold-out cross-fit calibration
    should genuinely improve the CV score (mirrors kpx_group_1/kpx_group_3 in
    reports/eda/group2_calibration_regression.md).
    """
    rng = np.random.RandomState(0)
    rows = []
    for fold_i in range(1, n_folds + 1):
        actual = rng.uniform(0.3 * capacity, 0.9 * capacity, size=n_per_fold)
        pred = actual * 0.8 + rng.normal(0, 50, size=n_per_fold)
        dtm = pd.date_range("2024-01-01", periods=n_per_fold, freq="h") + pd.Timedelta(hours=1000 * fold_i)
        rows.append(pd.DataFrame({"fold": fold_i, "forecast_kst_dtm": dtm, "pred": pred, "actual": actual}))
    return pd.concat(rows, ignore_index=True)


def _make_flip_signed_bias_oof(capacity: float, n_per_fold: int = 300) -> pd.DataFrame:
    """Synthetic 2-fold OOF frame where the two folds have opposite-signed
    constant biases (+1500 / -1500 kWh). A calibrator fit on fold A's data
    and applied to fold B (and vice versa) therefore *doubles* the bias
    instead of correcting it for most rows -- leave-one-fold-out cross-fit
    calibration should genuinely regress the CV score here (mirrors
    kpx_group_2 in reports/eda/group2_calibration_regression.md, where the
    isotonic curve's sign/direction is unstable/wrong for a chunk of the
    range that matters for scoring).
    """
    rng = np.random.RandomState(1)
    rows = []
    for fold_i in (1, 2):
        actual = rng.uniform(0.3 * capacity, 0.9 * capacity, size=n_per_fold)
        bias = 1500.0 if fold_i == 1 else -1500.0
        pred = actual + bias + rng.normal(0, 100, size=n_per_fold)
        dtm = pd.date_range("2024-01-01", periods=n_per_fold, freq="h") + pd.Timedelta(hours=1000 * fold_i)
        rows.append(pd.DataFrame({"fold": fold_i, "forecast_kst_dtm": dtm, "pred": pred, "actual": actual}))
    return pd.concat(rows, ignore_index=True)


def _fake_group_result(oof_df: pd.DataFrame) -> dict:
    fold_meta = [
        {"fold": int(f), "n_train": 1, "n_val": int((oof_df["fold"] == f).sum()), "best_iteration": None}
        for f in sorted(oof_df["fold"].unique())
    ]
    candidate_calibrator = PredictionCalibrator().fit(oof_df["pred"].to_numpy(), oof_df["actual"].to_numpy())
    return {"oof_df": oof_df, "fold_meta": fold_meta, "candidate_calibrator": candidate_calibrator}


def test_evaluate_and_save_calibrators_gates_per_group(tmp_path):
    """Fake scenario: kpx_group_1/kpx_group_3 have a consistent bias
    calibration genuinely fixes (should save), kpx_group_2 has a
    sign-flipping bias calibration makes worse (should be skipped) -- this is
    exactly the pattern reports/eda/group2_calibration_regression.md found in
    the real tuned run, reproduced here with synthetic data so the test does
    not depend on the real feature parquets.
    """
    all_results = {
        "kpx_group_1": _fake_group_result(_make_consistently_biased_oof(GROUP_CAPACITY_KWH["kpx_group_1"])),
        "kpx_group_2": _fake_group_result(_make_flip_signed_bias_oof(GROUP_CAPACITY_KWH["kpx_group_2"])),
        "kpx_group_3": _fake_group_result(_make_consistently_biased_oof(GROUP_CAPACITY_KWH["kpx_group_3"])),
    }

    decisions = _evaluate_and_save_calibrators(all_results, tmp_path)

    assert set(decisions.keys()) == set(KPX_GROUPS)
    assert decisions["kpx_group_1"]["saved"] is True
    assert decisions["kpx_group_2"]["saved"] is False
    assert decisions["kpx_group_3"]["saved"] is True
    assert decisions["kpx_group_1"]["delta"] > 0
    assert decisions["kpx_group_2"]["delta"] < 0
    assert decisions["kpx_group_3"]["delta"] > 0

    # Files on disk must match the per-group decision exactly.
    assert (tmp_path / "calibrator_kpx_group_1.joblib").exists()
    assert not (tmp_path / "calibrator_kpx_group_2.joblib").exists()
    assert (tmp_path / "calibrator_kpx_group_3.joblib").exists()

    decision_path = tmp_path / "calibration_decision.json"
    assert decision_path.exists()
    with open(decision_path, "r", encoding="utf-8") as f:
        saved_decisions = json.load(f)
    assert saved_decisions["kpx_group_1"]["saved"] is True
    assert saved_decisions["kpx_group_2"]["saved"] is False
    assert saved_decisions["kpx_group_3"]["saved"] is True
