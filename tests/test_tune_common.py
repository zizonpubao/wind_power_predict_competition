"""Light smoke test for src/training/tune_common.py's model-agnostic Optuna
objective + OOF helpers, exercised against both GroupXGBModel and
GroupCatBoostModel.

Does not test tuning quality (that needs many trials + a real held-out eval)
-- just confirms tune_group_generic, wired to the real block-aware CV
splitter (BlockTimeSeriesSplit) and the official competition_score, runs
end-to-end without crashing for both model types on a time-sliced small
subset of one group's real feature table (kept small purely for test speed).
"""
import numpy as np
import optuna
import pytest

from configs.paths import DATA_PROCESSED_DIR
from src.models.catboost_model import GroupCatBoostModel
from src.models.xgb_model import GroupXGBModel
from src.training.tune_common import tune_group_generic

KPX_GROUP = "kpx_group_1"
N_SPLITS = 3
N_TRIALS = 2


def _xgb_suggest(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
    }


def _catboost_suggest(trial: optuna.Trial) -> dict:
    return {
        "depth": trial.suggest_int("depth", 4, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.2, log=True),
    }


@pytest.fixture(autouse=True)
def _skip_if_no_feature_table():
    path = DATA_PROCESSED_DIR / f"features_{KPX_GROUP}_train.parquet"
    if not path.exists():
        pytest.skip(f"processed feature table not found: {path}")


def test_tune_group_generic_xgb_runs_end_to_end_and_writes_expected_shape():
    result = tune_group_generic(
        KPX_GROUP,
        GroupXGBModel,
        _xgb_suggest,
        "n_estimators",
        n_trials=N_TRIALS,
        n_splits=N_SPLITS,
        feature_set="pruned",
        early_stopping_rounds=20,
        sampler_seed=42,
    )

    assert np.isfinite(result["best_cv_score"])
    assert set(result["oof_df"].columns) >= {"fold", "forecast_kst_dtm", "pred", "actual"}
    assert len(result["fold_metrics"]) == N_SPLITS
    assert result["final_model"].best_iteration_ is None  # full-data refit has no eval_set
    assert isinstance(result["final_params"]["n_estimators"], int)


def test_tune_group_generic_catboost_runs_end_to_end_and_writes_expected_shape():
    result = tune_group_generic(
        KPX_GROUP,
        GroupCatBoostModel,
        _catboost_suggest,
        "iterations",
        n_trials=N_TRIALS,
        n_splits=N_SPLITS,
        feature_set="pruned",
        early_stopping_rounds=20,
        sampler_seed=42,
    )

    assert np.isfinite(result["best_cv_score"])
    assert set(result["oof_df"].columns) >= {"fold", "forecast_kst_dtm", "pred", "actual"}
    assert len(result["fold_metrics"]) == N_SPLITS
    assert result["final_model"].best_iteration_ is None  # full-data refit has no eval_set
    assert isinstance(result["final_params"]["iterations"], int)
