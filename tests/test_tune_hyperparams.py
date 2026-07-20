"""Light smoke test for src/training/tune_hyperparams.py's Optuna objective.

Does not test tuning quality (that would need many trials + a real held-out
eval) -- just confirms the objective function, wired to the real block-aware
CV splitter (BlockTimeSeriesSplit) and the official competition_score, runs
end-to-end without crashing and returns a finite score for a tiny
(n_trials=2) study on a time-sliced small subset of one group's real feature
table (sliced small purely to keep the test fast).
"""
import numpy as np
import optuna
import pandas as pd
import pytest

from configs.paths import DATA_PROCESSED_DIR, GROUP_CAPACITY_KWH
from src.training.train_baseline import _get_feature_cols
from src.training.tune_hyperparams import _make_objective

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
