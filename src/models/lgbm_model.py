"""Thin LightGBM regressor wrapper for a single KPX group's baseline model.

Kept intentionally simple (CLAUDE.md section 9 / task spec): default
hyperparameters are hand-picked for a small-ish tabular problem (~26k train
rows, ~140 features per group) rather than tuned via search -- that is future
work for a dedicated hyperparameter-search pass, not this baseline.

Regularization choices (num_leaves=15, min_child_samples=30, subsample /
colsample_bytree=0.8, small L1/L2) all lean toward under- rather than
over-fitting given the rows:features ratio (~180:1) is not generous for
gradient boosting. ``fit`` optionally takes an ``eval_set`` for early
stopping; ``predict`` clips to ``[0, capacity_kwh * 1.01]`` per CLAUDE.md
section 4's post-processing recommendation (installed-capacity overshoot in
the raw labels is real but tiny, <=0.62%, so this is a safety clamp on model
output, not a data-fidelity assumption).
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

DEFAULT_EARLY_STOPPING_ROUNDS = 50


class GroupLGBMModel:
    """LightGBM regressor for one KPX group, with capacity-aware prediction clipping.

    Parameters
    ----------
    capacity_kwh: the group's 1-hour-equivalent installed capacity in kWh
        (``configs.paths.GROUP_CAPACITY_KWH[kpx_group]``). Predictions are
        clipped to ``[0, capacity_kwh * 1.01]``.
    **lgbm_params: overrides merged on top of ``DEFAULT_PARAMS`` and passed
        straight through to ``lightgbm.LGBMRegressor``.
    """

    def __init__(self, capacity_kwh: float, **lgbm_params: Any):
        self.capacity_kwh = float(capacity_kwh)
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **lgbm_params}
        self.model_ = lgb.LGBMRegressor(**self.params)
        self.best_iteration_: int | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    ) -> "GroupLGBMModel":
        """Fit the underlying LGBMRegressor.

        If ``eval_set=(X_val, y_val)`` is given, trains with early stopping
        against it (``best_iteration_`` is recorded); otherwise trains for the
        full configured ``n_estimators``.
        """
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["eval_metric"] = "l1"
            fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

        self.model_.fit(X, y, **fit_kwargs)
        self.best_iteration_ = getattr(self.model_, "best_iteration_", None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and clip to ``[0, capacity_kwh * 1.01]`` (CLAUDE.md post-processing)."""
        preds = self.model_.predict(X)
        return np.clip(preds, 0.0, self.capacity_kwh * 1.01)
