"""Thin XGBoost regressor wrapper for a single KPX group's model.

Mirrors ``src/models/lgbm_model.py``'s ``GroupLGBMModel`` interface as closely
as XGBoost's sklearn API allows, so ``src/training``'s CV/refit loops can
treat both model types almost interchangeably: ``__init__(capacity_kwh,
**xgb_params)``, ``.fit(X, y, eval_set=None, early_stopping_rounds=None)``,
``.predict(X)`` (clips to ``[0, capacity_kwh * 1.01]``, same post-processing
rule as the LightGBM wrapper -- see CLAUDE.md section 4), and a
``.best_iteration_`` attribute.

Default hyperparameters (``DEFAULT_PARAMS``) are hand-picked in the same
spirit as the LightGBM baseline's: modest depth/subsampling for a
~20-26k-row / ~65-67-pruned-feature CPU regression problem, not yet
hyperparameter-searched (that's ``src/training/train_xgb.py``'s job).
``tree_method="hist"`` is used for CPU speed (no GPU detected in this dev
environment, per CLAUDE.md section 6); XGBoost's ``hist`` method runs
entirely on CPU when no GPU device is configured.

``best_iteration_`` convention (important for parity with LightGBM)
---------------------------------------------------------------------
XGBoost's sklearn API exposes ``model_.best_iteration`` as a **0-indexed**
round number (e.g. ``162`` means rounds ``0..162``, i.e. 163 trees), whereas
LightGBM's ``best_iteration_`` is already the actual tree *count* (directly
reusable as a future ``n_estimators``). To let ``src/training`` code reuse the
exact same "average CV folds' best_iteration_, use it as n_estimators for the
final full-data refit" pattern across both model types without special-casing
either one, this wrapper stores ``self.best_iteration_ = model_.best_iteration
+ 1`` (the actual tree count), NOT the raw 0-indexed attribute.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "verbosity": 0,
}

DEFAULT_EARLY_STOPPING_ROUNDS = 50


class GroupXGBModel:
    """XGBoost regressor for one KPX group, with capacity-aware prediction clipping.

    Parameters
    ----------
    capacity_kwh: the group's 1-hour-equivalent installed capacity in kWh
        (``configs.paths.GROUP_CAPACITY_KWH[kpx_group]``). Predictions are
        clipped to ``[0, capacity_kwh * 1.01]``.
    **xgb_params: overrides merged on top of ``DEFAULT_PARAMS`` and passed
        straight through to ``xgboost.XGBRegressor``.
    """

    def __init__(self, capacity_kwh: float, **xgb_params: Any):
        self.capacity_kwh = float(capacity_kwh)
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **xgb_params}
        self.model_: xgb.XGBRegressor | None = None
        self.best_iteration_: int | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    ) -> "GroupXGBModel":
        """Fit a fresh ``XGBRegressor``.

        If ``eval_set=(X_val, y_val)`` is given, ``early_stopping_rounds`` is
        set on the constructor (XGBoost>=2.0's required location for it, not
        ``.fit()``) and training stops early against that eval set;
        ``best_iteration_`` is then recorded as the actual tree count (see
        module docstring). Otherwise trains for the full configured
        ``n_estimators`` and ``best_iteration_`` is left ``None``.

        A fresh ``XGBRegressor`` is constructed on every call (mirroring
        ``GroupLGBMModel``, whose ``lgb.LGBMRegressor`` is built once in
        ``__init__`` but is stateless until ``.fit()``) so repeated ``.fit()``
        calls -- e.g. one CV fold followed by a full-data refit reusing the
        same wrapper instance -- never accidentally warm-start from a
        previous fold's trees.
        """
        params = dict(self.params)
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            params["early_stopping_rounds"] = early_stopping_rounds
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["verbose"] = False
        else:
            params.pop("early_stopping_rounds", None)

        self.model_ = xgb.XGBRegressor(**params)
        self.model_.fit(X, y, **fit_kwargs)

        raw_best_iteration = getattr(self.model_, "best_iteration", None)
        self.best_iteration_ = int(raw_best_iteration) + 1 if raw_best_iteration is not None else None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and clip to ``[0, capacity_kwh * 1.01]`` (CLAUDE.md post-processing).

        No explicit ``iteration_range`` is passed: XGBoost's sklearn API
        already restricts prediction to the best round automatically whenever
        ``best_iteration`` is set on the fitted booster (i.e. whenever this
        fit used early stopping).
        """
        preds = self.model_.predict(X)
        return np.clip(preds, 0.0, self.capacity_kwh * 1.01)
