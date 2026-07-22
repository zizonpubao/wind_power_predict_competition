"""Thin CatBoost regressor wrapper for a single KPX group's model.

Mirrors ``src/models/lgbm_model.py``'s ``GroupLGBMModel`` interface as closely
as CatBoost's sklearn-style API allows: ``__init__(capacity_kwh,
**catboost_params)``, ``.fit(X, y, eval_set=None, early_stopping_rounds=None)``,
``.predict(X)`` (clips to ``[0, capacity_kwh * 1.01]``, same post-processing
rule as the LightGBM wrapper -- see CLAUDE.md section 4), and a
``.best_iteration_`` attribute.

Default hyperparameters (``DEFAULT_PARAMS``) are hand-picked in the same
spirit as the LightGBM baseline's, for a ~20-26k-row / ~65-67-pruned-feature
CPU regression problem, not yet hyperparameter-searched (that's
``src/training/train_catboost.py``'s job). ``bootstrap_type="Bernoulli"`` is
set explicitly because CatBoost's default bootstrap (``Bayesian``) does not
accept a ``subsample`` fraction at all -- ``Bernoulli`` is the standard way to
get LightGBM/XGBoost-style row subsampling in CatBoost.

``best_iteration_`` convention (important for parity with LightGBM/XGBoost)
-----------------------------------------------------------------------------
CatBoost's ``get_best_iteration()`` is, like XGBoost's ``best_iteration``, a
**0-indexed** round number (``tree_count_ == get_best_iteration() + 1``,
empirically verified), not the actual tree count LightGBM's
``best_iteration_`` already is. To let ``src/training`` reuse the same
"average CV folds' best_iteration_, use it as n_estimators for the final
full-data refit" pattern across all three model types without special-casing
any of them, this wrapper stores ``self.best_iteration_ =
get_best_iteration() + 1`` (the actual tree count), NOT the raw 0-indexed
value.
"""
from __future__ import annotations

from typing import Any

import catboost as cb
import numpy as np
import pandas as pd

DEFAULT_PARAMS: dict[str, Any] = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "colsample_bylevel": 0.8,
    "random_seed": 42,
    "loss_function": "RMSE",
    "thread_count": -1,
    "verbose": False,
    "allow_writing_files": False,
}

DEFAULT_EARLY_STOPPING_ROUNDS = 50


class GroupCatBoostModel:
    """CatBoost regressor for one KPX group, with capacity-aware prediction clipping.

    Parameters
    ----------
    capacity_kwh: the group's 1-hour-equivalent installed capacity in kWh
        (``configs.paths.GROUP_CAPACITY_KWH[kpx_group]``). Predictions are
        clipped to ``[0, capacity_kwh * 1.01]``.
    **catboost_params: overrides merged on top of ``DEFAULT_PARAMS`` and
        passed straight through to ``catboost.CatBoostRegressor``.
    """

    def __init__(self, capacity_kwh: float, **catboost_params: Any):
        self.capacity_kwh = float(capacity_kwh)
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **catboost_params}
        self.model_: cb.CatBoostRegressor | None = None
        self.best_iteration_: int | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    ) -> "GroupCatBoostModel":
        """Fit a fresh ``CatBoostRegressor``.

        If ``eval_set=(X_val, y_val)`` is given, trains with
        ``early_stopping_rounds`` + ``use_best_model=True`` against it (so the
        deployed model's trees are truncated at the best round, matching
        LightGBM/XGBoost early-stopping behavior); ``best_iteration_`` is then
        recorded as the actual tree count (see module docstring). Otherwise
        trains for the full configured ``iterations`` and ``best_iteration_``
        is left ``None``.

        A fresh ``CatBoostRegressor`` is constructed on every call (mirroring
        ``GroupXGBModel``) so repeated ``.fit()`` calls on the same wrapper
        instance never warm-start from a previous fit's trees.
        """
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            fit_kwargs["use_best_model"] = True

        self.model_ = cb.CatBoostRegressor(**self.params)
        self.model_.fit(X, y, **fit_kwargs)

        if eval_set is not None:
            self.best_iteration_ = int(self.model_.get_best_iteration()) + 1
        else:
            self.best_iteration_ = None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict and clip to ``[0, capacity_kwh * 1.01]`` (CLAUDE.md post-processing)."""
        preds = self.model_.predict(X)
        return np.clip(preds, 0.0, self.capacity_kwh * 1.01)
