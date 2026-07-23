"""9-quantile LightGBM regressor + decision-theoretic post-processing for one
KPX group -- the GBM component of the v14 pipeline reproduction (see
``.claude/plans/logical-stirring-sphinx.md`` Phase B, and the reference
hand-off code at ``C:\\Users\\heelo\\Desktop\\files\\gbm_model.py``/
``config.py``, treated as a *starting point*, not validated ground truth --
see CLAUDE.md/plan section 1's warning about that hand-off).

Trains 9 independent ``lgb.LGBMRegressor(objective="quantile", alpha=q)``
models (one per ``src.features.decision_optimize.QUANTILES`` level) and, at
predict time, feeds their 9 outputs through
``src.features.decision_optimize.decision_optimal_point_prediction`` -- which
enforces monotonicity, builds a discretized pmf, and picks the point that
maximizes an expected-utility blend of the official 1-NMAE/FICR metrics --
rather than e.g. just returning the median quantile directly.

Mirrors ``src/models/lgbm_model.py`` (``GroupLGBMModel``) and
``src/models/xgb_model.py`` (``GroupXGBModel``)'s constructor/fit/predict
shape closely enough that this satisfies
``src/training/tune_common.py``'s ``_GroupModelProtocol`` unmodified -- see
``GroupLGBMQuantileModel.predict``'s docstring.
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.decision_optimize import (
    QUANTILES,
    decision_optimal_point_prediction,
    enforce_monotonic_quantiles,
)

# Hyperparameter *starting values* per the task spec / v14 hand-off's
# GBM_PARAMS -- CLAUDE.md/plan section 1 explicitly warns these are not
# validated ground truth, just a reasonable starting point subject to this
# repo's own CV re-verification (a later tuning pass, not this file).
# n_estimators is intentionally large (a ceiling, not a target) since actual
# tree count is controlled by early stopping / the tune_common.py
# "average CV folds' best_iteration_" convention, exactly as
# GroupLGBMModel/GroupXGBModel already do.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.75,
    "subsample_freq": 1,
    "colsample_bytree": 0.65,
    "reg_alpha": 0.3,
    "reg_lambda": 0.3,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

DEFAULT_EARLY_STOPPING_ROUNDS = 50

# Which of the 9 sub-models' best_iteration_ represents the whole wrapper's
# best_iteration_ (see class docstring's "best_iteration_ convention" section).
MEDIAN_QUANTILE = 0.5

# Sample-weighting rule (v14 hand-off's VALID_HOUR_THRESHOLD): rows below 10%
# utilization get a much smaller training weight, since the official metrics
# don't score them at all (CLAUDE.md section 5's eligibility filter) -- this
# steers the quantile fits' capacity toward the rows that actually matter for
# the leaderboard score instead of spending it on near-zero-generation noise.
VALID_HOUR_UTILIZATION = 0.10
LOW_UTILIZATION_SAMPLE_WEIGHT = 0.05


class GroupLGBMQuantileModel:
    """9 independent LightGBM quantile regressors + EU-optimal decision point,
    for one KPX group's capacity-aware prediction.

    Parameters
    ----------
    capacity_kwh : the group's 1-hour-equivalent installed capacity in kWh.
    **lgbm_params : overrides merged on top of ``DEFAULT_PARAMS`` and passed
        to every one of the 9 ``lightgbm.LGBMRegressor`` sub-models (each also
        gets its own ``objective="quantile"``/``alpha=q``, not overridable via
        this kwarg -- those are exactly what defines "9 quantiles").

    monotonic_constraints -- deliberately never set
    -------------------------------------------------
    This wrapper raises ``ValueError`` if ``monotonic_constraints`` is passed
    in ``**lgbm_params`` at all. LightGBM's ``objective="quantile"`` combined
    with ``monotonic_constraints`` is a known-bad combination flagged in the
    v14 hand-off this reproduction is based on -- quantile crossing (each of
    the 9 sub-models is an independent fit, so nothing structurally keeps
    e.g. the alpha=0.35 model below the alpha=0.5 model for the same row) is
    instead corrected post-hoc via
    ``src.features.decision_optimize.enforce_monotonic_quantiles`` (a cheap,
    always-safe fix applied at predict time), rather than constraining the
    per-quantile boosting objective itself.

    best_iteration_ convention
    -----------------------------
    ``src/training/tune_common.py``'s CV/refit loop reads a single scalar
    ``best_iteration_`` per fold (averages it across folds, then uses that as
    a fixed ``n_estimators`` for the final full-data refit) -- it has no
    notion of "9 sub-models, each with its own best_iteration_". Rather than
    changing that shared, model-agnostic machinery to special-case this one
    model class, this wrapper picks one consistent representative:
    the ``q=0.5`` (median) sub-model's own ``best_iteration_``. 0.5 is always
    present in ``QUANTILES``, so this is well-defined for every call; the
    median quantile is also the most natural single "typical" round-count
    among the 9 (the extreme quantiles, e.g. 0.05/0.95, tend to need
    somewhat different numbers of rounds before their own eval metric
    plateaus, so neither extreme is a better representative choice).

    Pickling
    ---------
    Unlike ``GroupLGBMModel``'s optional ``AsymmetricSquaredObjective`` (a
    custom Python callable, which needed to be a *module-level* class rather
    than a closure to survive ``joblib.dump``/``load``), this model only ever
    uses LightGBM's **built-in** string objective ``"quantile"`` -- no custom
    Python callable is involved anywhere in ``fit``/``predict``, so none of
    that closure-pickling constraint applies here. The whole
    ``GroupLGBMQuantileModel`` instance (a plain dict of 9 fitted
    ``LGBMRegressor``s plus a few scalars) pickles via a single
    ``joblib.dump`` exactly like ``GroupLGBMModel``/``GroupXGBModel`` do --
    see ``tests/test_lgbm_quantile_model.py``'s round-trip test.
    """

    def __init__(self, capacity_kwh: float, **lgbm_params: Any):
        if "monotonic_constraints" in lgbm_params:
            raise ValueError(
                "GroupLGBMQuantileModel must not be given monotonic_constraints -- "
                "LightGBM's quantile objective is incompatible with it (see class docstring)."
            )
        self.capacity_kwh = float(capacity_kwh)
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **lgbm_params}
        self.models_: dict[float, lgb.LGBMRegressor] = {}
        self.best_iteration_: int | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    ) -> "GroupLGBMQuantileModel":
        """Fit all 9 quantile sub-models, each on the same ``sample_weight``.

        If ``eval_set=(X_val, y_val)`` is given, every sub-model trains with
        early stopping against it using LightGBM's built-in ``"quantile"``
        eval metric (which automatically uses that sub-model's own ``alpha``
        -- no cross-quantile mismatch); ``best_iteration_`` is then recorded
        from the ``q=0.5`` sub-model (see class docstring). Otherwise every
        sub-model trains for the full configured ``n_estimators`` and
        ``best_iteration_`` is left ``None``.

        A fresh ``models_`` dict (and fresh ``LGBMRegressor`` per quantile) is
        built on every call, so repeated ``.fit()`` calls on the same wrapper
        instance never warm-start from a previous fit's trees.
        """
        y_arr = np.asarray(y, dtype=float)
        sample_weight = np.where(
            y_arr >= self.capacity_kwh * VALID_HOUR_UTILIZATION,
            1.0,
            LOW_UTILIZATION_SAMPLE_WEIGHT,
        )

        self.models_ = {}
        for q in QUANTILES:
            params = dict(self.params)
            params["objective"] = "quantile"
            params["alpha"] = q
            model = lgb.LGBMRegressor(**params)

            fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
            if eval_set is not None:
                fit_kwargs["eval_set"] = [eval_set]
                fit_kwargs["eval_metric"] = "quantile"
                fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

            model.fit(X, y, **fit_kwargs)
            self.models_[q] = model

        median_model = self.models_[MEDIAN_QUANTILE]
        self.best_iteration_ = getattr(median_model, "best_iteration_", None)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Raw (not yet monotonic-enforced) 9-quantile predictions, shape
        ``(n_rows, 9)`` in ``QUANTILES`` order. Diagnostic/testing use --
        ``predict`` is the one CV/inference actually calls.
        """
        preds = [self.models_[q].predict(X) for q in QUANTILES]
        return np.stack(preds, axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """9-quantile predict -> monotonic enforcement -> EU-optimal decision
        point -> capacity-safety clip.

        Signature matches ``src/training/tune_common.py``'s
        ``_GroupModelProtocol`` (``fit``/``predict``/``best_iteration_`` only)
        exactly, so ``oof_predict_generic``/``tune_group_generic`` and
        ``src/ensembling/blend_search.py`` need zero changes to use this model
        class -- the entire decision-theoretic post-processing pipeline is an
        implementation detail fully contained inside this one method.
        """
        raw = self.predict_quantiles(X)
        sorted_q = enforce_monotonic_quantiles(raw)
        decision = decision_optimal_point_prediction(sorted_q, QUANTILES, self.capacity_kwh)
        return np.clip(decision, 0.0, self.capacity_kwh * 1.01)
