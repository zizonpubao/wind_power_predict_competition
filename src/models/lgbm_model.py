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

Asymmetric objective (``asymmetry_alpha``)
-------------------------------------------
``reports/eda/ficr_gap_diagnosis.md`` (priority recommendation #2) found the
tuned baseline systematically *under*-predicts 62-66% of eligible hours, with
a consistently negative mean signed error, worst at high-generation hours --
a classic gradient-boosting regression-to-the-mean effect that a symmetric
squared-error loss has no way to counteract. ``GroupLGBMModel`` optionally
accepts ``asymmetry_alpha`` (a float in ``(0, 1)``, excluded means "use
LightGBM's plain built-in regression objective"): when set, it swaps in
``AsymmetricSquaredObjective(asymmetry_alpha)`` as the training objective,
which penalizes under-prediction residuals ``alpha`` times as hard and
over-prediction residuals ``(1-alpha)`` times as hard (see that class's
docstring for the exact grad/hess). ``alpha=0.5`` is mathematically identical
to plain squared error (grad=r, hess=1), so the parameter continuously
degrades to the symmetric baseline rather than being an on/off switch --
useful both for the unit tests below and for Optuna's search space in
``src/training/tune_hyperparams.py``.

Early stopping still uses a *standard* ``eval_metric`` (``"l1"``, unchanged)
rather than a custom eval metric matching the asymmetric objective. LightGBM
supports a custom ``eval_metric`` callable alongside a custom ``fobj``, but a
standard L1/L2 stopping criterion is deliberately kept here: it answers "has
the model stopped improving on held-out data at all", independent of which
training loss produced the boosting rounds, and avoids coupling early
stopping's patience to whatever alpha happens to be under search this trial
(a leaderboard-relevant eval metric like ``competition_score`` itself is not
usable here either -- it needs the full multi-group DataFrame shape and
capacity-based eligibility filter, not per-row grad/hess-style callbacks).
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


def asymmetric_squared_error_grad_hess(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Pure gradient/Hessian function for the asymmetric squared-error loss.

    Residual convention: ``r = pred - actual``.
      - ``r < 0`` (under-prediction): ``loss = alpha * r**2``,
        ``grad = 2*alpha*r``, ``hess = 2*alpha``.
      - ``r >= 0`` (over-prediction): ``loss = (1-alpha) * r**2``,
        ``grad = 2*(1-alpha)*r``, ``hess = 2*(1-alpha)``.

    ``alpha > 0.5`` penalizes under-prediction more heavily than
    over-prediction, pushing the fitted model's predictions up -- the
    opposite direction of the systematic under-prediction bias documented in
    ``reports/eda/ficr_gap_diagnosis.md``. ``alpha == 0.5`` reduces exactly to
    plain squared error (``grad = r``, ``hess = 1``), matching LightGBM's
    built-in ``"regression"`` (L2) objective.

    Kept as a standalone, side-effect-free function (rather than only living
    inside ``AsymmetricSquaredObjective.__call__``) so it's directly unit
    testable without constructing a LightGBM model.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    r = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    coef = np.where(r < 0.0, alpha, 1.0 - alpha)
    grad = 2.0 * coef * r
    hess = 2.0 * coef
    return grad, hess


class AsymmetricSquaredObjective:
    """Picklable callable wrapping ``asymmetric_squared_error_grad_hess`` at a
    fixed ``alpha``, for use as LightGBM's sklearn-API ``objective=`` callable.

    LightGBM's sklearn wrapper calls a custom ``objective`` callable as
    ``objective(y_true, y_pred) -> (grad, hess)`` (see
    ``lightgbm.sklearn._ObjectiveFunctionWrapper``'s docstring) -- this class
    matches that exact signature via ``__call__``.

    Implemented as a module-level class (not a closure/nested function)
    specifically so instances survive ``joblib.dump``/``joblib.load`` of the
    fitted ``LGBMRegressor``: plain closures over local variables cannot be
    pickled (``pickle.PicklingError: ... not found as __main__.<closure>``),
    but a class with a simple ``float`` attribute pickles fine since the
    class itself is importable by qualified name.
    """

    def __init__(self, alpha: float):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return asymmetric_squared_error_grad_hess(y_true, y_pred, self.alpha)

    def __repr__(self) -> str:
        return f"AsymmetricSquaredObjective(alpha={self.alpha})"


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
    asymmetry_alpha: if ``None`` (default), trains with LightGBM's plain
        built-in ``"regression"`` (L2) objective, unchanged from before this
        parameter existed. If a float in ``(0, 1)``, trains with
        ``AsymmetricSquaredObjective(asymmetry_alpha)`` instead (see module
        docstring / that class's docstring) -- ``alpha > 0.5`` penalizes
        under-prediction more than over-prediction.
    **lgbm_params: overrides merged on top of ``DEFAULT_PARAMS`` and passed
        straight through to ``lightgbm.LGBMRegressor``.
    """

    def __init__(self, capacity_kwh: float, asymmetry_alpha: float | None = None, **lgbm_params: Any):
        self.capacity_kwh = float(capacity_kwh)
        self.asymmetry_alpha = float(asymmetry_alpha) if asymmetry_alpha is not None else None
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **lgbm_params}
        if self.asymmetry_alpha is not None:
            self.params["objective"] = AsymmetricSquaredObjective(self.asymmetry_alpha)
            # LightGBM's default feature_pre_filter=True pre-filters features
            # it deems unsplittable (given min_data_in_leaf) once at Dataset
            # construction time; combined with a *custom* fobj objective
            # (any custom objective, not specific to this asymmetric one --
            # verified by reproducing the same crash at alpha=0.5, which is
            # mathematically identical to the built-in objective), this can
            # hit a real LightGBM engine bug on small folds/aggressive
            # min_child_samples ("LightGBMError: Check failed:
            # train_data->num_features() > 0" inside feature_histogram.hpp)
            # that the built-in objective path does not trigger. Disabling
            # feature_pre_filter (only when a custom objective is in play,
            # and only if the caller hasn't already set it explicitly) is
            # the documented LightGBM workaround and costs a little dataset-
            # construction speed, not correctness -- each CV fold/Optuna
            # trial builds its own fresh Dataset here anyway, so there is no
            # cross-fold pre-filter caching being given up.
            self.params.setdefault("feature_pre_filter", False)
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
