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

from src.evaluation.metrics import (
    DEFAULT_MIN_UTILIZATION,
    FICR_TIER1_NMAE_THRESHOLD,
    FICR_TIER1_RATE,
    FICR_TIER2_NMAE_THRESHOLD,
    FICR_TIER2_RATE,
    FICR_TIER3_RATE,
)


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


# ---------------------------------------------------------------------------
# FICR step-surrogate objective (idea A3)
# ---------------------------------------------------------------------------
# Directly optimize the money-shaped FICR metric at *training* time rather than
# only post-hoc (as ``src/features/decision_optimize.py`` does). The official
# per-hour settlement rate is a step function of the error rate
# ``e = |pred - actual| / capacity`` (see ``src/evaluation/metrics.py``):
#   e <= 0.06 -> 4 won/kWh ;  0.06 < e <= 0.08 -> 3 ;  e > 0.08 -> 0.
# Maximizing settlement == keeping ``e`` from crossing the 6%/8% cliffs. We
# make that objective differentiable with a **two-sigmoid smooth approximation**
# of the rate function and turn it into a boosting loss, reusing the confirmed
# tier boundaries/rates as named constants (never re-hardcoding the numbers).

# Numerical-safety floor so a zero-weight row never produces exactly-zero hess.
_HESS_EPS = 1e-12


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-safe logistic sigmoid (``steepness * (e - t)`` can be large)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def ficr_smooth_rate(
    e: np.ndarray,
    steepness: float,
    t1: float = FICR_TIER1_NMAE_THRESHOLD,
    t2: float = FICR_TIER2_NMAE_THRESHOLD,
    r1: float = FICR_TIER1_RATE,
    r2: float = FICR_TIER2_RATE,
    r3: float = FICR_TIER3_RATE,
) -> np.ndarray:
    """Differentiable approximation of the FICR per-hour settlement *rate*.

    Two logistic sigmoids stacked so the rate steps down ``r1 -> r2`` at the
    6% cliff (``t1``) and ``r2 -> r3`` at the 8% cliff (``t2``)::

        smooth_rate(e) = r1 - (r1-r2)*sigma(k(e-t1)) - (r2-r3)*sigma(k(e-t2))

    ``e -> 0``: ``~r1`` (top rate); between the cliffs: ``~r2``; ``e`` large:
    ``~r3``. Larger ``steepness`` (``k``) -> sharper (closer to the true step,
    but flatter gradients away from the cliffs).
    """
    e = np.asarray(e, dtype=np.float64)
    a = r1 - r2
    b = r2 - r3
    return r1 - a * _stable_sigmoid(steepness * (e - t1)) - b * _stable_sigmoid(steepness * (e - t2))


def _cliff_pull_and_curv(
    e: np.ndarray, steepness: float, t1: float, t2: float, a: float, b: float
) -> tuple[np.ndarray, np.ndarray]:
    """The (k-free) sigmoid-derivative "pull" shape and its derivative.

    ``pull(e) = a*sigma'(k(e-t1)) + b*sigma'(k(e-t2))`` is proportional to
    ``d(cliff_term)/de`` (where ``cliff_term = r1 - smooth_rate``), a bump that
    peaks *at* each cliff and is ~0 elsewhere -- this is the gradient signal
    that discourages predictions whose error rate is near a cliff. Using
    ``pull`` (dropping the explicit ``k`` factor of the exact derivative) keeps
    the cliff gradient O(1), so ``w_cliff`` stays a sanely-scaled hyperparameter
    comparable to ``w_l2``; the dropped constant is simply absorbed into
    ``w_cliff`` (a monotone reparametrization, same optimum shape).

    ``pull_prime = d(pull)/de`` (using ``sigma''(z)=sigma'(z)(1-2 sigma(z))``) is
    the exact curvature of that designed cliff surrogate; the caller takes its
    absolute value as a positive (Gauss-Newton-style) Hessian proxy, since the
    true curvature flips sign across each bump and LightGBM needs hess >= 0.
    """
    z1 = steepness * (e - t1)
    z2 = steepness * (e - t2)
    s1 = _stable_sigmoid(z1)
    s2 = _stable_sigmoid(z2)
    ds1 = s1 * (1.0 - s1)  # sigma'
    ds2 = s2 * (1.0 - s2)
    pull = a * ds1 + b * ds2
    pull_prime = steepness * (a * ds1 * (1.0 - 2.0 * s1) + b * ds2 * (1.0 - 2.0 * s2))
    return pull, pull_prime


def huber_grad_hess_eps(
    eps: np.ndarray, delta: float, tail_hess: float
) -> tuple[np.ndarray, np.ndarray]:
    """Huber gradient/Hessian in *error-rate* space ``eps = (pred-actual)/cap``.

    ``|eps| <= delta``: quadratic (``grad = eps``, ``hess = 1``); beyond:
    linear (``grad = delta*sign(eps)``, ``hess = tail_hess``). ``tail_hess`` is
    a positive constant (default 1.0, matching the in-band curvature -- i.e. a
    constant-Hessian Huber, exactly how LightGBM's own built-in huber
    approximates it) rather than the textbook 0: it keeps the combined
    objective's Hessian strictly positive everywhere (LightGBM stability) and
    makes the ``w_cliff=0`` case reduce to exactly this (weighted) Huber.

    NOTE on ``delta``: this is in **error-rate** units (fraction of capacity),
    so ``delta`` must be wide enough to cover the residuals seen during
    boosting -- crucially at cold-start, where a custom-objective LightGBM
    initializes every raw prediction to 0 and so every eligible row's residual
    is its full generation (``|eps|`` up to ~1.0). A too-small ``delta`` (e.g.
    0.1) puts every row in the flat linear tail, where ``grad/hess`` is an
    identical constant for all rows; no split then has positive gain and
    LightGBM emits a degenerate zero tree (the model collapses to constant 0).
    The default ``delta=1.0`` keeps the whole plausible residual range in the
    quadratic (residual-proportional) region -- so ``w_cliff=0`` behaves like a
    well-scaled squared error -- while still capping the gradient of physically
    impossible misses (``|pred-actual| > capacity``, e.g. sentinel-like blowups)
    at the linear tail.
    """
    eps = np.asarray(eps, dtype=np.float64)
    a = np.abs(eps)
    quad = a <= delta
    grad = np.where(quad, eps, delta * np.sign(eps))
    hess = np.where(quad, 1.0, tail_hess)
    return grad, hess


def ficr_surrogate_grad_hess(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity_kwh: float,
    w_cliff: float,
    w_l2: float,
    huber_delta: float,
    steepness: float,
    min_utilization: float,
    ineligible_weight: float,
    actual_weight_floor: float,
    huber_tail_hess: float,
    t1: float = FICR_TIER1_NMAE_THRESHOLD,
    t2: float = FICR_TIER2_NMAE_THRESHOLD,
    r1: float = FICR_TIER1_RATE,
    r2: float = FICR_TIER2_RATE,
    r3: float = FICR_TIER3_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient/Hessian (w.r.t. ``pred``, in kWh) of the FICR step-surrogate loss.

    Per row, in error-rate space ``eps = (pred - actual)/capacity`` (signed),
    ``e = |eps|``, the loss combines two terms:

      * **cliff term** ``w_cliff * cliff_term(e)`` where
        ``cliff_term = r1 - smooth_rate(e)`` rises from ~0 to ~(r1-r3) as ``e``
        crosses the 6%/8% cliffs -- its gradient (``w_cliff * pull(e) *
        sign(eps)``) always points toward ``actual`` but is only *strong* near a
        cliff, and vanishes both at ``e=0`` and far past 8% (once a prediction
        is grossly wrong the step function is already flat, so the loss can no
        longer rescue it -- an honest ceiling of this whole idea).
      * **Huber term** ``w_l2 * huber(eps, delta)`` -- a small always-on pull
        toward ``actual`` that keeps the objective well-behaved inside the <=6%
        band (where the cliff gradient is ~0) and caps the influence of gross
        misses (linear tail) so they don't dominate the FICR shaping.

    Row weighting mirrors what actually gets scored: eligibility
    (``actual >= min_utilization*capacity`` -> weight 1, else
    ``ineligible_weight``) times ``actual/capacity`` (floored at
    ``actual_weight_floor``), since settlement is proportional to generation so
    high-output hours matter more.

    Chain rule + fixed rescale: ``d/dpred = d/deps * (1/cap)`` and
    ``d2/dpred2 = d2/deps2 * (1/cap^2)``, but the whole loss is then multiplied
    by the positive constant ``cap**2`` (which cannot change the argmin) so the
    grad/hess land in the same numeric scale LightGBM's built-in / the
    asymmetric objective produce -- i.e. the Huber term reduces to a plain
    **kWh-space** Huber (``grad = w_l2*(pred-actual)``, ``hess = w_l2``). Without
    this rescale the raw ``1/cap**2`` Hessian is ~1e-9, far below LightGBM's
    ``min_sum_hessian_in_leaf`` (default 1e-3), so the tree never splits and the
    model collapses to a constant-0 predictor (verified empirically). Net
    effect per row: ``grad = weight * cap * grad_eps``, ``hess = weight *
    hess_eps``. Hessian is kept strictly positive (``|pull_prime|`` proxy for
    the non-convex cliff curvature + the always-positive Huber tail).
    """
    if capacity_kwh <= 0:
        raise ValueError(f"capacity_kwh must be positive, got {capacity_kwh}")
    if w_cliff < 0:
        raise ValueError(f"w_cliff must be >= 0, got {w_cliff}")
    if w_l2 <= 0:
        raise ValueError(f"w_l2 must be > 0 (needed for a strictly positive Hessian), got {w_l2}")
    if steepness <= 0:
        raise ValueError(f"steepness must be > 0, got {steepness}")

    actual = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    cap = float(capacity_kwh)

    eps = (pred - actual) / cap
    e = np.abs(eps)
    a_coef = r1 - r2
    b_coef = r2 - r3

    pull, pull_prime = _cliff_pull_and_curv(e, steepness, t1, t2, a_coef, b_coef)
    grad_eps_cliff = w_cliff * pull * np.sign(eps)
    hess_eps_cliff = w_cliff * np.abs(pull_prime)

    huber_grad, huber_hess = huber_grad_hess_eps(eps, huber_delta, huber_tail_hess)
    grad_eps = grad_eps_cliff + w_l2 * huber_grad
    hess_eps = hess_eps_cliff + w_l2 * huber_hess

    eligible = actual >= (min_utilization * cap)
    base_w = np.where(eligible, 1.0, ineligible_weight)
    actual_w = np.clip(actual / cap, actual_weight_floor, None)
    weight = base_w * actual_w

    # Chain rule (1/cap, 1/cap^2) then a fixed *cap^2 rescale of the whole
    # loss -> grad picks up one net factor of cap, hess picks up none. See the
    # docstring for why (keeps hess O(w_l2), above min_sum_hessian_in_leaf).
    grad = weight * cap * grad_eps
    hess = weight * hess_eps
    hess = np.maximum(hess, _HESS_EPS)
    return grad, hess


class FICRSurrogateObjective:
    """Picklable LightGBM ``objective=`` callable for the FICR step surrogate.

    Module-level (not a closure) for the same reason as
    ``AsymmetricSquaredObjective``: instances must survive ``joblib.dump`` of a
    fitted ``LGBMRegressor``. ``capacity_kwh`` (the group's 1-hour installed
    capacity in kWh) is required -- the error rate ``e = |pred-actual|/capacity``
    is what the FICR cliffs are defined on. All other knobs default to
    sensible starting values; ``w_cliff``/``w_l2`` are the two the training
    harness sweeps (cliff-shaping strength vs. always-on Huber pull).
    """

    def __init__(
        self,
        capacity_kwh: float,
        w_cliff: float = 1.0,
        w_l2: float = 1.0,
        huber_delta: float = 1.0,
        steepness: float = 400.0,
        min_utilization: float = DEFAULT_MIN_UTILIZATION,
        ineligible_weight: float = 0.05,
        actual_weight_floor: float = 0.05,
        huber_tail_hess: float = 1.0,
    ):
        if capacity_kwh <= 0:
            raise ValueError(f"capacity_kwh must be positive, got {capacity_kwh}")
        if w_cliff < 0:
            raise ValueError(f"w_cliff must be >= 0, got {w_cliff}")
        if w_l2 <= 0:
            raise ValueError(f"w_l2 must be > 0, got {w_l2}")
        if steepness <= 0:
            raise ValueError(f"steepness must be > 0, got {steepness}")
        self.capacity_kwh = float(capacity_kwh)
        self.w_cliff = float(w_cliff)
        self.w_l2 = float(w_l2)
        self.huber_delta = float(huber_delta)
        self.steepness = float(steepness)
        self.min_utilization = float(min_utilization)
        self.ineligible_weight = float(ineligible_weight)
        self.actual_weight_floor = float(actual_weight_floor)
        self.huber_tail_hess = float(huber_tail_hess)

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return ficr_surrogate_grad_hess(
            y_true,
            y_pred,
            capacity_kwh=self.capacity_kwh,
            w_cliff=self.w_cliff,
            w_l2=self.w_l2,
            huber_delta=self.huber_delta,
            steepness=self.steepness,
            min_utilization=self.min_utilization,
            ineligible_weight=self.ineligible_weight,
            actual_weight_floor=self.actual_weight_floor,
            huber_tail_hess=self.huber_tail_hess,
        )

    def __repr__(self) -> str:
        return (
            f"FICRSurrogateObjective(capacity_kwh={self.capacity_kwh}, "
            f"w_cliff={self.w_cliff}, w_l2={self.w_l2}, huber_delta={self.huber_delta}, "
            f"steepness={self.steepness})"
        )


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
    ficr_surrogate: if not ``None``, a dict of ``FICRSurrogateObjective``
        keyword args (e.g. ``{"w_cliff": 1.0, "w_l2": 1.0}``) -- trains with
        that FICR step-surrogate objective, which directly optimizes the
        money-shaped FICR metric at training time (idea A3, see
        ``FICRSurrogateObjective``). ``capacity_kwh`` is injected
        automatically. Mutually exclusive with ``asymmetry_alpha``.
    **lgbm_params: overrides merged on top of ``DEFAULT_PARAMS`` and passed
        straight through to ``lightgbm.LGBMRegressor``.
    """

    def __init__(
        self,
        capacity_kwh: float,
        asymmetry_alpha: float | None = None,
        ficr_surrogate: dict[str, Any] | None = None,
        **lgbm_params: Any,
    ):
        if asymmetry_alpha is not None and ficr_surrogate is not None:
            raise ValueError("asymmetry_alpha and ficr_surrogate are mutually exclusive; set at most one.")
        self.capacity_kwh = float(capacity_kwh)
        self.asymmetry_alpha = float(asymmetry_alpha) if asymmetry_alpha is not None else None
        self.ficr_surrogate = dict(ficr_surrogate) if ficr_surrogate is not None else None
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **lgbm_params}
        if self.ficr_surrogate is not None:
            self.params["objective"] = FICRSurrogateObjective(
                capacity_kwh=self.capacity_kwh, **self.ficr_surrogate
            )
            # Same custom-objective / feature_pre_filter workaround as the
            # asymmetric path below (any custom fobj can hit the LightGBM
            # feature_pre_filter crash on small folds).
            self.params.setdefault("feature_pre_filter", False)
        elif self.asymmetry_alpha is not None:
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
