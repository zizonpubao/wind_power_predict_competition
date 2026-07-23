"""Decision-theoretic post-processing for 9-quantile GBM predictions.

This turns a LightGBM quantile-regression model's 9 independent per-quantile
predictions into a single point prediction, chosen not by a plain statistic
(mean/median) but by maximizing an *expected utility* (EU) that mirrors the
official competition score directly: ``0.5 * (1-NMAE) + 0.5 * FICR`` (see
``src/evaluation/metrics.py`` / CLAUDE.md section 5).

Why EU instead of the simpler "reward = price * generation * validity" sum
used by the reference v14 hand-off code (``gbm_model.py`` at
``C:\\Users\\heelo\\Desktop\\files\\gbm_model.py``): that version only scores
FICR-shaped reward and ignores the NMAE half of the official score, and its
"reward" isn't actually an *expected utility* over a distribution -- it just
sums a reward-per-quantile-sample without weighting by how much probability
mass that quantile represents. The functions below instead (1) build an
explicit discretized probability mass function (pmf) over generation values
from the 9 quantiles, and (2) choose the decision point that maximizes a
proper expectation of a utility blending both official metrics, which is a
more direct proxy for what actually gets scored.

Pipeline (all pure functions, no I/O, no model objects -- easiest layer of
this project to unit test in isolation):

    q_preds (n_rows, 9), possibly quantile-crossed
        -> enforce_monotonic_quantiles           (sort each row's 9 values)
        -> quantiles_to_pmf                       (9 quantiles -> 101-point pmf)
        -> expected_utility_grid_search            (pmf -> per-row optimal decision)

``decision_optimal_point_prediction`` chains all three for convenience.

IMPORTANT approximation / known limitation (FICR term)
-------------------------------------------------------
The official FICR is a *global* ratio: ``sum(rate_h * actual_h) / sum(4 *
actual_h)`` computed over an entire fold/dataset's eligible hours -- it does
not decompose losslessly into a sum of independent per-row terms, because the
denominator couples every hour together. ``expected_utility_grid_search``'s
``ficr_term`` uses each row's *own* pmf-implied expected actual generation
(``E[actual]``) as a **local, row-level stand-in** for that global
denominator (i.e. "if every hour looked statistically like this one, what
would my share of the theoretical-max settlement be"). This is a genuine
approximation, not the literal official formula, and is most defensible when
generation levels don't vary wildly hour-to-hour relative to the pmf's own
spread; it cannot exactly reproduce the true dataset-level FICR value. Treat
the resulting decision as "utility-optimized under a locally-linearized proxy
for FICR", not as literally maximizing the real official FICR metric.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    FICR_TIER1_NMAE_THRESHOLD,
    FICR_TIER1_RATE,
    FICR_TIER2_NMAE_THRESHOLD,
    FICR_TIER2_RATE,
    FICR_TIER3_RATE,
)

QUANTILES: list[float] = [0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95]

# Numerical-safety epsilons -- never let a degenerate (zero-width / zero-mass)
# input produce a NaN or a divide-by-zero.
_WIDTH_EPS = 1e-9
_DENOM_EPS = 1e-9


def enforce_monotonic_quantiles(q_preds: np.ndarray) -> np.ndarray:
    """Force each row's quantile predictions to be non-decreasing.

    LightGBM's ``objective="quantile"`` trains 9 *independent* models (one per
    ``alpha``), so nothing structurally prevents "quantile crossing" -- e.g.
    the alpha=0.35 model predicting a larger value than the alpha=0.5 model
    for the same row. ``np.maximum.accumulate`` along the quantile axis is the
    standard cheap fix: it leaves already-sorted rows untouched and, for
    crossed rows, clamps each quantile up to at least the previous (lower)
    quantile's value -- the result is always within the original row's
    ``[min, max]`` range.

    Parameters
    ----------
    q_preds : array of shape ``(n_rows, n_quantiles)``.

    Returns
    -------
    Array of the same shape, non-decreasing along axis 1.
    """
    q_preds = np.asarray(q_preds, dtype=float)
    if q_preds.ndim != 2:
        raise ValueError(f"q_preds must be 2D (n_rows, n_quantiles), got shape {q_preds.shape}")
    return np.maximum.accumulate(q_preds, axis=1)


def quantiles_to_pmf(
    q_values: np.ndarray,
    quantile_levels: list[float] | np.ndarray = QUANTILES,
    capacity_kwh: float = 1.0,
    n_grid: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-row quantile predictions into a discretized pmf over
    ``[0, capacity_kwh]``.

    Construction: for each row, treat ``(0, 0.0)`` and ``(capacity_kwh, 1.0)``
    as two extra anchor points bracketing the row's (already
    monotonic-enforced) quantile predictions, giving ``n_quantiles + 2``
    ``(value, cumulative_probability)`` points that define a piecewise-linear
    CDF. That CDF is evaluated at ``n_grid`` equally spaced points spanning
    ``[0, capacity_kwh]`` (1% steps at the default ``n_grid=101``), and the pmf
    is the discrete first difference of that CDF.

    Row-values are clipped into ``[0, capacity_kwh]`` before building the
    anchors -- a raw quantile regressor has no notion of the installed-
    capacity bound, so an out-of-range quantile prediction is folded onto the
    nearest valid anchor rather than left to produce a decreasing/invalid CDF
    segment.

    Vectorization note: ``capacity_kwh`` is a single scalar shared by every
    row of one KPX group's model (not per-row), so the output ``grid`` is
    identical for every row -- only the pmf differs row to row. The CDF is
    built by looping over the (fixed, small: ``n_quantiles + 1 == 10``)
    piecewise-linear *segments* between anchors, broadcasting each segment's
    linear-interpolation formula across all rows and all grid points at once
    -- no per-row Python loop.

    Parameters
    ----------
    q_values : array of shape ``(n_rows, n_quantiles)``, ideally already
        passed through ``enforce_monotonic_quantiles``.
    quantile_levels : the probability level of each column of ``q_values``
        (defaults to the module's ``QUANTILES``).
    capacity_kwh : the group's 1-hour installed-capacity bound.
    n_grid : number of equally spaced grid points over ``[0, capacity_kwh]``.

    Returns
    -------
    ``(grid, pmf)``: ``grid`` has shape ``(n_grid,)`` (shared across rows),
    ``pmf`` has shape ``(n_rows, n_grid)`` and each row sums to 1.
    """
    q_values = np.asarray(q_values, dtype=float)
    if q_values.ndim != 2:
        raise ValueError(f"q_values must be 2D (n_rows, n_quantiles), got shape {q_values.shape}")
    quantile_levels = np.asarray(quantile_levels, dtype=float)
    n_rows, n_q = q_values.shape
    if len(quantile_levels) != n_q:
        raise ValueError(
            f"quantile_levels has {len(quantile_levels)} entries but q_values has {n_q} columns"
        )
    if capacity_kwh <= 0:
        raise ValueError(f"capacity_kwh must be positive, got {capacity_kwh}")

    q_clipped = np.clip(q_values, 0.0, capacity_kwh)

    # Anchors: x has shape (n_rows, n_q + 2), y has shape (n_q + 2,) (shared
    # across rows -- the cumulative-probability *levels* don't depend on the
    # row, only the *values* at which they occur do).
    x = np.concatenate(
        [np.zeros((n_rows, 1)), q_clipped, np.full((n_rows, 1), float(capacity_kwh))], axis=1
    )
    y = np.concatenate([[0.0], quantile_levels, [1.0]])

    grid = np.linspace(0.0, float(capacity_kwh), n_grid)  # (n_grid,), identical for every row

    cdf = np.zeros((n_rows, n_grid))
    n_segments = x.shape[1] - 1
    for k in range(n_segments):
        x_lo = x[:, k]  # (n_rows,)
        x_hi = x[:, k + 1]
        y_lo, y_hi = y[k], y[k + 1]
        width = x_hi - x_lo

        # Degenerate (zero-width) segments happen whenever two neighboring
        # anchors coincide (e.g. several quantile predictions tied, or a
        # quantile prediction sitting exactly on the 0/capacity anchor) --
        # they represent a probability *jump* at that single value rather
        # than a ramp, so assign frac=1 (-> cdf = y_hi, the *largest*
        # cumulative value reached at that point) instead of dividing by
        # zero.
        width_safe = np.where(width > _WIDTH_EPS, width, 1.0)  # (n_rows,), avoid 0-division
        frac = (grid[None, :] - x_lo[:, None]) / width_safe[:, None]  # (n_rows, n_grid)
        frac = np.where((width > _WIDTH_EPS)[:, None], frac, 1.0)
        frac = np.clip(frac, 0.0, 1.0)
        candidate = y_lo + frac * (y_hi - y_lo)

        # Small epsilon on the membership test guards against float
        # boundary misses at segment edges; ascending segment order means a
        # later (higher-y) segment's write wins at any point shared by two
        # adjacent segments, matching a right-continuous CDF convention.
        eps = max(capacity_kwh, 1.0) * 1e-9
        mask = (grid[None, :] >= x_lo[:, None] - eps) & (grid[None, :] <= x_hi[:, None] + eps)
        cdf = np.where(mask, candidate, cdf)

    # Numerical safety net: clip to [0, 1] and force monotonic non-decreasing
    # (should already hold given non-decreasing y/x, but float rounding in
    # the segment loop above could in principle produce a tiny local dip).
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf, axis=1)

    pmf = np.diff(cdf, axis=1, prepend=0.0)
    pmf = np.clip(pmf, 0.0, None)  # guard tiny negative float noise

    row_sums = pmf.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums > _DENOM_EPS, row_sums, 1.0)
    pmf = pmf / row_sums_safe

    return grid, pmf


def expected_utility_grid_search(
    grid: np.ndarray,
    pmf: np.ndarray,
    capacity_kwh: float,
    w_nmae: float = 0.5,
    w_ficr: float = 0.5,
) -> np.ndarray:
    """Pick, per row, the grid point maximizing an expected-utility blend of
    the 1-NMAE and FICR official metrics against that row's pmf.

    Builds one shared ``(n_grid, n_grid)`` reward matrix (indexed
    ``[decision_j, outcome_k]``) covering every candidate decision value
    against every possible outcome value on ``grid`` -- this construction
    happens exactly once (it only depends on ``grid``/``capacity_kwh``, not on
    any row's pmf), then every row's expected utility is computed via a single
    matrix multiply against its own pmf.

        nmae_term[j, k]  = 1 - |grid[j] - grid[k]| / capacity_kwh
        nmae_h           = |grid[j] - grid[k]| / capacity_kwh
        rate[j, k]       = FICR_TIER1_RATE if nmae_h <= FICR_TIER1_NMAE_THRESHOLD
                           FICR_TIER2_RATE if nmae_h <= FICR_TIER2_NMAE_THRESHOLD
                           FICR_TIER3_RATE otherwise
        ficr_term[j, k]  = rate[j, k] * grid[k] / (FICR_TIER1_RATE * E[actual])

    where ``E[actual]`` is *this row's* pmf-implied expected outcome value
    (``sum_k pmf[k] * grid[k]``) -- see the module docstring's "IMPORTANT
    approximation" note: this is a per-row local stand-in for the official
    FICR's dataset-level denominator, not the literal formula.

        EU[j] = w_nmae * sum_k(pmf[k] * nmae_term[j, k])
              + w_ficr * sum_k(pmf[k] * ficr_term[j, k])

    The returned decision for each row is ``grid[argmax_j EU[j]]``.

    Parameters
    ----------
    grid : shape ``(n_grid,)``, as returned by ``quantiles_to_pmf``.
    pmf : shape ``(n_rows, n_grid)``, as returned by ``quantiles_to_pmf``.
    capacity_kwh : the group's 1-hour installed-capacity bound (must match
        what ``grid``/``pmf`` were built against).
    w_nmae, w_ficr : utility blend weights (default 0.5/0.5, matching the
        official ``score = 0.5*(1-NMAE) + 0.5*FICR``).

    Returns
    -------
    Array of shape ``(n_rows,)``: the EU-maximizing decision value per row.
    """
    grid = np.asarray(grid, dtype=float)
    pmf = np.asarray(pmf, dtype=float)
    if pmf.ndim != 2 or pmf.shape[1] != len(grid):
        raise ValueError(f"pmf must have shape (n_rows, {len(grid)}), got {pmf.shape}")
    if capacity_kwh <= 0:
        raise ValueError(f"capacity_kwh must be positive, got {capacity_kwh}")

    abs_err = np.abs(grid[:, None] - grid[None, :]) / capacity_kwh  # (n_grid_j, n_grid_k)
    nmae_term = 1.0 - abs_err

    rate = np.where(
        abs_err <= FICR_TIER1_NMAE_THRESHOLD,
        FICR_TIER1_RATE,
        np.where(abs_err <= FICR_TIER2_NMAE_THRESHOLD, FICR_TIER2_RATE, FICR_TIER3_RATE),
    )
    ficr_numer = rate * grid[None, :]  # (n_grid_j, n_grid_k), broadcasts grid[k] over rows k

    expected_actual = pmf @ grid  # (n_rows,) -- this row's E[actual] under its own pmf
    denom = np.maximum(FICR_TIER1_RATE * expected_actual, _DENOM_EPS)

    eu_nmae = pmf @ nmae_term.T  # (n_rows, n_grid_j)
    eu_ficr_numer = pmf @ ficr_numer.T  # (n_rows, n_grid_j)
    eu_ficr = eu_ficr_numer / denom[:, None]

    eu = w_nmae * eu_nmae + w_ficr * eu_ficr
    best_j = np.argmax(eu, axis=1)
    return grid[best_j]


def decision_optimal_point_prediction(
    q_values: np.ndarray,
    quantile_levels: list[float] | np.ndarray = QUANTILES,
    capacity_kwh: float = 1.0,
    w_nmae: float = 0.5,
    w_ficr: float = 0.5,
    n_grid: int = 101,
) -> np.ndarray:
    """End-to-end convenience wrapper: raw (possibly quantile-crossed) 9-
    quantile predictions -> monotonic enforcement -> pmf -> EU-optimal point
    decision.

    See ``enforce_monotonic_quantiles``, ``quantiles_to_pmf``, and
    ``expected_utility_grid_search`` for the individual steps this composes.
    """
    q_sorted = enforce_monotonic_quantiles(np.asarray(q_values, dtype=float))
    grid, pmf = quantiles_to_pmf(q_sorted, quantile_levels, capacity_kwh, n_grid=n_grid)
    return expected_utility_grid_search(grid, pmf, capacity_kwh, w_nmae=w_nmae, w_ficr=w_ficr)
