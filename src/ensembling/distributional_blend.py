"""Distributional ensemble blending (idea A1): blend several models' *predictive
quantiles* into one common distribution, then apply the decision-theoretic
FICR-aware post-processing (``src.features.decision_optimize``) **once** to that
blended distribution -- instead of linearly blending three point predictions.

Why this exists
---------------
The verified single-biggest lever in this project has been attacking the FICR
step function decision-theoretically: the 9-quantile GBM feeds its distribution
through ``decision_optimal_point_prediction`` (an expected-utility grid search
over the official 1-NMAE/FICR score) rather than returning a plain median. But
the current best submission is a *point* blend
(``src.ensembling.blend_search``): it linearly averages three already-collapsed
point predictions, so the decision optimizer can no longer see any distribution
-- a point has no spread, so the whole FICR-aware machinery cannot be applied to
the ensemble.

A1 fixes that mismatch: represent all three models as quantiles on the shared
``QUANTILES`` grid, blend the *quantiles* (Vincentization: average same-level
quantiles across models), and run the decision optimizer exactly once on the
blended distribution. This extends the one proven lever to the whole ensemble
rather than throwing it away at the blend step.

Pure functions only (no I/O, no model objects) -- the harness
(``src.training.train_distributional_blend``) supplies the per-model quantile
arrays and does all fold/score bookkeeping.
"""
from __future__ import annotations

import numpy as np

from src.features.decision_optimize import (
    QUANTILES,
    decision_optimal_point_prediction,
    enforce_monotonic_quantiles,
)


def _normalize_weights(weights: list[float] | np.ndarray, n: int) -> np.ndarray:
    """Coerce ``weights`` to a length-``n`` non-negative array summing to 1."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights must have length {n} (one per quantile array), got shape {w.shape}")
    if np.any(w < 0):
        raise ValueError(f"weights must be non-negative, got {w.tolist()}")
    total = w.sum()
    if total <= 0:
        raise ValueError(f"weights must sum to a positive number, got {total}")
    return w / total


def blend_quantiles(
    quantile_arrays: list[np.ndarray], weights: list[float] | np.ndarray
) -> np.ndarray:
    """Vincentization: weighted-average same-level quantiles across models.

    Each array in ``quantile_arrays`` is one model's ``(n_rows, n_levels)``
    predictive quantiles on a **shared** level grid (same columns/levels, same
    row alignment). The blend averages column-by-column with ``weights``
    (renormalized to sum to 1), which is the standard "average the quantile
    functions" way to combine distributions (a.k.a. Vincent averaging) -- as
    opposed to averaging densities. The result is then passed through
    ``enforce_monotonic_quantiles`` so the blended quantiles are guaranteed
    non-decreasing per row even if rounding or a crossed input made them dip.

    Parameters
    ----------
    quantile_arrays : list of ``(n_rows, n_levels)`` arrays, all the same shape.
    weights : one weight per array (renormalized to sum to 1; non-negative).

    Returns
    -------
    ``(n_rows, n_levels)`` blended, monotonic-enforced quantiles.
    """
    if not quantile_arrays:
        raise ValueError("quantile_arrays must be non-empty")
    arrays = [np.asarray(a, dtype=float) for a in quantile_arrays]
    shape0 = arrays[0].shape
    for i, a in enumerate(arrays):
        if a.ndim != 2:
            raise ValueError(f"quantile_arrays[{i}] must be 2D (n_rows, n_levels), got shape {a.shape}")
        if a.shape != shape0:
            raise ValueError(
                f"all quantile arrays must share the same shape; quantile_arrays[{i}] is {a.shape} "
                f"but quantile_arrays[0] is {shape0}"
            )
    w = _normalize_weights(weights, len(arrays))
    blended = np.tensordot(w, np.stack(arrays, axis=0), axes=(0, 0))  # (n_rows, n_levels)
    return enforce_monotonic_quantiles(blended)


def distributional_blend_point(
    quantile_arrays: list[np.ndarray],
    weights: list[float] | np.ndarray,
    capacity_kwh: float,
    quantile_levels: list[float] | np.ndarray = QUANTILES,
    w_nmae: float = 0.5,
    w_ficr: float = 0.5,
    n_grid: int = 101,
) -> np.ndarray:
    """The core of A1: blend model quantiles into one distribution, then take a
    single decision-theoretic (EU-optimal) point per row from that blend.

    Blends via ``blend_quantiles`` and feeds the result straight into
    ``src.features.decision_optimize.decision_optimal_point_prediction`` (the
    same FICR/1-NMAE expected-utility grid search the standalone GBM uses),
    applied **once** to the blended distribution -- not per base model. The
    ``w_nmae``/``w_ficr`` decision weights are forwarded unchanged, so the
    B1 ``w_ficr`` sweep is free: fix the blended quantiles and only re-run this
    (cheap, vectorized) decision step at each weight.

    Returns ``(n_rows,)`` point predictions in ``[0, capacity_kwh]`` (the
    decision grid is bounded to capacity, so no extra clip is needed).
    """
    blended = blend_quantiles(quantile_arrays, weights)
    return decision_optimal_point_prediction(
        blended,
        quantile_levels=quantile_levels,
        capacity_kwh=capacity_kwh,
        w_nmae=w_nmae,
        w_ficr=w_ficr,
        n_grid=n_grid,
    )
