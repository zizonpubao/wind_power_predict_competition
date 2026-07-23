"""Unit tests for src/ensembling/distributional_blend.py (idea A1: blend
predictive quantiles, then decision-optimize once).

Structural invariants, not "match whatever the code computes":
  - extreme weights (all mass on one model) reproduce that model's quantiles,
  - the blend is monotonic per row even for crossed inputs,
  - weights are renormalized,
  - single-model distributional_blend_point == decision_optimal_point_prediction
    on that model alone,
  - a degenerate (point-mass) distribution decides to (nearest grid to) its mass.
"""
import numpy as np
import pytest

from src.ensembling.distributional_blend import blend_quantiles, distributional_blend_point
from src.features.decision_optimize import (
    QUANTILES,
    decision_optimal_point_prediction,
)

CAPACITY = 21_600.0
N_LEVELS = len(QUANTILES)


def _rand_monotonic_quantiles(n_rows: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = rng.uniform(0, CAPACITY, size=(n_rows, N_LEVELS))
    return np.sort(base, axis=1)  # monotonic per row


def test_blend_extreme_weight_reproduces_single_model():
    a = _rand_monotonic_quantiles(10, 0)
    b = _rand_monotonic_quantiles(10, 1)
    c = _rand_monotonic_quantiles(10, 2)
    out = blend_quantiles([a, b, c], [1.0, 0.0, 0.0])
    # a is already monotonic, so enforce_monotonic leaves it untouched.
    np.testing.assert_allclose(out, a, rtol=1e-9, atol=1e-6)


def test_blend_is_convex_combination_when_all_monotonic():
    a = _rand_monotonic_quantiles(8, 3)
    b = _rand_monotonic_quantiles(8, 4)
    out = blend_quantiles([a, b], [0.25, 0.75])
    np.testing.assert_allclose(out, 0.25 * a + 0.75 * b, rtol=1e-9, atol=1e-6)


def test_blend_weights_are_renormalized():
    a = _rand_monotonic_quantiles(5, 5)
    b = _rand_monotonic_quantiles(5, 6)
    unnorm = blend_quantiles([a, b], [2.0, 6.0])   # -> 0.25 / 0.75
    norm = blend_quantiles([a, b], [0.25, 0.75])
    np.testing.assert_allclose(unnorm, norm, rtol=1e-9, atol=1e-9)


def test_blend_output_is_monotonic_even_for_crossed_inputs():
    rng = np.random.RandomState(7)
    a = rng.uniform(0, CAPACITY, size=(20, N_LEVELS))   # deliberately unsorted / crossed
    b = rng.uniform(0, CAPACITY, size=(20, N_LEVELS))
    out = blend_quantiles([a, b], [0.5, 0.5])
    assert (np.diff(out, axis=1) >= -1e-9).all()


def test_blend_rejects_mismatched_shapes():
    a = _rand_monotonic_quantiles(5, 8)
    b = _rand_monotonic_quantiles(6, 9)
    with pytest.raises(ValueError):
        blend_quantiles([a, b], [0.5, 0.5])


def test_blend_rejects_negative_weight():
    a = _rand_monotonic_quantiles(5, 10)
    b = _rand_monotonic_quantiles(5, 11)
    with pytest.raises(ValueError):
        blend_quantiles([a, b], [-0.1, 1.1])


def test_distributional_point_single_model_matches_decision_optimal():
    a = _rand_monotonic_quantiles(15, 12)
    b = _rand_monotonic_quantiles(15, 13)
    point = distributional_blend_point([a, b], [1.0, 0.0], CAPACITY)
    ref = decision_optimal_point_prediction(a, QUANTILES, CAPACITY)
    np.testing.assert_allclose(point, ref, rtol=1e-9, atol=1e-6)


def test_distributional_point_degenerate_pointmass_decides_near_that_value():
    # every quantile equal to the same value v -> a (near) point mass at v; the
    # EU-optimal decision must be the grid node nearest v (grid step = cap/100).
    v = 12_000.0
    a = np.full((4, N_LEVELS), v)
    point = distributional_blend_point([a], [1.0], CAPACITY)
    grid_step = CAPACITY / 100.0
    assert np.all(np.abs(point - v) <= grid_step)


def test_distributional_point_wficr_sweep_runs_and_stays_in_bounds():
    a = _rand_monotonic_quantiles(12, 14)
    b = _rand_monotonic_quantiles(12, 15)
    for w_ficr in (0.5, 0.6, 0.7):
        point = distributional_blend_point([a, b], [0.6, 0.4], CAPACITY, w_nmae=1 - w_ficr, w_ficr=w_ficr)
        assert point.shape == (12,)
        assert (point >= 0.0).all() and (point <= CAPACITY + 1e-6).all()
