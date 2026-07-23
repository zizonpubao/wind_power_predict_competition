"""Unit tests for src/features/decision_optimize.py (pure-function
decision-theoretic post-processing of 9-quantile GBM predictions).

All expected values below are either hand-derived (see comments) or
structural invariants -- not "does it match whatever the implementation
happens to compute".
"""
import numpy as np
import pytest

from src.features.decision_optimize import (
    QUANTILES,
    decision_optimal_point_prediction,
    enforce_monotonic_quantiles,
    expected_utility_grid_search,
    quantiles_to_pmf,
)

CAPACITY = 21_600.0
N_GRID = 101
GRID_STEP = CAPACITY / (N_GRID - 1)


# ---------------------------------------------------------------------------
# enforce_monotonic_quantiles
# ---------------------------------------------------------------------------


def test_enforce_monotonic_quantiles_leaves_already_sorted_rows_unchanged():
    q = np.array([[100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0]])
    out = enforce_monotonic_quantiles(q)
    np.testing.assert_allclose(out, q)


def test_enforce_monotonic_quantiles_fixes_crossed_quantiles_and_stays_in_range():
    # Deliberately non-monotonic (crossed) row.
    q = np.array([[500.0, 100.0, 900.0, 200.0, 850.0, 300.0, 950.0, 400.0, 1000.0]])
    out = enforce_monotonic_quantiles(q)

    assert (np.diff(out, axis=1) >= 0).all(), "output must be non-decreasing per row"
    assert out.min() >= q.min() - 1e-9
    assert out.max() <= q.max() + 1e-9


def test_enforce_monotonic_quantiles_output_within_original_min_max_generic():
    rng = np.random.RandomState(0)
    q = rng.uniform(0, 21_600.0, size=(20, len(QUANTILES)))  # unsorted, arbitrary rows
    out = enforce_monotonic_quantiles(q)
    assert (np.diff(out, axis=1) >= -1e-9).all()
    assert out.min() >= q.min() - 1e-9
    assert out.max() <= q.max() + 1e-9


# ---------------------------------------------------------------------------
# quantiles_to_pmf
# ---------------------------------------------------------------------------


def test_quantiles_to_pmf_all_zero_concentrates_mass_at_grid_zero_no_div_by_zero():
    q = np.zeros((3, len(QUANTILES)))
    grid, pmf = quantiles_to_pmf(q, QUANTILES, CAPACITY, n_grid=N_GRID)

    assert not np.isnan(pmf).any()
    assert not np.isinf(pmf).any()
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-9)
    # 9 quantile anchors all sit at 0, with the largest (0.95) anchor jumping
    # the CDF to 0.95 right at x=0 -- so at least 95% of pmf mass must land
    # on grid index 0.
    assert (pmf[:, 0] >= 0.95 - 1e-9).all()
    assert grid[0] == pytest.approx(0.0)


def test_quantiles_to_pmf_all_near_capacity_concentrates_mass_at_grid_end():
    q = np.full((2, len(QUANTILES)), CAPACITY)
    grid, pmf = quantiles_to_pmf(q, QUANTILES, CAPACITY, n_grid=N_GRID)

    assert not np.isnan(pmf).any()
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-9)
    # All 9 quantile anchors sit at capacity, so CDF must reach 1.0 already
    # at (or essentially at) the last grid point.
    assert (pmf[:, -1] > 0.5).all()
    assert grid[-1] == pytest.approx(CAPACITY)


def test_quantiles_to_pmf_sums_to_one_for_generic_increasing_quantiles():
    q = np.array([[1000.0, 2000.0, 3500.0, 5000.0, 7000.0, 9000.0, 11000.0, 14000.0, 18000.0]])
    grid, pmf = quantiles_to_pmf(q, QUANTILES, CAPACITY, n_grid=N_GRID)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, atol=1e-9)
    assert (pmf >= 0).all()


# ---------------------------------------------------------------------------
# expected_utility_grid_search -- analytic w_nmae=0 / w_ficr=0 cases
# ---------------------------------------------------------------------------


def test_expected_utility_w_ficr_zero_recovers_the_median():
    """With w_ficr=0, EU reduces to maximizing E[1 - |c-X|/capacity], i.e.
    minimizing E|c-X| -- minimized (over real c) at the distribution's
    median. Our piecewise-linear CDF construction passes exactly through the
    (q_values[4]=0.5-quantile, y=0.5) anchor, so the true median of the
    constructed distribution is exactly the input's own 0.5-quantile value.
    """
    q_row = np.array([1000.0, 2000.0, 3500.0, 5000.0, 7000.0, 9000.0, 11000.0, 14000.0, 18000.0])
    q = q_row[None, :]
    decision = decision_optimal_point_prediction(q, QUANTILES, CAPACITY, w_nmae=1.0, w_ficr=0.0)
    assert decision[0] == pytest.approx(q_row[4], abs=GRID_STEP + 1e-6)


def test_expected_utility_w_nmae_zero_matches_hand_derived_discrete_optimum():
    """Pure-FICR objective (w_nmae=0) on a hand-built two-point pmf: 30% mass
    at value 20, 70% mass at value 80 (capacity=100, grid step=1).

    E[actual] = 0.3*20 + 0.7*80 = 62; denom = 4*62 = 248.
    For candidate c, EU_ficr(c) = [6*rate(c,20) + 56*rate(c,80)] / 248
    (since pmf(20)*20=6, pmf(80)*80=56).

    Putting all weight near 80 dominates (56 >> 6), and rate(c,80) is
    saturated at its max (4) for any c with |c-80|<=6 -- so the EU-maximizing
    plateau is exactly c in [74, 86]. Ties broken by np.argmax's
    "first occurrence" rule (ascending candidate order) put the decision at
    the *left* edge of that plateau, c=74 (since c=73 already drops to the
    3-tier rate: |73-80|=7 > 6).
    """
    grid = np.linspace(0.0, 100.0, 101)
    pmf = np.zeros((1, 101))
    pmf[0, 20] = 0.3
    pmf[0, 80] = 0.7

    decision = expected_utility_grid_search(grid, pmf, capacity_kwh=100.0, w_nmae=0.0, w_ficr=1.0)
    assert decision[0] == pytest.approx(74.0, abs=1e-9)


# ---------------------------------------------------------------------------
# decision_optimal_point_prediction -- end-to-end behavior
# ---------------------------------------------------------------------------


def test_decision_point_equals_common_value_when_all_quantiles_agree():
    v = 47.0 / 100.0 * CAPACITY  # lands exactly on a grid point (grid step = capacity/100)
    q = np.full((4, len(QUANTILES)), v)
    decision = decision_optimal_point_prediction(q, QUANTILES, CAPACITY)
    np.testing.assert_allclose(decision, v, atol=GRID_STEP / 2 + 1e-6)


def test_decision_point_all_zero_is_zero_no_nan():
    q = np.zeros((5, len(QUANTILES)))
    decision = decision_optimal_point_prediction(q, QUANTILES, CAPACITY)
    assert not np.isnan(decision).any()
    np.testing.assert_allclose(decision, 0.0, atol=1e-9)


def test_decision_point_all_near_capacity_is_near_capacity():
    q = np.full((3, len(QUANTILES)), CAPACITY)
    decision = decision_optimal_point_prediction(q, QUANTILES, CAPACITY)
    assert not np.isnan(decision).any()
    assert (decision <= CAPACITY + 1e-9).all()
    assert (decision >= CAPACITY - 2 * GRID_STEP).all()


def test_decision_point_within_range_after_quantile_crossing():
    # The *enforce_monotonic_quantiles* step's output is guaranteed to stay
    # within [min(q), max(q)] (tested directly above) -- but the final
    # EU-optimal decision is not required to: it's chosen against a pmf that
    # also includes the (capacity, 1.0) anchor's tail probability mass beyond
    # the largest observed quantile, so it may legitimately land a bit past
    # max(q) if that better balances the utility trade-off. The only hard
    # invariant at this final stage is staying within [0, capacity] and never
    # producing NaN -- checked here; the tighter "sorted output in range"
    # invariant is covered by the enforce_monotonic_quantiles tests above.
    q = np.array([[500.0, 100.0, 900.0, 200.0, 850.0, 300.0, 950.0, 400.0, 1000.0]])
    decision = decision_optimal_point_prediction(q, QUANTILES, CAPACITY)
    assert not np.isnan(decision).any()
    assert decision[0] >= 0.0 - 1e-6
    assert decision[0] <= CAPACITY + 1e-6


def test_decision_point_translation_invariance():
    q_row = np.array([1000.0, 2000.0, 3500.0, 5000.0, 7000.0, 9000.0, 11000.0, 14000.0, 18000.0])
    shift = 500.0

    decision_base = decision_optimal_point_prediction(q_row[None, :], QUANTILES, CAPACITY)
    decision_shifted = decision_optimal_point_prediction((q_row + shift)[None, :], QUANTILES, CAPACITY)

    assert decision_shifted[0] - decision_base[0] == pytest.approx(shift, abs=2 * GRID_STEP)


def test_decision_point_vectorized_matches_row_by_row():
    rng = np.random.RandomState(0)
    n_rows = 6
    base = np.sort(rng.uniform(0, CAPACITY, size=(n_rows, len(QUANTILES))), axis=1)

    batch_decision = decision_optimal_point_prediction(base, QUANTILES, CAPACITY)
    row_decisions = np.array(
        [decision_optimal_point_prediction(base[i : i + 1, :], QUANTILES, CAPACITY)[0] for i in range(n_rows)]
    )

    np.testing.assert_allclose(batch_decision, row_decisions, atol=1e-9)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_enforce_monotonic_quantiles_rejects_non_2d_input():
    with pytest.raises(ValueError):
        enforce_monotonic_quantiles(np.array([1.0, 2.0, 3.0]))


def test_quantiles_to_pmf_rejects_mismatched_quantile_levels_length():
    q = np.zeros((2, len(QUANTILES)))
    with pytest.raises(ValueError):
        quantiles_to_pmf(q, QUANTILES[:-1], CAPACITY, n_grid=N_GRID)


def test_expected_utility_grid_search_rejects_mismatched_pmf_shape():
    grid = np.linspace(0.0, CAPACITY, N_GRID)
    bad_pmf = np.zeros((2, N_GRID - 1))
    with pytest.raises(ValueError):
        expected_utility_grid_search(grid, bad_pmf, CAPACITY)
