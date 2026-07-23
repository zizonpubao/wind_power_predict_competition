"""Unit tests for the FICR step-surrogate objective (idea A3) in
src/models/lgbm_model.py -- the pure grad/hess functions, the smooth-rate
approximation, and the picklable ``FICRSurrogateObjective`` wrapper + its
``GroupLGBMModel`` integration.

Pure-function tests use tiny hand-checkable arrays; the end-to-end
"does it actually learn" property (which caught the cold-start/scaling bug)
is covered by running src/training/train_ficr_surrogate.py, not this suite.
"""
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    FICR_TIER1_NMAE_THRESHOLD,
    FICR_TIER1_RATE,
    FICR_TIER2_NMAE_THRESHOLD,
    FICR_TIER2_RATE,
    FICR_TIER3_RATE,
)
from src.models.lgbm_model import (
    FICRSurrogateObjective,
    GroupLGBMModel,
    ficr_smooth_rate,
    ficr_surrogate_grad_hess,
    huber_grad_hess_eps,
)

CAP = 21_600.0

# The full explicit kwarg set so pure-function tests don't depend on the
# class defaults (which are tuned for training, not for hand-checkable math).
BASE_KW = dict(
    capacity_kwh=CAP,
    w_cliff=1.0,
    w_l2=1.0,
    huber_delta=1.0,
    steepness=400.0,
    min_utilization=0.10,
    ineligible_weight=0.05,
    actual_weight_floor=0.05,
    huber_tail_hess=1.0,
)


def _kw(**overrides):
    kw = dict(BASE_KW)
    kw.update(overrides)
    return kw


# --------------------------------------------------------------------------
# smooth_rate: the two-sigmoid differentiable approximation of the step
# --------------------------------------------------------------------------


def test_smooth_rate_hits_the_three_tiers_between_cliffs():
    """Far below 6%, between 6-8%, and far above 8% the smooth rate must sit
    at ~tier1/tier2/tier3 respectively (the whole point of the approximation).
    """
    e = np.array([0.0, 0.07, 0.20])
    r = ficr_smooth_rate(e, steepness=400.0)
    assert r[0] == pytest.approx(FICR_TIER1_RATE, abs=1e-3)
    # midpoint between the cliffs: near tier2 but not exactly (finite steepness
    # means each sigmoid is only ~98% saturated 0.01 away from its threshold)
    assert r[1] == pytest.approx(FICR_TIER2_RATE, abs=0.1)
    assert r[2] == pytest.approx(FICR_TIER3_RATE, abs=1e-3)


def test_smooth_rate_is_monotonically_non_increasing_in_error():
    e = np.linspace(0.0, 0.2, 400)
    r = ficr_smooth_rate(e, steepness=400.0)
    assert np.all(np.diff(r) <= 1e-9)


def test_smooth_rate_at_cliff_midpoints_is_between_adjacent_tiers():
    """At the exact cliff thresholds the sigmoid is at its half-way point, so
    the rate should sit strictly between the two adjacent tier rates."""
    r = ficr_smooth_rate(
        np.array([FICR_TIER1_NMAE_THRESHOLD, FICR_TIER2_NMAE_THRESHOLD]), steepness=400.0
    )
    assert FICR_TIER2_RATE < r[0] < FICR_TIER1_RATE
    assert FICR_TIER3_RATE < r[1] < FICR_TIER2_RATE


# --------------------------------------------------------------------------
# huber_grad_hess_eps
# --------------------------------------------------------------------------


def test_huber_quadratic_region_is_identity_grad_unit_hess():
    eps = np.array([-0.5, 0.0, 0.3, 0.9])
    grad, hess = huber_grad_hess_eps(eps, delta=1.0, tail_hess=1.0)
    np.testing.assert_allclose(grad, eps)
    np.testing.assert_allclose(hess, np.ones_like(eps))


def test_huber_linear_tail_caps_grad_and_uses_tail_hess():
    eps = np.array([-3.0, 2.0])
    grad, hess = huber_grad_hess_eps(eps, delta=0.5, tail_hess=0.25)
    np.testing.assert_allclose(grad, [-0.5, 0.5])
    np.testing.assert_allclose(hess, [0.25, 0.25])


# --------------------------------------------------------------------------
# ficr_surrogate_grad_hess: the combined loss grad/hess w.r.t. pred (kWh)
# --------------------------------------------------------------------------


def test_zero_error_gives_zero_gradient_and_positive_hessian():
    """At pred==actual the loss is at its minimum: gradient must be ~0 (nothing
    to push), and Hessian strictly positive (LightGBM needs it)."""
    actual = np.array([10_000.0, 15_000.0, 5_000.0])
    grad, hess = ficr_surrogate_grad_hess(actual, actual.copy(), **_kw())
    np.testing.assert_allclose(grad, 0.0, atol=1e-6)
    assert np.all(hess > 0.0)


def test_gradient_points_toward_actual():
    """Over-prediction (pred>actual) must give positive gradient (LightGBM
    steps pred *down*); under-prediction negative gradient (steps pred up).
    """
    actual = np.array([10_000.0, 10_000.0])
    over = np.array([12_000.0, 10_000.0])
    under = np.array([8_000.0, 10_000.0])
    g_over, _ = ficr_surrogate_grad_hess(actual, over, **_kw())
    g_under, _ = ficr_surrogate_grad_hess(actual, under, **_kw())
    assert g_over[0] > 0.0  # pred above actual -> push down
    assert g_under[0] < 0.0  # pred below actual -> push up


def test_hessian_strictly_positive_across_error_range_including_cliffs():
    actual = np.full(200, 12_000.0)
    # sweep predictions so the error rate crosses both 6% and 8% cliffs
    pred = actual + np.linspace(-0.2, 0.2, 200) * CAP
    _, hess = ficr_surrogate_grad_hess(actual, pred, **_kw(w_cliff=5.0))
    assert np.all(hess > 0.0)


def test_cliff_term_raises_loss_gradient_magnitude_near_the_cliffs():
    """With w_cliff>0 the |gradient| for an error sitting right at a cliff must
    exceed the pure-Huber (w_cliff=0) gradient at the same point -- that extra
    "pull" away from the cliff is the entire mechanism of the surrogate.
    """
    actual = np.array([12_000.0])
    # error rate ~0.06 (right at the first cliff)
    pred = actual + FICR_TIER1_NMAE_THRESHOLD * CAP
    g_cliff, _ = ficr_surrogate_grad_hess(actual, pred, **_kw(w_cliff=3.0))
    g_plain, _ = ficr_surrogate_grad_hess(actual, pred, **_kw(w_cliff=0.0))
    assert abs(g_cliff[0]) > abs(g_plain[0])


def test_w_cliff_zero_reduces_to_weighted_scaled_huber():
    """w_cliff=0 must be exactly the (row-weighted, cap-scaled) Huber term --
    the "pure Huber floor" the sweep uses as its baseline config.
    """
    actual = np.array([12_000.0, 3_000.0, 18_000.0])
    pred = np.array([13_000.0, 3_500.0, 10_000.0])
    kw = _kw(w_cliff=0.0, w_l2=1.0)
    grad, hess = ficr_surrogate_grad_hess(actual, pred, **kw)

    # Reconstruct the expected weighted/scaled Huber by hand.
    cap = CAP
    eps = (pred - actual) / cap
    h_grad, h_hess = huber_grad_hess_eps(eps, kw["huber_delta"], kw["huber_tail_hess"])
    eligible = actual >= kw["min_utilization"] * cap
    base_w = np.where(eligible, 1.0, kw["ineligible_weight"])
    actual_w = np.clip(actual / cap, kw["actual_weight_floor"], None)
    weight = base_w * actual_w
    exp_grad = weight * cap * (kw["w_l2"] * h_grad)
    exp_hess = weight * (kw["w_l2"] * h_hess)

    np.testing.assert_allclose(grad, exp_grad)
    np.testing.assert_allclose(hess, exp_hess)


def test_ineligible_rows_get_downweighted():
    """A row whose actual is below the 10% eligibility threshold must get a
    strictly smaller-magnitude gradient than an otherwise-identical error on an
    eligible high-output row (settlement is proportional to generation + the
    eligibility filter, both baked into the weight).
    """
    # same *absolute* residual on a tiny-output vs a large-output row
    actual = np.array([500.0, 12_000.0])  # 500 << 0.1*cap (=2160) -> ineligible
    pred = actual + 1_000.0
    grad, _ = ficr_surrogate_grad_hess(actual, pred, **_kw())
    assert abs(grad[0]) < abs(grad[1])


def test_vectorized_matches_scalar_loop():
    rng = np.random.RandomState(3)
    actual = rng.uniform(0, CAP, size=50)
    pred = actual + rng.uniform(-0.3, 0.3, size=50) * CAP
    grad_vec, hess_vec = ficr_surrogate_grad_hess(actual, pred, **_kw(w_cliff=2.0))
    for i in range(len(actual)):
        g, h = ficr_surrogate_grad_hess(actual[i : i + 1], pred[i : i + 1], **_kw(w_cliff=2.0))
        assert g[0] == pytest.approx(grad_vec[i])
        assert h[0] == pytest.approx(hess_vec[i])


@pytest.mark.parametrize(
    "bad", [dict(capacity_kwh=0.0), dict(capacity_kwh=-1.0), dict(w_cliff=-0.1), dict(w_l2=0.0), dict(steepness=0.0)]
)
def test_grad_hess_rejects_invalid_params(bad):
    with pytest.raises(ValueError):
        ficr_surrogate_grad_hess(np.array([1.0]), np.array([1.0]), **_kw(**bad))


# --------------------------------------------------------------------------
# FICRSurrogateObjective wrapper + GroupLGBMModel integration
# --------------------------------------------------------------------------


def test_objective_call_matches_pure_function():
    obj = FICRSurrogateObjective(CAP, w_cliff=1.5, w_l2=1.0)
    actual = np.array([10_000.0, 4_000.0])
    pred = np.array([11_000.0, 3_000.0])
    grad, hess = obj(actual, pred)
    exp_grad, exp_hess = ficr_surrogate_grad_hess(
        actual,
        pred,
        capacity_kwh=obj.capacity_kwh,
        w_cliff=obj.w_cliff,
        w_l2=obj.w_l2,
        huber_delta=obj.huber_delta,
        steepness=obj.steepness,
        min_utilization=obj.min_utilization,
        ineligible_weight=obj.ineligible_weight,
        actual_weight_floor=obj.actual_weight_floor,
        huber_tail_hess=obj.huber_tail_hess,
    )
    np.testing.assert_allclose(grad, exp_grad)
    np.testing.assert_allclose(hess, exp_hess)


def test_objective_pickle_round_trip():
    obj = FICRSurrogateObjective(CAP, w_cliff=0.7, w_l2=1.3, huber_delta=0.9, steepness=350.0)
    loaded = pickle.loads(pickle.dumps(obj))
    actual = np.array([9_000.0, 15_000.0])
    pred = np.array([9_500.0, 12_000.0])
    np.testing.assert_allclose(loaded(actual, pred)[0], obj(actual, pred)[0])
    np.testing.assert_allclose(loaded(actual, pred)[1], obj(actual, pred)[1])


@pytest.mark.parametrize("bad", [dict(capacity_kwh=0.0), dict(w_cliff=-1.0), dict(w_l2=0.0), dict(steepness=-5.0)])
def test_objective_constructor_rejects_invalid_params(bad):
    kw = dict(capacity_kwh=CAP)
    kw.update(bad)
    with pytest.raises(ValueError):
        FICRSurrogateObjective(**kw)


def test_grouplgbmmodel_rejects_both_asymmetry_and_surrogate():
    with pytest.raises(ValueError):
        GroupLGBMModel(capacity_kwh=CAP, asymmetry_alpha=0.7, ficr_surrogate={"w_cliff": 1.0})


def test_grouplgbmmodel_injects_capacity_and_sets_feature_pre_filter():
    model = GroupLGBMModel(capacity_kwh=CAP, ficr_surrogate={"w_cliff": 0.5, "w_l2": 1.0}, n_estimators=10)
    obj = model.params["objective"]
    assert isinstance(obj, FICRSurrogateObjective)
    assert obj.capacity_kwh == pytest.approx(CAP)
    assert obj.w_cliff == pytest.approx(0.5)
    assert model.params["feature_pre_filter"] is False


def test_grouplgbmmodel_surrogate_trains_predicts_and_survives_joblib(tmp_path):
    """The surrogate must actually learn a non-trivial (non-constant-0) fit and
    survive a joblib round trip -- the regression guard for the cold-start/
    scaling bug where the model collapsed to all-zero predictions.
    """
    rng = np.random.RandomState(0)
    n = 800
    X = pd.DataFrame(rng.rand(n, 6), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((X.values.sum(axis=1) / 6) * CAP * 0.9 + rng.rand(n) * 500.0)

    model = GroupLGBMModel(
        capacity_kwh=CAP, ficr_surrogate={"w_cliff": 0.5, "w_l2": 1.0}, n_estimators=60
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (n,)
    assert preds.std() > 0.0  # not a collapsed constant-0 predictor
    assert np.corrcoef(preds, y)[0, 1] > 0.8

    dump_path = tmp_path / "surr.joblib"
    joblib.dump(model, dump_path)
    loaded = joblib.load(dump_path)
    np.testing.assert_allclose(loaded.predict(X), preds)
