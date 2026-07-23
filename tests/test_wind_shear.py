"""Unit tests for src/features/wind_shear.py.

Focused on:
  1. power_law_extrapolate's ratio formula (including the identity case
     h_target == h_ref, and scalar-vs-per-row-Series alpha).
  2. estimate_shear_exponent recovering a known, hand-constructed exponent,
     and its guards (v_low<=0 / v_high<=0 -> NaN per-row, not raise;
     h_high==h_low -> raise, not per-row NaN).
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.features.wind_shear import (
    DEFAULT_SHEAR_EXPONENT,
    HUB_HEIGHT_M,
    estimate_shear_exponent,
    power_law_extrapolate,
)


# ---------------------------------------------------------------------------
# power_law_extrapolate
# ---------------------------------------------------------------------------


def test_power_law_extrapolate_identity_when_target_equals_ref_height():
    v_ref = pd.Series([0.0, 3.0, 12.5])
    out = power_law_extrapolate(v_ref, h_ref=100.0, h_target=100.0, alpha=0.3)
    assert out.to_numpy() == pytest.approx(v_ref.to_numpy())


def test_power_law_extrapolate_known_ratio_scalar_alpha():
    # Hand-computed: v_ref=10, h_ref=10, h_target=80, alpha=1/7
    v_ref = 10.0
    expected = 10.0 * (80.0 / 10.0) ** (1.0 / 7.0)
    out = power_law_extrapolate(v_ref, h_ref=10.0, h_target=80.0, alpha=1.0 / 7.0)
    assert out == pytest.approx(expected)


def test_power_law_extrapolate_default_h_target_and_alpha():
    v_ref = 5.0
    expected = 5.0 * (HUB_HEIGHT_M / 100.0) ** DEFAULT_SHEAR_EXPONENT
    out = power_law_extrapolate(v_ref, h_ref=100.0)
    assert out == pytest.approx(expected)


def test_power_law_extrapolate_per_row_alpha_series():
    v_ref = pd.Series([10.0, 10.0, 10.0])
    alpha = pd.Series([0.1, 1.0 / 7.0, 0.3])
    out = power_law_extrapolate(v_ref, h_ref=10.0, h_target=100.0, alpha=alpha)

    expected = v_ref.to_numpy() * (100.0 / 10.0) ** alpha.to_numpy()
    assert isinstance(out, pd.Series)
    assert out.index.equals(v_ref.index)
    assert out.to_numpy() == pytest.approx(expected)


def test_power_law_extrapolate_returns_ndarray_for_non_series_input():
    out = power_law_extrapolate(np.array([1.0, 2.0]), h_ref=10.0, h_target=20.0)
    assert isinstance(out, np.ndarray)
    assert not isinstance(out, pd.Series)


# ---------------------------------------------------------------------------
# estimate_shear_exponent
# ---------------------------------------------------------------------------


def test_estimate_shear_exponent_recovers_known_alpha():
    true_alpha = 0.2
    h_low, h_high = 80.0, 100.0
    v_low = pd.Series([5.0, 8.0, 12.0])
    v_high = v_low * (h_high / h_low) ** true_alpha

    recovered = estimate_shear_exponent(v_low, h_low, v_high, h_high)
    assert recovered.to_numpy() == pytest.approx(np.full(3, true_alpha))


def test_estimate_shear_exponent_guards_non_positive_v_low_and_v_high():
    v_low = pd.Series([5.0, -1.0, 5.0, 0.0])
    v_high = pd.Series([6.0, 6.0, -2.0, 6.0])
    out = estimate_shear_exponent(v_low, 80.0, v_high, 100.0)

    assert not math.isnan(out.iloc[0])
    assert math.isnan(out.iloc[1])  # v_low <= 0
    assert math.isnan(out.iloc[2])  # v_high <= 0
    assert math.isnan(out.iloc[3])  # v_low == 0 (not > 0)


def test_estimate_shear_exponent_raises_on_equal_heights():
    with pytest.raises(ValueError):
        estimate_shear_exponent(pd.Series([5.0]), 100.0, pd.Series([6.0]), 100.0)


def test_estimate_shear_exponent_returns_ndarray_for_non_series_input():
    out = estimate_shear_exponent(5.0, 80.0, 6.0, 100.0)
    assert isinstance(out, np.ndarray)
