"""Unit tests for src/features/wake_features.py.

Focused on the boundary values reports/domain_research/wake_effect.md's
formulas are supposed to hit exactly:
  1. wind blowing from exactly WAKE_FROM_DIR_DEG (295 deg) -> perfect wake
     alignment (cos=1, full sector exposure, max deficit).
  2. wind blowing from the exact opposite direction (115 deg) -> alignment=-1,
     zero sector exposure, and (critically) zero deficit proxy -- the
     max(0, ...) clip must zero out a negative-cosine case, not let it leak
     through as a negative "deficit".
  3. wind blowing exactly at the sector edge (295 +/- 25 deg) -> sector
     exposure exactly 0 (ramp fully decayed).
  4. angle-wrap correctness near the 0/360 boundary (e.g. WAKE_FROM_DIR_DEG=295
     vs dir_deg=359.9 differ by only ~64.9 deg the "short way", not ~295 deg).
"""
import math

import numpy as np
from pytest import approx

from src.features.wake_features import (
    BEARING_G1_TO_G2_DEG,
    WAKE_DEFICIT_FRAC_CONST,
    WAKE_FROM_DIR_DEG,
    WAKE_SECTOR_HALF_WIDTH_DEG,
    add_group1_group2_wake_features,
    wake_alignment_cos,
    wake_deficit_proxy,
    wake_sector_exposure,
)

import pandas as pd


def _sin_cos(dir_deg: float) -> tuple[float, float]:
    rad = math.radians(dir_deg)
    return math.sin(rad), math.cos(rad)


# ---------------------------------------------------------------------------
# Constants sanity (the exact values the task spec / domain report fixed)
# ---------------------------------------------------------------------------


def test_constants_match_domain_report():
    assert BEARING_G1_TO_G2_DEG == 115.0
    assert WAKE_FROM_DIR_DEG == approx(295.0)
    assert WAKE_SECTOR_HALF_WIDTH_DEG == 25.0
    assert WAKE_DEFICIT_FRAC_CONST == 0.075


# ---------------------------------------------------------------------------
# wake_alignment_cos
# ---------------------------------------------------------------------------


def test_wake_alignment_cos_perfect_alignment_at_295():
    dir_sin, dir_cos = _sin_cos(295.0)
    result = wake_alignment_cos(dir_sin, dir_cos)
    assert float(result) == approx(1.0)


def test_wake_alignment_cos_opposite_direction_at_115():
    dir_sin, dir_cos = _sin_cos(115.0)
    result = wake_alignment_cos(dir_sin, dir_cos)
    assert float(result) == approx(-1.0)


def test_wake_alignment_cos_crosswind_is_zero():
    # 295 +/- 90 deg is a pure crosswind relative to the wake axis.
    dir_sin, dir_cos = _sin_cos(295.0 + 90.0)
    result = wake_alignment_cos(dir_sin, dir_cos)
    assert float(result) == approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# wake_sector_exposure
# ---------------------------------------------------------------------------


def test_wake_sector_exposure_full_at_center():
    assert float(wake_sector_exposure(295.0)) == approx(1.0)


def test_wake_sector_exposure_zero_at_opposite_direction():
    # 105 deg (the task spec's example): |105-295|=190 -> short way = 170 deg,
    # far outside the 25 deg half-width -> exactly 0 (clipped, not negative).
    assert float(wake_sector_exposure(105.0)) == 0.0


def test_wake_sector_exposure_zero_exactly_at_sector_edge():
    for edge in (295.0 + 25.0, 295.0 - 25.0):
        assert float(wake_sector_exposure(edge % 360.0)) == approx(0.0, abs=1e-9)


def test_wake_sector_exposure_half_at_half_width():
    # Halfway to the edge (12.5 deg off-center) should give ramp value 0.5.
    assert float(wake_sector_exposure(295.0 + 12.5)) == approx(0.5)


def test_wake_sector_exposure_handles_0_360_wrap():
    # dir_deg=359.9 vs wake_from_dir_deg=295: naive |diff| = 64.9, which
    # happens to already be the short way here, but check a wrap case where
    # naive diff would be wrong: dir_deg=0 (i.e. 360) vs a hypothetical
    # wake_from_dir_deg near 359 should resolve to a tiny angle_diff.
    exposure = wake_sector_exposure(0.0, wake_from_dir_deg=359.0, half_width_deg=25.0)
    # angle_diff should be 1 deg (360 - 359), not 359 deg.
    assert float(exposure) == approx(1.0 - 1.0 / 25.0)


# ---------------------------------------------------------------------------
# wake_deficit_proxy
# ---------------------------------------------------------------------------


def test_wake_deficit_proxy_scales_with_speed_and_const_at_perfect_alignment():
    speed = 10.0
    alignment = 1.0
    result = wake_deficit_proxy(speed, alignment)
    assert float(result) == approx(10.0 * 1.0 * WAKE_DEFICIT_FRAC_CONST)


def test_wake_deficit_proxy_is_zero_when_alignment_negative():
    # Group2 upwind of group1 (alignment_cos < 0): must clip to exactly 0,
    # never a negative "deficit".
    result = wake_deficit_proxy(speed_mean=15.0, alignment_cos=-0.8)
    assert float(result) == 0.0


def test_wake_deficit_proxy_array_input():
    speed = np.array([0.0, 5.0, 10.0])
    alignment = np.array([-1.0, 0.0, 1.0])
    result = wake_deficit_proxy(speed, alignment)
    expected = np.array([0.0, 0.0, 10.0 * WAKE_DEFICIT_FRAC_CONST])
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# add_group1_group2_wake_features (dataframe-level orchestrator)
# ---------------------------------------------------------------------------


def test_add_group1_group2_wake_features_adds_expected_columns():
    dir_sin_295, dir_cos_295 = _sin_cos(295.0)
    dir_sin_115, dir_cos_115 = _sin_cos(115.0)
    df = pd.DataFrame(
        {
            "ldaps_10m_dir_sin": [dir_sin_295, dir_sin_115],
            "ldaps_10m_dir_cos": [dir_cos_295, dir_cos_115],
            "ldaps_10m_dir_deg": [295.0, 115.0],
            "ldaps_10m_speed_mean": [10.0, 10.0],
        }
    )
    out = add_group1_group2_wake_features(df)

    for col in ("wake_alignment_cos_g1g2", "wake_sector_exposure_g1g2", "wake_deficit_proxy_g1g2"):
        assert col in out.columns

    # Row 0: wind exactly from 295 deg -> full alignment/exposure/deficit.
    assert out["wake_alignment_cos_g1g2"].iloc[0] == approx(1.0)
    assert out["wake_sector_exposure_g1g2"].iloc[0] == approx(1.0)
    assert out["wake_deficit_proxy_g1g2"].iloc[0] == approx(10.0 * WAKE_DEFICIT_FRAC_CONST)

    # Row 1: wind exactly from 115 deg (opposite) -> no wake possible.
    assert out["wake_alignment_cos_g1g2"].iloc[1] == approx(-1.0)
    assert out["wake_sector_exposure_g1g2"].iloc[1] == 0.0
    assert out["wake_deficit_proxy_g1g2"].iloc[1] == 0.0


def test_add_group1_group2_wake_features_missing_column_raises():
    df = pd.DataFrame({"ldaps_10m_dir_sin": [0.0]})
    try:
        add_group1_group2_wake_features(df)
        assert False, "expected KeyError for missing required columns"
    except KeyError:
        pass
