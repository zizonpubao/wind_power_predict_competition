"""Unit tests for src/features/weather_features.py.

Focused on the three failure modes most likely to bite silently:
  1. circular-mean bugs in wind direction aggregation (averaging raw degrees
     instead of sin/cos components) -- directions near the 0/360 wrap
     boundary are the classic trap.
  2. power_curve_transform not actually being monotonic / not respecting its
     cut-in / rated / cut-out breakpoints.
  3. lag_rolling_features leaking across the data_available_kst_dtm block
     boundary, the project's core leakage rule (CLAUDE.md section 3).
"""
import math

import numpy as np
import pandas as pd
from pytest import approx

from src.features.weather_features import (
    lag_rolling_features,
    power_curve_transform,
    wind_speed_direction,
)


# ---------------------------------------------------------------------------
# 1. Circular-mean sanity
# ---------------------------------------------------------------------------


def _uv_for_direction(dir_deg: float, speed: float = 1.0) -> tuple[float, float]:
    """Inverse of wind_speed_direction's forward formula
    (dir_rad = atan2(u, v) + pi), so tests can construct u/v pairs that are
    known to resolve to a target meteorological direction.
    """
    theta = math.radians(dir_deg) - math.pi
    u = speed * math.sin(theta)
    v = speed * math.cos(theta)
    return u, v


def test_wind_speed_direction_circular_mean_near_0_360_boundary():
    # Two grids reporting 350 deg and 10 deg should average to ~0 deg
    # (the short way around the compass), NOT ~180 deg (the naive
    # arithmetic-mean-of-degrees bug).
    u1, v1 = _uv_for_direction(350.0)
    u2, v2 = _uv_for_direction(10.0)

    df = pd.DataFrame(
        {
            "forecast_kst_dtm": [pd.Timestamp("2024-01-01 01:00")] * 2,
            "data_available_kst_dtm": [pd.Timestamp("2023-12-31 13:00")] * 2,
            "u": [u1, u2],
            "v": [v1, v2],
        }
    )

    out = wind_speed_direction(df, "u", "v", prefix="test")
    assert len(out) == 1
    dir_deg = out["test_dir_deg"].iloc[0]

    # Distance to 0/360 should be small; distance to 180 (the naive-average
    # bug's answer) should be large.
    dist_to_zero = min(dir_deg, 360 - dir_deg)
    assert dist_to_zero < 5.0, f"expected ~0 deg, got {dir_deg}"
    assert abs(dir_deg - 180) > 100, f"looks like a naive-degree-average bug: {dir_deg}"

    # sin/cos columns should be consistent with dir_deg.
    assert out["test_dir_sin"].iloc[0] == approx(math.sin(math.radians(dir_deg)))
    assert out["test_dir_cos"].iloc[0] == approx(math.cos(math.radians(dir_deg)))


def test_wind_speed_direction_speed_and_grouping():
    # Two forecast hours, two grids each; speed should be the plain grid-mean
    # of sqrt(u^2+v^2), and each forecast hour aggregated independently.
    df = pd.DataFrame(
        {
            "forecast_kst_dtm": [
                pd.Timestamp("2024-01-01 01:00"),
                pd.Timestamp("2024-01-01 01:00"),
                pd.Timestamp("2024-01-01 02:00"),
                pd.Timestamp("2024-01-01 02:00"),
            ],
            "data_available_kst_dtm": [pd.Timestamp("2023-12-31 13:00")] * 4,
            "u": [3.0, 3.0, 0.0, 0.0],
            "v": [4.0, 4.0, 5.0, 5.0],
        }
    )
    out = wind_speed_direction(df, "u", "v", prefix="w")
    assert len(out) == 2
    row0 = out[out["forecast_kst_dtm"] == pd.Timestamp("2024-01-01 01:00")].iloc[0]
    row1 = out[out["forecast_kst_dtm"] == pd.Timestamp("2024-01-01 02:00")].iloc[0]
    assert row0["w_speed_mean"] == approx(5.0)
    assert row1["w_speed_mean"] == approx(5.0)


# ---------------------------------------------------------------------------
# 2. power_curve_transform monotonicity / boundaries
# ---------------------------------------------------------------------------


def test_power_curve_transform_boundaries():
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    speeds = pd.Series([0.0, 1.0, cut_in, 7.5, rated, 18.0, cut_out, 30.0])
    out = power_curve_transform(speeds, cut_in=cut_in, rated=rated, cut_out=cut_out)

    assert out.iloc[0] == 0.0  # calm
    assert out.iloc[1] == 0.0  # below cut-in
    assert out.iloc[2] == 0.0  # exactly at cut-in
    assert 0.0 < out.iloc[3] < 1.0  # mid-ramp
    assert out.iloc[4] == 1.0  # exactly at rated
    assert out.iloc[5] == 1.0  # rated < v < cut-out: flat at 1.0
    assert out.iloc[6] == 1.0  # exactly at cut-out
    assert out.iloc[7] == 0.0  # above cut-out: shut down


def test_power_curve_transform_monotonic_in_ramp_region():
    cut_in, rated, cut_out = 3.0, 12.0, 25.0
    speeds = pd.Series(np.linspace(0.0, cut_out, 200))
    out = power_curve_transform(speeds, cut_in=cut_in, rated=rated, cut_out=cut_out)

    # Overall: never decreasing from 0 up to rated, then flat, then a single
    # drop to 0 above cut_out -- i.e. within [0, rated] it's non-decreasing.
    ramp_region = out[speeds <= rated]
    diffs = np.diff(ramp_region.values)
    assert (diffs >= -1e-12).all(), "power curve must be monotonic non-decreasing up to rated"
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_power_curve_transform_returns_series_with_matching_index():
    speeds = pd.Series([1.0, 5.0, 15.0], index=[10, 20, 30])
    out = power_curve_transform(speeds)
    assert list(out.index) == [10, 20, 30]


# ---------------------------------------------------------------------------
# 3. lag_rolling_features must not leak across data_available_kst_dtm blocks
# ---------------------------------------------------------------------------


def _make_two_block_df() -> pd.DataFrame:
    block_a_available = pd.Timestamp("2024-01-01 13:00")
    block_b_available = pd.Timestamp("2024-01-02 13:00")

    # Block A: 5 forecast hours with large values (ends right before block B
    # starts, so a leaking rolling window would pick these up).
    block_a = pd.DataFrame(
        {
            "data_available_kst_dtm": [block_a_available] * 5,
            "forecast_kst_dtm": pd.date_range("2024-01-02 01:00", periods=5, freq="h"),
            "val": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        }
    )
    # Block B: 5 forecast hours with small values, immediately following A in
    # forecast_kst_dtm order.
    block_b = pd.DataFrame(
        {
            "data_available_kst_dtm": [block_b_available] * 5,
            "forecast_kst_dtm": pd.date_range("2024-01-02 06:00", periods=5, freq="h"),
            "val": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    return pd.concat([block_a, block_b], ignore_index=True)


def test_lag_rolling_features_lead_hour_resets_per_block():
    df = _make_two_block_df()
    out = lag_rolling_features(df, "data_available_kst_dtm", ["val"], windows=[3])
    lead_hours_a = out[out["data_available_kst_dtm"] == df["data_available_kst_dtm"].iloc[0]][
        "lead_hour"
    ].tolist()
    lead_hours_b = out[out["data_available_kst_dtm"] == df["data_available_kst_dtm"].iloc[5]][
        "lead_hour"
    ].tolist()
    assert lead_hours_a == [1, 2, 3, 4, 5]
    assert lead_hours_b == [1, 2, 3, 4, 5]


def test_lag_rolling_features_no_leakage_across_block_boundary():
    df = _make_two_block_df()
    out = lag_rolling_features(df, "data_available_kst_dtm", ["val"], windows=[3])

    block_b_available = df["data_available_kst_dtm"].iloc[5]
    block_b_rows = out[out["data_available_kst_dtm"] == block_b_available].sort_values(
        "forecast_kst_dtm"
    )

    # First row of block B: only 1 value available *within the block*
    # (val=1.0). A leaking window would pull in block A's val=5000.0 and
    # produce a mean in the thousands.
    first_row = block_b_rows.iloc[0]
    assert first_row["val_roll_mean_3h"] == approx(1.0)
    assert first_row["lead_hour"] == 1

    # Third row of block B: window=3 now has exactly 3 in-block values
    # (1,2,3) available -- mean should be 2.0, not something pulled from
    # block A.
    third_row = block_b_rows.iloc[2]
    assert third_row["val_roll_mean_3h"] == approx(2.0)

    # Sanity: block A's last row's window is entirely within block A.
    block_a_available = df["data_available_kst_dtm"].iloc[0]
    block_a_rows = out[out["data_available_kst_dtm"] == block_a_available].sort_values(
        "forecast_kst_dtm"
    )
    last_row_a = block_a_rows.iloc[-1]
    # last 3 values of block A: 3000, 4000, 5000 -> mean 4000
    assert last_row_a["val_roll_mean_3h"] == approx(4000.0)


def test_lag_rolling_features_std_is_nan_for_single_observation():
    df = _make_two_block_df()
    out = lag_rolling_features(df, "data_available_kst_dtm", ["val"], windows=[3])
    block_b_available = df["data_available_kst_dtm"].iloc[5]
    first_row = out[
        (out["data_available_kst_dtm"] == block_b_available) & (out["lead_hour"] == 1)
    ].iloc[0]
    # std of a single observation is undefined (NaN), never leaks in a
    # cross-block value to make it computable.
    assert pd.isna(first_row["val_roll_std_3h"])
