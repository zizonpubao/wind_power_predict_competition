"""Integration test for src/features/build_features.py.

Runs the real assembly pipeline (_assemble_feature_table) on a one-week
time-sliced subset of the actual LDAPS/GFS train data -- fast enough for the
default test suite while still exercising the real column names, real
spatial_aggregate/wind_speed_direction/power_curve_transform/lag_rolling_features
composition, and the real label merge. Loading load_ldaps("train")/
load_gfs("train") once and then slicing is much cheaper than re-reading the
full CSVs per test case, so all assertions share one fixture-scoped load.
"""
import pandas as pd
import pytest

from src.data.loaders import load_gfs, load_ldaps
from src.features.build_features import (
    _GFS_RENAME,
    _LDAPS_RENAME,
    _assemble_feature_table,
)

_ONE_WEEK = pd.Timedelta(days=7)


@pytest.fixture(scope="module")
def one_week_frames():
    ldaps = load_ldaps("train").rename(columns=_LDAPS_RENAME)
    gfs = load_gfs("train").rename(columns=_GFS_RENAME)

    start = ldaps["forecast_kst_dtm"].min()
    end = start + _ONE_WEEK

    ldaps_small = ldaps[
        (ldaps["forecast_kst_dtm"] >= start) & (ldaps["forecast_kst_dtm"] < end)
    ].reset_index(drop=True)
    gfs_small = gfs[
        (gfs["forecast_kst_dtm"] >= start) & (gfs["forecast_kst_dtm"] < end)
    ].reset_index(drop=True)
    return ldaps_small, gfs_small


def test_build_feature_table_train_no_leakage_and_expected_columns(one_week_frames):
    ldaps_small, gfs_small = one_week_frames
    out = _assemble_feature_table(ldaps_small, gfs_small, "train", "kpx_group_1")

    assert len(out) > 0

    # 1. No leakage: every row's data_available_kst_dtm must be <= its
    #    forecast_kst_dtm (the project's core leakage rule, CLAUDE.md section 3).
    assert (out["data_available_kst_dtm"] <= out["forecast_kst_dtm"]).all()

    # 2. Expected key columns present: block keys, per-source IDW wind speed /
    #    power-curve outputs (the columns lag_rolling_features was scoped to),
    #    a lag/rolling-derived column, lead_hour, and the train-only target.
    expected_cols = {
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "ldaps_10m_speed_idw",
        "ldaps_10m_power_curve_idw",
        "gfs_10m_speed_idw",
        "gfs_10m_power_curve_idw",
        "gfs_80m_speed_idw",
        "gfs_100m_speed_idw",
        "gfs_pbl_speed_idw",
        "lead_hour",
        "ldaps_10m_speed_idw_roll_mean_3h",
        "target",
    }
    missing = expected_cols - set(out.columns)
    assert not missing, f"missing expected columns: {missing}"

    # 3. No fully-duplicate rows.
    assert not out.duplicated().any()


def test_build_feature_table_test_split_has_no_target_column(one_week_frames):
    ldaps_small, gfs_small = one_week_frames
    out = _assemble_feature_table(ldaps_small, gfs_small, "test", "kpx_group_2")

    assert len(out) > 0
    assert "target" not in out.columns
    assert (out["data_available_kst_dtm"] <= out["forecast_kst_dtm"]).all()
    assert not out.duplicated().any()


def test_lead_hour_resets_within_each_data_available_block(one_week_frames):
    ldaps_small, gfs_small = one_week_frames
    out = _assemble_feature_table(ldaps_small, gfs_small, "train", "kpx_group_3")

    # lead_hour is 1-indexed within each data_available_kst_dtm block, so its
    # max per block should never exceed 24 (one issuance covers 24 forecast
    # hours) -- a cross-block leak in lag_rolling_features' grouping would
    # show up as lead_hour counting past a block boundary.
    max_lead_per_block = out.groupby("data_available_kst_dtm")["lead_hour"].max()
    assert (max_lead_per_block <= 24).all()
