"""Tests for the ECMWF IFS third-source integration:

(a) the forecast_kst_dtm join never changes the feature-table row count,
(b) the ``previous_day2`` offset is leakage-safe by explicit time arithmetic
    (and ``previous_day1`` demonstrably is NOT -- documenting why 2 was
    chosen), against the D-1 14:00 KST information cutoff,
(c) rows before the ~2024-04 archive start stay NaN (never zero-filled).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from configs.paths import DATA_PROCESSED_DIR
from src.data.fetch_ecmwf import (
    DISSEMINATION_MAX_DELAY,
    OFFSET_DAYS,
    cutoff_kst_for,
    implied_init_utc,
    is_leakage_safe,
)
from src.features.build_features import ECMWF_FEATURE_COLS, _add_ecmwf_features


def _synthetic_merged(n_hours: int, start: str) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n_hours, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "forecast_kst_dtm": ts,
            "data_available_kst_dtm": ts.normalize() - pd.Timedelta(hours=11),
            "ldaps_10m_speed_idw": rng.uniform(0, 15, n_hours),
            "ldaps_ws_hub_fixed": rng.uniform(0, 20, n_hours),
            "gfs_100m_speed_idw": rng.uniform(0, 20, n_hours),
        }
    )


def _synthetic_ecmwf(start: str, n_hours: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n_hours, freq="h")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "forecast_kst_dtm": ts,
            "ecmwf_wind_speed_100m": rng.uniform(0, 25, n_hours),
            "ecmwf_wind_direction_100m": rng.uniform(0, 360, n_hours),
            "ecmwf_wind_speed_10m": rng.uniform(0, 15, n_hours),
            "ecmwf_temperature_2m": rng.uniform(-10, 30, n_hours),
            "ecmwf_surface_pressure": rng.uniform(880, 920, n_hours),
        }
    )


# ---------------------------------------------------------------- (a) join --
def test_join_preserves_row_count_and_adds_all_columns():
    merged = _synthetic_merged(72, "2024-06-01 01:00")
    ecmwf = _synthetic_ecmwf("2024-06-01 01:00", 72)
    out = _add_ecmwf_features(merged.copy(), ecmwf)
    assert len(out) == len(merged)
    for col in ECMWF_FEATURE_COLS:
        assert col in out.columns
    assert out["ecmwf_ws100"].notna().all()
    # disagreement features consistent with their definition
    np.testing.assert_allclose(
        out["ecmwf_minus_gfs_ws100"], out["ecmwf_ws100"] - merged["gfs_100m_speed_idw"]
    )


def test_join_with_partial_coverage_keeps_uncovered_rows_nan():
    # merged spans 2024-03-30 .. 2024-04-02; ECMWF only covers from 04-01.
    merged = _synthetic_merged(96, "2024-03-30 01:00")
    ecmwf = _synthetic_ecmwf("2024-04-01 00:00", 96)
    out = _add_ecmwf_features(merged.copy(), ecmwf)
    assert len(out) == len(merged)
    pre = out["forecast_kst_dtm"] < pd.Timestamp("2024-04-01")
    assert pre.sum() > 0
    for col in ECMWF_FEATURE_COLS:
        assert out.loc[pre, col].isna().all(), f"{col} must stay NaN before coverage"
    assert out.loc[~pre, "ecmwf_ws100"].notna().all()


def test_missing_backfill_gives_all_nan_columns_same_rows():
    merged = _synthetic_merged(24, "2022-05-01 01:00")
    out = _add_ecmwf_features(merged.copy(), None)
    assert len(out) == len(merged)
    for col in ECMWF_FEATURE_COLS:
        assert out[col].isna().all()


def test_duplicate_ecmwf_timestamps_raise():
    merged = _synthetic_merged(24, "2024-06-01 01:00")
    ecmwf = _synthetic_ecmwf("2024-06-01 01:00", 24)
    dup = pd.concat([ecmwf, ecmwf.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="row count"):
        _add_ecmwf_features(merged.copy(), dup)


# ---------------------------------------------------- (b) leakage arithmetic --
def _target_day_hours(day: datetime) -> list[datetime]:
    """The 24 forecast hours of one KST target day D: D 01:00 .. D+1 00:00."""
    return [day + timedelta(hours=h) for h in range(1, 25)]


def test_cutoff_is_14kst_on_previous_day():
    # forecast 2025-06-15 01:00 .. 2025-06-16 00:00 all share cutoff 06-14 14:00
    for t in _target_day_hours(datetime(2025, 6, 15)):
        assert cutoff_kst_for(t) == datetime(2025, 6, 14, 14, 0)


def test_offset2_is_leakage_safe_for_every_hour_of_a_target_day():
    for day in (datetime(2024, 5, 10), datetime(2025, 1, 1), datetime(2025, 12, 31)):
        for t in _target_day_hours(day):
            assert is_leakage_safe(t, offset_days=2), f"previous_day2 must be safe at {t}"


def test_offset1_leaks_for_late_hours():
    # previous_day1 resolves to D-1 00z/06z/12z runs for part of the day --
    # all disseminated after the D-1 14:00 KST cutoff. This is why
    # OFFSET_DAYS == 2 and must never be lowered.
    leaks = [t for t in _target_day_hours(datetime(2025, 6, 15)) if not is_leakage_safe(t, offset_days=1)]
    assert len(leaks) > 0


def test_configured_offset_is_2():
    assert OFFSET_DAYS == 2


def test_worst_case_init_is_d2_12z():
    # Latest forecast hour of target day D (D+1 00:00 KST = D 15:00 UTC):
    # implied init must be exactly D-2 12z, whose dissemination ends
    # 04:34 KST D-1 -- >9h before the 14:00 cutoff.
    t = datetime(2025, 6, 16, 0, 0)  # target day D = 2025-06-15, KST 24:00
    init = implied_init_utc(t - timedelta(hours=9), offset_days=2)
    assert init == datetime(2025, 6, 13, 12, 0)  # D-2 12z UTC
    available_kst = init + DISSEMINATION_MAX_DELAY + timedelta(hours=9)
    assert available_kst == datetime(2025, 6, 14, 4, 34)
    assert available_kst < cutoff_kst_for(t)


# --------------------------------------------------- (c) built parquet NaN --
_TRAIN_PARQUET = DATA_PROCESSED_DIR / "features_kpx_group_1_train.parquet"


@pytest.mark.skipif(not _TRAIN_PARQUET.exists(), reason="processed feature tables not built on this machine")
def test_train_parquet_nan_before_archive_start():
    df = pd.read_parquet(_TRAIN_PARQUET, columns=["forecast_kst_dtm", "ecmwf_ws100", "ecmwf_ldaps_ws_ratio"])
    pre = df["forecast_kst_dtm"] < pd.Timestamp("2024-04-01")
    post = df["forecast_kst_dtm"] >= pd.Timestamp("2024-06-01")
    assert pre.sum() > 0 and post.sum() > 0
    assert df.loc[pre, "ecmwf_ws100"].isna().all()
    assert df.loc[pre, "ecmwf_ldaps_ws_ratio"].isna().all()
    assert df.loc[post, "ecmwf_ws100"].notna().mean() > 0.99
