"""Unit tests for src/features/calendar_features.py.

Focused on:
  1. add_calendar_features's sin/cos encoding matching the theoretical
     2*pi*value/period formula at boundary values (hour=0/23, month=1/12).
  2. add_lead_hours's core leakage-safety invariant: lead_hours > 0 for
     every row, given this project's announcement-before-forecast-hour data
     shape (CLAUDE.md section 3) -- plus that it never touches a pre-existing
     `lead_hour` (singular) column.
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.features.calendar_features import add_calendar_features, add_lead_hours


# ---------------------------------------------------------------------------
# add_calendar_features
# ---------------------------------------------------------------------------


def test_add_calendar_features_hour_boundary_values():
    df = pd.DataFrame(
        {
            "forecast_kst_dtm": [
                pd.Timestamp("2024-06-15 00:00:00"),
                pd.Timestamp("2024-06-15 23:00:00"),
            ]
        }
    )
    out = add_calendar_features(df)

    assert out["hour"].tolist() == [0, 23]
    assert out["hour_sin"].iloc[0] == pytest.approx(math.sin(2 * math.pi * 0 / 24))
    assert out["hour_cos"].iloc[0] == pytest.approx(math.cos(2 * math.pi * 0 / 24))
    assert out["hour_sin"].iloc[1] == pytest.approx(math.sin(2 * math.pi * 23 / 24))
    assert out["hour_cos"].iloc[1] == pytest.approx(math.cos(2 * math.pi * 23 / 24))
    # hour=0 is the "reference angle": sin=0, cos=1 exactly.
    assert out["hour_sin"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert out["hour_cos"].iloc[0] == pytest.approx(1.0, abs=1e-12)


def test_add_calendar_features_month_boundary_values():
    df = pd.DataFrame(
        {
            "forecast_kst_dtm": [
                pd.Timestamp("2024-01-10 05:00:00"),
                pd.Timestamp("2024-12-10 05:00:00"),
            ]
        }
    )
    out = add_calendar_features(df)

    assert out["month"].tolist() == [1, 12]
    assert out["month_sin"].iloc[0] == pytest.approx(math.sin(2 * math.pi * 1 / 12))
    assert out["month_cos"].iloc[0] == pytest.approx(math.cos(2 * math.pi * 1 / 12))
    assert out["month_sin"].iloc[1] == pytest.approx(math.sin(2 * math.pi * 12 / 12))
    assert out["month_cos"].iloc[1] == pytest.approx(math.cos(2 * math.pi * 12 / 12))


def test_add_calendar_features_dayofweek_and_dayofyear_passthrough():
    ts = pd.Timestamp("2024-03-05 10:00:00")  # a Tuesday
    df = pd.DataFrame({"forecast_kst_dtm": [ts]})
    out = add_calendar_features(df)

    assert out["dayofweek"].iloc[0] == ts.dayofweek
    assert out["dayofyear"].iloc[0] == ts.dayofyear
    assert out["dayofyear_sin"].iloc[0] == pytest.approx(
        math.sin(2 * math.pi * ts.dayofyear / 365)
    )


def test_add_calendar_features_returns_copy_not_mutating_input():
    df = pd.DataFrame({"forecast_kst_dtm": [pd.Timestamp("2024-01-01 00:00:00")]})
    original_cols = list(df.columns)
    _ = add_calendar_features(df)
    assert list(df.columns) == original_cols


def test_add_calendar_features_adds_expected_column_set():
    df = pd.DataFrame({"forecast_kst_dtm": [pd.Timestamp("2024-01-01 00:00:00")]})
    out = add_calendar_features(df)
    expected_new = {
        "month", "hour", "dayofweek", "dayofyear",
        "month_sin", "month_cos", "hour_sin", "hour_cos",
        "dayofweek_sin", "dayofweek_cos", "dayofyear_sin", "dayofyear_cos",
    }
    assert expected_new.issubset(set(out.columns))


# ---------------------------------------------------------------------------
# add_lead_hours
# ---------------------------------------------------------------------------


def test_add_lead_hours_exact_value():
    df = pd.DataFrame(
        {
            "forecast_kst_dtm": [pd.Timestamp("2024-01-02 14:00:00")],
            "data_available_kst_dtm": [pd.Timestamp("2024-01-01 13:00:00")],
        }
    )
    out = add_lead_hours(df)
    assert out["lead_hours"].iloc[0] == pytest.approx(25.0)


def test_add_lead_hours_positive_invariant_on_realistic_block_structure():
    # Mimics a real data_available_kst_dtm block: one issuance (13:00) covers
    # 24 forecast hours starting the next hour -- CLAUDE.md section 3's
    # "13:00 issuance -> usable from then on" structure.
    available = pd.Timestamp("2024-03-01 13:00:00")
    forecast_hours = pd.date_range(available + pd.Timedelta(hours=1), periods=24, freq="h")
    df = pd.DataFrame(
        {
            "forecast_kst_dtm": forecast_hours,
            "data_available_kst_dtm": [available] * 24,
        }
    )
    out = add_lead_hours(df)
    assert (out["lead_hours"] > 0).all()
    assert out["lead_hours"].min() == pytest.approx(1.0)
    assert out["lead_hours"].max() == pytest.approx(24.0)


def test_add_lead_hours_does_not_touch_existing_singular_lead_hour_column():
    df = pd.DataFrame(
        {
            "forecast_kst_dtm": [pd.Timestamp("2024-01-01 14:00:00")],
            "data_available_kst_dtm": [pd.Timestamp("2024-01-01 13:00:00")],
            "lead_hour": [1],
        }
    )
    out = add_lead_hours(df)
    assert "lead_hour" in out.columns
    assert out["lead_hour"].iloc[0] == 1
    assert "lead_hours" in out.columns
    assert out["lead_hours"].iloc[0] == pytest.approx(1.0)
