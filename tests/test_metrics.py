"""Unit tests for src/evaluation/metrics.py (1-NMAE, FICR, competition score).

Formulas under test are the ones confirmed in:
  - reports/domain_research/nmae_formula.md
  - reports/domain_research/ficr_formula.md

All expected values below are computed by hand in comments next to the synthetic data,
not derived from the implementation itself.
"""
import math

import numpy as np
import pandas as pd
import pytest

from configs.paths import GROUP_CAPACITY_KWH
from src.evaluation.metrics import (
    competition_score,
    ficr,
    ficr_per_group,
    nmae_per_group,
    one_minus_nmae,
)

CAP1 = GROUP_CAPACITY_KWH["kpx_group_1"]  # 21,600 kWh
CAP2 = GROUP_CAPACITY_KWH["kpx_group_2"]  # 21,600 kWh
CAP3 = GROUP_CAPACITY_KWH["kpx_group_3"]  # 21,000 kWh


def _boundary_rows(capacity: float, base_actual: float = 10_000.0):
    """6 rows whose |pred-actual|/capacity hits exactly 0.00, 0.05, 0.06, 0.07, 0.08, 0.09.

    actual is held constant at base_actual (well above the 10% utilization cutoff for any
    of the 3 group capacities, since 10% of 21,600/21,000 is 2,160/2,100) so every row is
    eligible under the utilization filter.
    """
    targets = [0.00, 0.05, 0.06, 0.07, 0.08, 0.09]
    actual = [base_actual] * 6
    pred = [base_actual + t * capacity for t in targets]
    return pd.Series(pred), pd.Series(actual), targets


# ---------------------------------------------------------------------------
# nmae_per_group
# ---------------------------------------------------------------------------


def test_nmae_per_group_boundaries():
    pred, actual, targets = _boundary_rows(CAP1)
    # NMAE = mean(|pred-actual|/capacity) = mean(targets)
    #      = (0.00 + 0.05 + 0.06 + 0.07 + 0.08 + 0.09) / 6 = 0.35 / 6 = 0.058333...
    expected = sum(targets) / 6
    result = nmae_per_group(pred, actual, CAP1)
    assert result == pytest.approx(expected, abs=1e-9)
    assert result == pytest.approx(0.05833333333, abs=1e-9)


def test_nmae_per_group_simple_two_rows():
    # pred-actual errors of exactly 1,080 kWh and 2,160 kWh on a 21,600 kWh capacity group
    # -> nmae per row = 0.05 and 0.10 -> mean = 0.075
    pred = pd.Series([11_080.0, 12_160.0])
    actual = pd.Series([10_000.0, 10_000.0])
    result = nmae_per_group(pred, actual, CAP1)
    assert result == pytest.approx(0.075, abs=1e-9)


# ---------------------------------------------------------------------------
# ficr_per_group
# ---------------------------------------------------------------------------


def test_ficr_per_group_boundaries():
    pred, actual, targets = _boundary_rows(CAP1)
    # rate per row (nmae <= .06 -> 4, .06 < nmae <= .08 -> 3, nmae > .08 -> 0):
    #   0.00 -> 4   0.05 -> 4   0.06 -> 4 (inclusive)  0.07 -> 3   0.08 -> 3 (inclusive)  0.09 -> 0
    # actual is 10,000 for every row.
    # earned       = 10,000 * (4+4+4+3+3+0) = 10,000 * 18 = 180,000
    # max_possible = 10,000 * 4 * 6         = 240,000
    # FICR = 180,000 / 240,000 = 0.75
    result = ficr_per_group(pred, actual, CAP1)
    assert result == pytest.approx(0.75, abs=1e-9)


def test_ficr_per_group_all_top_tier():
    # All rows exactly on target (nmae=0) -> rate=4 always -> FICR = 1.0
    actual = pd.Series([5_000.0, 8_000.0, 12_000.0])
    pred = actual.copy()
    result = ficr_per_group(pred, actual, CAP1)
    assert result == pytest.approx(1.0, abs=1e-9)


def test_ficr_per_group_all_zero_tier():
    # nmae = 0.20 for every row (> 0.08) -> rate=0 always -> FICR = 0.0
    actual = pd.Series([10_000.0, 10_000.0])
    pred = actual + 0.20 * CAP1
    result = ficr_per_group(pred, actual, CAP1)
    assert result == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 10%-utilization filter
# ---------------------------------------------------------------------------


def test_utilization_filter_excludes_low_utilization_hours():
    # Row 0: actual = 2,000 kWh, which is below 10% of 21,600 (=2,160) -> must be excluded,
    #         even though its error is huge (would otherwise dominate the mean).
    # Row 1: actual = 10,000 kWh (eligible), pred off by exactly 5% of capacity (1,080 kWh).
    actual = pd.Series([2_000.0, 10_000.0])
    pred = pd.Series([2_000.0 + 0.99 * CAP1, 10_000.0 + 0.05 * CAP1])

    nmae_result = nmae_per_group(pred, actual, CAP1)
    # If the low-utilization row were (wrongly) included, nmae would be huge (~mean of 0.99, 0.05).
    # With the filter applied, only row 1 counts -> nmae == 0.05 exactly.
    assert nmae_result == pytest.approx(0.05, abs=1e-9)

    ficr_result = ficr_per_group(pred, actual, CAP1)
    # Only row 1 (nmae=0.05 <= 0.06 -> rate 4) counts -> FICR = 1.0, not dragged down by row 0.
    assert ficr_result == pytest.approx(1.0, abs=1e-9)


def test_utilization_filter_boundary_is_inclusive_of_threshold():
    # actual exactly at 10% of capacity (2,160.0) must be INCLUDED ("설비용량의 10% 이상").
    threshold_actual = 0.10 * CAP1  # 2,160.0
    actual = pd.Series([threshold_actual])
    pred = pd.Series([threshold_actual + 0.05 * CAP1])
    result = nmae_per_group(pred, actual, CAP1)
    assert result == pytest.approx(0.05, abs=1e-9)


# ---------------------------------------------------------------------------
# Divide-by-zero edge case
# ---------------------------------------------------------------------------


def test_nmae_per_group_no_eligible_rows_returns_nan():
    actual = pd.Series([100.0, 200.0])  # both well below 10% of 21,600 (=2,160)
    pred = pd.Series([150.0, 250.0])
    result = nmae_per_group(pred, actual, CAP1)
    assert math.isnan(result)


def test_ficr_per_group_no_eligible_rows_returns_nan():
    actual = pd.Series([100.0, 200.0])
    pred = pd.Series([150.0, 250.0])
    result = ficr_per_group(pred, actual, CAP1)
    assert math.isnan(result)


def test_ficr_per_group_zero_actual_with_zero_min_utilization_returns_nan():
    # With min_utilization=0, actual=0 rows become "eligible" but sum(actual)=0
    # -> max_possible=0 -> must return NaN instead of raising / dividing by zero.
    actual = pd.Series([0.0, 0.0])
    pred = pd.Series([0.0, 0.0])
    result = ficr_per_group(pred, actual, CAP1, min_utilization=0.0)
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# Multi-group aggregate functions (one_minus_nmae / ficr / competition_score)
# ---------------------------------------------------------------------------


def _three_group_frames():
    """3 groups, each with the same 6-row boundary pattern (scaled to each group's capacity),
    sharing a common forecast_kst_dtm key.
    """
    dtms = pd.date_range("2025-01-01 01:00:00", periods=6, freq="h")

    pred1, actual1, _ = _boundary_rows(CAP1)
    pred2, actual2, _ = _boundary_rows(CAP2)
    pred3, actual3, _ = _boundary_rows(CAP3)

    pred_df = pd.DataFrame(
        {
            "forecast_kst_dtm": dtms,
            "kpx_group_1": pred1,
            "kpx_group_2": pred2,
            "kpx_group_3": pred3,
        }
    )
    actual_df = pd.DataFrame(
        {
            "forecast_kst_dtm": dtms,
            "kpx_group_1": actual1,
            "kpx_group_2": actual2,
            "kpx_group_3": actual3,
        }
    )
    return pred_df, actual_df


def test_one_minus_nmae_three_groups():
    pred_df, actual_df = _three_group_frames()
    result = one_minus_nmae(pred_df, actual_df)

    # Each group individually has NMAE = 0.35/6 = 0.058333... (see test_nmae_per_group_boundaries)
    per_group_expected = sum([0.00, 0.05, 0.06, 0.07, 0.08, 0.09]) / 6
    for col in ["kpx_group_1", "kpx_group_2", "kpx_group_3"]:
        assert result[col] == pytest.approx(per_group_expected, abs=1e-9)

    # 1-NMAE = 1 - mean(three identical group NMAEs) = 1 - 0.058333... = 0.941666...
    assert result["1-NMAE"] == pytest.approx(1 - per_group_expected, abs=1e-9)


def test_ficr_three_groups():
    pred_df, actual_df = _three_group_frames()
    result = ficr(pred_df, actual_df)

    # Each group individually has FICR = 0.75 (see test_ficr_per_group_boundaries)
    for col in ["kpx_group_1", "kpx_group_2", "kpx_group_3"]:
        assert result[col] == pytest.approx(0.75, abs=1e-9)
    assert result["FICR"] == pytest.approx(0.75, abs=1e-9)


def test_competition_score_three_groups():
    pred_df, actual_df = _three_group_frames()
    result = competition_score(pred_df, actual_df)

    per_group_nmae_expected = sum([0.00, 0.05, 0.06, 0.07, 0.08, 0.09]) / 6
    expected_one_minus_nmae = 1 - per_group_nmae_expected
    expected_ficr = 0.75
    # score = 0.5*(1-NMAE) + 0.5*FICR
    #       = 0.5*0.9416666... + 0.5*0.75 = 0.4708333... + 0.375 = 0.8458333...
    expected_score = 0.5 * expected_one_minus_nmae + 0.5 * expected_ficr

    assert result["1-NMAE"] == pytest.approx(expected_one_minus_nmae, abs=1e-9)
    assert result["FICR"] == pytest.approx(expected_ficr, abs=1e-9)
    assert result["score"] == pytest.approx(expected_score, abs=1e-9)

    # per-group breakdown present and namespaced (not overwritten by each other)
    for col in ["kpx_group_1", "kpx_group_2", "kpx_group_3"]:
        assert result[f"{col}_nmae"] == pytest.approx(per_group_nmae_expected, abs=1e-9)
        assert result[f"{col}_ficr"] == pytest.approx(0.75, abs=1e-9)


def test_one_minus_nmae_accepts_renamed_actual_key_col():
    # Caller pattern from the spec: actual_df uses "kst_dtm" instead of "forecast_kst_dtm";
    # caller is expected to rename before calling, not the function guessing column names.
    pred_df, actual_df = _three_group_frames()
    actual_df_renamed = actual_df.rename(columns={"forecast_kst_dtm": "kst_dtm"}).rename(
        columns={"kst_dtm": "forecast_kst_dtm"}
    )
    result = one_minus_nmae(pred_df, actual_df_renamed)
    per_group_expected = sum([0.00, 0.05, 0.06, 0.07, 0.08, 0.09]) / 6
    assert result["1-NMAE"] == pytest.approx(1 - per_group_expected, abs=1e-9)


def test_one_minus_nmae_custom_key_col():
    pred_df, actual_df = _three_group_frames()
    pred_df = pred_df.rename(columns={"forecast_kst_dtm": "my_dtm"})
    actual_df = actual_df.rename(columns={"forecast_kst_dtm": "my_dtm"})
    result = one_minus_nmae(pred_df, actual_df, key_col="my_dtm")
    per_group_expected = sum([0.00, 0.05, 0.06, 0.07, 0.08, 0.09]) / 6
    assert result["1-NMAE"] == pytest.approx(1 - per_group_expected, abs=1e-9)


def test_one_minus_nmae_misaligned_rows_are_joined_on_key():
    # actual_df has an extra row (a timestamp not present in pred_df) and rows in a
    # different order; alignment must happen via the merge on forecast_kst_dtm, not
    # positional order.
    pred_df, actual_df = _three_group_frames()
    pred_shuffled = pred_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    extra_row = actual_df.iloc[[0]].copy()
    extra_row["forecast_kst_dtm"] = pd.Timestamp("2099-01-01 00:00:00")
    actual_with_extra = pd.concat([actual_df, extra_row], ignore_index=True)

    result = one_minus_nmae(pred_shuffled, actual_with_extra)
    per_group_expected = sum([0.00, 0.05, 0.06, 0.07, 0.08, 0.09]) / 6
    assert result["1-NMAE"] == pytest.approx(1 - per_group_expected, abs=1e-9)
