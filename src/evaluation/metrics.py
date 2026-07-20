"""Official competition metrics: 1-NMAE and FICR (정산금획득률).

Implements exactly what is documented in:
  - reports/domain_research/nmae_formula.md
  - reports/domain_research/ficr_formula.md
(both confirmed against the official DACON evaluation page; see CLAUDE.md section 4).

Formulas (per-group, per-hour h, with actual_h/pred_h in kWh and capacity_g the group's
1-hour-equivalent installed capacity in kWh from ``configs.paths.GROUP_CAPACITY_KWH``):

    Eligibility filter (applies to BOTH metrics):
        only hours where actual_h >= min_utilization * capacity_g (default min_utilization=0.10)
        — "실제 발전량이 설비용량의 10% 이상인 시간대".

    NMAE (group) = mean( |pred_h - actual_h| / capacity_g )   over eligible hours
    1-NMAE       = 1 - mean(NMAE over the 3 groups)

    nmae_h  = |pred_h - actual_h| / capacity_g
    rate_h  = 4  if nmae_h <= 0.06
              3  if 0.06 < nmae_h <= 0.08
              0  if nmae_h > 0.08
    FICR (group) = sum(rate_h * actual_h) / sum(4 * actual_h)   over eligible hours
    FICR         = mean(FICR over the 3 groups)

    score = 0.5 * (1-NMAE) + 0.5 * FICR

NOTE (assumption, not 100% confirmed by DACON — see ficr_formula.md "확인되지 않은 것"):
the "이론상 최대 정산금" (theoretical max settlement) denominator is assumed to be
4원 * actual generation for every eligible hour (i.e. as if every hour scored the top tier).
This is the most natural reading of the confirmed rate table but DACON has not spelled out
the denominator definition as an explicit formula.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from configs.paths import GROUP_CAPACITY_KWH

DEFAULT_GROUP_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
DEFAULT_KEY_COL = "forecast_kst_dtm"
DEFAULT_MIN_UTILIZATION = 0.10


def _eligible_mask(actual: pd.Series, capacity_kwh: float, min_utilization: float) -> pd.Series:
    """실제 발전량이 설비용량의 min_utilization 이상인 시간대만 True."""
    return actual >= (min_utilization * capacity_kwh)


def nmae_per_group(
    pred: pd.Series,
    actual: pd.Series,
    capacity_kwh: float,
    min_utilization: float = DEFAULT_MIN_UTILIZATION,
) -> float:
    """그룹별 NMAE = mean(|pred-actual|/capacity) over eligible (utilization >= 10%) rows.

    reports/domain_research/nmae_formula.md 확정 산식:
      "그룹별 NMAE = 평균( |예측 발전량 − 실제 발전량| / 그룹 설비용량 )"
    """
    pred = pd.Series(pred).reset_index(drop=True).astype(float)
    actual = pd.Series(actual).reset_index(drop=True).astype(float)
    if len(pred) != len(actual):
        raise ValueError(f"pred and actual must be the same length, got {len(pred)} vs {len(actual)}")

    mask = _eligible_mask(actual, capacity_kwh, min_utilization)
    if not mask.any():
        return float("nan")

    err = (pred[mask] - actual[mask]).abs() / capacity_kwh
    return float(err.mean())


def one_minus_nmae(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    group_cols: list[str] = DEFAULT_GROUP_COLS,
    key_col: str = DEFAULT_KEY_COL,
    min_utilization: float = DEFAULT_MIN_UTILIZATION,
) -> dict:
    """Aligns pred_df/actual_df on ``key_col`` and computes 1-NMAE across the 3 groups.

    Returns a dict: {group: group_NMAE, ..., "1-NMAE": 1 - mean(group NMAEs)}.
    """
    merged = pred_df.merge(actual_df, on=key_col, how="inner", suffixes=("_pred", "_actual"))

    result: dict = {}
    for col in group_cols:
        capacity = GROUP_CAPACITY_KWH[col]
        result[col] = nmae_per_group(
            merged[f"{col}_pred"],
            merged[f"{col}_actual"],
            capacity,
            min_utilization=min_utilization,
        )

    result["1-NMAE"] = 1 - float(np.mean([result[col] for col in group_cols]))
    return result


def ficr_per_group(
    pred: pd.Series,
    actual: pd.Series,
    capacity_kwh: float,
    min_utilization: float = DEFAULT_MIN_UTILIZATION,
) -> float:
    """그룹별 FICR = 획득 정산금 / 이론상 최대 정산금, over eligible (utilization >= 10%) rows.

    Confirmed tier table (reports/domain_research/ficr_formula.md, "✅ 확정" section):
      nmae_h <= 0.06            -> rate_h = 4 원/kWh
      0.06 < nmae_h <= 0.08     -> rate_h = 3 원/kWh
      nmae_h > 0.08             -> rate_h = 0 원/kWh
      FICR = sum(rate_h * actual_h) / sum(4 * actual_h)
    """
    pred = pd.Series(pred).reset_index(drop=True).astype(float)
    actual = pd.Series(actual).reset_index(drop=True).astype(float)
    if len(pred) != len(actual):
        raise ValueError(f"pred and actual must be the same length, got {len(pred)} vs {len(actual)}")

    mask = _eligible_mask(actual, capacity_kwh, min_utilization)
    if not mask.any():
        return float("nan")

    p = pred[mask]
    a = actual[mask]
    nmae_h = (p - a).abs() / capacity_kwh

    rate_h = np.where(nmae_h <= 0.06, 4.0, np.where(nmae_h <= 0.08, 3.0, 0.0))
    earned = float(np.sum(rate_h * a.to_numpy()))
    max_possible = float(np.sum(4.0 * a.to_numpy()))

    if max_possible == 0:
        # Edge case: eligible hours exist but their actual generation sums to 0
        # (only possible if min_utilization == 0). Avoid dividing by zero.
        return float("nan")

    return earned / max_possible


def ficr(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    group_cols: list[str] = DEFAULT_GROUP_COLS,
    key_col: str = DEFAULT_KEY_COL,
    min_utilization: float = DEFAULT_MIN_UTILIZATION,
) -> dict:
    """Aligns pred_df/actual_df on ``key_col`` and computes FICR across the 3 groups.

    Returns a dict: {group: group_FICR, ..., "FICR": mean(group FICRs)}.
    """
    merged = pred_df.merge(actual_df, on=key_col, how="inner", suffixes=("_pred", "_actual"))

    result: dict = {}
    for col in group_cols:
        capacity = GROUP_CAPACITY_KWH[col]
        result[col] = ficr_per_group(
            merged[f"{col}_pred"],
            merged[f"{col}_actual"],
            capacity,
            min_utilization=min_utilization,
        )

    result["FICR"] = float(np.mean([result[col] for col in group_cols]))
    return result


def competition_score(
    pred_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    group_cols: list[str] = DEFAULT_GROUP_COLS,
    key_col: str = DEFAULT_KEY_COL,
    min_utilization: float = DEFAULT_MIN_UTILIZATION,
) -> dict:
    """Combines 1-NMAE and FICR into the official competition score.

    score = 0.5 * (1-NMAE) + 0.5 * FICR

    Returns a dict with all per-group NMAE/FICR values (namespaced as
    ``"<group>_nmae"`` / ``"<group>_ficr"`` since both metrics reuse the group names),
    plus "1-NMAE", "FICR", and "score".
    """
    nmae_result = one_minus_nmae(
        pred_df, actual_df, group_cols=group_cols, key_col=key_col, min_utilization=min_utilization
    )
    ficr_result = ficr(
        pred_df, actual_df, group_cols=group_cols, key_col=key_col, min_utilization=min_utilization
    )

    out: dict = {}
    for col in group_cols:
        out[f"{col}_nmae"] = nmae_result[col]
        out[f"{col}_ficr"] = ficr_result[col]

    out["1-NMAE"] = nmae_result["1-NMAE"]
    out["FICR"] = ficr_result["FICR"]
    out["score"] = 0.5 * out["1-NMAE"] + 0.5 * out["FICR"]
    return out
