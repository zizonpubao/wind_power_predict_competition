"""Raw-file loaders for the BARAM 2026 wind power forecasting dataset.

All paths are resolved through ``configs.paths`` — never hardcode a path here.
Source CSVs live under the read-only ``C:\\Users\\aica_\\Desktop\\open (1)`` tree
(see CLAUDE.md); this module only reads them, it never writes there.

All CSVs are ``utf-8-sig`` and all timestamp columns are already KST — we parse
them as naive ``datetime64[ns]`` and never apply a timezone conversion (KST
throughout, per CLAUDE.md coding conventions).

SCADA-to-hourly aggregation convention (verified against real data, 2026-07-20)
---------------------------------------------------------------------------
``data_description.md`` states ``train_labels.csv``'s ``kst_dtm`` is "집계 구간의
종료 시각" (the *end* of the aggregation interval). SCADA rows are 10-minute
readings timestamped at the *start* of each sub-interval (``HH:00, HH:10, ...,
HH:50``). So the 6 SCADA readings timestamped in ``[HH:00, HH:50]`` belong to
the hour that *ends* at ``HH+1:00`` — i.e. the aggregated row's ``kst_dtm``
must be ``floor(kst_dtm) + 1 hour``, not the floor itself. This was confirmed
empirically: correlating VESTAS group-1 SCADA against ``train_labels`` gives
r=0.954 when paired with the same-hour label (``HH:00``) vs. r=0.998 when
paired with the next-hour label (``HH+1:00``) — the next-hour pairing is
unambiguously correct.

A second, more surprising finding while doing that same check: CLAUDE.md's
domain note assumes ``*_power_kw10m`` is a 10-minute *instantaneous average
power* (kW), to be converted to hourly kWh by **averaging** the 6 readings
(numerically: mean(kW) over the hour == kWh, since power x 1h == energy).
That does NOT match the data — averaging the 6 raw values under-shoots
``train_labels`` by a consistent ~6x (e.g. group-1 hourly mean of the 6 raw
values ~2,100-3,200 vs. an actual label of ~12,000-19,000 for the same hour;
6x is exactly the 10-min-to-1-hour reading count). **Summing** the 6 raw
values instead lines up with the label almost exactly (ratios ~0.94-1.10,
r=0.998), which implies each ``power_kw10m`` reading is already energy-like
(kWh generated in that 10-minute bucket) despite the "kW" name, not a power
rate that needs dividing by 6. `aggregate_scada_to_hourly` therefore **sums**
the 6 within-hour readings rather than averaging them.

This contradicts the literal CLAUDE.md phrasing ("6개 10분 값 평균 → 1시간
kWh") and should be sanity-checked by the team (e.g. via `eda-analyst`) before
being relied on for label validation — flagging prominently rather than
silently overriding stated domain knowledge.
"""
from __future__ import annotations

import pandas as pd

from configs.paths import (
    GFS_TEST_CSV,
    GFS_TRAIN_CSV,
    LDAPS_TEST_CSV,
    LDAPS_TRAIN_CSV,
    SAMPLE_SUBMISSION_CSV,
    SCADA_UNISON_TRAIN_CSV,
    SCADA_VESTAS_TRAIN_CSV,
    TRAIN_LABELS_CSV,
    TURBINE_GROUP_MAP,
)

_FORECAST_DTM_COLS = ["forecast_kst_dtm", "data_available_kst_dtm"]


def _read_csv(path, parse_dates=None) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=parse_dates)


def load_ldaps(split: str) -> pd.DataFrame:
    """Load the LDAPS weather forecast file (long format: 16 grids per forecast hour).

    Parameters
    ----------
    split: "train" or "test"
    """
    if split == "train":
        path = LDAPS_TRAIN_CSV
    elif split == "test":
        path = LDAPS_TEST_CSV
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return _read_csv(path, parse_dates=_FORECAST_DTM_COLS)


def load_gfs(split: str) -> pd.DataFrame:
    """Load the GFS weather forecast file (long format: 9 grids per forecast hour).

    Parameters
    ----------
    split: "train" or "test"
    """
    if split == "train":
        path = GFS_TRAIN_CSV
    elif split == "test":
        path = GFS_TEST_CSV
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return _read_csv(path, parse_dates=_FORECAST_DTM_COLS)


def load_train_labels() -> pd.DataFrame:
    """Load train_labels.csv (hourly kWh per kpx_group, kst_dtm = interval end)."""
    return _read_csv(TRAIN_LABELS_CSV, parse_dates=["kst_dtm"])


def load_scada_vestas() -> pd.DataFrame:
    """Load the 10-minute VESTAS SCADA training data."""
    return _read_csv(SCADA_VESTAS_TRAIN_CSV, parse_dates=["kst_dtm"])


def load_scada_unison() -> pd.DataFrame:
    """Load the 10-minute UNISON SCADA training data."""
    return _read_csv(SCADA_UNISON_TRAIN_CSV, parse_dates=["kst_dtm"])


def load_sample_submission() -> pd.DataFrame:
    """Load sample_submission.csv (forecast_id, forecast_kst_dtm untouched on submit)."""
    return _read_csv(SAMPLE_SUBMISSION_CSV, parse_dates=["forecast_kst_dtm"])


def pivot_grid_wide(df: pd.DataFrame, value_cols: list[str] | None = None) -> pd.DataFrame:
    """Pivot a long-format LDAPS/GFS dataframe (one row per forecast_kst_dtm x grid_id)
    into wide format: one row per (forecast_kst_dtm, data_available_kst_dtm), with
    columns ``{orig_col}_grid{grid_id}`` for every weather variable x grid_id pair.

    If ``value_cols`` is None, every numeric column other than
    forecast_kst_dtm/data_available_kst_dtm/grid_id/latitude/longitude is pivoted.
    """
    index_cols = _FORECAST_DTM_COLS
    non_value_cols = set(index_cols) | {"grid_id", "latitude", "longitude"}
    if value_cols is None:
        value_cols = [
            c
            for c in df.columns
            if c not in non_value_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

    indexed = df.set_index(index_cols + ["grid_id"])[value_cols]
    wide = indexed.unstack("grid_id")
    wide.columns = [f"{col}_grid{grid_id}" for col, grid_id in wide.columns]
    wide = wide.sort_index(axis=1).sort_index(axis=0).reset_index()
    return wide


def aggregate_scada_to_hourly(df: pd.DataFrame, power_cols: list[str]) -> pd.DataFrame:
    """Aggregate a 10-minute SCADA dataframe to hourly, summing the 6 readings within
    each hour (see the module docstring for why this is sum, not average).

    Returns one row per hour with a ``kst_dtm`` column set to the hour-*end*
    timestamp, matching train_labels.csv's "집계 구간의 종료 시각" convention:
    readings at HH:00..HH:50 map to kst_dtm = HH+1:00.
    """
    hour_start = df["kst_dtm"].dt.floor("h")
    hourly = df.groupby(hour_start)[power_cols].sum(min_count=1)
    hourly = hourly.reset_index(names="kst_dtm")
    hourly["kst_dtm"] = hourly["kst_dtm"] + pd.Timedelta(hours=1)
    return hourly.sort_values("kst_dtm").reset_index(drop=True)


def group_scada_power(vestas_df: pd.DataFrame, unison_df: pd.DataFrame) -> pd.DataFrame:
    """Sum per-turbine SCADA power into hourly kWh per kpx_group_1/2/3, using
    `configs.paths.TURBINE_GROUP_MAP`. Reuses `aggregate_scada_to_hourly` for the
    time aggregation, then sums the resulting per-turbine hourly kWh across each
    group's turbines.
    """
    source_by_group = {
        "kpx_group_1": vestas_df,
        "kpx_group_2": vestas_df,
        "kpx_group_3": unison_df,
    }

    group_frames = []
    for group, turbines in TURBINE_GROUP_MAP.items():
        power_cols = [f"{t}_power_kw10m" for t in turbines]
        hourly = aggregate_scada_to_hourly(source_by_group[group], power_cols)
        hourly[group] = hourly[power_cols].sum(axis=1, min_count=1)
        group_frames.append(hourly[["kst_dtm", group]])

    merged = group_frames[0]
    for frame in group_frames[1:]:
        merged = merged.merge(frame, on="kst_dtm", how="outer")
    return merged.sort_values("kst_dtm").reset_index(drop=True)
