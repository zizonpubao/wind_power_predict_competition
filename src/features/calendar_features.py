"""Calendar / seasonality feature functions -- pure DataFrame transforms.

Both functions return a **copy** of the input frame with new columns added
(unlike some of weather_features.py's mutate-and-return helpers) -- they are
meant to be called once, late in ``build_features._assemble_feature_table``,
on the fully-merged frame, so there is no repeated-mutation risk to guard
against and a copy keeps the call sites simple (``df = add_calendar_features(df)``).

Naming-collision warning
------------------------
``add_lead_hours``'s ``lead_hours`` (plural, continuous hour-fraction) is a
**different column** from ``weather_features.lag_rolling_features``'s
``lead_hour`` (singular, 1-indexed integer position within a
``data_available_kst_dtm`` block, e.g. 1..24). Do not rename, alias, or
otherwise conflate the two -- this module only ever adds the plural column
and never touches the existing singular one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Period (in the same units as the corresponding calendar field) used for
#: the sin/cos encoding of each field. dayofyear's period is always 365
#: (never 366) -- a deliberate simplification per the task spec: the 366th
#: day of a leap year reuses the 365-period formula rather than getting a
#: separate leap-year-aware period, which would make the encoding
#: discontinuous across years anyway.
_PERIODS = {"month": 12, "hour": 24, "dayofweek": 7, "dayofyear": 365}


def add_calendar_features(df: pd.DataFrame, dt_col: str = "forecast_kst_dtm") -> pd.DataFrame:
    """Add month/hour/dayofweek/dayofyear (ints, from pandas' own datetime
    accessors -- dayofweek is Monday=0) plus ``{col}_sin``/``{col}_cos``
    (``sin(2*pi*value/period)`` / ``cos(2*pi*value/period)``) for each of the
    4 fields -- 12 new columns total.

    Parameters
    ----------
    df: any dataframe with a `dt_col` datetime column.
    dt_col: name of the datetime column to derive calendar fields from
        (default "forecast_kst_dtm" -- the column build_features.py's
        _assemble_feature_table calls this with).

    Returns
    -------
    A copy of `df` with the 12 new columns appended.
    """
    out = df.copy()
    dt = pd.to_datetime(out[dt_col])

    fields = {
        "month": dt.dt.month,
        "hour": dt.dt.hour,
        "dayofweek": dt.dt.dayofweek,
        "dayofyear": dt.dt.dayofyear,
    }
    for name, series in fields.items():
        out[name] = series
        period = _PERIODS[name]
        angle = 2.0 * np.pi * series.to_numpy(dtype=float) / period
        out[f"{name}_sin"] = np.sin(angle)
        out[f"{name}_cos"] = np.cos(angle)

    return out


def add_lead_hours(
    df: pd.DataFrame,
    forecast_col: str = "forecast_kst_dtm",
    available_col: str = "data_available_kst_dtm",
) -> pd.DataFrame:
    """Add ``lead_hours`` = ``(df[forecast_col] - df[available_col]).total_seconds() / 3600``
    -- the continuous hour-gap between a forecast's issuance time and the
    hour it predicts ("forecast freshness"), distinct from the existing
    1-indexed ``lead_hour`` (see module docstring).

    Given this project's data (an announcement is always issued strictly
    before every forecast hour it covers -- CLAUDE.md section 3), every row
    is expected to have ``lead_hours > 0``; this function does not itself
    assert that (it is a pure transform of whatever timestamps it is given),
    but callers/tests should treat a non-positive value as a leakage red flag.

    Returns
    -------
    A copy of `df` with the 1 new column appended.
    """
    out = df.copy()
    forecast_dt = pd.to_datetime(out[forecast_col])
    available_dt = pd.to_datetime(out[available_col])
    out["lead_hours"] = (forecast_dt - available_dt).dt.total_seconds() / 3600.0
    return out
