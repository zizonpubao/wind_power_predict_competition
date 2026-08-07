"""Backfill ICON (DWD global) forecasts from the Open-Meteo Previous Runs
API into ``data/interim/icon.parquet`` -- the fourth, independent NWP source
next to the competition-provided LDAPS/GFS and the already-integrated ECMWF
IFS (``src.data.fetch_ecmwf``). See ``reports/experiment_queue.md`` #10/#C.

Why the Previous Runs API and why ``_previous_day2`` (leakage argument)
------------------------------------------------------------------------
Same cutoff argument as ``fetch_ecmwf`` (see that module's docstring for the
full derivation): the competition's information cutoff for target day D is
D-1 14:00 KST. DWD's ICON global run is disseminated much *faster* than
ECMWF IFS -- publicly documented as roughly 2-3 hours after each 00/06/12/18
UTC run init (DWD open-data schedule). This module does **not** re-verify
that exact delay empirically this session (no independent DWD confluence
page was fetched), so to stay honest it reuses ECMWF's own measured
worst-case dissemination delay (7h34m, the *slower* of the two models) as a
deliberately conservative stand-in -- if ICON is actually faster (which
public documentation suggests), the true margin is only larger, never
smaller. Under that conservative delay, the exact same run-cycle arithmetic
as ``fetch_ecmwf`` applies (6-hourly runs, ``previous_day2`` = 48h older run
than the plain series), so ``OFFSET_DAYS = 2`` is safe by the identical
hour-by-hour argument -- re-derived in ``tests/test_icon.py`` by importing
and reusing ``fetch_ecmwf.implied_init_utc``/``cutoff_kst_for`` unchanged
(only the model name differs, the leakage arithmetic does not depend on it).

Coverage (probed directly against this exact coordinate, 2026-08-07)
-----------------------------------------------------------------------
``previous_day2`` data for ``icon_global`` at this coordinate is **not**
archived before ~2024-02-16 (2024-01-01/2024-02-01 fully None, 2024-02-15
partial, 2024-03-01 fully present) and all 5 requested variables (wind
speed 100m/10m, wind direction 100m, 2m temperature, surface pressure) turn
on together at that point. This is *earlier* than ECMWF IFS's ~2024-04
start but comparable in order of magnitude -- **not** the wider
"2022-11-24+" coverage the un-offset historical-forecast API advertises for
this model (that plain archive is not leakage-safe to use directly since it
does not isolate a single fixed-offset run; only the previous-runs offset
series was checked/used here). We fetch 2024-02-01 .. 2026-01-01 (train
rows before the archive start simply keep NaN in every icon_* column --
LightGBM handles this natively, never zero-filled).

Coordinate: reuses the identical all-turbine centroid as ``fetch_ecmwf``
(~(37.2815, 128.9629)); at ICON global's native ~13km resolution all three
KPX groups still fall in essentially the same neighborhood, matching the
existing single-point-serves-all-groups convention.

Usage:  python -m src.data.fetch_icon
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import pandas as pd

from configs.paths import DATA_INTERIM_DIR
from src.data.fetch_ecmwf import DISSEMINATION_MAX_DELAY as _ECMWF_MAX_DELAY
from src.data.fetch_ecmwf import cutoff_kst_for, implied_init_utc

logger = logging.getLogger(__name__)

LATITUDE = 37.2815
LONGITUDE = 128.9629

MODEL = "icon_global"

# Leakage-safe previous-runs offset -- see module docstring: reuses the
# ECMWF dissemination-delay worst case (conservative stand-in, ICON is
# publicly documented to disseminate faster). Do NOT lower without a
# verified DWD dissemination-timing source.
OFFSET_DAYS = 2

HOURLY_VARS = [
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_speed_10m",
    "temperature_2m",
    "surface_pressure",
]

_OUT_RENAME = {f"{v}_previous_day{OFFSET_DAYS}": f"icon_{v}" for v in HOURLY_VARS}

# Archive start for this coordinate/model was probed at ~2024-02-16; start a
# little earlier (02-01) to capture the transition cleanly, NaN either way.
START_DATE = "2024-02-01"
END_DATE = "2026-01-01"  # covers the last test forecast_kst_dtm (2026-01-01 00:00 KST)

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

OUTPUT_PARQUET = DATA_INTERIM_DIR / "icon.parquet"

# Conservative stand-in for DWD's actual dissemination delay -- see module
# docstring. Reusing ECMWF's own measured worst case rather than guessing a
# smaller number keeps this provably safe even if the public "~2-3h" claim
# for ICON turns out to be optimistic.
DISSEMINATION_MAX_DELAY = _ECMWF_MAX_DELAY
_RUN_CYCLE_HOURS = 6


def is_leakage_safe(forecast_kst: datetime, offset_days: int = OFFSET_DAYS) -> bool:
    """True iff the run behind ``previous_day{offset_days}`` for this
    forecast hour finished dissemination before the D-1 14:00 KST cutoff.
    Identical arithmetic to ``fetch_ecmwf.is_leakage_safe`` (imported
    building blocks), just parameterized on this module's own conservative
    ``DISSEMINATION_MAX_DELAY``.
    """
    valid_utc = forecast_kst - timedelta(hours=9)
    init = implied_init_utc(valid_utc, offset_days)
    available_utc = init + DISSEMINATION_MAX_DELAY
    available_kst = available_utc + timedelta(hours=9)
    return available_kst < cutoff_kst_for(forecast_kst)


def _fetch_chunk(start_date: str, end_date: str, retries: int = 3) -> dict:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "models": MODEL,
        "hourly": ",".join(f"{v}_previous_day{OFFSET_DAYS}" for v in HOURLY_VARS),
        "start_date": start_date,
        "end_date": end_date,
        "wind_speed_unit": "ms",
        "timezone": "Asia/Seoul",
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001 -- retry on any transient network/API error
            last_err = e
            wait = 5 * attempt
            logger.warning("Fetch %s..%s failed (attempt %d/%d): %s -- retrying in %ds", start_date, end_date, attempt, retries, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Open-Meteo fetch failed after {retries} attempts: {last_err}")


def fetch_icon() -> pd.DataFrame:
    """Fetch the full backfill window in yearly chunks, return one frame with
    ``forecast_kst_dtm`` + ``icon_*`` raw columns (NaN where unarchived).
    """
    chunks = [
        (START_DATE, "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2026-01-01", END_DATE),
    ]
    frames: list[pd.DataFrame] = []
    for start, end in chunks:
        payload = _fetch_chunk(start, end)
        hourly = payload["hourly"]
        df = pd.DataFrame(hourly)
        df["forecast_kst_dtm"] = pd.to_datetime(df.pop("time"))
        frames.append(df)
        logger.info("Fetched %s..%s: %d rows", start, end, len(df))
        time.sleep(2)  # be polite to the free tier

    out = pd.concat(frames, ignore_index=True).rename(columns=_OUT_RENAME)
    out = out[["forecast_kst_dtm"] + list(_OUT_RENAME.values())]
    out = out.drop_duplicates("forecast_kst_dtm").sort_values("forecast_kst_dtm").reset_index(drop=True)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    probe_hours = [datetime(2025, 6, 15, 1) + timedelta(hours=h) for h in range(24)]
    unsafe = [t for t in probe_hours if not is_leakage_safe(t, OFFSET_DAYS)]
    if unsafe:
        raise RuntimeError(f"OFFSET_DAYS={OFFSET_DAYS} is not leakage-safe for hours: {unsafe[:3]}...")

    df = fetch_icon()
    n = len(df)
    nn = df["icon_wind_speed_100m"].notna().sum()
    first_valid = df.loc[df["icon_wind_speed_100m"].notna(), "forecast_kst_dtm"].min()
    logger.info(
        "Backfill complete: %d rows (%s .. %s), ws100 non-null %d (%.1f%%), first ws100 %s",
        n, df["forecast_kst_dtm"].min(), df["forecast_kst_dtm"].max(), nn, 100 * nn / n, first_valid,
    )
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Wrote %s", OUTPUT_PARQUET)


if __name__ == "__main__":
    main()
