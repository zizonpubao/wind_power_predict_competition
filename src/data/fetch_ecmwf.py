"""Backfill ECMWF IFS (0.25 deg) forecasts from the Open-Meteo Previous Runs
API into ``data/interim/ecmwf_ifs.parquet`` -- the third, independent NWP
source next to the competition-provided LDAPS/GFS (see
``reports/domain_research/external_data_candidates.md``).

Why the Previous Runs API and why ``_previous_day2`` (leakage argument)
------------------------------------------------------------------------
The competition's information cutoff for target day D is **D-1 14:00 KST**
(= D-1 05:00 UTC): the provided LDAPS/GFS blocks for D are issued D-1 09:00
KST and usable from D-1 13:00 KST. Any external feature must likewise be
fully determined before that cutoff.

ECMWF IFS dissemination schedule (confluence.ecmwf.int, checked 2026-07-24):
  - 00z run: disseminated 05:45-07:34 UTC same day = 14:45-16:34 KST
  - 12z run: disseminated 17:45-19:34 UTC same day = 02:45-04:34 KST next day
  - 06z/18z runs: analogous +5h45m..+7h34m after init.
So for target day D the **D-1 00z run is NOT usable** (lands 14:45 KST D-1,
after the cutoff), while anything initialized at or before **D-2 18z** is
safe (D-2 18z lands by ~10:34 KST D-1 < 14:00 KST).

Open-Meteo Previous Runs API semantics -- verified empirically on
2026-08-07 against this exact coordinate:
  - the plain variable (``wind_speed_100m``) equals the Historical Forecast
    API's stitched best-lead series exactly (checked value-for-value on
    2025-06-10..11) -> for valid time T it comes from the most recent model
    run initialized at or before T (lead ~0-9 h, runs every 6 h);
  - ``*_previous_dayN`` returns the same valid time from the run **24*N
    hours older** (values diverge progressively with N, as expected).
Therefore ``previous_day2``'s implied init for valid time T is
``floor6h(T_utc) - 48h``. Across one KST target day D (T = D 01:00..24:00
KST = D-1 16:00 .. D 15:00 UTC) that init ranges over D-3 12z ... D-2 12z --
every one of them at or before D-2 12z, i.e. disseminated by 04:34 KST D-1,
**comfortably before the 14:00 KST cutoff with >9 h margin**. Even if the
stitching picked a run one cycle later than assumed, the worst case is
D-2 18z, still safe. ``previous_day1`` by the same arithmetic reaches
D-1 12z for the late hours of D -> leaks -> banned. Hence OFFSET_DAYS = 2
(the conservative, provably safe choice; see ``tests/test_ecmwf.py`` which
re-derives this arithmetic hour by hour).

Coverage
--------
Open-Meteo's previous-runs archive for ``ecmwf_ifs025`` at this coordinate
starts ~2024-04 for the 100 m wind fields (probed directly: 2024-03-01 all
None, 2024-04-20 fully present; 10 m/2 m/surface fields exist a bit earlier).
``wind_gusts_10m`` is not archived for this model at all (all None) and is
therefore excluded. We fetch 2024-04-01 .. 2026-01-01 (the last test
forecast_kst_dtm is 2026-01-01 00:00 KST); earlier train rows simply have no
ECMWF features (NaN -- LightGBM handles natively, never zero-filled).

Coordinate: single point at the all-turbine centroid (mean of the three
group centroids from ``src.features.weather_features._group_centroid``,
~(37.2815, 128.9629)). At 0.25 deg resolution all three KPX groups fall in
the same grid cell (API snaps to 37.25, 129.0), so one point serves all.

Usage:  python -m src.data.fetch_ecmwf
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

logger = logging.getLogger(__name__)

# All-turbine centroid (docs/info_raw.csv via weather_features._group_centroid).
LATITUDE = 37.2815
LONGITUDE = 128.9629

MODEL = "ecmwf_ifs025"

# Leakage-safe previous-runs offset -- see module docstring. Do NOT lower
# this to 1 without redoing the dissemination arithmetic: previous_day1
# resolves to the D-1 00z/06z/12z runs for part of the target day, all of
# which land after the D-1 14:00 KST cutoff.
OFFSET_DAYS = 2

# gusts intentionally absent: not archived for ecmwf_ifs025 (probed, all None).
HOURLY_VARS = [
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_speed_10m",
    "temperature_2m",
    "surface_pressure",
]

# Output column rename: raw API name (with offset suffix) -> parquet column.
_OUT_RENAME = {f"{v}_previous_day{OFFSET_DAYS}": f"ecmwf_{v}" for v in HOURLY_VARS}

START_DATE = "2024-04-01"
END_DATE = "2026-01-01"  # covers the last test forecast_kst_dtm (2026-01-01 00:00 KST)

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

OUTPUT_PARQUET = DATA_INTERIM_DIR / "ecmwf_ifs.parquet"

# ECMWF dissemination worst-case delay after init (12z run: 17:45-19:34 UTC).
DISSEMINATION_MAX_DELAY = timedelta(hours=7, minutes=34)
_RUN_CYCLE_HOURS = 6


def implied_init_utc(valid_utc: datetime, offset_days: int = OFFSET_DAYS) -> datetime:
    """Init time of the model run that ``*_previous_day{offset_days}``
    returns for ``valid_utc``, under the verified Open-Meteo semantics
    (most recent 6-hourly run at/before the valid time, minus 24h*offset).
    """
    floored = valid_utc.replace(minute=0, second=0, microsecond=0)
    floored -= timedelta(hours=floored.hour % _RUN_CYCLE_HOURS)
    return floored - timedelta(days=offset_days)


def cutoff_kst_for(forecast_kst: datetime) -> datetime:
    """Information cutoff (KST) for a forecast target hour: 14:00 on the day
    before its target day D. Hour 00:00 belongs to the *previous* KST day's
    block (blocks run D 01:00 .. D+1 00:00), hence the 1-hour shift.
    """
    target_day = (forecast_kst - timedelta(hours=1)).date()
    return datetime.combine(target_day, datetime.min.time()) - timedelta(days=1) + timedelta(hours=14)


def is_leakage_safe(forecast_kst: datetime, offset_days: int = OFFSET_DAYS) -> bool:
    """True iff the run behind ``previous_day{offset_days}`` for this
    forecast hour finished dissemination before the D-1 14:00 KST cutoff.
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


def fetch_ecmwf() -> pd.DataFrame:
    """Fetch the full backfill window in yearly chunks, return one frame with
    ``forecast_kst_dtm`` + ``ecmwf_*`` raw columns (NaN where unarchived).
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

    # Hard leakage guard: refuse to write anything if the configured offset
    # is not provably safe for every hour of a target day.
    probe_hours = [datetime(2025, 6, 15, 1) + timedelta(hours=h) for h in range(24)]
    unsafe = [t for t in probe_hours if not is_leakage_safe(t, OFFSET_DAYS)]
    if unsafe:
        raise RuntimeError(f"OFFSET_DAYS={OFFSET_DAYS} is not leakage-safe for hours: {unsafe[:3]}...")

    df = fetch_ecmwf()
    n = len(df)
    nn = df["ecmwf_wind_speed_100m"].notna().sum()
    first_valid = df.loc[df["ecmwf_wind_speed_100m"].notna(), "forecast_kst_dtm"].min()
    logger.info(
        "Backfill complete: %d rows (%s .. %s), ws100 non-null %d (%.1f%%), first ws100 %s",
        n, df["forecast_kst_dtm"].min(), df["forecast_kst_dtm"].max(), nn, 100 * nn / n, first_valid,
    )
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Wrote %s", OUTPUT_PARQUET)


if __name__ == "__main__":
    main()
