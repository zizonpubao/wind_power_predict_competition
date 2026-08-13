"""Backfill GEM global (Canadian CMC, ~0.15 deg) forecasts from the Open-Meteo
Previous Runs API into ``data/interim/gem.parquet`` -- a fifth, independent NWP
source next to LDAPS/GFS (competition-provided), ECMWF IFS and ICON.

Leakage argument: identical to ``src.data.fetch_icon`` -- reuses ECMWF's own
measured worst-case dissemination delay (7h34m) as a conservative stand-in and
the same ``previous_day2`` run-cycle arithmetic (re-derived via
``fetch_ecmwf.implied_init_utc``/``cutoff_kst_for``), so ``OFFSET_DAYS = 2``
is safe by the identical hour-by-hour argument. GEM global runs 00/12 UTC
(12-hourly, slower cycle than ECMWF's 6-hourly); a 48h-older run is strictly
older information than ECMWF's already-verified-safe 48h offset, so the
cutoff margin only grows.

Coverage (probed 2026-08-13 at this exact coordinate, previous_day2):
2024-02-15 empty, 2024-02-20 full -- archive starts ~2024-02-16..20, same
partial-coverage regime as ECMWF (~2024-04) / ICON (~2024-02-16). Earlier
train rows keep NaN in every gem_* column (LightGBM handles this natively).
Unlike ECMWF/ICON, GEM exposes hub-height wind directly (80m/120m); we take
80m (turbine hub ~80-100m) plus 10m, direction 80m, t2m, surface pressure.

Usage:  python -m src.data.fetch_gem
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

MODEL = "gem_global"

OFFSET_DAYS = 2

HOURLY_VARS = [
    "wind_speed_80m",
    "wind_direction_80m",
    "wind_speed_10m",
    "temperature_2m",
    "surface_pressure",
]

_OUT_RENAME = {f"{v}_previous_day{OFFSET_DAYS}": f"gem_{v}" for v in HOURLY_VARS}

START_DATE = "2024-02-01"
END_DATE = "2026-01-01"

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

OUTPUT_PARQUET = DATA_INTERIM_DIR / "gem.parquet"

DISSEMINATION_MAX_DELAY = _ECMWF_MAX_DELAY
_RUN_CYCLE_HOURS = 12


def is_leakage_safe(forecast_kst: datetime, offset_days: int = OFFSET_DAYS) -> bool:
    """Same arithmetic as fetch_ecmwf/fetch_icon: the run behind
    ``previous_day{offset_days}`` must finish dissemination before the
    D-1 14:00 KST cutoff. ``implied_init_utc`` assumes a 6-hourly cycle;
    GEM's true 12-hourly cycle can only make the run *older* (larger
    margin), so this check is conservative.
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


def fetch_gem() -> pd.DataFrame:
    chunks = [
        (START_DATE, "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2026-01-01", END_DATE),
    ]
    frames: list[pd.DataFrame] = []
    for start, end in chunks:
        payload = _fetch_chunk(start, end)
        df = pd.DataFrame(payload["hourly"])
        df["forecast_kst_dtm"] = pd.to_datetime(df.pop("time"))
        frames.append(df)
        logger.info("Fetched %s..%s: %d rows", start, end, len(df))
        time.sleep(2)

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

    df = fetch_gem()
    n = len(df)
    nn = df["gem_wind_speed_80m"].notna().sum()
    first_valid = df.loc[df["gem_wind_speed_80m"].notna(), "forecast_kst_dtm"].min()
    logger.info(
        "Backfill complete: %d rows (%s .. %s), ws80 non-null %d (%.1f%%), first ws80 %s",
        n, df["forecast_kst_dtm"].min(), df["forecast_kst_dtm"].max(), nn, 100 * nn / n, first_valid,
    )
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Wrote %s", OUTPUT_PARQUET)


if __name__ == "__main__":
    main()
