"""Backfill JMA GSM (Japan Meteorological Agency global, 0.5 deg) forecasts
from the Open-Meteo Previous Runs API into ``data/interim/jma.parquet``.

Why JMA despite the coarse 0.5-deg grid: it is the only additional NWP whose
``previous_day2`` archive at this coordinate covers the **entire** training
period (probed 2026-08-13: 2022-06-01 already full 24/24, still full at
2025-12-15), unlike ECMWF (~2024-04+), ICON (~2024-02+) and GEM (~2024-02+).
Full coverage means every CV fold trains on non-NaN jma_* values, so the CV
verdict on this source is trustworthy (no partial-coverage caveat), and the
neural-net tracks could in principle also consume it (they collapsed on the
sparse ECMWF columns precisely because ~75% of train rows were NaN there).

Leakage argument: identical arithmetic to ``fetch_icon``/``fetch_gem`` --
ECMWF's measured worst-case dissemination delay reused as a conservative
stand-in, same ``previous_day2`` offset, verified against the D-1 14:00 KST
cutoff at import time in ``main``. JMA GSM runs 00/06/12/18 UTC (6-hourly,
same cycle as ECMWF), so the ECMWF-verified arithmetic transfers unchanged.

JMA GSM exposes only 10m wind (no hub-height level); we take wind speed and
direction at 10m plus 2m temperature and surface pressure.

Usage:  python -m src.data.fetch_jma
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

MODEL = "jma_gsm"

OFFSET_DAYS = 2

HOURLY_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "temperature_2m",
    "surface_pressure",
]

_OUT_RENAME = {f"{v}_previous_day{OFFSET_DAYS}": f"jma_{v}" for v in HOURLY_VARS}

START_DATE = "2022-01-01"
END_DATE = "2026-01-01"

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

OUTPUT_PARQUET = DATA_INTERIM_DIR / "jma.parquet"

DISSEMINATION_MAX_DELAY = _ECMWF_MAX_DELAY
_RUN_CYCLE_HOURS = 6


def is_leakage_safe(forecast_kst: datetime, offset_days: int = OFFSET_DAYS) -> bool:
    """Same arithmetic as fetch_ecmwf: the run behind ``previous_day{offset}``
    must finish dissemination before the D-1 14:00 KST cutoff."""
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


def fetch_jma() -> pd.DataFrame:
    chunks = [
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
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

    df = fetch_jma()
    n = len(df)
    nn = df["jma_wind_speed_10m"].notna().sum()
    first_valid = df.loc[df["jma_wind_speed_10m"].notna(), "forecast_kst_dtm"].min()
    logger.info(
        "Backfill complete: %d rows (%s .. %s), ws10 non-null %d (%.1f%%), first ws10 %s",
        n, df["forecast_kst_dtm"].min(), df["forecast_kst_dtm"].max(), nn, 100 * nn / n, first_valid,
    )
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Wrote %s", OUTPUT_PARQUET)


if __name__ == "__main__":
    main()
