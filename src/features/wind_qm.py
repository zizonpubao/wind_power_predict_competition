"""Quantile-mapping bias correction of LDAPS forecast wind speed against
SCADA-observed nacelle wind speed (experiment_queue.md #9, 2026-08-09 batch
exp B).

Pure, leakage-safe by construction: the mapping is fit *once* on train-period
paired (forecast, actual) percentiles and then applied as a frozen function of
the forecast value alone -- nothing about the mapping depends on any row it is
later applied to, so it can be safely applied to the test split without
looking at any test-period actual generation/SCADA data (there is none for
test anyway, per CLAUDE.md section 3).

Method: classic percentile-matching bias correction (not the "empirical CDF
per row" full distributional QM some hydrology literature uses) -- take
``n_percentiles`` evenly spaced percentiles of the paired forecast and actual
samples, then treat ``(forecast_percentile_value -> actual_percentile_value)``
as a piecewise-linear monotonic map, applied via ``np.interp`` (which clips
out-of-range inputs to the nearest edge value rather than extrapolating
unboundedly).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_quantile_mapping(
    forecast: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    n_percentiles: int = 99,
) -> dict[str, np.ndarray]:
    """Fit a percentile-matching map from ``forecast`` to ``actual``.

    ``forecast``/``actual`` must be paired (same length, same underlying
    rows/hours) -- e.g. LDAPS forecast wind speed and SCADA-observed nacelle
    wind speed for the same hours. Rows with either value ``NaN`` are dropped
    before fitting.

    Returns a dict ``{"forecast_pctiles": ..., "actual_pctiles": ...}``, both
    length ``n_percentiles``, monotonically non-decreasing (percentiles of a
    real sample are always sorted) -- directly usable by
    ``apply_quantile_mapping``.
    """
    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if forecast.shape != actual.shape:
        raise ValueError(f"forecast and actual must be paired (same shape), got {forecast.shape} vs {actual.shape}")

    mask = ~(np.isnan(forecast) | np.isnan(actual))
    forecast, actual = forecast[mask], actual[mask]
    if len(forecast) < n_percentiles:
        raise ValueError(f"need at least {n_percentiles} paired non-NaN rows to fit, got {len(forecast)}")

    pct_levels = np.linspace(1, 99, n_percentiles)
    forecast_pctiles = np.percentile(forecast, pct_levels)
    actual_pctiles = np.percentile(actual, pct_levels)
    return {"forecast_pctiles": forecast_pctiles, "actual_pctiles": actual_pctiles}


def apply_quantile_mapping(new_forecast: np.ndarray | pd.Series, mapping: dict[str, np.ndarray]) -> np.ndarray:
    """Apply a fitted mapping (see ``fit_quantile_mapping``) to new forecast
    values. Out-of-range inputs are clipped to the nearest fitted percentile's
    mapped value (``np.interp`` default behavior), never extrapolated.
    """
    new_forecast = np.asarray(new_forecast, dtype=float)
    return np.interp(new_forecast, mapping["forecast_pctiles"], mapping["actual_pctiles"])
