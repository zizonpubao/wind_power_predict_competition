"""First-principles physics feature functions for the BARAM wind-power model.

Every function here is a **pure transform of forecast-derived (or, for the
SCADA empirical power curve, a precomputed-and-stored) quantity** -- so every
feature is identically computable for train and test, and none depends on any
value unavailable at ``data_available_kst_dtm`` for its ``forecast_kst_dtm``
(the project leakage rule, CLAUDE.md section 3). The SCADA power curve is the
one function that *reads* the train-only SCADA files, but it does so once to
fit a small monotone 1-D lookup table that is then frozen to JSON and applied
to forecast wind speed -- the fitted curve, not SCADA, is what feeds the
feature table, so it evaluates unchanged at test time.

Domain grounding
----------------
- **Air density** ``rho = sp / (R_specific * T)`` (ideal gas law, dry-air
  ``R_specific = 287.05 J/(kg*K)``). LDAPS/GFS ``surface_0_sp`` is in Pa
  (~88,000-103,000, verified) and ``t`` in Kelvin (~257-282, verified), so no
  unit conversion is needed. Wind *power* in the airflow is
  ``P ~ 0.5 * rho * A * v**3`` -- power scales **linearly with density**, so a
  cold dense winter air mass yields materially more power at the same wind
  speed than warm summer air. The existing pipeline only had ``v`` and a
  ``v``-only power-curve transform; ``rho`` and ``rho * v**3`` inject the
  density axis the physics says matters (CLAUDE.md EDA: label-SCADA gap has a
  winter-up/summer-down seasonality consistent with a density effect).
- **Wind shear exponent** ``alpha = ln(v_hi/v_lo) / ln(h_hi/h_lo)`` (power law).
  Already used inside ``wind_shear.py`` for hub extrapolation but never exposed
  as a feature itself; ``alpha`` is also a stability proxy (large alpha = stable
  stratification / strong shear). **Veer** (turning of the wind vector with
  height) is a complementary stability/baroclinicity signal.
- **Directional sector x speed**: ridge terrain channels wind, so the
  speed->power relationship differs by wind direction. Per-sector speed columns
  let a tree split the power response by sector without having to rediscover the
  sector boundaries from raw ``dir_sin``/``dir_cos``.
- **Spatial dispersion** across the LDAPS-16 / GFS-9 grids (std / range / CV of
  per-grid wind speed): a large spread flags a front / strong spatial gradient
  passing the site, where a single IDW mean is least trustworthy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Specific gas constant for dry air, J/(kg*K).
R_SPECIFIC_DRY_AIR = 287.05
#: ISA sea-level standard air density (kg/m^3), used only to normalise the
#: density ratio into an O(1) feature -- not a physical assumption about the site.
RHO_SEA_LEVEL = 1.225

_BLOCK_COLS = ["forecast_kst_dtm", "data_available_kst_dtm"]


# ---------------------------------------------------------------------------
# Batch 1: air density and density-corrected power-in-wind
# ---------------------------------------------------------------------------


def air_density(sp, temp_k, r_specific: float = R_SPECIFIC_DRY_AIR):
    """Ideal-gas dry-air density ``rho = sp / (R_specific * T)`` (kg/m^3).

    Parameters
    ----------
    sp: surface pressure in **Pa** (LDAPS/GFS ``surface_0_sp`` already is).
    temp_k: temperature in **Kelvin** (LDAPS ``heightAboveGround_2_t`` /
        GFS ``heightAboveGround_2_2t`` already are).
    """
    sp_arr = np.asarray(sp, dtype=float)
    temp_arr = np.asarray(temp_k, dtype=float)
    rho = sp_arr / (r_specific * temp_arr)
    return _wrap(rho, sp, temp_k)


def wind_power_density(rho, speed):
    """Kinetic power flux in the wind, ``0.5 * rho * v**3`` (W/m^2) -- the
    physically-correct combination of density and speed that a turbine's output
    ultimately tracks (before the power-curve cap). Negative/So NaN speeds are
    guarded by clipping speed at 0 before cubing.
    """
    rho_arr = np.asarray(rho, dtype=float)
    speed_arr = np.clip(np.asarray(speed, dtype=float), 0.0, None)
    return _wrap(0.5 * rho_arr * speed_arr**3, rho, speed)


def density_ratio(rho, rho_ref: float = RHO_SEA_LEVEL):
    """``rho / rho_ref`` -- the multiplicative density correction to apply to a
    density-agnostic power estimate (O(1), ~0.95-1.05 typically)."""
    rho_arr = np.asarray(rho, dtype=float)
    return _wrap(rho_arr / rho_ref, rho)


# ---------------------------------------------------------------------------
# Batch 2: shear exponent (stability proxy) and veer
# ---------------------------------------------------------------------------


def shear_exponent(v_low, h_low: float, v_high, h_high: float):
    """Power-law shear exponent ``alpha = ln(v_high/v_low) / ln(h_high/h_low)``.

    Thin wrapper around the same math as ``wind_shear.estimate_shear_exponent``
    but returning ``0.0`` (neutral-ish) rather than NaN where a level's speed is
    non-positive, so this can be used **directly as a model feature column**
    without a downstream fillna step (the extrapolation use-case in wind_shear
    deliberately keeps NaN so its caller can fall back to the 1/7 constant; a
    feature column wants a finite value instead).
    """
    v_low = np.asarray(v_low, dtype=float)
    v_high = np.asarray(v_high, dtype=float)
    v_low, v_high = np.broadcast_arrays(v_low, v_high)
    denom = np.log(float(h_high) / float(h_low))
    valid = (v_low > 0) & (v_high > 0)
    alpha = np.zeros(v_high.shape, dtype=float)
    alpha[valid] = np.log(v_high[valid] / v_low[valid]) / denom
    return _wrap(alpha, v_high, v_low)


def veer_cos(u_low, v_low, u_high, v_high):
    """Cosine of the veer angle between the low-level and high-level wind
    vectors: ``(u_lo*u_hi + v_lo*v_hi) / (|lo| |hi|)`` in [-1, 1].

    1 = wind blows the same compass direction at both heights (no veer,
    typically well-mixed/unstable); < 1 = the wind turns with height (veer,
    associated with stable stratification / thermal-wind baroclinicity). A
    degenerate zero-length vector at either level -> 0.0 (undefined angle, kept
    finite for a feature column).
    """
    u_low = np.asarray(u_low, dtype=float)
    v_low = np.asarray(v_low, dtype=float)
    u_high = np.asarray(u_high, dtype=float)
    v_high = np.asarray(v_high, dtype=float)
    dot = u_low * u_high + v_low * v_high
    mag = np.sqrt(u_low**2 + v_low**2) * np.sqrt(u_high**2 + v_high**2)
    out = np.divide(dot, mag, out=np.zeros_like(dot), where=mag > 0)
    return _wrap(out, u_low, u_high)


# ---------------------------------------------------------------------------
# Batch 3: directional-sector x speed interaction
# ---------------------------------------------------------------------------


def sector_speed_features(
    dir_deg, speed, n_sectors: int = 8, prefix: str = "ldaps_10m"
) -> pd.DataFrame:
    """One column per compass sector holding the wind *speed* when the wind
    direction falls in that sector and 0 otherwise -- an explicit
    direction-conditioned speed the tree can split on per sector (terrain
    channelling), rather than forcing it to reconstruct sectors from
    ``dir_sin``/``dir_cos``.

    Sector ``k`` covers ``[k*360/n_sectors, (k+1)*360/n_sectors)`` degrees
    (sector 0 = ``[0, 45)`` deg for the 8-sector default, i.e. the N->NE
    quadrant edge; boundaries are half-open at the upper edge). Returns a
    DataFrame of ``n_sectors`` columns
    named ``{prefix}_secspeed_{k}`` (index-aligned to the inputs if they are
    Series).
    """
    dir_arr = np.asarray(dir_deg, dtype=float) % 360.0
    spd_arr = np.asarray(speed, dtype=float)
    width = 360.0 / n_sectors
    sector = np.floor(dir_arr / width).astype(int) % n_sectors
    index = dir_deg.index if isinstance(dir_deg, pd.Series) else None
    data = {}
    for k in range(n_sectors):
        col = np.where(sector == k, spd_arr, 0.0)
        data[f"{prefix}_secspeed_{k}"] = col
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# Batch 4: spatial dispersion across grids (operates on the long grid frame)
# ---------------------------------------------------------------------------


def grid_speed_dispersion(
    df: pd.DataFrame, u_col: str, v_col: str, prefix: str
) -> pd.DataFrame:
    """Per-forecast-hour dispersion of wind speed **across the grids** (LDAPS 16
    / GFS 9), from a long-format frame (one row per forecast_kst_dtm x grid_id).

    A large across-grid spread means the site sits on a strong spatial gradient
    (front, terrain-forced convergence) where any single aggregated value is
    least reliable -- a signal an IDW/nearest mean alone throws away.

    Returns one row per forecast_kst_dtm (carrying ``data_available_kst_dtm``)
    with:
      - ``{prefix}_grid_std``: std of per-grid speed
      - ``{prefix}_grid_range``: max - min of per-grid speed
      - ``{prefix}_grid_cv``: std / mean (coefficient of variation, 0 where mean=0)
    """
    work = df[_BLOCK_COLS + [u_col, v_col]].copy()
    work["_spd"] = np.sqrt(work[u_col] ** 2 + work[v_col] ** 2)
    grouped = work.groupby(_BLOCK_COLS, as_index=False).agg(
        _std=("_spd", "std"),
        _min=("_spd", "min"),
        _max=("_spd", "max"),
        _mean=("_spd", "mean"),
    )
    grouped[f"{prefix}_grid_std"] = grouped["_std"].fillna(0.0)
    grouped[f"{prefix}_grid_range"] = grouped["_max"] - grouped["_min"]
    grouped[f"{prefix}_grid_cv"] = np.where(
        grouped["_mean"] > 0, grouped[f"{prefix}_grid_std"] / grouped["_mean"], 0.0
    )
    out_cols = _BLOCK_COLS + [f"{prefix}_grid_std", f"{prefix}_grid_range", f"{prefix}_grid_cv"]
    return grouped[out_cols].sort_values("forecast_kst_dtm").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Batch 5: SCADA empirical (measured) power curve
# ---------------------------------------------------------------------------

#: Glitch mask threshold on per-turbine 10-min energy (kWh) -- CLAUDE.md section
#: 4: vestas SCADA has ~0.48% physically-impossible rows (|value| up to ~5e7);
#: a single 3.6-4.2 MW turbine cannot exceed ~700 kWh in a 10-min bucket.
SCADA_POWER_ABS_MAX = 700.0
#: Plausible measured-wind-speed range (m/s) for masking nacelle anemometer glitches.
SCADA_WS_MIN, SCADA_WS_MAX = 0.0, 40.0


def fit_empirical_power_curve(
    ws, power, bin_width: float = 0.5, ws_max: float = 28.0
) -> dict[str, list[float]]:
    """Fit a monotone-nondecreasing empirical power curve from a cloud of
    (measured wind speed, measured 10-min energy) points.

    Bins ``ws`` into ``bin_width``-m/s bins over ``[0, ws_max]``, takes each
    bin's **median** power (robust to the residual glitches/outliers), then
    enforces non-decreasing values via a running max (a real turbine curve is
    monotone up to rated and flat after; the site rarely sees cut-out winds, so
    a running max is a safe shape prior over the observed range). The values are
    normalised to [0, 1] by the running max's top so the feature is scale-free
    and lets the model learn its own kWh scaling.

    Returns ``{"centers": [...], "values": [...]}`` (JSON-serialisable), a
    lookup table to be applied to *forecast* wind speed via ``apply_power_curve``.
    """
    ws = np.asarray(ws, dtype=float)
    power = np.asarray(power, dtype=float)
    mask = (
        np.isfinite(ws)
        & np.isfinite(power)
        & (ws >= SCADA_WS_MIN)
        & (ws <= SCADA_WS_MAX)
        & (np.abs(power) <= SCADA_POWER_ABS_MAX)
        & (power >= 0.0)
    )
    ws, power = ws[mask], power[mask]

    edges = np.arange(0.0, ws_max + bin_width, bin_width)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx = np.clip(np.digitize(ws, edges) - 1, 0, len(centers) - 1)
    med = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        vals = power[idx == k]
        if vals.size:
            med[k] = np.median(vals)
    # Fill empty low bins with 0 (below cut-in) and forward-fill high empties.
    med = pd.Series(med).ffill().fillna(0.0).to_numpy()
    med = np.maximum.accumulate(med)  # enforce monotone nondecreasing
    top = med.max()
    values = med / top if top > 0 else med
    return {"centers": centers.tolist(), "values": values.tolist()}


def apply_power_curve(curve: dict[str, list[float]], speed):
    """Evaluate a fitted empirical power curve at ``speed`` (linear
    interpolation over the stored bin centres; flat-extrapolated at both ends).
    Returns [0, 1]-normalised expected power for that speed.
    """
    centers = np.asarray(curve["centers"], dtype=float)
    values = np.asarray(curve["values"], dtype=float)
    spd = np.clip(np.asarray(speed, dtype=float), 0.0, None)
    out = np.interp(spd, centers, values)
    return _wrap(out, spd)


# ---------------------------------------------------------------------------
# small shared helper
# ---------------------------------------------------------------------------


def _wrap(result: np.ndarray, *inputs):
    """Return ``result`` as a pd.Series (index from the first Series input) if
    any input is a Series, else the plain ndarray -- mirrors
    ``weather_features``/``wind_shear`` calling convention."""
    for value in inputs:
        if isinstance(value, pd.Series):
            return pd.Series(np.asarray(result), index=value.index)
    return np.asarray(result)
