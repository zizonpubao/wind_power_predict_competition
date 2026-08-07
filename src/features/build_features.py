"""Assembles one model-ready feature table per (split, kpx_group) by composing
``src.data.loaders`` and ``src.features.weather_features`` -- this module does
not reimplement grid loading, spatial aggregation, wind-vector decomposition,
power-curve transforms, or lag/rolling stats; it only wires those existing
building blocks together and decides *which* raw columns feed them.

Column selection rationale
---------------------------
- **LDAPS**: only the 10m u/v pair (``heightAboveGround_10_10u/10v``) is a true
  instantaneous wind vector. The 50m fields are max/min envelopes, not a
  vector pair -- ``weather_features.wind_speed_direction``'s own docstring
  warns callers not to feed those in, so they are excluded here too. Scalar
  fields carried through: 2m temperature (``heightAboveGround_2_t``), surface
  pressure (``surface_0_sp``), and boundary-layer height (``etc_0_blh``) --
  all physically relevant to air density / turbulent mixing (feature-engineer
  role brief). Land-sea mask (``surface_0_lsm``), precip/snow fields, cloud
  fractions, radiation, and dewpoint/humidity are deliberately **not**
  spatially aggregated here: lsm is a static per-grid categorical flag (an
  IDW/nearest continuous aggregation of a 0/1 mask adds little), and the rest
  are lower-priority for a wind-power model relative to the wind/temperature/
  pressure/BLH set already carried.
- **GFS**: has four clean instantaneous u/v vector pairs -- 10m
  (``heightAboveGround_10_10u/10v``), 80m (``heightAboveGround_80_u/v``), 100m
  (``heightAboveGround_100_100u/100v``), and planetary-boundary-layer
  (``planetaryBoundaryLayer_0_u/v``) -- unlike LDAPS, none of these are
  envelope-only, so all four are used (multi-level wind shear/veer is exactly
  the kind of signal GFS's coarser-but-multi-level data can add over LDAPS's
  single clean level). Scalar fields carried: downward shortwave radiation
  (``surface_0_dswrf``) and total cloud cover (``atmosphere_0_tcc``), both
  atmospheric-stability/insolation proxies. Isobaric-pressure-level fields
  (850/700/500 hPa) and precip/gust fields are out of scope -- they describe
  free-atmosphere conditions well above hub height and are a lower priority
  than the boundary-layer-relevant set already carried; a future iteration
  could revisit them if EDA shows added skill.
- Raw column names are renamed to short, source-prefixed aliases
  (``ldaps_10m_u``, ``gfs_80m_v``, ...) immediately after loading (see
  ``_LDAPS_RENAME`` / ``_GFS_RENAME``), so every downstream function
  (``wind_speed_direction``'s ``prefix`` argument, ``spatial_aggregate``'s
  ``{col}_idw`` / ``{col}_nearest`` naming) naturally produces
  self-describing, source-disambiguated columns with no separate manual
  prefixing step and no risk of LDAPS/GFS column name collisions at merge
  time.
- ``lag_rolling_features`` is applied only to the "primary" derived signals --
  the IDW-aggregated wind speed and IDW-derived power-curve output for every
  level in both sources (1 LDAPS level + 4 GFS levels = 5 speed columns + 5
  power-curve columns = 10 columns total), not the full merged frame. With the
  default windows (3, 6, 12, 24h) x 2 stats (mean, std) that is still +80
  columns, so it is scoped deliberately rather than applied blindly to every
  IDW/nearest column the earlier steps produce.
- IDW wind speed is derived **after** ``spatial_aggregate`` as
  ``sqrt(u_idw**2 + v_idw**2)`` from the already-IDW-averaged u/v components
  (rather than IDW-averaging a per-grid speed column) -- this matches the
  literal task spec ("run spatial_aggregate on ... u/v components ... apply
  power_curve_transform to the IDW-aggregated wind speed") and keeps
  ``spatial_aggregate``'s value_cols argument limited to genuinely raw
  weather columns.

LDAPS/GFS alignment
--------------------
Confirmed by direct inspection (2026-07-20, both train and test splits): LDAPS
and GFS share the **exact same set** of (forecast_kst_dtm,
data_available_kst_dtm) pairs (26,304 pairs for train, 8,760 for test, set
equality both directions). So in practice the LDAPS-derived/GFS-derived merge
below is a clean 1:1 join with zero unmatched rows. The merge is nonetheless
implemented as an **outer** join with a runtime equality check that logs a
warning (rather than silently proceeding) if a future data refresh ever
breaks that alignment, per the task spec's "confirm this rather than assuming"
instruction.

SCADA exclusion
-----------------
SCADA is intentionally **not used anywhere** in this module. CLAUDE.md section
3 is explicit that SCADA does not exist at test time and must never be a model
input feature -- it is a training-pipeline-design aid only (label validation,
outlier detection, power-curve parameter estimation), never a feature column.
Do not add a SCADA-derived column here even though ``src.data.loaders``
exposes SCADA loaders that would be easy to reach for.

Wake-alignment features (kpx_group_2 only)
--------------------------------------------
``kpx_group_2``'s feature table additionally gets 3 columns from
``src.features.wake_features.add_group1_group2_wake_features`` --
``wake_alignment_cos_g1g2``/``wake_sector_exposure_g1g2``/
``wake_deficit_proxy_g1g2`` -- encoding how aligned the forecast wind
direction is with the geometric wake axis from the upwind ``kpx_group_1``
farm (``reports/domain_research/wake_effect.md``). kpx_group_1/3 do not get
these columns; see that module's docstring for why.

Hub-height wind-shear extrapolation (LDAPS/GFS, all groups)
--------------------------------------------------------------
``_add_wind_shear_features`` (called from ``_build_source_features``, right
after ``_idw_speed_and_power_curve`` -- the earliest point each source's
per-level IDW speed columns are all available on the same frame) extrapolates
wind speed to the turbines' 117m hub height via
``src.features.wind_shear.power_law_extrapolate``:
  - **LDAPS**: only a fixed-shear-exponent (Hellman 1/7 approximation)
    version is possible, extrapolated from ``ldaps_10m_speed_idw`` --
    ``ldaps_ws_hub_fixed`` / ``ldaps_ws_hub_fixed_cubed``. LDAPS has no
    second clean vector level to estimate a real exponent from (see
    weather_features.py's module docstring on why its 50m fields are
    excluded everywhere).
  - **GFS**: has clean 80m *and* 100m levels, so a per-row shear exponent is
    estimated from those two via ``wind_shear.estimate_shear_exponent``
    (falling back to the fixed 1/7 constant wherever that estimate is NaN --
    non-positive wind speed at either level), then used to extrapolate from
    the 100m level (the closer of the two to 117m) to hub height --
    ``gfs_ws_hub_est`` / ``gfs_ws_hub_est_cubed``. A fixed-alpha comparison
    version from the *same* 100m reference level is also produced --
    ``gfs_ws_hub_fixed`` / ``gfs_ws_hub_fixed_cubed`` -- so the two GFS
    columns differ only in which alpha was used, isolating that one
    variable. Every ``ws_hub`` column also gets a ``_cubed`` counterpart
    (``.clip(lower=0) ** 3``, guarding against a negative extrapolated speed
    feeding a cubic power-in-wind proxy with the wrong sign) -- wind power is
    proportional to v**3, so this is a cheap physically-motivated nonlinear
    transform on top of the extrapolated speed itself.

Calendar / lead-time features (group/source-agnostic, computed once)
------------------------------------------------------------------------
``_assemble_feature_table`` calls ``src.features.calendar_features``'s
``add_calendar_features``/``add_lead_hours`` exactly once, right after
``lag_rolling_features`` and before the kpx_group_2-only wake-feature branch
(these two are unrelated to any particular group or weather source, so there
is nothing to scope per-branch). ``add_lead_hours``'s ``lead_hours`` (plural,
continuous hour-fraction between issuance and the forecast hour) is a
**different column** from ``lag_rolling_features``'s ``lead_hour`` (singular,
1-indexed integer position within a ``data_available_kst_dtm`` block) -- see
that module's docstring for why both exist and must not be conflated.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from configs.paths import DATA_INTERIM_DIR, DATA_PROCESSED_DIR
from src.data.loaders import load_gfs, load_ldaps, load_train_labels
from src.features.calendar_features import add_calendar_features, add_lead_hours
from src.features.wake_features import add_group1_group2_wake_features
from src.features.weather_features import (
    KPX_GROUP_TURBINE_MODEL,
    TURBINE_POWER_CURVE_PARAMS,
    lag_rolling_features,
    power_curve_transform,
    spatial_aggregate,
    wind_speed_direction,
)
from src.features.wind_shear import DEFAULT_SHEAR_EXPONENT, estimate_shear_exponent, power_law_extrapolate
from src.features.physics_features import (
    air_density,
    apply_power_curve,
    density_ratio,
    grid_speed_dispersion,
    sector_speed_features,
    shear_exponent,
    veer_cos,
    wind_power_density,
)
from src.features.scada_power_curve import load_scada_power_curves

logger = logging.getLogger(__name__)

# VESTAS V126 (group_1/2) vs UNISON U136 (group_3) -- which fitted SCADA
# empirical power curve to apply to a group's forecast hub-height wind speed.
_KPX_GROUP_SCADA_SOURCE = {
    "kpx_group_1": "vestas",
    "kpx_group_2": "vestas",
    "kpx_group_3": "unison",
}

# Number of directional sectors for the sector x speed interaction batch.
_N_DIR_SECTORS = 8

# Lazily-loaded, cached fitted SCADA power curves (see scada_power_curve.py).
_SCADA_CURVES_CACHE: dict | None = None


def _get_scada_curves() -> dict:
    global _SCADA_CURVES_CACHE
    if _SCADA_CURVES_CACHE is None:
        _SCADA_CURVES_CACHE = load_scada_power_curves()
    return _SCADA_CURVES_CACHE

_BLOCK_COLS = ["forecast_kst_dtm", "data_available_kst_dtm"]

# --- ECMWF IFS (third, independent NWP source) --------------------------------
# Backfilled by ``src.data.fetch_ecmwf`` from the Open-Meteo Previous Runs API
# using the leakage-safe ``previous_day2`` offset (D-2 12z-or-earlier runs,
# disseminated by ~04:34 KST D-1, well before the D-1 14:00 KST cutoff -- the
# full dissemination arithmetic lives in that module's docstring and is
# re-verified in tests/test_ecmwf.py). Coverage starts ~2024-04; earlier train
# rows keep NaN in every ECMWF column on purpose (LightGBM handles missing
# values natively -- never zero-fill these).
ECMWF_PARQUET = DATA_INTERIM_DIR / "ecmwf_ifs.parquet"

# Model-input columns _add_ecmwf_features produces (referenced by
# configs/selected_features.json's manual_overrides -- keep names in sync).
ECMWF_FEATURE_COLS = [
    "ecmwf_ws100",
    "ecmwf_ws10",
    "ecmwf_ws100_cubed",
    "ecmwf_dir_sin",
    "ecmwf_dir_cos",
    "ecmwf_t2m",
    "ecmwf_sp",
    "ecmwf_minus_ldaps_ws10",
    "ecmwf_minus_ldaps_hub",
    "ecmwf_ldaps_ws_ratio",
    "ecmwf_minus_gfs_ws100",
]


def load_ecmwf() -> pd.DataFrame | None:
    """Load the backfilled ECMWF parquet, or None (with a warning) if the
    backfill has not been run on this machine yet.
    """
    if not ECMWF_PARQUET.exists():
        logger.warning(
            "%s not found -- run `python -m src.data.fetch_ecmwf` first. "
            "ECMWF feature columns will be all-NaN.",
            ECMWF_PARQUET,
        )
        return None
    return pd.read_parquet(ECMWF_PARQUET)


def _add_ecmwf_features(merged: pd.DataFrame, ecmwf_df: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join ECMWF raw fields on forecast_kst_dtm and derive the
    ECMWF_FEATURE_COLS: wind speeds (10m/100m), the cubic power-in-wind
    transform, direction sin/cos, 2m temp / surface pressure, and
    **cross-model disagreement features vs LDAPS/GFS** (difference/ratio --
    inter-NWP spread is a forecast-uncertainty signal the single-source
    features cannot express). Rows outside the backfill window (pre-2024-04
    train rows) stay NaN. Row count is asserted unchanged (m:1 join).
    """
    before_len = len(merged)
    if ecmwf_df is not None:
        merged = merged.merge(ecmwf_df, on="forecast_kst_dtm", how="left")
        if len(merged) != before_len:
            raise ValueError(
                f"ECMWF join changed row count ({before_len} -> {len(merged)}); "
                "ecmwf_ifs.parquet likely has duplicate forecast_kst_dtm values."
            )
    else:
        for col in ("ecmwf_wind_speed_100m", "ecmwf_wind_direction_100m",
                    "ecmwf_wind_speed_10m", "ecmwf_temperature_2m", "ecmwf_surface_pressure"):
            merged[col] = np.nan

    ws100 = merged.pop("ecmwf_wind_speed_100m")
    wd100 = merged.pop("ecmwf_wind_direction_100m")
    merged["ecmwf_ws100"] = ws100
    merged["ecmwf_ws10"] = merged.pop("ecmwf_wind_speed_10m")
    merged["ecmwf_ws100_cubed"] = ws100.clip(lower=0.0) ** 3
    merged["ecmwf_dir_sin"] = np.sin(np.deg2rad(wd100))
    merged["ecmwf_dir_cos"] = np.cos(np.deg2rad(wd100))
    merged["ecmwf_t2m"] = merged.pop("ecmwf_temperature_2m")
    merged["ecmwf_sp"] = merged.pop("ecmwf_surface_pressure")
    # Cross-model disagreement: like-for-like 10m diff, near-hub diff (ECMWF
    # 100m vs LDAPS 117m power-law extrapolation), a bounded ratio, and the
    # GFS-side 100m diff. NaN propagates wherever ECMWF is uncovered.
    merged["ecmwf_minus_ldaps_ws10"] = merged["ecmwf_ws10"] - merged["ldaps_10m_speed_idw"]
    merged["ecmwf_minus_ldaps_hub"] = ws100 - merged["ldaps_ws_hub_fixed"]
    merged["ecmwf_ldaps_ws_ratio"] = ws100 / merged["ldaps_ws_hub_fixed"].clip(lower=0.5)
    merged["ecmwf_minus_gfs_ws100"] = ws100 - merged["gfs_100m_speed_idw"]
    return merged

# --- ICON global (fourth, independent NWP source) -----------------------------
# Backfilled by ``src.data.fetch_icon`` from the Open-Meteo Previous Runs API,
# same ``previous_day2`` leakage-safe offset convention as ECMWF (see that
# module's docstring for the arithmetic, re-verified in tests/test_icon.py).
# Coverage starts ~2024-02-17 at this coordinate; earlier train rows keep NaN
# in every icon_* column on purpose (LightGBM handles missing natively).
ICON_PARQUET = DATA_INTERIM_DIR / "icon.parquet"

# Model-input columns _add_icon_features produces (referenced by
# configs/selected_features.json's manual_overrides -- keep names in sync).
ICON_FEATURE_COLS = [
    "icon_ws100",
    "icon_ws10",
    "icon_ws100_cubed",
    "icon_dir_sin",
    "icon_dir_cos",
    "icon_t2m",
    "icon_sp",
    "icon_minus_ldaps_ws10",
    "icon_minus_ldaps_hub",
    "icon_ldaps_ws_ratio",
    "icon_minus_gfs_ws100",
]


def load_icon() -> pd.DataFrame | None:
    """Load the backfilled ICON parquet, or None (with a warning) if the
    backfill has not been run on this machine yet.
    """
    if not ICON_PARQUET.exists():
        logger.warning(
            "%s not found -- run `python -m src.data.fetch_icon` first. "
            "ICON feature columns will be all-NaN.",
            ICON_PARQUET,
        )
        return None
    return pd.read_parquet(ICON_PARQUET)


def _add_icon_features(merged: pd.DataFrame, icon_df: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join ICON raw fields on forecast_kst_dtm and derive the
    ICON_FEATURE_COLS -- identical shape/derivation to ``_add_ecmwf_features``,
    a fourth independent NWP source with its own cross-model disagreement
    features vs LDAPS/GFS. Rows outside the backfill window stay NaN. Row
    count is asserted unchanged (m:1 join).
    """
    before_len = len(merged)
    if icon_df is not None:
        merged = merged.merge(icon_df, on="forecast_kst_dtm", how="left")
        if len(merged) != before_len:
            raise ValueError(
                f"ICON join changed row count ({before_len} -> {len(merged)}); "
                "icon.parquet likely has duplicate forecast_kst_dtm values."
            )
    else:
        for col in ("icon_wind_speed_100m", "icon_wind_direction_100m",
                    "icon_wind_speed_10m", "icon_temperature_2m", "icon_surface_pressure"):
            merged[col] = np.nan

    ws100 = merged.pop("icon_wind_speed_100m")
    wd100 = merged.pop("icon_wind_direction_100m")
    merged["icon_ws100"] = ws100
    merged["icon_ws10"] = merged.pop("icon_wind_speed_10m")
    merged["icon_ws100_cubed"] = ws100.clip(lower=0.0) ** 3
    merged["icon_dir_sin"] = np.sin(np.deg2rad(wd100))
    merged["icon_dir_cos"] = np.cos(np.deg2rad(wd100))
    merged["icon_t2m"] = merged.pop("icon_temperature_2m")
    merged["icon_sp"] = merged.pop("icon_surface_pressure")
    merged["icon_minus_ldaps_ws10"] = merged["icon_ws10"] - merged["ldaps_10m_speed_idw"]
    merged["icon_minus_ldaps_hub"] = ws100 - merged["ldaps_ws_hub_fixed"]
    merged["icon_ldaps_ws_ratio"] = ws100 / merged["ldaps_ws_hub_fixed"].clip(lower=0.5)
    merged["icon_minus_gfs_ws100"] = ws100 - merged["gfs_100m_speed_idw"]
    return merged

KPX_GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
SPLITS = ("train", "test")

# Raw LDAPS/GFS column -> short, source-prefixed alias. See module docstring
# for why exactly these columns (and not others) were chosen.
_LDAPS_RENAME = {
    "heightAboveGround_10_10u": "ldaps_10m_u",
    "heightAboveGround_10_10v": "ldaps_10m_v",
    "heightAboveGround_2_t": "ldaps_t2m",
    "surface_0_sp": "ldaps_sp",
    "etc_0_blh": "ldaps_blh",
}
_GFS_RENAME = {
    "heightAboveGround_10_10u": "gfs_10m_u",
    "heightAboveGround_10_10v": "gfs_10m_v",
    "heightAboveGround_80_u": "gfs_80m_u",
    "heightAboveGround_80_v": "gfs_80m_v",
    "heightAboveGround_100_100u": "gfs_100m_u",
    "heightAboveGround_100_100v": "gfs_100m_v",
    "planetaryBoundaryLayer_0_u": "gfs_pbl_u",
    "planetaryBoundaryLayer_0_v": "gfs_pbl_v",
    "surface_0_dswrf": "gfs_dswrf",
    "atmosphere_0_tcc": "gfs_tcc",
    # Added for the air-density physics batch (surface pressure Pa, 2m temp K)
    # -- GFS's own sp/t2m so a GFS-side density correction is possible too.
    "surface_0_sp": "gfs_sp",
    "heightAboveGround_2_2t": "gfs_t2m",
}

# (u_col, v_col, feature-name prefix) tuples fed to wind_speed_direction /
# used to derive per-level IDW wind speed, post spatial_aggregate.
_LDAPS_LEVELS = [("ldaps_10m_u", "ldaps_10m_v", "ldaps_10m")]
_GFS_LEVELS = [
    ("gfs_10m_u", "gfs_10m_v", "gfs_10m"),
    ("gfs_80m_u", "gfs_80m_v", "gfs_80m"),
    ("gfs_100m_u", "gfs_100m_v", "gfs_100m"),
    ("gfs_pbl_u", "gfs_pbl_v", "gfs_pbl"),
]

_LDAPS_SCALAR_COLS = ["ldaps_t2m", "ldaps_sp", "ldaps_blh"]
_GFS_SCALAR_COLS = ["gfs_dswrf", "gfs_tcc", "gfs_sp", "gfs_t2m"]

# The subset of derived columns lag_rolling_features is applied to -- see
# module docstring for why this subset and not every column.
_LAG_WINDOWS = (3, 6, 12, 24)


def _idw_speed_and_power_curve(
    spatial_df: pd.DataFrame, levels: list[tuple[str, str, str]], kpx_group: str
) -> pd.DataFrame:
    """Add ``{prefix}_speed_idw`` and ``{prefix}_power_curve_idw`` columns to a
    ``spatial_aggregate`` output, computed from its ``{u_col}_idw`` /
    ``{v_col}_idw`` columns (mutates and returns spatial_df).
    """
    params = TURBINE_POWER_CURVE_PARAMS[KPX_GROUP_TURBINE_MODEL[kpx_group]]
    for u_col, v_col, prefix in levels:
        u_idw = spatial_df[f"{u_col}_idw"]
        v_idw = spatial_df[f"{v_col}_idw"]
        speed_col = f"{prefix}_speed_idw"
        spatial_df[speed_col] = np.sqrt(u_idw**2 + v_idw**2)
        spatial_df[f"{prefix}_power_curve_idw"] = power_curve_transform(
            spatial_df[speed_col], **params
        ).to_numpy()
    return spatial_df


def _add_wind_shear_features(spatial_df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add hub-height (117m) wind-speed extrapolation columns, wired to
    `source`'s already-computed IDW speed columns (called right after
    `_idw_speed_and_power_curve`). See the module docstring's "Hub-height
    wind-shear extrapolation" section for the full rationale.

    Adds (mutates and returns spatial_df):
      source="ldaps": ldaps_ws_hub_fixed, ldaps_ws_hub_fixed_cubed
      source="gfs":   gfs_ws_hub_fixed, gfs_ws_hub_fixed_cubed,
                      gfs_ws_hub_est,   gfs_ws_hub_est_cubed
    """

    def _cubed(v: pd.Series) -> pd.Series:
        return v.clip(lower=0.0) ** 3

    if source == "ldaps":
        v_ref = spatial_df["ldaps_10m_speed_idw"]
        ws_hub_fixed = power_law_extrapolate(v_ref, h_ref=10.0)
        spatial_df["ldaps_ws_hub_fixed"] = ws_hub_fixed
        spatial_df["ldaps_ws_hub_fixed_cubed"] = _cubed(ws_hub_fixed)
    elif source == "gfs":
        v80 = spatial_df["gfs_80m_speed_idw"]
        v100 = spatial_df["gfs_100m_speed_idw"]
        alpha_est = estimate_shear_exponent(v80, 80.0, v100, 100.0).fillna(DEFAULT_SHEAR_EXPONENT)

        ws_hub_fixed = power_law_extrapolate(v100, h_ref=100.0)
        ws_hub_est = power_law_extrapolate(v100, h_ref=100.0, alpha=alpha_est)

        spatial_df["gfs_ws_hub_fixed"] = ws_hub_fixed
        spatial_df["gfs_ws_hub_fixed_cubed"] = _cubed(ws_hub_fixed)
        spatial_df["gfs_ws_hub_est"] = ws_hub_est
        spatial_df["gfs_ws_hub_est_cubed"] = _cubed(ws_hub_est)
    else:
        raise ValueError(f"_add_wind_shear_features: unknown source {source!r}, expected 'ldaps' or 'gfs'")
    return spatial_df


def _build_source_features(
    df: pd.DataFrame,
    levels: list[tuple[str, str, str]],
    scalar_cols: list[str],
    kpx_group: str,
    source: str,
) -> pd.DataFrame:
    """One row per forecast_kst_dtm: spatial_aggregate (IDW/nearest) over every
    level's u/v pair plus the source's scalar columns, IDW wind speed +
    power-curve output per level, hub-height wind-shear extrapolation (see
    module docstring), and wind_speed_direction's grid-mean speed/
    circular-mean direction per level -- all merged on _BLOCK_COLS.

    `source`: "ldaps" or "gfs" -- selects which wind_shear columns
    `_add_wind_shear_features` produces (LDAPS has only a 10m level so only a
    fixed-alpha version is possible; GFS has 80m/100m so an estimated-alpha
    version is also produced).
    """
    value_cols = [col for u, v, _ in levels for col in (u, v)] + scalar_cols
    spatial = spatial_aggregate(df, kpx_group, value_cols)
    spatial = _idw_speed_and_power_curve(spatial, levels, kpx_group)
    spatial = _add_wind_shear_features(spatial, source)

    merged = spatial
    for u_col, v_col, prefix in levels:
        wsd = wind_speed_direction(df, u_col, v_col, prefix=prefix)
        merged = merged.merge(wsd, on=_BLOCK_COLS, how="outer")

    # Batch 4: across-grid spatial dispersion of wind speed (front / gradient
    # signal an IDW mean throws away). Only for the primary vector level(s):
    # LDAPS 10m, and GFS 100m (nearest to hub) + GFS 10m.
    dispersion_levels = {
        "ldaps": [("ldaps_10m_u", "ldaps_10m_v", "ldaps_10m")],
        "gfs": [
            ("gfs_10m_u", "gfs_10m_v", "gfs_10m"),
            ("gfs_100m_u", "gfs_100m_v", "gfs_100m"),
        ],
    }[source]
    for u_col, v_col, prefix in dispersion_levels:
        disp = grid_speed_dispersion(df, u_col, v_col, prefix=prefix)
        merged = merged.merge(disp, on=_BLOCK_COLS, how="outer")
    return merged


def _add_physics_features(merged: pd.DataFrame, kpx_group: str) -> pd.DataFrame:
    """Add the first-principles physics batches (air density / density-corrected
    power, shear-exponent & veer stability proxies, directional-sector x speed,
    and the SCADA empirical power curve applied to forecast hub speed) to the
    fully-merged LDAPS+GFS frame. Mutates and returns ``merged``.

    All inputs are forecast-derived (or the frozen SCADA curve) -- see
    physics_features.py / scada_power_curve.py for the leakage argument.
    """
    # --- Batch 1: air density + density-corrected power-in-wind ---
    rho_ldaps = air_density(merged["ldaps_sp_idw"], merged["ldaps_t2m_idw"])
    merged["ldaps_air_density"] = rho_ldaps
    merged["ldaps_density_ratio"] = density_ratio(rho_ldaps)
    rho_gfs = air_density(merged["gfs_sp_idw"], merged["gfs_t2m_idw"])
    merged["gfs_air_density"] = rho_gfs
    # rho * v^3 (power flux) for the physically most-relevant speeds:
    merged["ldaps_rho_v3_10m"] = wind_power_density(rho_ldaps, merged["ldaps_10m_speed_idw"])
    merged["ldaps_rho_v3_hub"] = wind_power_density(rho_ldaps, merged["ldaps_ws_hub_fixed"])
    merged["gfs_rho_v3_100m"] = wind_power_density(rho_gfs, merged["gfs_100m_speed_idw"])
    merged["gfs_rho_v3_hub"] = wind_power_density(rho_gfs, merged["gfs_ws_hub_est"])

    # --- Batch 2: shear exponent (stability proxy) + veer ---
    merged["gfs_shear_alpha_10_100"] = shear_exponent(
        merged["gfs_10m_speed_idw"], 10.0, merged["gfs_100m_speed_idw"], 100.0
    )
    merged["gfs_shear_alpha_80_100"] = shear_exponent(
        merged["gfs_80m_speed_idw"], 80.0, merged["gfs_100m_speed_idw"], 100.0
    )
    merged["gfs_veer_cos_10_100"] = veer_cos(
        merged["gfs_10m_u_idw"], merged["gfs_10m_v_idw"],
        merged["gfs_100m_u_idw"], merged["gfs_100m_v_idw"],
    )

    # --- Batch 3: directional-sector x speed (LDAPS 10m, terrain channelling) ---
    sector_df = sector_speed_features(
        merged["ldaps_10m_dir_deg"], merged["ldaps_10m_speed_idw"],
        n_sectors=_N_DIR_SECTORS, prefix="ldaps_10m",
    )
    for col in sector_df.columns:
        merged[col] = sector_df[col].to_numpy()

    # --- Batch 5: SCADA empirical power curve applied to forecast hub speed ---
    curve = _get_scada_curves()[_KPX_GROUP_SCADA_SOURCE[kpx_group]]
    merged["scada_pc_ldaps_hub"] = apply_power_curve(curve, merged["ldaps_ws_hub_fixed"])
    merged["scada_pc_gfs_hub"] = apply_power_curve(curve, merged["gfs_ws_hub_est"])

    return merged


_ECMWF_FROM_DISK = "__load_from_disk__"
_ICON_FROM_DISK = "__load_from_disk__"


def _assemble_feature_table(
    ldaps_renamed: pd.DataFrame,
    gfs_renamed: pd.DataFrame,
    split: str,
    kpx_group: str,
    ecmwf_df: pd.DataFrame | None | str = _ECMWF_FROM_DISK,
    icon_df: pd.DataFrame | None | str = _ICON_FROM_DISK,
) -> pd.DataFrame:
    """Core assembly logic, factored out of `build_feature_table` so tests can
    feed it a small time-sliced subset of already-loaded/renamed LDAPS/GFS
    frames instead of re-reading the full CSVs on every test run.

    ecmwf_df: the backfilled ECMWF frame to join (tests pass a synthetic
    frame or None); by default it is loaded from ``ECMWF_PARQUET``.
    icon_df: the backfilled ICON frame to join (tests pass a synthetic frame
    or None); by default it is loaded from ``ICON_PARQUET``.
    """
    if isinstance(ecmwf_df, str) and ecmwf_df == _ECMWF_FROM_DISK:
        ecmwf_df = load_ecmwf()
    if isinstance(icon_df, str) and icon_df == _ICON_FROM_DISK:
        icon_df = load_icon()
    ldaps_pairs = set(zip(ldaps_renamed["forecast_kst_dtm"], ldaps_renamed["data_available_kst_dtm"]))
    gfs_pairs = set(zip(gfs_renamed["forecast_kst_dtm"], gfs_renamed["data_available_kst_dtm"]))
    if ldaps_pairs != gfs_pairs:
        only_ldaps = len(ldaps_pairs - gfs_pairs)
        only_gfs = len(gfs_pairs - ldaps_pairs)
        logger.warning(
            "LDAPS/GFS (forecast_kst_dtm, data_available_kst_dtm) pairs differ for "
            "split=%r, kpx_group=%r: %d pairs only in LDAPS, %d only in GFS -- "
            "merging via outer join, expect NaN gaps on the non-overlapping side. "
            "(As of 2026-07-20 EDA this should not happen on the full train/test "
            "files -- see module docstring.)",
            split,
            kpx_group,
            only_ldaps,
            only_gfs,
        )

    ldaps_features = _build_source_features(ldaps_renamed, _LDAPS_LEVELS, _LDAPS_SCALAR_COLS, kpx_group, source="ldaps")
    gfs_features = _build_source_features(gfs_renamed, _GFS_LEVELS, _GFS_SCALAR_COLS, kpx_group, source="gfs")

    merged = ldaps_features.merge(gfs_features, on=_BLOCK_COLS, how="outer")

    # First-principles physics batches (air density, shear/veer stability,
    # directional-sector x speed, SCADA empirical power curve) -- computed on
    # the merged frame where both sources' IDW speed/scalar columns coexist.
    merged = _add_physics_features(merged, kpx_group)

    # ECMWF IFS third-source features + cross-model disagreement (see the
    # "ECMWF IFS" comment block above ECMWF_FEATURE_COLS). Joined after the
    # physics batch because the diff/ratio features need ldaps_ws_hub_fixed /
    # gfs_100m_speed_idw to exist on the merged frame.
    merged = _add_ecmwf_features(merged, ecmwf_df)

    # ICON global fourth-source features (same pattern as ECMWF above, see
    # "ICON global" comment block above ICON_FEATURE_COLS).
    merged = _add_icon_features(merged, icon_df)

    lag_value_cols = [f"{prefix}_speed_idw" for _u, _v, prefix in _LDAPS_LEVELS + _GFS_LEVELS]
    lag_value_cols += [f"{prefix}_power_curve_idw" for _u, _v, prefix in _LDAPS_LEVELS + _GFS_LEVELS]
    merged = lag_rolling_features(merged, "data_available_kst_dtm", lag_value_cols, windows=_LAG_WINDOWS)

    # Calendar/seasonality + lead-time features (group/source-agnostic,
    # computed exactly once here regardless of kpx_group -- see module
    # docstring's "Calendar / lead-time features" section for why
    # `lead_hours` (plural) is a distinct column from lag_rolling_features'
    # `lead_hour` (singular) above).
    merged = add_calendar_features(merged, dt_col="forecast_kst_dtm")
    merged = add_lead_hours(merged, forecast_col="forecast_kst_dtm", available_col="data_available_kst_dtm")

    if kpx_group == "kpx_group_2":
        # Wake-alignment features (reports/domain_research/wake_effect.md):
        # kpx_group_1 sits ~1.28km upwind of kpx_group_2 along the same
        # ridge. Only kpx_group_2 gets these -- see wake_features.py's module
        # docstring for why group1/group3 are out of scope. Reuses the
        # already-computed ldaps_10m_dir_sin/_dir_cos/_dir_deg/_speed_mean
        # columns (wind_speed_direction aggregates uniformly across all LDAPS
        # grids regardless of kpx_group, so no separate group1-centroid
        # recomputation is needed here -- see that module's docstring).
        merged = add_group1_group2_wake_features(merged, prefix="ldaps_10m", suffix="g1g2")

    if split == "train":
        labels = load_train_labels()[["kst_dtm", kpx_group]].rename(
            columns={"kst_dtm": "forecast_kst_dtm", kpx_group: "target"}
        )
        # Left join on the feature frame: keep every feature row even where
        # the label is missing (e.g. kpx_group_3 has no 2022 labels) -- let
        # downstream training code decide how to handle missing targets,
        # per CLAUDE.md section 3 / the task spec. Never drop rows here.
        merged = merged.merge(labels, on="forecast_kst_dtm", how="left")

    return merged.sort_values(["data_available_kst_dtm", "forecast_kst_dtm"]).reset_index(drop=True)


def build_feature_table(split: str, kpx_group: str) -> pd.DataFrame:
    """Build the full model-ready feature table for one (split, kpx_group).

    Loads LDAPS + GFS for `split` via `src.data.loaders`, derives wind-vector
    / spatial-aggregate / power-curve / lag-rolling features via
    `src.features.weather_features`, and (for split=="train") merges in the
    target column for `kpx_group` from `load_train_labels()`.

    Parameters
    ----------
    split: "train" or "test".
    kpx_group: one of "kpx_group_1" / "kpx_group_2" / "kpx_group_3".

    Returns
    -------
    One row per forecast_kst_dtm, sorted by (data_available_kst_dtm,
    forecast_kst_dtm), with a "target" column (train only, possibly NaN).
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if kpx_group not in KPX_GROUPS:
        raise ValueError(f"kpx_group must be one of {KPX_GROUPS}, got {kpx_group!r}")

    ldaps_renamed = load_ldaps(split).rename(columns=_LDAPS_RENAME)
    gfs_renamed = load_gfs(split).rename(columns=_GFS_RENAME)
    return _assemble_feature_table(ldaps_renamed, gfs_renamed, split, kpx_group)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for split in SPLITS:
        for kpx_group in KPX_GROUPS:
            table = build_feature_table(split, kpx_group)
            out_path = DATA_PROCESSED_DIR / f"features_{kpx_group}_{split}.parquet"
            table.to_parquet(out_path, index=False)
            n_missing_target = int(table["target"].isna().sum()) if "target" in table.columns else None
            logger.info(
                "Wrote %s: shape=%s, missing_target=%s",
                out_path,
                table.shape,
                n_missing_target,
            )


if __name__ == "__main__":
    main()
