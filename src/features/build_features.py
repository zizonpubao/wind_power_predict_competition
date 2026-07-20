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
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from configs.paths import DATA_PROCESSED_DIR
from src.data.loaders import load_gfs, load_ldaps, load_train_labels
from src.features.weather_features import (
    KPX_GROUP_TURBINE_MODEL,
    TURBINE_POWER_CURVE_PARAMS,
    lag_rolling_features,
    power_curve_transform,
    spatial_aggregate,
    wind_speed_direction,
)

logger = logging.getLogger(__name__)

_BLOCK_COLS = ["forecast_kst_dtm", "data_available_kst_dtm"]

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
_GFS_SCALAR_COLS = ["gfs_dswrf", "gfs_tcc"]

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


def _build_source_features(
    df: pd.DataFrame,
    levels: list[tuple[str, str, str]],
    scalar_cols: list[str],
    kpx_group: str,
) -> pd.DataFrame:
    """One row per forecast_kst_dtm: spatial_aggregate (IDW/nearest) over every
    level's u/v pair plus the source's scalar columns, IDW wind speed +
    power-curve output per level, and wind_speed_direction's grid-mean
    speed/circular-mean direction per level -- all merged on _BLOCK_COLS.
    """
    value_cols = [col for u, v, _ in levels for col in (u, v)] + scalar_cols
    spatial = spatial_aggregate(df, kpx_group, value_cols)
    spatial = _idw_speed_and_power_curve(spatial, levels, kpx_group)

    merged = spatial
    for u_col, v_col, prefix in levels:
        wsd = wind_speed_direction(df, u_col, v_col, prefix=prefix)
        merged = merged.merge(wsd, on=_BLOCK_COLS, how="outer")
    return merged


def _assemble_feature_table(
    ldaps_renamed: pd.DataFrame,
    gfs_renamed: pd.DataFrame,
    split: str,
    kpx_group: str,
) -> pd.DataFrame:
    """Core assembly logic, factored out of `build_feature_table` so tests can
    feed it a small time-sliced subset of already-loaded/renamed LDAPS/GFS
    frames instead of re-reading the full CSVs on every test run.
    """
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

    ldaps_features = _build_source_features(ldaps_renamed, _LDAPS_LEVELS, _LDAPS_SCALAR_COLS, kpx_group)
    gfs_features = _build_source_features(gfs_renamed, _GFS_LEVELS, _GFS_SCALAR_COLS, kpx_group)

    merged = ldaps_features.merge(gfs_features, on=_BLOCK_COLS, how="outer")

    lag_value_cols = [f"{prefix}_speed_idw" for _u, _v, prefix in _LDAPS_LEVELS + _GFS_LEVELS]
    lag_value_cols += [f"{prefix}_power_curve_idw" for _u, _v, prefix in _LDAPS_LEVELS + _GFS_LEVELS]
    merged = lag_rolling_features(merged, "data_available_kst_dtm", lag_value_cols, windows=_LAG_WINDOWS)

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
