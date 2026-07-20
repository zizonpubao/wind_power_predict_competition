"""Weather-derived feature functions for LDAPS/GFS forecast dataframes.

Every function here takes a *long-format* LDAPS or GFS dataframe (one row per
``forecast_kst_dtm`` x ``grid_id``, as returned by ``src.data.loaders.load_ldaps``
/ ``load_gfs``) and returns a dataframe **already aggregated across grids**,
keyed by ``forecast_kst_dtm`` (with ``data_available_kst_dtm`` carried through
as an attribute column, since it is constant within one ``forecast_kst_dtm``
and downstream code -- e.g. ``lag_rolling_features`` -- needs it as the
leakage-boundary group key). None of these functions return per-grid columns;
use ``src.data.loaders.pivot_grid_wide`` if raw per-grid columns are ever
wanted instead.

Domain grounding (see CLAUDE.md sections 3-4 and reports/eda/eda_summary.md):
  - LDAPS (16 grids, ~1.5km) has much stronger wind-speed/power correlation
    than GFS (9 grids, ~0.25 deg): r=0.73-0.74 vs 0.54-0.55. Treat LDAPS as the
    primary feature source and GFS as complementary/ensemble-diversity input.
  - The `data_available_kst_dtm` block boundary is the project's core leakage
    boundary -- a 24-row bundle of forecast hours sharing one issuance time.
    `lag_rolling_features` below must never let a rolling window reach across
    into a different block.
  - Weather files (LDAPS/GFS, train+test) have zero missingness in the raw
    data (EDA section 2.2), so none of these functions need to handle NaN
    forecast inputs as a normal case -- only defensive edge cases (e.g. a
    degenerate all-cancelling wind-direction vector) are guarded.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from configs.paths import PROJECT_ROOT

_INFO_RAW_CSV = PROJECT_ROOT / "docs" / "info_raw.csv"

# The two columns that uniquely key a forecast row within one LDAPS/GFS split
# (see CLAUDE.md section 3: forecast_kst_dtm alone happens to be unique too,
# since announcement windows are issued daily and do not overlap, but we
# carry data_available_kst_dtm through anyway -- lag_rolling_features and any
# leakage-boundary logic downstream needs it).
_BLOCK_COLS = ["forecast_kst_dtm", "data_available_kst_dtm"]

# Matches Google-style DMS coordinate strings, e.g.
# 37°16'55.61"N 128°57'02.10"E
_DMS_RE = re.compile(
    r"(?P<d1>\d+)\D+(?P<m1>\d+)\D+(?P<s1>[\d.]+)\D*(?P<hemi1>[NS])\s+"
    r"(?P<d2>\d+)\D+(?P<m2>\d+)\D+(?P<s2>[\d.]+)\D*(?P<hemi2>[EW])"
)

_EARTH_RADIUS_KM = 6371.0088


# ---------------------------------------------------------------------------
# 1. Wind speed / direction (circular-mean safe)
# ---------------------------------------------------------------------------


def wind_speed_direction(df: pd.DataFrame, u_col: str, v_col: str, prefix: str) -> pd.DataFrame:
    """Decompose a U/V wind-component pair into speed + direction per grid row,
    then aggregate across grids to one row per forecast hour.

    Meteorological convention: direction is the compass bearing the wind is
    blowing *from* (0/360=N, 90=E, 180=S, 270=W):
    ``dir_deg = (180 + degrees(atan2(u, v))) % 360``.

    Circular-mean safe by construction: directions are never averaged as raw
    degrees (350 deg and 10 deg would wrongly average to ~180 deg instead of
    ~0 deg). Instead we average sin/cos of each grid's direction across grids
    and recover the mean angle via atan2 -- the standard circular-mean
    construction, equivalent here to vector-averaging the underlying
    (normalized) U/V components.

    Works generically for any U/V pair present in a source -- LDAPS only has a
    clean vector pair at 10m (``heightAboveGround_10_10u/10v``; its 50m fields
    are max/min envelopes, not a true instantaneous vector, so callers should
    not feed those in here), GFS has 10m/80m/100m and planetary-boundary-layer
    pairs.

    Parameters
    ----------
    df: long-format LDAPS/GFS dataframe (one row per forecast_kst_dtm x grid_id).
    u_col, v_col: column names of the U/V wind components to use.
    prefix: feature-name prefix, e.g. "ldaps_10m" or "gfs_pbl".

    Returns
    -------
    One row per forecast_kst_dtm with:
      - data_available_kst_dtm (carried through)
      - {prefix}_speed_mean: grid-mean wind speed (m/s)
      - {prefix}_dir_sin / {prefix}_dir_cos: sin/cos of the circular-mean
        direction -- the ML-friendly encoding, safe to average/interpolate
        further downstream.
      - {prefix}_dir_deg: circular-mean direction in degrees [0, 360), kept
        for interpretability only -- never re-average this column directly.
    """
    work = df[_BLOCK_COLS + [u_col, v_col]].copy()
    work["_speed"] = np.sqrt(work[u_col] ** 2 + work[v_col] ** 2)
    # (180 + atan2(u, v)) in radians; sin/cos below are periodic so no
    # explicit wrap-to-[0, 2*pi) is needed before aggregating.
    dir_rad = np.arctan2(work[u_col], work[v_col]) + np.pi
    work["_dir_sin"] = np.sin(dir_rad)
    work["_dir_cos"] = np.cos(dir_rad)

    grouped = work.groupby(_BLOCK_COLS, as_index=False).agg(
        **{
            f"{prefix}_speed_mean": ("_speed", "mean"),
            "_sin_mean": ("_dir_sin", "mean"),
            "_cos_mean": ("_dir_cos", "mean"),
        }
    )
    resultant_len = np.sqrt(grouped["_sin_mean"] ** 2 + grouped["_cos_mean"] ** 2)
    # Guard the (very unlikely) degenerate case where grid directions cancel
    # out exactly (resultant length ~0, e.g. two grids pointing dead-opposite
    # with equal weight) -- direction is genuinely undefined there.
    safe_len = resultant_len.replace(0.0, np.nan)
    grouped[f"{prefix}_dir_sin"] = grouped["_sin_mean"] / safe_len
    grouped[f"{prefix}_dir_cos"] = grouped["_cos_mean"] / safe_len
    grouped[f"{prefix}_dir_deg"] = np.degrees(
        np.arctan2(grouped["_sin_mean"], grouped["_cos_mean"])
    ) % 360

    grouped = grouped.drop(columns=["_sin_mean", "_cos_mean"])
    return grouped.sort_values("forecast_kst_dtm").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Spatial aggregation toward a KPX group's turbine centroid
# ---------------------------------------------------------------------------


def _dms_to_decimal(deg, minute, second, hemisphere: str) -> float:
    value = float(deg) + float(minute) / 60.0 + float(second) / 3600.0
    if hemisphere in ("S", "W"):
        value = -value
    return value


def _parse_google_coord(coord_str: str) -> tuple[float, float]:
    """Parse a Google-style DMS coordinate string, e.g.
    ``37°16'55.61"N 128°57'02.10"E`` -> (37.28211..., 128.95058...) decimal
    degrees (lat, lon).
    """
    match = _DMS_RE.search(str(coord_str))
    if not match:
        raise ValueError(f"Could not parse coordinate string: {coord_str!r}")
    lat = _dms_to_decimal(match["d1"], match["m1"], match["s1"], match["hemi1"])
    lon = _dms_to_decimal(match["d2"], match["m2"], match["s2"], match["hemi2"])
    return lat, lon


def _load_turbine_coords() -> pd.DataFrame:
    """Load per-turbine decimal lat/lon + kpx_group from docs/info_raw.csv.

    info_raw.csv carries a stray literal first header row ("Unnamed: 0"..
    "Unnamed: 11", an artifact of however it was re-saved from info.xlsx) --
    the real Korean headers are on the *second* line, so we read with
    header=1. See docs/turbine_kpx_mapping.md for the turbine <-> kpx_group
    assignment this reproduces (VESTAS 1-6 -> group_1, VESTAS 7-12 ->
    group_2, UNISON 1-5 -> group_3).
    """
    raw = pd.read_csv(_INFO_RAW_CSV, encoding="utf-8-sig", header=1)
    manufacturer_col = "제작사"
    unit_col = "호기"
    coord_col = "좌표(Google)"

    coords = raw[coord_col].map(_parse_google_coord)
    raw = raw.assign(
        _lat=[c[0] for c in coords],
        _lon=[c[1] for c in coords],
    )

    def _kpx_group(row) -> str | None:
        mfr = row[manufacturer_col]
        unit = int(row[unit_col])
        if mfr == "VESTAS" and 1 <= unit <= 6:
            return "kpx_group_1"
        if mfr == "VESTAS" and 7 <= unit <= 12:
            return "kpx_group_2"
        if mfr == "UNISON" and 1 <= unit <= 5:
            return "kpx_group_3"
        return None

    raw["kpx_group"] = raw.apply(_kpx_group, axis=1)
    if raw["kpx_group"].isna().any():
        unmapped = raw.loc[raw["kpx_group"].isna(), [manufacturer_col, unit_col]]
        raise ValueError(f"Unmapped turbine rows in info_raw.csv:\n{unmapped}")

    out = raw[["kpx_group", "_lat", "_lon"]].rename(
        columns={"_lat": "latitude", "_lon": "longitude"}
    )
    return out.reset_index(drop=True)


def _group_centroid(kpx_group: str) -> tuple[float, float]:
    """Turbine-coordinate centroid (mean lat, mean lon) for a kpx_group,
    parsed from docs/info_raw.csv's Google-coordinate column.
    """
    coords = _load_turbine_coords()
    group_coords = coords[coords["kpx_group"] == kpx_group]
    if group_coords.empty:
        raise ValueError(f"No turbines found for kpx_group={kpx_group!r}")
    return group_coords["latitude"].mean(), group_coords["longitude"].mean()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def spatial_aggregate(
    df: pd.DataFrame,
    kpx_group: str,
    value_cols: list[str],
    power: float = 2.0,
) -> pd.DataFrame:
    """Reduce LDAPS/GFS grid rows to a single per-KPX-group series via
    inverse-distance-weighted (IDW) averaging toward the group's turbine
    centroid, plus the single nearest grid's raw value as an alternative
    feature.

    Grid coordinates are always taken from whatever `df` carries (its own
    ``latitude``/``longitude`` columns) rather than assumed -- LDAPS (16
    grids, ~1.5km) and GFS (9 grids, ~0.25 deg) have entirely different grid
    layouts and grid_id numbering.

    Parameters
    ----------
    df: long-format LDAPS/GFS dataframe (one row per forecast_kst_dtm x grid_id).
    kpx_group: "kpx_group_1" / "kpx_group_2" / "kpx_group_3".
    value_cols: weather columns in `df` to aggregate.
    power: IDW distance exponent (default 2, the standard inverse-square
        weighting).

    Returns
    -------
    One row per forecast_kst_dtm with, for every column in value_cols:
      - {col}_idw: inverse-distance-weighted mean across grids
        (weight = 1 / distance_km ** power)
      - {col}_nearest: the value at whichever single grid is geographically
        closest to the group centroid (the same grid for every timestamp,
        since grid locations are static within one LDAPS/GFS source).
    """
    centroid_lat, centroid_lon = _group_centroid(kpx_group)

    grid_coords = (
        df[["grid_id", "latitude", "longitude"]].drop_duplicates("grid_id").set_index("grid_id")
    )
    distances_km = grid_coords.apply(
        lambda r: _haversine_km(r["latitude"], r["longitude"], centroid_lat, centroid_lon),
        axis=1,
    )
    # Guard against a (very unlikely) zero-distance grid point blowing up 1/d**power.
    weights = 1.0 / (distances_km.clip(lower=1e-6) ** power)
    nearest_grid_id = distances_km.idxmin()

    work = df[_BLOCK_COLS + ["grid_id"] + value_cols].copy()
    work["_weight"] = work["grid_id"].map(weights)
    weighted_cols = {}
    for col in value_cols:
        wcol = f"_{col}_w"
        work[wcol] = work[col] * work["_weight"]
        weighted_cols[col] = wcol

    agg_spec = {f"{col}_wsum": (weighted_cols[col], "sum") for col in value_cols}
    agg_spec["_weight_sum"] = ("_weight", "sum")
    grouped = work.groupby(_BLOCK_COLS, as_index=False).agg(**agg_spec)

    for col in value_cols:
        grouped[f"{col}_idw"] = grouped[f"{col}_wsum"] / grouped["_weight_sum"]
    grouped = grouped[_BLOCK_COLS + [f"{col}_idw" for col in value_cols]]

    nearest = df.loc[df["grid_id"] == nearest_grid_id, _BLOCK_COLS + value_cols].rename(
        columns={col: f"{col}_nearest" for col in value_cols}
    )

    result = grouped.merge(nearest, on=_BLOCK_COLS, how="left")
    return result.sort_values("forecast_kst_dtm").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Power-curve-shaped transform
# ---------------------------------------------------------------------------


def power_curve_transform(
    wind_speed: pd.Series,
    cut_in: float = 3.0,
    rated: float = 12.0,
    cut_out: float = 25.0,
) -> pd.Series:
    """Monotonic nonlinear transform of wind speed shaped like a generic
    utility-scale wind-turbine power curve, normalized to [0, 1]:
      - 0 below cut_in (too little wind to turn the rotor)
      - a roughly cubic ramp from 0 to 1 between cut_in and rated (power in
        the wind scales with v**3, so this approximates -- without
        replicating exactly -- the physical torque/output ramp up to the
        generator's rated limit)
      - flat at 1.0 between rated and cut_out (generator-limited, output
        capped regardless of extra wind)
      - 0 above cut_out (the turbine feathers/shuts down to protect itself)

    # TODO: replace defaults once reports/domain_research/turbine_power_curves.md
    # lands. cut_in=3.0 / rated=12.0 / cut_out=25.0 (m/s) are generic
    # class-typical placeholders, NOT the confirmed VESTAS V126 / UNISON U136
    # spec -- domain-researcher is investigating the manufacturer curves in
    # parallel. cut_in/rated/cut_out are exposed as named parameters
    # specifically so the confirmed values can be swapped in per-group
    # without touching this function's body (V126 and U136 will likely need
    # different values, per EDA section 5's group3-vs-group1/2 power-curve
    # note in reports/eda/eda_summary.md).

    Parameters
    ----------
    wind_speed: wind speed in m/s.
    cut_in, rated, cut_out: turbine power-curve breakpoints in m/s.

    Returns
    -------
    A pd.Series of the same length/index as `wind_speed` (if it is one),
    normalized power-curve output in [0, 1].
    """
    index = wind_speed.index if isinstance(wind_speed, pd.Series) else None
    v = np.asarray(wind_speed, dtype=float)

    ramp = np.clip((v - cut_in) / (rated - cut_in), 0.0, 1.0) ** 3
    out = np.select(
        condlist=[v < cut_in, v <= rated, v <= cut_out],
        choicelist=[0.0, ramp, 1.0],
        default=0.0,
    )
    return pd.Series(out, index=index, name="power_curve")


# ---------------------------------------------------------------------------
# 4. Lag / rolling features within a data_available_kst_dtm block
# ---------------------------------------------------------------------------


def lag_rolling_features(
    df: pd.DataFrame,
    group_key: str,
    value_cols: list[str],
    windows: list[int] = (3, 6, 12, 24),
) -> pd.DataFrame:
    """Rolling mean/std of value_cols, computed strictly *within* each
    `group_key` block (typically `data_available_kst_dtm`, the 24-row bundle
    of forecast hours sharing one issuance time), ordered by
    `forecast_kst_dtm` -- never across into a different block. This is the
    project's core leakage boundary (CLAUDE.md section 3): a rolling window
    must not reach into a different announcement's forecasts.

    Also adds `lead_hour`: the 1-indexed position of each row within its
    block (1..24 for a full block), a cheap proxy for forecast lead time.

    Windows are trailing with `min_periods=1`, so the first row(s) of a block
    use whatever history is available *within that block only* -- e.g. the
    3h-rolling-mean for lead_hour=1 is just that single row's value, not an
    average that reaches back into yesterday's announcement.

    Parameters
    ----------
    df: any dataframe with `group_key`, `forecast_kst_dtm`, and value_cols
        (e.g. the output of wind_speed_direction / spatial_aggregate).
    group_key: the leakage-boundary column, e.g. "data_available_kst_dtm".
    value_cols: columns to compute rolling stats for.
    windows: window sizes in hours (== rows, since one row is one forecast hour).

    Returns
    -------
    `df` sorted by [group_key, forecast_kst_dtm], with `lead_hour` and
    `{col}_roll_mean_{w}h` / `{col}_roll_std_{w}h` columns added for every
    value_col x window combination.
    """
    out = df.sort_values([group_key, "forecast_kst_dtm"]).reset_index(drop=True)
    out["lead_hour"] = out.groupby(group_key).cumcount() + 1

    grouped = out.groupby(group_key, sort=False)
    for col in value_cols:
        for w in windows:
            out[f"{col}_roll_mean_{w}h"] = grouped[col].transform(
                lambda s, w=w: s.rolling(window=w, min_periods=1).mean()
            )
            out[f"{col}_roll_std_{w}h"] = grouped[col].transform(
                lambda s, w=w: s.rolling(window=w, min_periods=1).std()
            )
    return out
