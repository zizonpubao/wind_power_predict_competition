"""Turbine wake (후류) alignment features for the kpx_group_1 -> kpx_group_2
inter-farm relationship.

Motivation (see ``reports/domain_research/wake_effect.md`` for the full
derivation, sourcing, and caveats): ``kpx_group_1`` (VESTAS 1-6) sits ~1.28km
upwind of ``kpx_group_2`` (VESTAS 7-12) along the same ridge, at geographic
bearing 115 deg (group1 centroid -> group2 centroid). Two independent
code-writer experiments (OOF-isotonic calibration, asymmetric loss tuning)
found group2's residual bias behaves oppositely to group1/group3's, which
motivated this wake hypothesis. These features are a **directional-alignment
proxy**, not a physically exact wake-deficit calculation -- the domain report
is explicit that LDAPS's ~1.5km grid spacing is larger than the 1.28km
group1-group2 distance, so the raw forecast fields cannot represent this
sub-grid wake phenomenon directly. The model is left to learn whatever
correlation exists between "alignment with the wake axis" and group2's
residual, rather than us hand-tuning a exact Jensen-model deficit.

Because ``src.features.weather_features.wind_speed_direction`` aggregates
*every* grid uniformly (a plain circular mean across all LDAPS grids, not an
IDW-toward-one-group's-centroid computation -- see its docstring), its output
columns (``ldaps_10m_dir_sin`` / ``_dir_cos`` / ``_dir_deg`` /
``ldaps_10m_speed_mean``) are **identical regardless of which kpx_group's
feature table they end up in**. The wake_effect.md report's own section 4
argues this is fine here specifically because group1 and group2 are closer to
each other (1.28km) than one LDAPS grid cell (~1.5km) -- i.e. "group1's
aggregated wind" and "group2's aggregated wind" are, in this dataset, the same
number. So callers should NOT re-run spatial_aggregate toward kpx_group_1's
centroid separately; simply feed this module the kpx_group_2 feature table's
own already-computed ``ldaps_10m_dir_sin``/``_dir_cos``/``_dir_deg``/
``_speed_mean`` columns (see ``add_group1_group2_wake_features``).

Only kpx_group_2 gets these features -- group1 is the upwind farm (nothing
wakes it in this pairing) and group3 is a different turbine model/location
not covered by this specific geometric relationship (the report flags a
group2->group3 bearing as a distinct, *unverified* extension hypothesis, out
of scope here).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Geometric / physical constants (see reports/domain_research/wake_effect.md
# section 5 for the full derivation). All are time-invariant (pure geography)
# except WAKE_SECTOR_HALF_WIDTH_DEG and WAKE_DEFICIT_FRAC_CONST, which are
# literature-based approximations explicitly flagged as tunable once SCADA-
# based empirical recalibration is done (see that report's section 5 closing
# paragraph and "미확인" section).
# ---------------------------------------------------------------------------

#: Geographic bearing from kpx_group_1's turbine centroid to kpx_group_2's
#: turbine centroid (degrees, 0=N/90=E), independently re-derived from
#: docs/info_raw.csv coordinates in wake_effect.md section 1 and confirmed to
#: match this exact value.
BEARING_G1_TO_G2_DEG = 115.0

#: The meteorological wind direction ("blowing from") that aligns group1's
#: wake plume with group2 -- the reciprocal of the geographic bearing above.
#: wind_speed_direction's dir_deg convention is "direction wind blows FROM",
#: so group1 only wakes group2 when wind blows from 295 deg (not 115 deg).
WAKE_FROM_DIR_DEG = (BEARING_G1_TO_G2_DEG + 180.0) % 360.0  # = 295.0

#: Centroid-to-centroid distance (km), independently re-derived in
#: wake_effect.md section 1.
DIST_G1_G2_KM = 1.28

#: Half-width (degrees) of the wind-direction sector treated as "wake-exposed"
#: around WAKE_FROM_DIR_DEG -- literature range is +-15 to 30 deg (inter-farm
#: wake screening / IEC 61400-12-1 wake-affected-sector practice, wake_effect.md
#: section 2); 25.0 is the report's proposed midpoint default. Exposed here as
#: a named module constant (rather than buried in a function default) so it is
#: easy to find and override once/if scada-based recalibration narrows it.
WAKE_SECTOR_HALF_WIDTH_DEG = 25.0

#: Approximate fractional wind-speed deficit applied when wake alignment is
#: perfect (wake_alignment_cos_g1g2 == 1), from wake_effect.md section 2's
#: Jensen/PARK-model back-of-envelope calculation (Ct=0.8, k=0.075 "typical
#: onshore" decay constant, x/D=10.2 for this site's fixed 1.28km/126m
#: geometry -> deficit ~8.7%; this constant uses the more conservative 0.075
#: rather than the 8.7% result itself, matching the task's exact requested
#: value). This is a coarse, literature-approximate constant, NOT a
#: site-calibrated Ct/k fit -- see wake_effect.md's "미확인" section.
WAKE_DEFICIT_FRAC_CONST = 0.075

_WAKE_FROM_DIR_RAD = math.radians(WAKE_FROM_DIR_DEG)
_COS_WAKE_FROM_DIR = math.cos(_WAKE_FROM_DIR_RAD)
_SIN_WAKE_FROM_DIR = math.sin(_WAKE_FROM_DIR_RAD)


def wake_alignment_cos(dir_sin, dir_cos, wake_from_dir_deg: float = WAKE_FROM_DIR_DEG):
    """cos(dir_deg - wake_from_dir_deg) via the angle-difference trig identity,
    computed directly from already-available dir_sin/dir_cos components (no
    atan2 round-trip needed): ``cos(a-b) = cos(a)cos(b) + sin(a)sin(b)``.

    Returns a value in [-1, 1]: +1 when the wind blows exactly from
    ``wake_from_dir_deg`` (group1 directly upwind of group2, maximal wake
    alignment), -1 when it blows from the exact opposite direction (group2 is
    upwind of group1 instead -- no wake possible), 0 for a crosswind.
    """
    if wake_from_dir_deg == WAKE_FROM_DIR_DEG:
        cos_b, sin_b = _COS_WAKE_FROM_DIR, _SIN_WAKE_FROM_DIR
    else:
        rad = math.radians(wake_from_dir_deg)
        cos_b, sin_b = math.cos(rad), math.sin(rad)
    return np.asarray(dir_cos) * cos_b + np.asarray(dir_sin) * sin_b


def wake_sector_exposure(
    dir_deg,
    wake_from_dir_deg: float = WAKE_FROM_DIR_DEG,
    half_width_deg: float = WAKE_SECTOR_HALF_WIDTH_DEG,
):
    """Ramp membership in the +-``half_width_deg`` sector around
    ``wake_from_dir_deg``: 1.0 at perfect alignment, linearly down to 0.0 at
    ``angle_diff >= half_width_deg``, clipped at 0 beyond that (never
    negative). A smooth ramp rather than a hard 0/1 cutoff, so a tree model
    doesn't have to relearn a sharp boundary discontinuity.

    ``angle_diff`` is the shortest angular distance on the compass circle
    (correctly handles wrap-around, e.g. dir_deg=359 vs wake_from_dir_deg=295
    should not be treated as a huge 64+296 deg difference).
    """
    dir_deg = np.asarray(dir_deg, dtype=float)
    raw_diff = np.abs(dir_deg - wake_from_dir_deg)
    angle_diff = np.minimum(raw_diff, 360.0 - raw_diff)
    return np.maximum(0.0, 1.0 - angle_diff / half_width_deg)


def wake_deficit_proxy(
    speed_mean,
    alignment_cos,
    deficit_frac_const: float = WAKE_DEFICIT_FRAC_CONST,
):
    """Jensen-model-inspired scalar proxy (m/s) for the wind-speed deficit
    group2 is expected to see from group1's wake: upwind speed scaled by the
    deficit fraction constant and by ``max(0, alignment_cos)``.

    The ``max(0, ...)`` clip on alignment is physically required, not just a
    numerical nicety: when ``alignment_cos < 0`` the wind is blowing group2's
    way *toward* group1 instead (group2 is upwind), so no wake deficit exists
    and the proxy must be exactly 0, not a small negative number.
    """
    alignment_cos = np.asarray(alignment_cos, dtype=float)
    return np.asarray(speed_mean, dtype=float) * np.clip(alignment_cos, 0.0, None) * deficit_frac_const


def add_group1_group2_wake_features(
    df: pd.DataFrame,
    prefix: str = "ldaps_10m",
    suffix: str = "g1g2",
) -> pd.DataFrame:
    """Add the 3 wake-alignment features to a kpx_group_2 feature table
    in-place (mutates and returns ``df``, matching the mutate-and-return style
    ``build_features._idw_speed_and_power_curve`` already uses).

    Reads ``{prefix}_dir_sin`` / ``{prefix}_dir_cos`` / ``{prefix}_dir_deg`` /
    ``{prefix}_speed_mean`` (default prefix "ldaps_10m", the project's only
    true instantaneous LDAPS wind vector -- see weather_features.py's module
    docstring) and writes:
      - ``wake_alignment_cos_{suffix}``
      - ``wake_sector_exposure_{suffix}``
      - ``wake_deficit_proxy_{suffix}``

    Callers should apply this only to kpx_group_2's feature table -- see this
    module's docstring for why kpx_group_1/3 are out of scope.
    """
    required = [f"{prefix}_dir_sin", f"{prefix}_dir_cos", f"{prefix}_dir_deg", f"{prefix}_speed_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_group1_group2_wake_features: missing required column(s) {missing} in df")

    alignment_cos = wake_alignment_cos(df[f"{prefix}_dir_sin"], df[f"{prefix}_dir_cos"])
    df[f"wake_alignment_cos_{suffix}"] = alignment_cos
    df[f"wake_sector_exposure_{suffix}"] = wake_sector_exposure(df[f"{prefix}_dir_deg"])
    df[f"wake_deficit_proxy_{suffix}"] = wake_deficit_proxy(df[f"{prefix}_speed_mean"], alignment_cos)
    return df
