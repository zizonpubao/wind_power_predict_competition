"""Unit tests for src/features/physics_features.py -- the first-principles
physics feature batch (air density, shear/veer, directional-sector speed,
grid dispersion, SCADA empirical power curve).

Focused on the properties that would silently corrupt a feature if wrong:
  - air density direction/units (cold+high-pressure => denser air),
  - power-in-wind scaling with rho and v**3,
  - shear/veer guards on degenerate (zero/negative) wind vectors,
  - sector-speed one-hot-of-speed partitioning (exactly one nonzero per row),
  - grid dispersion respecting the per-forecast-hour grouping,
  - empirical power curve monotonicity + normalisation + interpolation.
"""
import numpy as np
import pandas as pd
from pytest import approx

from src.features.physics_features import (
    air_density,
    apply_power_curve,
    density_ratio,
    fit_empirical_power_curve,
    grid_speed_dispersion,
    sector_speed_features,
    shear_exponent,
    veer_cos,
    wind_power_density,
    R_SPECIFIC_DRY_AIR,
    RHO_SEA_LEVEL,
)


# --- air density -----------------------------------------------------------


def test_air_density_ideal_gas_value():
    # 101325 Pa, 288.15 K -> ISA sea-level ~1.225 kg/m^3
    rho = air_density(101325.0, 288.15)
    assert float(rho) == approx(101325.0 / (R_SPECIFIC_DRY_AIR * 288.15), rel=1e-9)
    assert float(rho) == approx(1.225, abs=0.005)


def test_air_density_colder_is_denser():
    warm = air_density(90000.0, 290.0)
    cold = air_density(90000.0, 260.0)
    assert float(cold) > float(warm)


def test_air_density_series_preserves_index():
    sp = pd.Series([90000.0, 95000.0], index=[5, 9])
    t = pd.Series([260.0, 280.0], index=[5, 9])
    rho = air_density(sp, t)
    assert isinstance(rho, pd.Series)
    assert list(rho.index) == [5, 9]


def test_density_ratio_reference():
    assert float(density_ratio(RHO_SEA_LEVEL)) == approx(1.0)


# --- power in wind ---------------------------------------------------------


def test_wind_power_density_cubes_speed_and_scales_density():
    p1 = float(wind_power_density(1.2, 5.0))
    p2 = float(wind_power_density(1.2, 10.0))
    assert p2 == approx(p1 * 8.0, rel=1e-9)  # doubling speed -> 8x power
    p3 = float(wind_power_density(2.4, 5.0))
    assert p3 == approx(p1 * 2.0, rel=1e-9)  # doubling density -> 2x power
    assert p1 == approx(0.5 * 1.2 * 125.0, rel=1e-9)


def test_wind_power_density_clips_negative_speed():
    assert float(wind_power_density(1.2, -3.0)) == 0.0


# --- shear exponent --------------------------------------------------------


def test_shear_exponent_known_value():
    # v doubles from 10m to 100m -> alpha = ln2 / ln10
    alpha = shear_exponent(np.array([5.0]), 10.0, np.array([10.0]), 100.0)
    assert float(alpha[0]) == approx(np.log(2) / np.log(10), rel=1e-9)


def test_shear_exponent_nonpositive_returns_zero_not_nan():
    alpha = shear_exponent(np.array([0.0, -1.0, 4.0]), 10.0, np.array([5.0, 5.0, 8.0]), 100.0)
    assert np.isfinite(alpha).all()
    assert alpha[0] == 0.0 and alpha[1] == 0.0
    assert alpha[2] == approx(np.log(2) / np.log(10), rel=1e-9)


# --- veer ------------------------------------------------------------------


def test_veer_cos_same_direction_is_one():
    assert float(veer_cos(1.0, 2.0, 2.0, 4.0)) == approx(1.0, rel=1e-9)


def test_veer_cos_orthogonal_is_zero():
    assert float(veer_cos(1.0, 0.0, 0.0, 1.0)) == approx(0.0, abs=1e-12)


def test_veer_cos_zero_vector_guarded():
    assert float(veer_cos(0.0, 0.0, 1.0, 1.0)) == 0.0


# --- directional sector speed ---------------------------------------------


def test_sector_speed_exactly_one_nonzero_per_row():
    dir_deg = pd.Series([0.0, 44.0, 46.0, 200.0, 359.0])
    speed = pd.Series([3.0, 4.0, 5.0, 6.0, 7.0])
    out = sector_speed_features(dir_deg, speed, n_sectors=8, prefix="ldaps_10m")
    assert out.shape == (5, 8)
    # exactly one column nonzero per row, and it equals the speed
    nonzero = (out != 0).sum(axis=1)
    assert (nonzero == 1).all()
    assert out.to_numpy().sum(axis=1) == approx(speed.to_numpy())


def test_sector_speed_bins_are_half_open():
    # width 45: 1 deg -> sector 0 [0,45); 359 deg -> sector 7 [315,360);
    # 360 deg wraps back to sector 0 via the modulo.
    out = sector_speed_features(pd.Series([1.0, 359.0, 360.0]), pd.Series([2.0, 3.0, 4.0]), n_sectors=8)
    assert out["ldaps_10m_secspeed_0"].tolist() == [2.0, 0.0, 4.0]
    assert out["ldaps_10m_secspeed_7"].tolist() == [0.0, 3.0, 0.0]


# --- grid dispersion -------------------------------------------------------


def test_grid_speed_dispersion_per_hour():
    df = pd.DataFrame({
        "forecast_kst_dtm": pd.to_datetime(["2025-01-01 01:00"] * 3 + ["2025-01-01 02:00"] * 3),
        "data_available_kst_dtm": pd.to_datetime(["2024-12-31 13:00"] * 6),
        "grid_id": [1, 2, 3, 1, 2, 3],
        "u": [3.0, 0.0, 0.0, 1.0, 1.0, 1.0],  # speeds 3,0,0 then 1,1,1
        "v": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    out = grid_speed_dispersion(df, "u", "v", prefix="ldaps_10m")
    assert len(out) == 2
    h1 = out.iloc[0]
    assert h1["ldaps_10m_grid_range"] == approx(3.0)
    assert h1["ldaps_10m_grid_std"] == approx(np.std([3.0, 0.0, 0.0], ddof=1))
    h2 = out.iloc[1]
    assert h2["ldaps_10m_grid_range"] == approx(0.0)
    assert h2["ldaps_10m_grid_cv"] == approx(0.0)  # uniform -> zero CV


# --- SCADA empirical power curve ------------------------------------------


def test_fit_empirical_power_curve_monotone_and_normalised():
    rng = np.random.default_rng(0)
    ws = rng.uniform(0, 25, 20000)
    # synthetic S-curve: 0 below 3, cubic ramp to 12, flat after
    p = np.clip(((ws - 3) / 9) ** 3, 0, 1) * 600.0
    curve = fit_empirical_power_curve(ws, p)
    vals = np.asarray(curve["values"])
    assert np.all(np.diff(vals) >= -1e-9)  # monotone nondecreasing
    assert vals.max() == approx(1.0)
    assert vals.min() == approx(0.0, abs=1e-9)
    # applied to a low speed -> ~0, high speed -> ~1
    assert float(apply_power_curve(curve, 1.0)) < 0.05
    assert float(apply_power_curve(curve, 20.0)) > 0.95


def test_fit_empirical_power_curve_masks_glitches():
    # inject huge sentinel power values; median-per-bin + |value|<=700 mask
    # must keep the curve bounded/sane despite them
    ws = np.concatenate([np.full(100, 10.0), np.full(5, 10.0)])
    p = np.concatenate([np.full(100, 300.0), np.full(5, 5e7)])
    curve = fit_empirical_power_curve(ws, p)
    assert np.isfinite(curve["values"]).all()
    assert max(curve["values"]) == approx(1.0)


def test_apply_power_curve_flat_extrapolates():
    curve = {"centers": [0.0, 5.0, 10.0], "values": [0.0, 0.5, 1.0]}
    assert float(apply_power_curve(curve, -2.0)) == approx(0.0)
    assert float(apply_power_curve(curve, 50.0)) == approx(1.0)
    assert float(apply_power_curve(curve, 2.5)) == approx(0.25)
