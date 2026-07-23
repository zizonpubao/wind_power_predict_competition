"""Power-law wind-shear extrapolation to hub height (117m).

Pure functions only -- no I/O, no knowledge of LDAPS/GFS column names (that
wiring lives in ``src/features/build_features.py``). Motivated by the v14
pipeline reproduction plan
(``C:\\Users\\heelo\\.claude\\plans\\logical-stirring-sphinx.md``, Phase A):
LDAPS only has a clean instantaneous vector at 10m (see
``weather_features.py``'s module docstring -- its 50m fields are max/min
envelopes, not usable here either), so it can only be extrapolated to hub
height with an assumed fixed shear exponent. GFS has both 80m and 100m clean
levels, so its exponent can instead be *estimated per row from the data
itself* rather than assumed -- a strictly better-grounded extrapolation that
LDAPS's single level cannot support.

Both functions accept scalars, numpy arrays, or ``pd.Series`` for their
value arguments and return a ``pd.Series`` (matching the input's index) if
any Series was passed, else a plain ``np.ndarray`` -- mirroring the calling
convention already used by ``weather_features.power_curve_transform``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: VESTAS V126 / UNISON U136 hub height (both confirmed identical). See
#: docs/turbine_kpx_mapping.md and reports/domain_research/turbine_power_curves.md
#: (Vestas' own "4 MW platform" brochure lists 117m as one of V126's stock
#: hub-height options, matching the project's spec exactly; UNISON's U136
#: product page confirms 117m directly too).
HUB_HEIGHT_M = 117.0

#: Hellman/power-law standard neutral-atmosphere approximation (1/7).
#: This is a generic textbook constant, NOT a measured or site-calibrated
#: shear exponent -- it is only used where no better per-row estimate is
#: available (LDAPS's single-level case), or as GFS's fixed-alpha comparison
#: column (see estimate_shear_exponent for the data-driven alternative).
DEFAULT_SHEAR_EXPONENT = 1.0 / 7.0


def _wrap_like_input(result: np.ndarray, *inputs) -> pd.Series | np.ndarray:
    """Return `result` as a pd.Series (index taken from the first Series
    found among `inputs`) if any of `inputs` is a pd.Series, else as-is.
    """
    for value in inputs:
        if isinstance(value, pd.Series):
            return pd.Series(result, index=value.index)
    return result


def power_law_extrapolate(
    v_ref,
    h_ref: float,
    h_target: float = HUB_HEIGHT_M,
    alpha=DEFAULT_SHEAR_EXPONENT,
):
    """Power-law wind-speed extrapolation: ``v_target = v_ref * (h_target/h_ref)**alpha``.

    Parameters
    ----------
    v_ref: reference-height wind speed (m/s). Scalar, array-like, or
        pd.Series.
    h_ref: reference height (m) that `v_ref` was measured/computed at.
    h_target: target height (m) to extrapolate to (default: hub height,
        117m).
    alpha: shear exponent. Scalar (e.g. DEFAULT_SHEAR_EXPONENT) or a
        per-row array-like/pd.Series (e.g. estimate_shear_exponent's output,
        already fillna'd by the caller) -- broadcasting against `v_ref` is
        left to numpy.

    Returns
    -------
    pd.Series (matching whichever of v_ref/alpha is a Series, v_ref taking
    priority) if either input is a Series, else np.ndarray.
    """
    v_ref_arr = np.asarray(v_ref, dtype=float)
    alpha_arr = np.asarray(alpha, dtype=float)
    ratio = (float(h_target) / float(h_ref)) ** alpha_arr
    result = v_ref_arr * ratio
    return _wrap_like_input(result, v_ref, alpha)


def estimate_shear_exponent(v_low, h_low: float, v_high, h_high: float):
    """Row-wise shear exponent estimated from two observed levels:
    ``alpha = ln(v_high/v_low) / ln(h_high/h_low)``.

    Guards
    ------
    - Any row where ``v_low <= 0`` or ``v_high <= 0`` -> NaN for that row
      (log of a non-positive value is undefined) instead of raising --
      callers should ``.fillna(DEFAULT_SHEAR_EXPONENT)`` to fall back
      per-row rather than dropping rows wholesale.
    - ``h_high == h_low`` -> raises ``ValueError`` upfront (``ln(h_high/h_low)
      == 0`` would divide every row by zero -- this is a caller bug, not a
      per-row data issue, so it is not silently NaN'd like the v_low/v_high
      guard above).

    Returns
    -------
    pd.Series (matching whichever of v_low/v_high is a Series, v_high taking
    priority) if either input is a Series, else np.ndarray.
    """
    if float(h_high) == float(h_low):
        raise ValueError(
            f"estimate_shear_exponent: h_high ({h_high}) must differ from h_low ({h_low})"
        )

    v_low_arr = np.asarray(v_low, dtype=float)
    v_high_arr = np.asarray(v_high, dtype=float)
    v_low_arr, v_high_arr = np.broadcast_arrays(v_low_arr, v_high_arr)

    valid = (v_low_arr > 0) & (v_high_arr > 0)
    alpha = np.full(v_high_arr.shape, np.nan, dtype=float)
    log_height_ratio = np.log(float(h_high) / float(h_low))
    alpha[valid] = np.log(v_high_arr[valid] / v_low_arr[valid]) / log_height_ratio

    return _wrap_like_input(alpha, v_high, v_low)
