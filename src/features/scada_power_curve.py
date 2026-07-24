"""Fit + persist the per-manufacturer SCADA *empirical* power curve.

This is the one place SCADA is read for a feature-adjacent purpose, and it is
deliberately leakage-safe: it fits the turbine's own **measured wind speed ->
measured 10-min energy** relationship (a physical property of the machine, not
the label and not the forecast), freezes it to
``configs/scada_power_curves.json``, and thereafter
``build_features._add_physics_features`` applies that frozen lookup table to the
*forecast* hub-height wind speed. So:

  - SCADA never enters the feature table directly (CLAUDE.md section 3).
  - The fitted curve is a fixed deterministic function -> identically computable
    for train and test (test has no SCADA, but it does not need any).
  - It uses measured ws and measured power, **not** ``train_labels`` -- so it
    introduces no target leakage into CV (unlike a curve fit on forecast-speed
    -> label, which would be a target-encoding of the very quantity being
    predicted).

Glitch masking (CLAUDE.md section 4) is handled inside
``physics_features.fit_empirical_power_curve`` (|value| <= 700 kWh per 10-min
turbine bucket, ws in a plausible range). vestas is pooled across turbines 1-12
(both group_1 and group_2 are VESTAS V126), unison across turbines 1-5.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from configs.paths import PROJECT_ROOT
from src.data.loaders import load_scada_unison, load_scada_vestas
from src.features.physics_features import fit_empirical_power_curve

logger = logging.getLogger(__name__)

SCADA_POWER_CURVES_JSON = PROJECT_ROOT / "configs" / "scada_power_curves.json"


def _pool_ws_power(df: pd.DataFrame, prefix: str, n_turbines: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack every turbine's (measured ws, measured 10-min energy) pair from a
    wide SCADA frame into two long 1-D arrays."""
    ws_parts, pw_parts = [], []
    for i in range(1, n_turbines + 1):
        ws_col = f"{prefix}_wtg{i:02d}_ws"
        pw_col = f"{prefix}_wtg{i:02d}_power_kw10m"
        if ws_col in df.columns and pw_col in df.columns:
            ws_parts.append(df[ws_col].to_numpy(dtype=float))
            pw_parts.append(df[pw_col].to_numpy(dtype=float))
    return np.concatenate(ws_parts), np.concatenate(pw_parts)


def build_scada_power_curves() -> dict[str, dict]:
    """Fit and return ``{"vestas": curve, "unison": curve}`` (each a
    ``{"centers": [...], "values": [...]}`` dict)."""
    vestas = load_scada_vestas()
    unison = load_scada_unison()
    ws_v, pw_v = _pool_ws_power(vestas, "vestas", 12)
    ws_u, pw_u = _pool_ws_power(unison, "unison", 5)
    curves = {
        "vestas": fit_empirical_power_curve(ws_v, pw_v),
        "unison": fit_empirical_power_curve(ws_u, pw_u),
    }
    for name, c in curves.items():
        logger.info("Fitted %s power curve: %d bins, value range [%.3f, %.3f]",
                    name, len(c["centers"]), min(c["values"]), max(c["values"]))
    return curves


def save_scada_power_curves(curves: dict[str, dict], path=SCADA_POWER_CURVES_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)
    logger.info("Wrote %s", path)


def load_scada_power_curves(path=SCADA_POWER_CURVES_JSON) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m src.features.scada_power_curve` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    curves = build_scada_power_curves()
    save_scada_power_curves(curves)


if __name__ == "__main__":
    main()
