"""Post-hoc monotonic bias calibration for a single KPX group's model
predictions.

Motivated by ``reports/eda/ficr_gap_diagnosis.md``: the tuned LightGBM models
systematically under-predict 62-66% of eligible hours (a classic
regression-to-the-mean effect of gradient boosting), which the report
identifies as the single highest-leverage explanation for the FICR gap.
``PredictionCalibrator`` fits a monotonic pred -> actual mapping (isotonic
regression) on out-of-fold ``(pred, actual)`` pairs, so it corrects the
*shape* of the bias (e.g. "predictions of ~9,000 kWh are on average too low
by ~700 kWh") without needing a parametric assumption about the correction.

One instance handles exactly one KPX group, mirroring how ``GroupLGBMModel``
itself is per-group -- callers keep one ``PredictionCalibrator`` per group,
saved alongside that group's model as ``calibrator_<group>.joblib`` in the
experiment directory. Kept intentionally minimal: fit / transform / save /
load, nothing else.

IMPORTANT (see ``src/training/evaluate_calibration.py``): fitting a
calibrator on the same OOF predictions it is then scored against would leak
(the calibrator would be allowed to see the exact residuals it's being
evaluated on). This module itself is leakage-agnostic -- it just fits
whatever ``(pred, actual)`` pairs it is given -- so it is the *caller's*
responsibility to pass properly cross-fit / held-out pairs when evaluating,
and the full pooled OOF pairs only when producing the final production
artifact (exactly analogous to how the final model refit uses all rows,
while CV scoring must not).
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


class PredictionCalibrator:
    """Isotonic (monotonic) calibration mapping from raw model predictions to
    actual targets, fit on out-of-fold ``(pred, actual)`` pairs for one KPX
    group.

    ``out_of_bounds="clip"`` means predictions outside the range seen during
    ``fit`` are clipped to the nearest fitted boundary value rather than
    extrapolated, which keeps calibrated predictions from blowing up on
    test-time predictions slightly outside the OOF prediction range.
    """

    def __init__(self) -> None:
        self.isotonic_ = IsotonicRegression(out_of_bounds="clip")
        self.fitted_: bool = False

    def fit(self, oof_pred: np.ndarray, oof_actual: np.ndarray) -> "PredictionCalibrator":
        """Fit the isotonic pred -> actual mapping on OOF (pred, actual) pairs."""
        oof_pred = np.asarray(oof_pred, dtype=float)
        oof_actual = np.asarray(oof_actual, dtype=float)
        if len(oof_pred) != len(oof_actual):
            raise ValueError(
                f"oof_pred and oof_actual must be the same length, got {len(oof_pred)} vs {len(oof_actual)}"
            )
        self.isotonic_.fit(oof_pred, oof_actual)
        self.fitted_ = True
        return self

    def transform(self, pred: np.ndarray) -> np.ndarray:
        """Map raw predictions through the fitted isotonic calibration curve."""
        if not self.fitted_:
            raise RuntimeError("PredictionCalibrator.transform() called before fit().")
        pred = np.asarray(pred, dtype=float)
        return self.isotonic_.predict(pred)

    def fit_transform(self, oof_pred: np.ndarray, oof_actual: np.ndarray) -> np.ndarray:
        return self.fit(oof_pred, oof_actual).transform(oof_pred)

    def save(self, path: Any) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Any) -> "PredictionCalibrator":
        return joblib.load(path)
