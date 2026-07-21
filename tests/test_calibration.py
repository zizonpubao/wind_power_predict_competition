"""Unit tests for src/models/calibration.py (PredictionCalibrator).

Uses small synthetic (pred, actual) arrays with a known systematic bias --
not the real feature parquets (that end-to-end evaluation is covered by
actually running src/training/evaluate_calibration.py, not by this suite).
"""
import numpy as np
import pytest

from src.models.calibration import PredictionCalibrator


def _make_biased_data(n: int = 200, seed: int = 0):
    """Synthetic data where predictions are consistently 20% below actual,
    plus a little noise -- a stand-in for the real under-prediction bias
    documented in reports/eda/ficr_gap_diagnosis.md.
    """
    rng = np.random.RandomState(seed)
    actual = rng.uniform(1_000, 20_000, size=n)
    pred = actual * 0.8 + rng.normal(0, 50, size=n)
    return pred, actual


def test_fit_transform_reduces_systematic_bias():
    pred, actual = _make_biased_data()

    raw_mean_signed_err = float(np.mean(pred - actual))
    assert raw_mean_signed_err < -1000  # sanity: the synthetic bias is real and large

    calibrator = PredictionCalibrator().fit(pred, actual)
    calibrated = calibrator.transform(pred)

    calib_mean_signed_err = float(np.mean(calibrated - actual))
    # Calibration should shrink the systematic bias substantially (it won't
    # be exactly zero in-sample since isotonic regression is a step function,
    # but it must be far closer to zero than the raw bias).
    assert abs(calib_mean_signed_err) < abs(raw_mean_signed_err) * 0.25

    # Overall absolute error should also improve, not just the signed mean.
    raw_mae = float(np.mean(np.abs(pred - actual)))
    calib_mae = float(np.mean(np.abs(calibrated - actual)))
    assert calib_mae < raw_mae


def test_transform_is_monotonic_nondecreasing():
    pred, actual = _make_biased_data(n=500, seed=1)
    calibrator = PredictionCalibrator().fit(pred, actual)

    test_pred = np.sort(np.random.RandomState(2).uniform(pred.min(), pred.max(), size=100))
    calibrated = calibrator.transform(test_pred)

    # test_pred is sorted ascending, so a monotonic map must produce a
    # non-decreasing calibrated sequence.
    assert np.all(np.diff(calibrated) >= -1e-9)


def test_out_of_bounds_predictions_are_clipped_not_extrapolated():
    pred, actual = _make_biased_data()
    calibrator = PredictionCalibrator().fit(pred, actual)

    far_below = np.array([pred.min() - 100_000])
    far_above = np.array([pred.max() + 100_000])

    calibrated_low = calibrator.transform(far_below)
    calibrated_high = calibrator.transform(far_above)

    # "out_of_bounds=clip" means the calibrated value for an extreme input
    # equals the calibrated value at the nearest boundary of the fitted
    # range, not a linear extrapolation beyond it.
    boundary_low = calibrator.transform(np.array([pred.min()]))
    boundary_high = calibrator.transform(np.array([pred.max()]))
    assert calibrated_low[0] == pytest.approx(boundary_low[0])
    assert calibrated_high[0] == pytest.approx(boundary_high[0])


def test_transform_before_fit_raises():
    calibrator = PredictionCalibrator()
    with pytest.raises(RuntimeError):
        calibrator.transform(np.array([1.0, 2.0]))


def test_mismatched_lengths_raise():
    calibrator = PredictionCalibrator()
    with pytest.raises(ValueError):
        calibrator.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_save_and_load_roundtrip(tmp_path):
    pred, actual = _make_biased_data()
    calibrator = PredictionCalibrator().fit(pred, actual)

    path = tmp_path / "calibrator_kpx_group_1.joblib"
    calibrator.save(path)
    assert path.exists()

    loaded = PredictionCalibrator.load(path)
    np.testing.assert_allclose(loaded.transform(pred), calibrator.transform(pred))
