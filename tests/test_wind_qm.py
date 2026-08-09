import numpy as np
import pytest

from src.features.wind_qm import apply_quantile_mapping, fit_quantile_mapping


def test_fit_and_apply_roundtrip_on_training_data():
    rng = np.random.default_rng(0)
    forecast = rng.normal(7, 2, 2000)
    actual = forecast * 1.2 + 1.0  # deterministic monotone bias

    mapping = fit_quantile_mapping(forecast, actual)
    applied = apply_quantile_mapping(forecast, mapping)

    # Bias-corrected values should track the true bias closely on in-sample data.
    assert np.corrcoef(applied, actual)[0, 1] > 0.99
    assert abs(applied.mean() - actual.mean()) < 0.5


def test_monotonic_map():
    mapping = {
        "forecast_pctiles": np.array([1.0, 2.0, 3.0, 4.0]),
        "actual_pctiles": np.array([2.0, 4.0, 5.0, 9.0]),
    }
    x = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    y = apply_quantile_mapping(x, mapping)
    assert np.all(np.diff(y) >= 0)


def test_out_of_range_clips_to_edges_not_extrapolated():
    mapping = {
        "forecast_pctiles": np.array([1.0, 2.0, 3.0]),
        "actual_pctiles": np.array([5.0, 6.0, 7.0]),
    }
    y = apply_quantile_mapping(np.array([-100.0, 100.0]), mapping)
    assert y[0] == pytest.approx(5.0)
    assert y[1] == pytest.approx(7.0)


def test_fit_requires_paired_shapes():
    with pytest.raises(ValueError):
        fit_quantile_mapping(np.zeros(10), np.zeros(20))


def test_fit_drops_nan_rows_and_requires_min_sample():
    forecast = np.concatenate([np.arange(150.0), [np.nan]])
    actual = np.concatenate([np.arange(150.0) * 2, [np.nan]])
    mapping = fit_quantile_mapping(forecast, actual, n_percentiles=99)
    assert len(mapping["forecast_pctiles"]) == 99
    assert np.all(np.diff(mapping["forecast_pctiles"]) >= 0)

    with pytest.raises(ValueError):
        fit_quantile_mapping(np.array([1.0, 2.0]), np.array([1.0, 2.0]), n_percentiles=99)
