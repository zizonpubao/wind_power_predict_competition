"""Tests for src/ensembling/recalibrate.py — the leaderboard-verified per-group
multiplicative recalibration of the v14 blend.

These tests use only synthetic in-memory data (no experiment artifacts / torch),
so they run fast and offline. The reproduction of the actual 0.6280 submission
lives in the module CLI, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from configs.paths import GROUP_CAPACITY_KWH
from src.ensembling.recalibrate import (
    DEFAULT_STRENGTH,
    MultiplicativeRecalibrator,
    crossfit_group,
    fit_group_factor,
)
from src.evaluation.metrics import competition_score


def _synthetic_group(n: int = 500, cap: float = 21_600.0, seed: int = 0) -> pd.DataFrame:
    """A single group's OOF frame where predictions are systematically low by a
    known factor, so the recovered factor is predictable.
    """
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2024-01-01", periods=n, freq="h")
    actual = rng.uniform(0.05 * cap, 0.95 * cap, size=n)
    # predictions ~8% low + a little noise
    pred = actual / 1.08 + rng.normal(0, 0.01 * cap, size=n)
    pred = np.clip(pred, 0, cap)
    fold = (np.arange(n) % 5) + 1
    return pd.DataFrame({"forecast_kst_dtm": dt, "pred": pred, "actual": actual, "fold": fold})


def test_perfect_prediction_factor_near_one():
    """When pred == actual, the score-maximizing factor must be ~1.0."""
    cap = 21_600.0
    dt = pd.date_range("2024-01-01", periods=300, freq="h")
    actual = np.random.default_rng(1).uniform(0.05 * cap, 0.95 * cap, size=300)
    f = fit_group_factor(actual.copy(), actual.copy(), pd.Series(dt), cap)
    assert abs(f - 1.0) <= 0.01


def test_underprediction_recovers_upward_factor():
    """Systematically-low predictions must yield a factor > 1 (recover ~1.08)."""
    df = _synthetic_group()
    f = fit_group_factor(df["pred"].to_numpy(), df["actual"].to_numpy(), df["forecast_kst_dtm"], 21_600.0)
    assert 1.0 < f <= 1.12
    assert abs(f - 1.08) <= 0.02


def test_strength_scaling_formula():
    """applied_factor must equal 1 + strength*(factor-1) exactly."""
    factors = {"kpx_group_1": 1.12, "kpx_group_2": 1.06, "kpx_group_3": 1.115}
    for s in (0.0, 0.25, 0.5, 1.0):
        rc = MultiplicativeRecalibrator(factors, strength=s)
        for g, f in factors.items():
            assert rc.applied_factor(g) == pytest.approx(1.0 + s * (f - 1.0))


def test_strength_zero_is_identity_up_to_clip():
    """strength=0 leaves predictions untouched (aside from the capacity clip)."""
    factors = {"kpx_group_1": 1.12}
    rc = MultiplicativeRecalibrator(factors, strength=0.0)
    pred = np.array([0.0, 5000.0, 21_000.0])
    out = rc.transform("kpx_group_1", pred, capacity=21_600.0)
    np.testing.assert_allclose(out, pred)


def test_clip_bounds():
    """transform clips to [0, capacity]."""
    rc = MultiplicativeRecalibrator({"g": 2.0}, strength=1.0)
    pred = np.array([-100.0, 10_000.0, 20_000.0])
    out = rc.transform("g", pred, capacity=21_600.0)
    assert out.min() >= 0.0
    assert out.max() <= 21_600.0
    # 10000 * 2.0 = 20000 (within cap, not clipped); -100 -> 0; 20000*2 -> clip to cap
    np.testing.assert_allclose(out, [0.0, 20_000.0, 21_600.0])


def test_default_strength_is_half():
    assert DEFAULT_STRENGTH == 0.5
    assert MultiplicativeRecalibrator({"g": 1.1}).strength == 0.5


def test_save_load_roundtrip(tmp_path):
    factors = {"kpx_group_1": 1.12, "kpx_group_2": 1.06, "kpx_group_3": 1.115}
    rc = MultiplicativeRecalibrator(factors, strength=0.5)
    path = tmp_path / "recal.joblib"
    rc.save(path)
    loaded = MultiplicativeRecalibrator.load(path)
    assert loaded.oof_factor == rc.oof_factor
    assert loaded.strength == rc.strength
    pred = np.array([1000.0, 5000.0, 12_000.0])
    for g in factors:
        np.testing.assert_allclose(
            loaded.transform(g, pred, 21_600.0), rc.transform(g, pred, 21_600.0)
        )


def test_factor_consistent_with_competition_score():
    """The fitted factor must actually maximize competition_score over the grid:
    scoring the recalibrated preds must beat scoring the raw preds.
    """
    cap = 21_600.0
    df = _synthetic_group(cap=cap)
    f = fit_group_factor(df["pred"].to_numpy(), df["actual"].to_numpy(), df["forecast_kst_dtm"], cap)

    key = df["forecast_kst_dtm"].to_numpy()
    actual_df = pd.DataFrame({"forecast_kst_dtm": key, "kpx_group_1": df["actual"].to_numpy()})
    raw_df = pd.DataFrame({"forecast_kst_dtm": key, "kpx_group_1": df["pred"].to_numpy()})
    scaled = np.clip(df["pred"].to_numpy() * f, 0, cap)
    scaled_df = pd.DataFrame({"forecast_kst_dtm": key, "kpx_group_1": scaled})

    raw_score = competition_score(raw_df, actual_df, group_cols=["kpx_group_1"])["score"]
    scaled_score = competition_score(scaled_df, actual_df, group_cols=["kpx_group_1"])["score"]
    assert scaled_score >= raw_score


def test_crossfit_group_reports_all_strengths():
    """crossfit_group returns a fold factor per fold and a score per strength,
    with strength 0 equal to the untouched base blend score.
    """
    df = _synthetic_group()
    cap = 21_600.0
    out = crossfit_group(df, cap, strengths=(0.0, 0.5, 1.0))
    assert set(out["fold_factors"]) == {1, 2, 3, 4, 5}
    assert set(out["scores"]) == {0.0, 0.5, 1.0}

    key = df["forecast_kst_dtm"].to_numpy()
    actual_df = pd.DataFrame({"forecast_kst_dtm": key, "kpx_group_1": df["actual"].to_numpy()})
    base_df = pd.DataFrame({"forecast_kst_dtm": key, "kpx_group_1": df["pred"].to_numpy()})
    base = competition_score(base_df, actual_df, group_cols=["kpx_group_1"])["score"]
    assert out["scores"][0.0] == pytest.approx(base)
    # correction should help on this systematically-low synthetic data
    assert out["scores"][0.5] >= out["scores"][0.0]
