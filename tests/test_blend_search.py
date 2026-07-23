"""Tests for src/ensembling/blend_search.py -- the nested-LOFO ensemble
weight search used to blend already-trained LightGBM/XGBoost/CatBoost OOF
predictions.

Uses small synthetic DataFrames (not real experiment runs) so these run fast
and deterministically, and so the "does the search actually recover a known
best weight" property can be checked exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ensembling.blend_search import (
    best_weight,
    blend_score,
    load_joined_oof,
    nested_lofo_search,
    weight_grid,
)

GROUP = "kpx_group_1"
CAPACITY = 21_600.0


def _make_oof_df(n_per_fold: int = 200, n_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Synthetic OOF frame: actual values comfortably above the 10% eligibility
    threshold, one perfect model ("good") and one noisy model ("bad") so the
    weight search has an unambiguous best answer (weight ~1 on "good").
    """
    rng = np.random.default_rng(seed)
    rows = []
    dtm0 = pd.Timestamp("2024-01-01")
    for fold in range(1, n_folds + 1):
        for i in range(n_per_fold):
            actual = float(rng.uniform(0.3, 0.9) * CAPACITY)
            rows.append(
                {
                    "forecast_kst_dtm": dtm0 + pd.Timedelta(hours=(fold - 1) * n_per_fold + i),
                    "fold": fold,
                    "actual": actual,
                    "pred_good": actual,  # perfect predictions
                    "pred_bad": actual + rng.normal(0, CAPACITY * 0.5),  # very noisy
                }
            )
    return pd.DataFrame(rows)


def test_weight_grid_three_models_sums_to_one_and_covers_corners():
    grid = weight_grid(["a", "b", "c"], step=0.1)
    for w in grid:
        assert abs(sum(w) - 1.0) < 1e-9
        assert all(x >= 0 for x in w)
    assert (1.0, 0.0, 0.0) in grid
    assert (0.0, 1.0, 0.0) in grid
    assert (0.0, 0.0, 1.0) in grid


def test_weight_grid_two_models():
    grid = weight_grid(["a", "b"], step=0.25)
    assert (1.0, 0.0) in grid
    assert (0.0, 1.0) in grid
    assert (0.5, 0.5) in grid
    for w in grid:
        assert abs(sum(w) - 1.0) < 1e-9


def test_weight_grid_rejects_unsupported_model_count():
    with pytest.raises(ValueError):
        weight_grid(["a", "b", "c", "d"], step=0.5)


def test_blend_score_all_weight_on_perfect_model_is_near_max_score():
    df = _make_oof_df()
    score = blend_score(df, GROUP, ["good", "bad"], (1.0, 0.0))
    # perfect predictions -> 1-NMAE=1.0, FICR=1.0 -> score=1.0
    assert score == pytest.approx(1.0, abs=1e-6)


def test_best_weight_recovers_perfect_model_over_noisy_one():
    df = _make_oof_df()
    grid = weight_grid(["good", "bad"], step=0.1)
    w, score = best_weight(df, GROUP, ["good", "bad"], grid)
    assert w == (1.0, 0.0)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_nested_lofo_search_never_evaluates_a_fold_with_weights_fit_on_itself():
    df = _make_oof_df()
    grid = weight_grid(["good", "bad"], step=0.2)
    rows = nested_lofo_search(df, GROUP, ["good", "bad"], grid)
    assert len(rows) == 5
    for r in rows:
        # the perfect model should always win the weight search regardless of
        # which fold is held out, since "good" is a noiseless oracle in every
        # fold not just the held-out one
        assert r["weight"]["good"] >= r["weight"]["bad"]
        assert r["blend_eval_score_on_held_fold"] == pytest.approx(1.0, abs=1e-6)


def test_load_joined_oof_raises_on_missing_run(tmp_path, monkeypatch):
    import src.ensembling.blend_search as blend_search_module

    monkeypatch.setattr(blend_search_module, "EXPERIMENTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_joined_oof({"lgbm": "does_not_exist_run"}, GROUP)


def test_load_joined_oof_raises_on_fold_mismatch_between_runs(tmp_path, monkeypatch):
    import src.ensembling.blend_search as blend_search_module

    monkeypatch.setattr(blend_search_module, "EXPERIMENTS_DIR", tmp_path)

    dtm = pd.date_range("2024-01-01", periods=10, freq="h")
    run_a_dir = tmp_path / "run_a"
    run_b_dir = tmp_path / "run_b"
    run_a_dir.mkdir()
    run_b_dir.mkdir()

    df_a = pd.DataFrame({"forecast_kst_dtm": dtm, "pred": 1.0, "actual": 2.0, "fold": 1})
    df_b = df_a.copy()
    df_b["fold"] = 2  # mismatched fold assignment for the same rows

    df_a.to_parquet(run_a_dir / f"oof_predictions_{GROUP}.parquet", index=False)
    df_b.to_parquet(run_b_dir / f"oof_predictions_{GROUP}.parquet", index=False)

    with pytest.raises(ValueError, match="fold assignment"):
        load_joined_oof({"a": "run_a", "b": "run_b"}, GROUP)


# --- Phase E additions: generic --runs CLI parsing, fixed-weight comparison,
# and mixed GBM/torch test-time prediction dispatch ----------------------------

from src.ensembling.blend_search import (  # noqa: E402
    _parse_kv,
    _predict_group_test,
    per_fold_mean_score,
    weights_dict_to_tuple,
)


def test_parse_kv_parses_repeated_name_equals_value():
    assert _parse_kv(["gbm=run_a", "lstm=run_b"]) == {"gbm": "run_a", "lstm": "run_b"}
    assert _parse_kv(None) == {}
    assert _parse_kv(["a=0.35", "b=0.65"], cast=float) == {"a": 0.35, "b": 0.65}


def test_parse_kv_rejects_item_without_equals():
    with pytest.raises(ValueError):
        _parse_kv(["no_equals_here"])


def test_weights_dict_to_tuple_aligns_to_model_order_and_defaults_zero():
    assert weights_dict_to_tuple(["gbm", "lstm", "transformer"], {"gbm": 0.5, "lstm": 0.5}) == (0.5, 0.5, 0.0)


def test_weights_dict_to_tuple_rejects_unknown_model_and_bad_sum():
    with pytest.raises(ValueError, match="unknown"):
        weights_dict_to_tuple(["gbm", "lstm"], {"xxx": 1.0})
    with pytest.raises(ValueError, match="sum to 1"):
        weights_dict_to_tuple(["gbm", "lstm"], {"gbm": 0.3, "lstm": 0.3})


def test_per_fold_mean_score_is_mean_of_per_fold_blend_scores():
    df = _make_oof_df(n_per_fold=100, n_folds=5)
    mean_s, per_fold = per_fold_mean_score(df, GROUP, ["good", "bad"], (1.0, 0.0))
    # perfect model on every fold -> 1.0 each -> mean 1.0
    assert set(per_fold) == {1, 2, 3, 4, 5}
    assert mean_s == pytest.approx(1.0, abs=1e-6)
    for s in per_fold.values():
        assert s == pytest.approx(1.0, abs=1e-6)


class _FakeGBM:
    """Feature-matrix model: predict takes only the feature columns (no block key)."""

    capacity_kwh = 21_600.0

    def predict(self, X):
        # would raise if handed the extra block/dt columns a sequence model needs
        assert list(X.columns) == ["f0", "f1"], f"GBM got unexpected columns {list(X.columns)}"
        return np.zeros(len(X))


class _FakeSeq:
    """Sequence model: predict needs the block key + datetime alongside features."""

    capacity_kwh = 21_600.0
    block_col = "data_available_kst_dtm"
    dt_col = "forecast_kst_dtm"
    feature_cols = ["f0", "f1"]

    def predict(self, df):
        assert self.block_col in df.columns and self.dt_col in df.columns, (
            "sequence model was not handed its block/datetime columns"
        )
        return np.ones(len(df))


def test_predict_group_test_dispatches_matrix_vs_sequence_models():
    test_df = pd.DataFrame(
        {
            "f0": [1.0, 2.0, 3.0],
            "f1": [4.0, 5.0, 6.0],
            "data_available_kst_dtm": pd.date_range("2025-01-01", periods=3, freq="h"),
            "forecast_kst_dtm": pd.date_range("2025-01-02", periods=3, freq="h"),
        }
    )
    feats = ["f0", "f1"]
    # GBM: called with exactly the feature columns (asserted inside _FakeGBM.predict)
    gbm_out = _predict_group_test(_FakeGBM(), test_df, feats)
    assert gbm_out.shape == (3,)
    # Sequence model: called with feature cols PLUS block/dt (asserted inside _FakeSeq.predict)
    seq_out = _predict_group_test(_FakeSeq(), test_df, feats)
    assert seq_out.shape == (3,)
    assert (seq_out == 1.0).all()
