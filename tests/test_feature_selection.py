"""Unit tests for src/features/feature_selection.py (gain-based feature
pruning) and its wiring into src/training/train_baseline.py's
``_get_feature_cols(..., feature_set=...)``.

``cumulative_gain_selection`` is pure (no I/O, no model objects) so it's
tested directly against small synthetic gain arrays. The save/load round
trip and the training-side filtering logic are tested against a temporary
JSON file (monkeypatched in place of the real configs/selected_features.json)
so these tests don't depend on -- or mutate -- the real, git-tracked config.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from configs.paths import DATA_PROCESSED_DIR
from src.features import feature_selection as fs
from src.training import train_baseline as tb


# --- cumulative_gain_selection (pure function) -----------------------------


def test_cumulative_gain_selection_picks_minimal_top_gain_subset():
    names = ["a", "b", "c", "d", "e"]
    gains = np.array([50.0, 30.0, 10.0, 5.0, 5.0])  # total 100
    # top-1 covers 50%, top-2 covers 80%, top-3 covers 90%, top-4 covers 95%.
    selected = fs.cumulative_gain_selection(names, gains, cum_gain_threshold=0.95, max_features=None)
    assert set(selected) == {"a", "b", "c", "d"}


def test_cumulative_gain_selection_preserves_original_order():
    names = ["a", "b", "c", "d", "e"]
    gains = np.array([5.0, 50.0, 5.0, 30.0, 10.0])
    selected = fs.cumulative_gain_selection(names, gains, cum_gain_threshold=0.90, max_features=None)
    # Whatever gets selected should come back in the *input* order, not
    # sorted by gain descending.
    assert selected == [n for n in names if n in selected]


def test_cumulative_gain_selection_respects_max_features_cap():
    names = [f"f{i}" for i in range(10)]
    gains = np.array([10.0] * 10)  # uniform gain -> 95% cutoff needs 10 features
    selected = fs.cumulative_gain_selection(names, gains, cum_gain_threshold=0.95, max_features=3)
    assert len(selected) == 3


def test_cumulative_gain_selection_all_features_needed_for_100_pct():
    names = ["a", "b", "c"]
    gains = np.array([1.0, 1.0, 1.0])
    selected = fs.cumulative_gain_selection(names, gains, cum_gain_threshold=1.0, max_features=None)
    assert set(selected) == {"a", "b", "c"}


def test_cumulative_gain_selection_zero_total_gain_falls_back_to_all():
    names = ["a", "b", "c"]
    gains = np.array([0.0, 0.0, 0.0])
    selected = fs.cumulative_gain_selection(names, gains, cum_gain_threshold=0.95, max_features=2)
    # Degenerate case: no signal to rank by, so nothing is pruned (safer than
    # an arbitrary/misleading subset), even though max_features=2 is set.
    assert set(selected) == {"a", "b", "c"}


def test_cumulative_gain_selection_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fs.cumulative_gain_selection(["a", "b"], np.array([1.0, 2.0, 3.0]))


def test_cumulative_gain_selection_rejects_bad_threshold():
    with pytest.raises(ValueError):
        fs.cumulative_gain_selection(["a"], np.array([1.0]), cum_gain_threshold=0.0)
    with pytest.raises(ValueError):
        fs.cumulative_gain_selection(["a"], np.array([1.0]), cum_gain_threshold=1.5)


# --- save / load round trip -------------------------------------------------


def test_save_and_load_selected_features_round_trip(tmp_path):
    result = {
        "source_run_id": "dummy_run",
        "cum_gain_threshold": 0.95,
        "max_features": 75,
        "generated_at": "2026-07-22T00:00:00",
        "groups": {
            "kpx_group_1": {
                "selected_features": ["a", "b"],
                "n_selected": 2,
                "n_total": 3,
                "excluded_features": ["c"],
            }
        },
    }
    out_path = tmp_path / "selected_features.json"
    fs.save_selected_features(result, path=out_path)
    assert out_path.exists()

    loaded = fs.load_selected_features("kpx_group_1", path=out_path)
    assert loaded == ["a", "b"]

    # round trip through raw JSON, too (guards against accidental non-JSON-
    # serializable values sneaking into the result dict, e.g. numpy scalars).
    with open(out_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["groups"]["kpx_group_1"]["selected_features"] == ["a", "b"]


def test_load_selected_features_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        fs.load_selected_features("kpx_group_1", path=missing_path)


def test_load_selected_features_missing_group_raises(tmp_path):
    result = {
        "source_run_id": "dummy_run",
        "cum_gain_threshold": 0.95,
        "max_features": 75,
        "generated_at": "2026-07-22T00:00:00",
        "groups": {"kpx_group_1": {"selected_features": ["a"], "n_selected": 1, "n_total": 1, "excluded_features": []}},
    }
    out_path = tmp_path / "selected_features.json"
    fs.save_selected_features(result, path=out_path)
    with pytest.raises(KeyError):
        fs.load_selected_features("kpx_group_2", path=out_path)


# --- train_baseline._get_feature_cols(feature_set=...) wiring ---------------


def _toy_feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "data_available_kst_dtm": pd.to_datetime(["2023-12-31", "2024-01-01"]),
            "target": [1.0, 2.0],
            "feat_a": [0.1, 0.2],
            "feat_b": [0.3, 0.4],
            "feat_c": [0.5, 0.6],
        }
    )


def test_get_feature_cols_full_is_unchanged_default():
    df = _toy_feature_table()
    cols = tb._get_feature_cols(df)
    assert cols == ["feat_a", "feat_b", "feat_c"]


def test_get_feature_cols_pruned_filters_to_selected_list(monkeypatch):
    df = _toy_feature_table()
    monkeypatch.setattr(tb, "load_selected_features", lambda kpx_group: ["feat_b"])
    cols = tb._get_feature_cols(df, kpx_group="kpx_group_1", feature_set="pruned")
    assert cols == ["feat_b"]


def test_get_feature_cols_pruned_requires_kpx_group():
    df = _toy_feature_table()
    with pytest.raises(ValueError):
        tb._get_feature_cols(df, kpx_group=None, feature_set="pruned")


def test_get_feature_cols_pruned_raises_on_stale_selection(monkeypatch):
    df = _toy_feature_table()
    # Selection references a column that no longer exists in this feature
    # table -- must fail loudly rather than silently dropping it, since a
    # stale configs/selected_features.json could otherwise mask a real
    # feature-pipeline regression.
    monkeypatch.setattr(tb, "load_selected_features", lambda kpx_group: ["feat_b", "feat_nonexistent"])
    with pytest.raises(ValueError):
        tb._get_feature_cols(df, kpx_group="kpx_group_1", feature_set="pruned")


def test_get_feature_cols_rejects_unknown_feature_set():
    df = _toy_feature_table()
    with pytest.raises(ValueError):
        tb._get_feature_cols(df, kpx_group="kpx_group_1", feature_set="bogus")


# --- integration: real configs/selected_features.json, if present ----------


@pytest.mark.parametrize("kpx_group", ["kpx_group_1", "kpx_group_2", "kpx_group_3"])
def test_real_selected_features_are_subset_of_real_feature_table(kpx_group):
    """If configs/selected_features.json has already been generated (it is
    committed to the repo once produced), every listed feature must actually
    exist in that group's real processed feature table, and the selection
    must be a strict, non-trivial pruning (fewer than the full column count).
    """
    from configs.paths import SELECTED_FEATURES_JSON

    if not SELECTED_FEATURES_JSON.exists():
        pytest.skip("configs/selected_features.json not generated yet")

    table_path = DATA_PROCESSED_DIR / f"features_{kpx_group}_train.parquet"
    if not table_path.exists():
        pytest.skip(f"processed feature table not found: {table_path}")

    df = pd.read_parquet(table_path)
    all_cols = set(tb._get_feature_cols(df))
    selected = fs.load_selected_features(kpx_group)

    assert len(selected) > 0
    assert set(selected).issubset(all_cols)
    assert len(selected) < len(all_cols)
