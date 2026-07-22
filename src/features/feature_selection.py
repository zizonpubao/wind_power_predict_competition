"""Gain-based feature pruning for the per-KPX-group LightGBM models.

Motivated by ``reports/eda/ficr_gap_diagnosis.md`` section 4: all 3 groups'
tuned models use the same 141-column feature table, but the great majority of
gain is concentrated in a small subset -- the report found 48-73 features
(out of 141) already reach 95% cumulative gain, with the rest a long tail of
near-zero-value GFS-derived rolling/power-curve columns. Given the
~20-26k-row / 141-feature ratio flagged in ``src/models/lgbm_model.py``'s own
docstring, pruning to that 95%-cumulative-gain cutoff (capped at
``max_features`` so a group's tail doesn't dominate the input to a fresh
tuning pass) is a cheap way to cut variance/overfitting risk.

This module only *computes and stores* the per-group selected-feature lists
(as ``configs/selected_features.json``, read via
``configs.paths.SELECTED_FEATURES_JSON``); it does not itself retrain
anything. ``src/training/train_baseline.py`` and
``src/training/tune_hyperparams.py`` optionally filter their feature columns
down to this list via their ``--feature-set pruned`` CLI flag (default
remains ``full``, i.e. unchanged behavior unless explicitly requested).

Why 95% cumulative gain, capped at 75
--------------------------------------
95% cumulative gain is the same threshold ``ficr_gap_diagnosis.md`` used, so
the pruning decision is directly traceable to that report's own numbers
rather than an arbitrarily different cutoff. The ``max_features=75`` cap
(recommended range in the task spec was "top 50-75") only matters if a
group's 95% cutoff would otherwise exceed it -- it did not for any of the 3
groups when this was actually computed against the
``20260721_205339_lgbm_tuned`` run (65/141, 63/141, 67/141 for groups 1/2/3
respectively; all comfortably under 75), so the cap is a no-op safety bound
here rather than the binding constraint, but is kept so a future re-run
against a differently-shaped model can't silently balloon back toward the
full 141.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from configs.paths import EXPERIMENTS_DIR, SELECTED_FEATURES_JSON

logger = logging.getLogger(__name__)

KPX_GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
DEFAULT_CUM_GAIN_THRESHOLD = 0.95
DEFAULT_MAX_FEATURES = 75


def cumulative_gain_selection(
    feature_names: list[str],
    gains: np.ndarray,
    cum_gain_threshold: float = DEFAULT_CUM_GAIN_THRESHOLD,
    max_features: int | None = DEFAULT_MAX_FEATURES,
) -> list[str]:
    """Pick the smallest top-gain feature subset covering >= ``cum_gain_threshold``
    of total gain, then cap it at ``max_features`` if that cutoff is larger.

    Pure function (no I/O, no model objects) so it's directly unit-testable.
    Returns the selected feature names **in original ``feature_names`` order**
    (not sorted by gain) -- downstream column selection just needs set
    membership, and preserving input order keeps the selected list easy to
    diff against the full feature table.

    Ties / edge cases: if every gain is 0 (a degenerate model), every feature
    is "selected" once threshold*total <= 0 is trivially satisfied by rank 1
    -- guarded explicitly below so a broken model can't silently produce an
    empty or single-feature selection.
    """
    feature_names = list(feature_names)
    gains = np.asarray(gains, dtype=float)
    if len(feature_names) != len(gains):
        raise ValueError(
            f"feature_names ({len(feature_names)}) and gains ({len(gains)}) must be the same length"
        )
    if not (0.0 < cum_gain_threshold <= 1.0):
        raise ValueError(f"cum_gain_threshold must be in (0, 1], got {cum_gain_threshold}")

    total = gains.sum()
    order = np.argsort(-gains, kind="stable")

    if total <= 0:
        # Degenerate (all-zero-gain) model: cumulative-gain ranking is
        # meaningless, so fall back to keeping everything rather than
        # guessing -- pruning should never silently drop features it has no
        # real signal to judge.
        logger.warning(
            "cumulative_gain_selection: total gain is %.4g (<=0) across %d features; "
            "returning all features unpruned.",
            total,
            len(feature_names),
        )
        return feature_names

    cum = np.cumsum(gains[order]) / total
    n_cutoff = int(np.searchsorted(cum, cum_gain_threshold) + 1)
    n_cutoff = min(n_cutoff, len(feature_names))
    if max_features is not None:
        n_cutoff = min(n_cutoff, max_features)

    selected_idx = set(order[:n_cutoff].tolist())
    return [name for i, name in enumerate(feature_names) if i in selected_idx]


def _feature_importance_from_model(model: Any) -> tuple[list[str], np.ndarray]:
    """Extract (feature_names, gain_importance) from a fitted ``GroupLGBMModel``."""
    booster = model.model_.booster_
    names = list(booster.feature_name())
    gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    return names, gains


def select_features_from_run(
    run_id: str,
    kpx_groups: tuple[str, ...] = KPX_GROUPS,
    cum_gain_threshold: float = DEFAULT_CUM_GAIN_THRESHOLD,
    max_features: int | None = DEFAULT_MAX_FEATURES,
) -> dict[str, Any]:
    """Load ``experiments/<run_id>/model_<group>.joblib`` for every group,
    compute each group's 95%-cumulative-gain (capped) feature selection, and
    return a JSON-serializable summary dict (also written by ``save_selected_features``).
    """
    run_dir = EXPERIMENTS_DIR / run_id
    per_group: dict[str, Any] = {}

    for kpx_group in kpx_groups:
        model_path = run_dir / f"model_{kpx_group}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file for {kpx_group}: {model_path}")
        model = joblib.load(model_path)

        names, gains = _feature_importance_from_model(model)
        selected = cumulative_gain_selection(
            names, gains, cum_gain_threshold=cum_gain_threshold, max_features=max_features
        )
        excluded = [n for n in names if n not in set(selected)]

        logger.info(
            "%s: %d/%d features selected (%.0f%% cum-gain cutoff, capped at %s)",
            kpx_group,
            len(selected),
            len(names),
            cum_gain_threshold * 100,
            max_features,
        )

        per_group[kpx_group] = {
            "selected_features": selected,
            "n_selected": len(selected),
            "n_total": len(names),
            "excluded_features": excluded,
        }

    return {
        "source_run_id": run_id,
        "cum_gain_threshold": cum_gain_threshold,
        "max_features": max_features,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "groups": per_group,
    }


def save_selected_features(result: dict[str, Any], path: Path = SELECTED_FEATURES_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Wrote selected-features config: %s", path)


def load_selected_features(kpx_group: str, path: Path = SELECTED_FEATURES_JSON) -> list[str]:
    """Return the stored ``selected_features`` list for one KPX group.

    Raises ``FileNotFoundError`` if ``path`` doesn't exist yet (run
    ``python -m src.features.feature_selection`` first) and ``KeyError`` if
    ``kpx_group`` isn't in it -- callers (``src/training/*``) should let both
    propagate rather than silently falling back, since a caller asking for
    ``feature_set="pruned"`` without a selection file is a configuration
    error, not something to paper over.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it first, e.g.:\n"
            f"  python -m src.features.feature_selection --run-id <tuned_run_id>"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["groups"][kpx_group]["selected_features"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Compute per-KPX-group gain-based feature pruning from a trained run's "
            "model_<group>.joblib files, and write configs/selected_features.json."
        )
    )
    parser.add_argument("--run-id", required=True, help="experiments/<run_id> to load models from")
    parser.add_argument("--cum-gain-threshold", type=float, default=DEFAULT_CUM_GAIN_THRESHOLD)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument(
        "--output",
        type=Path,
        default=SELECTED_FEATURES_JSON,
        help="Output path (default: configs/selected_features.json)",
    )
    args = parser.parse_args()

    result = select_features_from_run(
        args.run_id,
        cum_gain_threshold=args.cum_gain_threshold,
        max_features=args.max_features,
    )
    save_selected_features(result, path=args.output)

    print(f"\n=== Feature selection summary (source run: {args.run_id}) ===")
    header = f"{'group':<16}{'selected':>10}{'total':>10}{'excluded':>10}"
    print(header)
    print("-" * len(header))
    for g in KPX_GROUPS:
        info = result["groups"][g]
        print(f"{g:<16}{info['n_selected']:>10}{info['n_total']:>10}{info['n_total'] - info['n_selected']:>10}")
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
