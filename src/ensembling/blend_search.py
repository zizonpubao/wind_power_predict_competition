"""Per-KPX-group ensemble weight search over multiple already-trained models'
OOF predictions (LightGBM / XGBoost / CatBoost, pruned feature set), validated
with nested leave-one-fold-out (LOFO) CV so the reported score is not
optimistic.

Why nested LOFO, not "fit weights on all 5 folds' OOF then re-score on that
same OOF": that would let the grid search fit noise in the exact rows it is
then judged on -- a blend that looks better on training-period metrics but
was never evaluated the same way the base models were (CLAUDE.md's own
"leakage" caution, and the ensembler agent brief's explicit warning). Instead,
for each of the 5 CV folds k:
  1. Grid-search the best (w_lgbm, w_xgb, w_cat) simplex weight (step 0.05,
     weights >= 0, sum to 1) that maximizes the official ``competition_score``
     on the OOF rows from the OTHER 4 folds pooled together.
  2. Freeze that weight and evaluate it on fold k's OOF rows alone (rows the
     weight search never saw).
Averaging the 5 held-out scores gives a score directly comparable to each
base model's own mean-of-5-folds CV score (same held-out rows, same metric).

A "production" weight (fit on all 5 folds pooled, maximum data) is also
computed for deployment IF a group's nested LOFO score shows a genuine,
fold-consistent improvement over the current-best LightGBM-alone run; groups
that don't clear that bar fall back to pure LightGBM (weight (1,0,0)) rather
than being forced into an ensemble that doesn't actually help.
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
import pandas as pd
import yaml

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, SUBMISSIONS_DIR
from src.data.loaders import load_sample_submission
from src.evaluation.metrics import competition_score
from src.training.train_baseline import KPX_GROUPS

logger = logging.getLogger(__name__)

# Runs this blend combines -- all three trained on the identical pruned
# feature set + group2 wake features + identical 5-fold BlockTimeSeriesSplit,
# per the task spec, so their OOF rows/folds line up exactly (verified in
# main() before any weight search runs).
DEFAULT_RUNS: dict[str, str] = {
    "lgbm": "20260722_103636_lgbm_tuned_pruned",
    "xgb": "20260722_105650_xgb_tuned_pruned",
    "cat": "20260722_105902_catboost_tuned_pruned",
}
# The run whose standalone CV score is the bar a blend must clear per group.
BASELINE_MODEL_NAME = "lgbm"

WEIGHT_STEP = 0.05
# A group's nested-LOFO blend is only adopted over pure baseline if it beats
# the baseline's own mean CV score AND does so in a majority of the 5 folds
# (not just on average) -- guards against one lucky fold masking a blend that
# is actually worse most of the time.
MIN_FOLDS_IMPROVED = 3


def weight_grid(model_names: list[str], step: float = WEIGHT_STEP) -> list[tuple[float, ...]]:
    """All weight tuples over ``model_names`` on the ``step``-resolution
    simplex (each weight a multiple of ``step``, all >= 0, summing to 1).

    Only implemented for 2 or 3 models (this task's use case); raises for
    anything else rather than silently doing the wrong thing.
    """
    n = round(1.0 / step)
    if len(model_names) == 3:
        grid = []
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                grid.append((round(i * step, 4), round(j * step, 4), round(k * step, 4)))
        return grid
    if len(model_names) == 2:
        return [(round(i * step, 4), round(1.0 - i * step, 4)) for i in range(n + 1)]
    raise ValueError(f"weight_grid only supports 2 or 3 models, got {len(model_names)}")


def load_joined_oof(runs: dict[str, str], group: str) -> pd.DataFrame:
    """Join every run's ``oof_predictions_<group>.parquet`` on
    ``forecast_kst_dtm`` into one frame with columns ``forecast_kst_dtm``,
    ``fold``, ``actual``, and ``pred_<name>`` for each ``name`` in ``runs``.

    Asserts the join doesn't drop rows (i.e. every run covers the exact same
    OOF rows) and that ``fold``/``actual`` agree across runs -- both are
    expected since all runs share the same CV splitter and label source, and
    a mismatch would mean the runs aren't actually comparable.
    """
    base: pd.DataFrame | None = None
    for name, run_id in runs.items():
        path = EXPERIMENTS_DIR / run_id / f"oof_predictions_{group}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{name} ({run_id}): missing {path.name} -- ensembler cannot blend a run "
                f"without its OOF predictions on disk (see module docstring; report this "
                f"back rather than training the missing artifact ad hoc, unless it can be "
                f"deterministically regenerated from already-recorded hyperparameters)."
            )
        df = pd.read_parquet(path).rename(columns={"pred": f"pred_{name}"})
        cols = ["forecast_kst_dtm", "fold", "actual", f"pred_{name}"]
        if base is None:
            base = df[cols]
            continue
        before = len(base)
        merged = base.merge(
            df[["forecast_kst_dtm", "fold", "actual", f"pred_{name}"]],
            on="forecast_kst_dtm",
            how="inner",
            suffixes=("", f"_{name}"),
        )
        if len(merged) != before:
            raise ValueError(
                f"{group}: joining {name}'s OOF predictions dropped rows "
                f"({before} -> {len(merged)}) -- runs do not share the same OOF rows."
            )
        fold_mismatch = int((merged["fold"] != merged[f"fold_{name}"]).sum())
        actual_mismatch = float((merged["actual"] - merged[f"actual_{name}"]).abs().max())
        if fold_mismatch:
            raise ValueError(f"{group}: {fold_mismatch} row(s) disagree on fold assignment between runs.")
        if actual_mismatch > 1e-6:
            raise ValueError(f"{group}: actual/label mismatch between runs (max diff {actual_mismatch}).")
        base = merged.drop(columns=[f"fold_{name}", f"actual_{name}"])
    assert base is not None
    return base.reset_index(drop=True)


def blend_score(df: pd.DataFrame, group: str, model_names: list[str], weights: tuple[float, ...]) -> float:
    """Official ``competition_score`` of the weighted-average blend of
    ``pred_<name>`` columns against ``actual``, on the given rows.
    """
    blended = sum(w * df[f"pred_{name}"] for name, w in zip(model_names, weights))
    pred_df = pd.DataFrame({"forecast_kst_dtm": df["forecast_kst_dtm"].to_numpy(), group: blended.to_numpy()})
    actual_df = pd.DataFrame({"forecast_kst_dtm": df["forecast_kst_dtm"].to_numpy(), group: df["actual"].to_numpy()})
    return competition_score(pred_df, actual_df, group_cols=[group])["score"]


def best_weight(
    df: pd.DataFrame, group: str, model_names: list[str], grid: list[tuple[float, ...]]
) -> tuple[tuple[float, ...], float]:
    """Grid search: the weight tuple in ``grid`` maximizing ``blend_score`` on ``df``."""
    best_w, best_s = grid[0], -np.inf
    for w in grid:
        s = blend_score(df, group, model_names, w)
        if s > best_s:
            best_s, best_w = s, w
    return best_w, best_s


def nested_lofo_search(
    df: pd.DataFrame, group: str, model_names: list[str], grid: list[tuple[float, ...]]
) -> list[dict[str, Any]]:
    """Nested leave-one-fold-out weight search (see module docstring): for
    each fold, fit the best weight on the other folds, evaluate it on the
    held-out fold. Returns one dict per fold.
    """
    folds = sorted(df["fold"].unique())
    rows = []
    for k in folds:
        fit_df = df[df["fold"] != k]
        eval_df = df[df["fold"] == k]
        w, fit_score = best_weight(fit_df, group, model_names, grid)
        eval_score = blend_score(eval_df, group, model_names, w)
        rows.append(
            {
                "held_out_fold": int(k),
                "weight": {name: float(x) for name, x in zip(model_names, w)},
                "fit_score_on_other_folds": float(fit_score),
                "blend_eval_score_on_held_fold": float(eval_score),
            }
        )
    return rows


def analyze_group(
    group: str,
    runs: dict[str, str],
    baseline_fold_scores: dict[int, float],
    baseline_mean_score: float,
    step: float = WEIGHT_STEP,
    min_folds_improved: int = MIN_FOLDS_IMPROVED,
) -> dict[str, Any]:
    """Full per-group analysis: join OOF, nested LOFO search, production
    weight fit, adopt/fallback decision vs the baseline model's own CV score.
    """
    model_names = list(runs.keys())
    df = load_joined_oof(runs, group)
    grid = weight_grid(model_names, step=step)

    nested_rows = nested_lofo_search(df, group, model_names, grid)
    for r in nested_rows:
        lgbm_score = baseline_fold_scores[r["held_out_fold"]]
        r["baseline_eval_score_on_held_fold"] = lgbm_score
        r["delta_vs_baseline"] = r["blend_eval_score_on_held_fold"] - lgbm_score

    nested_mean = float(np.mean([r["blend_eval_score_on_held_fold"] for r in nested_rows]))
    n_improved = sum(1 for r in nested_rows if r["delta_vs_baseline"] > 0)

    prod_w, prod_fit_score = best_weight(df, group, model_names, grid)

    adopt_blend = (nested_mean > baseline_mean_score) and (n_improved >= min_folds_improved)
    fallback_weight = {name: (1.0 if name == BASELINE_MODEL_NAME else 0.0) for name in model_names}
    final_weight = {name: float(x) for name, x in zip(model_names, prod_w)} if adopt_blend else fallback_weight

    return {
        "group": group,
        "model_names": model_names,
        "weight_grid_step": step,
        "nested_lofo_rows": nested_rows,
        "nested_blend_mean_score": nested_mean,
        "baseline_mean_score": baseline_mean_score,
        "delta_vs_baseline_nested": nested_mean - baseline_mean_score,
        "n_folds_improved_of_5": n_improved,
        "min_folds_improved_required": min_folds_improved,
        "production_weight_fit_on_all_folds": {name: float(x) for name, x in zip(model_names, prod_w)},
        "production_weight_fit_score_optimistic_not_for_decisions": float(prod_fit_score),
        "adopt_blend": adopt_blend,
        "final_weight": final_weight,
    }


def generate_blend_submission(
    run_id: str, runs: dict[str, str], final_weights: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """Apply each group's final per-model weights to the 3 runs' TEST
    predictions (each model's own ``model_<group>.joblib`` + that run's own
    ``feature_cols_per_group`` from config.yaml), sum, clip to capacity, and
    assemble/validate a submission exactly like ``src.inference.predict``.
    """
    sample = load_sample_submission()
    submission = sample[["forecast_id", "forecast_kst_dtm"]].copy()

    run_configs: dict[str, dict[str, Any]] = {}
    for name, rid in runs.items():
        with open(EXPERIMENTS_DIR / rid / "config.yaml", "r", encoding="utf-8") as f:
            run_configs[name] = yaml.safe_load(f)

    for group in KPX_GROUPS:
        w = final_weights[group]
        blended = None
        capacity_kwh = None
        for name, rid in runs.items():
            weight = w.get(name, 0.0)
            if weight == 0.0:
                continue
            model_path = EXPERIMENTS_DIR / rid / f"model_{group}.joblib"
            model = joblib.load(model_path)
            capacity_kwh = model.capacity_kwh
            feature_cols = run_configs[name]["feature_cols_per_group"][group]
            test_df = pd.read_parquet(DATA_PROCESSED_DIR / f"features_{group}_test.parquet")
            preds = model.predict(test_df[feature_cols])
            blended = preds * weight if blended is None else blended + preds * weight
        if blended is None:
            raise ValueError(f"{group}: final weight has no nonzero component -- nothing to predict with.")
        blended = np.clip(blended, 0.0, capacity_kwh * 1.01)

        pred_df = pd.DataFrame({"forecast_kst_dtm": test_df["forecast_kst_dtm"].to_numpy(), group: blended})
        before_len = len(submission)
        submission = submission.merge(pred_df, on="forecast_kst_dtm", how="left")
        if len(submission) != before_len:
            raise ValueError(f"Merge for {group} changed row count ({before_len} -> {len(submission)}).")
        n_missing = int(submission[group].isna().sum())
        if n_missing:
            raise ValueError(f"{n_missing} rows could not be matched to a {group} test prediction.")

    submission_cols = ["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]
    submission = submission[submission_cols].reset_index(drop=True)

    assert len(submission) == 8760, f"expected 8,760 rows, got {len(submission)}"
    assert list(submission.columns) == submission_cols
    assert (submission["forecast_id"].to_numpy() == sample["forecast_id"].to_numpy()).all()
    assert (submission["forecast_kst_dtm"].to_numpy() == sample["forecast_kst_dtm"].to_numpy()).all()

    out_path = SUBMISSIONS_DIR / f"submission_{run_id}.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote blended submission: %s (%d rows)", out_path, len(submission))
    return submission


def main() -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Per-group nested-LOFO ensemble weight search.")
    parser.add_argument("--step", type=float, default=WEIGHT_STEP)
    parser.add_argument("--min-folds-improved", type=int, default=MIN_FOLDS_IMPROVED)
    parser.add_argument("--force-submission", action="store_true", help="Write a blended submission even if no group adopts the blend (writes the pure-baseline-equivalent blend).")
    args = parser.parse_args()

    runs = DEFAULT_RUNS
    with open(EXPERIMENTS_DIR / runs[BASELINE_MODEL_NAME] / "metrics.json", "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_ensemble_lgbm_xgb_cat"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    group_results: dict[str, dict[str, Any]] = {}
    for group in KPX_GROUPS:
        baseline_fold_scores = {fm["fold"]: fm["score"] for fm in baseline_metrics[group]["fold_metrics"]}
        baseline_mean = baseline_metrics[group]["agg_metrics"]["score_mean"]
        result = analyze_group(
            group, runs, baseline_fold_scores, baseline_mean, step=args.step, min_folds_improved=args.min_folds_improved
        )
        group_results[group] = result
        print(f"\n=== {group}: nested LOFO blend mean={result['nested_blend_mean_score']:.4f} "
              f"vs baseline({BASELINE_MODEL_NAME})={baseline_mean:.4f} "
              f"(delta={result['delta_vs_baseline_nested']:+.4f}, "
              f"improved {result['n_folds_improved_of_5']}/5 folds) -> "
              f"{'ADOPT BLEND' if result['adopt_blend'] else 'FALL BACK TO BASELINE'} "
              f"final_weight={result['final_weight']}")

    overall_baseline = float(np.mean([baseline_metrics[g]["agg_metrics"]["score_mean"] for g in KPX_GROUPS]))
    overall_adopted = float(np.mean([group_results[g]["nested_blend_mean_score"] if group_results[g]["adopt_blend"]
                                      else baseline_metrics[g]["agg_metrics"]["score_mean"] for g in KPX_GROUPS]))
    any_adopted = any(group_results[g]["adopt_blend"] for g in KPX_GROUPS)

    print(f"\n=== OVERALL: baseline({BASELINE_MODEL_NAME})={overall_baseline:.4f} "
          f"adopted-per-group(nested, honest)={overall_adopted:.4f} "
          f"(delta={overall_adopted - overall_baseline:+.4f}) ===")

    config = {
        "run_id": run_id,
        "method": "ensembler: per-group nested leave-one-fold-out (LOFO) weighted-average blend "
                  "over already-trained LightGBM/XGBoost/CatBoost OOF predictions",
        "base_runs": runs,
        "baseline_model_for_decision": BASELINE_MODEL_NAME,
        "weight_grid_step": args.step,
        "min_folds_improved_required": args.min_folds_improved,
        "final_weight_per_group": {g: group_results[g]["final_weight"] for g in KPX_GROUPS},
        "adopt_blend_per_group": {g: group_results[g]["adopt_blend"] for g in KPX_GROUPS},
        "any_group_adopted": any_adopted,
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    metrics_out = {
        "per_group": group_results,
        "overall": {
            "baseline_score": overall_baseline,
            "adopted_per_group_nested_score": overall_adopted,
            "delta": overall_adopted - overall_baseline,
        },
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    if any_adopted or args.force_submission:
        generate_blend_submission(run_id, runs, {g: group_results[g]["final_weight"] for g in KPX_GROUPS})
    else:
        print(
            "\nNo group's nested-LOFO blend robustly beat its baseline -> recommendation is "
            "to keep the baseline standalone submission as final; no new blended submission "
            "written (pass --force-submission to override)."
        )

    print(f"\nRun dir: {run_dir}")
    return run_id


if __name__ == "__main__":
    main()
