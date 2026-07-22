"""XGBoost ensemble-candidate training pipeline: per-KPX-group Optuna-tuned
CV scoring (via the official ``competition_score``) + a final full-data
refit, written to ``experiments/<run_id>/`` -- the XGBoost counterpart to
``src/training/tune_hyperparams.py`` (LightGBM).

Built for ensembling, not standalone submission: uses the exact same
block-aware CV splitter (``BlockTimeSeriesSplit``, via
``src/training/tune_common.py``), the same official competition metric, and
-- critically for a later ``ensembler`` agent -- the same pruned per-group
feature set (``configs/selected_features.json``, default
``--feature-set pruned``) the current-best LightGBM run
(``20260722_103636_lgbm_tuned_pruned``) uses, so scores are directly
comparable and OOF predictions are aligned on the same rows/folds.

In addition to the usual ``config.yaml``/``metrics.json``/
``model_<group>.joblib`` artifacts, this script writes
``oof_predictions_<group>.parquet`` per group (columns: ``forecast_kst_dtm``,
``pred``, ``actual``, ``fold``) -- the out-of-fold predictions an
``ensembler`` needs to blend this model with others without any risk of
leakage (every row is a held-out prediction from a fold that group's row
was never trained on).

Trial budget: 20 (vs LightGBM's 40) -- a lighter budget for a second/third
model family within the same time-boxed exploration, per the task spec;
override with ``--n-trials`` if more search is wanted later.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import optuna
import yaml

from configs.paths import EXPERIMENTS_DIR
from src.models.xgb_model import DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_PARAMS, GroupXGBModel
from src.training.train_baseline import FEATURE_SETS, KPX_GROUPS, N_SPLITS, _get_git_commit
from src.training.tune_common import tune_group_generic

logger = logging.getLogger(__name__)

N_TRIALS = 20
SAMPLER_SEED = 42
N_ESTIMATORS_PARAM = "n_estimators"

# Current-best LightGBM run to diff the XGBoost scores against in the printed
# comparison table (CLAUDE.md-logged: overall CV score 0.5881, pruned feature
# set + group2 wake features + per-group asymmetric loss).
BASELINE_RUN_ID = "20260722_103636_lgbm_tuned_pruned"


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Search space sized for a ~20-26k row / 65-67-pruned-feature CPU
    XGBoost regression, with a 20-trial budget (lighter than LightGBM's 40 --
    see module docstring).
    """
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _load_baseline_metrics() -> dict[str, Any] | None:
    baseline_path = EXPERIMENTS_DIR / BASELINE_RUN_ID / "metrics.json"
    if not baseline_path.exists():
        logger.warning("Baseline metrics not found at %s; comparison table will omit baseline scores.", baseline_path)
        return None
    with open(baseline_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser(description="Optuna-tuned XGBoost training for the per-group ensemble.")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default="pruned",
        help=(
            "'pruned' (default): configs/selected_features.json's per-group gain-based "
            "selection, matching the current-best LightGBM run for a fair ensemble "
            "comparison. 'full': all feature columns."
        ),
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_xgb_tuned"
    if args.feature_set == "pruned":
        run_id += "_pruned"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, Any]] = {}
    for kpx_group in KPX_GROUPS:
        logger.info("=== Tuning %s (%d trials, XGBoost) ===", kpx_group, args.n_trials)
        all_results[kpx_group] = tune_group_generic(
            kpx_group,
            GroupXGBModel,
            _suggest_params,
            N_ESTIMATORS_PARAM,
            n_trials=args.n_trials,
            n_splits=args.n_splits,
            feature_set=args.feature_set,
            early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
            sampler_seed=SAMPLER_SEED,
        )

        model_path = run_dir / f"model_{kpx_group}.joblib"
        joblib.dump(all_results[kpx_group]["final_model"], model_path)
        logger.info("Saved final tuned model: %s", model_path)

        oof_path = run_dir / f"oof_predictions_{kpx_group}.parquet"
        oof_out = all_results[kpx_group]["oof_df"][["forecast_kst_dtm", "pred", "actual", "fold"]]
        oof_out.to_parquet(oof_path, index=False)
        logger.info("Saved OOF predictions (%d rows): %s", len(oof_out), oof_path)

    config = {
        "run_id": run_id,
        "model_type": "xgboost",
        "n_splits": args.n_splits,
        "feature_set": args.feature_set,
        "model_default_params": DEFAULT_PARAMS,
        "early_stopping_rounds": DEFAULT_EARLY_STOPPING_ROUNDS,
        "git_commit": _get_git_commit(),
        "final_n_estimators_per_group": {g: all_results[g]["final_n_estimators"] for g in KPX_GROUPS},
        "feature_count_per_group": {g: len(all_results[g]["feature_cols"]) for g in KPX_GROUPS},
        "feature_cols_per_group": {g: all_results[g]["feature_cols"] for g in KPX_GROUPS},
        "tuning_n_trials": args.n_trials,
        "tuning_sampler": "TPESampler",
        "tuning_sampler_seed": SAMPLER_SEED,
        "tuned_params_per_group": {g: all_results[g]["final_params"] for g in KPX_GROUPS},
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    metrics_out: dict[str, Any] = {}
    for g in KPX_GROUPS:
        metrics_out[g] = {
            "n_rows_used": all_results[g]["n_rows_used"],
            "n_missing_target_dropped": all_results[g]["n_missing_target_dropped"],
            "fold_metrics": all_results[g]["fold_metrics"],
            "agg_metrics": all_results[g]["agg_metrics"],
        }
    overall = {
        "score_mean": float(np.nanmean([all_results[g]["agg_metrics"]["score_mean"] for g in KPX_GROUPS])),
        "1-NMAE_mean": float(np.nanmean([all_results[g]["agg_metrics"]["1-NMAE_mean"] for g in KPX_GROUPS])),
        "FICR_mean": float(np.nanmean([all_results[g]["agg_metrics"]["FICR_mean"] for g in KPX_GROUPS])),
    }
    metrics_out["overall"] = overall
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    tuning_out: dict[str, Any] = {}
    for g in KPX_GROUPS:
        tuning_out[g] = {
            "n_trials": all_results[g]["n_trials"],
            "sampler_seed": all_results[g]["sampler_seed"],
            "tuning_wall_clock_seconds": all_results[g]["tuning_wall_clock_seconds"],
            "best_params": all_results[g]["best_params"],
            "best_cv_score": all_results[g]["best_cv_score"],
            "final_n_estimators": all_results[g]["final_n_estimators"],
            "trials": all_results[g]["trials"],
        }
    with open(run_dir / "tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(tuning_out, f, indent=2)

    baseline_metrics = _load_baseline_metrics()
    print(f"\n=== LightGBM ({BASELINE_RUN_ID}) vs XGBoost ({run_id}) CV Score (mean across folds) ===")
    header = f"{'group':<16}{'lgbm':>12}{'xgboost':>12}{'delta':>12}"
    print(header)
    print("-" * len(header))
    xgb_scores = {g: all_results[g]["agg_metrics"]["score_mean"] for g in KPX_GROUPS}
    for g in KPX_GROUPS:
        b = baseline_metrics[g]["agg_metrics"]["score_mean"] if baseline_metrics else float("nan")
        t = xgb_scores[g]
        print(f"{g:<16}{b:>12.4f}{t:>12.4f}{t - b:>12.4f}")
    print("-" * len(header))
    b_overall = baseline_metrics["overall"]["score_mean"] if baseline_metrics else float("nan")
    t_overall = overall["score_mean"]
    print(f"{'overall':<16}{b_overall:>12.4f}{t_overall:>12.4f}{t_overall - b_overall:>12.4f}")
    print(f"\nRun dir: {run_dir}")

    return run_id


if __name__ == "__main__":
    main()
