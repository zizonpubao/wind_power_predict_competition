"""Optuna hyperparameter search for the per-KPX-group LightGBM models.

Extends -- does not replace -- ``src.training.train_baseline``: reuses the
exact same block-aware CV splitter (``BlockTimeSeriesSplit``), the exact same
official competition metric (``competition_score``), the same feature-column
selection logic (imported from ``train_baseline``, not re-implemented), and
the same "average the CV folds' best_iteration, use it as a fixed
n_estimators for the full-data refit" strategy the baseline already uses.

For each of the 3 kpx_groups, an Optuna study searches over
``num_leaves``, ``learning_rate``, ``min_child_samples``, ``subsample``,
``colsample_bytree``, ``reg_alpha``, ``reg_lambda`` (ranges chosen for a
~20-26k row / 141-feature CPU LightGBM regression -- see ``_suggest_params``)
to maximize the mean 5-fold CV ``competition_score`` -- the official
0.5*(1-NMAE)+0.5*FICR score, not RMSE/MAE. ``n_estimators`` itself is left at
the model's default (2000) both during search and for the final refit; early
stopping (same 50-round patience as the baseline) picks the effective number
of boosting rounds per trial/fold, exactly mirroring how the baseline handles
n_estimators.

Writes ``experiments/<run_id>_tuned/`` with the same config.yaml/metrics.json
*shape* train_baseline.py writes (so evaluator/ensembler can consume either
run interchangeably), plus a ``tuning_results.json`` with best params, best CV
score, and every trial's params/value for later inspection.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, GROUP_CAPACITY_KWH
from src.evaluation.metrics import competition_score
from src.inference.predict import generate_submission
from src.models.lgbm_model import DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_PARAMS, GroupLGBMModel
from src.training.train_baseline import KPX_GROUPS, N_SPLITS, _get_feature_cols, _get_git_commit
from src.validation.splitter import BlockTimeSeriesSplit

logger = logging.getLogger(__name__)

N_TRIALS = 40
SAMPLER_SEED = 42

# Baseline run to diff the tuned scores against in the printed comparison
# table (CLAUDE.md-logged real CV run: score 0.572 / 1-NMAE 0.856 / FICR 0.288).
BASELINE_RUN_ID = "20260720_160818_lgbm_baseline"


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Search space sized for a ~20-26k row / 141-feature CPU LightGBM regression."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _cv_score(
    df: pd.DataFrame,
    feature_cols: list[str],
    capacity: float,
    kpx_group: str,
    params: dict[str, Any],
    n_splits: int = N_SPLITS,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Run the same block-aware 5-fold CV train_baseline.run_group runs, for an
    arbitrary hyperparameter dict. Returns (fold_metrics, best_iterations),
    same shapes as train_baseline.run_group's internals -- reused both as the
    Optuna objective's value and, after the study finishes, to get the
    best-params fold metrics/best_iterations needed for the final refit.
    """
    splitter = BlockTimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        X_train, y_train = train_df[feature_cols], train_df["target"]
        X_val, y_val = val_df[feature_cols], val_df["target"]

        model = GroupLGBMModel(capacity_kwh=capacity, **params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
        )

        val_pred = model.predict(X_val)
        pred_df = pd.DataFrame(
            {"forecast_kst_dtm": val_df["forecast_kst_dtm"].to_numpy(), kpx_group: val_pred}
        )
        actual_df = pd.DataFrame(
            {"forecast_kst_dtm": val_df["forecast_kst_dtm"].to_numpy(), kpx_group: y_val.to_numpy()}
        )
        scores = competition_score(pred_df, actual_df, group_cols=[kpx_group])

        fold_metrics.append(
            {
                "fold": fold_i,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "score": scores["score"],
                "1-NMAE": scores["1-NMAE"],
                "FICR": scores["FICR"],
                "best_iteration": int(model.best_iteration_) if model.best_iteration_ else None,
            }
        )
        if model.best_iteration_:
            best_iterations.append(int(model.best_iteration_))

    return fold_metrics, best_iterations


def _make_objective(df: pd.DataFrame, feature_cols: list[str], capacity: float, kpx_group: str, n_splits: int):
    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        fold_metrics, _ = _cv_score(df, feature_cols, capacity, kpx_group, params, n_splits=n_splits)
        return float(np.nanmean([f["score"] for f in fold_metrics]))

    return objective


def tune_group(kpx_group: str, n_trials: int = N_TRIALS, n_splits: int = N_SPLITS) -> dict[str, Any]:
    """Run an Optuna study for one kpx_group, then refit a final GroupLGBMModel
    on all available rows using the best-found hyperparameters (same
    average-CV-best_iteration-as-n_estimators approach train_baseline.run_group
    uses). Returns a dict with everything needed to write config.yaml,
    metrics.json, tuning_results.json and the model artifact.
    """
    path = DATA_PROCESSED_DIR / f"features_{kpx_group}_train.parquet"
    df = pd.read_parquet(path)

    n_missing = int(df["target"].isna().sum())
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    logger.info("%s: dropped %d rows with missing target, %d rows remain", kpx_group, n_missing, len(df))

    feature_cols = _get_feature_cols(df)
    capacity = GROUP_CAPACITY_KWH[kpx_group]

    sampler = optuna.samplers.TPESampler(seed=SAMPLER_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t0 = time.time()
    study.optimize(
        _make_objective(df, feature_cols, capacity, kpx_group, n_splits),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0
    logger.info(
        "%s: Optuna study finished (%d trials, %.1fs, %.2fs/trial), best_value=%.4f, best_params=%s",
        kpx_group,
        len(study.trials),
        elapsed,
        elapsed / max(len(study.trials), 1),
        study.best_value,
        study.best_params,
    )

    # Re-run CV once more with the winning params to get fold_metrics +
    # best_iterations for the final refit (Optuna only records the scalar
    # objective value per trial, not the full fold breakdown).
    best_params = dict(study.best_params)
    fold_metrics, best_iterations = _cv_score(df, feature_cols, capacity, kpx_group, best_params, n_splits=n_splits)

    agg_metrics = {
        "score_mean": float(np.nanmean([f["score"] for f in fold_metrics])),
        "score_std": float(np.nanstd([f["score"] for f in fold_metrics])),
        "1-NMAE_mean": float(np.nanmean([f["1-NMAE"] for f in fold_metrics])),
        "1-NMAE_std": float(np.nanstd([f["1-NMAE"] for f in fold_metrics])),
        "FICR_mean": float(np.nanmean([f["FICR"] for f in fold_metrics])),
        "FICR_std": float(np.nanstd([f["FICR"] for f in fold_metrics])),
    }

    # Final refit on ALL available rows -- same strategy as
    # train_baseline.run_group: fixed n_estimators = mean CV best_iteration
    # (floored at 50), since early stopping needs a held-out eval set which
    # would contradict "train on all rows".
    final_params: dict[str, Any] = dict(best_params)
    if best_iterations:
        final_params["n_estimators"] = max(int(round(float(np.mean(best_iterations)))), 50)

    final_model = GroupLGBMModel(capacity_kwh=capacity, **final_params)
    final_model.fit(df[feature_cols], df["target"])
    logger.info(
        "%s: final tuned model refit on all %d rows (n_estimators=%s)",
        kpx_group,
        len(df),
        final_model.params["n_estimators"],
    )

    trials_records = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        }
        for t in study.trials
    ]

    return {
        "kpx_group": kpx_group,
        "n_rows_used": len(df),
        "n_missing_target_dropped": n_missing,
        "feature_cols": feature_cols,
        "n_trials": n_trials,
        "sampler_seed": SAMPLER_SEED,
        "tuning_wall_clock_seconds": elapsed,
        "best_params": best_params,
        "best_cv_score": agg_metrics["score_mean"],
        "fold_metrics": fold_metrics,
        "agg_metrics": agg_metrics,
        "trials": trials_records,
        "final_model": final_model,
        "final_params": final_params,
        "final_n_estimators": final_model.params["n_estimators"],
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

    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for the per-group LightGBM models.")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--skip-submission", action="store_true", help="Skip generating a submission CSV at the end.")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_lgbm_tuned"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, Any]] = {}
    for kpx_group in KPX_GROUPS:
        logger.info("=== Tuning %s (%d trials) ===", kpx_group, args.n_trials)
        all_results[kpx_group] = tune_group(kpx_group, n_trials=args.n_trials, n_splits=args.n_splits)

        model_path = run_dir / f"model_{kpx_group}.joblib"
        joblib.dump(all_results[kpx_group]["final_model"], model_path)
        logger.info("Saved final tuned model: %s", model_path)

    # --- config.yaml: same top-level shape as train_baseline.py's, plus
    # tuning-specific extras, so evaluator/ensembler can read either run's
    # config.yaml with the same keys they already rely on. ---
    config = {
        "run_id": run_id,
        "n_splits": args.n_splits,
        "model_default_params": DEFAULT_PARAMS,
        "early_stopping_rounds": DEFAULT_EARLY_STOPPING_ROUNDS,
        "git_commit": _get_git_commit(),
        "final_n_estimators_per_group": {g: all_results[g]["final_n_estimators"] for g in KPX_GROUPS},
        "feature_count_per_group": {g: len(all_results[g]["feature_cols"]) for g in KPX_GROUPS},
        "feature_cols_per_group": {g: all_results[g]["feature_cols"] for g in KPX_GROUPS},
        # tuning-specific extras (not present in the baseline's config.yaml)
        "tuning_n_trials": args.n_trials,
        "tuning_sampler": "TPESampler",
        "tuning_sampler_seed": SAMPLER_SEED,
        "tuned_params_per_group": {g: all_results[g]["final_params"] for g in KPX_GROUPS},
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    # --- metrics.json: identical shape to train_baseline.py's (per-group
    # n_rows_used/n_missing_target_dropped/fold_metrics/agg_metrics + overall). ---
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

    # --- tuning_results.json: best params + best CV score per group + every
    # trial's params/value, for later inspection (separate from metrics.json
    # so metrics.json stays schema-compatible with train_baseline's). ---
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

    # --- Comparison table: baseline vs tuned score per group ---
    baseline_metrics = _load_baseline_metrics()
    print("\n=== Baseline vs Tuned CV Score (mean across folds) ===")
    header = f"{'group':<16}{'baseline':>12}{'tuned':>12}{'delta':>12}"
    print(header)
    print("-" * len(header))
    tuned_scores = {g: all_results[g]["agg_metrics"]["score_mean"] for g in KPX_GROUPS}
    baseline_scores: dict[str, float] = {}
    for g in KPX_GROUPS:
        b = baseline_metrics[g]["agg_metrics"]["score_mean"] if baseline_metrics else float("nan")
        baseline_scores[g] = b
        t = tuned_scores[g]
        print(f"{g:<16}{b:>12.4f}{t:>12.4f}{t - b:>12.4f}")
    print("-" * len(header))
    b_overall = baseline_metrics["overall"]["score_mean"] if baseline_metrics else float("nan")
    t_overall = overall["score_mean"]
    print(f"{'overall':<16}{b_overall:>12.4f}{t_overall:>12.4f}{t_overall - b_overall:>12.4f}")
    print(f"\nRun dir: {run_dir}")

    if not args.skip_submission:
        logger.info("Generating submission from tuned run %s ...", run_id)
        submission = generate_submission(run_id)
        print(f"\nSubmission generated: {len(submission)} rows -> submissions/submission_{run_id}.csv")

    return run_id


if __name__ == "__main__":
    main()
