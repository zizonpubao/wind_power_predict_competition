"""Baseline LightGBM training pipeline: per-KPX-group CV scoring (via the
official ``competition_score``) plus a final full-data refit, written to
``experiments/<run_id>/``.

Composes existing, already-tested building blocks -- it does not reimplement
CV splitting (``src.validation.splitter.BlockTimeSeriesSplit``), metrics
(``src.evaluation.metrics.competition_score``), or the model itself
(``src.models.lgbm_model.GroupLGBMModel``).

Per CLAUDE.md section 3 / the task spec, each group's train parquet has a
small number of missing-target rows (kpx_group_3: ~8,766 rows, the whole of
2022; kpx_group_1/2: ~103-104 scattered hours) -- these are simply dropped
before CV/training, no imputation.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, GROUP_CAPACITY_KWH
from src.evaluation.metrics import competition_score
from src.models.lgbm_model import DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_PARAMS, GroupLGBMModel
from src.validation.splitter import BlockTimeSeriesSplit

logger = logging.getLogger(__name__)

KPX_GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
N_SPLITS = 5

# Columns present in the feature parquets that are identifiers/target, not
# model inputs. Everything else in the parquet is a feature -- computed from
# the actual columns rather than an assumed fixed list, per the task spec.
NON_FEATURE_COLS = {"forecast_kst_dtm", "data_available_kst_dtm", "target"}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def _get_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = out.stdout.strip()
        return commit or None
    except Exception:
        return None


def run_group(kpx_group: str, n_splits: int = N_SPLITS) -> dict[str, Any]:
    """Run block-aware CV + a final full-data refit for one kpx_group.

    Returns a dict with: kpx_group, n_rows_used, n_missing_target_dropped,
    feature_cols, fold_metrics (list of per-fold dicts), agg_metrics
    (mean/std across folds), and final_model (a fitted GroupLGBMModel).
    """
    path = DATA_PROCESSED_DIR / f"features_{kpx_group}_train.parquet"
    df = pd.read_parquet(path)

    n_missing = int(df["target"].isna().sum())
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    logger.info(
        "%s: dropped %d rows with missing target, %d rows remain",
        kpx_group,
        n_missing,
        len(df),
    )

    feature_cols = _get_feature_cols(df)
    capacity = GROUP_CAPACITY_KWH[kpx_group]

    splitter = BlockTimeSeriesSplit(n_splits=n_splits)
    fold_metrics: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        X_train, y_train = train_df[feature_cols], train_df["target"]
        X_val, y_val = val_df[feature_cols], val_df["target"]

        model = GroupLGBMModel(capacity_kwh=capacity)
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
        fold_result = {
            "fold": fold_i,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "score": scores["score"],
            "1-NMAE": scores["1-NMAE"],
            "FICR": scores["FICR"],
            "best_iteration": int(model.best_iteration_) if model.best_iteration_ else None,
        }
        fold_metrics.append(fold_result)
        if model.best_iteration_:
            best_iterations.append(int(model.best_iteration_))

        logger.info(
            "%s fold %d/%d: n_train=%d n_val=%d score=%.4f 1-NMAE=%.4f FICR=%.4f best_iter=%s",
            kpx_group,
            fold_i,
            n_splits,
            len(train_idx),
            len(val_idx),
            scores["score"],
            scores["1-NMAE"],
            scores["FICR"],
            fold_result["best_iteration"],
        )

    agg_metrics = {
        "score_mean": float(np.nanmean([f["score"] for f in fold_metrics])),
        "score_std": float(np.nanstd([f["score"] for f in fold_metrics])),
        "1-NMAE_mean": float(np.nanmean([f["1-NMAE"] for f in fold_metrics])),
        "1-NMAE_std": float(np.nanstd([f["1-NMAE"] for f in fold_metrics])),
        "FICR_mean": float(np.nanmean([f["FICR"] for f in fold_metrics])),
        "FICR_std": float(np.nanstd([f["FICR"] for f in fold_metrics])),
    }
    if any(np.isnan(f["score"]) for f in fold_metrics):
        logger.warning(
            "%s: at least one CV fold had no eligible (>=10%% utilization) validation "
            "rows and scored NaN; aggregates use nanmean/nanstd.",
            kpx_group,
        )

    # Final refit on ALL available (non-missing-target) rows for this group --
    # this is the model used for test inference. Early stopping needs a held-
    # out eval set, which would contradict "train on all rows", so instead we
    # reuse the CV folds' average best_iteration (when early stopping actually
    # triggered) as a fixed n_estimators for the full-data fit; otherwise fall
    # back to the model's configured default n_estimators.
    final_params: dict[str, Any] = {}
    if best_iterations:
        final_params["n_estimators"] = max(int(round(float(np.mean(best_iterations)))), 50)

    final_model = GroupLGBMModel(capacity_kwh=capacity, **final_params)
    final_model.fit(df[feature_cols], df["target"])
    logger.info(
        "%s: final model refit on all %d rows (n_estimators=%s)",
        kpx_group,
        len(df),
        final_model.params["n_estimators"],
    )

    return {
        "kpx_group": kpx_group,
        "n_rows_used": len(df),
        "n_missing_target_dropped": n_missing,
        "feature_cols": feature_cols,
        "fold_metrics": fold_metrics,
        "agg_metrics": agg_metrics,
        "final_model": final_model,
    }


def main() -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train the LightGBM baseline for all 3 KPX groups.")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_lgbm_baseline"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, Any]] = {}
    for kpx_group in KPX_GROUPS:
        logger.info("=== Running %s ===", kpx_group)
        all_results[kpx_group] = run_group(kpx_group, n_splits=args.n_splits)

        model_path = run_dir / f"model_{kpx_group}.joblib"
        joblib.dump(all_results[kpx_group]["final_model"], model_path)
        logger.info("Saved final model: %s", model_path)

    config = {
        "run_id": run_id,
        "n_splits": args.n_splits,
        "model_default_params": DEFAULT_PARAMS,
        "early_stopping_rounds": DEFAULT_EARLY_STOPPING_ROUNDS,
        "git_commit": _get_git_commit(),
        "feature_count_per_group": {
            g: len(all_results[g]["feature_cols"]) for g in KPX_GROUPS
        },
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

    print("\n=== Baseline CV Summary (mean across folds) ===")
    header = f"{'group':<16}{'score':>10}{'1-NMAE':>10}{'FICR':>10}"
    print(header)
    print("-" * len(header))
    for g in KPX_GROUPS:
        m = all_results[g]["agg_metrics"]
        print(f"{g:<16}{m['score_mean']:>10.4f}{m['1-NMAE_mean']:>10.4f}{m['FICR_mean']:>10.4f}")
    print("-" * len(header))
    print(f"{'overall':<16}{overall['score_mean']:>10.4f}{overall['1-NMAE_mean']:>10.4f}{overall['FICR_mean']:>10.4f}")
    print(f"\nRun dir: {run_dir}")

    return run_id


if __name__ == "__main__":
    main()
