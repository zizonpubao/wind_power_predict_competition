"""FICR step-surrogate GBM training pipeline (idea A3): per-KPX-group
block-aware CV over a small ``w_cliff``/``w_l2`` sweep, best-config final
full-data refit, written to ``experiments/<run_id>/`` in the same standard
layout every other model track uses.

This is the *training-time* counterpart to the post-hoc decision-theoretic FICR
optimization in ``src/features/decision_optimize.py``: instead of only reshaping
an already-trained model's point predictions, it trains a single-point GBM with
``src.models.lgbm_model.FICRSurrogateObjective`` -- a differentiable two-sigmoid
approximation of the money-shaped FICR settlement-rate step (4/3/0 won at the
6%/8% error-rate cliffs), plus an always-on kWh-scale Huber pull toward the
target. See that class's docstring for the loss math.

Reuses the exact same building blocks as ``train_gbm_quantile.py`` /
``train_baseline.py`` (``BlockTimeSeriesSplit`` via ``tune_common.oof_predict_generic``,
the official ``competition_score`` via ``tune_common.score_oof``, and the pruned
per-group feature set from ``configs/selected_features.json``) so the saved
``oof_predictions_<group>.parquet`` join with every other run's the same way
``src/ensembling/blend_search.py`` expects (identical CV splitter + label source
=> identical ``forecast_kst_dtm`` set as the GBM-quantile run this is compared
against).

Rather than a full Optuna study, a tiny fixed grid over ``w_cliff`` (cliff-
shaping strength) at fixed ``w_l2=1`` is swept per group and the best-CV config
kept -- CPU-cheap, deterministic, and enough to answer the only question that
matters here: does directly shaping the FICR cliffs at training time beat the
existing GBM-quantile / asymmetric-loss tracks, or is the honest ceiling of the
idea (gross misses can't be rescued by a loss, only near-cliff hours) too low to
adopt.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, GROUP_CAPACITY_KWH
from src.models.lgbm_model import DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_PARAMS, GroupLGBMModel
from src.training.train_baseline import FEATURE_SETS, KPX_GROUPS, N_SPLITS, _get_feature_cols, _get_git_commit
from src.training.tune_common import oof_predict_generic, score_oof

logger = logging.getLogger(__name__)

# Reference runs to diff the standalone FICR-surrogate CV score against in the
# printed table (CLAUDE.md-logged current tracks): the GBM-quantile + decision-
# optimal run (overall CV 0.6038) and the tuned LightGBM asymmetric-loss run
# (overall CV 0.5881). Not perfectly apples-to-apples -- both use tuned
# hyperparameters and extra machinery (9 quantiles + EU decision optimization /
# per-group asymmetric loss) -- but they're the bar an A3 point model would
# have to clear to earn a place in the blend.
GBM_QUANTILE_RUN_ID = "20260723_111840_gbm_quantile_pruned"
ASYM_RUN_ID = "20260722_103636_lgbm_tuned_pruned"

# Small per-group cliff-strength sweep at fixed w_l2=1.0 (see module docstring).
# Includes w_cliff=0.0 -- the pure well-scaled Huber/L2 point model -- as the
# floor, so "did the cliff shaping help at all" is answered directly.
DEFAULT_SWEEP: list[dict[str, float]] = [
    {"w_cliff": 0.0, "w_l2": 1.0},
    {"w_cliff": 0.25, "w_l2": 1.0},
    {"w_cliff": 0.5, "w_l2": 1.0},
    {"w_cliff": 1.0, "w_l2": 1.0},
]


def _load_overall(run_id: str) -> dict[str, Any] | None:
    path = EXPERIMENTS_DIR / run_id / "metrics.json"
    if not path.exists():
        logger.warning("Reference metrics not found at %s; comparison will omit it.", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _agg(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "score_mean": float(np.nanmean([f["score"] for f in fold_metrics])),
        "score_std": float(np.nanstd([f["score"] for f in fold_metrics])),
        "1-NMAE_mean": float(np.nanmean([f["1-NMAE"] for f in fold_metrics])),
        "1-NMAE_std": float(np.nanstd([f["1-NMAE"] for f in fold_metrics])),
        "FICR_mean": float(np.nanmean([f["FICR"] for f in fold_metrics])),
        "FICR_std": float(np.nanstd([f["FICR"] for f in fold_metrics])),
    }


def run_group(
    kpx_group: str,
    n_splits: int,
    feature_set: str,
    early_stopping_rounds: int,
    sweep: list[dict[str, float]],
) -> dict[str, Any]:
    """Block-aware CV sweep over ``sweep`` configs for one group; keep the best-
    CV-score config; refit it on all rows. Returns the standard result dict plus
    ``sweep_results`` (every config's mean score) for the metrics record.
    """
    path = DATA_PROCESSED_DIR / f"features_{kpx_group}_train.parquet"
    df = pd.read_parquet(path)

    n_missing = int(df["target"].isna().sum())
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    logger.info("%s: dropped %d rows with missing target, %d rows remain", kpx_group, n_missing, len(df))

    feature_cols = _get_feature_cols(df, kpx_group=kpx_group, feature_set=feature_set)
    capacity = GROUP_CAPACITY_KWH[kpx_group]

    sweep_results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in sweep:
        oof_df, fold_meta = oof_predict_generic(
            df,
            feature_cols,
            capacity,
            kpx_group,
            GroupLGBMModel,
            {"ficr_surrogate": dict(cfg)},
            early_stopping_rounds,
            n_splits,
        )
        fold_metrics = score_oof(oof_df, fold_meta, kpx_group)
        agg = _agg(fold_metrics)
        sweep_results.append({"config": dict(cfg), "agg_metrics": agg})
        logger.info(
            "%s cfg=%s -> score=%.4f (1-NMAE=%.4f FICR=%.4f)",
            kpx_group, cfg, agg["score_mean"], agg["1-NMAE_mean"], agg["FICR_mean"],
        )
        if best is None or agg["score_mean"] > best["agg_metrics"]["score_mean"]:
            best = {
                "config": dict(cfg),
                "fold_metrics": fold_metrics,
                "agg_metrics": agg,
                "oof_df": oof_df,
                "fold_meta": fold_meta,
            }

    assert best is not None
    if any(np.isnan(f["score"]) for f in best["fold_metrics"]):
        logger.warning(
            "%s: at least one CV fold had no eligible (>=10%% utilization) validation "
            "rows and scored NaN; aggregates use nanmean/nanstd.", kpx_group,
        )

    best_iterations = [f["best_iteration"] for f in best["fold_metrics"] if f["best_iteration"]]
    final_params: dict[str, Any] = {"ficr_surrogate": dict(best["config"])}
    if best_iterations:
        final_params["n_estimators"] = max(int(round(float(np.mean(best_iterations)))), 50)

    final_model = GroupLGBMModel(capacity_kwh=capacity, **final_params)
    final_model.fit(df[feature_cols], df["target"])
    logger.info(
        "%s: best config %s refit on all %d rows (n_estimators=%s)",
        kpx_group, best["config"], len(df), final_model.params["n_estimators"],
    )

    return {
        "kpx_group": kpx_group,
        "n_rows_used": len(df),
        "n_missing_target_dropped": n_missing,
        "feature_cols": feature_cols,
        "best_config": best["config"],
        "sweep_results": sweep_results,
        "fold_metrics": best["fold_metrics"],
        "agg_metrics": best["agg_metrics"],
        "final_model": final_model,
        "final_params": {k: v for k, v in final_params.items() if k != "ficr_surrogate"},
        "final_n_estimators": final_model.params["n_estimators"],
        "oof_df": best["oof_df"],
    }


def main() -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train the FICR step-surrogate single-point GBM (idea A3) for all 3 KPX groups."
    )
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default="pruned",
        help="'pruned' (default): configs/selected_features.json's per-group selection, matching "
        "the GBM-quantile run for a fair ensemble comparison. 'full': all feature columns.",
    )
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_ficr_surrogate"
    if args.feature_set == "pruned":
        run_id += "_pruned"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, Any]] = {}
    for kpx_group in KPX_GROUPS:
        logger.info("=== Running %s (FICR-surrogate) ===", kpx_group)
        all_results[kpx_group] = run_group(
            kpx_group,
            n_splits=args.n_splits,
            feature_set=args.feature_set,
            early_stopping_rounds=args.early_stopping_rounds,
            sweep=DEFAULT_SWEEP,
        )

        model_path = run_dir / f"model_{kpx_group}.joblib"
        joblib.dump(all_results[kpx_group]["final_model"], model_path)
        logger.info("Saved final model: %s", model_path)

        oof_path = run_dir / f"oof_predictions_{kpx_group}.parquet"
        oof_out = all_results[kpx_group]["oof_df"][["forecast_kst_dtm", "pred", "actual", "fold"]]
        oof_out.to_parquet(oof_path, index=False)
        logger.info("Saved OOF predictions (%d rows): %s", len(oof_out), oof_path)

    config = {
        "run_id": run_id,
        "model_type": "lgbm_ficr_surrogate_pointwise",
        "n_splits": args.n_splits,
        "feature_set": args.feature_set,
        "sweep": DEFAULT_SWEEP,
        "best_config_per_group": {g: all_results[g]["best_config"] for g in KPX_GROUPS},
        "model_default_params": DEFAULT_PARAMS,
        "early_stopping_rounds": args.early_stopping_rounds,
        "git_commit": _get_git_commit(),
        "final_n_estimators_per_group": {g: all_results[g]["final_n_estimators"] for g in KPX_GROUPS},
        "feature_count_per_group": {g: len(all_results[g]["feature_cols"]) for g in KPX_GROUPS},
        "feature_cols_per_group": {g: all_results[g]["feature_cols"] for g in KPX_GROUPS},
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    metrics_out: dict[str, Any] = {}
    for g in KPX_GROUPS:
        metrics_out[g] = {
            "n_rows_used": all_results[g]["n_rows_used"],
            "n_missing_target_dropped": all_results[g]["n_missing_target_dropped"],
            "best_config": all_results[g]["best_config"],
            "sweep_results": all_results[g]["sweep_results"],
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

    gbmq = _load_overall(GBM_QUANTILE_RUN_ID)
    asym = _load_overall(ASYM_RUN_ID)
    print(f"\n=== FICR-surrogate ({run_id}) vs GBM-quantile / asym-LGBM CV score (mean of folds) ===")
    header = f"{'group':<16}{'asym':>10}{'gbm_quant':>12}{'ficr_surr':>12}{'best_cfg':>22}"
    print(header)
    print("-" * len(header))
    for g in KPX_GROUPS:
        a = asym[g]["agg_metrics"]["score_mean"] if asym else float("nan")
        q = gbmq[g]["agg_metrics"]["score_mean"] if gbmq else float("nan")
        t = all_results[g]["agg_metrics"]["score_mean"]
        cfg = all_results[g]["best_config"]
        print(f"{g:<16}{a:>10.4f}{q:>12.4f}{t:>12.4f}{'w_cliff=' + str(cfg['w_cliff']):>22}")
    print("-" * len(header))
    a_o = asym["overall"]["score_mean"] if asym else float("nan")
    q_o = gbmq["overall"]["score_mean"] if gbmq else float("nan")
    t_o = overall["score_mean"]
    print(f"{'overall':<16}{a_o:>10.4f}{q_o:>12.4f}{t_o:>12.4f}")
    print(f"\nRun dir: {run_dir}")

    return run_id


if __name__ == "__main__":
    main()
