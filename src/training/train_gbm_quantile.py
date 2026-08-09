"""GBM-quantile + decision-theoretic-post-processing training pipeline: per-
KPX-group block-aware CV scoring (via the official ``competition_score``) plus
a final full-data refit, written to ``experiments/<run_id>/`` -- the v14-
reproduction GBM component (see
``.claude/plans/logical-stirring-sphinx.md`` Phase B).

Unlike ``src/training/train_xgb.py``/``train_catboost.py``, this script does
**not** run an Optuna search: the task spec is to first faithfully reproduce
the v14 hand-off's GBM component with its own starting hyperparameters
(``src.models.lgbm_quantile_model.DEFAULT_PARAMS``) via a single CV pass,
since re-tuning before knowing whether the decision-theoretic post-processing
itself is worth anything would confound "did the post-processing help" with
"did the hyperparameters change" -- a later pass can add Optuna search once
this baseline is established (mirrors ``tune_common.py``'s
``oof_predict_generic``/``score_oof`` being used directly, without
``tune_group_generic``'s Optuna wrapper, exactly as the task spec asks).

Still reuses the same building blocks as every other model track in this
repo: ``BlockTimeSeriesSplit`` (via ``tune_common.oof_predict_generic``), the
official ``competition_score`` (via ``tune_common.score_oof``), and
``configs/selected_features.json``'s pruned per-group feature set (default
``--feature-set pruned``, matching the current-best LightGBM run
``20260722_103636_lgbm_tuned_pruned`` so scores/OOF predictions are directly
ensemble-comparable) -- so ``experiments/<run_id>/oof_predictions_<group>.parquet``
joins with every other run's the same way ``src/ensembling/blend_search.py``
already expects.
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
from src.features.decision_optimize import QUANTILES, QUANTILES_19
from src.models.lgbm_quantile_model import (
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_PARAMS,
    GroupLGBMQuantileModel,
)
from src.training.train_baseline import FEATURE_SETS, KPX_GROUPS, N_SPLITS, _get_feature_cols, _get_git_commit
from src.training.tune_common import oof_predict_generic, score_oof

logger = logging.getLogger(__name__)

# Current-best LightGBM run to diff the GBM-quantile scores against in the
# printed comparison table (CLAUDE.md-logged: overall CV score 0.5881, pruned
# feature set + group2 wake features + per-group asymmetric loss). Not a
# fair apples-to-apples comparison (different model family entirely -- 9
# quantile sub-models + EU decision optimization vs a single point-regression
# model) but useful context on how far off this independent model family
# lands, which is what actually matters for an ensemble candidate.
BASELINE_RUN_ID = "20260722_103636_lgbm_tuned_pruned"


def _load_baseline_metrics() -> dict[str, Any] | None:
    baseline_path = EXPERIMENTS_DIR / BASELINE_RUN_ID / "metrics.json"
    if not baseline_path.exists():
        logger.warning("Baseline metrics not found at %s; comparison table will omit baseline scores.", baseline_path)
        return None
    with open(baseline_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_group(
    kpx_group: str,
    n_splits: int,
    feature_set: str,
    early_stopping_rounds: int,
    quantiles: list[float] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run block-aware CV + a final full-data refit for one kpx_group's
    ``GroupLGBMQuantileModel``, using the fixed spec starting hyperparameters
    (no Optuna search -- see module docstring).

    ``quantiles``: probability levels each sub-model is fit at. ``None``
    (default) keeps the original 9-level ``decision_optimize.QUANTILES``
    behavior; experiment_queue.md #4 passes ``QUANTILES_19`` (19 levels) via
    ``--n-quantiles 19`` to test whether finer distribution resolution helps
    the decision-optimal post-processing.

    ``seed``: overrides ``DEFAULT_PARAMS["random_state"]`` for every one of
    the 9 (or 19) quantile sub-models when given (``None`` keeps the default
    42, unchanged behavior) -- experiment_queue.md #8's seed-bagging probe
    trains 3 otherwise-identical models at different seeds to average away
    quantile-estimation variance.
    """
    path = DATA_PROCESSED_DIR / f"features_{kpx_group}_train.parquet"
    df = pd.read_parquet(path)

    n_missing = int(df["target"].isna().sum())
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    logger.info("%s: dropped %d rows with missing target, %d rows remain", kpx_group, n_missing, len(df))

    feature_cols = _get_feature_cols(df, kpx_group=kpx_group, feature_set=feature_set)
    capacity = GROUP_CAPACITY_KWH[kpx_group]

    params: dict[str, Any] = dict(DEFAULT_PARAMS)
    if quantiles is not None:
        params["quantiles"] = list(quantiles)
    if seed is not None:
        params["random_state"] = int(seed)

    oof_df, fold_meta = oof_predict_generic(
        df,
        feature_cols,
        capacity,
        kpx_group,
        GroupLGBMQuantileModel,
        params,
        early_stopping_rounds,
        n_splits,
    )
    fold_metrics = score_oof(oof_df, fold_meta, kpx_group)

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

    # Final refit on ALL available rows -- n_estimators fixed to the CV
    # folds' average best_iteration_ (the q=0.5 sub-model's representative
    # value, see GroupLGBMQuantileModel's docstring), same convention every
    # other model track in this repo uses.
    best_iterations = [f["best_iteration"] for f in fold_metrics if f["best_iteration"]]
    final_params: dict[str, Any] = dict(params)
    if best_iterations:
        final_params["n_estimators"] = max(int(round(float(np.mean(best_iterations)))), 50)

    final_model = GroupLGBMQuantileModel(capacity_kwh=capacity, **final_params)
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
        "final_params": final_params,
        "final_n_estimators": final_model.params["n_estimators"],
        "oof_df": oof_df,
    }


def main() -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train the 9-quantile GBM + decision-theoretic post-processing model for all 3 KPX groups."
    )
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
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument(
        "--n-quantiles",
        type=int,
        choices=(9, 19),
        default=9,
        help=(
            "9 (default, unchanged behavior): decision_optimize.QUANTILES. "
            "19: decision_optimize.QUANTILES_19 (0.05 step, [0.05..0.95]) -- "
            "experiment_queue.md #4's higher-resolution decision-optimization probe."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Overrides DEFAULT_PARAMS['random_state'] (42) for all quantile sub-models. "
            "Default None keeps the original behavior. experiment_queue.md #8's seed-"
            "bagging probe trains 3 runs at different seeds and averages their quantile "
            "arrays before decision-optimization."
        ),
    )
    args = parser.parse_args()

    quantiles = QUANTILES_19 if args.n_quantiles == 19 else QUANTILES

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_gbm_quantile"
    if args.feature_set == "pruned":
        run_id += "_pruned"
    if args.n_quantiles != 9:
        run_id += f"_q{args.n_quantiles}"
    if args.seed is not None:
        run_id += f"_seed{args.seed}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, Any]] = {}
    for kpx_group in KPX_GROUPS:
        logger.info("=== Running %s (GBM-quantile) ===", kpx_group)
        all_results[kpx_group] = run_group(
            kpx_group,
            n_splits=args.n_splits,
            feature_set=args.feature_set,
            early_stopping_rounds=args.early_stopping_rounds,
            quantiles=quantiles,
            seed=args.seed,
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
        "model_type": "lgbm_quantile_decision_optimal",
        "n_splits": args.n_splits,
        "feature_set": args.feature_set,
        "model_default_params": DEFAULT_PARAMS,
        "n_quantiles": args.n_quantiles,
        "quantiles": quantiles,
        "seed": args.seed,
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

    baseline_metrics = _load_baseline_metrics()
    print(f"\n=== LightGBM ({BASELINE_RUN_ID}) vs GBM-quantile ({run_id}) CV Score (mean across folds) ===")
    header = f"{'group':<16}{'lgbm':>12}{'gbm_quantile':>14}{'delta':>12}"
    print(header)
    print("-" * len(header))
    gq_scores = {g: all_results[g]["agg_metrics"]["score_mean"] for g in KPX_GROUPS}
    for g in KPX_GROUPS:
        b = baseline_metrics[g]["agg_metrics"]["score_mean"] if baseline_metrics else float("nan")
        t = gq_scores[g]
        print(f"{g:<16}{b:>12.4f}{t:>14.4f}{t - b:>12.4f}")
    print("-" * len(header))
    b_overall = baseline_metrics["overall"]["score_mean"] if baseline_metrics else float("nan")
    t_overall = overall["score_mean"]
    print(f"{'overall':<16}{b_overall:>12.4f}{t_overall:>14.4f}{t_overall - b_overall:>12.4f}")
    print(f"\nRun dir: {run_dir}")

    return run_id


if __name__ == "__main__":
    main()
