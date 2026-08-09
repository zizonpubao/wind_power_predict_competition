"""Seed-bagging variant of ``src.training.train_gbm_quantile``: fits
``GroupLGBMQuantileModel`` at several ``random_state`` seeds per fold/final
refit, averages the *raw quantile arrays* (before monotonic enforcement /
decision-optimization) across seeds, then runs the shared decision-theoretic
post-processing once on the averaged quantiles.

experiment_queue.md #8: quantile-estimation variance reduction via bagging.
Averaging at the quantile-array level (not the final decided point) matters --
it lets ``enforce_monotonic_quantiles``/``decision_optimal_point_prediction``
see a genuinely smoother 9-point distribution estimate rather than just
averaging 3 already-decided scalars, which would throw away the extra
distributional information bagging is meant to sharpen.

Mirrors ``train_gbm_quantile.py``'s CV-then-final-refit structure and CLI/
output shape as closely as possible (same ``configs/selected_features.json``
pruned feature set, same ``BlockTimeSeriesSplit``, same
``experiments/<run_id>/`` artifact layout: one model per seed *and* group is
saved so any seed can be inspected individually) but cannot reuse
``tune_common.oof_predict_generic`` directly, since that helper only ever
sees a model's already-decided ``predict()`` output -- this script needs the
intermediate ``predict_quantiles()`` array to average across seeds before
decision-optimization runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, GROUP_CAPACITY_KWH
from src.evaluation.metrics import competition_score
from src.features.decision_optimize import (
    QUANTILES,
    decision_optimal_point_prediction,
    enforce_monotonic_quantiles,
)
from src.models.lgbm_quantile_model import DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_PARAMS, GroupLGBMQuantileModel
from src.training.train_baseline import KPX_GROUPS, N_SPLITS, _get_feature_cols, _get_git_commit
from src.validation.splitter import BlockTimeSeriesSplit

logger = logging.getLogger(__name__)

DEFAULT_SEEDS: list[int] = [42, 202, 777]


def _score_fold(forecast_kst_dtm: np.ndarray, pred: np.ndarray, actual: np.ndarray, kpx_group: str) -> dict[str, float]:
    pred_df = pd.DataFrame({"forecast_kst_dtm": forecast_kst_dtm, kpx_group: pred})
    actual_df = pd.DataFrame({"forecast_kst_dtm": forecast_kst_dtm, kpx_group: actual})
    return competition_score(pred_df, actual_df, group_cols=[kpx_group])


def run_group(
    kpx_group: str,
    n_splits: int,
    feature_set: str,
    early_stopping_rounds: int,
    seeds: list[int],
) -> dict[str, Any]:
    """Block-aware CV (seed-averaged quantiles) + a final full-data refit
    (also seed-averaged) for one kpx_group.
    """
    path = DATA_PROCESSED_DIR / f"features_{kpx_group}_train.parquet"
    df = pd.read_parquet(path)

    n_missing = int(df["target"].isna().sum())
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    logger.info("%s: dropped %d rows with missing target, %d rows remain", kpx_group, n_missing, len(df))

    feature_cols = _get_feature_cols(df, kpx_group=kpx_group, feature_set=feature_set)
    capacity = GROUP_CAPACITY_KWH[kpx_group]

    splitter = BlockTimeSeriesSplit(n_splits=n_splits)
    oof_records: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        X_train, y_train = train_df[feature_cols], train_df["target"]
        X_val, y_val = val_df[feature_cols], val_df["target"]

        seed_quantile_preds = []
        fold_best_iters = []
        for seed in seeds:
            model = GroupLGBMQuantileModel(capacity_kwh=capacity, random_state=seed, **{
                k: v for k, v in DEFAULT_PARAMS.items() if k != "random_state"
            })
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=early_stopping_rounds)
            seed_quantile_preds.append(model.predict_quantiles(X_val))
            if model.best_iteration_:
                fold_best_iters.append(model.best_iteration_)

        avg_q = np.mean(np.stack(seed_quantile_preds, axis=0), axis=0)  # (n_rows, n_quantiles)
        sorted_q = enforce_monotonic_quantiles(avg_q)
        val_pred = decision_optimal_point_prediction(sorted_q, QUANTILES, capacity)
        val_pred = np.clip(val_pred, 0.0, capacity * 1.01)

        oof_records.append(
            pd.DataFrame(
                {
                    "fold": fold_i,
                    "forecast_kst_dtm": val_df["forecast_kst_dtm"].to_numpy(),
                    "pred": val_pred,
                    "actual": y_val.to_numpy(),
                }
            )
        )
        scores = _score_fold(val_df["forecast_kst_dtm"].to_numpy(), val_pred, y_val.to_numpy(), kpx_group)
        best_iter = int(round(float(np.mean(fold_best_iters)))) if fold_best_iters else None
        if best_iter:
            best_iterations.append(best_iter)
        fold_metrics.append(
            {
                "fold": fold_i,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "score": scores["score"],
                "1-NMAE": scores["1-NMAE"],
                "FICR": scores["FICR"],
                "best_iteration": best_iter,
            }
        )

    oof_df = pd.concat(oof_records, ignore_index=True)

    agg_metrics = {
        "score_mean": float(np.nanmean([f["score"] for f in fold_metrics])),
        "score_std": float(np.nanstd([f["score"] for f in fold_metrics])),
        "1-NMAE_mean": float(np.nanmean([f["1-NMAE"] for f in fold_metrics])),
        "1-NMAE_std": float(np.nanstd([f["1-NMAE"] for f in fold_metrics])),
        "FICR_mean": float(np.nanmean([f["FICR"] for f in fold_metrics])),
        "FICR_std": float(np.nanstd([f["FICR"] for f in fold_metrics])),
    }

    final_n_estimators = max(int(round(float(np.mean(best_iterations)))), 50) if best_iterations else DEFAULT_PARAMS["n_estimators"]

    final_models: dict[int, GroupLGBMQuantileModel] = {}
    for seed in seeds:
        params = {k: v for k, v in DEFAULT_PARAMS.items() if k != "random_state"}
        params["n_estimators"] = final_n_estimators
        model = GroupLGBMQuantileModel(capacity_kwh=capacity, random_state=seed, **params)
        model.fit(df[feature_cols], df["target"])
        final_models[seed] = model
    logger.info("%s: %d final seed models refit on all %d rows (n_estimators=%d)", kpx_group, len(seeds), len(df), final_n_estimators)

    return {
        "kpx_group": kpx_group,
        "n_rows_used": len(df),
        "n_missing_target_dropped": n_missing,
        "feature_cols": feature_cols,
        "fold_metrics": fold_metrics,
        "agg_metrics": agg_metrics,
        "final_models": final_models,
        "final_n_estimators": final_n_estimators,
        "oof_df": oof_df,
    }


class SeedBaggedGBMQuantileModel:
    """Predict-time wrapper: averages ``predict_quantiles`` across a dict of
    per-seed ``GroupLGBMQuantileModel``s, then runs the shared decision-
    theoretic post-processing once. Mirrors ``GroupLGBMQuantileModel``'s
    ``predict(X)`` signature so ``src.inference.predict.generate_submission``
    can use it exactly like any other saved model.
    """

    def __init__(self, models: dict[int, GroupLGBMQuantileModel], capacity_kwh: float):
        self.models = models
        self.capacity_kwh = float(capacity_kwh)
        # best_iteration_ kept for interface parity; not meaningful post-bagging.
        self.best_iteration_ = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        seed_q = [m.predict_quantiles(X) for m in self.models.values()]
        avg_q = np.mean(np.stack(seed_q, axis=0), axis=0)
        sorted_q = enforce_monotonic_quantiles(avg_q)
        decision = decision_optimal_point_prediction(sorted_q, QUANTILES, self.capacity_kwh)
        return np.clip(decision, 0.0, self.capacity_kwh * 1.01)


def main() -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # When this file is executed as `python -m src.training.train_gbm_seedbag`,
    # Python runs it as the `__main__` module, so `SeedBaggedGBMQuantileModel`
    # (defined at this module's top level) gets `__class__.__module__ ==
    # "__main__"` and joblib pickles it with that module reference -- any
    # other process trying to `joblib.load()` the saved model later (e.g.
    # `src.inference.predict`) then fails with "Can't get attribute
    # 'SeedBaggedGBMQuantileModel' on <module ...>" because it's a different
    # `__main__`. The fix has two parts, both required: (1) alias this
    # already-loaded `__main__` module object under its real dotted path in
    # `sys.modules` -- so `sys.modules["src.training.train_gbm_seedbag"] is
    # sys.modules["__main__"]`, i.e. the *same* module object, not a second,
    # separately-imported copy (pickle's `save_global` requires
    # `getattr(sys.modules[module_name], name) is obj` by identity, which a
    # fresh `import src.training.train_gbm_seedbag` would fail since that
    # creates a distinct class object from the `__main__` one currently in
    # use); (2) only then is it safe to rewrite `__module__` to the dotted
    # path, since that lookup will now resolve to the identical class.
    if __name__ == "__main__" and "src.training.train_gbm_seedbag" not in sys.modules:
        sys.modules["src.training.train_gbm_seedbag"] = sys.modules["__main__"]
    if SeedBaggedGBMQuantileModel.__module__ == "__main__":
        SeedBaggedGBMQuantileModel.__module__ = "src.training.train_gbm_seedbag"

    parser = argparse.ArgumentParser(
        description="Seed-bagged 9-quantile GBM: average quantile arrays across seeds before decision-optimization."
    )
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--feature-set", choices=("full", "pruned"), default="pruned")
    parser.add_argument("--early-stopping-rounds", type=int, default=DEFAULT_EARLY_STOPPING_ROUNDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_gbm_seedbag"
    if args.feature_set == "pruned":
        run_id += "_pruned"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, Any]] = {}
    for kpx_group in KPX_GROUPS:
        logger.info("=== Running %s (GBM seed-bag, seeds=%s) ===", kpx_group, args.seeds)
        all_results[kpx_group] = run_group(
            kpx_group,
            n_splits=args.n_splits,
            feature_set=args.feature_set,
            early_stopping_rounds=args.early_stopping_rounds,
            seeds=args.seeds,
        )

        # OOF predictions saved BEFORE the model dump: they're the expensive,
        # irreplaceable-without-a-full-CV-rerun artifact, whereas the final
        # model is comparatively cheap to reproduce -- if joblib.dump ever
        # fails again (e.g. a future pickling edge case), the CV work already
        # done for this group isn't silently lost.
        oof_path = run_dir / f"oof_predictions_{kpx_group}.parquet"
        oof_out = all_results[kpx_group]["oof_df"][["forecast_kst_dtm", "pred", "actual", "fold"]]
        oof_out.to_parquet(oof_path, index=False)
        logger.info("Saved OOF predictions (%d rows): %s", len(oof_out), oof_path)

        capacity = GROUP_CAPACITY_KWH[kpx_group]
        wrapper = SeedBaggedGBMQuantileModel(all_results[kpx_group]["final_models"], capacity)
        model_path = run_dir / f"model_{kpx_group}.joblib"
        joblib.dump(wrapper, model_path)
        logger.info("Saved seed-bagged final model: %s", model_path)

    config = {
        "run_id": run_id,
        "model_type": "lgbm_quantile_seedbag_decision_optimal",
        "n_splits": args.n_splits,
        "feature_set": args.feature_set,
        "seeds": args.seeds,
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

    print(f"\n=== GBM seed-bag ({run_id}, seeds={args.seeds}) CV Score (mean across folds) ===")
    header = f"{'group':<16}{'score':>12}"
    print(header)
    print("-" * len(header))
    for g in KPX_GROUPS:
        print(f"{g:<16}{all_results[g]['agg_metrics']['score_mean']:>12.4f}")
    print("-" * len(header))
    print(f"{'overall':<16}{overall['score_mean']:>12.4f}")
    print(f"\nRun dir: {run_dir}")

    return run_id


if __name__ == "__main__":
    main()
