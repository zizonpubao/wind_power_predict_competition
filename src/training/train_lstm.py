"""LSTM (bidirectional seq2seq) training pipeline: per-KPX-group block-aware CV
scoring (official ``competition_score``) + a final full-data refit, written to
``experiments/<run_id>/`` -- the LSTM component of the v14 pipeline
reproduction (see ``.claude/plans/logical-stirring-sphinx.md`` Phase C).

Harness choice -- option (b), a dedicated sequence-model harness
------------------------------------------------------------------
The GBM/XGBoost tracks run through ``tune_common.oof_predict_generic``, which
(1) passes only ``df[feature_cols]`` to the model -- stripping the
``data_available_kst_dtm`` block key the sequence models need to assemble
24-hour sequences -- and (2) relies on its callers having already
``dropna(subset=["target"])`` before splitting. Both are wrong for a sequence
model: it needs the block key, and dropping individual missing-label rows
would shatter a block's 24-row alignment. Rather than bolt block-key passing
and a "skip the dropna" flag onto that shared, GBM-tuned machinery (option a),
this harness runs its own ``BlockTimeSeriesSplit`` fold loop and hands each
model whole DataFrame slices (features + block key + datetime + target),
leaving ``tune_common.py`` and the existing GBM/XGB/CatBoost behavior
completely untouched.

Missing-label handling -- block-set alignment with the GBM tracks
-----------------------------------------------------------------
Instead of dropping every missing-target row (which the GBM tracks do), this
harness drops only **fully-missing blocks** (blocks whose 24 hours are *all*
unlabeled -- e.g. kpx_group_3's entire 2022) and keeps **partially-missing
blocks intact**, masking their few missing hours inside the loss. This yields
the exact same set of forecast blocks the GBM tracks' ``dropna`` leaves
(dropping every row of an all-missing block == dropping the block; a
partially-missing block survives in both), so after excluding NaN-actual rows
from the saved OOF, this run's ``oof_predictions_<group>.parquet`` covers the
identical rows/folds as every GBM run -- exactly what
``src/ensembling/blend_search.load_joined_oof`` requires to join them.

GPU is used when available (``src.models.torch_common.DEVICE``); the per-group
wall-clock is logged so the seed/fold budget can be judged against it.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, GROUP_CAPACITY_KWH
from src.evaluation.metrics import competition_score
from src.models.lstm_model import DEFAULT_N_SEEDS, GroupLSTMModel
from src.models.torch_common import BLOCK_COL, DEVICE
from src.training.train_baseline import FEATURE_SETS, KPX_GROUPS, N_SPLITS, _get_feature_cols, _get_git_commit
from src.validation.splitter import BlockTimeSeriesSplit, assert_no_leakage

logger = logging.getLogger(__name__)

# group3 gets mixup augmentation (its labels span only 2023-2024) -- v14 rule.
MIXUP_GROUP = "kpx_group_3"

# Runs to diff LSTM CV scores against in the printed comparison table.
BASELINE_RUN_ID = "20260722_103636_lgbm_tuned_pruned"  # current-best LightGBM (overall 0.5881)
GBM_QUANTILE_HINT = "the GBM-quantile run (overall 0.6038)"  # Phase B, for context


def _load_metrics(run_id: str) -> dict[str, Any] | None:
    path = EXPERIMENTS_DIR / run_id / "metrics.json"
    if not path.exists():
        logger.warning("Metrics not found at %s; comparison table will omit its scores.", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _drop_fully_missing_blocks(df: pd.DataFrame, target_col: str = "target") -> tuple[pd.DataFrame, int]:
    """Drop blocks whose every row has a missing target (keep partially-missing
    blocks intact for in-loss masking). Returns (filtered_df, n_blocks_dropped).
    """
    all_missing = df.groupby(BLOCK_COL)[target_col].transform(lambda s: s.isna().all())
    n_dropped = int(df.loc[all_missing, BLOCK_COL].nunique())
    return df.loc[~all_missing].reset_index(drop=True), n_dropped


def run_group(
    kpx_group: str,
    n_splits: int,
    feature_set: str,
    n_seeds: int,
    max_epochs: int,
) -> dict[str, Any]:
    path = DATA_PROCESSED_DIR / f"features_{kpx_group}_train.parquet"
    df = pd.read_parquet(path)

    n_missing_rows = int(df["target"].isna().sum())
    df, n_fully_missing_blocks = _drop_fully_missing_blocks(df)
    n_partial_missing_rows = int(df["target"].isna().sum())
    logger.info(
        "%s: dropped %d fully-missing block(s); %d rows remain (%d still have masked "
        "missing labels inside surviving blocks)",
        kpx_group,
        n_fully_missing_blocks,
        len(df),
        n_partial_missing_rows,
    )

    feature_cols = _get_feature_cols(df, kpx_group=kpx_group, feature_set=feature_set)
    capacity = GROUP_CAPACITY_KWH[kpx_group]
    apply_mixup = kpx_group == MIXUP_GROUP

    splitter = BlockTimeSeriesSplit(n_splits=n_splits)
    oof_records: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    t0 = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        # Extra guard on top of BlockTimeSeriesSplit's own ordering: no block
        # straddles the boundary and all train blocks precede all val blocks.
        assert_no_leakage(train_df, val_df)

        fold_t0 = time.time()
        model = GroupLSTMModel(
            capacity,
            feature_cols,
            n_seeds=n_seeds,
            apply_mixup=apply_mixup,
            max_epochs=max_epochs,
        )
        model.fit(train_df)
        val_pred = model.predict(val_df)  # aligned to val_df rows
        fold_secs = time.time() - fold_t0

        actual = val_df["target"].to_numpy(dtype=float)
        keep = ~np.isnan(actual)  # NaN-actual rows can't be scored/ensembled
        fk = val_df["forecast_kst_dtm"].to_numpy()[keep]
        p = val_pred[keep]
        a = actual[keep]

        oof_records.append(pd.DataFrame({"fold": fold_i, "forecast_kst_dtm": fk, "pred": p, "actual": a}))

        pred_df = pd.DataFrame({"forecast_kst_dtm": fk, kpx_group: p})
        actual_df = pd.DataFrame({"forecast_kst_dtm": fk, kpx_group: a})
        scores = competition_score(pred_df, actual_df, group_cols=[kpx_group])
        fold_metrics.append(
            {
                "fold": fold_i,
                "n_train": int(len(train_idx)),
                "n_val": int(keep.sum()),
                "score": scores["score"],
                "1-NMAE": scores["1-NMAE"],
                "FICR": scores["FICR"],
                "fold_wall_clock_seconds": fold_secs,
            }
        )
        logger.info(
            "%s fold %d/%d: n_train=%d n_val=%d score=%.4f 1-NMAE=%.4f FICR=%.4f (%.1fs)",
            kpx_group,
            fold_i,
            n_splits,
            len(train_idx),
            int(keep.sum()),
            scores["score"],
            scores["1-NMAE"],
            scores["FICR"],
            fold_secs,
        )

    cv_secs = time.time() - t0
    oof_df = pd.concat(oof_records, ignore_index=True)

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

    # Final refit on ALL surviving rows (all non-fully-missing blocks) -- the
    # model used for test inference.
    refit_t0 = time.time()
    final_model = GroupLSTMModel(
        capacity, feature_cols, n_seeds=n_seeds, apply_mixup=apply_mixup, max_epochs=max_epochs
    )
    final_model.fit(df)
    refit_secs = time.time() - refit_t0
    logger.info("%s: final model refit on all %d rows (%.1fs)", kpx_group, len(df), refit_secs)

    return {
        "kpx_group": kpx_group,
        "n_rows_used": len(df),
        "n_missing_target_rows": n_missing_rows,
        "n_fully_missing_blocks_dropped": n_fully_missing_blocks,
        "n_partial_missing_rows_masked": n_partial_missing_rows,
        "feature_cols": feature_cols,
        "apply_mixup": apply_mixup,
        "fold_metrics": fold_metrics,
        "agg_metrics": agg_metrics,
        "cv_wall_clock_seconds": cv_secs,
        "refit_wall_clock_seconds": refit_secs,
        "final_model": final_model,
        "oof_df": oof_df,
    }


def main() -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train the bidirectional-LSTM seq2seq model for all 3 KPX groups.")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default="pruned",
        help=(
            "'pruned' (default): configs/selected_features.json's per-group gain-based "
            "selection, matching the current-best LightGBM/GBM-quantile runs for a fair "
            "ensemble comparison. 'full': all feature columns."
        ),
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(KPX_GROUPS),
        help="Subset of KPX groups to train (default: all 3). Useful for a single-group timing probe.",
    )
    args = parser.parse_args()

    logger.info("Torch device: %s", DEVICE)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_lstm"
    if args.feature_set == "pruned":
        run_id += "_pruned"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    groups = list(args.groups)
    all_results: dict[str, dict[str, Any]] = {}
    for kpx_group in groups:
        logger.info("=== Training %s (LSTM, %d seeds, %d splits) ===", kpx_group, args.n_seeds, args.n_splits)
        all_results[kpx_group] = run_group(
            kpx_group,
            n_splits=args.n_splits,
            feature_set=args.feature_set,
            n_seeds=args.n_seeds,
            max_epochs=args.max_epochs,
        )

        model_path = run_dir / f"model_{kpx_group}.joblib"
        all_results[kpx_group]["final_model"].save(model_path)
        logger.info("Saved final model: %s", model_path)

        oof_path = run_dir / f"oof_predictions_{kpx_group}.parquet"
        oof_out = all_results[kpx_group]["oof_df"][["forecast_kst_dtm", "pred", "actual", "fold"]]
        oof_out.to_parquet(oof_path, index=False)
        logger.info("Saved OOF predictions (%d rows): %s", len(oof_out), oof_path)

    config = {
        "run_id": run_id,
        "model_type": "lstm_bidirectional_seq2seq",
        "device": str(DEVICE),
        "n_splits": args.n_splits,
        "n_seeds": args.n_seeds,
        "max_epochs": args.max_epochs,
        "feature_set": args.feature_set,
        "mixup_group": MIXUP_GROUP,
        "git_commit": _get_git_commit(),
        "feature_count_per_group": {g: len(all_results[g]["feature_cols"]) for g in groups},
        "feature_cols_per_group": {g: all_results[g]["feature_cols"] for g in groups},
        "apply_mixup_per_group": {g: all_results[g]["apply_mixup"] for g in groups},
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    metrics_out: dict[str, Any] = {}
    for g in groups:
        metrics_out[g] = {
            "n_rows_used": all_results[g]["n_rows_used"],
            "n_missing_target_rows": all_results[g]["n_missing_target_rows"],
            "n_fully_missing_blocks_dropped": all_results[g]["n_fully_missing_blocks_dropped"],
            "cv_wall_clock_seconds": all_results[g]["cv_wall_clock_seconds"],
            "refit_wall_clock_seconds": all_results[g]["refit_wall_clock_seconds"],
            "fold_metrics": all_results[g]["fold_metrics"],
            "agg_metrics": all_results[g]["agg_metrics"],
        }
    overall = {
        "score_mean": float(np.nanmean([all_results[g]["agg_metrics"]["score_mean"] for g in groups])),
        "1-NMAE_mean": float(np.nanmean([all_results[g]["agg_metrics"]["1-NMAE_mean"] for g in groups])),
        "FICR_mean": float(np.nanmean([all_results[g]["agg_metrics"]["FICR_mean"] for g in groups])),
    }
    metrics_out["overall"] = overall
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    baseline_metrics = _load_metrics(BASELINE_RUN_ID)
    print(f"\n=== LightGBM ({BASELINE_RUN_ID}) vs LSTM ({run_id}) CV Score (mean across folds) ===")
    header = f"{'group':<16}{'lgbm':>12}{'lstm':>12}{'delta':>12}"
    print(header)
    print("-" * len(header))
    for g in groups:
        b = baseline_metrics[g]["agg_metrics"]["score_mean"] if baseline_metrics else float("nan")
        t = all_results[g]["agg_metrics"]["score_mean"]
        print(f"{g:<16}{b:>12.4f}{t:>12.4f}{t - b:>12.4f}")
    print("-" * len(header))
    b_overall = baseline_metrics["overall"]["score_mean"] if baseline_metrics else float("nan")
    print(f"{'overall':<16}{b_overall:>12.4f}{overall['score_mean']:>12.4f}{overall['score_mean'] - b_overall:>12.4f}")
    print(f"\nRun dir: {run_dir}")

    return run_id


if __name__ == "__main__":
    main()
