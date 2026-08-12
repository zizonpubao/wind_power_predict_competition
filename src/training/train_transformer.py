"""TransformerEncoder (seq2seq) training pipeline: per-KPX-group block-aware CV
scoring (official ``competition_score``) + a final full-data refit, written to
``experiments/<run_id>/`` -- the Transformer component of the v14 pipeline
reproduction (see ``.claude/plans/logical-stirring-sphinx.md`` Phase D).

This is deliberately the same harness as ``src/training/train_lstm.py`` (the
shared fold-loop/missing-block logic is reused from there), with three
differences that are all v14-spec:
  - it trains ``GroupTransformerModel`` instead of ``GroupLSTMModel``,
  - there is **no mixup** (mixup is a group3-only LSTM rule; the Transformer
    never uses it), and
  - the run_id / comparison table are Transformer-flavored.

Because it reuses ``train_lstm._drop_fully_missing_blocks`` (drop only blocks
whose 24 hours are *all* unlabeled; keep partially-missing blocks intact and
mask their few missing hours in the loss), this run's
``oof_predictions_<group>.parquet`` covers the identical set of forecast blocks
as the LSTM and GBM runs -- exactly what
``src/ensembling/blend_search.load_joined_oof`` needs to join the three tracks
for the Phase E blend.

GPU is used when available (``src.models.torch_common.DEVICE``); per-group
wall-clock is logged so the seed/fold budget can be judged against it.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, GROUP_CAPACITY_KWH
from src.evaluation.metrics import competition_score
from src.models.torch_common import BLOCK_COL, DEVICE, add_ecmwf_all_groups_features
from src.models.transformer_model import DEFAULT_N_SEEDS, GroupTransformerModel
from src.training.train_baseline import FEATURE_SETS, KPX_GROUPS, N_SPLITS, _get_feature_cols, _get_git_commit
from src.training.train_lstm import _drop_fully_missing_blocks, _load_metrics
from src.validation.splitter import BlockTimeSeriesSplit, assert_no_leakage

logger = logging.getLogger(__name__)

# Runs to diff Transformer CV scores against in the printed comparison table.
BASELINE_RUN_ID = "20260722_103636_lgbm_tuned_pruned"  # current-best LightGBM (overall 0.5881)
LSTM_RUN_ID = "20260723_131338_lstm_pruned"  # Phase C LSTM (overall 0.5953)


def run_group(
    kpx_group: str,
    n_splits: int,
    feature_set: str,
    n_seeds: int,
    max_epochs: int,
    ecmwf_all: bool = False,
) -> dict[str, Any]:
    """``ecmwf_all=True`` (experiment_queue.md #20/I): see
    ``src.training.train_lstm.run_group``'s matching docstring -- identical
    feature-set extension, does not touch ``configs/selected_features.json``.
    """
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
    if ecmwf_all:
        df, feature_cols = add_ecmwf_all_groups_features(df, feature_cols)
        logger.info("%s: ecmwf_all=True -> %d feature columns (incl. ecmwf_available)", kpx_group, len(feature_cols))
    capacity = GROUP_CAPACITY_KWH[kpx_group]

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
        model = GroupTransformerModel(
            capacity,
            feature_cols,
            n_seeds=n_seeds,
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
    final_model = GroupTransformerModel(
        capacity, feature_cols, n_seeds=n_seeds, max_epochs=max_epochs
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
        "fold_metrics": fold_metrics,
        "agg_metrics": agg_metrics,
        "cv_wall_clock_seconds": cv_secs,
        "refit_wall_clock_seconds": refit_secs,
        "final_model": final_model,
        "oof_df": oof_df,
    }


def main() -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train the TransformerEncoder seq2seq model for all 3 KPX groups.")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default="pruned",
        help=(
            "'pruned' (default): configs/selected_features.json's per-group gain-based "
            "selection, matching the LightGBM/GBM-quantile/LSTM runs for a fair ensemble "
            "comparison. 'full': all feature columns."
        ),
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(KPX_GROUPS),
        help="Subset of KPX groups to train (default: all 3). Useful for a single-group timing probe.",
    )
    parser.add_argument(
        "--ecmwf-all",
        action="store_true",
        help=(
            "Experiment queue #20 (I): extend the 'pruned' feature set with every group's "
            "ECMWF columns (src.models.torch_common.add_ecmwf_all_groups_features) plus an "
            "ecmwf_available indicator. Does NOT touch configs/selected_features.json -- "
            "recorded only in this run's config.yaml."
        ),
    )
    args = parser.parse_args()

    logger.info("Torch device: %s", DEVICE)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_transformer"
    if args.feature_set == "pruned":
        run_id += "_pruned"
    if args.ecmwf_all:
        run_id += "_ecmwfall"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    groups = list(args.groups)
    all_results: dict[str, dict[str, Any]] = {}
    for kpx_group in groups:
        logger.info("=== Training %s (Transformer, %d seeds, %d splits) ===", kpx_group, args.n_seeds, args.n_splits)
        all_results[kpx_group] = run_group(
            kpx_group,
            n_splits=args.n_splits,
            feature_set=args.feature_set,
            n_seeds=args.n_seeds,
            max_epochs=args.max_epochs,
            ecmwf_all=args.ecmwf_all,
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
        "model_type": "transformer_encoder_seq2seq",
        "device": str(DEVICE),
        "n_splits": args.n_splits,
        "n_seeds": args.n_seeds,
        "max_epochs": args.max_epochs,
        "feature_set": args.feature_set,
        "ecmwf_all": args.ecmwf_all,
        "git_commit": _get_git_commit(),
        "feature_count_per_group": {g: len(all_results[g]["feature_cols"]) for g in groups},
        "feature_cols_per_group": {g: all_results[g]["feature_cols"] for g in groups},
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
    lstm_metrics = _load_metrics(LSTM_RUN_ID)
    print(f"\n=== CV Score (mean across folds): LightGBM vs LSTM vs Transformer ({run_id}) ===")
    header = f"{'group':<16}{'lgbm':>12}{'lstm':>12}{'transf':>12}{'d_vs_lstm':>12}"
    print(header)
    print("-" * len(header))
    for g in groups:
        b = baseline_metrics[g]["agg_metrics"]["score_mean"] if baseline_metrics else float("nan")
        l = lstm_metrics[g]["agg_metrics"]["score_mean"] if lstm_metrics else float("nan")
        t = all_results[g]["agg_metrics"]["score_mean"]
        print(f"{g:<16}{b:>12.4f}{l:>12.4f}{t:>12.4f}{t - l:>12.4f}")
    print("-" * len(header))
    b_overall = baseline_metrics["overall"]["score_mean"] if baseline_metrics else float("nan")
    l_overall = lstm_metrics["overall"]["score_mean"] if lstm_metrics else float("nan")
    print(f"{'overall':<16}{b_overall:>12.4f}{l_overall:>12.4f}{overall['score_mean']:>12.4f}{overall['score_mean'] - l_overall:>12.4f}")
    print(f"\nRun dir: {run_dir}")

    return run_id


if __name__ == "__main__":
    main()
