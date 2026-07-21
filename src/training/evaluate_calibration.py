"""Evaluate whether an OOF-residual-based isotonic calibration fixes the
systematic under-prediction bias found in ``reports/eda/ficr_gap_diagnosis.md``
(the tuned LightGBM run under-predicts 62-66% of eligible hours, mean signed
error -721 to -1,246 kWh, worst at high generation).

For each of the 3 KPX groups:
  1. Reconstruct genuine OOF predictions the same way the diagnosis report
     did: refit ``GroupLGBMModel`` per CV fold with the tuned
     ``best_params`` from ``experiments/<TUNED_RUN_ID>/tuning_results.json``,
     over the same ``BlockTimeSeriesSplit``. Reuses
     ``src.training.tune_hyperparams._oof_predict`` (the exact same fit loop
     ``tune_hyperparams._cv_score``/the diagnosis report use) instead of
     duplicating it.
  2. Fit a ``PredictionCalibrator`` (isotonic pred -> actual) and evaluate its
     effect on OOF predictions it did NOT see during its own fit.
  3. Print a before/after comparison table: score / 1-NMAE / FICR / tier
     distribution, per group and overall, aggregated the same way
     ``train_baseline.py``/``tune_hyperparams.py`` do (mean of per-fold
     ``competition_score``, then mean across the 3 groups for "overall").

Methodological care -- avoiding leakage in the *evaluation* itself
--------------------------------------------------------------------
Fitting the calibrator on the exact same OOF (pred, actual) pairs it is then
scored on would leak: the calibrator would be allowed to "see" the residuals
it's being evaluated against, producing an optimistic before/after delta.

This script uses **leave-one-fold-out cross-fitting**: the 5 CV folds
already computed by ``_oof_predict`` double as the cross-fitting folds. For
each fold k, an isotonic calibrator is fit on the *other 4 folds'* OOF
(pred, actual) pairs only, then applied to fold k's predictions; the 5
resulting calibrated-fold predictions are concatenated back into one series
(exactly the shape ``_score_oof`` expects, so the existing per-fold
``competition_score`` scoring path can be reused unchanged). This is chosen
over a temporal walk-forward split (fit on folds 1..k-1, apply to fold k)
because walk-forward would leave fold 1 with no prior fold to calibrate
from and is otherwise no safer here: a calibrator is a global monotonic
pred -> actual curve, not a feature that could encode future information
into a row's own features, so the leakage that actually matters is a row
being calibrated by a curve fit including that row (or its fold-mates) --
which leave-one-fold-out cross-fitting rules out for every row, by
construction. This mirrors the leave-one-fold-out approach ``sklearn``'s own
``CalibratedClassifierCV`` uses for probability calibration.

The *production* calibrator saved for inference (see ``tune_hyperparams.py``,
if this evaluation shows a genuine improvement) is different on purpose: it
is fit on ALL 5 folds' pooled OOF pairs, exactly analogous to how the final
model refit uses all training rows -- there is no "held-out" set left to
protect once the artifact is deployed on genuinely unseen test data.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, GROUP_CAPACITY_KWH
from src.models.calibration import PredictionCalibrator
from src.training.train_baseline import KPX_GROUPS, N_SPLITS, _get_feature_cols
from src.training.tune_hyperparams import _oof_predict, _score_oof

logger = logging.getLogger(__name__)

TUNED_RUN_ID = "20260720_161850_lgbm_tuned"


def _load_best_params(run_id: str, kpx_group: str) -> dict:
    path = EXPERIMENTS_DIR / run_id / "tuning_results.json"
    with open(path, "r", encoding="utf-8") as f:
        tuning_results = json.load(f)
    return dict(tuning_results[kpx_group]["best_params"])


def _cross_fit_calibrated_column(oof_df: pd.DataFrame) -> np.ndarray:
    """Leave-one-fold-out cross-fit calibration (see module docstring).

    For each fold present in ``oof_df["fold"]``, fits a fresh
    ``PredictionCalibrator`` on every OTHER fold's (pred, actual) pairs, and
    uses it to transform that fold's own ``pred`` column. Returns a numpy
    array aligned with ``oof_df``'s row order (a "pred_calibrated" column
    where no row was ever transformed by a calibrator that saw that row, or
    any row sharing its fold, during fitting).
    """
    calibrated = np.empty(len(oof_df), dtype=float)
    folds = sorted(oof_df["fold"].unique())
    for fold_i in folds:
        fit_mask = (oof_df["fold"] != fold_i).to_numpy()
        apply_mask = (oof_df["fold"] == fold_i).to_numpy()

        calibrator = PredictionCalibrator()
        calibrator.fit(
            oof_df.loc[fit_mask, "pred"].to_numpy(),
            oof_df.loc[fit_mask, "actual"].to_numpy(),
        )
        calibrated[apply_mask] = calibrator.transform(oof_df.loc[apply_mask, "pred"].to_numpy())
    return calibrated


def _tier_distribution(oof_df: pd.DataFrame, capacity: float, pred_col: str) -> dict[str, float]:
    """Recompute the ficr_gap_diagnosis.md-style tier distribution (% of
    eligible hours in each nMAE settlement tier) for a given prediction
    column, pooled across all folds.
    """
    eligible = oof_df[oof_df["actual"] >= 0.10 * capacity]
    if len(eligible) == 0:
        return {"n_eligible": 0, "pct_le_6": float("nan"), "pct_6_8": float("nan"), "pct_gt_8": float("nan")}

    nmae_h = (eligible[pred_col] - eligible["actual"]).abs() / capacity
    n = len(nmae_h)
    return {
        "n_eligible": int(n),
        "pct_le_6": float((nmae_h <= 0.06).sum() / n * 100),
        "pct_6_8": float(((nmae_h > 0.06) & (nmae_h <= 0.08)).sum() / n * 100),
        "pct_gt_8": float((nmae_h > 0.08).sum() / n * 100),
    }


def evaluate_group(kpx_group: str, tuned_run_id: str = TUNED_RUN_ID, n_splits: int = N_SPLITS) -> dict:
    path = DATA_PROCESSED_DIR / f"features_{kpx_group}_train.parquet"
    df = pd.read_parquet(path)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    feature_cols = _get_feature_cols(df)
    capacity = GROUP_CAPACITY_KWH[kpx_group]
    best_params = _load_best_params(tuned_run_id, kpx_group)

    logger.info("%s: reconstructing OOF predictions with tuned best_params ...", kpx_group)
    oof_df, fold_meta = _oof_predict(df, feature_cols, capacity, kpx_group, best_params, n_splits=n_splits)

    logger.info("%s: leave-one-fold-out cross-fit calibration ...", kpx_group)
    oof_df = oof_df.copy()
    oof_df["pred_calibrated"] = _cross_fit_calibrated_column(oof_df)

    raw_fold_metrics = _score_oof(oof_df, fold_meta, kpx_group, pred_col="pred")
    calib_fold_metrics = _score_oof(oof_df, fold_meta, kpx_group, pred_col="pred_calibrated")

    def _agg(fold_metrics: list[dict]) -> dict:
        return {
            "score_mean": float(np.nanmean([f["score"] for f in fold_metrics])),
            "1-NMAE_mean": float(np.nanmean([f["1-NMAE"] for f in fold_metrics])),
            "FICR_mean": float(np.nanmean([f["FICR"] for f in fold_metrics])),
        }

    raw_agg = _agg(raw_fold_metrics)
    calib_agg = _agg(calib_fold_metrics)

    raw_tiers = _tier_distribution(oof_df, capacity, "pred")
    calib_tiers = _tier_distribution(oof_df, capacity, "pred_calibrated")

    # Mean signed error, for a quick before/after bias sanity check (matches
    # the "-721 to -1,246 kWh" figures from ficr_gap_diagnosis.md section 3).
    eligible = oof_df[oof_df["actual"] >= 0.10 * capacity]
    raw_mean_signed_err = float((eligible["pred"] - eligible["actual"]).mean())
    calib_mean_signed_err = float((eligible["pred_calibrated"] - eligible["actual"]).mean())

    # Production calibrator: fit on ALL folds' pooled OOF pairs (not the
    # cross-fit version used for scoring above) -- see module docstring for
    # why this is not leakage for a deployed artifact. Only actually saved by
    # main() if the leak-free evaluation above shows a genuine improvement.
    production_calibrator = PredictionCalibrator().fit(
        oof_df["pred"].to_numpy(), oof_df["actual"].to_numpy()
    )

    return {
        "kpx_group": kpx_group,
        "n_oof_rows": len(oof_df),
        "raw_agg": raw_agg,
        "calib_agg": calib_agg,
        "raw_tiers": raw_tiers,
        "calib_tiers": calib_tiers,
        "raw_mean_signed_err": raw_mean_signed_err,
        "calib_mean_signed_err": calib_mean_signed_err,
        "production_calibrator": production_calibrator,
    }


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate isotonic OOF-residual calibration vs. the raw tuned LightGBM predictions."
    )
    parser.add_argument("--run-id", default=TUNED_RUN_ID, help="experiments/<run_id> to load tuned best_params from")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument(
        "--save-calibrators",
        action="store_true",
        help=(
            "If the leak-free evaluation shows a genuine overall CV score improvement, save each "
            "group's production PredictionCalibrator (fit on ALL pooled OOF rows) as "
            "experiments/<run-id>/calibrator_<group>.joblib, ready for src/inference/predict.py to load."
        ),
    )
    args = parser.parse_args()

    results = {}
    for kpx_group in KPX_GROUPS:
        logger.info("=== Evaluating calibration for %s ===", kpx_group)
        results[kpx_group] = evaluate_group(kpx_group, tuned_run_id=args.run_id, n_splits=args.n_splits)

    print(f"\n=== Calibration evaluation (tuned run: {args.run_id}) ===")
    print("Cross-fit (leak-free) before/after comparison, mean across CV folds\n")

    header = f"{'group':<14}{'metric':<10}{'raw':>10}{'calibrated':>12}{'delta':>10}"
    print(header)
    print("-" * len(header))
    for g in KPX_GROUPS:
        r = results[g]
        for metric, key in (("score", "score_mean"), ("1-NMAE", "1-NMAE_mean"), ("FICR", "FICR_mean")):
            raw_v = r["raw_agg"][key]
            calib_v = r["calib_agg"][key]
            print(f"{g:<14}{metric:<10}{raw_v:>10.4f}{calib_v:>12.4f}{calib_v - raw_v:>10.4f}")
        print(
            f"{'':<14}{'mean_err':<10}{r['raw_mean_signed_err']:>10.1f}{r['calib_mean_signed_err']:>12.1f}"
            f"{r['calib_mean_signed_err'] - r['raw_mean_signed_err']:>10.1f}"
        )
        print(
            f"{'':<14}{'tier<=6%':<10}{r['raw_tiers']['pct_le_6']:>9.2f}%{r['calib_tiers']['pct_le_6']:>11.2f}%"
            f"{r['calib_tiers']['pct_le_6'] - r['raw_tiers']['pct_le_6']:>9.2f}%"
        )
        print(
            f"{'':<14}{'tier6-8%':<10}{r['raw_tiers']['pct_6_8']:>9.2f}%{r['calib_tiers']['pct_6_8']:>11.2f}%"
            f"{r['calib_tiers']['pct_6_8'] - r['raw_tiers']['pct_6_8']:>9.2f}%"
        )
        print(
            f"{'':<14}{'tier>8%':<10}{r['raw_tiers']['pct_gt_8']:>9.2f}%{r['calib_tiers']['pct_gt_8']:>11.2f}%"
            f"{r['calib_tiers']['pct_gt_8'] - r['raw_tiers']['pct_gt_8']:>9.2f}%"
        )
        print("-" * len(header))

    overall_raw = {
        "score_mean": float(np.mean([results[g]["raw_agg"]["score_mean"] for g in KPX_GROUPS])),
        "1-NMAE_mean": float(np.mean([results[g]["raw_agg"]["1-NMAE_mean"] for g in KPX_GROUPS])),
        "FICR_mean": float(np.mean([results[g]["raw_agg"]["FICR_mean"] for g in KPX_GROUPS])),
    }
    overall_calib = {
        "score_mean": float(np.mean([results[g]["calib_agg"]["score_mean"] for g in KPX_GROUPS])),
        "1-NMAE_mean": float(np.mean([results[g]["calib_agg"]["1-NMAE_mean"] for g in KPX_GROUPS])),
        "FICR_mean": float(np.mean([results[g]["calib_agg"]["FICR_mean"] for g in KPX_GROUPS])),
    }
    for metric, key in (("score", "score_mean"), ("1-NMAE", "1-NMAE_mean"), ("FICR", "FICR_mean")):
        raw_v = overall_raw[key]
        calib_v = overall_calib[key]
        print(f"{'OVERALL':<14}{metric:<10}{raw_v:>10.4f}{calib_v:>12.4f}{calib_v - raw_v:>10.4f}")

    improves = overall_calib["score_mean"] > overall_raw["score_mean"]
    verdict = "IMPROVES" if improves else "DOES NOT IMPROVE"
    print(
        f"\nVerdict: calibration {verdict} the overall CV score "
        f"({overall_raw['score_mean']:.4f} -> {overall_calib['score_mean']:.4f}, "
        f"delta {overall_calib['score_mean'] - overall_raw['score_mean']:+.4f})"
    )

    if args.save_calibrators:
        if improves:
            run_dir = EXPERIMENTS_DIR / args.run_id
            for g in KPX_GROUPS:
                calib_path = run_dir / f"calibrator_{g}.joblib"
                results[g]["production_calibrator"].save(calib_path)
                logger.info("Saved production calibrator: %s", calib_path)
            print(f"\nSaved production calibrators to {run_dir}")
        else:
            print("\n--save-calibrators given but verdict was DOES NOT IMPROVE -- not saving anything.")

    return {"per_group": results, "overall_raw": overall_raw, "overall_calib": overall_calib}


if __name__ == "__main__":
    main()
