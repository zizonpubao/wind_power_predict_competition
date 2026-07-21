"""Test-time inference: load the 3 per-group final models trained by
``src.training.train_baseline`` (saved under ``experiments/<run_id>/``),
predict on the processed test feature tables, and assemble a submission CSV
matching ``sample_submission.csv``'s exact schema and row order.

``forecast_id`` is never regenerated here -- it, and the row order, come
straight from ``load_sample_submission()`` via a left join on
``forecast_kst_dtm``, per CLAUDE.md / the code-writer role brief.

If ``experiments/<run_id>/calibrator_<group>.joblib`` exists (written by
``src.training.tune_hyperparams`` / ``src.training.evaluate_calibration
--save-calibrators`` when the isotonic OOF-residual calibration was found to
genuinely improve CV score -- see ``reports/eda/ficr_gap_diagnosis.md``),
it is applied to that group's raw predictions: predict -> calibrate -> clip
to capacity -> assemble submission. Calibration runs on top of
``model.predict()``'s own capacity-clipped output (the calibrator was itself
fit on those same clipped OOF predictions), and the capacity clip is
re-applied afterward since isotonic regression can map a near-capacity input
slightly above the safety bound -- clipping stays the very last
post-processing step either way.
"""
from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, SUBMISSIONS_DIR
from src.data.loaders import load_sample_submission
from src.models.calibration import PredictionCalibrator

logger = logging.getLogger(__name__)

KPX_GROUPS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")
SUBMISSION_COLS = ["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]

# Must match src.training.train_baseline.NON_FEATURE_COLS so the feature list
# a model was fit on and the feature list it's asked to predict on line up.
NON_FEATURE_COLS = {"forecast_kst_dtm", "data_available_kst_dtm", "target"}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def generate_submission(run_id: str) -> pd.DataFrame:
    """Load experiments/<run_id>'s 3 final models, predict on the test feature
    tables, and write submissions/submission_<run_id>.csv. Returns the
    assembled submission dataframe.
    """
    run_dir = EXPERIMENTS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    sample = load_sample_submission()
    submission = sample[["forecast_id", "forecast_kst_dtm"]].copy()

    for kpx_group in KPX_GROUPS:
        model_path = run_dir / f"model_{kpx_group}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file for {kpx_group}: {model_path}")
        model = joblib.load(model_path)

        test_path = DATA_PROCESSED_DIR / f"features_{kpx_group}_test.parquet"
        test_df = pd.read_parquet(test_path)
        feature_cols = _get_feature_cols(test_df)

        preds = model.predict(test_df[feature_cols])

        calibrator_path = run_dir / f"calibrator_{kpx_group}.joblib"
        if calibrator_path.exists():
            calibrator = PredictionCalibrator.load(calibrator_path)
            preds = calibrator.transform(preds)
            # Re-clip after calibration: the calibrator was fit on already
            # capacity-clipped OOF predictions, but isotonic regression can
            # still map a near-capacity input a little above the bound, so
            # the safety clamp stays the last step before assembling preds.
            preds = np.clip(preds, 0.0, model.capacity_kwh * 1.01)
            logger.info("%s: applied calibrator %s", kpx_group, calibrator_path.name)

        pred_df = pd.DataFrame(
            {"forecast_kst_dtm": test_df["forecast_kst_dtm"].to_numpy(), kpx_group: preds}
        )

        before_len = len(submission)
        submission = submission.merge(pred_df, on="forecast_kst_dtm", how="left")
        if len(submission) != before_len:
            raise ValueError(
                f"Merge for {kpx_group} changed row count ({before_len} -> {len(submission)}); "
                f"test parquet likely has duplicate forecast_kst_dtm values."
            )

        n_missing = int(submission[kpx_group].isna().sum())
        if n_missing:
            raise ValueError(
                f"{n_missing} rows could not be matched to a {kpx_group} test prediction "
                f"(forecast_kst_dtm mismatch between sample_submission.csv and the test parquet)."
            )

    submission = submission[SUBMISSION_COLS].reset_index(drop=True)

    # --- Sanity checks against sample_submission.csv (assert, don't eyeball) ---
    assert len(submission) == 8760, f"expected 8,760 rows, got {len(submission)}"
    assert list(submission.columns) == SUBMISSION_COLS, (
        f"column mismatch: {list(submission.columns)} != {SUBMISSION_COLS}"
    )
    assert (submission["forecast_id"].to_numpy() == sample["forecast_id"].to_numpy()).all(), (
        "forecast_id does not exactly match sample_submission.csv"
    )
    assert (submission["forecast_kst_dtm"].to_numpy() == sample["forecast_kst_dtm"].to_numpy()).all(), (
        "forecast_kst_dtm does not exactly match sample_submission.csv"
    )

    out_path = SUBMISSIONS_DIR / f"submission_{run_id}.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote submission: %s (%d rows)", out_path, len(submission))

    return submission


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Generate a submission CSV from a trained experiments/<run_id> run."
    )
    parser.add_argument("run_id", help="experiments/<run_id> directory name to load models from")
    args = parser.parse_args()
    generate_submission(args.run_id)


if __name__ == "__main__":
    main()
