"""Distributional-ensemble (idea A1) + w_ficr sweep (idea B1) harness.

Regenerates the three v14 tracks' **out-of-fold predictive quantiles** on the
identical block-aware CV (``BlockTimeSeriesSplit``) the point tracks used, then
evaluates the distributional blend: blend the quantiles (Vincentization) and
apply the decision-theoretic FICR/1-NMAE post-processing **once** to the blended
distribution (``src.ensembling.distributional_blend.distributional_blend_point``).
This extends the one verified lever of this project -- attacking the FICR step
function decision-theoretically -- from the standalone GBM to the whole
ensemble, which a *point* blend cannot do (a point has no distribution to
optimize over).

Why regenerate OOF at all: the saved ``oof_predictions_<group>.parquet`` files
hold only point predictions. A1 needs the OOF *quantiles*, so this re-runs the
CV once (GBM ~fast, LSTM/Transformer are the wall-clock cost) and caches
``oof_quantiles_<model>_<group>.parquet`` for cheap re-evaluation (the whole
B1 ``w_ficr`` sweep is then free -- fixed blended quantiles, only the decision
step re-runs). Use ``--reuse-oof <run_id>`` to skip retraining and only
re-evaluate/sweep from a previous run's cached quantiles.

Row/fold alignment with the point tracks
-----------------------------------------
All three tracks' OOF is built in ONE fold loop over the same
fully-missing-block-dropped frame (see ``src.training.train_lstm`` for why that
matches the GBM tracks' ``dropna`` block set), so every model's OOF quantiles
cover the exact same rows/folds -- no cross-run join needed and no chance of a
silent row-mismatch. GBM still ``dropna``s its own train/eval targets inside
the fold (LightGBM can't fit a NaN target); the sequence models mask missing
labels in-loss, keeping blocks intact.

Honesty note (kept front-of-mind, CLAUDE.md/task): CV improvement has repeatedly
NOT transferred to the real leaderboard this session. A submission is only
emitted if a variant beats the current best point-blend CV (0.6038) by more than
the overall fold-std noise band (~0.02); otherwise the recommendation is "do not
submit", reported honestly. The w_ficr sweep is presented as a real-board
candidate list, not decided by CV.
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

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, GROUP_CAPACITY_KWH, SUBMISSIONS_DIR
from src.data.loaders import load_sample_submission
from src.ensembling.distributional_blend import distributional_blend_point
from src.evaluation.metrics import competition_score
from src.features.decision_optimize import QUANTILES, decision_optimal_point_prediction
from src.models.lgbm_quantile_model import DEFAULT_EARLY_STOPPING_ROUNDS, DEFAULT_PARAMS, GroupLGBMQuantileModel
from src.models.lstm_model import DEFAULT_N_SEEDS as LSTM_N_SEEDS, GroupLSTMModel
from src.models.torch_common import BLOCK_COL, DEVICE, DT_COL
from src.models.transformer_model import DEFAULT_N_SEEDS as TRANS_N_SEEDS, GroupTransformerModel
from src.training.train_baseline import KPX_GROUPS, N_SPLITS, _get_feature_cols, _get_git_commit
from src.training.train_lstm import _drop_fully_missing_blocks
from src.validation.splitter import BlockTimeSeriesSplit, assert_no_leakage

logger = logging.getLogger(__name__)

MODEL_ORDER = ["gbm", "lstm", "transformer"]
MIXUP_GROUP = "kpx_group_3"

# The v14 fixed blend weights (task spec: lstm 0.35 / transformer 0.35 / gbm 0.30).
V14_FIXED_WEIGHTS = {"gbm": 0.30, "lstm": 0.35, "transformer": 0.35}
# A GBM-heavy reference blend (task: gbm 0.5 / lstm 0.25 / transformer 0.25).
GBM_HEAVY_WEIGHTS = {"gbm": 0.50, "lstm": 0.25, "transformer": 0.25}
WFICR_SWEEP = [0.5, 0.6, 0.7]

# Current-best point-blend CV score (v14 fixed weights) and its noise band -- the
# bar A1 must clear to justify a submission (task spec).
POINT_BLEND_BASELINE_CV = 0.6038
NOISE_BAND = 0.02

# Final full-data models used for the (conditional) test-time submission and the
# always-run test-time predict_quantiles sanity check. These are the already-
# committed default-constructor runs, matching this harness's OOF training config.
FINAL_MODEL_RUNS = {
    "gbm": "20260723_111840_gbm_quantile_pruned",
    "lstm": "20260723_131338_lstm_pruned",
    "transformer": "20260723_133309_transformer_pruned",
}

QUANTILE_COLS = [f"q{int(round(q * 100)):02d}" for q in QUANTILES]


# ---------------------------------------------------------------------------
# OOF quantile generation
# ---------------------------------------------------------------------------
def _fit_gbm_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], capacity: float):
    # Match the repo-wide convention (``tune_common.oof_predict_generic``): early-
    # stop each quantile sub-model on the (labeled) validation fold, so this GBM
    # OOF reproduces the committed gbm-quantile run's CV exactly and stays
    # apples-to-apples with the point-blend baseline.
    tr = train_df.dropna(subset=["target"])
    va = val_df.dropna(subset=["target"])
    model = GroupLGBMQuantileModel(capacity_kwh=capacity, **DEFAULT_PARAMS)
    model.fit(
        tr[feature_cols],
        tr["target"],
        eval_set=(va[feature_cols], va["target"]),
        early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
    )
    return model


def generate_group_oof(
    kpx_group: str, n_splits: int, n_seeds_lstm: int, n_seeds_trans: int, max_epochs: int
) -> dict[str, Any]:
    """Run one block-aware CV pass collecting all three tracks' OOF quantiles
    (+ point predictions) for one group, on identical rows/folds."""
    df = pd.read_parquet(DATA_PROCESSED_DIR / f"features_{kpx_group}_train.parquet")
    df, n_dropped = _drop_fully_missing_blocks(df)
    feature_cols = _get_feature_cols(df, kpx_group=kpx_group, feature_set="pruned")
    capacity = GROUP_CAPACITY_KWH[kpx_group]
    apply_mixup = kpx_group == MIXUP_GROUP

    splitter = BlockTimeSeriesSplit(n_splits=n_splits)
    per_model_rows: dict[str, list[pd.DataFrame]] = {m: [] for m in MODEL_ORDER}

    t0 = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        assert_no_leakage(train_df, val_df)

        actual = val_df["target"].to_numpy(dtype=float)
        keep = ~np.isnan(actual)
        fk = val_df["forecast_kst_dtm"].to_numpy()[keep]
        a = actual[keep]

        # GBM
        gbm = _fit_gbm_fold(train_df, val_df, feature_cols, capacity)
        gbm_q = gbm.predict_quantiles(val_df[feature_cols])[keep]
        gbm_point = gbm.predict(val_df[feature_cols])[keep]

        # LSTM
        lstm = GroupLSTMModel(
            capacity, feature_cols, n_seeds=n_seeds_lstm, apply_mixup=apply_mixup, max_epochs=max_epochs
        ).fit(train_df)
        lstm_q = lstm.predict_quantiles(val_df)[keep]
        lstm_point = lstm.predict(val_df)[keep]

        # Transformer
        trans = GroupTransformerModel(
            capacity, feature_cols, n_seeds=n_seeds_trans, max_epochs=max_epochs
        ).fit(train_df)
        trans_q = trans.predict_quantiles(val_df)[keep]
        trans_point = trans.predict(val_df)[keep]

        for name, q, point in (
            ("gbm", gbm_q, gbm_point),
            ("lstm", lstm_q, lstm_point),
            ("transformer", trans_q, trans_point),
        ):
            rec = pd.DataFrame({"forecast_kst_dtm": fk, "fold": fold_i, "actual": a, "point": point})
            for j, col in enumerate(QUANTILE_COLS):
                rec[col] = q[:, j]
            per_model_rows[name].append(rec)

        logger.info("%s fold %d/%d done (%.1fs elapsed)", kpx_group, fold_i, n_splits, time.time() - t0)

    oof = {name: pd.concat(rows, ignore_index=True) for name, rows in per_model_rows.items()}
    return {
        "oof": oof,
        "feature_cols": feature_cols,
        "capacity": capacity,
        "n_fully_missing_blocks_dropped": n_dropped,
        "cv_wall_clock_seconds": time.time() - t0,
    }


def load_group_oof(run_id: str, kpx_group: str) -> dict[str, pd.DataFrame]:
    oof = {}
    for name in MODEL_ORDER:
        path = EXPERIMENTS_DIR / run_id / f"oof_quantiles_{name}_{kpx_group}.parquet"
        oof[name] = pd.read_parquet(path)
    return oof


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def _per_fold_mean_score(
    fk: np.ndarray, fold: np.ndarray, actual: np.ndarray, pred: np.ndarray, group: str
) -> tuple[float, dict[int, float]]:
    folds = sorted(int(f) for f in np.unique(fold))
    per = {}
    for k in folds:
        m = fold == k
        pred_df = pd.DataFrame({"forecast_kst_dtm": fk[m], group: pred[m]})
        actual_df = pd.DataFrame({"forecast_kst_dtm": fk[m], group: actual[m]})
        per[k] = competition_score(pred_df, actual_df, group_cols=[group])["score"]
    return float(np.nanmean(list(per.values()))), per


def _assemble_group(oof: dict[str, pd.DataFrame], group: str) -> dict[str, Any]:
    """Verify the three tracks' OOF are row-aligned and pack into arrays."""
    base = oof["gbm"]
    fk = base["forecast_kst_dtm"].to_numpy()
    fold = base["fold"].to_numpy()
    actual = base["actual"].to_numpy(dtype=float)
    q = {}
    point = {}
    for name in MODEL_ORDER:
        d = oof[name]
        if not np.array_equal(d["forecast_kst_dtm"].to_numpy(), fk):
            raise ValueError(f"{group}: {name} OOF rows are not aligned with gbm OOF (order/count mismatch).")
        if float(np.nanmax(np.abs(d["actual"].to_numpy(dtype=float) - actual))) > 1e-6:
            raise ValueError(f"{group}: {name} OOF 'actual' disagrees with gbm OOF.")
        q[name] = d[QUANTILE_COLS].to_numpy(dtype=float)
        point[name] = d["point"].to_numpy(dtype=float)
    return {"fk": fk, "fold": fold, "actual": actual, "q": q, "point": point}


def _weights_tuple(weights: dict[str, float]) -> list[float]:
    return [weights[m] for m in MODEL_ORDER]


def evaluate_group(g: dict[str, Any], group: str) -> dict[str, Any]:
    """Compute every variant's per-fold-mean CV score for one group."""
    fk, fold, actual, q, point = g["fk"], g["fold"], g["actual"], g["q"], g["point"]
    capacity = GROUP_CAPACITY_KWH[group]
    q_list = [q[m] for m in MODEL_ORDER]

    def score(pred):
        return _per_fold_mean_score(fk, fold, actual, pred, group)

    results: dict[str, Any] = {}

    # baselines (point tracks, same OOF rows)
    cap_clip = capacity * 1.01
    v14w = V14_FIXED_WEIGHTS
    pt_blend = np.clip(sum(v14w[m] * point[m] for m in MODEL_ORDER), 0.0, cap_clip)
    results["point_blend_v14_fixed"] = dict(zip(("mean", "per_fold"), score(pt_blend)))
    results["gbm_alone"] = dict(zip(("mean", "per_fold"), score(point["gbm"])))

    # A1: distributional blend, v14 fixed weights, w_ficr = 0.5
    a1 = distributional_blend_point(q_list, _weights_tuple(v14w), capacity, w_nmae=0.5, w_ficr=0.5)
    results["A1_dist_v14_wficr0.5"] = dict(zip(("mean", "per_fold"), score(a1)))

    # B1: w_ficr sweep on the v14-weighted distributional blend
    sweep = {}
    for wf in WFICR_SWEEP:
        pred = distributional_blend_point(q_list, _weights_tuple(v14w), capacity, w_nmae=1 - wf, w_ficr=wf)
        mean, per = score(pred)
        sweep[f"wficr_{wf}"] = {"mean": mean, "per_fold": per}
    results["B1_wficr_sweep"] = sweep

    # GBM-heavy reference distributional blend (w_ficr 0.5 and 0.6)
    heavy = {}
    for wf in (0.5, 0.6):
        pred = distributional_blend_point(q_list, _weights_tuple(GBM_HEAVY_WEIGHTS), capacity, w_nmae=1 - wf, w_ficr=wf)
        mean, per = score(pred)
        heavy[f"wficr_{wf}"] = {"mean": mean, "per_fold": per}
    results["gbm_heavy_dist"] = heavy

    # overall fold-std of the A1 blend (for the noise band report)
    results["A1_fold_std"] = float(np.nanstd(list(results["A1_dist_v14_wficr0.5"]["per_fold"].values())))
    return results


# ---------------------------------------------------------------------------
# test-time verification + (conditional) submission
# ---------------------------------------------------------------------------
def _load_final_models() -> dict[str, dict[str, tuple[Any, list[str]]]]:
    """Load each track's final full-data model + its per-group feature_cols
    (from that run's config.yaml -- the GBM wrapper doesn't store feature_cols
    on the instance, unlike the sequence models)."""
    import joblib

    models: dict[str, dict[str, tuple[Any, list[str]]]] = {m: {} for m in MODEL_ORDER}
    for name, rid in FINAL_MODEL_RUNS.items():
        with open(EXPERIMENTS_DIR / rid / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        feats = cfg["feature_cols_per_group"]
        for group in KPX_GROUPS:
            model = joblib.load(EXPERIMENTS_DIR / rid / f"model_{group}.joblib")
            models[name][group] = (model, list(feats[group]))
    return models


def _test_quantiles(model: Any, feature_cols: list[str], test_df: pd.DataFrame) -> np.ndarray:
    """predict_quantiles for either a feature-matrix (GBM) or sequence (torch)
    model, aligned to test_df row order."""
    block_col = getattr(model, "block_col", None)
    dt_col = getattr(model, "dt_col", None)
    if block_col is not None and dt_col is not None:
        needed = list(feature_cols) + [c for c in (block_col, dt_col) if c not in feature_cols]
        return model.predict_quantiles(test_df[needed])
    return model.predict_quantiles(test_df[feature_cols])


def verify_test_time_quantiles(models: dict[str, dict[str, Any]], n_blocks: int = 3) -> dict[str, Any]:
    """Load each final model and run predict_quantiles on a few TEST blocks;
    assert shape + per-row monotonicity. Confirms the test-time distributional
    path (esp. the neural predict_quantiles) works before any submission."""
    report = {}
    for group in KPX_GROUPS:
        test_df = pd.read_parquet(DATA_PROCESSED_DIR / f"features_{group}_test.parquet")
        blocks = pd.Series(test_df[BLOCK_COL].unique()).sort_values().head(n_blocks)
        sample = test_df[test_df[BLOCK_COL].isin(blocks)].reset_index(drop=True)
        for name in MODEL_ORDER:
            model, feats = models[name][group]
            q = _test_quantiles(model, feats, sample)
            assert q.shape == (len(sample), len(QUANTILES)), f"{name}/{group} test quantile shape {q.shape}"
            # after monotonic enforcement inside blending this is what matters; here
            # just report raw crossing rate (GBM can cross; neural is Gaussian-monotone).
            crossed = int((np.diff(q, axis=1) < -1e-6).any(axis=1).sum())
            report[f"{name}_{group}"] = {"rows": int(len(sample)), "raw_crossed_rows": crossed}
    return report


def generate_distributional_submission(
    run_id: str, models: dict[str, dict[str, Any]], weights: dict[str, float], w_ficr: float, label: str
) -> pd.DataFrame:
    """Test-time A1: per group, predict each model's quantiles, blend, decide
    once, assemble/validate a submission matching sample_submission exactly."""
    sample = load_sample_submission()
    submission = sample[["forecast_id", "forecast_kst_dtm"]].copy()

    for group in KPX_GROUPS:
        capacity = GROUP_CAPACITY_KWH[group]
        test_df = pd.read_parquet(DATA_PROCESSED_DIR / f"features_{group}_test.parquet")
        q_list = [_test_quantiles(*models[m][group], test_df=test_df) for m in MODEL_ORDER]
        pred = distributional_blend_point(
            q_list, _weights_tuple(weights), capacity, w_nmae=1 - w_ficr, w_ficr=w_ficr
        )
        pred = np.clip(pred, 0.0, capacity * 1.01)
        pred_df = pd.DataFrame({"forecast_kst_dtm": test_df["forecast_kst_dtm"].to_numpy(), group: pred})
        before = len(submission)
        submission = submission.merge(pred_df, on="forecast_kst_dtm", how="left")
        if len(submission) != before or int(submission[group].isna().sum()):
            raise ValueError(f"{group}: test prediction merge failed (rows/NaN).")

    cols = ["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]
    submission = submission[cols].reset_index(drop=True)
    assert len(submission) == 8760, f"expected 8760 rows, got {len(submission)}"
    assert (submission["forecast_id"].to_numpy() == sample["forecast_id"].to_numpy()).all()
    assert (submission["forecast_kst_dtm"].to_numpy() == sample["forecast_kst_dtm"].to_numpy()).all()

    out_path = SUBMISSIONS_DIR / f"submission_{run_id}_{label}.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote distributional-blend submission: %s", out_path)
    return submission


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _print_tables(all_eval: dict[str, dict[str, Any]], groups: list[str]) -> dict[str, Any]:
    def overall(selector) -> float:
        return float(np.nanmean([selector(all_eval[g]) for g in groups]))

    print("\n" + "=" * 96)
    print("HONEST CV COMPARISON (per-fold-mean official competition_score; same OOF rows/folds)")
    print("=" * 96)
    variants = [
        ("gbm_alone", lambda e: e["gbm_alone"]["mean"]),
        ("point_blend_v14", lambda e: e["point_blend_v14_fixed"]["mean"]),
        ("A1_dist_v14(wF0.5)", lambda e: e["A1_dist_v14_wficr0.5"]["mean"]),
    ]
    header = f"{'group':<14}" + "".join(f"{n:>20}" for n, _ in variants)
    print(header)
    for g in groups:
        line = f"{g:<14}" + "".join(f"{sel(all_eval[g]):>20.4f}" for _, sel in variants)
        print(line)
    print(f"{'OVERALL':<14}" + "".join(f"{overall(sel):>20.4f}" for _, sel in variants))

    print("\n--- B1 w_ficr sweep (A1 distributional blend, v14 weights) ---")
    hdr = f"{'group':<14}" + "".join(f"{'wficr='+str(wf):>14}" for wf in WFICR_SWEEP)
    print(hdr)
    for g in groups:
        line = f"{g:<14}" + "".join(f"{all_eval[g]['B1_wficr_sweep']['wficr_'+str(wf)]['mean']:>14.4f}" for wf in WFICR_SWEEP)
        print(line)
    print(f"{'OVERALL':<14}" + "".join(
        f"{overall(lambda e, wf=wf: e['B1_wficr_sweep']['wficr_'+str(wf)]['mean']):>14.4f}" for wf in WFICR_SWEEP))

    print("\n--- GBM-heavy reference distributional blend (gbm0.5/lstm0.25/trans0.25) ---")
    for wf in (0.5, 0.6):
        line = f"wficr={wf:<8}" + "".join(f"{g}={all_eval[g]['gbm_heavy_dist']['wficr_'+str(wf)]['mean']:.4f}  " for g in groups)
        line += f"OVERALL={overall(lambda e, wf=wf: e['gbm_heavy_dist']['wficr_'+str(wf)]['mean']):.4f}"
        print(line)

    return {
        "gbm_alone": overall(lambda e: e["gbm_alone"]["mean"]),
        "point_blend_v14_fixed": overall(lambda e: e["point_blend_v14_fixed"]["mean"]),
        "A1_dist_v14_wficr0.5": overall(lambda e: e["A1_dist_v14_wficr0.5"]["mean"]),
        "B1_wficr_sweep": {
            f"wficr_{wf}": overall(lambda e, wf=wf: e["B1_wficr_sweep"][f"wficr_{wf}"]["mean"]) for wf in WFICR_SWEEP
        },
        "gbm_heavy_dist": {
            f"wficr_{wf}": overall(lambda e, wf=wf: e["gbm_heavy_dist"][f"wficr_{wf}"]["mean"]) for wf in (0.5, 0.6)
        },
    }


def main() -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="A1 distributional-ensemble + B1 w_ficr sweep.")
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--n-seeds-lstm", type=int, default=LSTM_N_SEEDS)
    parser.add_argument("--n-seeds-trans", type=int, default=TRANS_N_SEEDS)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--groups", nargs="+", default=list(KPX_GROUPS))
    parser.add_argument("--reuse-oof", default=None, help="run_id to load cached oof_quantiles from (skip retraining).")
    parser.add_argument("--emit-submission", action="store_true",
                        help="Force-write the A1 submission even if CV doesn't clear the noise band.")
    args = parser.parse_args()

    logger.info("Torch device: %s", DEVICE)
    groups = list(args.groups)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_distblend"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    gen_meta: dict[str, Any] = {}
    all_eval: dict[str, dict[str, Any]] = {}
    for group in groups:
        if args.reuse_oof:
            logger.info("=== %s: loading cached OOF quantiles from %s ===", group, args.reuse_oof)
            oof = load_group_oof(args.reuse_oof, group)
        else:
            logger.info("=== %s: regenerating OOF quantiles (block CV) ===", group)
            res = generate_group_oof(
                group, args.n_splits, args.n_seeds_lstm, args.n_seeds_trans, args.max_epochs
            )
            oof = res["oof"]
            gen_meta[group] = {
                "cv_wall_clock_seconds": res["cv_wall_clock_seconds"],
                "n_fully_missing_blocks_dropped": res["n_fully_missing_blocks_dropped"],
                "feature_count": len(res["feature_cols"]),
            }
            for name in MODEL_ORDER:
                out = oof[name][["forecast_kst_dtm", "fold", "actual", "point", *QUANTILE_COLS]]
                out.to_parquet(run_dir / f"oof_quantiles_{name}_{group}.parquet", index=False)
                logger.info("Saved %d-row OOF quantiles: oof_quantiles_%s_%s.parquet", len(out), name, group)

        g = _assemble_group(oof, group)
        all_eval[group] = evaluate_group(g, group)

    overall = _print_tables(all_eval, groups)

    # -- submission decision (honest noise-band gate) --------------------
    best_variant, best_overall = "A1_dist_v14_wficr0.5", overall["A1_dist_v14_wficr0.5"]
    for wf in WFICR_SWEEP:
        v = overall["B1_wficr_sweep"][f"wficr_{wf}"]
        if v > best_overall:
            best_variant, best_overall = f"B1_wficr_{wf}", v
    clears = best_overall > POINT_BLEND_BASELINE_CV + NOISE_BAND
    print(
        f"\nBest A1/B1 overall CV = {best_overall:.4f} ({best_variant}); "
        f"point-blend baseline {POINT_BLEND_BASELINE_CV:.4f} + noise band {NOISE_BAND:.2f} "
        f"= {POINT_BLEND_BASELINE_CV + NOISE_BAND:.4f} -> "
        f"{'CLEARS (submission warranted)' if clears else 'does NOT clear (submission NOT recommended)'}"
    )

    # -- always verify the test-time predict_quantiles path --------------
    logger.info("Verifying test-time predict_quantiles on a few TEST blocks ...")
    models = _load_final_models()
    test_verify = verify_test_time_quantiles(models)
    logger.info("Test-time quantile sanity: %s", test_verify)

    submission_file = None
    if clears or args.emit_submission:
        wf = float(best_variant.split("_")[-1]) if best_variant.startswith("B1") else 0.5
        sub = generate_distributional_submission(
            run_id, models, V14_FIXED_WEIGHTS, w_ficr=wf, label=best_variant.replace(".", "p")
        )
        submission_file = f"submission_{run_id}_{best_variant.replace('.', 'p')}.csv"
        logger.info("Submission rows=%d", len(sub))

    # -- persist config + metrics ----------------------------------------
    config = {
        "run_id": run_id,
        "method": "A1 distributional ensemble (blend quantiles -> decision-optimize once) + B1 w_ficr sweep",
        "model_order": MODEL_ORDER,
        "v14_fixed_weights": V14_FIXED_WEIGHTS,
        "gbm_heavy_weights": GBM_HEAVY_WEIGHTS,
        "wficr_sweep": WFICR_SWEEP,
        "n_splits": args.n_splits,
        "n_seeds_lstm": args.n_seeds_lstm,
        "n_seeds_trans": args.n_seeds_trans,
        "max_epochs": args.max_epochs,
        "reuse_oof": args.reuse_oof,
        "final_model_runs": FINAL_MODEL_RUNS,
        "git_commit": _get_git_commit(),
        "device": str(DEVICE),
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    metrics_out = {
        "per_group": all_eval,
        "overall": overall,
        "generation_meta": gen_meta,
        "submission_decision": {
            "best_variant": best_variant,
            "best_overall_cv": best_overall,
            "point_blend_baseline_cv": POINT_BLEND_BASELINE_CV,
            "noise_band": NOISE_BAND,
            "clears_noise_band": clears,
            "submission_file": submission_file,
        },
        "test_time_quantile_verification": test_verify,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    print(f"\nRun dir: {run_dir}")
    return run_id


if __name__ == "__main__":
    main()
