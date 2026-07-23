"""Optuna hyperparameter re-tuning for the sequence models (LSTM / Transformer)
of the v14 pipeline reproduction -- shared machinery behind
``src/training/tune_lstm.py`` and ``src/training/tune_transformer.py``.

Why a dedicated tuner instead of ``tune_common.tune_group_generic``
-------------------------------------------------------------------
``tune_common``'s generic Optuna loop drives models through
``oof_predict_generic``, which hands each model a *bare feature matrix*
(``df[feature_cols]``) -- stripping the ``data_available_kst_dtm`` block key the
sequence models need to assemble 24-hour sequences (same reason
``train_lstm.py`` exists as a separate harness). So this module reuses
``train_lstm``'s own block-aware fold loop conventions (whole-DataFrame slices,
``_drop_fully_missing_blocks``, in-loss masking of partially-missing blocks)
rather than the GBM-tuned generic path, keeping ``tune_common.py`` untouched.

Honest-expectations note (see the task brief / CLAUDE.md §5)
-----------------------------------------------------------
The hand-off doc records that an LSTM random search (15 combos) and a
Transformer layer-count sweep both *failed to help in the real leaderboard*,
and today's own experiments confirmed CV-optimization (nested-LOFO) lost to a
fixed blend in the real world. This tuner is therefore run with the prior that
**CV improvement does not guarantee real-world improvement**, and it is built
to *stop early and report "no meaningful gain"* rather than manufacture one:

  * Cost control: search uses reduced seeds/folds (``--tune-n-seeds`` /
    ``--tune-n-splits``); the winning config is refit at the model's full
    seed count and 5 folds only once, at the end.
  * A per-group reduced-config **baseline** (default/starting hyperparameters,
    same reduced seeds/folds) is measured first -- its per-fold std is the
    noise yardstick, and its wall-clock is the measured 1-trial cost used to
    project (and cap, via ``--time-budget-seconds``) the search.
  * ``NoiseTimeEarlyStop`` halts a group's study once either (a) after
    ``--min-trials`` trials the best CV still sits within the noise band above
    the reduced baseline, or (b) the per-group wall-clock budget is exceeded.

Everything is scored with the official ``competition_score`` and split only
with ``BlockTimeSeriesSplit`` -- never a plain/random split.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
import yaml

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, GROUP_CAPACITY_KWH
from src.evaluation.metrics import competition_score
from src.models.torch_common import DEVICE
from src.training.train_baseline import KPX_GROUPS, N_SPLITS, _get_feature_cols, _get_git_commit
from src.training.train_lstm import _drop_fully_missing_blocks, _load_metrics
from src.validation.splitter import BlockTimeSeriesSplit, assert_no_leakage

logger = logging.getLogger(__name__)

SAMPLER_SEED = 42

# --- cost-control / early-stop defaults (see module docstring) ---
DEFAULT_N_TRIALS = 30
DEFAULT_TUNE_N_SEEDS = 2          # reduced from 6 (LSTM) / 4 (Transformer) during search
DEFAULT_TUNE_N_SPLITS = 3         # reduced from 5 during search (>= 3 per the brief)
DEFAULT_MIN_TRIALS = 12           # don't early-stop before this many trials
DEFAULT_NOISE_FLOOR = 0.02        # min "meaningful" delta even if fold-std is tinier
DEFAULT_TIME_BUDGET_SECONDS = 600  # per-group wall-clock cap for the search itself


# A ``make_model`` factory: zero-arg callable returning a fresh, unfitted model.
MakeModel = Callable[[], Any]


def build_arg_parser(model_kind: str, default_n_seeds: int) -> argparse.ArgumentParser:
    """Argparse namespace shared by the LSTM/Transformer tuning CLIs. The only
    per-model difference is the default full-refit seed count (6 for the LSTM,
    4 for the Transformer)."""
    p = argparse.ArgumentParser(description=f"Optuna re-tuning of the {model_kind} seq2seq model per KPX group.")
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Max Optuna trials per group (may stop early).")
    p.add_argument("--n-splits", type=int, default=N_SPLITS, help="Folds for the final full-config refit CV (5).")
    p.add_argument("--n-seeds", type=int, default=default_n_seeds, help="Seeds for the final full-config refit.")
    p.add_argument("--tune-n-seeds", type=int, default=DEFAULT_TUNE_N_SEEDS, help="Reduced seeds during search.")
    p.add_argument("--tune-n-splits", type=int, default=DEFAULT_TUNE_N_SPLITS, help="Reduced folds during search (>=3).")
    p.add_argument("--min-trials", type=int, default=DEFAULT_MIN_TRIALS, help="Don't early-stop before this many trials.")
    p.add_argument("--noise-floor", type=float, default=DEFAULT_NOISE_FLOOR, help="Min meaningful CV delta over baseline.")
    p.add_argument("--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS, help="Per-group search cap.")
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--feature-set", choices=("full", "pruned"), default="pruned")
    p.add_argument("--groups", nargs="+", default=list(KPX_GROUPS), help="Subset of KPX groups (default: all 3).")
    return p


def _cv_fold_metrics(
    df: pd.DataFrame,
    kpx_group: str,
    make_model: MakeModel,
    n_splits: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Block-aware CV fold loop for a sequence model, identical in shape to
    ``train_lstm.run_group``'s inner loop: fit on each fold's train blocks,
    predict its val blocks, score with the official ``competition_score``.

    Returns ``(fold_metrics, oof_df)`` where ``oof_df`` has columns
    ``fold``/``forecast_kst_dtm``/``pred``/``actual`` (NaN-actual val rows
    excluded, so it aligns with the GBM/LSTM/Transformer OOF parquets).
    """
    splitter = BlockTimeSeriesSplit(n_splits=n_splits)
    oof_records: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        assert_no_leakage(train_df, val_df)

        fold_t0 = time.time()
        model = make_model()
        model.fit(train_df)
        val_pred = model.predict(val_df)
        fold_secs = time.time() - fold_t0

        actual = val_df["target"].to_numpy(dtype=float)
        keep = ~np.isnan(actual)
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

    oof_df = pd.concat(oof_records, ignore_index=True)
    return fold_metrics, oof_df


def _agg(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "score_mean": float(np.nanmean([f["score"] for f in fold_metrics])),
        "score_std": float(np.nanstd([f["score"] for f in fold_metrics])),
        "1-NMAE_mean": float(np.nanmean([f["1-NMAE"] for f in fold_metrics])),
        "1-NMAE_std": float(np.nanstd([f["1-NMAE"] for f in fold_metrics])),
        "FICR_mean": float(np.nanmean([f["FICR"] for f in fold_metrics])),
        "FICR_std": float(np.nanstd([f["FICR"] for f in fold_metrics])),
    }


class NoiseTimeEarlyStop:
    """Optuna callback that stops a study when tuning is either (a) not beating
    the reduced-config baseline by more than the noise band, checked only after
    ``min_trials`` trials, or (b) over the per-group wall-clock budget.

    Records ``stopped_early`` / ``stop_reason`` for the run report.
    """

    def __init__(self, baseline_mean: float, noise_thresh: float, min_trials: int, time_budget_s: float):
        self.baseline_mean = baseline_mean
        self.noise_thresh = noise_thresh
        self.min_trials = min_trials
        self.time_budget_s = time_budget_s
        self.t0 = time.time()
        self.stopped_early = False
        self.stop_reason: str | None = None

    def decide(self, n_done: int, best_value: float, elapsed: float) -> str | None:
        """Pure stop-decision (no side effects): return a human-readable reason
        to stop, or ``None`` to keep going. Split out from ``__call__`` so the
        conditions are unit-testable without an active Optuna optimize loop
        (``study.stop()`` may only be called from inside one)."""
        if elapsed > self.time_budget_s:
            return f"time budget exceeded ({elapsed:.0f}s > {self.time_budget_s:.0f}s)"
        if n_done >= self.min_trials and best_value - self.baseline_mean < self.noise_thresh:
            return (
                f"no meaningful gain after {n_done} trials "
                f"(best {best_value:.4f} - baseline {self.baseline_mean:.4f} = "
                f"{best_value - self.baseline_mean:+.4f} < noise {self.noise_thresh:.4f})"
            )
        return None

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        reason = self.decide(n_done, study.best_value, time.time() - self.t0)
        if reason is not None:
            self.stopped_early = True
            self.stop_reason = reason
            study.stop()


def tune_group(
    kpx_group: str,
    model_cls: Callable[..., Any],
    suggest_fn: Callable[[optuna.Trial], dict[str, Any]],
    extra_kwargs_fn: Callable[[str], dict[str, Any]],
    feature_set: str,
    n_trials: int,
    n_seeds: int,
    n_splits: int,
    tune_n_seeds: int,
    tune_n_splits: int,
    max_epochs: int,
    min_trials: int,
    noise_floor: float,
    time_budget_s: float,
) -> dict[str, Any]:
    """Tune one KPX group's sequence model, then refit the winning config at the
    full seed count / 5 folds and return everything needed to write standard
    ``experiments/<run_id>/`` artifacts.

    ``model_cls(capacity_kwh, feature_cols, n_seeds=..., max_epochs=...,
    **extra, **suggested)`` must construct the model (matches both
    ``GroupLSTMModel`` and ``GroupTransformerModel`` signatures). ``extra`` is
    supplied per group by ``extra_kwargs_fn`` (e.g. ``apply_mixup=True`` for the
    LSTM on group3; ``{}`` for the Transformer).
    """
    path = DATA_PROCESSED_DIR / f"features_{kpx_group}_train.parquet"
    df = pd.read_parquet(path)

    n_missing_rows = int(df["target"].isna().sum())
    df, n_fully_missing_blocks = _drop_fully_missing_blocks(df)
    feature_cols = _get_feature_cols(df, kpx_group=kpx_group, feature_set=feature_set)
    capacity = GROUP_CAPACITY_KWH[kpx_group]
    extra = extra_kwargs_fn(kpx_group)

    # --- reduced-config baseline (default/starting hyperparameters) ---
    # This measures both the noise yardstick (per-fold std) and the ~1-trial
    # wall-clock, exactly at the reduced seeds/folds the search will use.
    def make_default() -> Any:
        return model_cls(capacity, feature_cols, n_seeds=tune_n_seeds, max_epochs=max_epochs, **extra)

    t0 = time.time()
    base_fm, _ = _cv_fold_metrics(df, kpx_group, make_default, tune_n_splits)
    baseline_reduced_secs = time.time() - t0
    base_agg = _agg(base_fm)
    baseline_mean = base_agg["score_mean"]
    baseline_std = base_agg["score_std"]
    noise_thresh = max(baseline_std, noise_floor)
    projected_total = baseline_reduced_secs * n_trials
    logger.info(
        "%s: reduced baseline (%d seeds, %d folds) score=%.4f +-%.4f in %.1fs/trial "
        "(noise band=%.4f); projected %d-trial search ~%.0fs (per-group cap %.0fs)",
        kpx_group, tune_n_seeds, tune_n_splits, baseline_mean, baseline_std,
        baseline_reduced_secs, noise_thresh, n_trials, projected_total, time_budget_s,
    )

    # --- Optuna search (reduced config) ---
    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)

        def mk() -> Any:
            return model_cls(
                capacity, feature_cols, n_seeds=tune_n_seeds, max_epochs=max_epochs, **extra, **params
            )

        fm, _ = _cv_fold_metrics(df, kpx_group, mk, tune_n_splits)
        return float(np.nanmean([f["score"] for f in fm]))

    sampler = optuna.samplers.TPESampler(seed=SAMPLER_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    stopper = NoiseTimeEarlyStop(baseline_mean, noise_thresh, min_trials, time_budget_s)

    search_t0 = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[stopper], show_progress_bar=False)
    search_secs = time.time() - search_t0

    best_params = dict(study.best_params)
    best_reduced = study.best_value
    logger.info(
        "%s: study done (%d trials, %.0fs, %.1fs/trial), best_reduced=%.4f (baseline %.4f, "
        "delta %+.4f), early_stop=%s (%s), best_params=%s",
        kpx_group, len(study.trials), search_secs, search_secs / max(len(study.trials), 1),
        best_reduced, baseline_mean, best_reduced - baseline_mean, stopper.stopped_early,
        stopper.stop_reason, best_params,
    )

    # --- refit winning config at FULL seeds / 5 folds ---
    def make_best_full() -> Any:
        return model_cls(capacity, feature_cols, n_seeds=n_seeds, max_epochs=max_epochs, **extra, **best_params)

    full_t0 = time.time()
    fold_metrics, oof_df = _cv_fold_metrics(df, kpx_group, make_best_full, n_splits)
    full_cv_secs = time.time() - full_t0
    agg_metrics = _agg(fold_metrics)

    refit_t0 = time.time()
    final_model = make_best_full()
    final_model.fit(df)
    refit_secs = time.time() - refit_t0
    logger.info(
        "%s: full-config CV (%d seeds, %d folds) score=%.4f +-%.4f in %.0fs; refit on all %d rows in %.0fs",
        kpx_group, n_seeds, n_splits, agg_metrics["score_mean"], agg_metrics["score_std"],
        full_cv_secs, len(df), refit_secs,
    )

    trials_records = [
        {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)} for t in study.trials
    ]

    return {
        "kpx_group": kpx_group,
        "n_rows_used": len(df),
        "n_missing_target_rows": n_missing_rows,
        "n_fully_missing_blocks_dropped": n_fully_missing_blocks,
        "feature_cols": feature_cols,
        "extra_kwargs": extra,
        "reduced_baseline_score_mean": baseline_mean,
        "reduced_baseline_score_std": baseline_std,
        "reduced_baseline_secs_per_trial": baseline_reduced_secs,
        "noise_threshold": noise_thresh,
        "best_reduced_score": best_reduced,
        "delta_reduced_vs_baseline": best_reduced - baseline_mean,
        "best_params": best_params,
        "n_trials_run": len(study.trials),
        "stopped_early": stopper.stopped_early,
        "stop_reason": stopper.stop_reason,
        "search_wall_clock_seconds": search_secs,
        "full_cv_wall_clock_seconds": full_cv_secs,
        "refit_wall_clock_seconds": refit_secs,
        "fold_metrics": fold_metrics,
        "agg_metrics": agg_metrics,
        "trials": trials_records,
        "final_model": final_model,
        "oof_df": oof_df,
    }


def run_tuning(
    model_kind: str,
    model_type_label: str,
    model_cls: Callable[..., Any],
    suggest_fn: Callable[[optuna.Trial], dict[str, Any]],
    extra_kwargs_fn: Callable[[str], dict[str, Any]],
    default_n_seeds: int,
    baseline_run_id: str,
    args: Any,
) -> str:
    """Drive ``tune_group`` for every KPX group, write standard artifacts
    (``model_<g>.joblib`` / ``oof_predictions_<g>.parquet`` / ``config.yaml`` /
    ``metrics.json`` / ``tuning_results.json``) to ``experiments/<run_id>/``,
    and print an honest comparison table vs the pre-tuning run.

    ``args`` is the parsed argparse namespace from the thin CLI wrappers.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("Torch device: %s", DEVICE)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{model_kind}_tuned"
    if args.feature_set == "pruned":
        run_id += "_pruned"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    groups = list(args.groups)
    all_results: dict[str, dict[str, Any]] = {}
    for kpx_group in groups:
        logger.info(
            "=== Tuning %s (%s, up to %d trials, reduced %d seeds/%d folds; refit %d seeds/%d folds) ===",
            kpx_group, model_kind, args.n_trials, args.tune_n_seeds, args.tune_n_splits, args.n_seeds, args.n_splits,
        )
        all_results[kpx_group] = tune_group(
            kpx_group,
            model_cls=model_cls,
            suggest_fn=suggest_fn,
            extra_kwargs_fn=extra_kwargs_fn,
            feature_set=args.feature_set,
            n_trials=args.n_trials,
            n_seeds=args.n_seeds,
            n_splits=args.n_splits,
            tune_n_seeds=args.tune_n_seeds,
            tune_n_splits=args.tune_n_splits,
            max_epochs=args.max_epochs,
            min_trials=args.min_trials,
            noise_floor=args.noise_floor,
            time_budget_s=args.time_budget_seconds,
        )

        model_path = run_dir / f"model_{kpx_group}.joblib"
        all_results[kpx_group]["final_model"].save(model_path)
        logger.info("Saved final tuned model: %s", model_path)

        oof_path = run_dir / f"oof_predictions_{kpx_group}.parquet"
        oof_out = all_results[kpx_group]["oof_df"][["forecast_kst_dtm", "pred", "actual", "fold"]]
        oof_out.to_parquet(oof_path, index=False)
        logger.info("Saved OOF predictions (%d rows): %s", len(oof_out), oof_path)

    config = {
        "run_id": run_id,
        "model_type": model_type_label,
        "device": str(DEVICE),
        "n_splits": args.n_splits,
        "n_seeds": args.n_seeds,
        "max_epochs": args.max_epochs,
        "feature_set": args.feature_set,
        "git_commit": _get_git_commit(),
        "tuning_n_trials_max": args.n_trials,
        "tuning_sampler": "TPESampler",
        "tuning_sampler_seed": SAMPLER_SEED,
        "tuning_reduced_n_seeds": args.tune_n_seeds,
        "tuning_reduced_n_splits": args.tune_n_splits,
        "tuning_min_trials": args.min_trials,
        "tuning_noise_floor": args.noise_floor,
        "tuning_time_budget_seconds_per_group": args.time_budget_seconds,
        "feature_count_per_group": {g: len(all_results[g]["feature_cols"]) for g in groups},
        "feature_cols_per_group": {g: all_results[g]["feature_cols"] for g in groups},
        "tuned_params_per_group": {g: all_results[g]["best_params"] for g in groups},
        "extra_kwargs_per_group": {g: all_results[g]["extra_kwargs"] for g in groups},
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    metrics_out: dict[str, Any] = {}
    for g in groups:
        metrics_out[g] = {
            "n_rows_used": all_results[g]["n_rows_used"],
            "n_missing_target_rows": all_results[g]["n_missing_target_rows"],
            "n_fully_missing_blocks_dropped": all_results[g]["n_fully_missing_blocks_dropped"],
            "full_cv_wall_clock_seconds": all_results[g]["full_cv_wall_clock_seconds"],
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

    tuning_out: dict[str, Any] = {}
    for g in groups:
        r = all_results[g]
        tuning_out[g] = {
            "n_trials_run": r["n_trials_run"],
            "n_trials_max": args.n_trials,
            "sampler_seed": SAMPLER_SEED,
            "reduced_baseline_score_mean": r["reduced_baseline_score_mean"],
            "reduced_baseline_score_std": r["reduced_baseline_score_std"],
            "reduced_baseline_secs_per_trial": r["reduced_baseline_secs_per_trial"],
            "noise_threshold": r["noise_threshold"],
            "best_reduced_score": r["best_reduced_score"],
            "delta_reduced_vs_baseline": r["delta_reduced_vs_baseline"],
            "best_params": r["best_params"],
            "stopped_early": r["stopped_early"],
            "stop_reason": r["stop_reason"],
            "search_wall_clock_seconds": r["search_wall_clock_seconds"],
            "trials": r["trials"],
        }
    with open(run_dir / "tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(tuning_out, f, indent=2)

    # --- honest comparison table: pre-tuning run vs re-tuned (full-config CV) ---
    prev = _load_metrics(baseline_run_id)
    print(f"\n=== {model_type_label}: pre-tuning ({baseline_run_id}) vs re-tuned ({run_id}) full-config CV ===")
    header = f"{'group':<16}{'pre-tune':>12}{'re-tuned':>12}{'delta':>12}{'noise':>10}{'early_stop':>12}"
    print(header)
    print("-" * len(header))
    for g in groups:
        b = prev[g]["agg_metrics"]["score_mean"] if prev else float("nan")
        t = all_results[g]["agg_metrics"]["score_mean"]
        noise = all_results[g]["noise_threshold"]
        es = "Y" if all_results[g]["stopped_early"] else "n"
        print(f"{g:<16}{b:>12.4f}{t:>12.4f}{t - b:>12.4f}{noise:>10.4f}{es:>12}")
    print("-" * len(header))
    b_overall = prev["overall"]["score_mean"] if prev else float("nan")
    print(f"{'overall':<16}{b_overall:>12.4f}{overall['score_mean']:>12.4f}{overall['score_mean'] - b_overall:>12.4f}")
    print(f"\nRun dir: {run_dir}")

    return run_id
