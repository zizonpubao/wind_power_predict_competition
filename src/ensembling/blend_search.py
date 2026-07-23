"""Per-KPX-group ensemble weight search over multiple already-trained models'
OOF predictions, validated with nested leave-one-fold-out (LOFO) CV so the
reported score is not optimistic.

The weight-search machinery (``weight_grid``/``load_joined_oof``/
``nested_lofo_search``/``generate_blend_submission``) is fully generic in
``model_names``: it was written for the LightGBM/XGBoost/CatBoost pruned blend
(``DEFAULT_RUNS``/``BASELINE_MODEL_NAME`` below) but takes any set of runs whose
OOF predictions share the same CV splitter and label source. The v14 pipeline
reproduction (Phase E) reuses it unchanged for a gbm-quantile / LSTM /
Transformer blend by passing ``--runs name=run_id`` on the CLI -- see ``main``.

Why nested LOFO, not "fit weights on all folds' OOF then re-score on that same
OOF": that would let the grid search fit noise in the exact rows it is then
judged on -- a blend that looks better on training-period metrics but was never
evaluated the same way the base models were (CLAUDE.md's own "leakage" caution,
and the ensembler agent brief's explicit warning). Instead, for each CV fold k:
  1. Grid-search the best simplex weight (step 0.05, weights >= 0, sum to 1)
     that maximizes the official ``competition_score`` on the OOF rows from the
     OTHER folds pooled together.
  2. Freeze that weight and evaluate it on fold k's OOF rows alone (rows the
     weight search never saw).
Averaging the held-out scores gives a score directly comparable to each base
model's own mean-of-folds CV score (same held-out rows, same metric).

A "production" weight (fit on all folds pooled, maximum data) is also computed
for deployment IF a group's nested LOFO score shows a genuine, fold-consistent
improvement over the baseline model's own CV score; groups that don't clear
that bar fall back to the pure baseline model (weight 1 on it) rather than
being forced into an ensemble that doesn't actually help.

Test-time note (torch models): the sequence models (LSTM/Transformer) don't
keep a live ``nn.Module`` -- ``joblib.load`` restores a wrapper whose
``predict`` takes a *DataFrame* carrying the forecast-block key + datetime, not
a bare feature matrix like the GBM wrappers. ``_predict_group_test`` detects
this via the wrapper's ``block_col``/``dt_col`` attributes and passes the extra
columns through, so a blend can mix GBM (joblib feature-matrix predict) and
torch (block-sequence predict) models transparently.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from configs.paths import DATA_PROCESSED_DIR, EXPERIMENTS_DIR, SUBMISSIONS_DIR
from src.data.loaders import load_sample_submission
from src.evaluation.metrics import competition_score
from src.training.train_baseline import KPX_GROUPS

logger = logging.getLogger(__name__)

# Default runs this blend combines -- the LightGBM/XGBoost/CatBoost pruned
# blend, all trained on the identical pruned feature set + group2 wake features
# + identical 5-fold BlockTimeSeriesSplit, so their OOF rows/folds line up
# exactly (verified in main() before any weight search runs). Override on the
# CLI with ``--runs name=run_id`` (e.g. the v14 gbm/lstm/transformer blend).
DEFAULT_RUNS: dict[str, str] = {
    "lgbm": "20260722_103636_lgbm_tuned_pruned",
    "xgb": "20260722_105650_xgb_tuned_pruned",
    "cat": "20260722_105902_catboost_tuned_pruned",
}
# The run whose standalone CV score is the bar a blend must clear per group.
BASELINE_MODEL_NAME = "lgbm"

WEIGHT_STEP = 0.05
# A group's nested-LOFO blend is only adopted over pure baseline if it beats
# the baseline's own mean CV score AND does so in a majority of the folds
# (not just on average) -- guards against one lucky fold masking a blend that
# is actually worse most of the time.
MIN_FOLDS_IMPROVED = 3


def weight_grid(model_names: list[str], step: float = WEIGHT_STEP) -> list[tuple[float, ...]]:
    """All weight tuples over ``model_names`` on the ``step``-resolution
    simplex (each weight a multiple of ``step``, all >= 0, summing to 1).

    Only implemented for 2 or 3 models (this task's use case); raises for
    anything else rather than silently doing the wrong thing.
    """
    n = round(1.0 / step)
    if len(model_names) == 3:
        grid = []
        for i in range(n + 1):
            for j in range(n + 1 - i):
                k = n - i - j
                grid.append((round(i * step, 4), round(j * step, 4), round(k * step, 4)))
        return grid
    if len(model_names) == 2:
        return [(round(i * step, 4), round(1.0 - i * step, 4)) for i in range(n + 1)]
    raise ValueError(f"weight_grid only supports 2 or 3 models, got {len(model_names)}")


def load_joined_oof(runs: dict[str, str], group: str) -> pd.DataFrame:
    """Join every run's ``oof_predictions_<group>.parquet`` on
    ``forecast_kst_dtm`` into one frame with columns ``forecast_kst_dtm``,
    ``fold``, ``actual``, and ``pred_<name>`` for each ``name`` in ``runs``.

    Asserts the join doesn't drop rows (i.e. every run covers the exact same
    OOF rows) and that ``fold``/``actual`` agree across runs -- both are
    expected since all runs share the same CV splitter and label source, and
    a mismatch would mean the runs aren't actually comparable.
    """
    base: pd.DataFrame | None = None
    for name, run_id in runs.items():
        path = EXPERIMENTS_DIR / run_id / f"oof_predictions_{group}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{name} ({run_id}): missing {path.name} -- ensembler cannot blend a run "
                f"without its OOF predictions on disk (see module docstring; report this "
                f"back rather than training the missing artifact ad hoc, unless it can be "
                f"deterministically regenerated from already-recorded hyperparameters)."
            )
        df = pd.read_parquet(path).rename(columns={"pred": f"pred_{name}"})
        cols = ["forecast_kst_dtm", "fold", "actual", f"pred_{name}"]
        if base is None:
            base = df[cols]
            continue
        before = len(base)
        merged = base.merge(
            df[["forecast_kst_dtm", "fold", "actual", f"pred_{name}"]],
            on="forecast_kst_dtm",
            how="inner",
            suffixes=("", f"_{name}"),
        )
        if len(merged) != before:
            raise ValueError(
                f"{group}: joining {name}'s OOF predictions dropped rows "
                f"({before} -> {len(merged)}) -- runs do not share the same OOF rows."
            )
        fold_mismatch = int((merged["fold"] != merged[f"fold_{name}"]).sum())
        actual_mismatch = float((merged["actual"] - merged[f"actual_{name}"]).abs().max())
        if fold_mismatch:
            raise ValueError(f"{group}: {fold_mismatch} row(s) disagree on fold assignment between runs.")
        if actual_mismatch > 1e-6:
            raise ValueError(f"{group}: actual/label mismatch between runs (max diff {actual_mismatch}).")
        base = merged.drop(columns=[f"fold_{name}", f"actual_{name}"])
    assert base is not None
    return base.reset_index(drop=True)


def blend_score(df: pd.DataFrame, group: str, model_names: list[str], weights: tuple[float, ...]) -> float:
    """Official ``competition_score`` of the weighted-average blend of
    ``pred_<name>`` columns against ``actual``, on the given rows.
    """
    blended = sum(w * df[f"pred_{name}"] for name, w in zip(model_names, weights))
    pred_df = pd.DataFrame({"forecast_kst_dtm": df["forecast_kst_dtm"].to_numpy(), group: blended.to_numpy()})
    actual_df = pd.DataFrame({"forecast_kst_dtm": df["forecast_kst_dtm"].to_numpy(), group: df["actual"].to_numpy()})
    return competition_score(pred_df, actual_df, group_cols=[group])["score"]


def per_fold_mean_score(
    df: pd.DataFrame, group: str, model_names: list[str], weights: tuple[float, ...]
) -> tuple[float, dict[int, float]]:
    """Mean over folds of a **fixed** (not fitted) weight's per-fold held-out
    ``competition_score``.

    Because the weight is fixed, evaluating it fold-by-fold and averaging is an
    honest, directly-comparable CV number -- exactly how each base model's own
    ``score_mean`` is computed (mean of per-fold held-out scores). Used for the
    v14 fixed-weight and single-track rows of the comparison table, so they sit
    on the same footing as the nested-LOFO blend and each other.
    """
    folds = sorted(int(f) for f in df["fold"].unique())
    per_fold = {k: blend_score(df[df["fold"] == k], group, model_names, weights) for k in folds}
    return float(np.mean(list(per_fold.values()))), per_fold


def weights_dict_to_tuple(model_names: list[str], weights: dict[str, float]) -> tuple[float, ...]:
    """Convert a ``{name: weight}`` mapping into a tuple aligned to
    ``model_names`` order (missing names default to 0.0). Raises if the weights
    don't sum to ~1 or reference an unknown model name.
    """
    unknown = set(weights) - set(model_names)
    if unknown:
        raise ValueError(f"fixed weights reference unknown model(s): {sorted(unknown)} (known: {model_names})")
    tup = tuple(float(weights.get(name, 0.0)) for name in model_names)
    if abs(sum(tup) - 1.0) > 1e-6:
        raise ValueError(f"fixed weights must sum to 1.0, got {sum(tup)} for {weights}")
    return tup


def best_weight(
    df: pd.DataFrame, group: str, model_names: list[str], grid: list[tuple[float, ...]]
) -> tuple[tuple[float, ...], float]:
    """Grid search: the weight tuple in ``grid`` maximizing ``blend_score`` on ``df``."""
    best_w, best_s = grid[0], -np.inf
    for w in grid:
        s = blend_score(df, group, model_names, w)
        if s > best_s:
            best_s, best_w = s, w
    return best_w, best_s


def nested_lofo_search(
    df: pd.DataFrame, group: str, model_names: list[str], grid: list[tuple[float, ...]]
) -> list[dict[str, Any]]:
    """Nested leave-one-fold-out weight search (see module docstring): for
    each fold, fit the best weight on the other folds, evaluate it on the
    held-out fold. Returns one dict per fold.
    """
    folds = sorted(df["fold"].unique())
    rows = []
    for k in folds:
        fit_df = df[df["fold"] != k]
        eval_df = df[df["fold"] == k]
        w, fit_score = best_weight(fit_df, group, model_names, grid)
        eval_score = blend_score(eval_df, group, model_names, w)
        rows.append(
            {
                "held_out_fold": int(k),
                "weight": {name: float(x) for name, x in zip(model_names, w)},
                "fit_score_on_other_folds": float(fit_score),
                "blend_eval_score_on_held_fold": float(eval_score),
            }
        )
    return rows


def analyze_group(
    group: str,
    runs: dict[str, str],
    baseline_fold_scores: dict[int, float],
    baseline_mean_score: float,
    baseline_model_name: str,
    step: float = WEIGHT_STEP,
    min_folds_improved: int = MIN_FOLDS_IMPROVED,
    fixed_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Full per-group analysis: join OOF, nested LOFO search, production weight
    fit, adopt/fallback decision vs the baseline model's own CV score, plus an
    honest comparison table (single-track means + optional fixed-weight blend).
    """
    model_names = list(runs.keys())
    df = load_joined_oof(runs, group)
    grid = weight_grid(model_names, step=step)

    nested_rows = nested_lofo_search(df, group, model_names, grid)
    for r in nested_rows:
        base_score = baseline_fold_scores[r["held_out_fold"]]
        r["baseline_eval_score_on_held_fold"] = base_score
        r["delta_vs_baseline"] = r["blend_eval_score_on_held_fold"] - base_score

    nested_mean = float(np.mean([r["blend_eval_score_on_held_fold"] for r in nested_rows]))
    n_improved = sum(1 for r in nested_rows if r["delta_vs_baseline"] > 0)

    prod_w, prod_fit_score = best_weight(df, group, model_names, grid)

    adopt_blend = (nested_mean > baseline_mean_score) and (n_improved >= min_folds_improved)
    fallback_weight = {name: (1.0 if name == baseline_model_name else 0.0) for name in model_names}
    final_weight = {name: float(x) for name, x in zip(model_names, prod_w)} if adopt_blend else fallback_weight

    # -- honest comparison table (same held-out-mean footing for every row) --
    single_track = {}
    for name in model_names:
        w = tuple(1.0 if m == name else 0.0 for m in model_names)
        mean_s, folds_s = per_fold_mean_score(df, group, model_names, w)
        single_track[name] = {"mean_score": mean_s, "per_fold": folds_s}

    fixed_block = None
    if fixed_weights is not None:
        fw_tuple = weights_dict_to_tuple(model_names, fixed_weights)
        mean_s, folds_s = per_fold_mean_score(df, group, model_names, fw_tuple)
        fixed_block = {"weights": dict(fixed_weights), "mean_score": mean_s, "per_fold": folds_s}

    return {
        "group": group,
        "model_names": model_names,
        "weight_grid_step": step,
        "nested_lofo_rows": nested_rows,
        "nested_blend_mean_score": nested_mean,
        "baseline_model_name": baseline_model_name,
        "baseline_mean_score": baseline_mean_score,
        "delta_vs_baseline_nested": nested_mean - baseline_mean_score,
        "n_folds_improved": n_improved,
        "n_folds_total": len(nested_rows),
        "min_folds_improved_required": min_folds_improved,
        "production_weight_fit_on_all_folds": {name: float(x) for name, x in zip(model_names, prod_w)},
        "production_weight_fit_score_optimistic_not_for_decisions": float(prod_fit_score),
        "adopt_blend": adopt_blend,
        "final_weight": final_weight,
        "single_track_cv": single_track,
        "fixed_weight_cv": fixed_block,
    }


def _predict_group_test(model: Any, test_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Predict one group's TEST rows with either a feature-matrix model
    (GBM/LightGBM/XGBoost/CatBoost wrappers -- ``predict`` takes ``X`` columns)
    or a sequence model (LSTM/Transformer -- ``predict`` takes a DataFrame that
    also carries the forecast-block key + datetime so it can assemble
    per-block sequences). Detected via the wrapper's ``block_col``/``dt_col``
    attributes; returns predictions aligned to ``test_df`` row order.
    """
    block_col = getattr(model, "block_col", None)
    dt_col = getattr(model, "dt_col", None)
    if block_col is not None and dt_col is not None:
        needed = list(feature_cols)
        for c in (block_col, dt_col):
            if c not in needed:
                needed.append(c)
        missing = [c for c in needed if c not in test_df.columns]
        if missing:
            raise ValueError(
                f"sequence model needs columns absent from the test parquet: {missing[:10]}"
            )
        return model.predict(test_df[needed])
    return model.predict(test_df[feature_cols])


def generate_blend_submission(
    run_id: str, runs: dict[str, str], final_weights: dict[str, dict[str, float]], label: str = ""
) -> pd.DataFrame:
    """Apply each group's per-model weights to the runs' TEST predictions (each
    model's own ``model_<group>.joblib`` + that run's own ``feature_cols_per_group``
    from config.yaml), sum, clip to capacity, and assemble/validate a submission
    exactly like ``src.inference.predict``.

    ``label`` suffixes the output filename (``submission_<run_id>_<label>.csv``)
    so multiple candidate blends (e.g. nested-LOFO vs a fixed v14 weight) from
    one run don't overwrite each other.
    """
    sample = load_sample_submission()
    submission = sample[["forecast_id", "forecast_kst_dtm"]].copy()

    run_configs: dict[str, dict[str, Any]] = {}
    for name, rid in runs.items():
        with open(EXPERIMENTS_DIR / rid / "config.yaml", "r", encoding="utf-8") as f:
            run_configs[name] = yaml.safe_load(f)

    for group in KPX_GROUPS:
        w = final_weights[group]
        blended = None
        capacity_kwh = None
        last_test_df = None
        for name, rid in runs.items():
            weight = w.get(name, 0.0)
            if weight == 0.0:
                continue
            model = joblib.load(EXPERIMENTS_DIR / rid / f"model_{group}.joblib")
            capacity_kwh = model.capacity_kwh
            feature_cols = run_configs[name]["feature_cols_per_group"][group]
            test_df = pd.read_parquet(DATA_PROCESSED_DIR / f"features_{group}_test.parquet")
            last_test_df = test_df
            preds = _predict_group_test(model, test_df, feature_cols)
            blended = preds * weight if blended is None else blended + preds * weight
        if blended is None:
            raise ValueError(f"{group}: final weight has no nonzero component -- nothing to predict with.")
        blended = np.clip(blended, 0.0, capacity_kwh * 1.01)

        pred_df = pd.DataFrame({"forecast_kst_dtm": last_test_df["forecast_kst_dtm"].to_numpy(), group: blended})
        before_len = len(submission)
        submission = submission.merge(pred_df, on="forecast_kst_dtm", how="left")
        if len(submission) != before_len:
            raise ValueError(f"Merge for {group} changed row count ({before_len} -> {len(submission)}).")
        n_missing = int(submission[group].isna().sum())
        if n_missing:
            raise ValueError(f"{n_missing} rows could not be matched to a {group} test prediction.")

    submission_cols = ["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]
    submission = submission[submission_cols].reset_index(drop=True)

    assert len(submission) == 8760, f"expected 8,760 rows, got {len(submission)}"
    assert list(submission.columns) == submission_cols
    assert (submission["forecast_id"].to_numpy() == sample["forecast_id"].to_numpy()).all()
    assert (submission["forecast_kst_dtm"].to_numpy() == sample["forecast_kst_dtm"].to_numpy()).all()

    suffix = f"_{label}" if label else ""
    out_path = SUBMISSIONS_DIR / f"submission_{run_id}{suffix}.csv"
    submission.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote blended submission: %s (%d rows)", out_path, len(submission))
    return submission


def _parse_kv(items: list[str] | None, cast=str) -> dict[str, Any]:
    """Parse repeated ``name=value`` CLI args into a dict (order preserved)."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"expected name=value, got {item!r}")
        name, value = item.split("=", 1)
        out[name.strip()] = cast(value.strip())
    return out


def main() -> str:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Per-group nested-LOFO ensemble weight search.")
    parser.add_argument(
        "--runs", action="append", metavar="name=run_id",
        help="Repeatable: a model name and its experiments/<run_id>. Defaults to the "
             "lgbm/xgb/cat pruned blend if omitted.",
    )
    parser.add_argument(
        "--baseline-name", default=None,
        help="Which --runs model is the standalone bar a blend must clear (default: first run, "
             "or BASELINE_MODEL_NAME for the built-in default runs).",
    )
    parser.add_argument(
        "--baseline-run-id", default=None,
        help="Run id whose metrics.json supplies the baseline per-fold/mean CV scores "
             "(default: the baseline model's own run).",
    )
    parser.add_argument(
        "--fixed-weights", action="append", metavar="name=weight",
        help="Repeatable: a fixed reference blend weight per model (e.g. the v14 "
             "lstm=0.35/transformer=0.35/gbm=0.30). Added to the comparison table and, "
             "if it sums to 1, written as an additional candidate submission.",
    )
    parser.add_argument("--run-suffix", default="ensemble_lgbm_xgb_cat",
                        help="Suffix for the generated experiments/<timestamp>_<suffix> run dir.")
    parser.add_argument("--step", type=float, default=WEIGHT_STEP)
    parser.add_argument("--min-folds-improved", type=int, default=MIN_FOLDS_IMPROVED)
    parser.add_argument("--force-submission", action="store_true",
                        help="Write the nested-LOFO blend submission even if no group adopts a blend "
                             "(writes the pure-baseline-equivalent blend).")
    args = parser.parse_args()

    runs = _parse_kv(args.runs) if args.runs else dict(DEFAULT_RUNS)
    model_names = list(runs.keys())
    baseline_name = args.baseline_name or (BASELINE_MODEL_NAME if BASELINE_MODEL_NAME in runs else model_names[0])
    if baseline_name not in runs:
        raise ValueError(f"--baseline-name {baseline_name!r} is not among --runs {model_names}")
    baseline_run_id = args.baseline_run_id or runs[baseline_name]

    fixed_weights = _parse_kv(args.fixed_weights, cast=float) if args.fixed_weights else None

    with open(EXPERIMENTS_DIR / baseline_run_id / "metrics.json", "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.run_suffix}"
    run_dir = EXPERIMENTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    group_results: dict[str, dict[str, Any]] = {}
    for group in KPX_GROUPS:
        baseline_fold_scores = {fm["fold"]: fm["score"] for fm in baseline_metrics[group]["fold_metrics"]}
        baseline_mean = baseline_metrics[group]["agg_metrics"]["score_mean"]
        result = analyze_group(
            group, runs, baseline_fold_scores, baseline_mean, baseline_name,
            step=args.step, min_folds_improved=args.min_folds_improved, fixed_weights=fixed_weights,
        )
        group_results[group] = result

    # -- honest comparison table -----------------------------------------
    def overall_of(selector) -> float:
        return float(np.mean([selector(group_results[g]) for g in KPX_GROUPS]))

    print("\n" + "=" * 92)
    print(f"HONEST COMPARISON (held-out CV competition_score; runs={runs}, baseline={baseline_name})")
    print("=" * 92)
    header = f"{'group':<14}" + "".join(f"{name+'-alone':>16}" for name in model_names)
    if fixed_weights is not None:
        header += f"{'v14-fixed':>14}"
    header += f"{'nestedLOFO':>13}{'adopt':>8}"
    print(header)
    for g in KPX_GROUPS:
        r = group_results[g]
        line = f"{g:<14}"
        for name in model_names:
            line += f"{r['single_track_cv'][name]['mean_score']:>16.4f}"
        if fixed_weights is not None:
            line += f"{r['fixed_weight_cv']['mean_score']:>14.4f}"
        line += f"{r['nested_blend_mean_score']:>13.4f}{('Y' if r['adopt_blend'] else 'n'):>8}"
        print(line)
    # overall row
    line = f"{'OVERALL':<14}"
    for name in model_names:
        line += f"{overall_of(lambda r, n=name: r['single_track_cv'][n]['mean_score']):>16.4f}"
    if fixed_weights is not None:
        line += f"{overall_of(lambda r: r['fixed_weight_cv']['mean_score']):>14.4f}"
    line += f"{overall_of(lambda r: r['nested_blend_mean_score']):>13.4f}"
    print(line)
    for g in KPX_GROUPS:
        r = group_results[g]
        print(f"  {g}: nested final_weight={r['final_weight']} "
              f"(improved {r['n_folds_improved']}/{r['n_folds_total']} folds, "
              f"delta_vs_{baseline_name}={r['delta_vs_baseline_nested']:+.4f})")

    overall_baseline = overall_of(lambda r: r["baseline_mean_score"])
    overall_nested_adopted = float(np.mean([
        group_results[g]["nested_blend_mean_score"] if group_results[g]["adopt_blend"]
        else group_results[g]["baseline_mean_score"] for g in KPX_GROUPS
    ]))
    any_adopted = any(group_results[g]["adopt_blend"] for g in KPX_GROUPS)
    print(f"\nOVERALL baseline({baseline_name})={overall_baseline:.4f}  "
          f"adopted-per-group(honest nested)={overall_nested_adopted:.4f}  "
          f"delta={overall_nested_adopted - overall_baseline:+.4f}")

    config = {
        "run_id": run_id,
        "method": "ensembler: per-group nested leave-one-fold-out (LOFO) weighted-average blend "
                  "over already-trained model OOF predictions",
        "base_runs": runs,
        "baseline_model_for_decision": baseline_name,
        "baseline_run_id_for_metrics": baseline_run_id,
        "weight_grid_step": args.step,
        "min_folds_improved_required": args.min_folds_improved,
        "fixed_reference_weights": fixed_weights,
        "final_weight_per_group_nested": {g: group_results[g]["final_weight"] for g in KPX_GROUPS},
        "adopt_blend_per_group": {g: group_results[g]["adopt_blend"] for g in KPX_GROUPS},
        "any_group_adopted": any_adopted,
    }
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    metrics_out = {
        "per_group": group_results,
        "overall": {
            "baseline_score": overall_baseline,
            "single_track": {n: overall_of(lambda r, nn=n: r["single_track_cv"][nn]["mean_score"]) for n in model_names},
            "fixed_weight_score": (overall_of(lambda r: r["fixed_weight_cv"]["mean_score"]) if fixed_weights else None),
            "nested_blend_score": overall_of(lambda r: r["nested_blend_mean_score"]),
            "adopted_per_group_nested_score": overall_nested_adopted,
        },
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    # -- candidate submissions -------------------------------------------
    if any_adopted or args.force_submission:
        generate_blend_submission(
            run_id, runs, {g: group_results[g]["final_weight"] for g in KPX_GROUPS}, label="nested_lofo"
        )
    else:
        print(
            "\nNo group's nested-LOFO blend robustly beat its baseline -> the honest nested "
            "recommendation is the baseline standalone; no nested blended submission written "
            "(pass --force-submission to also emit the pure-baseline-equivalent blend)."
        )
    if fixed_weights is not None:
        generate_blend_submission(
            run_id, runs, {g: dict(fixed_weights) for g in KPX_GROUPS}, label="v14_fixed"
        )

    print(f"\nRun dir: {run_dir}")
    return run_id


if __name__ == "__main__":
    main()
