"""Model-agnostic Optuna tuning + block-aware CV/OOF helpers shared by
``src/training/train_xgb.py`` and ``src/training/train_catboost.py``.

Factored out of ``src/training/tune_hyperparams.py`` (the LightGBM-specific
tuning script) so the XGBoost/CatBoost ensemble-candidate models reuse the
exact same building blocks -- ``BlockTimeSeriesSplit``, the official
``competition_score``, ``train_baseline``'s feature-column selection, and the
"average CV folds' best_iteration_, use it as a fixed n_estimators-equivalent
for the final full-data refit" strategy -- rather than duplicating them a
second and third time. LightGBM's own ``tune_hyperparams.py`` is intentionally
left untouched (it has extra LightGBM-only machinery: the asymmetric
objective search dimension and per-group calibration) -- this module only
factors out the parts that are genuinely identical across model types.

The one thing that differs per model type is the ``n_estimators``-equivalent
constructor kwarg name (``"n_estimators"`` for ``GroupXGBModel``,
``"iterations"`` for ``GroupCatBoostModel``) -- callers pass that in as
``n_estimators_param``.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Protocol

import numpy as np
import optuna
import pandas as pd

from configs.paths import DATA_PROCESSED_DIR, GROUP_CAPACITY_KWH
from src.evaluation.metrics import competition_score
from src.training.train_baseline import _get_feature_cols
from src.validation.splitter import BlockTimeSeriesSplit

logger = logging.getLogger(__name__)


class _GroupModelProtocol(Protocol):
    best_iteration_: int | None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int = 50,
    ) -> "_GroupModelProtocol": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def oof_predict_generic(
    df: pd.DataFrame,
    feature_cols: list[str],
    capacity: float,
    kpx_group: str,
    model_cls: Callable[..., _GroupModelProtocol],
    params: dict[str, Any],
    early_stopping_rounds: int,
    n_splits: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run the block-aware CV fold loop once for an arbitrary model class +
    hyperparameter dict, collecting every fold's held-out (out-of-fold)
    predictions into one concatenated frame.

    Same shape as ``src/training/tune_hyperparams.py``'s ``_oof_predict``:
    returns ``(oof_df, fold_meta)`` with ``oof_df`` columns ``fold``
    (1-indexed), ``forecast_kst_dtm``, ``pred``, ``actual``.
    """
    splitter = BlockTimeSeriesSplit(n_splits=n_splits)
    oof_records: list[pd.DataFrame] = []
    fold_meta: list[dict[str, Any]] = []

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(df), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        X_train, y_train = train_df[feature_cols], train_df["target"]
        X_val, y_val = val_df[feature_cols], val_df["target"]

        model = model_cls(capacity_kwh=capacity, **params)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=early_stopping_rounds,
        )

        val_pred = model.predict(X_val)
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
        fold_meta.append(
            {
                "fold": fold_i,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "best_iteration": int(model.best_iteration_) if model.best_iteration_ else None,
            }
        )

    oof_df = pd.concat(oof_records, ignore_index=True)
    return oof_df, fold_meta


def score_oof(
    oof_df: pd.DataFrame,
    fold_meta: list[dict[str, Any]],
    kpx_group: str,
    pred_col: str = "pred",
) -> list[dict[str, Any]]:
    """Reduce an ``oof_predict_generic`` frame to per-fold ``competition_score``
    dicts. Identical logic to ``tune_hyperparams._score_oof``.
    """
    fold_metrics: list[dict[str, Any]] = []
    for meta in fold_meta:
        fold_i = meta["fold"]
        fold_rows = oof_df[oof_df["fold"] == fold_i]
        pred_df = pd.DataFrame(
            {"forecast_kst_dtm": fold_rows["forecast_kst_dtm"].to_numpy(), kpx_group: fold_rows[pred_col].to_numpy()}
        )
        actual_df = pd.DataFrame(
            {"forecast_kst_dtm": fold_rows["forecast_kst_dtm"].to_numpy(), kpx_group: fold_rows["actual"].to_numpy()}
        )
        scores = competition_score(pred_df, actual_df, group_cols=[kpx_group])
        fold_metrics.append(
            {
                "fold": fold_i,
                "n_train": meta["n_train"],
                "n_val": meta["n_val"],
                "score": scores["score"],
                "1-NMAE": scores["1-NMAE"],
                "FICR": scores["FICR"],
                "best_iteration": meta["best_iteration"],
            }
        )
    return fold_metrics


def _make_objective(
    df: pd.DataFrame,
    feature_cols: list[str],
    capacity: float,
    kpx_group: str,
    model_cls: Callable[..., _GroupModelProtocol],
    suggest_params_fn: Callable[[optuna.Trial], dict[str, Any]],
    early_stopping_rounds: int,
    n_splits: int,
):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params_fn(trial)
        oof_df, fold_meta = oof_predict_generic(
            df, feature_cols, capacity, kpx_group, model_cls, params, early_stopping_rounds, n_splits
        )
        fold_metrics = score_oof(oof_df, fold_meta, kpx_group)
        return float(np.nanmean([f["score"] for f in fold_metrics]))

    return objective


def tune_group_generic(
    kpx_group: str,
    model_cls: Callable[..., _GroupModelProtocol],
    suggest_params_fn: Callable[[optuna.Trial], dict[str, Any]],
    n_estimators_param: str,
    n_trials: int,
    n_splits: int,
    feature_set: str,
    early_stopping_rounds: int,
    sampler_seed: int = 42,
) -> dict[str, Any]:
    """Run an Optuna study for one kpx_group with an arbitrary model class,
    then refit a final model on all available rows using the best-found
    hyperparameters. Same overall shape/strategy as
    ``tune_hyperparams.tune_group`` (LightGBM-specific version), generalized
    over the model class.

    ``n_estimators_param``: the constructor kwarg name this model class uses
    for its "how many boosting rounds" parameter (``"n_estimators"`` for
    ``GroupXGBModel``, ``"iterations"`` for ``GroupCatBoostModel``) -- the
    final refit sets this to the mean of the CV folds' ``best_iteration_``
    (floored at 50), the same convention ``train_baseline.run_group`` uses.
    """
    path = DATA_PROCESSED_DIR / f"features_{kpx_group}_train.parquet"
    df = pd.read_parquet(path)

    n_missing = int(df["target"].isna().sum())
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    logger.info("%s: dropped %d rows with missing target, %d rows remain", kpx_group, n_missing, len(df))

    feature_cols = _get_feature_cols(df, kpx_group=kpx_group, feature_set=feature_set)
    capacity = GROUP_CAPACITY_KWH[kpx_group]

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    t0 = time.time()
    study.optimize(
        _make_objective(
            df, feature_cols, capacity, kpx_group, model_cls, suggest_params_fn, early_stopping_rounds, n_splits
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0
    logger.info(
        "%s: Optuna study finished (%d trials, %.1fs, %.2fs/trial), best_value=%.4f, best_params=%s",
        kpx_group,
        len(study.trials),
        elapsed,
        elapsed / max(len(study.trials), 1),
        study.best_value,
        study.best_params,
    )

    # Re-run CV once more with the winning params to get fold_metrics/OOF
    # predictions for the final refit and for the saved oof_predictions_<group>.parquet
    # (Optuna only records the scalar objective value per trial).
    best_params = dict(study.best_params)
    oof_df, fold_meta = oof_predict_generic(
        df, feature_cols, capacity, kpx_group, model_cls, best_params, early_stopping_rounds, n_splits
    )
    fold_metrics = score_oof(oof_df, fold_meta, kpx_group)
    best_iterations = [f["best_iteration"] for f in fold_metrics if f["best_iteration"]]

    agg_metrics = {
        "score_mean": float(np.nanmean([f["score"] for f in fold_metrics])),
        "score_std": float(np.nanstd([f["score"] for f in fold_metrics])),
        "1-NMAE_mean": float(np.nanmean([f["1-NMAE"] for f in fold_metrics])),
        "1-NMAE_std": float(np.nanstd([f["1-NMAE"] for f in fold_metrics])),
        "FICR_mean": float(np.nanmean([f["FICR"] for f in fold_metrics])),
        "FICR_std": float(np.nanstd([f["FICR"] for f in fold_metrics])),
    }

    final_params: dict[str, Any] = dict(best_params)
    if best_iterations:
        final_params[n_estimators_param] = max(int(round(float(np.mean(best_iterations)))), 50)

    final_model = model_cls(capacity_kwh=capacity, **final_params)
    final_model.fit(df[feature_cols], df["target"])
    final_n_estimators = final_model.params[n_estimators_param]
    logger.info(
        "%s: final tuned model refit on all %d rows (%s=%s)",
        kpx_group,
        len(df),
        n_estimators_param,
        final_n_estimators,
    )

    trials_records = [
        {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)} for t in study.trials
    ]

    return {
        "kpx_group": kpx_group,
        "n_rows_used": len(df),
        "n_missing_target_dropped": n_missing,
        "feature_cols": feature_cols,
        "n_trials": n_trials,
        "sampler_seed": sampler_seed,
        "tuning_wall_clock_seconds": elapsed,
        "best_params": best_params,
        "best_cv_score": agg_metrics["score_mean"],
        "fold_metrics": fold_metrics,
        "agg_metrics": agg_metrics,
        "trials": trials_records,
        "final_model": final_model,
        "final_params": final_params,
        "final_n_estimators": final_n_estimators,
        "oof_df": oof_df,
        "fold_meta": fold_meta,
    }
