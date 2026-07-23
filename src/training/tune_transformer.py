"""Optuna re-tuning CLI for the TransformerEncoder seq2seq model
(``GroupTransformerModel``) -- thin wrapper over ``src.training.tune_torch``.

The Transformer was carried over from the v14 hand-off with its *starting*
hyperparameters (d_model=64/nhead=4/2-layer/dropout=0.2/lr=5e-4/wd=1e-5/
batch=64), never tuned against this repo's pruned feature set. This searches,
per group, over the space in ``suggest_transformer_params`` and refits the
winner at the full 4-seed / 5-fold config. See ``tune_torch``'s module
docstring for the cost-control + early-stop design and the honest "CV gain may
not transfer to the leaderboard" prior. The Transformer uses **no mixup** (v14
rule), so ``_extra_kwargs`` is always empty.

d_model must be divisible by nhead; every combination in the search space
(d_model in {32,64,128}, nhead in {2,4}) satisfies this, so no combinations are
pruned.
"""
from __future__ import annotations

import optuna

from src.models.transformer_model import DEFAULT_N_SEEDS, GroupTransformerModel
from src.training.tune_torch import build_arg_parser, run_tuning

# The pre-tuning Transformer run this re-tune is compared against (overall 0.5963).
BASELINE_RUN_ID = "20260723_133309_transformer_pruned"


def suggest_transformer_params(trial: optuna.Trial) -> dict:
    """Search space (keys are exactly ``GroupTransformerModel`` constructor
    kwargs). Every d_model in {32,64,128} is divisible by every nhead in {2,4},
    so no trial is invalid."""
    return {
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "nhead": trial.suggest_categorical("nhead", [2, 4]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


def _extra_kwargs(kpx_group: str) -> dict:
    return {}  # Transformer never uses mixup (v14 rule)


def main() -> str:
    args = build_arg_parser("Transformer", DEFAULT_N_SEEDS).parse_args()
    return run_tuning(
        model_kind="transformer",
        model_type_label="transformer_encoder_seq2seq_tuned",
        model_cls=GroupTransformerModel,
        suggest_fn=suggest_transformer_params,
        extra_kwargs_fn=_extra_kwargs,
        default_n_seeds=DEFAULT_N_SEEDS,
        baseline_run_id=BASELINE_RUN_ID,
        args=args,
    )


if __name__ == "__main__":
    main()
