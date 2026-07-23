"""Optuna re-tuning CLI for the bidirectional-LSTM seq2seq model
(``GroupLSTMModel``) -- thin wrapper over ``src.training.tune_torch``.

The LSTM was carried over from the v14 hand-off with its *starting*
hyperparameters (hidden=64/1-layer/dropout=0.2/lr=1e-3/batch=64/
huber_delta_kwh=1800), never tuned against this repo's pruned feature set. This
searches, per group, over the space in ``suggest_lstm_params`` and refits the
winner at the full 6-seed / 5-fold config. See ``tune_torch``'s module
docstring for the cost-control + early-stop (noise/time) design and the honest
"CV gain may not transfer to the leaderboard" prior this is run under.

group3 keeps its mixup augmentation (``apply_mixup=True``) during both search
and refit -- that is a fixed v14 rule, not a tuned knob.
"""
from __future__ import annotations

import optuna

from src.models.lstm_model import DEFAULT_N_SEEDS, GroupLSTMModel
from src.training.tune_torch import build_arg_parser, run_tuning

# The pre-tuning LSTM run this re-tune is compared against (overall 0.5953).
BASELINE_RUN_ID = "20260723_131338_lstm_pruned"
MIXUP_GROUP = "kpx_group_3"


def suggest_lstm_params(trial: optuna.Trial) -> dict:
    """Search space (keys are exactly ``GroupLSTMModel`` constructor kwargs)."""
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 1e-3),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "huber_delta_kwh": trial.suggest_float("huber_delta_kwh", 900.0, 2700.0),
    }


def _extra_kwargs(kpx_group: str) -> dict:
    # group3-only mixup augmentation (v14 rule), passed explicitly since
    # capacity alone can't identify the group.
    return {"apply_mixup": kpx_group == MIXUP_GROUP}


def main() -> str:
    args = build_arg_parser("LSTM", DEFAULT_N_SEEDS).parse_args()
    return run_tuning(
        model_kind="lstm",
        model_type_label="lstm_bidirectional_seq2seq_tuned",
        model_cls=GroupLSTMModel,
        suggest_fn=suggest_lstm_params,
        extra_kwargs_fn=_extra_kwargs,
        default_n_seeds=DEFAULT_N_SEEDS,
        baseline_run_id=BASELINE_RUN_ID,
        args=args,
    )


if __name__ == "__main__":
    main()
