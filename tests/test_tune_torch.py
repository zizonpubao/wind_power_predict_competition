"""Light unit tests for src/training/tune_torch.py's cost-control / early-stop
machinery and the two sequence-model search spaces.

Deliberately does NOT train any LSTM/Transformer (that is expensive and GPU-
bound, and the real re-tuning runs cover it end-to-end). It only pins down the
cheap, pure-logic pieces that guard the search: the noise/time early-stop
callback's stop conditions, and that every hyperparameter combination the
suggest functions can emit is a valid model constructor kwarg set -- in
particular that Transformer's ``d_model`` is always divisible by ``nhead``.
"""
import optuna
import pytest

from src.models.lstm_model import GroupLSTMModel
from src.models.transformer_model import GroupTransformerModel
from src.training.tune_lstm import suggest_lstm_params
from src.training.tune_transformer import suggest_transformer_params
from src.training.tune_torch import NoiseTimeEarlyStop, build_arg_parser


# ---------------------------------------------------------------------------
# NoiseTimeEarlyStop
# ---------------------------------------------------------------------------


def test_early_stop_triggers_when_gain_below_noise_after_min_trials():
    # baseline 0.60, noise band 0.02; best 0.605 -> gain 0.005 < 0.02 -> stop.
    stopper = NoiseTimeEarlyStop(baseline_mean=0.60, noise_thresh=0.02, min_trials=1, time_budget_s=1e9)
    reason = stopper.decide(n_done=1, best_value=0.605, elapsed=0.0)
    assert reason is not None and "no meaningful gain" in reason


def test_no_early_stop_when_gain_clears_noise():
    # gain 0.05 > noise 0.02 -> keep going.
    stopper = NoiseTimeEarlyStop(baseline_mean=0.60, noise_thresh=0.02, min_trials=1, time_budget_s=1e9)
    assert stopper.decide(n_done=1, best_value=0.65, elapsed=0.0) is None


def test_no_early_stop_before_min_trials():
    # Even a below-noise best must NOT stop before min_trials completed trials.
    stopper = NoiseTimeEarlyStop(baseline_mean=0.60, noise_thresh=0.02, min_trials=5, time_budget_s=1e9)
    assert stopper.decide(n_done=2, best_value=0.601, elapsed=0.0) is None


def test_time_budget_stop_fires_regardless_of_gain():
    # An exceeded time budget must stop immediately, even with a big gain / high min_trials.
    stopper = NoiseTimeEarlyStop(baseline_mean=0.60, noise_thresh=0.02, min_trials=100, time_budget_s=10.0)
    reason = stopper.decide(n_done=1, best_value=0.99, elapsed=11.0)
    assert reason is not None and "time budget" in reason


# ---------------------------------------------------------------------------
# Search spaces -> valid constructor kwargs
# ---------------------------------------------------------------------------

FEATURES = ["f0", "f1", "f2"]


def _sample_params(suggest_fn, n=40):
    """Draw n random parameter sets from a suggest function via independent
    Optuna trials (RandomSampler covers the categorical/int corners)."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    out = []
    for _ in range(n):
        trial = study.ask()
        out.append(suggest_fn(trial))
        study.tell(trial, 0.0)
    return out


def test_lstm_suggest_params_construct_a_model():
    for params in _sample_params(suggest_lstm_params):
        assert set(params) == {"hidden_size", "dropout", "lr", "weight_decay", "batch_size", "huber_delta_kwh"}
        # must construct without error (n_seeds tiny for speed; not fitted).
        GroupLSTMModel(21600, FEATURES, n_seeds=1, **params)


def test_transformer_suggest_params_dmodel_divisible_by_nhead_and_construct():
    for params in _sample_params(suggest_transformer_params):
        assert params["d_model"] % params["nhead"] == 0
        GroupTransformerModel(21600, FEATURES, n_seeds=1, **params)


# ---------------------------------------------------------------------------
# CLI arg parser defaults
# ---------------------------------------------------------------------------


def test_build_arg_parser_defaults():
    args = build_arg_parser("LSTM", default_n_seeds=6).parse_args([])
    assert args.n_seeds == 6
    assert args.n_splits == 5
    assert args.tune_n_splits >= 3  # brief: never fewer than 3 folds during search
    assert args.feature_set == "pruned"
    assert list(args.groups) == ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
