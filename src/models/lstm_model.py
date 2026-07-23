"""Bidirectional-LSTM seq2seq point model for one KPX group -- the LSTM
component of the v14 pipeline reproduction (see
``.claude/plans/logical-stirring-sphinx.md`` Phase C, and the reference
hand-off code at ``C:\\Users\\heelo\\Desktop\\files\\lstm_model.py``, treated
as a *starting point*, not validated ground truth).

Architecture: a 1-layer bidirectional LSTM (hidden 64) reads one whole
forecast block (24 hours, both directions at once) and a small Linear head
emits a per-hour generation prediction -- i.e. it is a seq2seq model over the
block, NOT a sliding-window "predict the next hour" model. Training bags over
``n_seeds`` random seeds (predictions averaged) and uses a Huber loss with
sample weights that (a) down-weight sub-10%-utilization hours (which the
official metric doesn't score) and (b) **mask out missing-label hours** so a
block can be kept intact even when a few of its 24 hours have no label
(dropping those rows instead would break the 24-row block alignment the
whole sequence approach relies on).

Unlike the GBM/XGBoost wrappers, this model's ``fit``/``predict`` take a whole
DataFrame slice (features + block key + datetime), not a bare feature matrix,
because it needs the ``data_available_kst_dtm`` block key to assemble
sequences -- so it is driven by its own harness (``src/training/train_lstm.py``)
rather than ``tune_common.oof_predict_generic`` (whose ``X = df[feature_cols]``
strips the block key). See that harness's module docstring for the "harness
option (b)" rationale.

Persistence follows ``src/models/calibration.PredictionCalibrator``'s pattern:
the model never keeps a live ``nn.Module`` as an attribute -- only CPU
``state_dict``s plus the fitted scaler and hyperparameters -- so a plain
``joblib.dump`` of the whole instance is safe and portable (GPU-trained
weights load and predict fine on a CPU-only machine, since state_dicts are
stored on CPU and rebuilt onto whatever ``DEVICE`` is available).
"""
from __future__ import annotations

from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.torch_common import (
    BLOCK_COL,
    DEVICE,
    DT_COL,
    GroupFeatureScaler,
    build_block_sequences,
    set_seed,
)

# v14 hand-off starting hyperparameters (LSTM_PARAMS / LSTM_SEEDS in the
# reference config.py). Per the plan section 1, these are a starting point
# subject to this repo's own CV re-verification, not validated ground truth.
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_LAYERS = 1
DEFAULT_DROPOUT = 0.2
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_HUBER_DELTA_KWH = 1800.0  # divided by capacity -> delta in y/capacity space
DEFAULT_N_SEEDS = 6
DEFAULT_MAX_EPOCHS = 150
DEFAULT_PATIENCE = 25
DEFAULT_VAL_HOLDOUT_FRAC = 0.15
BASE_SEEDS = [11, 23, 42, 7, 99, 123]

# Sample-weighting rule (v14 VALID_HOUR_THRESHOLD): hours below 10% capacity
# utilization get a small weight (the official metrics ignore them), and
# missing-label hours get weight 0 (masked out entirely).
VALID_HOUR_UTILIZATION = 0.10
LOW_UTILIZATION_SAMPLE_WEIGHT = 0.05

DEFAULT_MIXUP_ALPHA = 0.1


class LSTMPointModel(nn.Module):
    """1-layer bidirectional LSTM + Linear head, emitting one value per input
    timestep (``(batch, seq_len, n_features) -> (batch, seq_len)``)."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, delta: float) -> torch.Tensor:
    """Weighted Huber loss (weights carry the 10%-utilization down-weighting and
    the missing-label 0-mask), averaged over total weight."""
    err = pred - target
    abs_err = torch.abs(err)
    quad = torch.clamp(abs_err, max=delta)
    lin = abs_err - quad
    loss = 0.5 * quad ** 2 + delta * lin
    return (loss * weight).sum() / (weight.sum() + 1e-6)


def _mixup_batch(X: torch.Tensor, Y: torch.Tensor, W: torch.Tensor, alpha: float):
    """group3-only mixup augmentation: blend batch sample pairs by a random
    Beta ratio (group3's label span is only 2023-2024, so it benefits from the
    extra augmentation -- v14 rule)."""
    if alpha <= 0:
        return X, Y, W
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(X.size(0), device=X.device)
    X_mix = lam * X + (1 - lam) * X[perm]
    Y_mix = lam * Y + (1 - lam) * Y[perm]
    W_mix = lam * W + (1 - lam) * W[perm]
    return X_mix, Y_mix, W_mix


class GroupLSTMModel:
    """Seed-bagged bidirectional-LSTM seq2seq model for one KPX group.

    Parameters
    ----------
    capacity_kwh : the group's 1-hour-equivalent installed capacity (kWh);
        targets are trained in ``y/capacity`` space and predictions scaled back
        by it, and the Huber ``delta`` is ``huber_delta_kwh/capacity`` (so
        ~1800/21600 ≈ 0.083, right at the FICR 8% threshold -- an intentional
        design point, see the plan's Phase C notes).
    feature_cols : the model-input columns; the ``fit``/``predict`` DataFrames
        must also carry ``block_col`` and ``dt_col`` (and ``target_col`` for
        ``fit``) so sequences can be assembled by forecast block.
    apply_mixup : whether to use group3's mixup augmentation. Passed
        **explicitly** by the harness (capacity alone can't identify the group,
        since group1/2 share 21,600 kWh) -- ``train_lstm.py`` sets it True only
        for kpx_group_3.
    n_seeds : number of random-seed models to bag (predictions averaged).
    """

    def __init__(
        self,
        capacity_kwh: float,
        feature_cols: list[str],
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        n_seeds: int = DEFAULT_N_SEEDS,
        apply_mixup: bool = False,
        mixup_alpha: float = DEFAULT_MIXUP_ALPHA,
        *,
        block_col: str = BLOCK_COL,
        dt_col: str = DT_COL,
        target_col: str = "target",
        layers: int = DEFAULT_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        batch_size: int = DEFAULT_BATCH_SIZE,
        huber_delta_kwh: float = DEFAULT_HUBER_DELTA_KWH,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        patience: int = DEFAULT_PATIENCE,
        val_holdout_frac: float = DEFAULT_VAL_HOLDOUT_FRAC,
        seeds: Optional[list[int]] = None,
    ):
        self.capacity_kwh = float(capacity_kwh)
        self.feature_cols = list(feature_cols)
        self.n_features = len(self.feature_cols)
        self.hidden_size = int(hidden_size)
        self.n_seeds = int(n_seeds)
        self.apply_mixup = bool(apply_mixup)
        self.mixup_alpha = float(mixup_alpha)
        self.block_col = block_col
        self.dt_col = dt_col
        self.target_col = target_col
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.huber_delta_kwh = float(huber_delta_kwh)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.val_holdout_frac = float(val_holdout_frac)
        self.seeds = self._resolve_seeds(seeds, self.n_seeds)

        self.scaler: Optional[GroupFeatureScaler] = None
        self.state_dicts_: list[dict[str, Any]] = []
        # For _GroupModelProtocol shape parity / harness convenience only --
        # this model has no boosting-round early stopping to report.
        self.best_iteration_: Optional[int] = None

    @staticmethod
    def _resolve_seeds(seeds: Optional[list[int]], n_seeds: int) -> list[int]:
        base = list(seeds) if seeds is not None else list(BASE_SEEDS)
        while len(base) < n_seeds:
            base.append(base[-1] + 1000)
        return base[:n_seeds]

    # -- training --------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "GroupLSTMModel":
        """Fit the seed-bagged model on one fold's training rows.

        ``train_df`` must contain ``feature_cols`` + ``block_col`` + ``dt_col``
        + ``target_col``. The feature scaler is fit on ``train_df``'s feature
        rows only (never val/test -- leakage-safe), then every seed model is
        trained on the same sequences.
        """
        self.scaler = GroupFeatureScaler().fit(train_df[self.feature_cols].to_numpy(dtype=float))
        X_seq, Y_seq, _pos, _blocks = build_block_sequences(
            train_df, self.feature_cols, self.target_col, self.block_col, self.dt_col
        )
        X_scaled = self.scaler.transform(X_seq)

        self.state_dicts_ = []
        for seed in self.seeds:
            self.state_dicts_.append(self._train_one_seed(X_scaled, Y_seq, seed))
        return self

    def _train_one_seed(self, X_scaled: np.ndarray, Y_seq: np.ndarray, seed: int) -> dict[str, Any]:
        set_seed(seed)

        label_mask = ~np.isnan(Y_seq)
        Y_filled = np.nan_to_num(Y_seq, nan=0.0)
        weight = np.where(
            Y_filled >= self.capacity_kwh * VALID_HOUR_UTILIZATION,
            1.0,
            LOW_UTILIZATION_SAMPLE_WEIGHT,
        ) * label_mask
        Y_norm = Y_filled / self.capacity_kwh

        n = X_scaled.shape[0]
        n_val = max(1, int(n * self.val_holdout_frac)) if n > 1 else 0
        tr_idx = np.arange(0, n - n_val)
        va_idx = np.arange(n - n_val, n)

        Xt = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
        Yt = torch.tensor(Y_norm, dtype=torch.float32, device=DEVICE)
        Wt = torch.tensor(weight, dtype=torch.float32, device=DEVICE)
        Xtr, Ytr, Wtr = Xt[tr_idx], Yt[tr_idx], Wt[tr_idx]
        has_val = len(va_idx) > 0
        Xva, Yva, Wva = (Xt[va_idx], Yt[va_idx], Wt[va_idx]) if has_val else (Xtr, Ytr, Wtr)

        model = LSTMPointModel(self.n_features, self.hidden_size, self.layers, self.dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
        delta = self.huber_delta_kwh / self.capacity_kwh

        best_val, best_state, bad = np.inf, None, 0
        bs = self.batch_size
        for _epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(len(Xtr), device=DEVICE)
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                xb, yb, wb = Xtr[idx], Ytr[idx], Wtr[idx]
                if self.apply_mixup:
                    xb, yb, wb = _mixup_batch(xb, yb, wb, self.mixup_alpha)
                opt.zero_grad()
                loss = huber_loss(model(xb), yb, wb, delta)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = huber_loss(model(Xva), Yva, Wva, delta).item()
            sched.step(val_loss)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= self.patience:
                break

        if best_state is None:  # never improved (e.g. max_epochs very small)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return best_state

    # -- inference -------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict generation (kWh) for every row of ``df``, returned aligned to
        ``df``'s exact row order and count.

        ``df`` must contain ``feature_cols`` + ``block_col`` + ``dt_col`` (no
        target needed). Predictions are averaged over the bagged seed models,
        scaled back to kWh, and clipped to ``[0, capacity*1.01]``.
        """
        if not self.state_dicts_:
            raise RuntimeError("GroupLSTMModel.predict() called before fit().")
        X_seq, _Y, pos_idx, _blocks = build_block_sequences(
            df, self.feature_cols, None, self.block_col, self.dt_col
        )
        X_scaled = self.scaler.transform(X_seq)
        Xt = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)

        seed_preds: list[np.ndarray] = []
        for state in self.state_dicts_:
            model = LSTMPointModel(self.n_features, self.hidden_size, self.layers, self.dropout)
            model.load_state_dict(state)  # state_dicts are CPU tensors
            model.to(DEVICE)
            model.eval()
            with torch.no_grad():
                p = model(Xt).cpu().numpy()  # (n_blocks, seq_len)
            seed_preds.append(np.clip(p * self.capacity_kwh, 0.0, self.capacity_kwh * 1.01))

        pred_seq = np.mean(seed_preds, axis=0)  # (n_blocks, seq_len)
        out = np.empty(len(df), dtype=float)
        out[pos_idx.ravel()] = pred_seq.ravel()
        return out

    # -- persistence -----------------------------------------------------
    def save(self, path: Any) -> None:
        """Persist the whole instance (CPU state_dicts + scaler + hyperparams)
        via joblib -- no live nn.Module is ever held, mirroring
        ``PredictionCalibrator.save``."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: Any) -> "GroupLSTMModel":
        return joblib.load(path)
