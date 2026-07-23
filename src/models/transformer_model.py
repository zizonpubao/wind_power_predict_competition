"""TransformerEncoder seq2seq point model for one KPX group -- the Transformer
component of the v14 pipeline reproduction (see
``.claude/plans/logical-stirring-sphinx.md`` Phase D, and the reference
hand-off code at ``C:\\Users\\heelo\\Desktop\\files\\transformer_model.py``,
treated as a *starting point*, not validated ground truth).

Architecture: a sinusoidal positional encoding + a 2-layer
``nn.TransformerEncoder`` (d_model=64, nhead=4, GELU) reads one whole forecast
block (24 hours, self-attention over the whole block at once) and a small
Linear head emits a per-hour generation prediction -- i.e. it is a seq2seq
model over the block, the same input/output shape as ``GroupLSTMModel``.
Training bags over ``n_seeds`` random seeds (predictions averaged) and uses the
exact same Huber loss + sample-weighting utilities as the LSTM
(``src.models.torch_common`` / ``src.models.lstm_model.huber_loss``).

Deliberately identical to ``GroupLSTMModel`` in every way that matters for the
blend: it consumes the same ``data_available_kst_dtm``-block sequences from
``build_block_sequences``, so its OOF predictions align cell-for-cell with the
LSTM's and the two are automatically aligned for the Phase E blend. Two things
are *removed* versus the LSTM, per the v14 spec: this model is point-estimate
only (no quantile/decision-optimize variant) and has **no mixup** (mixup is a
group3-only LSTM rule; the Transformer never uses it).

Persistence follows the same ``PredictionCalibrator``/``GroupLSTMModel``
pattern: the instance holds only CPU ``state_dict``s + the fitted scaler +
hyperparameters, so a plain ``joblib.dump`` is safe and portable (GPU-trained
weights load and predict on a CPU-only machine via ``map_location``-style
rebuild onto whatever ``DEVICE`` is available).
"""
from __future__ import annotations

from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.lstm_model import (
    LOW_UTILIZATION_SAMPLE_WEIGHT,
    VALID_HOUR_UTILIZATION,
    huber_loss,
)
from src.models.torch_common import (
    BLOCK_COL,
    DEVICE,
    DT_COL,
    SEQ_LEN,
    GroupFeatureScaler,
    build_block_sequences,
    set_seed,
)

# v14 hand-off starting hyperparameters (TRANSFORMER_PARAMS / TRANSFORMER_SEEDS
# in the reference config.py). Per plan section 1, a starting point subject to
# this repo's own CV re-verification, not validated ground truth.
DEFAULT_D_MODEL = 64
DEFAULT_NHEAD = 4
DEFAULT_N_LAYERS = 2
DEFAULT_DIM_FEEDFORWARD = 128
DEFAULT_DROPOUT = 0.2
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_HUBER_DELTA_KWH = 1800.0  # divided by capacity -> delta in y/capacity space
DEFAULT_N_SEEDS = 4
DEFAULT_MAX_EPOCHS = 150
DEFAULT_PATIENCE = 25
DEFAULT_VAL_HOLDOUT_FRAC = 0.15
# Reuse the LSTM's first 4 base seeds so the two tracks bag over an overlapping
# seed set (harmless for diversity; keeps runs comparable).
BASE_SEEDS = [11, 23, 42, 7]


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding added to the projected
    inputs (max length = one forecast block = ``SEQ_LEN`` hours)."""

    def __init__(self, d_model: int, max_len: int = SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        # guard the odd-d_model case: cos slots may be one shorter than sin slots.
        pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerPointModel(nn.Module):
    """Input projection + sinusoidal positional encoding + N-layer
    ``nn.TransformerEncoder`` + Linear head, emitting one value per input
    timestep (``(batch, seq_len, n_features) -> (batch, seq_len)``)."""

    def __init__(
        self,
        n_features: int,
        d_model: int = DEFAULT_D_MODEL,
        nhead: int = DEFAULT_NHEAD,
        num_layers: int = DEFAULT_N_LAYERS,
        dim_feedforward: int = DEFAULT_DIM_FEEDFORWARD,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        out = self.encoder(x)
        return self.head(out).squeeze(-1)


class GroupTransformerModel:
    """Seed-bagged TransformerEncoder seq2seq model for one KPX group.

    Same ``fit``/``predict``/``save``/``load`` contract as ``GroupLSTMModel``
    (whole-DataFrame slices carrying ``feature_cols`` + ``block_col`` +
    ``dt_col`` [+ ``target_col`` for ``fit``]), so it plugs into the identical
    ``src/training/train_transformer.py`` harness and produces OOF that joins
    with the LSTM/GBM runs. Differences from the LSTM: point-only and **no
    mixup** (v14 spec).

    Parameters
    ----------
    capacity_kwh : the group's 1-hour-equivalent installed capacity (kWh);
        targets are trained in ``y/capacity`` space, predictions scaled back and
        clipped, and the Huber ``delta`` is ``huber_delta_kwh/capacity``.
    feature_cols : the model-input columns.
    d_model, nhead, n_layers : Transformer encoder shape (v14 defaults 64/4/2).
    n_seeds : number of random-seed models to bag (predictions averaged).
    """

    def __init__(
        self,
        capacity_kwh: float,
        feature_cols: list[str],
        d_model: int = DEFAULT_D_MODEL,
        nhead: int = DEFAULT_NHEAD,
        n_layers: int = DEFAULT_N_LAYERS,
        n_seeds: int = DEFAULT_N_SEEDS,
        *,
        block_col: str = BLOCK_COL,
        dt_col: str = DT_COL,
        target_col: str = "target",
        dim_feedforward: int = DEFAULT_DIM_FEEDFORWARD,
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
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.n_layers = int(n_layers)
        self.n_seeds = int(n_seeds)
        self.block_col = block_col
        self.dt_col = dt_col
        self.target_col = target_col
        self.dim_feedforward = int(dim_feedforward)
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
        # _GroupModelProtocol shape parity / harness convenience only -- no
        # boosting-round early stopping to report.
        self.best_iteration_: Optional[int] = None

    @staticmethod
    def _resolve_seeds(seeds: Optional[list[int]], n_seeds: int) -> list[int]:
        base = list(seeds) if seeds is not None else list(BASE_SEEDS)
        while len(base) < n_seeds:
            base.append(base[-1] + 1000)
        return base[:n_seeds]

    def _new_module(self) -> TransformerPointModel:
        return TransformerPointModel(
            self.n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.n_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

    # -- training --------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "GroupTransformerModel":
        """Fit the seed-bagged model on one fold's training rows.

        ``train_df`` must contain ``feature_cols`` + ``block_col`` + ``dt_col``
        + ``target_col``. The scaler is fit on ``train_df``'s feature rows only
        (never val/test -- leakage-safe), then every seed model is trained on
        the same block sequences.
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

        model = self._new_module().to(DEVICE)
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
                opt.zero_grad()
                loss = huber_loss(model(Xtr[idx]), Ytr[idx], Wtr[idx], delta)
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
        """Predict generation (kWh) for every row of ``df``, aligned to ``df``'s
        exact row order and count. Predictions are averaged over the bagged seed
        models, scaled back to kWh, and clipped to ``[0, capacity*1.01]``."""
        if not self.state_dicts_:
            raise RuntimeError("GroupTransformerModel.predict() called before fit().")
        X_seq, _Y, pos_idx, _blocks = build_block_sequences(
            df, self.feature_cols, None, self.block_col, self.dt_col
        )
        X_scaled = self.scaler.transform(X_seq)
        Xt = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)

        seed_preds: list[np.ndarray] = []
        for state in self.state_dicts_:
            model = self._new_module()
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
        via joblib -- no live nn.Module is ever held."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: Any) -> "GroupTransformerModel":
        return joblib.load(path)
