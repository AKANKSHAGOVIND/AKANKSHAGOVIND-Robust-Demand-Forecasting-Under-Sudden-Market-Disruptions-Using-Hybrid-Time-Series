"""
src/models/transformer_model.py
---------------------------------
Lightweight Transformer encoder (PyTorch) for temporal forecasting.
Falls back to Holt-Winters ETS if PyTorch is unavailable.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseForecaster

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ── PyTorch Architecture ──────────────────────────────────────────────────────
if _TORCH_OK:

    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe  = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1), :]

    class _LiteTransformer(nn.Module):
        def __init__(self, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1):
            super().__init__()
            self.proj    = nn.Linear(1, d_model)
            self.pos_enc = _PositionalEncoding(d_model)
            enc_layer    = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_ff, dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
            self.out     = nn.Linear(d_model, 1)

        def forward(self, x):                   # x: (B, T, 1)
            x = self.proj(x)
            x = self.pos_enc(x)
            x = self.encoder(x)
            return self.out(x[:, -1, :])        # predict next step


class TransformerForecaster(BaseForecaster):
    """
    Parameters
    ----------
    lookback   : int   Input window
    d_model    : int   Embedding dimension (must be divisible by nhead)
    nhead      : int   Number of attention heads
    num_layers : int   Encoder layers
    epochs     : int   Max epochs
    patience   : int   Early-stopping patience
    lr         : float Learning rate
    """

    name = "Transformer"

    def __init__(
        self,
        horizon: int    = 28,
        lookback: int   = 28,
        d_model: int    = 64,
        nhead: int      = 4,
        num_layers: int = 2,
        epochs: int     = 60,
        batch_size: int = 32,
        lr: float       = 1e-3,
        patience: int   = 10,
    ):
        super().__init__(horizon)
        self.lookback   = lookback
        self.d_model    = d_model
        self.nhead      = nhead
        self.num_layers = num_layers
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience
        self._model     = None
        self._mu        = 0.0
        self._sigma     = 1.0

    def _make_sequences(self, scaled: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i - self.lookback: i])
            y.append(scaled[i])
        Xt = torch.tensor(np.array(X)[..., None], dtype=torch.float32)
        yt = torch.tensor(np.array(y)[:, None],   dtype=torch.float32)
        return Xt, yt

    def fit(self, train: pd.DataFrame, target: str = "sales") -> None:
        self._validate_train(train, target)
        values     = train[target].values.astype(float)
        self._mu   = values.mean()
        self._sigma = values.std() + 1e-8
        scaled     = (values - self._mu) / self._sigma

        X, y      = self._make_sequences(scaled)
        loader    = DataLoader(TensorDataset(X, y),
                               batch_size=self.batch_size, shuffle=True)

        self._model = _LiteTransformer(
            d_model=self.d_model, nhead=self.nhead,
            num_layers=self.num_layers,
        )
        optim     = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)
        criterion = nn.HuberLoss()

        best_loss, no_imp, best_state = np.inf, 0, None

        for _ in range(self.epochs):
            self._model.train()
            ep_loss = 0.0
            for xb, yb in loader:
                optim.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optim.step()
                ep_loss += loss.item()
            sched.step(ep_loss)

            if ep_loss < best_loss:
                best_loss  = ep_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                no_imp     = 0
            else:
                no_imp += 1
            if no_imp >= self.patience:
                break

        if best_state:
            self._model.load_state_dict(best_state)
        self._model.eval()

        self._last_scaled  = scaled[-self.lookback:]
        self._train_values = values
        self._last_date    = pd.to_datetime(train["date"].max())
        self.is_fitted     = True

    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        assert self.is_fitted, "Call fit() first."
        horizon = horizon or self.horizon
        context = self._last_scaled.tolist()
        preds   = []

        self._model.eval()
        with torch.no_grad():
            for _ in range(horizon):
                seq  = torch.tensor(
                    np.array(context[-self.lookback:])[None, :, None],
                    dtype=torch.float32,
                )
                pred = self._model(seq).item()
                preds.append(pred)
                context.append(pred)

        denorm = np.array(preds) * self._sigma + self._mu
        denorm = np.clip(denorm, 0, None)
        rs     = self._train_values[-28:].std()
        lower  = np.clip(denorm - 1.645 * rs, 0, None)
        upper  = denorm + 1.645 * rs
        future = self._make_future_dates(self._last_date, horizon)

        return pd.DataFrame({
            "date": future, "yhat": denorm,
            "yhat_lower": lower, "yhat_upper": upper, "model": self.name,
        })


# ── Holt-Winters ETS Fallback ─────────────────────────────────────────────────
class ETSForecaster(BaseForecaster):
    """Triple Exponential Smoothing (Holt-Winters) fallback."""

    name = "Transformer (ETS)"

    def fit(self, train: pd.DataFrame, target: str = "sales") -> None:
        self._validate_train(train, target)
        self._values    = train[target].values.astype(float)
        self._last_date = pd.to_datetime(train["date"].max())
        self.is_fitted  = True

    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        horizon   = horizon or self.horizon
        v         = self._values
        alpha, beta, gamma, s = 0.3, 0.1, 0.2, 7

        level = float(v[0])
        trend = float((v[s] - v[0]) / s) if len(v) > s else 0.0
        season = [float(v[i]) - level for i in range(min(s, len(v)))]
        season = (season * (len(v) // s + 2))[: len(v)]

        for i in range(1, len(v)):
            pl    = level
            level = alpha * (v[i] - season[i % s]) + (1 - alpha) * (level + trend)
            trend = beta  * (level - pl)            + (1 - beta)  * trend
            season[i % s] = gamma * (v[i] - level) + (1 - gamma) * season[i % s]

        preds = []
        for m in range(1, horizon + 1):
            preds.append(max(0.0, level + m * trend + season[(len(v) + m) % s]))

        preds = np.array(preds)
        resid = v[-14:] - np.mean(v[-14:])
        lower, upper = self.get_confidence_interval(preds, resid)
        future = self._make_future_dates(self._last_date, horizon)
        return pd.DataFrame({
            "date": future, "yhat": preds,
            "yhat_lower": lower, "yhat_upper": upper, "model": self.name,
        })


def get_transformer_forecaster(**kwargs) -> BaseForecaster:
    return TransformerForecaster(**kwargs) if _TORCH_OK else ETSForecaster(**kwargs)
