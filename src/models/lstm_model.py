"""
src/models/lstm_model.py
-------------------------
Stacked LSTM (TensorFlow/Keras) with early stopping and normalization.
Falls back to rolling linear regression if TF is unavailable.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseForecaster

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.get_logger().setLevel("ERROR")
    _TF_OK = True
except ImportError:
    _TF_OK = False


def _make_sequences(values: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback: i])
        y.append(values[i])
    return np.array(X), np.array(y)


class LSTMForecaster(BaseForecaster):
    """
    Parameters
    ----------
    lookback   : int   Input window length (days)
    units      : int   LSTM hidden units in first layer
    dropout    : float Dropout rate
    epochs     : int   Max training epochs
    batch_size : int   Mini-batch size
    patience   : int   Early-stopping patience
    """

    name = "LSTM"

    def __init__(
        self,
        horizon: int    = 28,
        lookback: int   = 28,
        units: int      = 64,
        dropout: float  = 0.2,
        epochs: int     = 50,
        batch_size: int = 32,
        patience: int   = 10,
    ):
        super().__init__(horizon)
        self.lookback   = lookback
        self.units      = units
        self.dropout    = dropout
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self._model     = None
        self._mu        = 0.0
        self._sigma     = 1.0

    def _build_model(self):
        model = keras.Sequential([
            layers.Input(shape=(self.lookback, 1)),
            layers.LSTM(self.units, return_sequences=True,
                        dropout=self.dropout, recurrent_dropout=0.1),
            layers.LSTM(self.units // 2, dropout=self.dropout),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber")
        return model

    def fit(self, train: pd.DataFrame, target: str = "sales") -> None:
        self._validate_train(train, target)
        values     = train[target].values.astype(float)
        self._mu   = values.mean()
        self._sigma = values.std() + 1e-8
        scaled     = (values - self._mu) / self._sigma

        X, y = _make_sequences(scaled, self.lookback)
        X    = X[..., np.newaxis]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = self._build_model()
            cbs = [
                keras.callbacks.EarlyStopping(
                    patience=self.patience, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            ]
            self._model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.15,
                callbacks=cbs,
                verbose=0,
            )

        self._last_scaled  = scaled[-self.lookback:]
        self._train_values = values
        self._last_date    = pd.to_datetime(train["date"].max())
        self.is_fitted     = True

    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        assert self.is_fitted, "Call fit() first."
        horizon  = horizon or self.horizon
        context  = self._last_scaled.tolist()
        preds    = []

        for _ in range(horizon):
            x    = np.array(context[-self.lookback:])[np.newaxis, :, np.newaxis]
            pred = float(self._model.predict(x, verbose=0)[0, 0])
            preds.append(pred)
            context.append(pred)

        denorm   = np.array(preds) * self._sigma + self._mu
        denorm   = np.clip(denorm, 0, None)
        rs       = self._train_values[-28:].std()
        lower    = np.clip(denorm - 1.645 * rs, 0, None)
        upper    = denorm + 1.645 * rs
        future   = self._make_future_dates(self._last_date, horizon)

        return pd.DataFrame({
            "date": future, "yhat": denorm,
            "yhat_lower": lower, "yhat_upper": upper, "model": self.name,
        })


# ── Fallback ──────────────────────────────────────────────────────────────────
class RollingLinearForecaster(BaseForecaster):
    """Rolling linear trend + seasonal decomposition fallback."""

    name = "LSTM (Linear Approx)"

    def fit(self, train: pd.DataFrame, target: str = "sales") -> None:
        self._validate_train(train, target)
        self._values    = train[target].values.astype(float)
        self._last_date = pd.to_datetime(train["date"].max())
        self.is_fitted  = True

    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        horizon = horizon or self.horizon
        v       = self._values
        w       = min(60, len(v))
        coeffs  = np.polyfit(np.arange(w), v[-w:], 1)
        preds   = []
        for i in range(horizon):
            t    = w + i
            base = np.polyval(coeffs, t)
            s_ix = (len(v) + i) % 7
            seas = np.mean(v[s_ix::7]) - np.mean(v) if len(v) >= 28 else 0
            preds.append(max(0, base + seas))
        preds = np.array(preds)
        resid = v[-14:] - np.mean(v[-14:])
        lower, upper = self.get_confidence_interval(preds, resid)
        future = self._make_future_dates(self._last_date, horizon)
        return pd.DataFrame({
            "date": future, "yhat": preds,
            "yhat_lower": lower, "yhat_upper": upper, "model": self.name,
        })


def get_lstm_forecaster(**kwargs) -> BaseForecaster:
    return LSTMForecaster(**kwargs) if _TF_OK else RollingLinearForecaster(**kwargs)
