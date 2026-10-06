"""
src/models/base_model.py
------------------------
Abstract base class that all forecasting models implement.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """
    Common interface for ARIMA, Prophet, LSTM, Transformer and Ensemble.

    Subclasses MUST implement:
        fit(train, target)
        predict(horizon)
    """

    name: str = "BaseForecaster"

    def __init__(self, horizon: int = 28):
        self.horizon     = horizon
        self.is_fitted   = False
        self._train_data = None

    @abstractmethod
    def fit(self, train: pd.DataFrame, target: str = "sales") -> None:
        """Fit model on DataFrame containing 'date' and target columns."""
        ...

    @abstractmethod
    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        """
        Return forecast DataFrame with columns:
            date | yhat | yhat_lower | yhat_upper | model
        """
        ...

    # ── Convenience ──────────────────────────────────────────────────────────
    def fit_predict(self, train: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
        self.fit(train, target)
        return self.predict()

    def get_confidence_interval(
        self,
        predictions: np.ndarray,
        residuals:   np.ndarray,
        alpha: float = 0.10,
    ):
        """90 % CI derived from residual std — override for model-specific CI."""
        sigma = np.std(residuals) if len(residuals) > 1 else 1.0
        z     = 1.645  # 90 % z-score
        lower = np.clip(predictions - z * sigma, 0, None)
        upper = predictions + z * sigma
        return lower, upper

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _make_future_dates(self, last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
        return pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

    def _validate_train(self, train: pd.DataFrame, target: str) -> None:
        assert "date"  in train.columns, "train must have a 'date' column"
        assert target  in train.columns, f"train must have a '{target}' column"
        assert len(train) >= 14,         "Need at least 14 training samples"
