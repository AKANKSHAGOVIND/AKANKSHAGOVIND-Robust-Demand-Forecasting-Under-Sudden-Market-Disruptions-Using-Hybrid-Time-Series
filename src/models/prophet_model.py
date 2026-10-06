"""
src/models/prophet_model.py
----------------------------
Facebook Prophet wrapper with M5 holiday injection and SNAP regressors.
Falls back to trend+seasonal decomposition if prophet is unavailable.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseForecaster

try:
    from prophet import Prophet
    _PROPHET_OK = True
except ImportError:
    _PROPHET_OK = False


class ProphetForecaster(BaseForecaster):
    """
    Parameters
    ----------
    changepoint_prior_scale : float  Flexibility of trend (default 0.05)
    seasonality_prior_scale : float  Strength of seasonality (default 10)
    """

    name = "Prophet"

    def __init__(
        self,
        horizon: int = 28,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality:  bool = False,
    ):
        super().__init__(horizon)
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality      = yearly_seasonality
        self.weekly_seasonality      = weekly_seasonality
        self.daily_seasonality       = daily_seasonality
        self._model       = None
        self._regressors  = []

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _build_holidays(self, train: pd.DataFrame):
        if "event_name_1" not in train.columns:
            return None
        rows = []
        for _, row in train[["date", "event_name_1"]].iterrows():
            if pd.notna(row["event_name_1"]) and row["event_name_1"] != "":
                rows.append({
                    "holiday"       : row["event_name_1"],
                    "ds"            : row["date"],
                    "lower_window"  : -1,
                    "upper_window"  : 1,
                })
        return pd.DataFrame(rows) if rows else None

    def _get_regressors(self, df: pd.DataFrame):
        regs = [c for c in df.columns if c.startswith("snap_")]
        if "sell_price" in df.columns:
            regs.append("sell_price")
        return regs

    # ── fit / predict ─────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame, target: str = "sales") -> None:
        self._validate_train(train, target)
        self._train_data = train.copy()

        prophet_df       = pd.DataFrame()
        prophet_df["ds"] = pd.to_datetime(train["date"])
        prophet_df["y"]  = train[target].values.astype(float)

        self._regressors = self._get_regressors(train)
        for col in self._regressors:
            prophet_df[col] = train[col].fillna(0).values

        holidays = self._build_holidays(train)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                holidays=holidays,
                interval_width=0.90,
            )
            for col in self._regressors:
                self._model.add_regressor(col)
            self._model.fit(prophet_df)

        self._last_date = pd.to_datetime(train["date"].max())
        self.is_fitted  = True

    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        assert self.is_fitted, "Call fit() first."
        horizon = horizon or self.horizon

        future = self._model.make_future_dataframe(periods=horizon, freq="D")
        for col in self._regressors:
            if col in self._train_data.columns:
                last_val = self._train_data[col].iloc[-1]
                future[col] = 0.0
                future.loc[future.index[-horizon:], col] = last_val

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._model.predict(future)

        tail = fc.tail(horizon)
        return pd.DataFrame({
            "date"      : tail["ds"].values,
            "yhat"      : np.clip(tail["yhat"].values, 0, None),
            "yhat_lower": np.clip(tail["yhat_lower"].values, 0, None),
            "yhat_upper": tail["yhat_upper"].values,
            "model"     : self.name,
        })


# ── Fallback ──────────────────────────────────────────────────────────────────
class NaiveProphetForecaster(BaseForecaster):
    """Linear trend + weekly seasonality fallback."""

    name = "Prophet (Trend+Season)"

    def fit(self, train: pd.DataFrame, target: str = "sales") -> None:
        self._validate_train(train, target)
        self._values    = train[target].values.astype(float)
        self._last_date = pd.to_datetime(train["date"].max())
        self.is_fitted  = True

    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        horizon = horizon or self.horizon
        v       = self._values
        trend   = np.polyfit(np.arange(len(v)), v, 1)
        preds   = []
        for i in range(horizon):
            t    = len(v) + i
            base = np.polyval(trend, t)
            s_ix = (len(v) + i) % 7
            seas = np.mean(v[s_ix::7]) - np.mean(v) if len(v) > 7 else 0
            preds.append(max(0, base + seas))
        preds = np.array(preds)
        resid = v[-14:] - np.mean(v[-14:])
        lower, upper = self.get_confidence_interval(preds, resid)
        future = self._make_future_dates(self._last_date, horizon)
        return pd.DataFrame({
            "date": future, "yhat": preds,
            "yhat_lower": lower, "yhat_upper": upper, "model": self.name,
        })


def get_prophet_forecaster(**kwargs) -> BaseForecaster:
    return ProphetForecaster(**kwargs) if _PROPHET_OK else NaiveProphetForecaster(**kwargs)
