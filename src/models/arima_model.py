"""
src/models/arima_model.py
--------------------------
ARIMA / SARIMAX wrapper with automatic order selection via AIC grid search.
Falls back to seasonal-naive if statsmodels is unavailable.
"""

import warnings
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseForecaster

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    _STATSMODELS_OK = True
except (ImportError, TypeError):
    _STATSMODELS_OK = False


class ARIMAForecaster(BaseForecaster):
    """
    SARIMA(p,d,q)(P,D,Q,s) wrapper.

    Parameters
    ----------
    horizon        : int   Forecast horizon in days
    order          : tuple (p,d,q) — auto-selected if None
    seasonal_order : tuple (P,D,Q,s) — default (0,0,0,7)
    auto_order     : bool  Run AIC grid search if True
    """

    name = "ARIMA"

    def __init__(
        self,
        horizon: int = 28,
        order: Optional[tuple] = None,
        seasonal_order: tuple = (0, 0, 0, 7),
        auto_order: bool = True,
    ):
        super().__init__(horizon)
        self.order          = order or (2, 1, 2)
        self.seasonal_order = seasonal_order
        self.auto_order     = auto_order
        self._model_fit     = None
        self._residuals     = np.array([])

    # ── Auto order selection ──────────────────────────────────────────────────
    def _select_order(self, values: np.ndarray) -> tuple:
        best_aic, best_order = np.inf, (1, 1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pval = adfuller(values)[1]
                d_range = [0] if pval < 0.05 else [1]
            except Exception:
                d_range = [1]

            for p, d, q in product(range(3), d_range, range(3)):
                try:
                    res = SARIMAX(
                        values, order=(p, d, q),
                        seasonal_order=(0, 0, 0, 0),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=50)
                    if res.aic < best_aic:
                        best_aic, best_order = res.aic, (p, d, q)
                except Exception:
                    continue
        return best_order

    # ── fit / predict ─────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame, target: str = "sales") -> None:
        self._validate_train(train, target)
        self._train_data = train.copy()
        series = train.set_index("date")[target].asfreq("D").fillna(0)
        values = series.values.astype(float)

        if self.auto_order:
            self.order = self._select_order(values)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                values,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._model_fit = model.fit(disp=False, maxiter=200)

        self._residuals = self._model_fit.resid
        self._last_date = series.index[-1]
        self.is_fitted  = True

    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        assert self.is_fitted, "Call fit() first."
        horizon = horizon or self.horizon

        fc       = self._model_fit.get_forecast(steps=horizon)
        yhat     = np.clip(fc.predicted_mean, 0, None)
        ci_raw   = fc.conf_int(alpha=0.10)
        # conf_int() returns DataFrame or ndarray depending on statsmodels version
        if hasattr(ci_raw, "iloc"):
            ci_lower = ci_raw.iloc[:, 0].values
            ci_upper = ci_raw.iloc[:, 1].values
        else:
            ci_arr   = np.asarray(ci_raw)
            ci_lower = ci_arr[:, 0]
            ci_upper = ci_arr[:, 1]
        future   = self._make_future_dates(self._last_date, horizon)

        return pd.DataFrame({
            "date"      : future,
            "yhat"      : yhat,
            "yhat_lower": np.clip(ci_lower, 0, None),
            "yhat_upper": ci_upper,
            "model"     : self.name,
        })


# ── Seasonal Naive Fallback ───────────────────────────────────────────────────
class NaiveARIMAForecaster(BaseForecaster):
    """Seasonal-naive substitute when statsmodels is not installed."""

    name = "ARIMA (Naive)"

    def fit(self, train: pd.DataFrame, target: str = "sales") -> None:
        self._validate_train(train, target)
        self._values    = train[target].values.astype(float)
        self._last_date = pd.to_datetime(train["date"].max())
        self.is_fitted  = True

    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        horizon  = horizon or self.horizon
        baseline = self._values[-7:]
        preds    = np.array([baseline[i % 7] for i in range(horizon)])
        preds    = np.clip(preds, 0, None)
        resid    = self._values[-28:] - np.mean(self._values[-28:])
        lower, upper = self.get_confidence_interval(preds, resid)
        future   = self._make_future_dates(self._last_date, horizon)
        return pd.DataFrame({
            "date": future, "yhat": preds,
            "yhat_lower": lower, "yhat_upper": upper, "model": self.name,
        })


def get_arima_forecaster(**kwargs) -> BaseForecaster:
    """Return the best available ARIMA implementation."""
    return ARIMAForecaster(**kwargs) if _STATSMODELS_OK else NaiveARIMAForecaster(**kwargs)