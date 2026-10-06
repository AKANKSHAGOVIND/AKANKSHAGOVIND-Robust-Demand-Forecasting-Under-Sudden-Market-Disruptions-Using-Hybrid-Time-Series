"""
src/models/ensemble_model.py
-----------------------------
Regime-adaptive weighted ensemble.
Routes different model weight profiles based on the detected market regime.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.base_model import BaseForecaster
from src.detection.regime_classifier import Regime


# ── Regime weight tables ──────────────────────────────────────────────────────
DEFAULT_WEIGHTS: Dict[Regime, Dict[str, float]] = {
    Regime.STABLE: {
        "ARIMA": 0.35, "Prophet": 0.35, "LSTM": 0.20, "Transformer": 0.10,
    },
    Regime.DISRUPTED: {
        "ARIMA": 0.10, "Prophet": 0.15, "LSTM": 0.40, "Transformer": 0.35,
    },
    Regime.RECOVERY: {
        "ARIMA": 0.20, "Prophet": 0.30, "LSTM": 0.30, "Transformer": 0.20,
    },
}


class EnsembleForecaster(BaseForecaster):
    """
    Combines ARIMA, Prophet, LSTM, Transformer with regime-dependent weights.

    Parameters
    ----------
    regime         : current market regime (determines weight profile)
    custom_weights : override default weights per model name
    """

    name = "Ensemble"

    def __init__(
        self,
        horizon: int = 28,
        regime: Regime = Regime.STABLE,
        custom_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__(horizon)
        self.regime         = regime
        self.custom_weights = custom_weights
        self._forecasters:  List[BaseForecaster] = []
        self._predictions:  Dict[str, pd.DataFrame] = {}
        self._weights:      Dict[str, float] = {}

    def set_forecasters(self, forecasters: List[BaseForecaster]) -> None:
        self._forecasters = forecasters

    # ── fit / predict ─────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame, target: str = "sales") -> None:
        self._validate_train(train, target)
        self._train_data = train.copy()
        self._last_date  = pd.to_datetime(train["date"].max())

        for fc in self._forecasters:
            try:
                fc.fit(train, target)
            except Exception as e:
                print(f"[Ensemble] {fc.name} fit failed: {e}")

        self.is_fitted = True

    def predict(self, horizon: Optional[int] = None) -> pd.DataFrame:
        assert self.is_fitted, "Call fit() first."
        horizon = horizon or self.horizon

        weight_table = self.custom_weights or DEFAULT_WEIGHTS[self.regime]
        fc_map: Dict[str, pd.DataFrame] = {}

        for fc in self._forecasters:
            try:
                pred      = fc.predict(horizon)
                base_name = fc.name.split(" ")[0]   # strip fallback suffix
                fc_map[base_name] = pred
            except Exception as e:
                print(f"[Ensemble] {fc.name} predict failed: {e}")

        if not fc_map:
            raise RuntimeError("All child models failed.")

        # Normalize weights to available models
        avail   = {k: v for k, v in weight_table.items() if k in fc_map}
        total_w = sum(avail.values())
        if total_w == 0:
            avail   = {k: 1.0 / len(fc_map) for k in fc_map}
            total_w = 1.0
        norm_w        = {k: v / total_w for k, v in avail.items()}
        self._weights = norm_w

        # Weighted average
        future     = self._make_future_dates(self._last_date, horizon)
        yhat       = np.zeros(horizon)
        yhat_lower = np.zeros(horizon)
        yhat_upper = np.zeros(horizon)

        for mname, w in norm_w.items():
            df         = fc_map[mname]
            yhat       += w * df["yhat"].values[:horizon]
            yhat_lower += w * df["yhat_lower"].values[:horizon]
            yhat_upper += w * df["yhat_upper"].values[:horizon]

        self._predictions = fc_map

        return pd.DataFrame({
            "date"      : future,
            "yhat"      : np.clip(yhat, 0, None),
            "yhat_lower": np.clip(yhat_lower, 0, None),
            "yhat_upper": yhat_upper,
            "model"     : f"{self.name} ({self.regime.value})",
        })

    # ── Accessors ─────────────────────────────────────────────────────────────
    def get_all_predictions(self, horizon: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        if not self._predictions:
            self.predict(horizon)
        return self._predictions

    def get_weights(self) -> Dict[str, float]:
        return self._weights

    # ── Dynamic reweighting via inverse-MAE ───────────────────────────────────
    def performance_based_reweight(self, val_errors: Dict[str, float]) -> None:
        """
        Blend regime-default weights with inverse-MAE weights (70/30 split).

        Parameters
        ----------
        val_errors : {model_name: MAE_value}
        """
        inv     = {k: 1.0 / (v + 1e-8) for k, v in val_errors.items()}
        total   = sum(inv.values())
        perf_w  = {k: v / total for k, v in inv.items()}

        default = DEFAULT_WEIGHTS[self.regime]
        blended = {}
        for k in perf_w:
            base = default.get(k, 0.25)
            blended[k] = 0.70 * perf_w[k] + 0.30 * base

        total_b = sum(blended.values())
        self.custom_weights = {k: v / total_b for k, v in blended.items()}
