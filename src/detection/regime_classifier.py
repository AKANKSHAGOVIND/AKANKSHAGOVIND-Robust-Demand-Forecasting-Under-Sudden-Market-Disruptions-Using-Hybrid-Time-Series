"""
src/detection/regime_classifier.py
------------------------------------
Classifies each day as STABLE / DISRUPTED / RECOVERY
based on CUSUM change-point positions and rolling volatility.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
import pandas as pd

from src.detection.cusum import CUSUMResult


class Regime(str, Enum):
    STABLE    = "Stable"
    DISRUPTED = "Disrupted"
    RECOVERY  = "Recovery"


REGIME_COLORS = {
    Regime.STABLE   : "#22c55e",
    Regime.DISRUPTED: "#ef4444",
    Regime.RECOVERY : "#f59e0b",
}

REGIME_ICONS = {
    Regime.STABLE   : "",
    Regime.DISRUPTED: "",
    Regime.RECOVERY : "",
}


@dataclass
class RegimeResult:
    current_regime: Regime
    regime_series:  pd.Series
    regime_df:      pd.DataFrame
    summary:        dict


class RegimeClassifier:
    """
    Parameters
    ----------
    disruption_window : int   Days after a change point -> DISRUPTED
    recovery_window   : int   Days following disruption window -> RECOVERY
    vol_multiplier    : float Extra volatility threshold for STABLE override
    """

    def __init__(
        self,
        disruption_window: int = 28,
        recovery_window:   int = 42,
        vol_multiplier: float  = 1.5,
    ):
        self.disruption_window = disruption_window
        self.recovery_window   = recovery_window
        self.vol_multiplier    = vol_multiplier

    def classify(self, series: pd.Series, cusum_result: CUSUMResult) -> RegimeResult:
        n = len(series)

        # Store as plain strings to avoid pandas enum/idxmax ambiguity across versions
        S = "Stable"
        D = "Disrupted"
        R = "Recovery"

        regimes = np.full(n, S, dtype=object)

        # Label windows around each change point
        for cp in cusum_result.change_points:
            d_end = min(n, cp + self.disruption_window)
            r_end = min(n, cp + self.disruption_window + self.recovery_window)
            regimes[cp:d_end] = D
            for j in range(d_end, r_end):
                if regimes[j] == S:
                    regimes[j] = R

        # Volatility override
        roll_std   = series.rolling(28, min_periods=7).std()
        global_std = series.std()
        for i in range(n):
            if (regimes[i] == S
                    and not np.isnan(roll_std.iloc[i])
                    and roll_std.iloc[i] > self.vol_multiplier * global_std):
                regimes[i] = D

        # regime_series holds plain strings — idxmax always returns a str
        regime_series = pd.Series(regimes, index=series.index, dtype=object)

        dates = series.index if isinstance(series.index, pd.DatetimeIndex) \
                else pd.RangeIndex(n)

        regime_df = pd.DataFrame({
            "date"  : dates,
            "sales" : series.values,
            "regime": regime_series.values,
        })

        # Safe: idxmax() now returns "Stable" / "Disrupted" / "Recovery"
        current_str    = str(regime_series.iloc[-7:].value_counts().idxmax())
        current_regime = Regime(current_str)

        summary = {
            "current": current_regime.value,
            "counts": {r.value: int((regime_series == r.value).sum()) for r in Regime},
            "n_changes": len(cusum_result.change_points),
            "change_dates": [str(d)[:10] for d in cusum_result.change_dates],
        }

        return RegimeResult(
            current_regime=current_regime,
            regime_series=regime_series,
            regime_df=regime_df,
            summary=summary,
        )


def get_current_regime(series: pd.Series, cusum_result: CUSUMResult) -> Regime:
    return RegimeClassifier().classify(series, cusum_result).current_regime