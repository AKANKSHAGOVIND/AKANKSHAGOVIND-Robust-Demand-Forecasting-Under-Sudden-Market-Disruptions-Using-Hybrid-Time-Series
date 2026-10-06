"""
src/detection/cusum.py
-----------------------
Two-sided CUSUM (Cumulative Sum Control Chart) for change-point detection.
Detects both upward and downward structural breaks in a time series.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class CUSUMResult:
    """Container for CUSUM detection results."""
    change_points: List[int]            # Sample indices of detected breaks
    change_dates:  List[pd.Timestamp]  # Corresponding dates (if DatetimeIndex)
    cusum_pos:     np.ndarray           # Upward cumulative sum statistic
    cusum_neg:     np.ndarray           # Downward cumulative sum statistic
    threshold:     float                # Decision threshold (h × sigma)
    magnitudes:    List[float]          # |mean shift| at each change point


class CUSUMDetector:
    """
    Two-sided CUSUM detector for demand disruptions.

    Parameters
    ----------
    threshold : float
        Decision threshold multiplier h (signal when CUSUM > h × sigma).
        Typical range: 4–8. Higher = fewer, more confident detections.
    drift : float
        Allowable drift k in sigma units. Typical: 0.5–1.0.
    min_segment : int
        Minimum number of samples between consecutive change points.
    """

    def __init__(
        self,
        threshold: float = 5.0,
        drift: float = 0.5,
        min_segment: int = 14,
    ):
        self.threshold   = threshold
        self.drift       = drift
        self.min_segment = min_segment

    def fit(self, series: pd.Series) -> CUSUMResult:
        """
        Run CUSUM on a pandas Series (preferably DatetimeIndex).

        Returns a CUSUMResult with change-point indices, dates, statistics.
        """
        values = series.ffill().fillna(0).values.astype(float)
        dates  = (series.index
                  if isinstance(series.index, pd.DatetimeIndex)
                  else pd.RangeIndex(len(values)))

        mu    = np.mean(values)
        sigma = np.std(values) + 1e-8
        k     = self.drift    * sigma  # allowable slack
        h     = self.threshold * sigma  # decision boundary

        n       = len(values)
        s_pos   = np.zeros(n)
        s_neg   = np.zeros(n)
        changes = []

        for i in range(1, n):
            xi       = values[i]
            s_pos[i] = max(0.0, s_pos[i - 1] + (xi - mu) - k)
            s_neg[i] = max(0.0, s_neg[i - 1] - (xi - mu) - k)

            triggered = s_pos[i] > h or s_neg[i] > h
            if triggered:
                gap_ok = not changes or (i - changes[-1]) >= self.min_segment
                if gap_ok:
                    changes.append(i)
                # Reset after detection (regardless of gap)
                s_pos[i] = 0.0
                s_neg[i] = 0.0

        # Compute magnitude of shift at each change point
        magnitudes = []
        for cp in changes:
            before = values[max(0, cp - 14) : cp]
            after  = values[cp : min(n, cp + 14)]
            mag    = abs(np.mean(after) - np.mean(before))
            magnitudes.append(round(float(mag), 4))

        # Resolve dates
        change_dates = []
        for idx in changes:
            try:
                change_dates.append(dates[idx])
            except Exception:
                change_dates.append(idx)

        return CUSUMResult(
            change_points=changes,
            change_dates=change_dates,
            cusum_pos=s_pos,
            cusum_neg=s_neg,
            threshold=h,
            magnitudes=magnitudes,
        )


def detect_change_points(
    series: pd.Series,
    threshold: float = 5.0,
    drift: float = 0.5,
    min_segment: int = 14,
) -> CUSUMResult:
    """Convenience wrapper — create detector and fit in one call."""
    return CUSUMDetector(threshold=threshold, drift=drift,
                         min_segment=min_segment).fit(series)


def cusum_to_dataframe(result: CUSUMResult, series: pd.Series) -> pd.DataFrame:
    """Package CUSUM statistics + original values into a plottable DataFrame."""
    n = min(len(series), len(result.cusum_pos))
    df = pd.DataFrame({
        "date"      : series.index[:n],
        "sales"     : series.values[:n],
        "cusum_pos" : result.cusum_pos[:n],
        "cusum_neg" : result.cusum_neg[:n],
        "threshold" : result.threshold,
    })
    df["is_change_point"] = False
    for cp in result.change_points:
        if cp < len(df):
            df.loc[cp, "is_change_point"] = True
    return df.reset_index(drop=True)
