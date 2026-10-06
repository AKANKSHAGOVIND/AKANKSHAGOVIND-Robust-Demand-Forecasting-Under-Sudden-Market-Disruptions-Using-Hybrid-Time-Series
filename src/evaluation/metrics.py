"""
src/evaluation/metrics.py
--------------------------
MAE, RMSE, MAPE, sMAPE, R² calculators.
"""

from typing import Dict

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    """MAPE — excludes zero-actual rows to avoid division by zero."""
    mask = np.abs(actual) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(100.0 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def smape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    denom = (np.abs(actual) + np.abs(predicted)) / 2 + eps
    return float(100.0 * np.mean(np.abs(actual - predicted) / denom))


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2) + 1e-8
    return float(1.0 - ss_res / ss_tot)


def compute_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    a = np.asarray(actual,    dtype=float)
    p = np.asarray(predicted, dtype=float)
    n = min(len(a), len(p))
    a, p = a[:n], p[:n]
    return {
        "Model": model_name,
        "MAE"  : round(mae(a, p),      4),
        "RMSE" : round(rmse(a, p),     4),
        "MAPE" : round(mape(a, p),     2),
        "sMAPE": round(smape(a, p),    2),
        "R²"   : round(r2_score(a, p), 4),
    }


def build_metrics_table(
    forecasts: Dict[str, pd.DataFrame],
    actual:    pd.Series,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    forecasts : {model_name: DataFrame(date, yhat, ...)}
    actual    : pd.Series indexed by date

    Returns
    -------
    pd.DataFrame with metrics per model, sorted by MAE ascending.
    """
    rows = []
    for name, df in forecasts.items():
        aligned = df.set_index("date")["yhat"].reindex(actual.index)
        valid   = actual.notna() & aligned.notna()
        if valid.sum() == 0:
            continue
        rows.append(compute_all_metrics(
            actual[valid].values, aligned[valid].values, model_name=name))

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("Model")
    return out.sort_values("MAE")
