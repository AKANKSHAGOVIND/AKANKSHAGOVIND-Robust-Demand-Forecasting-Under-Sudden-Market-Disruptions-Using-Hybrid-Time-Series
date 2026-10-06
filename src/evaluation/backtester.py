"""
src/evaluation/backtester.py
-----------------------------
Walk-forward (expanding-window) cross-validation for fair model comparison.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics


@dataclass
class BacktestResult:
    model_name:   str
    metrics:      Dict[str, float]
    predictions:  pd.DataFrame
    fold_metrics: List[Dict]


def walk_forward_validate(
    train_df:     pd.DataFrame,
    model_cls,                     # BaseForecaster subclass (class object, not instance)
    model_kwargs: Optional[dict] = None,
    n_folds:      int = 3,
    min_train:    int = 90,
    horizon:      int = 28,
    target:       str = "sales",
) -> BacktestResult:
    """
    Expanding-window walk-forward cross-validation.

    For each fold the model is re-fitted on all data up to the fold boundary,
    then evaluated on the next `horizon` days.
    """
    model_kwargs = model_kwargs or {}
    df           = train_df.sort_values("date").reset_index(drop=True)
    n            = len(df)

    usable  = n - min_train
    n_folds = max(1, min(n_folds, usable // max(horizon, 1)))

    fold_ends    = np.linspace(min_train, n - horizon, n_folds, dtype=int)
    all_preds    = []
    fold_metrics = []

    for fold, end_idx in enumerate(fold_ends):
        train_fold = df.iloc[:end_idx].copy()
        test_fold  = df.iloc[end_idx: end_idx + horizon].copy()
        if len(test_fold) == 0:
            continue

        try:
            m = model_cls(horizon=horizon, **model_kwargs)
            m.fit(train_fold, target)
            pred_df  = m.predict(len(test_fold))
            actual   = test_fold[target].values
            pred_arr = pred_df["yhat"].values[: len(actual)]

            fold_metrics.append(
                compute_all_metrics(actual, pred_arr, model_name=f"Fold {fold+1}")
            )
            all_preds.append(pd.DataFrame({
                "date"     : test_fold["date"].values,
                "actual"   : actual,
                "predicted": pred_arr,
                "fold"     : fold + 1,
            }))
        except Exception as e:
            print(f"[Backtest] Fold {fold+1} failed ({model_cls.__name__}): {e}")

    if not all_preds:
        return BacktestResult(model_cls.__name__, {}, pd.DataFrame(), [])

    all_df  = pd.concat(all_preds, ignore_index=True)
    overall = compute_all_metrics(
        all_df["actual"].values, all_df["predicted"].values,
        model_name=getattr(model_cls, "name", model_cls.__name__),
    )

    return BacktestResult(
        model_name   = overall["Model"],
        metrics      = overall,
        predictions  = all_df,
        fold_metrics = fold_metrics,
    )


def compare_models(
    train_df:    pd.DataFrame,
    model_specs: List[Dict[str, Any]],
    n_folds:     int = 3,
    min_train:   int = 90,
    horizon:     int = 28,
    target:      str = "sales",
) -> pd.DataFrame:
    """
    Run walk-forward CV for multiple models and return a sorted comparison table.

    Parameters
    ----------
    model_specs : list of {"cls": <class>, "kwargs": <dict>, "name": <str>}
    """
    rows = []
    for spec in model_specs:
        result = walk_forward_validate(
            train_df, spec["cls"], spec.get("kwargs", {}),
            n_folds, min_train, horizon, target,
        )
        if result.metrics:
            rows.append(result.metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "Model" in df.columns:
        df = df.set_index("Model")
    return df.sort_values("MAE")
