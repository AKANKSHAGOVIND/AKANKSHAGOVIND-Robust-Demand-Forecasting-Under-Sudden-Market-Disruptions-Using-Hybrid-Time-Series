"""
src/export/csv_exporter.py
---------------------------
Build and serialize the downloadable 28-day forecast CSV.
"""

import io
from typing import Dict

import numpy as np
import pandas as pd


def build_export_df(
    forecasts: Dict[str, pd.DataFrame],
    store_id:  str,
    item_id:   str,
) -> pd.DataFrame:
    """
    Combine all model forecasts into a wide-format DataFrame.

    Output columns:
        date | store_id | item_id | ARIMA_yhat | Prophet_yhat | … |
        Ensemble_yhat | Ensemble_lower | Ensemble_upper
    """
    base_df = None

    for mname, fc_df in forecasts.items():
        safe = mname.replace(" ", "_").replace("(", "").replace(")", "")
        sub  = fc_df[["date", "yhat"]].rename(columns={"yhat": f"{safe}_yhat"})

        if "yhat_lower" in fc_df.columns:
            sub[f"{safe}_lower"] = fc_df["yhat_lower"].values
            sub[f"{safe}_upper"] = fc_df["yhat_upper"].values

        base_df = sub if base_df is None else base_df.merge(sub, on="date", how="outer")

    if base_df is None:
        return pd.DataFrame()

    base_df.insert(0, "store_id", store_id)
    base_df.insert(1, "item_id",  item_id)
    base_df["date"] = pd.to_datetime(base_df["date"]).dt.strftime("%Y-%m-%d")

    num_cols = base_df.select_dtypes(include="number").columns
    base_df[num_cols] = base_df[num_cols].round(2)

    return base_df.sort_values("date").reset_index(drop=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to UTF-8 CSV bytes (for st.download_button)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def forecast_summary_stats(df: pd.DataFrame, col: str = None) -> dict:
    """Quick summary statistics over the forecast horizon."""
    if col is None or col not in df.columns:
        yhat_cols = [c for c in df.columns if c.endswith("_yhat")]
        col = next((c for c in yhat_cols if "Ensemble" in c), yhat_cols[0] if yhat_cols else None)
    if col is None:
        return {}

    v = df[col].dropna()
    return {
        "total": round(float(v.sum()),  2),
        "mean" : round(float(v.mean()), 2),
        "min"  : round(float(v.min()),  2),
        "max"  : round(float(v.max()),  2),
        "std"  : round(float(v.std()),  2),
    }
