"""
src/data/preprocessor.py
------------------------
Merges wide sales + calendar + prices into a long-format time-series DataFrame.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# Wide → Long
# ═════════════════════════════════════════════════════════════════════════════

def wide_to_long(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt wide sales matrix (d_1 … d_N cols) into long format,
    then join calendar metadata on the 'd' key.
    """
    id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    id_cols  = [c for c in id_cols if c in sales_df.columns]
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]

    long_df = sales_df[id_cols + day_cols].melt(
        id_vars=id_cols, value_vars=day_cols,
        var_name="d", value_name="sales",
    )
    long_df["d_num"] = long_df["d"].str.extract(r"(\d+)").astype(int)
    long_df = long_df.sort_values(["item_id", "store_id", "d_num"]).drop(columns="d_num")

    # Calendar join
    cal_base  = ["d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
                 "event_name_1", "event_type_1"]
    snap_cols = [c for c in calendar_df.columns if c.startswith("snap_")]
    cal_cols  = [c for c in cal_base + snap_cols if c in calendar_df.columns]

    long_df = long_df.merge(calendar_df[cal_cols], on="d", how="left")
    long_df["date"] = pd.to_datetime(long_df["date"])
    return long_df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Price join
# ═════════════════════════════════════════════════════════════════════════════

def attach_prices(long_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Join weekly sell prices on store_id + item_id + wm_yr_wk."""
    merged = long_df.merge(
        prices_df[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    merged["sell_price"] = (
        merged.groupby(["store_id", "item_id"])["sell_price"]
              .transform(lambda x: x.ffill().bfill())
    )
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# Series helpers
# ═════════════════════════════════════════════════════════════════════════════

def filter_series(df: pd.DataFrame, store_id: str, item_id: str) -> pd.DataFrame:
    """Extract a single store × item time series."""
    ts = df[(df["store_id"] == store_id) & (df["item_id"] == item_id)].copy()
    return ts.sort_values("date").reset_index(drop=True)


def apply_cutoff(df: pd.DataFrame, cutoff_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split at cutoff_date → (train, test)."""
    cutoff = pd.Timestamp(cutoff_date)
    return df[df["date"] <= cutoff].copy(), df[df["date"] > cutoff].copy()


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sales"]      = df["sales"].fillna(0).clip(lower=0)
    if "sell_price" in df.columns:
        df["sell_price"] = df["sell_price"].ffill().bfill().fillna(0)
    return df


def build_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to daily frequency, filling gaps with 0."""
    df = df.set_index("date")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range)
    df["sales"] = df["sales"].fillna(0)
    df.index.name = "date"
    return df.reset_index()


def get_store_item_list(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return {store_id: [item_id, …]} for UI dropdowns."""
    result = {}
    for store, grp in df.groupby("store_id"):
        result[store] = sorted(grp["item_id"].unique().tolist())
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═════════════════════════════════════════════════════════════════════════════

def full_pipeline(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    store_id: Optional[str] = None,
    item_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    wide_to_long → attach_prices → handle_missing
    Optionally filter to a single store+item series.
    """
    long_df = wide_to_long(sales_df, calendar_df)
    long_df = attach_prices(long_df, prices_df)
    long_df = handle_missing(long_df)

    if store_id and item_id:
        long_df = filter_series(long_df, store_id, item_id)
        long_df = build_time_index(long_df)

    return long_df
