"""
src/data/feature_engineer.py
-----------------------------
Lag features, rolling statistics, calendar dummies, Fourier terms, price features.
Used as input to ML/DL models.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ── Lag & Rolling ─────────────────────────────────────────────────────────────

def add_lags(df: pd.DataFrame, target: str = "sales",
             lags: List[int] = [1, 7, 14, 28]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
    return df


def add_rolling_stats(df: pd.DataFrame, target: str = "sales",
                      windows: List[int] = [7, 14, 28]) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        s = df[target].shift(1)
        df[f"roll_mean_{w}"] = s.rolling(w).mean()
        df[f"roll_std_{w}"]  = s.rolling(w).std()
        df[f"roll_max_{w}"]  = s.rolling(w).max()
    return df


def add_ewm(df: pd.DataFrame, target: str = "sales",
            spans: List[int] = [7, 14]) -> pd.DataFrame:
    df = df.copy()
    for span in spans:
        df[f"ewm_{span}"] = df[target].shift(1).ewm(span=span, adjust=False).mean()
    return df


# ── Calendar Features ─────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["dayofmonth"] = df["date"].dt.day
    df["dayofyear"]  = df["date"].dt.dayofyear
    df["week"]       = df["date"].dt.isocalendar().week.astype(int)
    df["month"]      = df["date"].dt.month
    df["year"]       = df["date"].dt.year
    df["quarter"]    = df["date"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Fourier terms for weekly & annual seasonality
    for k in [1, 2, 3]:
        df[f"sin_week_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 7)
        df[f"cos_week_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 7)
        df[f"sin_year_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365.25)
        df[f"cos_year_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365.25)
    return df


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "event_name_1" in df.columns:
        df["has_event"] = (df["event_name_1"].notna() & (df["event_name_1"] != "")).astype(int)
    else:
        df["has_event"] = 0

    snap_cols = [c for c in df.columns if c.startswith("snap_")]
    df["is_snap"] = df[snap_cols].max(axis=1).fillna(0).astype(int) if snap_cols else 0
    return df


# ── Price Features ────────────────────────────────────────────────────────────

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "sell_price" not in df.columns:
        return df
    df["price_lag_1"]  = df["sell_price"].shift(1)
    df["price_change"] = df["sell_price"].pct_change().fillna(0)
    mu = df["sell_price"].mean()
    sg = df["sell_price"].std() + 1e-8
    df["price_norm"]   = (df["sell_price"] - mu) / sg
    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, target: str = "sales") -> pd.DataFrame:
    """Apply all feature engineering steps."""
    df = add_lags(df, target)
    df = add_rolling_stats(df, target)
    df = add_ewm(df, target)
    df = add_calendar_features(df)
    df = add_event_flags(df)
    df = add_price_features(df)
    return df


def get_feature_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """Return numeric feature columns (excludes meta & target)."""
    exclude = exclude or []
    skip = {"date","d","id","item_id","dept_id","cat_id","store_id",
            "state_id","wm_yr_wk","weekday","event_name_1","event_type_1","sales"}
    skip.update(exclude)
    return [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Min-max scale features; fit on train, apply to both."""
    stats = {}
    train_df, test_df = train_df.copy(), test_df.copy()
    for col in feature_cols:
        mn  = train_df[col].min()
        mx  = train_df[col].max()
        rng = (mx - mn) if mx != mn else 1.0
        train_df[col] = (train_df[col] - mn) / rng
        test_df[col]  = (test_df[col]  - mn) / rng
        stats[col]    = {"min": mn, "max": mx}
    return train_df, test_df, stats
