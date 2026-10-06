"""
src/data/loader.py
------------------
CSV ingestion, validation, caching and synthetic M5 data generation.
"""

import hashlib
import io
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# ── Dev-mode subset defaults ──────────────────────────────────────────────────
SAMPLE_STORES = ["CA_1", "TX_1"]
SAMPLE_DEPTS  = ["FOODS_3", "HOUSEHOLD_1"]
MAX_ITEMS     = 50
CACHE_DIR     = Path("data/processed")
RAW_DIR       = Path("data/raw")

# ── Required column sets ──────────────────────────────────────────────────────
REQUIRED_SALES    = {"item_id", "dept_id", "cat_id", "store_id", "state_id"}
REQUIRED_CALENDAR = {"date", "wm_yr_wk", "weekday", "d"}
REQUIRED_PRICES   = {"store_id", "item_id", "wm_yr_wk", "sell_price"}


# ═════════════════════════════════════════════════════════════════════════════
# Validators
# ═════════════════════════════════════════════════════════════════════════════

def validate_sales(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = REQUIRED_SALES - set(df.columns)
    if missing:
        return False, f"Missing columns: {missing}"
    day_cols = [c for c in df.columns if c.startswith("d_")]
    if len(day_cols) < 100:
        return False, "Expected 1 000+ day columns (d_1, d_2, …)."
    return True, "OK"


def validate_calendar(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = REQUIRED_CALENDAR - set(df.columns)
    if missing:
        return False, f"Missing columns: {missing}"
    return True, "OK"


def validate_prices(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = REQUIRED_PRICES - set(df.columns)
    if missing:
        return False, f"Missing columns: {missing}"
    return True, "OK"


# ═════════════════════════════════════════════════════════════════════════════
# Loaders
# ═════════════════════════════════════════════════════════════════════════════

def load_sales(source) -> pd.DataFrame:
    """Load sales CSV — accepts file path or Streamlit UploadedFile."""
    return pd.read_csv(source)


def load_calendar(source) -> pd.DataFrame:
    df = pd.read_csv(source)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_prices(source) -> pd.DataFrame:
    return pd.read_csv(source)


# ═════════════════════════════════════════════════════════════════════════════
# Subset helper
# ═════════════════════════════════════════════════════════════════════════════

def subset_sales(
    df: pd.DataFrame,
    stores: Optional[list] = None,
    depts: Optional[list] = None,
    max_items: Optional[int] = None,
) -> pd.DataFrame:
    """Reduce sales matrix to a manageable dev subset."""
    stores    = stores    or SAMPLE_STORES
    depts     = depts     or SAMPLE_DEPTS
    max_items = max_items or MAX_ITEMS

    mask = df["store_id"].isin(stores) & df["dept_id"].isin(depts)
    df   = df[mask].copy()

    if max_items and df["item_id"].nunique() > max_items:
        chosen = (
            df["item_id"]
            .drop_duplicates()
            .sample(min(max_items, df["item_id"].nunique()), random_state=42)
        )
        df = df[df["item_id"].isin(chosen)]
    return df.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# Cache helpers
# ═════════════════════════════════════════════════════════════════════════════

def cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.parquet"


def save_cache(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(cache_path(name), index=False)


def load_cache(name: str) -> Optional[pd.DataFrame]:
    p = cache_path(name)
    return pd.read_parquet(p) if p.exists() else None


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic M5 data generator  (zero Kaggle dependency)
# ═════════════════════════════════════════════════════════════════════════════

def generate_sample_data(
    n_stores: int = 2,
    n_items: int = 5,
    n_days: int = 365 * 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic synthetic M5-style data with:
    - Linear trend + weekly + annual seasonality
    - Random noise
    - One injected demand disruption (~day 600)
    """
    rng        = np.random.default_rng(42)
    start_date = pd.Timestamp("2013-01-01")
    dates      = pd.date_range(start_date, periods=n_days, freq="D")

    stores = [f"CA_{i+1}" for i in range(n_stores)]
    items  = [f"FOODS_3_{str(i+1).zfill(3)}" for i in range(n_items)]
    depts  = ["FOODS_3"] * n_items
    cats   = ["FOODS"]   * n_items
    states = ["CA"]      * n_stores

    # ── Calendar ──────────────────────────────────────────────────────────────
    cal_rows = []
    wk = 11101
    for i, d in enumerate(dates):
        if i % 7 == 0 and i > 0:
            wk += 1
        event = ""
        if   d.month == 11 and d.day == 25: event = "Thanksgiving"
        elif d.month == 12 and d.day == 25: event = "Christmas"
        elif d.month == 1  and d.day == 1:  event = "NewYear"
        elif d.month == 7  and d.day == 4:  event = "IndependenceDay"
        cal_rows.append({
            "d": f"d_{i+1}", "date": d, "wm_yr_wk": wk,
            "weekday": d.strftime("%A"), "wday": d.dayofweek + 1,
            "month": d.month, "year": d.year,
            "event_name_1": event, "event_type_1": "National" if event else "",
            "snap_CA": int(rng.random() > 0.8),
            "snap_TX": int(rng.random() > 0.8),
            "snap_WI": int(rng.random() > 0.8),
        })
    calendar_df = pd.DataFrame(cal_rows)

    # ── Sales (wide) ──────────────────────────────────────────────────────────
    day_cols  = [f"d_{i+1}" for i in range(n_days)]
    sales_rows = []
    t = np.arange(n_days, dtype=float)

    for s_idx, store in enumerate(stores):
        for i_idx, item in enumerate(items):
            trend   = 5 + 0.003 * t
            weekly  = 3 * np.sin(2 * np.pi * t / 7 + i_idx)
            annual  = 4 * np.sin(2 * np.pi * t / 365.25)
            # Disruption: sudden drop then recovery
            shock   = np.zeros(n_days)
            shock[600:640] = -4.0
            shock[640:700] =  2.5
            noise   = rng.normal(0, 1.5, n_days)
            sales   = np.clip(trend + weekly + annual + shock + noise, 0, None).astype(int)

            row = {
                "id": f"{item}_{store}_validation",
                "item_id": item, "dept_id": depts[i_idx],
                "cat_id": cats[i_idx], "store_id": store,
                "state_id": states[s_idx % len(states)],
            }
            for j, dc in enumerate(day_cols):
                row[dc] = int(sales[j])
            sales_rows.append(row)

    sales_df = pd.DataFrame(sales_rows)

    # ── Prices ────────────────────────────────────────────────────────────────
    wm_yr_wks = calendar_df["wm_yr_wk"].unique()
    price_rows = []
    for store in stores:
        for item in items:
            base = rng.uniform(1.5, 8.0)
            for wk in wm_yr_wks:
                price_rows.append({
                    "store_id": store, "item_id": item,
                    "wm_yr_wk": wk,
                    "sell_price": round(float(base * rng.uniform(0.9, 1.1)), 2),
                })
    prices_df = pd.DataFrame(price_rows)

    return sales_df, calendar_df, prices_df


def save_sample_csvs(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    out_dir: str = "data/sample",
) -> None:
    """Persist generated sample data as CSV files."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Save only id cols + first 30 day cols to keep file small
    id_cols  = [c for c in sales_df.columns if not c.startswith("d_")]
    day_cols = [c for c in sales_df.columns if c.startswith("d_")][:30]
    sales_df[id_cols + day_cols].to_csv(f"{out_dir}/sample_sales.csv",    index=False)
    calendar_df.to_csv(f"{out_dir}/sample_calendar.csv", index=False)
    prices_df.to_csv(  f"{out_dir}/sample_prices.csv",   index=False)
