"""
src/detection/change_point_viz.py
----------------------------------
Builds timeline marker data for the change-point overlay chart.
"""

from typing import List

import pandas as pd

from src.detection.cusum import CUSUMResult


def build_marker_data(
    result: CUSUMResult,
    history_df: pd.DataFrame,
) -> List[dict]:
    """
    Return a list of marker dicts for Plotly annotations.

    Each dict contains:
        date, magnitude, direction, label
    """
    markers = []
    sales   = history_df.set_index("date")["sales"]

    for cp_date, mag in zip(result.change_dates, result.magnitudes):
        ts = pd.Timestamp(cp_date)

        # Determine direction
        try:
            before = float(sales.loc[:ts].iloc[-7:].mean())
            after  = float(sales.loc[ts:].iloc[:7].mean())
            direction = "↑ Upward" if after > before else "↓ Downward"
        except Exception:
            direction = "Unknown"

        markers.append({
            "date"     : ts,
            "magnitude": mag,
            "direction": direction,
            "label"    : f"Δ{mag:.1f} {direction[:1]}",
        })

    return markers


def markers_to_dataframe(markers: List[dict]) -> pd.DataFrame:
    """Convert marker list to a DataFrame for display in Streamlit."""
    if not markers:
        return pd.DataFrame(columns=["Date", "Magnitude", "Direction"])
    return pd.DataFrame([{
        "Date"     : m["date"].strftime("%Y-%m-%d"),
        "Magnitude": m["magnitude"],
        "Direction": m["direction"],
    } for m in markers])
