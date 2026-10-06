"""
src/visualization/forecast_chart.py
Reusable Plotly chart builders shared across pages.
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.detection.cusum import CUSUMResult
from src.detection.regime_classifier import RegimeResult, REGIME_COLORS, REGIME_ICONS

MODEL_COLORS = {
    "ARIMA":       "#3b82f6",
    "Prophet":     "#8b5cf6",
    "LSTM":        "#10b981",
    "Transformer": "#f59e0b",
    "Ensemble":    "#dc2626",
}


def build_forecast_chart(
    history:       pd.Series,
    forecasts:     Dict[str, pd.DataFrame],
    cusum_result:  Optional[CUSUMResult]  = None,
    regime_result: Optional[RegimeResult] = None,
    cutoff_date:   Optional[str]          = None,
    actual_test:   Optional[pd.Series]    = None,
    history_tail:  int = 120,
    title:         str = "Demand Forecast",
    height:        int = 500,
) -> go.Figure:
    """
    Full forecast chart:
      - Historical sales (last `history_tail` days)
      - Optional regime-shaded background
      - Change-point vertical markers
      - Per-model forecast lines + 90% CI bands
      - Optional actual test data
    """
    fig = go.Figure()

    # ── Regime background shading ─────────────────────────────────────────────
    if regime_result is not None:
        rdf = regime_result.regime_df.copy()
        for r_name, r_color in REGIME_COLORS.items():
            mask = rdf["regime"] == r_name
            if mask.any():
                dates = rdf["date"][mask]
                sales = rdf["sales"][mask]
                fig.add_trace(go.Scatter(
                    x=dates, y=sales,
                    fill="tozeroy", fillcolor=r_color + "22",
                    line=dict(width=0), showlegend=True,
                    name=f"{REGIME_ICONS[r_name]} {r_name.value}",
                    hoverinfo="skip",
                ))

    # ── Historical sales ──────────────────────────────────────────────────────
    tail = history.tail(history_tail)
    fig.add_trace(go.Scatter(
        x=tail.index, y=tail.values,
        line=dict(color="#1e293b", width=1.8),
        name="Historical Sales",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:.0f}<extra></extra>",
    ))

    # ── Actual test data ──────────────────────────────────────────────────────
    if actual_test is not None and len(actual_test) > 0:
        fig.add_trace(go.Scatter(
            x=actual_test.index, y=actual_test.values,
            line=dict(color="#64748b", width=1.5, dash="dot"),
            name="Actual (post-cutoff)",
        ))

    # ── Change-point markers ──────────────────────────────────────────────────
    if cusum_result is not None:
        for cp_date, mag in zip(cusum_result.change_dates, cusum_result.magnitudes):
            fig.add_vline(
                x=str(cp_date)[:10], line_width=1.8, line_dash="dash", line_color="#dc2626",
                annotation_text=f"Δ{mag:.1f}", annotation_position="top",
                annotation_font=dict(color="#dc2626", size=10),
            )

    # ── Cutoff marker ─────────────────────────────────────────────────────────
    if cutoff_date:
        fig.add_vline(
            x=str(cutoff_date)[:10], line_width=2, line_color="#7c3aed", line_dash="dashdot",
            annotation_text="Cutoff →", annotation_position="top right",
            annotation_font=dict(color="#7c3aed", size=11),
        )

    # ── Forecasts + confidence bands ──────────────────────────────────────────
    for model_name, fc_df in forecasts.items():
        color  = MODEL_COLORS.get(model_name, "#94a3b8")
        lwidth = 3 if model_name == "Ensemble" else 1.8

        # CI band
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_df["date"], fc_df["date"].iloc[::-1]]),
            y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"].iloc[::-1]]),
            fill="toself", fillcolor=color + "22",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        # Forecast line
        fig.add_trace(go.Scatter(
            x=fc_df["date"], y=fc_df["yhat"],
            line=dict(color=color, width=lwidth),
            name=model_name,
            hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{model_name}: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Units Sold",
        height=height,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    return fig


def build_cusum_chart(
    sales:        pd.Series,
    cusum_pos:    np.ndarray,
    cusum_neg:    np.ndarray,
    threshold:    float,
    change_dates: list,
    height:       int = 420,
) -> go.Figure:
    """Two-panel CUSUM statistics chart."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("S⁺ (Upward CUSUM)", "S⁻ (Downward CUSUM)"),
                        vertical_spacing=0.08)

    dates = sales.index

    fig.add_trace(go.Scatter(x=dates, y=cusum_pos, fill="tozeroy",
                             fillcolor="rgba(59,130,246,0.15)",
                             line=dict(color="#3b82f6", width=1.5), name="S⁺"),
                  row=1, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="#dc2626",
                  annotation_text=f"h={threshold:.1f}", row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=cusum_neg, fill="tozeroy",
                             fillcolor="rgba(239,68,68,0.15)",
                             line=dict(color="#ef4444", width=1.5), name="S⁻"),
                  row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="#dc2626", row=2, col=1)

    for cp in change_dates:
        for r in (1, 2):
            fig.add_vline(x=str(cp)[:10], line_width=1.5, line_dash="dash",
                          line_color="#f59e0b", row=r, col=1)

    fig.update_layout(height=height, template="plotly_white",
                      title="Two-sided CUSUM Control Statistics", showlegend=True)
    return fig
