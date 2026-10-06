"""
pages/2_Explore_Data.py
Interactive EDA: sales history, CUSUM change-point detection, regime timeline.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.preprocessor import filter_series, apply_cutoff
from src.detection.cusum import detect_change_points, cusum_to_dataframe
from src.detection.regime_classifier import RegimeClassifier, REGIME_COLORS, REGIME_ICONS
from src.detection.change_point_viz import build_marker_data, markers_to_dataframe

st.set_page_config(page_title="Explore Data", page_icon="🔍", layout="wide")
st.markdown("#  Explore Data")
st.divider()

# ── Guard ─────────────────────────────────────────────────────────────────────
if "long_df" not in st.session_state:
    st.warning(" No data loaded yet. Please go to **1 · Upload Data** first.")
    st.stop()

long_df = st.session_state["long_df"]

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Series Selection")
    stores = sorted(long_df["store_id"].unique())
    store  = st.selectbox("Store", stores)

    items  = sorted(long_df[long_df["store_id"] == store]["item_id"].unique())
    item   = st.selectbox("Item / Product", items)

    st.divider()
    st.markdown("###  CUSUM Parameters")
    cusum_thresh  = st.slider("Threshold (h · σ)", 2.0, 12.0, 5.0, 0.5,
                               help="Higher = fewer, more confident detections")
    cusum_drift   = st.slider("Drift (k · σ)", 0.1, 2.0, 0.5, 0.1)
    min_seg       = st.slider("Min segment (days)", 7, 60, 14)

    st.divider()
    st.markdown("###  Cutoff Date")
    ts = filter_series(long_df, store, item)
    min_date = ts["date"].min().date()
    max_date = ts["date"].max().date()
    cutoff   = st.date_input("Training cutoff", value=max_date, min_value=min_date, max_value=max_date)
    st.session_state["cutoff_date"] = str(cutoff)

# ── Load series ───────────────────────────────────────────────────────────────
ts = filter_series(long_df, store, item)
train_ts, test_ts = apply_cutoff(ts, str(cutoff))
sales = train_ts.set_index("date")["sales"]

st.session_state["store_id"] = store
st.session_state["item_id"]  = item

# ── CUSUM detection ───────────────────────────────────────────────────────────
cusum_result = detect_change_points(sales, cusum_thresh, cusum_drift, min_seg)
cusum_df     = cusum_to_dataframe(cusum_result, sales)

# ── Regime classification ─────────────────────────────────────────────────────
rc            = RegimeClassifier()
regime_result = rc.classify(sales, cusum_result)
current       = regime_result.current_regime

# ── Top KPI row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(" Total Sales", f"{int(sales.sum()):,}")
c2.metric(" Days", f"{len(sales):,}")
c3.metric(" Daily Avg", f"{sales.mean():.1f}")
c4.metric(" Change Points", f"{len(cusum_result.change_points)}")

regime_color = {"Stable": "normal", "Disrupted": "inverse", "Recovery": "off"}
c5.metric("🚦 Current Regime", f"{REGIME_ICONS[current]} {current.value}")

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([" Sales History", " CUSUM Statistics", " Regime Timeline", "📋 Change Points"])

# ─── Tab 1: Sales history + change-point markers ──────────────────────────────
with tab1:
    fig = go.Figure()

    # Regime background shading
    rdf = regime_result.regime_df
    for r_name, r_color in REGIME_COLORS.items():
        mask = rdf["regime"] == r_name
        if mask.any():
            fig.add_trace(go.Scatter(
                x=rdf["date"][mask], y=rdf["sales"][mask],
                fill="tozeroy", fillcolor="rgba({},{},{},0.13)".format(int(r_color[1:3],16),int(r_color[3:5],16),int(r_color[5:7],16)),
                line=dict(width=0), showlegend=True,
                name=f"{REGIME_ICONS[r_name]} {r_name.value}",
                hoverinfo="skip",
            ))

    # Sales line
    fig.add_trace(go.Scatter(
        x=sales.index, y=sales.values,
        line=dict(color="#1e40af", width=1.8),
        name="Daily Sales",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Sales: %{y:.0f}<extra></extra>",
    ))

    # 7-day rolling mean
    roll7 = sales.rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=roll7.index, y=roll7.values,
        line=dict(color="#f59e0b", width=2, dash="dot"),
        name="7-day rolling avg",
    ))

    # Change-point vertical lines (use add_shape to avoid Plotly annotation bug)
    y_max = float(sales.max()) if len(sales) > 0 else 1.0
    for cp_date, mag in zip(cusum_result.change_dates, cusum_result.magnitudes):
        x_str = str(cp_date)[:10]
        fig.add_shape(type="line", x0=x_str, x1=x_str, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="#dc2626", width=2, dash="dash"))
        fig.add_annotation(x=x_str, y=y_max, text=f"Δ{mag:.1f}",
                           showarrow=False, yanchor="bottom",
                           font=dict(color="#dc2626", size=10),
                           bgcolor="rgba(255,255,255,0.7)")

    # Cutoff line
    x_cutoff = str(cutoff)
    fig.add_shape(type="line", x0=x_cutoff, x1=x_cutoff, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color="#7c3aed", width=2, dash="dashdot"))
    fig.add_annotation(x=x_cutoff, y=y_max, text="Cutoff",
                       showarrow=False, yanchor="bottom",
                       font=dict(color="#7c3aed", size=11),
                       bgcolor="rgba(255,255,255,0.7)")

    fig.update_layout(
        title=f"{store} · {item} — Sales History",
        xaxis_title="Date", yaxis_title="Units Sold",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=440, template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── Tab 2: CUSUM statistics ──────────────────────────────────────────────────
with tab2:
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("S⁺ (Upward CUSUM)", "S⁻ (Downward CUSUM)"),
                         vertical_spacing=0.08)

    fig2.add_trace(go.Scatter(
        x=sales.index, y=cusum_df["cusum_pos"],
        fill="tozeroy", fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="#3b82f6", width=1.5), name="S⁺"),
        row=1, col=1)
    fig2.add_hline(y=cusum_result.threshold, line_dash="dash", line_color="#dc2626",
                   annotation_text=f"Threshold = {cusum_result.threshold:.1f}", row=1, col=1)

    fig2.add_trace(go.Scatter(
        x=sales.index, y=cusum_df["cusum_neg"],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="#ef4444", width=1.5), name="S⁻"),
        row=2, col=1)
    fig2.add_hline(y=cusum_result.threshold, line_dash="dash", line_color="#dc2626", row=2, col=1)

    for cp_date in cusum_result.change_dates:
        for r in [1, 2]:
            fig2.add_vline(x=str(cp_date)[:10], line_width=1.5,
                           line_dash="dash", line_color="#f59e0b", row=r, col=1)

    fig2.update_layout(height=420, template="plotly_white", showlegend=True,
                       title="Two-sided CUSUM Control Statistics")
    st.plotly_chart(fig2, use_container_width=True)

# ─── Tab 3: Regime timeline ───────────────────────────────────────────────────
with tab3:
    rdf2 = regime_result.regime_df.copy()
    color_map = {
        "Stable":    "#22c55e",
        "Disrupted": "#ef4444",
        "Recovery":  "#f59e0b",
    }
    rdf2["color"] = rdf2["regime"].apply(lambda r: color_map.get(r.value if hasattr(r,"value") else r, "#94a3b8"))

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=rdf2["date"], y=rdf2["sales"],
        marker_color=rdf2["color"],
        name="Sales (colored by regime)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Sales: %{y:.0f}<extra></extra>",
    ))

    # Legend traces
    for label, color in color_map.items():
        fig3.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                  marker=dict(color=color, size=12, symbol="square"),
                                  name=label))

    fig3.update_layout(
        title="Regime Timeline (bar color = regime)",
        xaxis_title="Date", yaxis_title="Units Sold",
        height=380, template="plotly_white", showlegend=True,
        bargap=0,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Regime summary
    summary = regime_result.summary
    sc1, sc2, sc3 = st.columns(3)
    for col_w, r_key, label, bg in [
        (sc1, "Stable",    " Stable Days",    "#dcfce7"),
        (sc2, "Disrupted", " Disrupted Days", "#fee2e2"),
        (sc3, "Recovery",  " Recovery Days",  "#fef9c3"),
    ]:
        cnt = summary["counts"].get(r_key, 0)
        pct = cnt / len(rdf2) * 100 if len(rdf2) > 0 else 0
        col_w.markdown(
            f"<div style='background:{bg};border-radius:10px;padding:12px;text-align:center'>"
            f"<b style='font-size:1.3rem'>{cnt:,}</b><br>{label}<br>"
            f"<small>{pct:.1f}% of history</small></div>",
            unsafe_allow_html=True,
        )

# ─── Tab 4: Change-point table ────────────────────────────────────────────────
with tab4:
    markers = build_marker_data(cusum_result, train_ts)
    mdf     = markers_to_dataframe(markers)
    if len(mdf) == 0:
        st.info("No change points detected with current parameters. Try reducing the threshold.")
    else:
        st.markdown(f"**{len(mdf)} change point(s) detected**")
        st.dataframe(mdf, use_container_width=True, hide_index=True)
        st.caption("Magnitude = absolute mean-shift in sales units between 14-day windows around each break.")