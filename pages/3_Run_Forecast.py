"""
pages/3_Run_Forecast.py
Train all four models + ensemble, display interactive forecast chart with
confidence bands, regime badge, change-point markers, and CSV download.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data.preprocessor import filter_series, apply_cutoff
from src.detection.cusum import detect_change_points
from src.detection.regime_classifier import RegimeClassifier, REGIME_COLORS, REGIME_ICONS
from src.models.arima_model import get_arima_forecaster
from src.models.prophet_model import get_prophet_forecaster
from src.models.lstm_model import get_lstm_forecaster
from src.models.transformer_model import get_transformer_forecaster
from src.models.ensemble_model import EnsembleForecaster
from src.evaluation.metrics import build_metrics_table
from src.export.csv_exporter import build_export_df, to_csv_bytes

st.set_page_config(page_title="Run Forecast", page_icon="🚀", layout="wide")
st.markdown("#  Run Forecast")
st.divider()

# ── Guard ─────────────────────────────────────────────────────────────────────
if "long_df" not in st.session_state:
    st.warning(" No data loaded. Please go to **1 · Upload Data** first.")
    st.stop()

long_df = st.session_state["long_df"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Forecast Settings")
    stores  = sorted(long_df["store_id"].unique())
    default_store = st.session_state.get("store_id", stores[0])
    store   = st.selectbox("Store", stores, index=stores.index(default_store) if default_store in stores else 0)

    items   = sorted(long_df[long_df["store_id"] == store]["item_id"].unique())
    default_item = st.session_state.get("item_id", items[0])
    item    = st.selectbox("Item / Product", items, index=items.index(default_item) if default_item in items else 0)

    horizon = st.slider("Forecast horizon (days)", 7, 28, 28)

    ts_all    = filter_series(long_df, store, item)
    min_date  = ts_all["date"].min().date()
    max_date  = ts_all["date"].max().date()
    default_c = st.session_state.get("cutoff_date", str(max_date))
    cutoff    = st.date_input("Training cutoff", value=pd.Timestamp(default_c).date(),
                               min_value=min_date, max_value=max_date)

    st.divider()
    st.markdown("###  Model Selection")
    run_arima  = st.checkbox("ARIMA",       value=True)
    run_prophet= st.checkbox("Prophet",     value=True)
    run_lstm   = st.checkbox("LSTM",        value=True)
    run_trans  = st.checkbox("Transformer", value=True)

    st.divider()
    run_btn = st.button(" Run Forecast", type="primary", use_container_width=True)

# ── Pre-process series ────────────────────────────────────────────────────────
ts_all    = filter_series(long_df, store, item)
train_ts, test_ts = apply_cutoff(ts_all, str(cutoff))
sales_series = train_ts.set_index("date")["sales"]

# Detect regime
cusum_result  = detect_change_points(sales_series, threshold=5.0, drift=0.5, min_segment=14)
rc            = RegimeClassifier()
regime_result = rc.classify(sales_series, cusum_result)
current_regime= regime_result.current_regime

# Show regime badge
r_class = current_regime.value.lower()
st.markdown(
    f"**Current Regime:** "
    f"<span class='regime-{r_class}'>{REGIME_ICONS[current_regime]} {current_regime.value}</span>"
    f"  ·  **{len(cusum_result.change_points)}** change point(s) detected  "
    f"  ·  Training on **{len(train_ts):,}** days up to **{cutoff}**",
    unsafe_allow_html=True,
)
st.markdown("""
<style>
.regime-stable   {background:#dcfce7;color:#15803d;padding:3px 12px;border-radius:20px;font-weight:700;}
.regime-disrupted{background:#fee2e2;color:#dc2626;padding:3px 12px;border-radius:20px;font-weight:700;}
.regime-recovery {background:#fef9c3;color:#b45309;padding:3px 12px;border-radius:20px;font-weight:700;}
</style>""", unsafe_allow_html=True)

# ── Run models ────────────────────────────────────────────────────────────────
if run_btn:
    forecasts: dict = {}
    progress = st.progress(0, text="Starting…")
    status   = st.empty()
    steps    = sum([run_arima, run_prophet, run_lstm, run_trans]) + 1
    done     = [0]

    def tick(name):
        done[0] += 1
        progress.progress(done[0] / steps, text=f"Training {name}…")

    if run_arima:
        status.info("Training ARIMA…")
        try:
            m = get_arima_forecaster(horizon=horizon)
            m.fit(train_ts)
            forecasts["ARIMA"] = m.predict(horizon)
        except Exception as e:
            st.warning(f"ARIMA failed: {e}")
        tick("ARIMA")

    if run_prophet:
        status.info("Training Prophet…")
        try:
            m = get_prophet_forecaster(horizon=horizon)
            m.fit(train_ts)
            forecasts["Prophet"] = m.predict(horizon)
        except Exception as e:
            st.warning(f"Prophet failed: {e}")
        tick("Prophet")

    if run_lstm:
        status.info("Training LSTM…")
        try:
            m = get_lstm_forecaster(horizon=horizon)
            m.fit(train_ts)
            forecasts["LSTM"] = m.predict(horizon)
        except Exception as e:
            st.warning(f"LSTM failed: {e}")
        tick("LSTM")

    if run_trans:
        status.info("Training Transformer…")
        try:
            m = get_transformer_forecaster(horizon=horizon)
            m.fit(train_ts)
            forecasts["Transformer"] = m.predict(horizon)
        except Exception as e:
            st.warning(f"Transformer failed: {e}")
        tick("Transformer")

    # Ensemble
    if len(forecasts) >= 2:
        status.info("Building Ensemble…")
        try:
            ens = EnsembleForecaster(horizon=horizon, regime=current_regime)
            ens.set_forecasters([
                get_arima_forecaster(horizon=horizon),
                get_prophet_forecaster(horizon=horizon),
                get_lstm_forecaster(horizon=horizon),
                get_transformer_forecaster(horizon=horizon),
            ])
            ens.fit(train_ts)
            forecasts["Ensemble"] = ens.predict(horizon)
        except Exception as e:
            st.warning(f"Ensemble failed: {e}")
    tick("Ensemble")

    progress.progress(1.0, text="Done!")
    status.empty()

    st.session_state["forecasts"]      = forecasts
    st.session_state["train_ts"]       = train_ts
    st.session_state["test_ts"]        = test_ts
    st.session_state["cusum_result"]   = cusum_result
    st.session_state["current_regime"] = current_regime
    st.session_state["horizon"]        = horizon

# ── Display ───────────────────────────────────────────────────────────────────
if "forecasts" in st.session_state and st.session_state["forecasts"]:
    forecasts    = st.session_state["forecasts"]
    train_ts     = st.session_state["train_ts"]
    test_ts      = st.session_state["test_ts"]
    cusum_result = st.session_state["cusum_result"]
    horizon      = st.session_state["horizon"]

    MODEL_COLORS = {
        "ARIMA":       "#3b82f6",
        "Prophet":     "#8b5cf6",
        "LSTM":        "#10b981",
        "Transformer": "#f59e0b",
        "Ensemble":    "#dc2626",
    }

    fig = go.Figure()

    # Historical sales
    hist = train_ts.set_index("date")["sales"]
    display_hist = hist.tail(120)  # last 120 days for clarity
    fig.add_trace(go.Scatter(
        x=display_hist.index, y=display_hist.values,
        line=dict(color="#1e293b", width=1.8),
        name="Historical Sales",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual: %{y:.0f}<extra></extra>",
    ))

    # Actual test data if available
    if len(test_ts) > 0:
        fig.add_trace(go.Scatter(
            x=test_ts["date"], y=test_ts["sales"],
            line=dict(color="#64748b", width=1.5, dash="dot"),
            name="Actual (post-cutoff)",
        ))

    # Change-point markers
    y_max3 = float(hist.max()) if len(hist) > 0 else 1.0
    for cp_date, mag in zip(cusum_result.change_dates, cusum_result.magnitudes):
        x_str = str(cp_date)[:10]
        fig.add_shape(type="line", x0=x_str, x1=x_str, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="#dc2626", width=1.5, dash="dash"))
        fig.add_annotation(x=x_str, y=y_max3, text=f"Δ{mag:.1f}",
                           showarrow=False, yanchor="bottom",
                           font=dict(color="#dc2626", size=10),
                           bgcolor="rgba(255,255,255,0.7)")

    # Cutoff line
    x_cutoff3 = str(cutoff)
    fig.add_shape(type="line", x0=x_cutoff3, x1=x_cutoff3, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color="#7c3aed", width=2, dash="dashdot"))
    fig.add_annotation(x=x_cutoff3, y=y_max3, text="Cutoff →",
                       showarrow=False, yanchor="bottom",
                       font=dict(color="#7c3aed", size=11),
                       bgcolor="rgba(255,255,255,0.7)")

    # Forecasts + confidence bands
    for model_name, fc_df in forecasts.items():
        color = MODEL_COLORS.get(model_name, "#94a3b8")
        width = 3 if model_name == "Ensemble" else 1.8
        dash  = "solid" if model_name == "Ensemble" else "solid"

        # CI band
        fig.add_trace(go.Scatter(
            x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
            y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba({},{},{},0.13)".format(int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)),
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=fc_df["date"], y=fc_df["yhat"],
            line=dict(color=color, width=width, dash=dash),
            name=model_name,
            hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{model_name}: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"28-Day Forecast — {store} · {item}",
        xaxis_title="Date", yaxis_title="Predicted Units Sold",
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Forecast summary metrics ───────────────────────────────────────────────
    st.divider()
    st.markdown("###  Forecast Summary (28-day totals)")
    if forecasts:
        cols = st.columns(min(len(forecasts), 5))
        for i, (mname, fc_df) in enumerate(forecasts.items()):
            with cols[i % len(cols)]:
                total = fc_df["yhat"].sum()
                daily = fc_df["yhat"].mean()
                st.metric(mname, f"{total:.0f} units", f"avg {daily:.1f}/day")

    # ── Metrics table (if test data available) ────────────────────────────────
    if len(test_ts) > 0:
        st.divider()
        st.markdown("###  Model Performance on Post-Cutoff Data")
        actual_series = test_ts.set_index("date")["sales"]
        metrics_df = build_metrics_table(forecasts, actual_series)
        if not metrics_df.empty:
            st.dataframe(
                metrics_df.style.highlight_min(subset=["MAE","RMSE","MAPE"], color="#dcfce7")
                                .highlight_min(subset=["R²"],              color="#dbeafe"),
                use_container_width=True,
            )

    # ── Download CSV ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("###  Download Forecast CSV")
    export_df   = build_export_df(forecasts, store, item)
    csv_bytes   = to_csv_bytes(export_df)
    st.dataframe(export_df.head(10), use_container_width=True)
    st.download_button(
        label="⬇ Download 28-day Forecast CSV",
        data=csv_bytes,
        file_name=f"forecast_{store}_{item}_{cutoff}.csv",
        mime="text/csv",
    )

else:
    st.info(" Configure settings in the sidebar and press **▶ Run Forecast** to begin.")