"""
pages/4_Model_Comparison.py
Side-by-side model comparison: metrics table, radar chart, residual analysis,
and walk-forward backtesting results.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.preprocessor import filter_series, apply_cutoff
from src.detection.cusum import detect_change_points
from src.detection.regime_classifier import RegimeClassifier
from src.models.arima_model import get_arima_forecaster
from src.models.prophet_model import get_prophet_forecaster
from src.models.lstm_model import get_lstm_forecaster
from src.models.transformer_model import get_transformer_forecaster
from src.evaluation.metrics import build_metrics_table, compute_all_metrics
from src.evaluation.backtester import walk_forward_validate

st.set_page_config(page_title="Model Comparison", page_icon="", layout="wide")
st.markdown("# Model Comparison")
st.divider()

# ── Guard ─────────────────────────────────────────────────────────────────────
if "long_df" not in st.session_state:
    st.warning(" No data loaded. Please go to **1 · Upload Data** first.")
    st.stop()

long_df = st.session_state["long_df"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Settings")
    stores  = sorted(long_df["store_id"].unique())
    default_store = st.session_state.get("store_id", stores[0])
    store   = st.selectbox("Store", stores, index=stores.index(default_store) if default_store in stores else 0)

    items   = sorted(long_df[long_df["store_id"] == store]["item_id"].unique())
    default_item = st.session_state.get("item_id", items[0])
    item    = st.selectbox("Item / Product", items, index=items.index(default_item) if default_item in items else 0)

    ts_all   = filter_series(long_df, store, item)
    min_date = ts_all["date"].min().date()
    max_date = ts_all["date"].max().date()
    default_c = st.session_state.get("cutoff_date", str(max_date))
    cutoff   = st.date_input("Training cutoff", value=pd.Timestamp(default_c).date(),
                              min_value=min_date, max_value=max_date)
    horizon  = st.slider("Horizon", 7, 28, st.session_state.get("horizon", 28))

    st.divider()
    run_backtest = st.checkbox("Run walk-forward backtest", value=False,
                                help="Runs 3-fold expanding-window CV. Slower but more rigorous.")
    n_folds = st.slider("Backtest folds", 2, 5, 3) if run_backtest else 3
    run_btn = st.button(" Compare Models", type="primary", use_container_width=True)

# ── Pre-process ────────────────────────────────────────────────────────────────
ts_all = filter_series(long_df, store, item)
train_ts, test_ts = apply_cutoff(ts_all, str(cutoff))

MODEL_COLORS = {
    "ARIMA":       "#3b82f6",
    "Prophet":     "#8b5cf6",
    "LSTM":        "#10b981",
    "Transformer": "#f59e0b",
    "Ensemble":    "#dc2626",
}

if run_btn or "comparison_results" in st.session_state:

    if run_btn:
        # Train all models fresh
        cusum_result  = detect_change_points(train_ts.set_index("date")["sales"])
        regime_result = RegimeClassifier().classify(train_ts.set_index("date")["sales"], cusum_result)
        current       = regime_result.current_regime

        forecasts = {}
        bt_results = {}

        progress = st.progress(0, "Training models…")
        steps    = 4
        done     = [0]

        def step(name):
            done[0] += 1
            progress.progress(done[0] / steps, text=f"Finished {name}")

        for mname, factory in [
            ("ARIMA",       get_arima_forecaster),
            ("Prophet",     get_prophet_forecaster),
            ("LSTM",        get_lstm_forecaster),
            ("Transformer", get_transformer_forecaster),
        ]:
            try:
                m = factory(horizon=horizon)
                m.fit(train_ts)
                forecasts[mname] = m.predict(horizon)

                if run_backtest:
                    bt = walk_forward_validate(
                        train_ts, type(m), model_kwargs={"horizon": horizon},
                        n_folds=n_folds, horizon=horizon,
                    )
                    bt_results[mname] = bt
            except Exception as e:
                st.warning(f"{mname} failed: {e}")
            step(mname)

        progress.empty()
        st.session_state["comparison_results"] = {
            "forecasts":    forecasts,
            "bt_results":   bt_results,
            "train_ts":     train_ts,
            "test_ts":      test_ts,
            "current":      current,
            "run_backtest": run_backtest,
        }

    res       = st.session_state["comparison_results"]
    forecasts = res["forecasts"]
    bt_results= res["bt_results"]
    train_ts  = res["train_ts"]
    test_ts   = res["test_ts"]
    current   = res["current"]

    if not forecasts:
        st.error("All models failed. Check data quality.")
        st.stop()

    # ── Metrics table ──────────────────────────────────────────────────────────
    st.markdown(f"###  Model Metrics — {store} · {item}")
    if len(test_ts) > 0:
        actual_series = test_ts.set_index("date")["sales"]
        metrics_df = build_metrics_table(forecasts, actual_series)
    else:
        # Evaluate on last horizon days of training data
        val_actual = train_ts.tail(horizon).set_index("date")["sales"]
        metrics_rows = []
        for mname, fc_df in forecasts.items():
            aligned = fc_df.set_index("date")["yhat"].reindex(val_actual.index)
            valid   = val_actual.notna() & aligned.notna()
            if valid.sum() > 0:
                metrics_rows.append(compute_all_metrics(val_actual[valid].values, aligned[valid].values, mname))
        metrics_df = pd.DataFrame(metrics_rows).set_index("Model") if metrics_rows else pd.DataFrame()

    if not metrics_df.empty:
        st.dataframe(
            metrics_df.style
                .highlight_min(subset=["MAE","RMSE","MAPE"], color="#dcfce7", axis=0)
                .format({"MAE":"{:.3f}","RMSE":"{:.3f}","MAPE":"{:.2f}","sMAPE":"{:.2f}","R²":"{:.4f}"}),
            use_container_width=True,
        )

        # Best model callout
        best = metrics_df["MAE"].idxmin()
        st.success(f" Best model by MAE: **{best}** (MAE = {metrics_df.loc[best,'MAE']:.3f})")
    else:
        st.info("Metrics require at least some overlapping dates between forecast and actuals.")

    st.divider()

    # ── Tab: Forecast overlay + Radar + Residuals ─────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["Forecast Overlay", " Radar Chart", " Residuals", " Backtest"])

    with tab1:
        fig = go.Figure()
        hist = train_ts.set_index("date")["sales"].tail(90)
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, line=dict(color="#1e293b", width=1.8), name="Historical"))
        if len(test_ts) > 0:
            fig.add_trace(go.Scatter(x=test_ts["date"], y=test_ts["sales"],
                                     line=dict(color="#64748b", width=1.5, dash="dot"), name="Actual"))
        for mname, fc_df in forecasts.items():
            c = MODEL_COLORS.get(mname, "#94a3b8")
            fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["yhat"],
                                     line=dict(color=c, width=2), name=mname))
        fig.update_layout(title="All Model Forecasts", height=420, template="plotly_white",
                          legend=dict(orientation="h", y=1.1), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not metrics_df.empty and len(metrics_df) >= 2:
            cats   = ["MAE","RMSE","MAPE","sMAPE"]
            cats_r = [c for c in cats if c in metrics_df.columns]
            # Normalize 0-1 (lower = better → invert for radar)
            norm   = metrics_df[cats_r].copy()
            for c in cats_r:
                mn, mx = norm[c].min(), norm[c].max()
                norm[c] = 1 - (norm[c] - mn) / (mx - mn + 1e-8)  # inverted: higher=better

            fig_r = go.Figure()
            for mname in norm.index:
                vals = norm.loc[mname, cats_r].tolist()
                vals += [vals[0]]
                fig_r.add_trace(go.Scatterpolar(
                    r=vals, theta=cats_r + [cats_r[0]],
                    fill="toself", name=mname,
                    line=dict(color=MODEL_COLORS.get(mname, "#94a3b8")),
                ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Model Performance Radar (higher=better, normalized)",
                height=420, showlegend=True,
            )
            st.plotly_chart(fig_r, use_container_width=True)
            st.caption("Each axis is normalized so the best model scores 1.0 (outermost). Metrics are MAE, RMSE, MAPE, sMAPE — all lower-is-better, inverted here.")
        else:
            st.info("Need ≥ 2 models with computed metrics to draw radar chart.")

    with tab3:
        if len(test_ts) > 0:
            actual = test_ts.set_index("date")["sales"]
            fig_res = make_subplots(rows=len(forecasts), cols=1, shared_xaxes=True,
                                    subplot_titles=list(forecasts.keys()),
                                    vertical_spacing=0.04)
            for i, (mname, fc_df) in enumerate(forecasts.items(), start=1):
                aligned = fc_df.set_index("date")["yhat"].reindex(actual.index)
                residuals = actual - aligned
                c = MODEL_COLORS.get(mname, "#94a3b8")
                fig_res.add_trace(go.Bar(x=residuals.index, y=residuals.values,
                                         marker_color=c, name=mname, showlegend=False),
                                  row=i, col=1)
                fig_res.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=i, col=1)
            fig_res.update_layout(height=120 * len(forecasts) + 80, template="plotly_white",
                                  title="Residuals (Actual − Predicted)")
            st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.info("Residual plot requires post-cutoff actual data.")

    with tab4:
        if res["run_backtest"] and bt_results:
            st.markdown("#### Walk-Forward Cross-Validation Results")
            bt_rows = []
            for mname, btr in bt_results.items():
                row = {"Model": mname}
                row.update({k: v for k, v in btr.metrics.items() if k != "Model"})
                bt_rows.append(row)
            bt_df = pd.DataFrame(bt_rows).set_index("Model")
            st.dataframe(bt_df.style.highlight_min(subset=["MAE","RMSE","MAPE"] if "MAE" in bt_df.columns else [],
                                                    color="#dcfce7"), use_container_width=True)
            st.caption(f"Walk-forward expanding-window CV with {n_folds} folds, horizon={horizon} days each.")
        else:
            st.info("Enable **Run walk-forward backtest** in the sidebar and press Compare Models.")

else:
    st.info(" Configure settings and press **▶ Compare Models** to begin.")