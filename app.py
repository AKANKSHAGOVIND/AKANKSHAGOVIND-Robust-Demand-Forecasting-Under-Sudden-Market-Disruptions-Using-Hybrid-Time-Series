"""
app.py  –  Adaptive Demand Forecasting System
Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Adaptive Demand Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a 0%,#1e293b 100%);border-right:1px solid #334155;}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
[data-testid="stSidebar"] .stSelectbox label,[data-testid="stSidebar"] .stSlider label{color:#94a3b8 !important;font-size:0.78rem !important;text-transform:uppercase;letter-spacing:0.08em;}
.main .block-container{background:#f8fafc;padding-top:1.5rem;}
[data-testid="metric-container"]{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:1rem;box-shadow:0 1px 3px rgba(0,0,0,.06);}
h1{font-family:'Space Mono',monospace !important;color:#0f172a !important;font-size:1.6rem !important;}
h2{font-family:'Space Mono',monospace !important;color:#1e293b !important;font-size:1.2rem !important;}
.stButton>button{background:linear-gradient(135deg,#3b82f6,#2563eb);color:white !important;border:none;border-radius:8px;font-weight:600;padding:.5rem 1.5rem;transition:all .2s;}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(59,130,246,.4);}
.stDownloadButton>button{background:linear-gradient(135deg,#10b981,#059669);color:white !important;border:none;border-radius:8px;font-weight:600;}
.regime-stable{background:#dcfce7;color:#15803d;padding:4px 14px;border-radius:20px;font-weight:700;font-size:.9rem;}
.regime-disrupted{background:#fee2e2;color:#dc2626;padding:4px 14px;border-radius:20px;font-weight:700;font-size:.9rem;}
.regime-recovery{background:#fef9c3;color:#b45309;padding:4px 14px;border-radius:20px;font-weight:700;font-size:.9rem;}
.info-box{background:white;border-left:4px solid #3b82f6;border-radius:0 8px 8px 0;padding:.8rem 1rem;margin:.5rem 0;box-shadow:0 1px 3px rgba(0,0,0,.05);}
.nav-pill{display:inline-block;background:#dbeafe;color:#1d4ed8;padding:3px 10px;border-radius:20px;font-size:.75rem;font-weight:600;margin:2px;}
.stTabs [data-baseweb="tab-list"]{gap:6px;}
.stTabs [data-baseweb="tab"]{background:#f1f5f9;border-radius:8px 8px 0 0;color:#64748b;font-weight:500;padding:8px 20px;}
.stTabs [aria-selected="true"]{background:white;color:#1d4ed8 !important;border-bottom:2px solid #3b82f6;}
</style>
""", unsafe_allow_html=True)

st.markdown("# Adaptive Demand Forecasting")
st.markdown("##### *AI-Powered Market Disruption Detection & Multi-Model Forecasting*")
st.divider()

col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
    <div class="info-box">
    <strong>What this system does:</strong><br>
    Detects structural breaks in retail sales data using CUSUM, classifies market regimes,
    and applies a regime-weighted ensemble of ARIMA, Prophet, LSTM, and Transformer models
    to generate adaptive 28-day demand forecasts.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("###  Navigation Guide")
    st.markdown("""
| Page | Purpose |
|------|---------|
| **1 · Upload Data** | Load M5 CSVs or use built-in sample data |
| **2 · Explore Data** | Visualize sales, CUSUM change points, regime timeline |
| **3 · Run Forecast** | Train all models & generate 28-day predictions |
| **4 · Model Comparison** | Compare MAE / RMSE / MAPE across all models |
    """)
with col2:
    st.markdown("###  System Capabilities")
    for icon, cap in [
        ("","CUSUM Change-Point Detection"),
        ("","Regime Classification (Stable / Disrupted / Recovery)"),
        ("","ARIMA with auto order selection"),
        ("","Facebook Prophet with holidays & SNAP"),
        ("","Stacked LSTM (TensorFlow)"),
        ("","Transformer Encoder (PyTorch)"),
        ("","Regime-Adaptive Ensemble Weighting"),
        ("","Downloadable 28-day forecast CSV"),
    ]:
        st.markdown(f"<span class='nav-pill'>{icon} {cap}</span>", unsafe_allow_html=True)

st.divider()
st.markdown("####  Get Started → Open **1_Upload_Data** in the sidebar")
st.caption("M5 Walmart Forecasting Dataset · CUSUM · ARIMA · Prophet · LSTM · Transformer · Ensemble")
