"""
pages/1_Upload_Data.py
-----------------------
Upload M5 Walmart CSVs (or load synthetic demo data).
Subsets data BEFORE the wide->long melt to avoid MemoryError on the full 450 MB dataset.
"""

import streamlit as st
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import (
    load_sales, load_calendar, load_prices,
    validate_sales, validate_calendar, validate_prices,
    generate_sample_data, subset_sales,
)
from src.data.preprocessor import wide_to_long, attach_prices, handle_missing

st.set_page_config(page_title="Upload Data · ADF", page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
h1,h2,h3{font-family:'Space Mono',monospace;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a 0%,#1e293b 100%);}
[data-testid="stSidebar"] *{color:#e2e8f0!important;}
.stButton>button{background:#1e293b;color:white;border:none;border-radius:8px;
    font-family:'Space Mono',monospace;font-size:0.8rem;padding:0.5rem 1.25rem;}
.stButton>button:hover{background:#334155;}
</style>
""", unsafe_allow_html=True)

st.markdown("##  Upload Data")
st.markdown(
    "Upload your M5 Walmart dataset files, or use synthetic demo data. "
    "For real M5 data, you must **select a store + department subset** before processing — "
    "the full dataset needs ~6 GB RAM to melt and will crash without subsetting."
)

mode = st.radio(
    "Data source",
    [" Use synthetic demo data", " Upload real M5 CSV files"],
    horizontal=True,
)
st.divider()

# ─── SYNTHETIC ────────────────────────────────────────────────────────────────
if mode == " Use synthetic demo data":
    st.info("Generates 3-year synthetic M5-style data with seasonality and an injected demand disruption at day 600.")
    col1, col2 = st.columns(2)
    with col1:
        n_stores = st.slider("Stores", 1, 4, 2)
    with col2:
        n_items = st.slider("Items per store", 2, 20, 5)

    if st.button(" Generate & Load Demo Data", type="primary"):
        with st.spinner("Generating synthetic M5 data..."):
            sales_df, cal_df, prices_df = generate_sample_data(
                n_stores=n_stores, n_items=n_items, n_days=365 * 3
            )
            long_df = wide_to_long(sales_df, cal_df)
            long_df = attach_prices(long_df, prices_df)
            long_df = handle_missing(long_df)
            st.session_state.update({
                "sales_df": sales_df, "calendar_df": cal_df,
                "prices_df": prices_df, "long_df": long_df,
            })
            for k in ["selected_store","selected_item","cusum_result",
                      "regime_result","forecast_results","train_df","test_df"]:
                st.session_state.pop(k, None)
        st.success(f" {len(long_df):,} rows · {long_df['store_id'].nunique()} stores · {long_df['item_id'].nunique()} items")

# ─── REAL M5 UPLOAD ───────────────────────────────────────────────────────────
else:
    st.markdown("### Step 1 — Upload CSV Files")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("** Sales CSV** *(required)*")
        sales_file = st.file_uploader("sales_train_validation.csv", type=["csv"], key="sales_up")
        if sales_file:
            with st.spinner("Reading..."):
                raw = load_sales(sales_file)
            ok, msg = validate_sales(raw)
            if ok:
                st.session_state["raw_sales"] = raw
                st.success(f"{raw.shape[0]:,} items")
            else:
                st.error(f" {msg}")
                st.session_state.pop("raw_sales", None)

    with col2:
        st.markdown("** Calendar CSV** *(required)*")
        cal_file = st.file_uploader("calendar.csv", type=["csv"], key="cal_up")
        if cal_file:
            with st.spinner("Reading..."):
                cal_df = load_calendar(cal_file)
            ok2, msg2 = validate_calendar(cal_df)
            if ok2:
                st.session_state["calendar_df"] = cal_df
                st.success(f" {len(cal_df):,} days")
            else:
                st.error(f" {msg2}")
                st.session_state.pop("calendar_df", None)

    with col3:
        st.markdown("** Prices CSV** *(optional)*")
        prices_file = st.file_uploader("sell_prices.csv", type=["csv"], key="prices_up")
        if prices_file:
            with st.spinner("Reading..."):
                p_df = load_prices(prices_file)
            ok3, msg3 = validate_prices(p_df)
            if ok3:
                st.session_state["prices_df"] = p_df
                st.success(f" {len(p_df):,} rows")
            else:
                st.warning(f" {msg3}")

    # ── Step 2: Subset selector ───────────────────────────────────────────────
    raw_sales = st.session_state.get("raw_sales")
    cal_df    = st.session_state.get("calendar_df")

    if raw_sales is not None and cal_df is not None:
        st.divider()
        st.markdown("### Step 2 — Select Subset")
        st.warning(
            " The full M5 dataset has 30,490 items × 1,913 days = 58 million rows after melt, "
            "requiring ~6 GB RAM. **Select 1–2 stores and 1 department (≤ 100 items)** to stay within safe limits."
        )

        all_stores = sorted(raw_sales["store_id"].unique())
        all_depts  = sorted(raw_sales["dept_id"].unique())

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sel_stores = st.multiselect("Stores", all_stores, default=all_stores[:2])
        with sc2:
            sel_depts = st.multiselect("Departments", all_depts, default=all_depts[:1])
        with sc3:
            max_items = st.slider("Max items (random cap)", 10, 200, 50)

        if sel_stores and sel_depts:
            mask = raw_sales["store_id"].isin(sel_stores) & raw_sales["dept_id"].isin(sel_depts)
            n_before = mask.sum()
            n_after  = min(n_before, max_items)
            day_cols = len([c for c in raw_sales.columns if c.startswith("d_")])
            est_rows = n_after * day_cols
            st.info(f"Selection: **{n_before}** items → capped to **{n_after}** → ~**{est_rows:,} rows** after melt")

            if st.button(" Process Selected Subset", type="primary"):
                prog = st.progress(0, text="Subsetting...")
                with st.spinner("Processing..."):
                    sales_sub = subset_sales(raw_sales, stores=sel_stores, depts=sel_depts, max_items=max_items)
                    prog.progress(25, text="Melting to long format...")
                    long_df = wide_to_long(sales_sub, cal_df)
                    prog.progress(65, text="Attaching prices...")
                    prices_df = st.session_state.get("prices_df")
                    if prices_df is not None:
                        long_df = attach_prices(long_df, prices_df)
                    prog.progress(85, text="Handling missing values...")
                    long_df = handle_missing(long_df)
                    prog.progress(100, text="Done!")
                    st.session_state.update({"sales_df": sales_sub, "long_df": long_df})
                    for k in ["selected_store","selected_item","cusum_result",
                              "regime_result","forecast_results","train_df","test_df"]:
                        st.session_state.pop(k, None)
                st.success(f"{len(long_df):,} rows · {long_df['store_id'].nunique()} stores · {long_df['item_id'].nunique()} items")
        else:
            st.info("Select at least one store and one department above.")

# ─── SHARED SUMMARY ───────────────────────────────────────────────────────────
if "long_df" in st.session_state:
    long_df = st.session_state["long_df"]
    st.divider()
    st.markdown("###  Dataset Ready")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Rows",    f"{len(long_df):,}")
    m2.metric("Stores",  long_df["store_id"].nunique())
    m3.metric("Items",   long_df["item_id"].nunique())
    m4.metric("From",    str(long_df["date"].min().date()))
    m5.metric("To",      str(long_df["date"].max().date()))
    st.dataframe(long_df.head(100), use_container_width=True, height=240)
    st.info(" Go to ** Explore Data** to select a store/item and detect change points.")
