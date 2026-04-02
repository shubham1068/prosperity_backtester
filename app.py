import time
import itertools
import numpy as np
import pandas as pd
import streamlit as st

from strategies import STRATEGY_REGISTRY
from utils.parser import parse_uploaded_file, generate_demo_data
from utils.metrics import compute_all_metrics, compute_drawdown_series, rolling_sharpe
from utils.charts import equity_curve_chart, rolling_sharpe_chart

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🚀 Prosperity Pro", layout="wide")

# ---------------- STATE ----------------
if "prices_df" not in st.session_state:
    st.session_state.prices_df = pd.DataFrame()
    st.session_state.trades_df = pd.DataFrame()
    st.session_state.backtest_result = None

# ---------------- SAFE FUNCTIONS ----------------
def safe_df(df):
    return df.copy() if df is not None and not df.empty else pd.DataFrame()

def run_backtest(strategy_cls, prices, trades, params):
    try:
        strat = strategy_cls()
        return strat.run(prices, trades, **params)
    except Exception as e:
        st.error(f"❌ Backtest failed: {e}")
        return None

@st.cache_data
def cached_metrics(equity):
    return compute_all_metrics(equity)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

data_mode = st.sidebar.radio("Data Source", ["Upload", "Demo"])

if data_mode == "Upload":
    file = st.sidebar.file_uploader("Upload CSV/log")
    if file:
        prices, trades, _ = parse_uploaded_file(file.read(), file.name)
        st.session_state.prices_df = prices
        st.session_state.trades_df = trades
else:
    if st.sidebar.button("🎲 Generate Demo"):
        prices, trades = generate_demo_data()
        st.session_state.prices_df = prices
        st.session_state.trades_df = trades

# ---------------- STRATEGY ----------------
strategy_name = st.sidebar.selectbox("Strategy", list(STRATEGY_REGISTRY.keys()))
strategy_cls = STRATEGY_REGISTRY[strategy_name]

params = {}
schema = strategy_cls.param_schema()

st.sidebar.markdown("### ⚙️ Parameters")

for p in schema:
    name = p["name"]
    default = p.get("default", 0)

    if p["type"] == "slider":
        val = st.sidebar.slider(
            name,
            float(p["min"]),
            float(p["max"]),
            float(default),
            float(p["step"])
        )
    elif p["type"] == "number":
        val = st.sidebar.number_input(
            name,
            min_value=float(p["min"]),
            max_value=float(p["max"]),
            value=float(default),
            step=float(p["step"])
        )
    else:
        val = default

    params[name] = val

# ---------------- RUN BACKTEST ----------------
if st.sidebar.button("🚀 Run Backtest"):
    with st.spinner("Running backtest..."):
        result = run_backtest(
            strategy_cls,
            safe_df(st.session_state.prices_df),
            safe_df(st.session_state.trades_df),
            params
        )
        st.session_state.backtest_result = result

# ---------------- MAIN UI ----------------
st.title("📈 Prosperity Pro Backtester")

result = st.session_state.backtest_result

if result is None:
    st.info("👈 Upload data or generate demo to start.")
    st.stop()

# ---------------- METRICS ----------------
metrics = compute_all_metrics(result.equity_curve, result.trades)
col1, col2, col3, col4 = st.columns(4)
col1.metric("PnL", round(metrics["Final PnL"], 2))
col2.metric("Sharpe", round(metrics["Sharpe Ratio"], 2))
col3.metric("Drawdown", round(metrics["Max Drawdown"], 2))
col4.metric("Trades", metrics.get("Total Trades", 0))

# ---------------- TABS ----------------
tabs = st.tabs(["📈 Equity", "📊 Sharpe", "🔥 Optimizer"])

# ---- EQUITY ----
with tabs[0]:
    drawdown = compute_drawdown_series(result.equity_curve)
    fig = equity_curve_chart(
        result.equity_curve,
        result.timestamps,
        drawdown,
        result.strategy_name
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- SHARPE ----
with tabs[1]:
    rs = rolling_sharpe(result.equity_curve, 100)
    fig2 = rolling_sharpe_chart(rs, result.timestamps)
    st.plotly_chart(fig2, use_container_width=True)

# ---- OPTIMIZER ----
with tabs[2]:
    st.subheader("Grid Search")

    if st.button("Run Optimization"):
        df = safe_df(st.session_state.prices_df)

        x_vals = np.linspace(1, 10, 4)
        y_vals = np.linspace(1, 10, 4)

        total = len(x_vals) * len(y_vals)
        progress = st.progress(0)

        rows = []

        for i, (xv, yv) in enumerate(itertools.product(x_vals, y_vals)):
            p = dict(params)
            key = list(p.keys())[0]
            p[key] = xv

            try:
                res = run_backtest(strategy_cls, df, pd.DataFrame(), p)
                if res is None:
                    continue

                m = cached_metrics(res.equity_curve)

                rows.append({
                    "x": xv,
                    "y": yv,
                    "PnL": m["Final PnL"],
                    "Sharpe": m["Sharpe Ratio"]
                })
            except:
                pass

            progress.progress((i + 1) / total)

        opt_df = pd.DataFrame(rows)

        if not opt_df.empty:
            best = opt_df.loc[opt_df["Sharpe"].idxmax()]
            st.success(f"🏆 Best Sharpe: {best['Sharpe']:.2f}")

            st.dataframe(opt_df, use_container_width=True)
