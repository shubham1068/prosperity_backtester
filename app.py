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

# ---------------- SAFE ----------------
def safe_df(df):
    return df.copy() if df is not None and not df.empty else pd.DataFrame()

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def run_backtest(strategy_cls, prices, trades, params):
    try:
        strat = strategy_cls()
        res = strat.run(prices, trades, **params)

        # 🔥 SAFETY FIXES
        if not hasattr(res, "equity_curve") or res.equity_curve is None:
            raise ValueError("No equity_curve returned")

        if not hasattr(res, "timestamps"):
            res.timestamps = pd.Series(range(len(res.equity_curve)))

        if not hasattr(res, "trades"):
            res.trades = pd.DataFrame(columns=["pnl"])

        return res

    except Exception as e:
        st.error(f"❌ Backtest failed: {e}")
        return None

@st.cache_data
def cached_metrics(equity, trades):
    return compute_all_metrics(equity, trades)

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

    # 🔥 FIX: consistent float everywhere
    min_v = safe_float(p.get("min", 0))
    max_v = safe_float(p.get("max", 100))
    step  = safe_float(p.get("step", 1))
    val0  = safe_float(default)

    if p["type"] == "slider":
        val = st.sidebar.slider(name, min_v, max_v, val0, step)
    elif p["type"] == "number":
        val = st.sidebar.number_input(name, min_value=min_v, max_value=max_v, value=val0, step=step)
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

        # 🔍 DEBUG (VERY IMPORTANT)
        if result is not None:
            st.write("📊 Equity sample:", result.equity_curve.head())
            st.write("📊 Equity std:", result.equity_curve.std())

            if hasattr(result, "trades"):
                st.write("📊 Trades sample:", result.trades.head())

        st.session_state.backtest_result = result

# ---------------- MAIN ----------------
st.title("📈 Prosperity Pro Backtester")

result = st.session_state.backtest_result

if result is None:
    st.info("👈 Upload data or generate demo to start.")
    st.stop()

# ---------------- METRICS ----------------
metrics = cached_metrics(result.equity_curve, result.trades)

col1, col2, col3, col4 = st.columns(4)

col1.metric("PnL", safe_float(metrics.get("Final PnL")))
col2.metric("Sharpe", safe_float(metrics.get("Sharpe Ratio")))
col3.metric("Drawdown", safe_float(metrics.get("Max Drawdown")))
col4.metric("Trades", metrics.get("Total Trades", 0))

# ---------------- CHECK (IMPORTANT) ----------------
if result.equity_curve.std() == 0:
    st.warning("⚠️ Strategy not trading (equity flat)")

# ---------------- TABS ----------------
tabs = st.tabs(["📈 Equity", "📊 Sharpe", "🔥 Optimizer"])

# ---- EQUITY ----
with tabs[0]:
    drawdown = compute_drawdown_series(result.equity_curve)

    fig = equity_curve_chart(
        result.equity_curve,
        result.timestamps,
        drawdown,
        strategy_name
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

        if df.empty:
            st.error("No data available")
            st.stop()

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

                if res is None or res.equity_curve.std() == 0:
                    continue

                m = compute_all_metrics(res.equity_curve)

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
        else:
            st.warning("⚠️ No valid strategies found")
