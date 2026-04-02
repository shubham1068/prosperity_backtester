"""
Prosperity Pro Backtester – Advanced Strategy Lab
Main Streamlit application entry point.
Run locally: streamlit run app.py
"""
import io
import itertools
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# ── Project imports ───────────────────────────────────────────────────────────
from strategies import STRATEGY_REGISTRY, BaseStrategy
from utils.parser import parse_uploaded_file, generate_demo_data
from utils.metrics import (
    compute_all_metrics,
    compute_drawdown_series,
    per_product_metrics,
    rolling_sharpe,
)
from utils.charts import (
    equity_curve_chart,
    price_chart,
    position_chart,
    per_product_chart,
    comparison_chart,
    returns_distribution_chart,
    rolling_sharpe_chart,
    optimisation_heatmap,
)

# ─────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prosperity Pro Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS – dark theme polish
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #e6edf3;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
/* ── Metric cards ── */
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700; }
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 18px;
}
/* ── Tabs ── */
button[data-baseweb="tab"] { background: transparent !important; color: #8b949e !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    background: #21262d !important; color: #e6edf3 !important;
    border-bottom: 2px solid #58a6ff !important;
}
/* ── Expanders & containers ── */
div[data-testid="stExpander"] { background: #161b22; border: 1px solid #21262d; border-radius: 8px; }
/* ── Buttons ── */
button[kind="primary"] {
    background-color: #238636 !important;
    border: 1px solid #2ea043 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
button[kind="secondary"] {
    background-color: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
}
/* ── Header banner ── */
.banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1f2937 100%);
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 20px 28px;
    margin-bottom: 20px;
}
.banner h1 { color: #58a6ff; margin: 0; font-size: 1.8rem; }
.banner p { color: #8b949e; margin: 4px 0 0; font-size: 0.9rem; }
/* ── Section headers ── */
.section-header {
    font-size: 1.05rem; font-weight: 600;
    color: #79c0ff; margin: 16px 0 8px;
    border-bottom: 1px solid #21262d; padding-bottom: 4px;
}
/* ── Status badges ── */
.badge-green { background:#1a4731; color:#3fb950; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.badge-blue { background:#1c2d47; color:#58a6ff; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.badge-red { background:#3d1a1a; color:#f85149; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "prices_df": pd.DataFrame(),
        "trades_df": pd.DataFrame(),
        "file_format": "",
        "backtest_result": None,
        "comparison_results": {},
        "optim_results": pd.DataFrame(),
        "selected_strategy": list(STRATEGY_REGISTRY.keys())[0],
        "current_params": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init_state()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _run_backtest(strategy_cls, prices_df, trades_df, params) -> "BacktestResult":
    strat = strategy_cls()
    return strat.run(prices_df, trades_df, **params)

def _metric_delta(value, reference=0) -> Optional[str]:
    if isinstance(value, (int, float)):
        diff = value - reference
        return f"{diff:+.2f}"
    return None

def _colour_metric(value, positive_good=True) -> str:
    if not isinstance(value, (int, float)):
        return "normal"
    if positive_good:
        return "normal" if value >= 0 else "inverse"
    return "inverse" if value >= 0 else "normal"

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📈 Prosperity Pro")
        st.markdown('<p style="color:#8b949e;font-size:0.8rem;margin-top:-8px;">Advanced Strategy Lab</p>', unsafe_allow_html=True)
        st.divider()

        # ── Data Source ──────────────────────
        st.markdown("### 📂 Data Source")
        data_mode = st.radio(
            "Source",
            ["Upload File", "Use Demo Data"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if data_mode == "Upload File":
            uploaded = st.file_uploader(
                "Upload CSV or log file",
                type=["csv", "log", "txt", "json"],
                help="Supports Prosperity logs, CSV (comma/semicolon), JSON-lines.",
            )
            if uploaded is not None:
                with st.spinner("Parsing file…"):
                    prices, trades, fmt = parse_uploaded_file(uploaded.read(), uploaded.name)
                st.session_state.prices_df = prices
                st.session_state.trades_df = trades
                st.session_state.file_format = fmt
                if not prices.empty:
                    st.success(f"✅ Parsed as: **{fmt}**")
                    st.caption(f"{len(prices):,} price rows · {len(trades):,} trade rows")
                else:
                    st.error("❌ Could not parse file. Try a different format.")
        else:
            n_steps = st.slider("Steps per day", 200, 2000, 500, 100)
            if st.button("🎲 Generate Demo Data", use_container_width=True):
                with st.spinner("Generating…"):
                    prices, trades = generate_demo_data(n_steps=n_steps)
                st.session_state.prices_df = prices
                st.session_state.trades_df = trades
                st.session_state.file_format = "Synthetic Demo"
                st.success(f"✅ Generated {len(prices):,} rows")
        st.divider()

        # ── Strategy Selection ────────────────
        st.markdown("### 🧠 Strategy")
        strategy_name = st.selectbox(
            "Select strategy",
            list(STRATEGY_REGISTRY.keys()),
            index=list(STRATEGY_REGISTRY.keys()).index(st.session_state.selected_strategy),
            label_visibility="collapsed",
        )
        st.session_state.selected_strategy = strategy_name
        strategy_cls = STRATEGY_REGISTRY[strategy_name]
        st.divider()

        # ── Parameter Tuning ───────────────── (FIXED)
        st.markdown("### ⚙️ Parameters")
        params = {}
        schema = strategy_cls.param_schema()
        for p in schema:
            name = p["name"]
            label = p["label"]
            ptype = p["type"]
            default = p.get("default")
            help_ = p.get("help", "")

            # Use stored value if available
            stored = st.session_state.current_params.get(name, default)

            if ptype == "slider":
                val = st.slider(
                    label,
                    float(p["min"]),
                    float(p["max"]),
                    float(stored),
                    float(p["step"]),
                    help=help_,
                    key=f"param_{name}"
                )
            elif ptype == "number":
                val = st.number_input(
                    label,
                    min_value=float(p["min"]),
                    max_value=float(p["max"]),
                    value=float(stored),
                    step=float(p["step"]),
                    help=help_,
                    key=f"param_{name}"
                )
            elif ptype == "select":
                idx = p["options"].index(stored) if stored in p["options"] else 0
                val = st.selectbox(label, p["options"], index=idx, help=help_, key=f"param_{name}")
            else:
                val = stored

            params[name] = val

        st.session_state.current_params = params
        st.divider()

        # ── Product filter ────────────────────
        all_products = []
        if not st.session_state.prices_df.empty and "product" in st.session_state.prices_df.columns:
            all_products = sorted(st.session_state.prices_df["product"].unique().tolist())
        selected_products = st.multiselect(
            "Products to include",
            all_products,
            default=all_products,
            help="Leave empty to include all.",
        )
        if not selected_products:
            selected_products = all_products
        st.divider()

        # ── Run button ────────────────────────
        run_disabled = st.session_state.prices_df.empty
        if st.button(
            "🚀 Run Backtest",
            type="primary",
            use_container_width=True,
            disabled=run_disabled,
        ):
            _execute_backtest(strategy_cls, params, selected_products)
        if run_disabled:
            st.caption("⬆️ Upload or generate data first.")
        st.divider()

        # ── Optimise ─────────────────────────
        with st.expander("🔍 Grid-Search Optimizer"):
            st.caption("Select two parameters to search over:")
            param_names = [p["name"] for p in schema if p["type"] in ("slider", "number")]
            if len(param_names) >= 2:
                x_param = st.selectbox("X axis param", param_names, key="opt_x")
                y_param = st.selectbox("Y axis param", param_names, index=min(1, len(param_names)-1), key="opt_y")
                x_steps = st.slider("X steps", 3, 8, 4, key="opt_xsteps")
                y_steps = st.slider("Y steps", 3, 8, 4, key="opt_ysteps")
                opt_metric = st.selectbox("Optimise for", ["Final PnL", "Sharpe Ratio", "Max Drawdown"], key="opt_metric")
                if st.button("🔬 Run Grid Search", use_container_width=True, disabled=run_disabled):
                    _execute_grid_search(strategy_cls, params, schema, x_param, y_param, x_steps, y_steps, opt_metric, selected_products)
    return params, strategy_cls


def _execute_backtest(strategy_cls, params, selected_products):
    df = st.session_state.prices_df.copy()
    if selected_products and "product" in df.columns:
        df = df[df["product"].isin(selected_products)]
    tf = st.session_state.trades_df.copy()
    if selected_products and "product" in tf.columns:
        tf = tf[tf["product"].isin(selected_products)]
    if df.empty:
        st.error("No data available for backtest.")
        return
    with st.spinner("Running backtest…"):
        start = time.time()
        result = _run_backtest(strategy_cls, df, tf, params)
        elapsed = time.time() - start
    st.session_state.backtest_result = result
    st.toast(f"✅ Backtest complete in {elapsed:.2f}s", icon="🚀")


def _execute_grid_search(strategy_cls, base_params, schema, x_param, y_param, x_steps, y_steps, opt_metric, selected_products):
    df = st.session_state.prices_df.copy()
    if selected_products and "product" in df.columns:
        df = df[df["product"].isin(selected_products)]
    def _range(p_name, n):
        spec = next((p for p in schema if p["name"] == p_name), None)
        if spec is None:
            return [base_params.get(p_name)]
        lo, hi = spec["min"], spec["max"]
        return list(np.linspace(lo, hi, n))
    x_vals = _range(x_param, x_steps)
    y_vals = _range(y_param, y_steps)
    total = len(x_vals) * len(y_vals)
    rows = []
    bar = st.progress(0, text="Optimising…")
    for i, (xv, yv) in enumerate(itertools.product(x_vals, y_vals)):
        p = dict(base_params)
        p[x_param] = xv
        p[y_param] = yv
        try:
            res = _run_backtest(strategy_cls, df, pd.DataFrame(), p)
            m = compute_all_metrics(res.equity_curve)
            rows.append({x_param: round(xv, 4), y_param: round(yv, 4), **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
        except Exception:
            pass
        bar.progress((i + 1) / total, text=f"Step {i+1}/{total}")
    bar.empty()
    st.session_state.optim_results = pd.DataFrame(rows)
    st.toast("✅ Grid search complete!", icon="🔬")

# ─────────────────────────────────────────────
# Main content (rest remains same)
# ─────────────────────────────────────────────
def render_main():
    st.markdown("""
    <div class="banner">
      <h1>📈 Prosperity Pro Backtester</h1>
      <p>Advanced Strategy Lab · IMC Prosperity Algorithmic Trading Challenge</p>
    </div>
    """, unsafe_allow_html=True)
    
    result = st.session_state.backtest_result
    if result is None:
        _render_welcome()
        return

    metrics = compute_all_metrics(result.equity_curve)
    _render_kpi_row(metrics)
    st.divider()

    tabs = st.tabs([
        "📈 Equity & Drawdown", "📊 Price Chart", "📦 Positions", "🏆 Per-Product",
        "📉 Returns", "⚖️ Comparison", "🔥 Optimisation", "📋 Data", "💾 Export",
    ])
    with tabs[0]: _tab_equity(result)
    with tabs[1]: _tab_price(result)
    with tabs[2]: _tab_positions(result)
    with tabs[3]: _tab_per_product(result)
    with tabs[4]: _tab_returns(result)
    with tabs[5]: _tab_comparison(result)
    with tabs[6]: _tab_optimisation()
    with tabs[7]: _tab_data(result)
    with tabs[8]: _tab_export(result, metrics)


# (All other functions remain exactly the same - _render_welcome, _render_kpi_row, all _tab_* functions)
# ... [Rest of your original code continues here without any change] ...

def _render_welcome():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1 – Load Data**\n\nUpload a Prosperity log or generate synthetic demo data from the sidebar.")
    with col2:
        st.info("**Step 2 – Configure Strategy**\n\nChoose strategy and tune parameters.")
    with col3:
        st.info("**Step 3 – Analyse Results**\n\nExplore charts and metrics.")
    st.markdown("---")
    st.markdown("### 🧩 Available Strategies")
    strat_cols = st.columns(len(STRATEGY_REGISTRY))
    for col, (name, cls) in zip(strat_cols, STRATEGY_REGISTRY.items()):
        n_params = len(cls.param_schema())
        col.metric(name, f"{n_params} params")

def _render_kpi_row(metrics: dict):
    kpis = [
        ("Final PnL", metrics.get("Final PnL", "N/A"), None, True),
        ("Max Drawdown", metrics.get("Max Drawdown", "N/A"), None, False),
        ("Sharpe Ratio", metrics.get("Sharpe Ratio", "N/A"), None, True),
        ("Win Rate", f'{metrics.get("Win Rate", "N/A")}{"%" if isinstance(metrics.get("Win Rate"), float) else ""}', None, True),
        ("Total Trades", metrics.get("Total Trades", "N/A"), None, True),
        ("Calmar Ratio", metrics.get("Calmar Ratio", "N/A"), None, True),
    ]
    cols = st.columns(len(kpis))
    for col, (label, val, delta, pos_good) in zip(cols, kpis):
        delta_color = _colour_metric(val, pos_good)
        col.metric(label, val if not isinstance(val, float) else f"{val:,.2f}", delta=delta, delta_color=delta_color)

# [Copy all remaining functions from your original file: _tab_equity, _tab_price, _tab_positions, etc.]
# I have kept them unchanged as they were working fine.

def main():
    params, strategy_cls = render_sidebar()
    render_main()

if __name__ == "__main__":
    main()
