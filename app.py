"""
app.py
======
Prosperity Pro Backtester – Advanced Strategy Lab
Main Streamlit application entry point.

Run locally:
    streamlit run app.py
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
[data-testid="stMetricValue"]  { font-size: 1.4rem !important; font-weight: 700; }
[data-testid="stMetricDelta"]  { font-size: 0.85rem !important; }
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
.banner p  { color: #8b949e; margin: 4px 0 0; font-size: 0.9rem; }

/* ── Section headers ── */
.section-header {
    font-size: 1.05rem; font-weight: 600;
    color: #79c0ff; margin: 16px 0 8px;
    border-bottom: 1px solid #21262d; padding-bottom: 4px;
}

/* ── Status badges ── */
.badge-green { background:#1a4731; color:#3fb950; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.badge-blue  { background:#1c2d47; color:#58a6ff; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.badge-red   { background:#3d1a1a; color:#f85149; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────

def _init_state():
    defaults = {
        "prices_df":         pd.DataFrame(),
        "trades_df":         pd.DataFrame(),
        "file_format":       "",
        "backtest_result":   None,
        "comparison_results": {},
        "optim_results":     pd.DataFrame(),
        "selected_strategy": list(STRATEGY_REGISTRY.keys())[0],
        "current_params":    {},
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
                st.session_state.prices_df  = prices
                st.session_state.trades_df  = trades
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
                st.session_state.prices_df   = prices
                st.session_state.trades_df   = trades
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

        # ── Parameter Tuning ─────────────────
        st.markdown("### ⚙️ Parameters")
        params = {}
        schema = strategy_cls.param_schema()

        for p in schema:
            name    = p["name"]
            label   = p["label"]
            ptype   = p["type"]
            default = p.get("default")
            help_   = p.get("help", "")

            # Use stored value if available
            stored = st.session_state.current_params.get(name, default)

            if ptype == "slider":
                val = st.slider(label, p["min"], p["max"], float(stored), p["step"], help=help_, key=f"param_{name}")
            elif ptype == "number":
                val = st.number_input(label, min_value=p["min"], max_value=p["max"],
                                      value=float(stored), step=float(p["step"]), help=help_, key=f"param_{name}")
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

    # Build grid ranges
    def _range(p_name, n):
        spec = next((p for p in schema if p["name"] == p_name), None)
        if spec is None:
            return [base_params.get(p_name)]
        lo, hi = spec["min"], spec["max"]
        return list(np.linspace(lo, hi, n))

    x_vals = _range(x_param, x_steps)
    y_vals = _range(y_param, y_steps)
    total  = len(x_vals) * len(y_vals)

    rows = []
    bar  = st.progress(0, text="Optimising…")

    for i, (xv, yv) in enumerate(itertools.product(x_vals, y_vals)):
        p = dict(base_params)
        p[x_param] = xv
        p[y_param] = yv

        try:
            res = _run_backtest(strategy_cls, df, pd.DataFrame(), p)
            m   = compute_all_metrics(res.equity_curve)
            rows.append({x_param: round(xv, 4), y_param: round(yv, 4), **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
        except Exception:
            pass

        bar.progress((i + 1) / total, text=f"Step {i+1}/{total}")

    bar.empty()
    st.session_state.optim_results = pd.DataFrame(rows)
    st.toast("✅ Grid search complete!", icon="🔬")


# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────

def render_main():
    # Header
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

    # ── KPI Row ──────────────────────────────
    metrics = compute_all_metrics(result.equity_curve)
    _render_kpi_row(metrics)

    st.divider()

    # ── Tabs ─────────────────────────────────
    tabs = st.tabs([
        "📈 Equity & Drawdown",
        "📊 Price Chart",
        "📦 Positions",
        "🏆 Per-Product",
        "📉 Returns",
        "⚖️ Comparison",
        "🔥 Optimisation",
        "📋 Data",
        "💾 Export",
    ])

    with tabs[0]:
        _tab_equity(result)

    with tabs[1]:
        _tab_price(result)

    with tabs[2]:
        _tab_positions(result)

    with tabs[3]:
        _tab_per_product(result)

    with tabs[4]:
        _tab_returns(result)

    with tabs[5]:
        _tab_comparison(result)

    with tabs[6]:
        _tab_optimisation()

    with tabs[7]:
        _tab_data(result)

    with tabs[8]:
        _tab_export(result, metrics)


def _render_welcome():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1 – Load Data**\n\nUpload a Prosperity log (CSV / semicolon / JSON-lines) or generate synthetic demo data from the sidebar.")
    with col2:
        st.info("**Step 2 – Configure Strategy**\n\nChoose from 4 built-in strategies. Tune parameters live with sliders and number inputs.")
    with col3:
        st.info("**Step 3 – Analyse Results**\n\nExplore equity curves, trade markers, per-product stats, returns distribution, and more.")

    st.markdown("---")
    st.markdown("### 🧩 Available Strategies")
    strat_cols = st.columns(len(STRATEGY_REGISTRY))
    for col, (name, cls) in zip(strat_cols, STRATEGY_REGISTRY.items()):
        n_params = len(cls.param_schema())
        col.metric(name, f"{n_params} params")


def _render_kpi_row(metrics: dict):
    kpis = [
        ("Final PnL",    metrics.get("Final PnL", "N/A"),    None,          True),
        ("Max Drawdown", metrics.get("Max Drawdown", "N/A"), None,          False),
        ("Sharpe Ratio", metrics.get("Sharpe Ratio", "N/A"), None,          True),
        ("Win Rate",     f'{metrics.get("Win Rate", "N/A")}{"%" if isinstance(metrics.get("Win Rate"), float) else ""}', None, True),
        ("Total Trades", metrics.get("Total Trades", "N/A"), None,          True),
        ("Calmar Ratio", metrics.get("Calmar Ratio", "N/A"), None,          True),
    ]
    cols = st.columns(len(kpis))
    for col, (label, val, delta, pos_good) in zip(cols, kpis):
        delta_color = _colour_metric(val, pos_good)
        col.metric(label, val if not isinstance(val, float) else f"{val:,.2f}", delta=delta, delta_color=delta_color)


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

def _tab_equity(result):
    st.markdown('<div class="section-header">Equity Curve & Drawdown</div>', unsafe_allow_html=True)
    drawdown = compute_drawdown_series(result.equity_curve)
    fig = equity_curve_chart(result.equity_curve, result.timestamps, drawdown, result.strategy_name)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Rolling Sharpe (window=100)</div>', unsafe_allow_html=True)
        rs = rolling_sharpe(result.equity_curve, 100)
        fig2 = rolling_sharpe_chart(rs, result.timestamps)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Full Metrics</div>', unsafe_allow_html=True)
        metrics = compute_all_metrics(result.equity_curve)
        df_m = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
        st.dataframe(df_m, hide_index=True, use_container_width=True, height=350)


def _tab_price(result):
    st.markdown('<div class="section-header">Price Chart with Trade Markers</div>', unsafe_allow_html=True)
    prices_df = st.session_state.prices_df
    trades_df = st.session_state.trades_df

    products = []
    if "product" in prices_df.columns:
        products = sorted(prices_df["product"].unique().tolist())

    selected_prod = st.selectbox("Product", ["All"] + products, key="price_product_sel") if products else "All"
    prod = None if selected_prod == "All" else selected_prod

    fig = price_chart(prices_df, trades_df, prod)
    st.plotly_chart(fig, use_container_width=True)


def _tab_positions(result):
    st.markdown('<div class="section-header">Position Over Time</div>', unsafe_allow_html=True)
    if result.positions.empty:
        st.info("No position data available.")
        return
    fig = position_chart(result.positions, result.timestamps)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Long",   f"{result.positions.max():.0f}")
    col2.metric("Max Short",  f"{result.positions.min():.0f}")
    col3.metric("Avg |Pos|",  f"{result.positions.abs().mean():.1f}")
    col4.metric("Time Flat",  f"{(result.positions == 0).mean() * 100:.1f}%")


def _tab_per_product(result):
    st.markdown('<div class="section-header">Per-Product Performance</div>', unsafe_allow_html=True)
    if not result.per_product_pnl:
        st.info("No per-product data.")
        return

    # Build equity for each product
    prod_metrics_df = per_product_metrics(result.per_product_pnl)
    fig = per_product_chart(prod_metrics_df)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(prod_metrics_df, hide_index=True, use_container_width=True)

    # Individual equity curves
    st.markdown('<div class="section-header">Per-Product Equity Curves</div>', unsafe_allow_html=True)
    eq_dict = result.per_product_pnl
    fig2 = comparison_chart(eq_dict)
    st.plotly_chart(fig2, use_container_width=True)


def _tab_returns(result):
    st.markdown('<div class="section-header">Returns Distribution</div>', unsafe_allow_html=True)
    fig = returns_distribution_chart(result.equity_curve)
    st.plotly_chart(fig, use_container_width=True)

    metrics = compute_all_metrics(result.equity_curve)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Skewness",  metrics.get("Skewness", "N/A"))
    col2.metric("Kurtosis",  metrics.get("Kurtosis", "N/A"))
    col3.metric("Return Std",metrics.get("Return Std", "N/A"))
    col4.metric("Avg Return",metrics.get("Avg Step Return", "N/A"))


def _tab_comparison(result):
    st.markdown('<div class="section-header">Side-by-Side Strategy Comparison</div>', unsafe_allow_html=True)

    st.info("Run multiple strategies and add them here for comparison.")
    add_current = st.button("➕ Add Current Result to Comparison", key="add_to_compare")
    if add_current and result is not None:
        label = f"{result.strategy_name} ({len(st.session_state.comparison_results)+1})"
        st.session_state.comparison_results[label] = result.equity_curve
        st.success(f"Added '{label}'")

    if st.button("🗑️ Clear Comparison", key="clear_compare"):
        st.session_state.comparison_results = {}

    if st.session_state.comparison_results:
        fig = comparison_chart(st.session_state.comparison_results)
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        rows = []
        for name, eq in st.session_state.comparison_results.items():
            m = compute_all_metrics(eq)
            rows.append({
                "Strategy":     name,
                "Final PnL":    m.get("Final PnL"),
                "Max Drawdown": m.get("Max Drawdown"),
                "Sharpe":       m.get("Sharpe Ratio"),
                "Sortino":      m.get("Sortino Ratio"),
                "Calmar":       m.get("Calmar Ratio"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No strategies added yet. Run backtests and click 'Add' above.")


def _tab_optimisation():
    st.markdown('<div class="section-header">Grid-Search Optimisation Results</div>', unsafe_allow_html=True)

    if st.session_state.optim_results.empty:
        st.info("Run the Grid-Search Optimizer from the sidebar to see results here.")
        return

    df = st.session_state.optim_results
    numeric_cols = [c for c in df.columns if c not in df.columns[:2]]
    metric_sel = st.selectbox("Metric to visualise", [c for c in numeric_cols if c in ["Final PnL", "Sharpe Ratio", "Max Drawdown"] or True], key="opt_vis_metric")

    x_col = df.columns[0]
    y_col = df.columns[1]

    fig = optimisation_heatmap(df, x_col, y_col, metric_sel)
    st.plotly_chart(fig, use_container_width=True)

    best_row = df.loc[df[metric_sel].idxmax()] if metric_sel != "Max Drawdown" else df.loc[df[metric_sel].idxmax()]
    st.success(f"🏆 Best **{metric_sel}**: `{best_row[metric_sel]:.2f}` at {x_col}={best_row[x_col]:.4f}, {y_col}={best_row[y_col]:.4f}")

    st.markdown("**Full results table:**")
    st.dataframe(df.sort_values(metric_sel, ascending=(metric_sel == "Max Drawdown")), hide_index=True, use_container_width=True)


def _tab_data(result):
    st.markdown('<div class="section-header">Raw Data Preview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Price Data** (first 200 rows)")
        prices = st.session_state.prices_df
        if not prices.empty:
            st.dataframe(prices.head(200), use_container_width=True, height=350)
            st.caption(f"Format: **{st.session_state.file_format}** · {len(prices):,} rows · {prices.shape[1]} columns")
        else:
            st.info("No price data loaded.")

    with c2:
        st.markdown("**Trade Data** (first 200 rows)")
        trades = st.session_state.trades_df
        if not trades.empty:
            st.dataframe(trades.head(200), use_container_width=True, height=350)
            st.caption(f"{len(trades):,} trades")
        else:
            st.info("No trade data loaded.")

    st.markdown('<div class="section-header">Backtest Equity (first 500 steps)</div>', unsafe_allow_html=True)
    eq_df = pd.DataFrame({"timestamp": result.timestamps[:500] if not result.timestamps.empty else range(min(500, len(result.equity_curve))), "equity": result.equity_curve.values[:500]})
    st.dataframe(eq_df, hide_index=True, use_container_width=True, height=250)


def _tab_export(result, metrics):
    st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # ── Equity CSV ────────────────────────────
    with col1:
        st.markdown("**Equity Curve CSV**")
        eq_df = pd.DataFrame({
            "timestamp": result.timestamps.values if not result.timestamps.empty else range(len(result.equity_curve)),
            "equity":    result.equity_curve.values,
            "position":  result.positions.values if not result.positions.empty else np.zeros(len(result.equity_curve)),
        })
        csv_eq = eq_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Equity CSV", csv_eq, "equity_curve.csv", "text/csv", use_container_width=True)

    # ── Metrics CSV ───────────────────────────
    with col2:
        st.markdown("**Performance Metrics CSV**")
        metrics_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
        csv_m = metrics_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Metrics CSV", csv_m, "metrics.csv", "text/csv", use_container_width=True)

    # ── Params CSV ────────────────────────────
    with col3:
        st.markdown("**Strategy Parameters CSV**")
        params_df = pd.DataFrame({"Parameter": list(result.params_used.keys()), "Value": list(result.params_used.values())})
        csv_p = params_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Params CSV", csv_p, "params.csv", "text/csv", use_container_width=True)

    st.divider()

    # ── Per-product CSV ───────────────────────
    if result.per_product_pnl:
        st.markdown("**Per-Product Equity CSV**")
        pp_df = pd.DataFrame(result.per_product_pnl)
        csv_pp = pp_df.to_csv(index=True).encode("utf-8")
        st.download_button("⬇️ Download Per-Product CSV", csv_pp, "per_product_equity.csv", "text/csv", use_container_width=True)

    # ── Optimisation results ──────────────────
    if not st.session_state.optim_results.empty:
        st.markdown("**Grid-Search Results CSV**")
        csv_opt = st.session_state.optim_results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Optimisation CSV", csv_opt, "optimisation.csv", "text/csv", use_container_width=True)

    st.info("💡 To export charts as images, use the camera icon (🎥) in the top-right of each Plotly chart.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    params, strategy_cls = render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
