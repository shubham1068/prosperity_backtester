"""
utils/charts.py
---------------
Plotly chart builders for the Prosperity Pro Backtester.
All charts use a consistent dark theme.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

# ─────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────

THEME = dict(
    bg          = "#0e1117",
    panel_bg    = "#161b22",
    grid        = "#21262d",
    text        = "#e6edf3",
    subtext     = "#8b949e",
    accent1     = "#58a6ff",   # blue
    accent2     = "#3fb950",   # green
    accent3     = "#f85149",   # red
    accent4     = "#d2a8ff",   # purple
    accent5     = "#ffa657",   # orange
    accent6     = "#79c0ff",   # light blue
)

_LAYOUT_BASE = dict(
    paper_bgcolor = THEME["bg"],
    plot_bgcolor  = THEME["panel_bg"],
    font          = dict(color=THEME["text"], family="Inter, sans-serif", size=12),
    xaxis         = dict(gridcolor=THEME["grid"], zeroline=False, showgrid=True),
    yaxis         = dict(gridcolor=THEME["grid"], zeroline=False, showgrid=True),
    margin        = dict(l=50, r=30, t=50, b=40),
    hoverlabel    = dict(bgcolor=THEME["panel_bg"], font_color=THEME["text"]),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=THEME["grid"]),
)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=dict(text=title, font_size=16, font_color=THEME["text"]), **_LAYOUT_BASE)
    fig.update_xaxes(gridcolor=THEME["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=THEME["grid"], zeroline=False)
    return fig


# ─────────────────────────────────────────────
# 1. Equity Curve
# ─────────────────────────────────────────────

def equity_curve_chart(
    equity: pd.Series,
    timestamps: pd.Series = None,
    drawdown: pd.Series = None,
    strategy_name: str = "Strategy",
) -> go.Figure:
    """Combined equity + drawdown chart with dual y-axes."""
    x = timestamps if timestamps is not None else equity.index

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Equity Curve", "Drawdown"),
    )

    # Equity
    fig.add_trace(go.Scatter(
        x=x, y=equity,
        name=strategy_name,
        line=dict(color=THEME["accent1"], width=2),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
    ), row=1, col=1)

    # Zero line
    fig.add_hline(y=0, line=dict(color=THEME["subtext"], dash="dash", width=1), row=1, col=1)

    # Drawdown
    if drawdown is not None and not drawdown.empty:
        fig.add_trace(go.Scatter(
            x=x, y=drawdown,
            name="Drawdown",
            line=dict(color=THEME["accent3"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(248,81,73,0.15)",
        ), row=2, col=1)

    fig = _apply_theme(fig, "📈 Equity Curve & Drawdown")
    fig.update_layout(height=480, showlegend=True)
    return fig


# ─────────────────────────────────────────────
# 2. Price Chart with Trade Markers
# ─────────────────────────────────────────────

def price_chart(
    prices_df: pd.DataFrame,
    trades_df: pd.DataFrame = None,
    product: str = None,
) -> go.Figure:
    """Candlestick-style price chart with buy/sell markers."""

    if product:
        df = prices_df[prices_df["product"] == product].copy() if "product" in prices_df.columns else prices_df.copy()
    else:
        df = prices_df.copy()

    if df.empty:
        return _empty_chart("No price data available")

    x = df.get("timestamp", df.index)

    fig = go.Figure()

    # Mid price fill band (bid/ask spread)
    if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
        fig.add_trace(go.Scatter(
            x=list(x) + list(x)[::-1],
            y=list(df["ask_price_1"]) + list(df["bid_price_1"])[::-1],
            fill="toself",
            fillcolor="rgba(88,166,255,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Bid-Ask Spread",
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(x=x, y=df["bid_price_1"], line=dict(color=THEME["accent3"], width=1, dash="dot"), name="Bid L1"))
        fig.add_trace(go.Scatter(x=x, y=df["ask_price_1"], line=dict(color=THEME["accent2"], width=1, dash="dot"), name="Ask L1"))

    # Mid price
    if "mid_price" in df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=df["mid_price"],
            line=dict(color=THEME["accent1"], width=2),
            name="Mid Price",
        ))

    # Trade markers
    if trades_df is not None and not trades_df.empty:
        if product and "product" in trades_df.columns:
            tf = trades_df[trades_df["product"] == product]
        else:
            tf = trades_df

        if not tf.empty and "timestamp" in tf.columns and "price" in tf.columns:
            side_col = "side" if "side" in tf.columns else None

            buys  = tf[tf[side_col].str.upper() == "BUY"]  if side_col else tf
            sells = tf[tf[side_col].str.upper() == "SELL"] if side_col else pd.DataFrame()

            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys["timestamp"], y=buys["price"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color=THEME["accent2"]),
                    name="Buy",
                ))
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["timestamp"], y=sells["price"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=10, color=THEME["accent3"]),
                    name="Sell",
                ))

    title = f"📊 Price Chart – {product}" if product else "📊 Price Chart"
    _apply_theme(fig, title)
    fig.update_layout(height=420)
    return fig


# ─────────────────────────────────────────────
# 3. Position over Time
# ─────────────────────────────────────────────

def position_chart(position_series: pd.Series, timestamps: pd.Series = None) -> go.Figure:
    """Step-chart showing position size over time."""
    x = timestamps if timestamps is not None else position_series.index
    fig = go.Figure()

    # Colour by sign
    fig.add_trace(go.Bar(
        x=x,
        y=position_series.clip(lower=0),
        name="Long",
        marker_color=THEME["accent2"],
        opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=position_series.clip(upper=0),
        name="Short",
        marker_color=THEME["accent3"],
        opacity=0.8,
    ))
    fig.add_hline(y=0, line=dict(color=THEME["subtext"], width=1))

    _apply_theme(fig, "📦 Position Over Time")
    fig.update_layout(height=300, barmode="relative")
    return fig


# ─────────────────────────────────────────────
# 4. Per-Product Performance
# ─────────────────────────────────────────────

def per_product_chart(metrics_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of final PnL per product."""
    if metrics_df.empty:
        return _empty_chart("No product metrics available")

    colors = [THEME["accent2"] if v >= 0 else THEME["accent3"] for v in metrics_df["Final PnL"]]

    fig = go.Figure(go.Bar(
        x=metrics_df["Final PnL"],
        y=metrics_df["Product"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+,.0f}" for v in metrics_df["Final PnL"]],
        textposition="outside",
    ))

    _apply_theme(fig, "🏆 Final PnL by Product")
    fig.update_layout(height=max(250, len(metrics_df) * 60))
    return fig


# ─────────────────────────────────────────────
# 5. Strategy Comparison
# ─────────────────────────────────────────────

def comparison_chart(equity_dict: Dict[str, pd.Series]) -> go.Figure:
    """Overlay multiple equity curves for side-by-side comparison."""
    colours = [
        THEME["accent1"], THEME["accent2"], THEME["accent3"],
        THEME["accent4"], THEME["accent5"], THEME["accent6"],
    ]
    fig = go.Figure()

    for i, (name, eq) in enumerate(equity_dict.items()):
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq,
            name=name,
            line=dict(color=colours[i % len(colours)], width=2),
        ))

    fig.add_hline(y=0, line=dict(color=THEME["subtext"], dash="dash", width=1))
    _apply_theme(fig, "⚖️ Strategy Comparison – Equity Curves")
    fig.update_layout(height=420)
    return fig


# ─────────────────────────────────────────────
# 6. Returns Distribution
# ─────────────────────────────────────────────

def returns_distribution_chart(equity: pd.Series) -> go.Figure:
    """Histogram of per-step returns with normal overlay."""
    returns = equity.diff().dropna()
    if returns.empty:
        return _empty_chart("No returns data")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=60,
        name="Returns",
        marker_color=THEME["accent1"],
        opacity=0.7,
    ))

    # Normal overlay
    mu, sigma = returns.mean(), returns.std()
    if sigma > 0:
        x_range = np.linspace(returns.min(), returns.max(), 200)
        norm_y  = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
        # Scale to histogram height
        scale = len(returns) * (returns.max() - returns.min()) / 60
        fig.add_trace(go.Scatter(
            x=x_range, y=norm_y * scale,
            name="Normal Fit",
            line=dict(color=THEME["accent5"], width=2, dash="dash"),
        ))

    _apply_theme(fig, "📉 Returns Distribution")
    fig.update_layout(height=320, bargap=0.05)
    return fig


# ─────────────────────────────────────────────
# 7. Rolling Sharpe
# ─────────────────────────────────────────────

def rolling_sharpe_chart(sharpe_series: pd.Series, timestamps: pd.Series = None) -> go.Figure:
    x = timestamps if timestamps is not None else sharpe_series.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=sharpe_series,
        name="Rolling Sharpe",
        line=dict(color=THEME["accent4"], width=2),
        fill="tozeroy",
        fillcolor="rgba(210,168,255,0.1)",
    ))
    fig.add_hline(y=0, line=dict(color=THEME["subtext"], dash="dash", width=1))
    fig.add_hline(y=1, line=dict(color=THEME["accent2"], dash="dot", width=1))
    _apply_theme(fig, "📐 Rolling Sharpe Ratio (window=100)")
    fig.update_layout(height=300)
    return fig


# ─────────────────────────────────────────────
# 8. Optimisation heatmap
# ─────────────────────────────────────────────

def optimisation_heatmap(results_df: pd.DataFrame, x_param: str, y_param: str, metric: str = "Final PnL") -> go.Figure:
    """2-D heatmap of grid-search results."""
    try:
        pivot = results_df.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc="mean")
    except Exception:
        return _empty_chart("Could not pivot optimisation results")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale="RdYlGn",
        colorbar=dict(title=metric),
        text=np.round(pivot.values, 1),
        texttemplate="%{text}",
    ))

    _apply_theme(fig, f"🔥 Optimisation Heatmap – {metric}")
    fig.update_layout(height=420)
    return fig


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _empty_chart(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False, font=dict(size=16, color=THEME["subtext"]),
    )
    _apply_theme(fig)
    fig.update_layout(height=300)
    return fig
