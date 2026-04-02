"""
utils/metrics.py
----------------
Performance metric calculators for backtest results.
All functions accept a pandas Series of equity/PnL values.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_all_metrics(equity: pd.Series, trades_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Compute full suite of performance metrics.
    """

    metrics = {}

    if equity.empty or equity.isna().all():
        return _empty_metrics()

    equity = equity.dropna().reset_index(drop=True)

    # ✅ FIX 1: Correct returns (percentage-based)
    returns = equity.pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    # ── Core P&L ──────────────────────────────
    metrics["Final PnL"]  = round(float(equity.iloc[-1]), 2)
    metrics["Peak PnL"]   = round(float(equity.max()), 2)
    metrics["Trough PnL"] = round(float(equity.min()), 2)
    metrics["PnL Range"]  = round(metrics["Peak PnL"] - metrics["Trough PnL"], 2)

    # ── Drawdown (FIXED: % based) ────────────
    roll_max = equity.cummax()

    # Avoid division by zero
    roll_max_safe = roll_max.replace(0, np.nan)

    drawdown = (equity - roll_max_safe) / roll_max_safe
    drawdown = drawdown.fillna(0)

    metrics["Max Drawdown"] = round(float(drawdown.min()) * 100, 2)
    metrics["Avg Drawdown"] = round(float(drawdown[drawdown < 0].mean()) * 100 if (drawdown < 0).any() else 0.0, 2)

    # Drawdown duration
    in_dd = (drawdown < 0).astype(int)
    metrics["Max DD Duration"] = int(_longest_streak(in_dd))

    # ── Returns statistics ────────────────────
    metrics["Total Steps"]     = len(equity)
    metrics["Avg Step Return"] = round(float(returns.mean()), 6)
    metrics["Return Std"]      = round(float(returns.std()), 6)
    metrics["Skewness"]        = round(float(returns.skew()), 3)
    metrics["Kurtosis"]        = round(float(returns.kurtosis()), 3)

    # ── Risk-adjusted metrics (FIXED) ─────────
    std = returns.std()

    if std > 1e-8:
        metrics["Sharpe Ratio"] = round(float(returns.mean() / std * np.sqrt(252)), 3)
    else:
        metrics["Sharpe Ratio"] = 0.0

    downside = returns[returns < 0].std()

    if downside > 1e-8:
        metrics["Sortino Ratio"] = round(float(returns.mean() / downside * np.sqrt(252)), 3)
    else:
        metrics["Sortino Ratio"] = 0.0

    # Calmar Ratio
    if metrics["Max Drawdown"] != 0:
        metrics["Calmar Ratio"] = round(
            float(metrics["Final PnL"] / abs(metrics["Max Drawdown"])),
            3
        )
    else:
        metrics["Calmar Ratio"] = 0.0

    # ── Trade-level metrics ───────────────────
    if trades_df is not None and not trades_df.empty and "pnl" in trades_df.columns:
        t = trades_df["pnl"].dropna()

        metrics["Total Trades"]  = int(len(t))
        metrics["Win Rate"]      = round(float((t > 0).mean() * 100), 2)
        metrics["Avg Trade PnL"] = round(float(t.mean()), 2)
        metrics["Best Trade"]    = round(float(t.max()), 2)
        metrics["Worst Trade"]   = round(float(t.min()), 2)
        metrics["Profit Factor"] = _profit_factor(t)
    else:
        metrics["Total Trades"]  = 0
        metrics["Win Rate"]      = 0
        metrics["Avg Trade PnL"] = 0
        metrics["Best Trade"]    = 0
        metrics["Worst Trade"]   = 0
        metrics["Profit Factor"] = 0

    return metrics


def compute_drawdown_series(equity: pd.Series) -> pd.Series:
    """Return drawdown series in percentage."""
    if equity.empty:
        return pd.Series(dtype=float)

    roll_max = equity.cummax().replace(0, np.nan)
    dd = (equity - roll_max) / roll_max
    return dd.fillna(0)


def rolling_sharpe(equity: pd.Series, window: int = 100) -> pd.Series:
    """Rolling Sharpe ratio."""
    returns = equity.pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    roll_mean = returns.rolling(window).mean()
    roll_std  = returns.rolling(window).std()

    return (roll_mean / roll_std.replace(0, np.nan) * np.sqrt(252)).fillna(0)


def per_product_metrics(equity_by_product: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for prod, eq in equity_by_product.items():
        m = compute_all_metrics(eq)
        rows.append({
            "Product":       prod,
            "Final PnL":     m.get("Final PnL", 0),
            "Max Drawdown":  m.get("Max Drawdown", 0),
            "Sharpe":        m.get("Sharpe Ratio", 0),
            "Sortino":       m.get("Sortino Ratio", 0),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _empty_metrics() -> Dict[str, Any]:
    keys = [
        "Final PnL", "Peak PnL", "Trough PnL", "PnL Range",
        "Max Drawdown", "Avg Drawdown", "Max DD Duration",
        "Total Steps", "Avg Step Return", "Return Std",
        "Skewness", "Kurtosis",
        "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
        "Total Trades", "Win Rate", "Avg Trade PnL",
        "Best Trade", "Worst Trade", "Profit Factor",
    ]
    return {k: 0 for k in keys}


def _longest_streak(binary_series: pd.Series) -> int:
    max_streak = cur = 0
    for v in binary_series:
        if v:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return max_streak


def _profit_factor(trade_pnl: pd.Series) -> float:
    gross_profit = trade_pnl[trade_pnl > 0].sum()
    gross_loss   = abs(trade_pnl[trade_pnl < 0].sum())

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return round(float(gross_profit / gross_loss), 3)
