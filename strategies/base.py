"""
strategies/base.py
------------------
Abstract base class all strategies must inherit from.
Defines the required interface and shared helpers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    """Container for a single backtest run's output."""
    equity_curve:     pd.Series        = field(default_factory=pd.Series)
    positions:        pd.Series        = field(default_factory=pd.Series)
    trades:           pd.DataFrame     = field(default_factory=pd.DataFrame)
    per_product_pnl:  Dict[str, pd.Series] = field(default_factory=dict)
    timestamps:       pd.Series        = field(default_factory=pd.Series)
    params_used:      Dict[str, Any]   = field(default_factory=dict)
    strategy_name:    str              = ""


class BaseStrategy(ABC):
    """
    All strategies must:
      1. Declare their parameter schema via `param_schema()`
      2. Implement `run(prices_df, trades_df, **params)` returning BacktestResult
    """

    # ── Abstract interface ────────────────────

    @classmethod
    @abstractmethod
    def param_schema(cls) -> List[Dict[str, Any]]:
        """
        Return a list of parameter descriptor dicts:
        {
          "name":    str,           # parameter key
          "label":   str,           # display label
          "type":    "slider" | "number" | "select",
          "min":     float,         # for slider/number
          "max":     float,
          "step":    float,
          "default": Any,
          "help":    str,           # tooltip text
          "options": list,          # for select type
        }
        """

    @abstractmethod
    def run(
        self,
        prices_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        **params,
    ) -> BacktestResult:
        """Execute the strategy and return a BacktestResult."""

    # ── Shared helpers ────────────────────────

    @staticmethod
    def ewm_zscore(series: pd.Series, alpha: float, min_periods: int = 20) -> pd.Series:
        """Exponentially weighted moving z-score."""
        ewm_mean = series.ewm(alpha=alpha, min_periods=min_periods).mean()
        ewm_std  = series.ewm(alpha=alpha, min_periods=min_periods).std()
        return (series - ewm_mean) / ewm_std.replace(0, np.nan)

    @staticmethod
    def _mid(df: pd.DataFrame) -> pd.Series:
        """Best available mid-price estimate."""
        if "mid_price" in df.columns:
            return df["mid_price"]
        if "bid_price_1" in df.columns and "ask_price_1" in df.columns:
            return (df["bid_price_1"] + df["ask_price_1"]) / 2
        for col in ("price", "close", "last"):
            if col in df.columns:
                return df[col]
        raise ValueError("Cannot determine mid price – no suitable column found")

    @staticmethod
    def _bid(df: pd.DataFrame) -> pd.Series:
        for col in ("bid_price_1", "bid", "bid_price"):
            if col in df.columns:
                return df[col]
        return BaseStrategy._mid(df) - 1

    @staticmethod
    def _ask(df: pd.DataFrame) -> pd.Series:
        for col in ("ask_price_1", "ask", "ask_price"):
            if col in df.columns:
                return df[col]
        return BaseStrategy._mid(df) + 1

    @staticmethod
    def _pnl_from_positions(mid: pd.Series, position: pd.Series) -> pd.Series:
        """Compute mark-to-market P&L from position series."""
        price_change = mid.diff().fillna(0)
        return (position.shift(1).fillna(0) * price_change).cumsum()

    @staticmethod
    def _trade_log(timestamps, prices, quantities, sides) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": timestamps,
            "price":     prices,
            "quantity":  quantities,
            "side":      sides,
            "pnl":       np.nan,  # filled in post-run
        })
