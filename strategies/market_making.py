"""
strategies/market_making.py
---------------------------
Market-Making strategy: post bids below and asks above mid,
capture the spread while managing inventory risk.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

from .base import BaseStrategy, BacktestResult


class MarketMakingStrategy(BaseStrategy):

    @classmethod
    def param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "SPREAD_FACTOR", "label": "Quote Spread Factor",
                "type": "slider", "min": 0.1, "max": 3.0, "step": 0.05, "default": 0.5,
                "help": "Multiply the observed bid-ask spread by this factor for our quotes.",
            },
            {
                "name": "INV_SKEW", "label": "Inventory Skew Strength",
                "type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.3,
                "help": "How aggressively to skew quotes based on inventory (higher = more aggressive skew).",
            },
            {
                "name": "MAX_INVENTORY", "label": "Max Inventory",
                "type": "number", "min": 5, "max": 200, "step": 5, "default": 40,
                "help": "Maximum position size before stopping new orders.",
            },
            {
                "name": "QUOTE_SIZE", "label": "Quote Size (units)",
                "type": "number", "min": 1, "max": 20, "step": 1, "default": 5,
                "help": "Size of each market-making order.",
            },
            {
                "name": "FILL_PROB", "label": "Fill Probability",
                "type": "slider", "min": 0.05, "max": 0.8, "step": 0.05, "default": 0.3,
                "help": "Simulated probability that a resting order gets filled each step.",
            },
            {
                "name": "EWM_ALPHA", "label": "EWM Alpha (Fair Value)",
                "type": "slider", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.05,
                "help": "Decay factor for the fair-value estimate.",
            },
        ]

    def run(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame, **params) -> BacktestResult:
        SPREAD_F  = params.get("SPREAD_FACTOR", 0.5)
        INV_SKEW  = params.get("INV_SKEW", 0.3)
        MAX_INV   = int(params.get("MAX_INVENTORY", 40))
        Q_SIZE    = int(params.get("QUOTE_SIZE", 5))
        FILL_P    = params.get("FILL_PROB", 0.3)
        EWM_A     = params.get("EWM_ALPHA", 0.05)

        rng = np.random.default_rng(42)
        products = prices_df["product"].unique() if "product" in prices_df.columns else ["ALL"]
        per_product_pnl = {}
        all_equity = pd.Series(dtype=float)
        all_positions = pd.Series(dtype=float)
        all_timestamps = None

        for prod in products:
            df = prices_df[prices_df["product"] == prod].copy() if "product" in prices_df.columns else prices_df.copy()
            df = df.sort_values("timestamp").reset_index(drop=True) if "timestamp" in df.columns else df.reset_index(drop=True)

            mid = self._mid(df).ffill().bfill()
            bid = self._bid(df).ffill().bfill()
            ask = self._ask(df).ffill().bfill()

            fair_value = mid.ewm(alpha=EWM_A).mean()
            nat_spread = (ask - bid).clip(lower=0.5)

            cash       = np.zeros(len(df))
            position   = np.zeros(len(df))
            inventory  = 0
            total_cash = 0.0

            for i in range(1, len(df)):
                # Inventory-based skew
                skew_adj = -inventory / MAX_INV * INV_SKEW * nat_spread.iloc[i]
                our_bid  = fair_value.iloc[i] - nat_spread.iloc[i] * SPREAD_F / 2 + skew_adj
                our_ask  = fair_value.iloc[i] + nat_spread.iloc[i] * SPREAD_F / 2 + skew_adj

                # Simulate fills
                if abs(inventory) < MAX_INV:
                    if rng.random() < FILL_P:
                        # Bid filled → we buy
                        fill_qty = min(Q_SIZE, MAX_INV - abs(min(inventory, 0)))
                        inventory  += fill_qty
                        total_cash -= our_bid * fill_qty
                    if rng.random() < FILL_P:
                        # Ask filled → we sell
                        fill_qty = min(Q_SIZE, MAX_INV - abs(max(inventory, 0)))
                        inventory  -= fill_qty
                        total_cash += our_ask * fill_qty

                position[i] = inventory
                cash[i]     = total_cash

            pos_series = pd.Series(position, index=df.index)
            # MTM P&L = cash + inventory × current mid
            pnl = pd.Series(cash, index=df.index) + pos_series * mid
            pnl = pnl.cumsum() if (pnl.cumsum() != pnl).any() else pnl  # already cumulative
            per_product_pnl[prod] = pnl

            ts = df["timestamp"] if "timestamp" in df.columns else pd.Series(df.index)
            if all_timestamps is None:
                all_timestamps = ts.values

            all_equity    = all_equity.add(pnl, fill_value=0)
            all_positions = all_positions.add(pos_series, fill_value=0)

        all_equity.index    = range(len(all_equity))
        all_positions.index = range(len(all_positions))

        return BacktestResult(
            equity_curve    = all_equity,
            positions       = all_positions,
            trades          = pd.DataFrame(),
            per_product_pnl = per_product_pnl,
            timestamps      = pd.Series(all_timestamps) if all_timestamps is not None else pd.Series(),
            params_used     = params,
            strategy_name   = "Market Making",
        )
