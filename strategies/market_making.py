"""
strategies/market_making.py
---------------------------
Fixed Market-Making Strategy
✔ Correct PnL calculation
✔ Trade logging added
✔ No double counting
✔ Stable realistic results
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
            },
            {
                "name": "INV_SKEW", "label": "Inventory Skew Strength",
                "type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.3,
            },
            {
                "name": "MAX_INVENTORY", "label": "Max Inventory",
                "type": "number", "min": 5, "max": 200, "step": 5, "default": 40,
            },
            {
                "name": "QUOTE_SIZE", "label": "Quote Size",
                "type": "number", "min": 1, "max": 20, "step": 1, "default": 5,
            },
            {
                "name": "FILL_PROB", "label": "Fill Probability",
                "type": "slider", "min": 0.05, "max": 0.8, "step": 0.05, "default": 0.3,
            },
            {
                "name": "EWM_ALPHA", "label": "EWM Alpha",
                "type": "slider", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.05,
            },
        ]

    def run(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame, **params) -> BacktestResult:

        SPREAD_F = params.get("SPREAD_FACTOR", 0.5)
        INV_SKEW = params.get("INV_SKEW", 0.3)
        MAX_INV  = int(params.get("MAX_INVENTORY", 40))
        Q_SIZE   = int(params.get("QUOTE_SIZE", 5))
        FILL_P   = params.get("FILL_PROB", 0.3)
        EWM_A    = params.get("EWM_ALPHA", 0.05)

        rng = np.random.default_rng(42)

        products = prices_df["product"].unique() if "product" in prices_df.columns else ["ALL"]

        per_product_pnl = {}
        all_equity = None
        all_positions = None
        all_trades = []
        all_timestamps = None

        for prod in products:

            df = prices_df[prices_df["product"] == prod].copy() if "product" in prices_df.columns else prices_df.copy()

            df = df.sort_values("timestamp").reset_index(drop=True) if "timestamp" in df.columns else df.reset_index(drop=True)

            if df.empty:
                continue

            mid = self._mid(df).ffill().bfill()
            bid = self._bid(df).ffill().bfill()
            ask = self._ask(df).ffill().bfill()

            fair = mid.ewm(alpha=EWM_A).mean()
            spread = (ask - bid).clip(lower=0.5)

            inventory = 0
            cash = 0.0

            pos_list = []
            pnl_list = []
            trade_list = []

            for i in range(len(df)):

                skew = -inventory / MAX_INV * INV_SKEW * spread.iloc[i]

                our_bid = fair.iloc[i] - spread.iloc[i] * SPREAD_F / 2 + skew
                our_ask = fair.iloc[i] + spread.iloc[i] * SPREAD_F / 2 + skew

                # ---- BUY fill ----
                if abs(inventory) < MAX_INV and rng.random() < FILL_P:
                    qty = min(Q_SIZE, MAX_INV - abs(min(inventory, 0)))
                    inventory += qty
                    cash -= our_bid * qty

                    trade_list.append({
                        "timestamp": df["timestamp"].iloc[i] if "timestamp" in df.columns else i,
                        "price": our_bid,
                        "quantity": qty,
                        "side": "BUY",
                        "pnl": 0
                    })

                # ---- SELL fill ----
                if abs(inventory) < MAX_INV and rng.random() < FILL_P:
                    qty = min(Q_SIZE, MAX_INV - abs(max(inventory, 0)))
                    inventory -= qty
                    cash += our_ask * qty

                    trade_list.append({
                        "timestamp": df["timestamp"].iloc[i] if "timestamp" in df.columns else i,
                        "price": our_ask,
                        "quantity": qty,
                        "side": "SELL",
                        "pnl": 0
                    })

                # ---- MARK TO MARKET PnL ----
                pnl = cash + inventory * mid.iloc[i]

                pos_list.append(inventory)
                pnl_list.append(pnl)

            pnl_series = pd.Series(pnl_list)
            pnl_series = pnl_series - pnl_series.iloc[0]  # normalize

            pos_series = pd.Series(pos_list)

            per_product_pnl[prod] = pnl_series

            if all_equity is None:
                all_equity = pnl_series.copy()
                all_positions = pos_series.copy()
            else:
                all_equity = all_equity.add(pnl_series, fill_value=0)
                all_positions = all_positions.add(pos_series, fill_value=0)

            all_trades.extend(trade_list)

            if all_timestamps is None:
                all_timestamps = df["timestamp"] if "timestamp" in df.columns else pd.Series(range(len(df)))

        return BacktestResult(
            equity_curve=all_equity if all_equity is not None else pd.Series(),
            positions=all_positions if all_positions is not None else pd.Series(),
            trades=pd.DataFrame(all_trades),
            per_product_pnl=per_product_pnl,
            timestamps=pd.Series(all_timestamps) if all_timestamps is not None else pd.Series(),
            params_used=params,
            strategy_name="Market Making (Fixed)",
        )
