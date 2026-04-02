"""
strategies/momentum.py
----------------------
Trend-following / momentum strategy.
Enters in the direction of recent price movement.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

from .base import BaseStrategy, BacktestResult


class MomentumStrategy(BaseStrategy):

    @classmethod
    def param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "FAST_WINDOW", "label": "Fast EMA Window",
                "type": "number", "min": 2, "max": 50, "step": 1, "default": 10,
                "help": "Short-period EMA window for signal generation.",
            },
            {
                "name": "SLOW_WINDOW", "label": "Slow EMA Window",
                "type": "number", "min": 10, "max": 200, "step": 5, "default": 50,
                "help": "Long-period EMA window for trend baseline.",
            },
            {
                "name": "SIGNAL_THRESH", "label": "Signal Threshold (%)",
                "type": "slider", "min": 0.0, "max": 2.0, "step": 0.05, "default": 0.1,
                "help": "Minimum % deviation of fast from slow EMA to enter.",
            },
            {
                "name": "POSITION_SIZE", "label": "Base Position Size",
                "type": "number", "min": 1, "max": 100, "step": 1, "default": 20,
                "help": "Number of units to trade on a signal.",
            },
            {
                "name": "STOP_LOSS_PCT", "label": "Stop Loss (%)",
                "type": "slider", "min": 0.0, "max": 5.0, "step": 0.1, "default": 1.0,
                "help": "Close position if unrealised loss exceeds this % of entry price. 0 = disabled.",
            },
            {
                "name": "TAKE_PROFIT_PCT", "label": "Take Profit (%)",
                "type": "slider", "min": 0.0, "max": 10.0, "step": 0.1, "default": 2.0,
                "help": "Close position when profit reaches this % of entry price. 0 = disabled.",
            },
        ]

    def run(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame, **params) -> BacktestResult:
        FAST      = int(params.get("FAST_WINDOW", 10))
        SLOW      = int(params.get("SLOW_WINDOW", 50))
        SIG_THR   = params.get("SIGNAL_THRESH", 0.1) / 100.0
        POS_SIZE  = int(params.get("POSITION_SIZE", 20))
        STOP      = params.get("STOP_LOSS_PCT", 1.0) / 100.0
        TAKE_P    = params.get("TAKE_PROFIT_PCT", 2.0) / 100.0

        products = prices_df["product"].unique() if "product" in prices_df.columns else ["ALL"]
        per_product_pnl = {}
        all_equity = pd.Series(dtype=float)
        all_positions = pd.Series(dtype=float)
        all_timestamps = None

        for prod in products:
            df = prices_df[prices_df["product"] == prod].copy() if "product" in prices_df.columns else prices_df.copy()
            df = df.sort_values("timestamp").reset_index(drop=True) if "timestamp" in df.columns else df.reset_index(drop=True)

            mid  = self._mid(df).ffill().bfill()
            fast = mid.ewm(span=FAST, adjust=False).mean()
            slow = mid.ewm(span=SLOW, adjust=False).mean()

            position  = np.zeros(len(df))
            pos       = 0
            entry_px  = 0.0

            for i in range(SLOW, len(df)):
                px = mid.iloc[i]
                deviation = (fast.iloc[i] - slow.iloc[i]) / slow.iloc[i]

                # Stop / take-profit on open position
                if pos != 0 and entry_px > 0:
                    pnl_pct = (px - entry_px) / entry_px * np.sign(pos)
                    if STOP > 0 and pnl_pct < -STOP:
                        pos = 0
                    elif TAKE_P > 0 and pnl_pct > TAKE_P:
                        pos = 0

                # Signal entry / flip
                if deviation > SIG_THR and pos <= 0:
                    pos = POS_SIZE
                    entry_px = px
                elif deviation < -SIG_THR and pos >= 0:
                    pos = -POS_SIZE
                    entry_px = px
                elif abs(deviation) < SIG_THR / 2:
                    pos = 0

                position[i] = pos

            pos_series = pd.Series(position, index=df.index)
            pnl = self._pnl_from_positions(mid, pos_series)
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
            strategy_name   = "Momentum",
        )
