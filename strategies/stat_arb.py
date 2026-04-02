"""
strategies/stat_arb.py
----------------------
Statistical Arbitrage strategy using EWM z-score signals.
Enters when price deviates significantly from its moving average.
Uses tiered entry thresholds and inventory targets.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

from .base import BaseStrategy, BacktestResult


class StatArbStrategy(BaseStrategy):

    @classmethod
    def param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Z_ENTER", "label": "Z-Score Entry Threshold",
                "type": "slider", "min": 0.5, "max": 3.0, "step": 0.1, "default": 1.5,
                "help": "Enter a position when |z-score| exceeds this value.",
            },
            {
                "name": "Z_STRONG", "label": "Z-Score Strong Entry",
                "type": "slider", "min": 1.0, "max": 5.0, "step": 0.1, "default": 2.5,
                "help": "Add aggressively when |z-score| exceeds this stronger threshold.",
            },
            {
                "name": "EWM_ALPHA", "label": "EWM Alpha",
                "type": "slider", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.1,
                "help": "Decay factor for the exponentially weighted moving statistics.",
            },
            {
                "name": "INV_TARGET_MILD", "label": "Inventory Target (Mild)",
                "type": "number", "min": 1, "max": 50, "step": 1, "default": 10,
                "help": "Target position size for a mild z-score signal.",
            },
            {
                "name": "INV_TARGET_STRONG", "label": "Inventory Target (Strong)",
                "type": "number", "min": 5, "max": 100, "step": 1, "default": 25,
                "help": "Target position size for a strong z-score signal.",
            },
            {
                "name": "NEUT_THRESH", "label": "Neutralise Threshold",
                "type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.2,
                "help": "Close position when |z-score| falls below this value.",
            },
            {
                "name": "EMERGENCY_EXIT_THRESH", "label": "Emergency Exit Z",
                "type": "slider", "min": 3.0, "max": 8.0, "step": 0.1, "default": 5.0,
                "help": "Immediately flatten position if |z-score| exceeds this extreme value.",
            },
            {
                "name": "MAX_POSITION", "label": "Max Absolute Position",
                "type": "number", "min": 5, "max": 200, "step": 5, "default": 50,
                "help": "Hard position limit.",
            },
        ]

    def run(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame, **params) -> BacktestResult:
        Z_ENTER      = params.get("Z_ENTER", 1.5)
        Z_STRONG     = params.get("Z_STRONG", 2.5)
        EWM_ALPHA    = params.get("EWM_ALPHA", 0.1)
        INV_MILD     = int(params.get("INV_TARGET_MILD", 10))
        INV_STRONG   = int(params.get("INV_TARGET_STRONG", 25))
        NEUT         = params.get("NEUT_THRESH", 0.2)
        EMERG        = params.get("EMERGENCY_EXIT_THRESH", 5.0)
        MAX_POS      = int(params.get("MAX_POSITION", 50))

        products = prices_df["product"].unique() if "product" in prices_df.columns else ["ALL"]
        per_product_pnl = {}
        all_equity = pd.Series(dtype=float)
        all_positions = pd.Series(dtype=float)
        all_timestamps = None

        for prod in products:
            df = prices_df[prices_df["product"] == prod].copy() if "product" in prices_df.columns else prices_df.copy()
            df = df.sort_values("timestamp").reset_index(drop=True) if "timestamp" in df.columns else df.reset_index(drop=True)

            mid = self._mid(df).ffill().bfill()
            z   = self.ewm_zscore(mid, alpha=EWM_ALPHA)

            position = np.zeros(len(df))
            pos = 0

            for i in range(1, len(df)):
                zi = z.iloc[i] if not np.isnan(z.iloc[i]) else 0.0

                # Emergency exit
                if abs(zi) > EMERG:
                    pos = 0
                # Neutralise
                elif abs(zi) < NEUT:
                    pos = 0
                # Strong signal
                elif zi < -Z_STRONG:
                    pos = min(INV_STRONG, MAX_POS)      # long
                elif zi > Z_STRONG:
                    pos = max(-INV_STRONG, -MAX_POS)    # short
                # Mild signal
                elif zi < -Z_ENTER:
                    pos = min(INV_MILD, MAX_POS)
                elif zi > Z_ENTER:
                    pos = max(-INV_MILD, -MAX_POS)

                position[i] = pos

            pos_series = pd.Series(position, index=df.index)
            pnl = self._pnl_from_positions(mid, pos_series)
            per_product_pnl[prod] = pnl

            ts = df["timestamp"] if "timestamp" in df.columns else pd.Series(df.index)
            if all_timestamps is None:
                all_timestamps = ts.values

            all_equity = all_equity.add(pnl, fill_value=0)
            all_positions = all_positions.add(pos_series, fill_value=0)

        all_equity.index = range(len(all_equity))
        all_positions.index = range(len(all_positions))

        return BacktestResult(
            equity_curve    = all_equity,
            positions       = all_positions,
            trades          = pd.DataFrame(),
            per_product_pnl = per_product_pnl,
            timestamps      = pd.Series(all_timestamps) if all_timestamps is not None else pd.Series(),
            params_used     = params,
            strategy_name   = "Statistical Arbitrage",
        )
