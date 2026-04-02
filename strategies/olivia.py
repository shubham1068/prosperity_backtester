"""
strategies/olivia.py
--------------------
Olivia Adaptive Strategy: combines z-score mean-reversion with
dynamic quote skewing and aggression scaling based on recent fill rates.
Inspired by Prosperity challenge lore around the "Olivia" reference trader.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

from .base import BaseStrategy, BacktestResult


class OliviaStrategy(BaseStrategy):

    @classmethod
    def param_schema(cls) -> List[Dict[str, Any]]:
        return [
            {
                "name": "Z_ENTER", "label": "Z-Score Entry",
                "type": "slider", "min": 0.5, "max": 3.0, "step": 0.1, "default": 1.2,
                "help": "Z-score threshold to initiate a position.",
            },
            {
                "name": "Z_STRONG", "label": "Z-Score Strong",
                "type": "slider", "min": 1.0, "max": 5.0, "step": 0.1, "default": 2.2,
                "help": "Increase aggression beyond this z-score.",
            },
            {
                "name": "EWM_ALPHA", "label": "EWM Alpha",
                "type": "slider", "min": 0.01, "max": 0.5, "step": 0.01, "default": 0.08,
                "help": "Decay factor for EWM statistics.",
            },
            {
                "name": "INV_TARGET_MILD", "label": "Inventory Target (Mild)",
                "type": "number", "min": 1, "max": 50, "step": 1, "default": 8,
                "help": "Position size for mild signals.",
            },
            {
                "name": "INV_TARGET_STRONG", "label": "Inventory Target (Strong)",
                "type": "number", "min": 5, "max": 100, "step": 1, "default": 20,
                "help": "Position size for strong signals.",
            },
            {
                "name": "OLIVIA_AGGRESSION", "label": "Olivia Aggression",
                "type": "slider", "min": 0.0, "max": 2.0, "step": 0.1, "default": 1.0,
                "help": "Multiplier on trade size when mimicking Olivia's aggressive fills.",
            },
            {
                "name": "SKEW", "label": "Quote Skew Bias",
                "type": "slider", "min": -1.0, "max": 1.0, "step": 0.05, "default": 0.0,
                "help": "Static directional bias (-1 = always lean short, +1 = always lean long).",
            },
            {
                "name": "NEUT_THRESH", "label": "Neutralise Threshold",
                "type": "slider", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.15,
                "help": "Flatten position when |z-score| falls below this.",
            },
            {
                "name": "EMERGENCY_EXIT_THRESH", "label": "Emergency Exit Z",
                "type": "slider", "min": 3.0, "max": 8.0, "step": 0.1, "default": 4.5,
                "help": "Immediately flatten if |z-score| exceeds this extreme.",
            },
            {
                "name": "ADAPT_WINDOW", "label": "Adaptation Window",
                "type": "number", "min": 10, "max": 500, "step": 10, "default": 100,
                "help": "Steps over which to measure recent performance and adapt aggression.",
            },
        ]

    def run(self, prices_df: pd.DataFrame, trades_df: pd.DataFrame, **params) -> BacktestResult:
        Z_ENTER    = params.get("Z_ENTER", 1.2)
        Z_STRONG   = params.get("Z_STRONG", 2.2)
        EWM_ALPHA  = params.get("EWM_ALPHA", 0.08)
        INV_MILD   = int(params.get("INV_TARGET_MILD", 8))
        INV_STRONG = int(params.get("INV_TARGET_STRONG", 20))
        AGGRESSION = params.get("OLIVIA_AGGRESSION", 1.0)
        SKEW       = params.get("SKEW", 0.0)
        NEUT       = params.get("NEUT_THRESH", 0.15)
        EMERG      = params.get("EMERGENCY_EXIT_THRESH", 4.5)
        ADAPT_W    = int(params.get("ADAPT_WINDOW", 100))

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

            position   = np.zeros(len(df))
            pnl_buffer = []   # recent step-level pnl for adaptation
            pos = 0

            for i in range(1, len(df)):
                zi = (z.iloc[i] if not np.isnan(z.iloc[i]) else 0.0) + SKEW * 0.5

                # Adaptive aggression: reduce if recent pnl is negative
                if len(pnl_buffer) >= ADAPT_W:
                    recent_pnl = sum(pnl_buffer[-ADAPT_W:])
                    adapt_mult = np.clip(1.0 + recent_pnl / (abs(recent_pnl) + 1e-6) * 0.2, 0.5, 1.5)
                else:
                    adapt_mult = 1.0

                eff_mild   = max(1, int(INV_MILD   * AGGRESSION * adapt_mult))
                eff_strong = max(1, int(INV_STRONG * AGGRESSION * adapt_mult))

                # Emergency exit
                if abs(zi) > EMERG:
                    pos = 0
                elif abs(zi) < NEUT:
                    pos = 0
                elif zi < -Z_STRONG:
                    pos = eff_strong
                elif zi > Z_STRONG:
                    pos = -eff_strong
                elif zi < -Z_ENTER:
                    pos = eff_mild
                elif zi > Z_ENTER:
                    pos = -eff_mild

                position[i] = pos

                # Track step pnl for adaptation
                if i > 0:
                    step_pnl = position[i - 1] * (mid.iloc[i] - mid.iloc[i - 1])
                    pnl_buffer.append(step_pnl)

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
            strategy_name   = "Olivia Adaptive",
        )
