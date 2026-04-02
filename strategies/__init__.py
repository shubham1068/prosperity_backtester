"""
strategies/__init__.py
----------------------
Strategy registry. Import all strategies here so the app can discover them.
"""

from .base import BaseStrategy
from .stat_arb import StatArbStrategy
from .market_making import MarketMakingStrategy
from .momentum import MomentumStrategy
from .olivia import OliviaStrategy

# Registry: display_name → class
STRATEGY_REGISTRY = {
    "Statistical Arbitrage (Z-Score)":  StatArbStrategy,
    "Market Making (Spread Capture)":   MarketMakingStrategy,
    "Momentum / Trend Following":        MomentumStrategy,
    "Olivia Adaptive":                   OliviaStrategy,
}

__all__ = [
    "BaseStrategy",
    "StatArbStrategy",
    "MarketMakingStrategy",
    "MomentumStrategy",
    "OliviaStrategy",
    "STRATEGY_REGISTRY",
]
