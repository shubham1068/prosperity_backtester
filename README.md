# 📈 Prosperity Pro Backtester – Advanced Strategy Lab

A production-ready backtesting dashboard for the **IMC Prosperity Algorithmic Trading Challenge**, built with Streamlit.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Smart File Parser** | Auto-detects semicolon CSV, comma CSV, JSON-lines, mixed Prosperity logs |
| **4 Built-in Strategies** | Stat Arb (Z-Score), Market Making, Momentum, Olivia Adaptive |
| **Live Parameter Tuning** | Sliders + number inputs for every parameter, instant re-run |
| **Rich Analytics** | Final PnL, Max Drawdown, Sharpe, Sortino, Calmar, Win Rate, Profit Factor |
| **Interactive Charts** | Equity + Drawdown, Price + Trade Markers, Positions, Per-Product, Returns Distribution, Rolling Sharpe |
| **Strategy Comparison** | Side-by-side equity overlay of any number of runs |
| **Grid-Search Optimizer** | One-click 2D grid search with heatmap visualisation |
| **Export** | Equity CSV, Metrics CSV, Params CSV, Per-Product CSV, Optimisation CSV |
| **Dark UI** | GitHub-dark inspired colour palette throughout |

---

## 📂 Project Structure

```
prosperity_backtester/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── README.md
│
├── strategies/
│   ├── __init__.py           # Strategy registry
│   ├── base.py               # Abstract BaseStrategy
│   ├── stat_arb.py           # Statistical Arbitrage
│   ├── market_making.py      # Market Making
│   ├── momentum.py           # Momentum / Trend
│   └── olivia.py             # Olivia Adaptive
│
└── utils/
    ├── __init__.py
    ├── parser.py             # Intelligent file parser
    ├── metrics.py            # Performance metrics
    └── charts.py             # Plotly chart builders
```

---

## 🚀 Quick Start

### Local Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/prosperity-pro-backtester.git
cd prosperity-pro-backtester

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### Streamlit Cloud Deployment

1. Push the repo to GitHub (all files including `requirements.txt`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy**

> **Note:** Streamlit Cloud automatically installs from `requirements.txt`.

---

## 📊 Supported File Formats

| Format | Example |
|---|---|
| **Prosperity Log** | `Activities log:\n<semicolon CSV>...Trade History:\n<semicolon CSV>` |
| **Semicolon CSV** | `timestamp;product;bid_price_1;...` |
| **Comma CSV** | `timestamp,product,mid_price,...` |
| **JSON Lines** | `{"timestamp":0,"product":"TOMATOES",...}` |

Column names are auto-detected with fuzzy matching (e.g. `bid1` → `bid_price_1`).

---

## ⚙️ Strategy Parameters

### Statistical Arbitrage
| Parameter | Default | Description |
|---|---|---|
| Z_ENTER | 1.5 | Z-score entry threshold |
| Z_STRONG | 2.5 | Strong entry (larger position) |
| EWM_ALPHA | 0.10 | EWM decay factor |
| INV_TARGET_MILD | 10 | Position size for mild signal |
| INV_TARGET_STRONG | 25 | Position size for strong signal |
| NEUT_THRESH | 0.20 | Flatten when |z| < this |
| EMERGENCY_EXIT_THRESH | 5.0 | Flatten when |z| > this |

### Market Making
| Parameter | Default | Description |
|---|---|---|
| SPREAD_FACTOR | 0.5 | Quote spread multiplier |
| INV_SKEW | 0.3 | Inventory skew aggressiveness |
| MAX_INVENTORY | 40 | Hard position limit |
| QUOTE_SIZE | 5 | Size per order |
| FILL_PROB | 0.3 | Simulated fill probability |

### Momentum
| Parameter | Default | Description |
|---|---|---|
| FAST_WINDOW | 10 | Fast EMA period |
| SLOW_WINDOW | 50 | Slow EMA period |
| SIGNAL_THRESH | 0.1% | Min EMA deviation to enter |
| STOP_LOSS_PCT | 1.0% | Stop loss (0 = off) |
| TAKE_PROFIT_PCT | 2.0% | Take profit (0 = off) |

### Olivia Adaptive
| Parameter | Default | Description |
|---|---|---|
| Z_ENTER | 1.2 | Z-score entry |
| OLIVIA_AGGRESSION | 1.0 | Size multiplier |
| SKEW | 0.0 | Directional bias (-1 to +1) |
| ADAPT_WINDOW | 100 | Adaptation lookback |

---

## 🔧 Adding a Custom Strategy

1. Create `strategies/my_strategy.py`:

```python
from .base import BaseStrategy, BacktestResult

class MyStrategy(BaseStrategy):
    
    @classmethod
    def param_schema(cls):
        return [
            {"name": "MY_PARAM", "label": "My Parameter",
             "type": "slider", "min": 0.0, "max": 2.0, "step": 0.1, "default": 1.0,
             "help": "Controls something important."},
        ]
    
    def run(self, prices_df, trades_df, **params):
        mid = self._mid(prices_df)
        # ... your logic ...
        return BacktestResult(equity_curve=pnl, ...)
```

2. Register it in `strategies/__init__.py`:

```python
from .my_strategy import MyStrategy
STRATEGY_REGISTRY["My Strategy Name"] = MyStrategy
```

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
scipy>=1.11.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
kaleido>=0.2.1      # chart image export
```

---

## 📄 License

MIT License. Free to use, modify, and distribute.

---

*Built for the IMC Prosperity Algorithmic Trading Challenge community.*
