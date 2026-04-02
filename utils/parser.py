"""
utils/parser.py
---------------
Intelligent file parser for Prosperity trading logs.
Handles CSV (comma/semicolon), JSON-lines, and mixed-format logs.
"""

import io
import json
import re
import pandas as pd
import numpy as np
from typing import Tuple, Optional


# ─────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────

def parse_uploaded_file(file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Main entry point.
    Returns (prices_df, trades_df, detected_format_str).
    Both DataFrames may be empty if the section is absent.
    """
    text = _decode(file_bytes)

    # Try each format in priority order
    for fmt, fn in [
        ("Prosperity Log (semicolon CSV)",  _parse_prosperity_log),
        ("JSON Lines",                       _parse_jsonl),
        ("Comma-separated CSV",              _parse_csv_comma),
        ("Semicolon-separated CSV",          _parse_csv_semicolon),
        ("Generic / fallback",               _parse_generic),
    ]:
        try:
            prices, trades = fn(text)
            if prices is not None and not prices.empty:
                prices  = _normalise_prices(prices)
                trades  = _normalise_trades(trades) if trades is not None else pd.DataFrame()
                return prices, trades, fmt
        except Exception:
            continue

    return pd.DataFrame(), pd.DataFrame(), "Unknown – could not parse"


# ─────────────────────────────────────────────
# Format-specific parsers
# ─────────────────────────────────────────────

def _parse_prosperity_log(text: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Prosperity logs are divided by section headers like:
        Sandbox logs:
        Activities log:
        Trade History:
    Each section is a semicolon-delimited CSV block.
    """
    sections = _split_prosperity_sections(text)

    prices_df = pd.DataFrame()
    trades_df = pd.DataFrame()

    # Activities log → price data
    if "Activities log" in sections:
        prices_df = _read_semicolon_block(sections["Activities log"])

    # Trade History → trades
    if "Trade History" in sections:
        trades_df = _read_semicolon_block(sections["Trade History"])

    if prices_df.empty:
        raise ValueError("No price data found in Prosperity log")

    return prices_df, trades_df


def _parse_jsonl(text: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Parse JSON-lines format."""
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not records:
        raise ValueError("No JSON objects found")

    df = pd.json_normalize(records)
    return df, None


def _parse_csv_comma(text: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df = pd.read_csv(io.StringIO(text), sep=",", engine="python", on_bad_lines="skip")
    if df.empty or len(df.columns) < 2:
        raise ValueError("Not a valid comma CSV")
    return df, None


def _parse_csv_semicolon(text: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df = pd.read_csv(io.StringIO(text), sep=";", engine="python", on_bad_lines="skip")
    if df.empty or len(df.columns) < 2:
        raise ValueError("Not a valid semicolon CSV")
    return df, None


def _parse_generic(text: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Last-resort: try to sniff separator."""
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python", on_bad_lines="skip")
    if df.empty:
        raise ValueError("Could not parse generically")
    return df, None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _decode(file_bytes: bytes) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _split_prosperity_sections(text: str) -> dict:
    """
    Split a Prosperity log into named sections.
    Section headers end with a colon on their own line.
    """
    sections = {}
    current_name = None
    current_lines = []

    # Common section headers
    HEADERS = re.compile(
        r"^(Sandbox logs|Activities log|Trade History|"
        r"Profit and loss|Lambda log|[A-Z][a-z]+ log)\s*:?\s*$",
        re.IGNORECASE,
    )

    for line in text.splitlines():
        m = HEADERS.match(line.strip())
        if m:
            if current_name:
                sections[current_name] = "\n".join(current_lines)
            current_name = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_name:
        sections[current_name] = "\n".join(current_lines)

    return sections


def _read_semicolon_block(block: str) -> pd.DataFrame:
    """Parse a semicolon-delimited CSV text block."""
    lines = [l for l in block.splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()
    text = "\n".join(lines)
    try:
        return pd.read_csv(io.StringIO(text), sep=";", engine="python", on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Normalisation – map arbitrary column names
# ─────────────────────────────────────────────

# Ordered candidate column names → canonical name
_PRICE_COL_MAP = {
    "timestamp":   ["timestamp", "time", "ts", "day"],
    "product":     ["product", "symbol", "instrument", "asset"],
    "bid_price_1": ["bid_price_1", "bid1", "bidprice1", "bid_price"],
    "ask_price_1": ["ask_price_1", "ask1", "askprice1", "ask_price"],
    "mid_price":   ["mid_price", "mid", "price", "last", "close"],
    "profit_and_loss": ["profit_and_loss", "pnl", "p&l", "profit"],
}

_TRADE_COL_MAP = {
    "timestamp": ["timestamp", "time", "ts"],
    "product":   ["product", "symbol", "instrument"],
    "price":     ["price", "trade_price", "fill_price"],
    "quantity":  ["quantity", "qty", "size", "volume", "amount"],
    "side":      ["side", "direction", "type", "buyer", "seller"],
}


def _normalise_prices(df: pd.DataFrame) -> pd.DataFrame:
    return _remap_columns(df, _PRICE_COL_MAP)


def _normalise_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return _remap_columns(df, _TRADE_COL_MAP)


def _remap_columns(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Rename df columns to canonical names using fuzzy matching.
    Unknown columns are kept as-is.
    """
    rename = {}
    lower_cols = {c.lower().replace(" ", "_").replace(".", "_"): c for c in df.columns}

    for canonical, candidates in col_map.items():
        if canonical in df.columns:
            continue  # already correct
        for cand in candidates:
            if cand in lower_cols:
                rename[lower_cols[cand]] = canonical
                break

    return df.rename(columns=rename)


# ─────────────────────────────────────────────
# Synthetic data generator (for demo / testing)
# ─────────────────────────────────────────────

def generate_demo_data(products=("TOMATOES", "COCONUTS"), n_steps=1000, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic-looking synthetic price + trade data."""
    rng = np.random.default_rng(seed)
    rows = []
    trade_rows = []

    for day in range(3):
        for step in range(n_steps):
            ts = day * n_steps + step
            for prod in products:
                base = 5000 if prod == "TOMATOES" else 8000
                noise = rng.normal(0, 5)
                mid = base + noise + rng.normal(0, 2) * step / n_steps * 20

                spread = rng.uniform(5, 15)
                bid = mid - spread / 2
                ask = mid + spread / 2
                vol1 = int(rng.integers(5, 15))
                vol2 = int(rng.integers(10, 25))

                rows.append({
                    "day": day - 2,
                    "timestamp": ts,
                    "product": prod,
                    "bid_price_1": round(bid),
                    "bid_volume_1": vol1,
                    "bid_price_2": round(bid - spread),
                    "bid_volume_2": vol2,
                    "ask_price_1": round(ask),
                    "ask_volume_1": vol1,
                    "ask_price_2": round(ask + spread),
                    "ask_volume_2": vol2,
                    "mid_price": round(mid, 2),
                    "profit_and_loss": round(rng.normal(500, 300) * (ts / n_steps), 2),
                })

                # Occasional trades
                if rng.random() < 0.05:
                    trade_rows.append({
                        "timestamp": ts,
                        "product": prod,
                        "price": round(mid + rng.choice([-1, 1]) * spread / 2, 2),
                        "quantity": int(rng.integers(1, 10)),
                        "side": rng.choice(["BUY", "SELL"]),
                    })

    return pd.DataFrame(rows), pd.DataFrame(trade_rows)
