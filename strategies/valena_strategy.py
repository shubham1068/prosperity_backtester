# ─────────────────────────────────────────────────────────────
# CLASS 3c : ValenaStrategy  (Gap-Up / Gap-Down on Monthly Expiry)
# ─────────────────────────────────────────────────────────────
class ValenaStrategy(_BaseStrategy):
    """
    Valena Strategy — Gap-Based ATM Option Buy
    -------------------------------------------
    Logic  : On every monthly expiry day, check the opening gap:
               Gap-Up   (Open > Prev Close)  →  Buy ATM CALL
               Gap-Down (Open < Prev Close)  →  Buy ATM PUT
               No Gap   (Open == Prev Close) →  No Trade (skipped)

    Gap threshold (default 0.1%) filters out near-flat opens.

    Entry  : Monthly expiry day open price premium
    Exit   : Next monthly expiry day (hold one full cycle)
    Lot    : 50 qty (1 Nifty lot)
    """

    name = "Valena Gap Strategy"

    def __init__(self, gap_threshold_pct: float = 0.1):
        """
        Parameters
        ----------
        gap_threshold_pct : float
            Minimum gap size in % to trigger a trade.
            Default 0.1% — avoids flat/noise opens.
        """
        self.gap_threshold = gap_threshold_pct / 100.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"[{self.name}] Generating signals …")

        # Compute previous day's close (shifted by 1 row)
        df = df.copy()
        df["Prev_Close"] = df["Close"].shift(1)
        df["Gap_Pct"] = (df["Open"] - df["Prev_Close"]) / df["Prev_Close"]

        trades = []
        expiry_dates = df[df["Monthly_Expiry"]].index.tolist()

        for i, entry_date in enumerate(expiry_dates[:-1]):
            exit_date = expiry_dates[i + 1]
            row = df.loc[entry_date]

            gap_pct = row["Gap_Pct"]

            # Skip if gap is too small (flat/noise open)
            if abs(gap_pct) < self.gap_threshold:
                continue

            # Signal: Gap-Up → CALL, Gap-Down → PUT
            if gap_pct > 0:
                signal = 1
                option = "CALL"
            else:
                signal = -1
                option = "PUT"

            close  = row["Close"]
            strike = row["ATM_Strike"]
            dte    = (exit_date - entry_date).days

            entry_prem = self.estimate_premium(close, row["ATR_14"], dte)
            exit_close = df.loc[exit_date, "Close"]
            exit_prem  = self._exit_premium(
                close, exit_close, signal,
                entry_prem, df.loc[exit_date, "ATR_14"]
            )

            pnl     = (exit_prem - entry_prem) * 50   # 1 lot = 50 qty
            pnl_pct = (exit_prem - entry_prem) / entry_prem * 100

            trades.append({
                "strategy"    : self.name,
                "entry_date"  : entry_date,
                "exit_date"   : exit_date,
                "signal"      : option,
                "gap_pct"     : round(gap_pct * 100, 3),   # in %
                "strike"      : int(strike),
                "entry_spot"  : round(close, 2),
                "entry_prem"  : round(entry_prem, 2),
                "exit_prem"   : round(exit_prem, 2),
                "pnl"         : round(pnl, 2),
                "pnl_pct"     : round(pnl_pct, 2),
                "dte"         : dte,
            })

        return pd.DataFrame(trades)

    # ── private ─────────────────────────────────────────────
    @staticmethod
    def _exit_premium(entry_spot, exit_spot, signal, entry_prem, exit_atr):
        """Estimate exit premium based on directional spot move."""
        move   = (exit_spot - entry_spot) * signal   # positive = favourable
        profit = move * 0.6                           # delta ≈ 0.5 ATM, decayed
        return max(entry_prem + profit, 0)
