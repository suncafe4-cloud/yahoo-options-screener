#!/usr/bin/env python3
"""
Yahoo Options Premium Screener (via yfinance)

Goal:
- Scan a ticker list using Yahoo options chains (free)
- Identify contracts targeting ~3–6% monthly premium yield (income-style)
- Focus on 30–45 DTE
- Output ranked candidates for:
  - Cash-secured puts (CSP)
  - Covered calls (CC)

Notes:
- Yahoo data may be delayed/missing; script is defensive and skips gracefully.
- "Monthly return" here is premium/price scaled to 30 days:
    monthly_yield = (mid_price / denom) * (30 / DTE)
  where denom is strike for puts, underlying price for calls.

Outputs:
- candidates_csp.csv
- candidates_cc.csv
- summary_top.csv

Usage:
  pip install yfinance pandas
  python yahoo_options_premium_screener.py
  python yahoo_options_premium_screener.py --min_monthly 0.03 --max_monthly 0.06
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import time
from typing import Optional, List, Dict, Tuple

import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency. Run: pip install yfinance pandas") from e


# ----------------------------
# User configuration: Tickers
# ----------------------------
TICKERS: List[str] = [
    # Core
    "AAPL", "AMD", "NVDA", "TSLA", "GOOG", "GOOGL",

    # Expanded list (from your screenshots / earlier)
    "META", "AMZN", "ORCL", "ADBE", "QCOM", "AVGO", "PLTR", "DELL",
    "JPM", "BAC", "C", "GS", "GE", "CAT", "RTX", "HIG", "BA", "CMG", "FDX", "DAL", "BP", "IBM",
    "NFLX", "TGT", "KO", "PEP", "GIS", "F", "DASH",
    "ARKK", "JEPI", "JEPQ", "XYLD", "SQQQ", "VIX",
    "GME", "RIOT", "HOOD", "RIVN", "GPRO", "SSTK", "PRAA", "QMCO", "RDDT", "TEM",

    # Money market funds / mutual funds (no options) — auto-skipped
    "FZDXX", "VMFXX",
]


# ----------------------------
# Screener defaults
# ----------------------------
DEFAULT_DTE_MIN = 30
DEFAULT_DTE_MAX = 45

DEFAULT_MIN_OI = 200            # minimum open interest
DEFAULT_MIN_VOL = 50            # minimum daily option volume
DEFAULT_MAX_SPREAD_PCT = 0.10   # bid/ask spread <= 10% of mid

DEFAULT_MIN_MONTHLY = 0.03      # 3% monthly
DEFAULT_MAX_MONTHLY = 0.06      # 6% monthly

DEFAULT_SLEEP_SEC = 0.35        # throttle between tickers to reduce rate limits


def today_utc_date() -> dt.date:
    return dt.datetime.utcnow().date()


def parse_expiration(exp: str) -> dt.date:
    y, m, d = exp.split("-")
    return dt.date(int(y), int(m), int(d))


def calc_dte(exp: dt.date, asof: dt.date) -> int:
    return (exp - asof).days


def safe_mid(bid: float, ask: float, last: float) -> Optional[float]:
    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0
    if last is not None and last > 0:
        return float(last)
    return None


def spread_pct(bid: float, ask: float, mid: float) -> Optional[float]:
    if mid is None or mid <= 0:
        return None
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    return (ask - bid) / mid


def is_probably_non_optionable(ticker: str) -> bool:
    # Money market / mutual fund style tickers
    return ticker.endswith("XX") or ticker in {"FZDXX", "VMFXX"}


def try_get_earnings_date(t: "yf.Ticker") -> Optional[dt.date]:
    """
    Best-effort attempt to get next earnings date using yfinance.
    Not guaranteed; returns None if unavailable.
    """
    try:
        ed = t.get_earnings_dates(limit=1)
        if isinstance(ed, pd.DataFrame) and len(ed) > 0:
            ts = ed.index[0]
            if hasattr(ts, "date"):
                return ts.date()
    except Exception:
        pass

    try:
        cal = t.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].values
                if len(val) > 0 and pd.notna(val[0]):
                    ts = pd.to_datetime(val[0])
                    return ts.date()
    except Exception:
        pass

    return None


def get_underlying_price(tk: "yf.Ticker") -> Optional[float]:
    # Prefer fast_info
    try:
        fi = getattr(tk, "fast_info", None)
        if fi and fi.get("last_price"):
            p = float(fi["last_price"])
            if p > 0:
                return p
    except Exception:
        pass

    # Fallback: last close
    try:
        hist = tk.history(period="2d", interval="1d")
        if isinstance(hist, pd.DataFrame) and len(hist) > 0:
            p = float(hist["Close"].iloc[-1])
            if p > 0:
                return p
    except Exception:
        pass

    return None


def build_candidates_for_ticker(
    ticker: str,
    dte_min: int,
    dte_max: int,
    min_oi: int,
    min_vol: int,
    max_spread_pct: float,
    min_monthly: float,
    max_monthly: float,
    earnings_block_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (puts_df, calls_df) candidates for ticker.
    """
    if is_probably_non_optionable(ticker):
        return (pd.DataFrame(), pd.DataFrame())

    asof = today_utc_date()
    tk = yf.Ticker(ticker)

    underlying = get_underlying_price(tk)
    if not underlying:
        return (pd.DataFrame(), pd.DataFrame())

    earnings_date = try_get_earnings_date(tk)

    try:
        expirations = tk.options
    except Exception:
        return (pd.DataFrame(), pd.DataFrame())

    put_rows: List[Dict] = []
    call_rows: List[Dict] = []

    for exp_str in expirations:
        exp_date = parse_expiration(exp_str)
        dte = calc_dte(exp_date, asof)
        if dte < dte_min or dte > dte_max:
            continue

        # Conservative earnings avoidance: skip expirations that include earnings inside window (best-effort)
        if earnings_date:
            if 0 <= (earnings_date - asof).days <= dte + earnings_block_days:
                continue

        try:
            chain = tk.option_chain(exp_str)
        except Exception:
            continue

        # -----------------
        # CSP Put candidates
        # -----------------
        puts = chain.puts.copy()
        if isinstance(puts, pd.DataFrame) and not puts.empty:
            for _, r in puts.iterrows():
                strike = float(r.get("strike", math.nan))
                bid = float(r.get("bid", 0.0) or 0.0)
                ask = float(r.get("ask", 0.0) or 0.0)
                last = float(r.get("lastPrice", 0.0) or 0.0)
                oi = int(r.get("openInterest", 0) or 0)
                vol = int(r.get("volume", 0) or 0)

                # CSP wants OTM puts
                if not (strike < underlying):
                    continue

                mid = safe_mid(bid, ask, last)
                if mid is None or mid <= 0:
                    continue

                sp = spread_pct(bid, ask, mid)
                if sp is None:
                    continue

                if oi < min_oi or vol < min_vol:
                    continue
                if sp > max_spread_pct:
                    continue

                monthly = (mid / strike) * (30.0 / dte)
                if monthly < min_monthly or monthly > max_monthly:
                    continue

                put_rows.append({
                    "ticker": ticker,
                    "underlying": underlying,
                    "exp": exp_date,
                    "dte": dte,
                    "strategy": "CSP_PUT",
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "mid": mid,
                    "oi": oi,
                    "vol": vol,
                    "spread_pct": sp,
                    "monthly_yield": monthly,
                    "annualized_yield": monthly * 12.0,
                    "otm_pct": (underlying - strike) / underlying,
                    "earnings_date": earnings_date,
                })

        # --------------------
        # Covered Call candidates
        # --------------------
        calls = chain.calls.copy()
        if isinstance(calls, pd.DataFrame) and not calls.empty:
            for _, r in calls.iterrows():
                strike = float(r.get("strike", math.nan))
                bid = float(r.get("bid", 0.0) or 0.0)
                ask = float(r.get("ask", 0.0) or 0.0)
                last = float(r.get("lastPrice", 0.0) or 0.0)
                oi = int(r.get("openInterest", 0) or 0)
                vol = int(r.get("volume", 0) or 0)

                # CC wants OTM calls
                if not (strike > underlying):
                    continue

                mid = safe_mid(bid, ask, last)
                if mid is None or mid <= 0:
                    continue

                sp = spread_pct(bid, ask, mid)
                if sp is None:
                    continue

                if oi < min_oi or vol < min_vol:
                    continue
                if sp > max_spread_pct:
                    continue

                monthly = (mid / underlying) * (30.0 / dte)
                if monthly < min_monthly or monthly > max_monthly:
                    continue

                call_rows.append({
                    "ticker": ticker,
                    "underlying": underlying,
                    "exp": exp_date,
                    "dte": dte,
                    "strategy": "COVERED_CALL",
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "mid": mid,
                    "oi": oi,
                    "vol": vol,
                    "spread_pct": sp,
                    "monthly_yield": monthly,
                    "annualized_yield": monthly * 12.0,
                    "otm_pct": (strike - underlying) / underlying,
                    "earnings_date": earnings_date,
                })

    return pd.DataFrame(put_rows), pd.DataFrame(call_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dte_min", type=int, default=DEFAULT_DTE_MIN)
    ap.add_argument("--dte_max", type=int, default=DEFAULT_DTE_MAX)
    ap.add_argument("--min_oi", type=int, default=DEFAULT_MIN_OI)
    ap.add_argument("--min_vol", type=int, default=DEFAULT_MIN_VOL)
    ap.add_argument("--max_spread_pct", type=float, default=DEFAULT_MAX_SPREAD_PCT)
    ap.add_argument("--min_monthly", type=float, default=DEFAULT_MIN_MONTHLY)
    ap.add_argument("--max_monthly", type=float, default=DEFAULT_MAX_MONTHLY)
    ap.add_argument("--earnings_block_days", type=int, default=2)
    ap.add_argument("--sleep_sec", type=float, default=DEFAULT_SLEEP_SEC)
    ap.add_argument("--top_n", type=int, default=30)
    args = ap.parse_args()

    all_puts: List[pd.DataFrame] = []
    all_calls: List[pd.DataFrame] = []

    for i, ticker in enumerate(TICKERS, start=1):
        ticker = ticker.strip().upper()
        if not ticker:
            continue

        print(f"[{i}/{len(TICKERS)}] Scanning {ticker} ...")
        try:
            puts_df, calls_df = build_candidates_for_ticker(
                ticker=ticker,
                dte_min=args.dte_min,
                dte_max=args.dte_max,
                min_oi=args.min_oi,
                min_vol=args.min_vol,
                max_spread_pct=args.max_spread_pct,
                min_monthly=args.min_monthly,
                max_monthly=args.max_monthly,
                earnings_block_days=args.earnings_block_days,
            )
        except Exception as e:
            print(f"  - Skipped {ticker} (error): {e}")
            puts_df, calls_df = pd.DataFrame(), pd.DataFrame()

        if not puts_df.empty:
            all_puts.append(puts_df)
        if not calls_df.empty:
            all_calls.append(calls_df)

        time.sleep(args.sleep_sec)

    puts_all = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()
    calls_all = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()

    def rank_df(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.sort_values(
            by=["monthly_yield", "spread_pct", "otm_pct", "oi", "vol"],
            ascending=[False, True, False, False, False],
        )

    puts_ranked = rank_df(puts_all)
    calls_ranked = rank_df(calls_all)

    puts_ranked.to_csv("candidates_csp.csv", index=False)
    calls_ranked.to_csv("candidates_cc.csv", index=False)

    combined = pd.concat([puts_ranked, calls_ranked], ignore_index=True) if (not puts_ranked.empty or not calls_ranked.empty) else pd.DataFrame()
    if not combined.empty:
        combined = combined.sort_values(
            by=["monthly_yield", "spread_pct", "strategy"],
            ascending=[False, True, True],
        )
        combined.head(args.top_n).to_csv("summary_top.csv", index=False)
    else:
        pd.DataFrame().to_csv("summary_top.csv", index=False)

    print("\nDone.")
    print(f" - candidates_csp.csv: {len(puts_ranked)} rows")
    print(f" - candidates_cc.csv : {len(calls_ranked)} rows")
    if not combined.empty:
        print(f" - summary_top.csv   : top {min(args.top_n, len(combined))} rows")
    else:
        print(" - summary_top.csv   : no candidates found within filters.")


if __name__ == "__main__":
    main()
