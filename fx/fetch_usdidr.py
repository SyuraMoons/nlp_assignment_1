"""Pull the daily USD/IDR exchange rate for the project's date range (Person B's
Task 1 deliverable) and build a gap-free daily series ready to merge with Person A's
data/processed/daily_news_features.csv on `date`.

Source: Yahoo Finance (`USDIDR=X`, via yfinance) -- free, no billing/auth, and covers
the full 2021-2026 range in one call. Forex trades ~24/5 and is closed weekends
(and some holidays), so those dates have no traded close; they are forward-filled
from the last known close (`is_trading_day=False`) so every calendar day in the
range has a value and the merge with the news table never drops a date.

Output (data/processed/):
  usd_idr_raw.parquet   -- raw daily bars as returned by yfinance (trading days only)
  usd_idr_daily.csv     -- date, usd_idr_close, usd_idr_pct_change, is_trading_day
                           (one row per calendar day, weekends/holidays forward-filled)

Run with: python -m fx.fetch_usdidr
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config


def fetch_raw(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise SystemExit(f"yfinance returned no data for {ticker} ({start} to {end})")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": "usd_idr_close"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.reset_index()


def build_daily_series(raw, date_range):
    raw = raw.copy()
    raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")

    all_days = pd.DataFrame({
        "date": pd.date_range(date_range["start"], date_range["end"], freq="D").strftime("%Y-%m-%d")
    })

    daily = all_days.merge(raw, on="date", how="left")
    daily["is_trading_day"] = daily["usd_idr_close"].notna()
    daily["usd_idr_close"] = daily["usd_idr_close"].ffill()
    daily["usd_idr_pct_change"] = daily["usd_idr_close"].pct_change()
    return daily


def main():
    config = load_config()
    fx_config = config["fx"]
    date_range = config["date_range"]
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = fetch_raw(fx_config["ticker"], date_range["start"], date_range["end"])
    raw.to_parquet(raw_dir / "usd_idr_raw.parquet", index=False)

    daily = build_daily_series(raw, date_range)
    out_path = processed_dir / "usd_idr_daily.csv"
    daily.to_csv(out_path, index=False)

    trading_days = int(daily["is_trading_day"].sum())
    print(f"[fetch_usdidr] {len(raw)} raw trading-day bars from yfinance ({fx_config['ticker']})")
    print(f"[fetch_usdidr] wrote {len(daily)} daily rows ({date_range['start']} to {date_range['end']}) to {out_path}")
    print(f"[fetch_usdidr] {trading_days} real trading days, {len(daily) - trading_days} forward-filled (weekends/holidays)")


if __name__ == "__main__":
    main()
