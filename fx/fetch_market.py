"""Phase 3 market-data modality: pull daily closes for a handful of USD-strength,
risk-sentiment, commodity, and regional-peer-FX tickers via yfinance, alongside the
existing USD/IDR series from fx/fetch_usdidr.py.

These give the models a numeric time-series signal to compare news features against --
the Phase 3 hypothesis is "does news add anything on top of what market data already
shows", not "does news predict FX in a vacuum".

Leakage handling: some tickers (DXY, oil, gold, VIX, US10Y) trade on US/London
exchanges that close hours after Jakarta's trading day ends. Their same-day close is
not actually observable yet when a same-day WIB news feature is. Every ticker marked
`lag_utc_close: true` in config.yaml -> market.tickers is shifted forward one calendar
day here, so the value stored under date `t` is what was actually known by the end of
WIB day `t` (i.e. the close from day `t-1`). IHSG and the regional FX pairs trade
during the Asian day and are left unshifted.

Output (data/processed/):
  market_daily.csv -- one row per calendar day, columns
    <name>_close, <name>_pct_change, <name>_is_trading_day  per configured ticker

Run with: python -m fx.fetch_market
"""
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config


def fetch_one(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        print(f"[fetch_market] WARNING: no data for {symbol} ({start} to {end}), skipping")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.reset_index()


def build_daily_series(raw, date_range, name, lag_utc_close):
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    if lag_utc_close:
        # Shift the observation date forward one day: a close recorded for UTC day d
        # was not yet knowable at the end of WIB day d (WIB is UTC+7, and these
        # exchanges close hours after Jakarta's trading day is already over).
        raw["date"] = raw["date"] + pd.Timedelta(days=1)
    raw["date"] = raw["date"].dt.strftime("%Y-%m-%d")
    raw = raw.groupby("date", as_index=False)["close"].last()  # dedupe if the shift collided

    all_days = pd.DataFrame({
        "date": pd.date_range(date_range["start"], date_range["end"], freq="D").strftime("%Y-%m-%d")
    })
    daily = all_days.merge(raw, on="date", how="left")
    daily[f"{name}_is_trading_day"] = daily["close"].notna()
    daily["close"] = daily["close"].ffill()
    daily[f"{name}_close"] = daily["close"]
    daily[f"{name}_pct_change"] = daily["close"].pct_change()
    return daily[["date", f"{name}_close", f"{name}_pct_change", f"{name}_is_trading_day"]]


def main():
    config = load_config()
    date_range = config["date_range"]
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    tickers = config["market"]["tickers"]

    all_days = pd.DataFrame({
        "date": pd.date_range(date_range["start"], date_range["end"], freq="D").strftime("%Y-%m-%d")
    })
    merged = all_days.copy()

    for name, spec in tickers.items():
        raw = fetch_one(spec["symbol"], date_range["start"], date_range["end"])
        if raw is None:
            continue
        raw.to_parquet(raw_dir / f"market_{name}_raw.parquet", index=False)
        daily = build_daily_series(raw, date_range, name, spec.get("lag_utc_close", False))
        merged = merged.merge(daily, on="date", how="left")
        trading_days = int(daily[f"{name}_is_trading_day"].sum())
        print(f"[fetch_market] {name} ({spec['symbol']}): {trading_days} trading days, "
              f"lag_utc_close={spec.get('lag_utc_close', False)}")

    out_path = processed_dir / "market_daily.csv"
    merged.to_csv(out_path, index=False)
    print(f"[fetch_market] wrote {len(merged)} rows, {len(merged.columns)} columns to {out_path}")


if __name__ == "__main__":
    main()
