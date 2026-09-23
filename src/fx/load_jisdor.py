"""Load Bank Indonesia's JISDOR USD/IDR reference rate from the Excel file
downloaded from bi.go.id and turn it into a clean trading-day series.

Why JISDOR: Task 1 requires Bank Indonesia as the ultimate source of the USD rate.
JISDOR (Jakarta Interbank Spot Dollar Rate) is BI's official daily USD/IDR
reference rate, fixed once per Indonesian business day and published at 10:00 WIB.
BI's website blocks scripted downloads, so the Excel is downloaded by hand from
https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx
(period 01-09-2021 to 01-09-2026, or wider) and saved to the path in
config.yaml -> fx.raw_file.

The JISDOR dates double as the project's trading calendar: BI only publishes on
days the Indonesian interbank market is open, so weekends AND Indonesian public
holidays / cuti bersama are excluded automatically -- no hand-maintained holiday
list needed. Non-trading days are NOT forward-filled into fake rows; news that falls
on them is moved to the next trading day instead (see src/alignment/align_news.py).

Output (data/processed/):
  usd_idr_jisdor.csv -- one row per JISDOR fixing day:
      date, jisdor_rate, pct_change, log_return, direction, calendar_days_since_prev

Run with: python -m src.fx.load_jisdor
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import PROJECT_ROOT, data_path, load_config


def read_bi_excel(path):
    """BI's export has a few title rows above a header row 'NO | Tanggal | Kurs'.
    Find that header instead of hard-coding the offset, since BI changes the layout
    from time to time."""
    raw = pd.read_excel(path, header=None, dtype=object)
    header_row = raw.index[raw.apply(
        lambda r: r.astype(str).str.strip().str.lower().isin(["tanggal"]).any(), axis=1
    )][0]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(c).strip().lower() for c in raw.iloc[header_row]]
    df = df[["tanggal", "kurs"]].dropna()

    # Dates come as strings like '9/23/2026 12:00:00 AM' (month/day/year).
    df["date"] = pd.to_datetime(df["tanggal"].astype(str), format="%m/%d/%Y %I:%M:%S %p",
                                errors="coerce")
    bad = df["date"].isna()
    if bad.any():  # fall back for cells Excel already stored as real dates
        df.loc[bad, "date"] = pd.to_datetime(df.loc[bad, "tanggal"], errors="coerce")
    df["jisdor_rate"] = pd.to_numeric(df["kurs"], errors="coerce")
    return df[["date", "jisdor_rate"]].dropna()


def build_series(df, start, end):
    df = df.drop_duplicates("date").sort_values("date")
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    df["pct_change"] = df["jisdor_rate"].pct_change()
    df["log_return"] = np.log(df["jisdor_rate"]).diff()
    # +1 = IDR weakened (USD up), -1 = IDR strengthened, 0 = unchanged fixing.
    df["direction"] = np.sign(df["jisdor_rate"].diff()).astype("Int64")
    # >1 means a weekend/holiday gap before this fixing (e.g. Monday = 3).
    df["calendar_days_since_prev"] = df["date"].diff().dt.days.astype("Int64")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main():
    config = load_config()
    raw_path = PROJECT_ROOT / config["fx"]["raw_file"]
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found -- download the JISDOR Excel from bi.go.id first")

    start = pd.Timestamp(config["date_range"]["start"])
    end = pd.Timestamp(config["date_range"]["end"])
    series = build_series(read_bi_excel(raw_path), start, end)

    out_dir = data_path(config, "processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "usd_idr_jisdor.csv"
    series.to_csv(out_path, index=False)

    print(f"[load_jisdor] {len(series)} JISDOR fixing days "
          f"({series['date'].iloc[0]} to {series['date'].iloc[-1]}) -> {out_path}")
    print(f"[load_jisdor] rate range: {series['jisdor_rate'].min():,.0f} - "
          f"{series['jisdor_rate'].max():,.0f} IDR per USD")


if __name__ == "__main__":
    main()
