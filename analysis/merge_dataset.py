"""Merge the daily news-sentiment features, the daily USD/IDR exchange rate, and
(Phase 3) the market-data modality into one dataset, ready for the modeling/
hypothesis-testing tasks.

Inputs (data/processed/):
  daily_news_features.csv, usd_idr_daily.csv, market_daily.csv (optional -- Phase 3)

Output (data/processed/):
  merged_dataset.csv -- one row per calendar day, all tables' columns joined on date

Run with: python -m analysis.merge_dataset
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config


def main():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])

    news_path = processed_dir / "daily_news_features.csv"
    fx_path = processed_dir / "usd_idr_daily.csv"
    market_path = processed_dir / "market_daily.csv"
    if not news_path.exists():
        raise SystemExit(f"{news_path} not found -- run sentiment/build_daily.py first")
    if not fx_path.exists():
        raise SystemExit(f"{fx_path} not found -- run fx/fetch_usdidr.py first")

    news = pd.read_csv(news_path)
    fx = pd.read_csv(fx_path)

    merged = news.merge(fx, on="date", how="inner")
    if len(merged) != len(news) or len(merged) != len(fx):
        raise SystemExit(
            f"row count mismatch after merge: news={len(news)}, fx={len(fx)}, "
            f"merged={len(merged)} -- date ranges/coverage no longer line up, "
            "investigate before treating this as the Task 1 deliverable"
        )

    if market_path.exists():
        market = pd.read_csv(market_path)
        merged = merged.merge(market, on="date", how="left")
        if len(merged) != len(news):
            raise SystemExit(
                f"row count mismatch after market merge: expected {len(news)}, "
                f"got {len(merged)} -- market_daily.csv's date range no longer lines up"
            )
        print(f"[merge_dataset] joined market_daily.csv ({len(market.columns) - 1} market columns)")
    else:
        print("[merge_dataset] market_daily.csv not found -- skipping market modality "
              "(run fx/fetch_market.py for Phase 3 features)")

    out_path = processed_dir / "merged_dataset.csv"
    merged.to_csv(out_path, index=False)
    print(f"[merge_dataset] wrote {len(merged)} rows, {len(merged.columns)} columns to {out_path}")
    print(f"[merge_dataset] columns: {merged.columns.tolist()}")


if __name__ == "__main__":
    main()
