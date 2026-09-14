"""Aggregate scored Indonesian and global headlines/articles into one row per
calendar day (WIB) across the full configured date range -- including days with no
news at all, so a downstream merge with the daily USD/IDR series never silently drops
a date.

Inputs: data/processed/id_articles_scored.parquet, global_headlines_scored.parquet
Output: data/processed/daily_news_features.csv

Run with: python -m sentiment.build_daily
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config


def daily_id_features(df):
    if df.empty:
        return pd.DataFrame(columns=[
            "date", "id_title_sent_mean", "id_title_count",
            "id_body_sent_mean", "id_body_count", "id_gdelt_tone_mean",
        ])

    df = df.copy()
    df["gdelt_tone_numeric"] = pd.to_numeric(df["gdelt_tone"].str.split(",").str[0], errors="coerce")

    daily = df.groupby("date").agg(
        id_title_sent_mean=("title_sentiment", "mean"),
        id_title_count=("title_sentiment", "count"),
        id_gdelt_tone_mean=("gdelt_tone_numeric", "mean"),
    ).reset_index()

    body_daily = (
        df.dropna(subset=["body_sentiment"])
        .astype({"body_sentiment": "float"})
        .groupby("date")["body_sentiment"]
        .agg(id_body_sent_mean="mean", id_body_count="count")
        .reset_index()
    )

    per_source = df.groupby(["date", "source"]).size().unstack(fill_value=0)
    per_source.columns = [f"id_{c}_count" for c in per_source.columns]
    per_source = per_source.reset_index()

    daily = daily.merge(body_daily, on="date", how="left").merge(per_source, on="date", how="left")
    daily["id_body_count"] = daily["id_body_count"].fillna(0).astype(int)
    return daily


def daily_global_features(df):
    if df.empty:
        return pd.DataFrame(columns=[
            "date", "global_title_sent_mean", "global_count", "global_gdelt_tone_mean",
        ])

    df = df.copy()
    df["gdelt_tone_numeric"] = pd.to_numeric(df["gdelt_tone"].str.split(",").str[0], errors="coerce")

    daily = df.groupby("date").agg(
        global_title_sent_mean=("title_sentiment", "mean"),
        global_count=("title_sentiment", "count"),
        global_gdelt_tone_mean=("gdelt_tone_numeric", "mean"),
    ).reset_index()
    return daily


def main():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    date_range = config["date_range"]

    id_path = processed_dir / "id_articles_scored.parquet"
    global_path = processed_dir / "global_headlines_scored.parquet"
    if not id_path.exists():
        raise SystemExit(f"{id_path} not found -- run sentiment/run_indobert.py first")
    if not global_path.exists():
        raise SystemExit(f"{global_path} not found -- run sentiment/run_finbert.py first")

    id_df = pd.read_parquet(id_path)
    global_df = pd.read_parquet(global_path)

    id_daily = daily_id_features(id_df)
    global_daily = daily_global_features(global_df)

    all_days = pd.DataFrame({
        "date": pd.date_range(date_range["start"], date_range["end"], freq="D").strftime("%Y-%m-%d")
    })

    daily = all_days.merge(id_daily, on="date", how="left").merge(global_daily, on="date", how="left")

    count_cols = [c for c in daily.columns if c.endswith("_count")]
    daily[count_cols] = daily[count_cols].fillna(0).astype(int)

    out_path = processed_dir / "daily_news_features.csv"
    daily.to_csv(out_path, index=False)
    print(f"[build_daily] wrote {len(daily)} daily rows ({date_range['start']} to {date_range['end']}) to {out_path}")
    print(f"[build_daily] days with at least one ID article: {(daily['id_title_count'] > 0).sum()}")
    print(f"[build_daily] days with at least one global article: {(daily['global_count'] > 0).sum()}")


if __name__ == "__main__":
    main()
