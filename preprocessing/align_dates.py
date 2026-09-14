"""Merge cleaned Indonesian headlines with their scraped bodies (where a scrape
succeeded), and finalize the global-headlines table. Both outputs carry the full WIB
timestamp alongside the WIB calendar date, so a later task can decide how to shift
news published after market close onto the next trading day.

Inputs (data/interim/, from preprocessing/clean_text.py):
  id_headlines_clean.parquet, global_headlines_clean.parquet, bodies_clean.parquet

Outputs (data/processed/):
  id_articles.parquet      -- timestamp_wib, date, source, section, url, title, body, themes, gdelt_tone
  global_headlines.parquet -- timestamp_wib, date, source, url, title, themes, gdelt_tone

Run with: python -m preprocessing.align_dates
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config


def build_id_articles(interim_dir):
    headlines = pd.read_parquet(interim_dir / "id_headlines_clean.parquet")
    bodies = pd.read_parquet(interim_dir / "bodies_clean.parquet")

    merged = headlines.merge(bodies, on="url", how="left")
    merged = merged.rename(columns={
        "wib_datetime": "timestamp_wib",
        "wib_date": "date",
        "site": "source",
        "SourceCommonName": "source_domain",
        "V2Themes": "themes",
        "V2Tone": "gdelt_tone",
    })
    return merged[[
        "timestamp_wib", "date", "source", "section", "url", "title", "body",
        "themes", "gdelt_tone",
    ]]


def build_global_headlines(interim_dir):
    df = pd.read_parquet(interim_dir / "global_headlines_clean.parquet")
    df = df.rename(columns={
        "wib_datetime": "timestamp_wib",
        "SourceCommonName": "source",
        "V2Themes": "themes",
        "V2Tone": "gdelt_tone",
    })
    df["date"] = df["timestamp_wib"].apply(lambda dt: dt.date().isoformat())
    return df[["timestamp_wib", "date", "source", "url", "title", "themes", "gdelt_tone"]]


def main():
    config = load_config()
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    id_articles = build_id_articles(interim_dir)
    global_headlines = build_global_headlines(interim_dir)

    id_articles.to_parquet(processed_dir / "id_articles.parquet", index=False)
    global_headlines.to_parquet(processed_dir / "global_headlines.parquet", index=False)

    body_coverage = id_articles["body"].notna().mean() if len(id_articles) else 0.0
    print(f"id_articles: {len(id_articles)} rows, {body_coverage:.1%} with a scraped body")
    print(f"global_headlines: {len(global_headlines)} rows")


if __name__ == "__main__":
    main()
