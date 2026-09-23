"""Build the final Task 1 dataset: one row per USD/IDR trading day, joining the
JISDOR rate with the news that was aligned to that day by
src/alignment/align_news.py.

Only raw, model-free aggregates are computed here (counts, GDELT's own tone score,
and the concatenated headlines). Turning text into NLP features -- sentiment,
TF-IDF, etc. -- is deliberately left to Task 2, so this dataset stays a clean,
reusable base.

Inputs (data/processed/):
  usd_idr_jisdor.csv, articles_aligned/ (from align_news)

Output (data/processed/):
  final_dataset.csv -- one row per JISDOR trading day:
      date, jisdor_rate, pct_change, log_return, direction, calendar_days_since_prev,
      n_articles, n_bisnis, n_kontan, n_with_body, n_non_trading_day_articles,
      gdelt_tone_mean, titles   (headlines joined with " || ", oldest first)

Run with: python -m src.alignment.build_dataset
"""
import pandas as pd

from src.common.config import data_path, load_config

TITLE_SEP = " || "


def daily_news(articles):
    articles = articles.sort_values("timestamp_wib")
    g = articles.groupby("trading_date")
    daily = pd.DataFrame({
        "n_articles": g.size(),
        "n_with_body": g["has_body"].sum(),
        "n_non_trading_day_articles": g["alignment_case"].apply(lambda s: (s == "non_trading_day").sum()),
        "gdelt_tone_mean": g["gdelt_tone"].mean(),
        "titles": g["title"].apply(TITLE_SEP.join),
    })
    per_source = articles.pivot_table(index="trading_date", columns="source",
                                      values="article_id", aggfunc="count", fill_value=0)
    per_source.columns = [f"n_{c}" for c in per_source.columns]
    return daily.join(per_source).reset_index().rename(columns={"trading_date": "date"})


def main():
    config = load_config()
    processed_dir = data_path(config, "processed_dir")
    fx = pd.read_csv(processed_dir / "usd_idr_jisdor.csv")
    articles = pd.read_parquet(processed_dir / "articles_aligned")

    dataset = fx.merge(daily_news(articles), on="date", how="left")
    count_cols = [c for c in dataset.columns if c.startswith("n_")]
    dataset[count_cols] = dataset[count_cols].fillna(0).astype(int)
    dataset["titles"] = dataset["titles"].fillna("")

    front = ["date", "jisdor_rate", "pct_change", "log_return", "direction",
             "calendar_days_since_prev", "n_articles", "n_bisnis", "n_kontan",
             "n_with_body", "n_non_trading_day_articles", "gdelt_tone_mean", "titles"]
    dataset = dataset[[c for c in front if c in dataset.columns] +
                      [c for c in dataset.columns if c not in front]]

    out_path = processed_dir / "final_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"[build_dataset] {len(dataset)} trading days, "
          f"{int(dataset['n_articles'].sum())} articles -> {out_path}")
    print(f"[build_dataset] trading days with zero news: {(dataset['n_articles'] == 0).sum()}")


if __name__ == "__main__":
    main()
