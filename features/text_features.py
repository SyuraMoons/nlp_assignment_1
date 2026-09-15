"""Phase 3 text-modality features, built on top of the Phase 1/2 daily sentiment
table (sentiment/build_daily.py) and the scored article table
(sentiment/run_indobert.py). Three families of features are added, each targeting a
gap in the Phase 2 baseline (same-day sentiment mean/count only):

1. **Temporal dynamics** -- rolling 3/7/14-day means of sentiment and GDELT tone, a
   "sentiment surprise" (today vs. the trailing 30-day mean), and an article-volume
   z-score. These let a news shock register even on days with no article at all
   (same-day features are all still there via daily_news_features.csv, unchanged).
2. **Theme-group counts** -- daily counts per config.yaml -> theme_groups, parsed out
   of each article's raw GDELT V2Themes string. This is the most direct geopolitical
   signal (conflict/sanctions/trade activity), separate from general tone.
3. **Source-mix share** -- id_bisnis_count / id_kontan_count as a share of that day's
   total, isolating the 2023-03 Kontan-coverage-start structural break (see README)
   from a genuine change in article volume or sentiment.

All rolling/trailing stats are computed strictly causally (day t only ever uses
data from day <= t), so they're safe to use as-is without a future re-check for
leakage inside a single training run; the model script still must not let any
cross-validation fold see rows from its own future.

Sentence embeddings (the 3rd text feature named in the Phase 3 plan) are intentionally
NOT produced here -- PCA-reducing them must be fit on the training fold only, so
embedding generation is done once (features/embed_titles.py) and the *fitting* of any
dimensionality reduction happens inside modeling/train_refined.py's CV loop instead.

Input (data/processed/):
  daily_news_features.csv (sentiment/build_daily.py)
  id_articles_scored.parquet (sentiment/run_indobert.py)

Output (data/processed/):
  model_features.parquet -- daily_news_features.csv's columns plus the features above

Run with: python -m features.text_features
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config

ROLL_WINDOWS = [3, 7, 14]
SURPRISE_WINDOW = 30
VOLUME_ZSCORE_WINDOW = 30


def add_rolling_dynamics(daily):
    daily = daily.sort_values("date").reset_index(drop=True)

    for w in ROLL_WINDOWS:
        daily[f"id_title_sent_roll{w}"] = (
            daily["id_title_sent_mean"].rolling(w, min_periods=1).mean()
        )
        daily[f"id_gdelt_tone_roll{w}"] = (
            daily["id_gdelt_tone_mean"].rolling(w, min_periods=1).mean()
        )

    trailing_mean = daily["id_title_sent_mean"].shift(1).rolling(
        SURPRISE_WINDOW, min_periods=5
    ).mean()
    daily["id_sentiment_surprise"] = daily["id_title_sent_mean"] - trailing_mean

    vol_mean = daily["id_title_count"].shift(1).rolling(
        VOLUME_ZSCORE_WINDOW, min_periods=5
    ).mean()
    vol_std = daily["id_title_count"].shift(1).rolling(
        VOLUME_ZSCORE_WINDOW, min_periods=5
    ).std()
    daily["id_volume_zscore"] = (daily["id_title_count"] - vol_mean) / vol_std.replace(0, np.nan)

    return daily


def add_source_mix(daily):
    source_cols = [c for c in daily.columns if c.startswith("id_") and c.endswith("_count")
                   and c not in ("id_title_count", "id_body_count")]
    total = daily[source_cols].sum(axis=1).replace(0, np.nan)
    for c in source_cols:
        daily[f"{c}_share"] = daily[c] / total
        daily[f"{c}_share"] = daily[f"{c}_share"].fillna(0.0)
    return daily


def parse_theme_group_counts(themes_str, group_prefixes):
    if not isinstance(themes_str, str) or not themes_str:
        return {g: 0 for g in group_prefixes}
    tokens = {t.split(",")[0] for t in themes_str.split(";") if t}
    counts = {}
    for group, prefixes in group_prefixes.items():
        counts[group] = int(any(tok.startswith(p) for tok in tokens for p in prefixes))
    return counts


def add_theme_group_counts(daily, processed_dir, theme_groups, date_range):
    articles_path = processed_dir / "id_articles_scored.parquet"
    articles = pd.read_parquet(articles_path, columns=["date", "themes"])

    rows = articles["themes"].apply(lambda t: parse_theme_group_counts(t, theme_groups))
    theme_df = pd.DataFrame(rows.tolist())
    theme_df["date"] = articles["date"].values

    daily_theme = theme_df.groupby("date").sum().reset_index()
    daily_theme = daily_theme.rename(columns={g: f"id_theme_{g}_count" for g in theme_groups})

    daily = daily.merge(daily_theme, on="date", how="left")
    theme_cols = [f"id_theme_{g}_count" for g in theme_groups]
    daily[theme_cols] = daily[theme_cols].fillna(0).astype(int)
    return daily


def main():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    date_range = config["date_range"]
    theme_groups = config["theme_groups"]

    daily_path = processed_dir / "daily_news_features.csv"
    if not daily_path.exists():
        raise SystemExit(f"{daily_path} not found -- run sentiment/build_daily.py first")
    daily = pd.read_csv(daily_path)

    daily = add_rolling_dynamics(daily)
    daily = add_source_mix(daily)
    daily = add_theme_group_counts(daily, processed_dir, theme_groups, date_range)

    out_path = processed_dir / "model_features.parquet"
    daily.to_parquet(out_path, index=False)
    print(f"[text_features] wrote {len(daily)} rows, {len(daily.columns)} columns to {out_path}")
    new_cols = [c for c in daily.columns if c not in pd.read_csv(daily_path, nrows=0).columns]
    print(f"[text_features] added columns: {new_cols}")


if __name__ == "__main__":
    main()
