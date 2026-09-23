"""Assign every cleaned news article to the USD/IDR trading day it can influence.

The problem: news runs 24/7 (and is timestamped in WIB by src/preprocessing), but
the exchange rate only exists on Indonesian business days, as one JISDOR fixing
published at 10:00 WIB (config.yaml -> alignment.cutoff_time_wib).

The rule -- each article goes to the FIRST JISDOR fixing at or after its timestamp:

  case              example (cutoff 10:00 WIB)              -> trading_date
  same_day          Tue 08:30, Tue is a trading day           Tue
  after_cutoff      Tue 14:00 (market already fixed)          Wed (next trading day)
  non_trading_day   Sat 11:00, or a public holiday            next trading day (e.g. Mon)

Why this rule:
  * No look-ahead leakage: an article is only ever linked to a fixing that happens
    AFTER it was observed, so it can never "explain" a rate that was already set.
  * Weekend / holiday news is not thrown away -- it piles onto the next fixing,
    which is when the market actually gets to react to it.
  * The trading calendar comes from the JISDOR dates themselves, so Indonesian
    public holidays and cuti bersama are handled without a manual holiday list.
  * GDELT timestamps are the time GDELT *captured* the article (15-minute
    batches), which is always at or slightly after real publication, so the
    rule errs on the safe (later) side.

Inputs:
  data/interim/id_headlines_clean.parquet  (src/preprocessing/clean_text.py)
  data/interim/bodies_clean.parquet        (src/preprocessing/clean_text.py)
  data/processed/usd_idr_jisdor.csv        (src/fx/load_jisdor.py)

Output (data/processed/):
  articles_aligned/articles_<year>.parquet -- one row per article, split by
  trading-date year to keep files small (read the whole folder with
  pd.read_parquet("data/processed/articles_aligned")):
      article_id, timestamp_wib, calendar_date_wib, trading_date, alignment_case,
      source, section, url, title, body, has_body, themes, gdelt_tone

Run with: python -m src.alignment.align_news
"""
import json

import pandas as pd

from src.common.config import data_path, load_config


def fixing_times(jisdor, cutoff):
    """One row per trading day with the exact WIB datetime of its JISDOR fixing."""
    hours, minutes = (int(x) for x in cutoff.split(":"))
    fx = pd.DataFrame({"trading_date": pd.to_datetime(jisdor["date"])})
    fx["fixing_time"] = (fx["trading_date"] + pd.Timedelta(hours=hours, minutes=minutes)
                         ).dt.tz_localize("Asia/Jakarta")
    return fx.sort_values("fixing_time").reset_index(drop=True)


def load_articles(interim_dir):
    headlines = pd.read_parquet(interim_dir / "id_headlines_clean.parquet")
    bodies = pd.read_parquet(interim_dir / "bodies_clean.parquet")
    df = headlines.merge(bodies, on="url", how="left")
    df = df.rename(columns={
        "GKGRECORDID": "article_id",
        "wib_datetime": "timestamp_wib",
        "wib_date": "calendar_date_wib",
        "site": "source",
        "V2Themes": "themes",
    })
    # V2Tone is "tone,pos,neg,polarity,..."; the first value is GDELT's overall
    # document tone (roughly -10 negative .. +10 positive).
    df["gdelt_tone"] = pd.to_numeric(df["V2Tone"].str.split(",").str[0], errors="coerce")
    df["has_body"] = df["body"].notna()
    # V2Themes lists every theme mention with a character offset
    # ("ECON_INFLATION,693;ECON_INFLATION,812;..."). Keep just the unique theme names.
    df["themes"] = df["themes"].fillna("").map(
        lambda s: ";".join(sorted({t.split(",")[0] for t in s.split(";") if t})))
    df["timestamp_wib"] = pd.to_datetime(df["timestamp_wib"]).dt.tz_convert("Asia/Jakarta")
    return df


def align(articles, fx):
    articles = articles.sort_values("timestamp_wib").reset_index(drop=True)
    # merge_asof requires identical datetime dtypes on both keys.
    articles["timestamp_wib"] = articles["timestamp_wib"].astype(fx["fixing_time"].dtype)
    aligned = pd.merge_asof(
        articles, fx, left_on="timestamp_wib", right_on="fixing_time",
        direction="forward", allow_exact_matches=True,
    )

    trading_days = set(fx["trading_date"].dt.strftime("%Y-%m-%d"))
    aligned["trading_date"] = aligned["trading_date"].dt.strftime("%Y-%m-%d")
    same_day = aligned["calendar_date_wib"] == aligned["trading_date"]
    on_trading_day = aligned["calendar_date_wib"].isin(trading_days)
    aligned["alignment_case"] = "non_trading_day"
    aligned.loc[on_trading_day & ~same_day, "alignment_case"] = "after_cutoff"
    aligned.loc[same_day, "alignment_case"] = "same_day"
    return aligned


def main():
    config = load_config()
    interim_dir = data_path(config, "interim_dir")
    processed_dir = data_path(config, "processed_dir")

    jisdor_path = processed_dir / "usd_idr_jisdor.csv"
    if not jisdor_path.exists():
        raise SystemExit(f"{jisdor_path} not found -- run src.fx.load_jisdor first")
    fx = fixing_times(pd.read_csv(jisdor_path), config["alignment"]["cutoff_time_wib"])

    articles = load_articles(interim_dir)
    n_loaded = len(articles)
    start = pd.Timestamp(config["date_range"]["start"], tz="Asia/Jakarta")
    articles = articles[articles["timestamp_wib"] >= start]
    n_in_range = len(articles)

    aligned = align(articles, fx)
    # Articles after the last fixing in range have no trading day to map to.
    n_unmapped = int(aligned["trading_date"].isna().sum())
    aligned = aligned.dropna(subset=["trading_date"])

    cols = ["article_id", "timestamp_wib", "calendar_date_wib", "trading_date",
            "alignment_case", "source", "section", "url", "title", "body", "has_body",
            "themes", "gdelt_tone"]
    aligned = aligned[cols].reset_index(drop=True)
    out_path = processed_dir / "articles_aligned"
    out_path.mkdir(parents=True, exist_ok=True)
    for old in out_path.glob("articles_*.parquet"):
        old.unlink()
    for year, part in aligned.groupby(aligned["trading_date"].str[:4]):
        part.to_parquet(out_path / f"articles_{year}.parquet", index=False, compression="zstd")

    cases = aligned["alignment_case"].value_counts().to_dict()
    report = {
        "articles_loaded": n_loaded,
        "articles_on_or_after_start": n_in_range,
        "articles_after_last_fixing_dropped": n_unmapped,
        "articles_aligned": len(aligned),
        "alignment_cases": cases,
        "cutoff_time_wib": config["alignment"]["cutoff_time_wib"],
    }
    with open(processed_dir / "alignment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[align_news] {len(aligned)} articles aligned -> {out_path}")
    print(f"[align_news] cases: {cases}; dropped after last fixing: {n_unmapped}")


if __name__ == "__main__":
    main()
