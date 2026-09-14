"""Clean headline text (Indonesian + global) and scraped article bodies.

Inputs:
  data/processed/id_headlines_filtered.parquet   (from selection/build_queue.py)
  data/raw/gdelt_global_*.parquet                (from gdelt/fetch.py --query global)
  data/raw/articles/<site>.jsonl                 (from scrapers/*.py, targeted scrape)

Outputs (data/interim/):
  id_headlines_clean.parquet
  global_headlines_clean.parquet
  bodies_clean.parquet

Appends stage counts to data/processed/filter_report.json (created by build_queue.py).
Run with: python -m preprocessing.clean_text
"""
import glob
import html
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config
from common.gdelt_utils import gdelt_timestamp_to_wib, normalize_url

# Known non-content lines each site injects into scraped article bodies.
BOILERPLATE_PATTERNS = [
    re.compile(r"^Reporter:.*\|\s*Editor:.*$"),          # Kontan byline
    re.compile(r"^Cek Berita dan Artikel yang lain di"),  # Bisnis / follow-us promo
    re.compile(r"^Baca Juga$"),
    re.compile(r"^Nyaman tanpa iklan\."),                 # Bisnis subscription promo
]
MIN_BODY_CHARS = 200


def clean_title(raw_title):
    if not raw_title:
        return None
    title = html.unescape(raw_title).strip()
    return title or None


def normalize_title_key(title):
    return re.sub(r"\s+", " ", title.lower()).strip()


def clean_id_headlines(processed_dir):
    path = processed_dir / "id_headlines_filtered.parquet"
    df = pd.read_parquet(path)
    before = len(df)

    df["title"] = df["title"].apply(clean_title)
    df = df[df["title"].notna()].reset_index(drop=True)
    after_empty = len(df)

    df["_title_key"] = df["title"].apply(normalize_title_key)
    df = df.sort_values("wib_datetime").drop_duplicates(subset="_title_key", keep="first")
    df = df.drop(columns=["_title_key"]).reset_index(drop=True)
    after_dedupe = len(df)

    return df, [
        {"stage": "id_headlines_raw", "count": before},
        {"stage": "id_headlines_after_empty_title_drop", "count": after_empty},
        {"stage": "id_headlines_after_cross_site_title_dedupe", "count": after_dedupe},
    ]


def clean_global_headlines(raw_dir):
    paths = sorted(glob.glob(str(raw_dir / "gdelt_global_*.parquet")))
    empty_cols = ["GKGRECORDID", "gdelt_timestamp", "SourceCommonName", "url", "title",
                  "V2Themes", "V2Tone", "wib_datetime"]
    if not paths:
        return pd.DataFrame(columns=empty_cols), [{"stage": "global_headlines_raw", "count": 0}]

    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    before = len(df)

    df["url"] = df["url"].apply(normalize_url)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)

    df["title"] = df["title"].apply(clean_title)
    df = df[df["title"].notna()].reset_index(drop=True)
    after_empty = len(df)

    df["_title_key"] = df["title"].apply(normalize_title_key)
    df["wib_datetime"] = df["gdelt_timestamp"].apply(gdelt_timestamp_to_wib)
    df = df.sort_values("wib_datetime").drop_duplicates(subset="_title_key", keep="first")
    df = df.drop(columns=["_title_key"]).reset_index(drop=True)
    after_dedupe = len(df)

    return df, [
        {"stage": "global_headlines_raw", "count": before},
        {"stage": "global_headlines_after_empty_title_drop", "count": after_empty},
        {"stage": "global_headlines_after_title_dedupe", "count": after_dedupe},
    ]


def strip_boilerplate(body):
    lines = [line.strip() for line in body.split("\n")]
    kept = [
        line for line in lines
        if line and not any(pat.search(line) for pat in BOILERPLATE_PATTERNS)
    ]
    return "\n".join(kept)


def clean_bodies(raw_dir):
    articles_dir = raw_dir / "articles"
    rows = []
    for jsonl_path in sorted(articles_dir.glob("*.jsonl")):
        if jsonl_path.name == "_failed.jsonl":
            continue
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame(columns=["url", "body"]), [{"stage": "bodies_scraped", "count": 0}]

    df = pd.DataFrame(rows)
    before = len(df)

    df["url"] = df["url"].apply(normalize_url)
    df["body"] = df["body"].apply(lambda b: strip_boilerplate(b) if b else None)
    df = df[df["body"].notna() & (df["body"].str.len() >= MIN_BODY_CHARS)]
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    after = len(df)

    return df[["url", "body"]], [
        {"stage": "bodies_scraped", "count": before},
        {"stage": "bodies_after_boilerplate_and_length_filter", "count": after},
    ]


def append_report(processed_dir, new_stages):
    report_path = processed_dir / "filter_report.json"
    report = {"stages": []}
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
    report["stages"].extend(new_stages)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)


def main():
    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    id_df, id_stages = clean_id_headlines(processed_dir)
    global_df, global_stages = clean_global_headlines(raw_dir)
    bodies_df, body_stages = clean_bodies(raw_dir)

    id_df.to_parquet(interim_dir / "id_headlines_clean.parquet", index=False)
    global_df.to_parquet(interim_dir / "global_headlines_clean.parquet", index=False)
    bodies_df.to_parquet(interim_dir / "bodies_clean.parquet", index=False)

    append_report(processed_dir, id_stages + global_stages + body_stages)

    print(f"id_headlines_clean: {len(id_df)} rows")
    print(f"global_headlines_clean: {len(global_df)} rows")
    print(f"bodies_clean: {len(bodies_df)} rows")


if __name__ == "__main__":
    main()
