"""Turn raw GDELT headline pulls (data/raw/gdelt_id_*.parquet) into:
1. data/processed/id_headlines_filtered.parquet -- every headline that passes the
   section + theme filter, deduped. This is the full set used for headline-level
   sentiment, uncapped.
2. data/interim/scrape_queue.parquet -- a per-site, per-WIB-day capped subset of (1),
   the only URLs that get a full-text scrape. The cap exists purely to keep the scrape
   runtime bounded; it does not limit headline sentiment coverage.

Drop counts at each stage go to data/processed/filter_report.json as evidence for the
strategic-preprocessing writeup.

Run with: python -m selection.build_queue
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config
from common.gdelt_utils import (
    gdelt_timestamp_to_wib,
    matches_theme_prefixes,
    normalize_url,
    section_for_url,
    site_for_url,
)


def load_raw_id_headlines(raw_dir):
    paths = sorted(glob.glob(str(raw_dir / "gdelt_id_*.parquet")))
    if not paths:
        raise SystemExit(f"No gdelt_id_*.parquet files in {raw_dir} -- run gdelt/fetch.py first")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def sort_key(url):
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def build(config):
    raw_dir = Path(config["paths"]["raw_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_id_headlines(raw_dir)
    report = {"stages": []}
    report["stages"].append({"stage": "raw_pulled", "count": len(df)})

    df["url"] = df["url"].apply(normalize_url)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    report["stages"].append({"stage": "after_url_dedupe", "count": len(df)})

    df["site"] = df["url"].apply(site_for_url)
    df = df[df["site"].notna()].reset_index(drop=True)
    df["section"] = df.apply(lambda r: section_for_url(r["url"], r["site"]), axis=1)

    sections_cfg = config["sections"]
    df["section_ok"] = df.apply(
        lambda r: r["section"] in sections_cfg.get(r["site"], []), axis=1
    )
    dropped_sections = df[~df["section_ok"]]["section"].value_counts().to_dict()
    df = df[df["section_ok"]].drop(columns=["section_ok"]).reset_index(drop=True)
    report["stages"].append({
        "stage": "after_section_filter",
        "count": len(df),
        "dropped_sections_sample": dict(list(dropped_sections.items())[:20]),
    })

    theme_prefixes = config["relevant_theme_prefixes"]
    df["theme_ok"] = df["V2Themes"].apply(lambda t: matches_theme_prefixes(t, theme_prefixes))
    df = df[df["theme_ok"]].drop(columns=["theme_ok"]).reset_index(drop=True)
    report["stages"].append({"stage": "after_theme_filter", "count": len(df)})

    df["wib_datetime"] = df["gdelt_timestamp"].apply(gdelt_timestamp_to_wib)
    df["wib_date"] = df["wib_datetime"].apply(lambda dt: dt.date().isoformat())

    filtered_path = processed_dir / "id_headlines_filtered.parquet"
    df.to_parquet(filtered_path, index=False)

    cap = config["daily_cap_per_site"]
    df["_sort_key"] = df["url"].apply(sort_key)
    df = df.sort_values(["site", "wib_date", "_sort_key"])
    queue = df.groupby(["site", "wib_date"], group_keys=False).head(cap).drop(columns=["_sort_key"])
    report["stages"].append({
        "stage": "scrape_queue_after_daily_cap",
        "count": len(queue),
        "daily_cap_per_site": cap,
    })

    queue_path = interim_dir / "scrape_queue.parquet"
    queue.to_parquet(queue_path, index=False)

    report_path = processed_dir / "filter_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"wrote {len(df)} filtered headlines to {filtered_path}")
    print(f"wrote {len(queue)} queued articles to {queue_path}")
    print(f"wrote filter report to {report_path}")
    print(json.dumps(report, indent=2))


def main():
    config = load_config()
    build(config)


if __name__ == "__main__":
    main()
