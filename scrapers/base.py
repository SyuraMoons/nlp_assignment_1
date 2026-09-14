"""Shared HTTP session, rate limiting/retries, and JSONL checkpointing for the
targeted per-article scrapers. Article discovery no longer happens here -- URLs come
from the GDELT-derived scrape queue built by selection/build_queue.py, since the
Wayback Machine CDX API (used in an earlier version of this pipeline) returned
504 timeouts on domain-wide queries and would have taken ~1M fetches to cover every
archived article across 5 years.
"""
import json
import random
import time
from pathlib import Path

import pandas as pd
import requests

from common.config import load_config

WAYBACK_SNAPSHOT_URL = "https://web.archive.org/web/{timestamp}id_/{url}"


def load_queue_for_site(site_name, config=None):
    """Rows of data/interim/scrape_queue.parquet (built by selection/build_queue.py)
    for one site, sorted by WIB date so a killed run resumes in roughly chronological
    order.
    """
    config = config or load_config()
    queue_path = Path(config["paths"]["interim_dir"]) / "scrape_queue.parquet"
    if not queue_path.exists():
        raise SystemExit(f"{queue_path} not found -- run selection/build_queue.py first")
    df = pd.read_parquet(queue_path)
    df = df[df["site"] == site_name].sort_values("wib_date")
    return df


class PoliteSession:
    """requests.Session wrapper that adds a delay + retry-with-backoff around every GET."""

    def __init__(self, config=None):
        self.config = config or load_config()
        scraping_cfg = self.config["scraping"]
        self.min_delay = scraping_cfg["min_delay_seconds"]
        self.max_delay = scraping_cfg["max_delay_seconds"]
        self.max_retries = scraping_cfg["max_retries"]
        self.timeout = scraping_cfg["timeout_seconds"]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": scraping_cfg["user_agent"]})

    def get(self, url, **kwargs):
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                time.sleep(random.uniform(self.min_delay, self.max_delay))
                resp = self.session.get(url, timeout=self.timeout, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                backoff = self.min_delay * (2 ** attempt)
                time.sleep(backoff)
        raise last_exc


def fetch_article_html(session, url, wayback_timestamp=None):
    """Fetch an article's HTML from the live site; fall back to its Wayback snapshot
    (if a capture timestamp is known, e.g. from the GDELT row) when the live page is
    gone (404) or otherwise fails. Returns None if both attempts fail.
    """
    try:
        return session.get(url).text
    except requests.RequestException:
        pass

    if not wayback_timestamp:
        return None

    snapshot_url = WAYBACK_SNAPSHOT_URL.format(timestamp=wayback_timestamp, url=url)
    try:
        return session.get(snapshot_url).text
    except requests.RequestException:
        return None


class JsonlStore:
    """Append-only JSONL store with URL-based resume, so a killed scrape run can pick
    up where it left off without refetching articles already saved.
    """

    def __init__(self, site_name, raw_dir=None, config=None):
        config = config or load_config()
        self.dir = Path(raw_dir or config["paths"]["raw_dir"]) / "articles"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{site_name}.jsonl"
        self.failed_path = self.dir / "_failed.jsonl"

    def load_existing_urls(self):
        if not self.path.exists():
            return set()
        urls = set()
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    urls.add(json.loads(line)["url"])
        return urls

    def append(self, article):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(article, ensure_ascii=False) + "\n")

    def append_failed(self, url, reason):
        with open(self.failed_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"url": url, "reason": reason}, ensure_ascii=False) + "\n")
