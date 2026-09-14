"""Targeted scraper for Bisnis.com articles in the GDELT-derived scrape queue
(data/interim/scrape_queue.parquet, built by selection/build_queue.py).

Run with: python -m scrapers.bisnis
"""
from bs4 import BeautifulSoup
from tqdm import tqdm

from scrapers.base import JsonlStore, PoliteSession, fetch_article_html, load_queue_for_site
from common.config import load_config


def parse_article(html):
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else None

    # Two elements share the "detailsContent" class: an "mt20" promo blurb ("follow us
    # on Google News") and an "mt40" <article> that holds the real body paragraphs.
    body_el = soup.find("article", class_="detailsContent")
    body = None
    if body_el:
        paragraphs = [p.get_text(" ", strip=True) for p in body_el.find_all("p", recursive=False)]
        paragraphs = [p for p in paragraphs if p and p != "Baca Juga"]
        body = "\n".join(paragraphs)

    return title, body


def main():
    config = load_config()
    session = PoliteSession(config)
    store = JsonlStore("bisnis", config=config)

    queue = load_queue_for_site("bisnis", config)
    existing = store.load_existing_urls()
    queue = queue[~queue["url"].isin(existing)]
    print(f"[bisnis] {len(queue)} queued URLs to fetch ({len(existing)} already done)")

    for _, row in tqdm(queue.iterrows(), total=len(queue), desc="bisnis"):
        url = row["url"]
        html = fetch_article_html(session, url, wayback_timestamp=str(int(row["gdelt_timestamp"])))
        if html is None:
            store.append_failed(url, "fetch_failed")
            continue

        title, body = parse_article(html)
        if not title or not body:
            store.append_failed(url, "missing_title_or_body")
            continue

        store.append({
            "url": url,
            "title": title,
            "published_at": row["wib_date"],
            "body": body,
            "source": "bisnis",
        })


if __name__ == "__main__":
    main()
