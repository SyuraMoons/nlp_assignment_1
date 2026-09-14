"""Targeted scraper for CNBC Indonesia articles in the GDELT-derived scrape queue
(data/interim/scrape_queue.parquet, built by selection/build_queue.py).

Run with: python -m scrapers.cnbc_indonesia
"""
from bs4 import BeautifulSoup
from tqdm import tqdm

from scrapers.base import JsonlStore, PoliteSession, fetch_article_html, load_queue_for_site
from common.config import load_config


def parse_article(html):
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else None

    # The page nests an outer "detail-text min-w-0" wrapper around the real
    # "detail-text" div that actually holds the paragraphs; pick whichever
    # "detail-text" element has the most direct <p> children.
    candidates = soup.find_all("div", class_="detail-text")
    body_el = max(candidates, key=lambda d: len(d.find_all("p", recursive=False)), default=None)
    body = None
    if body_el:
        paragraphs = [p.get_text(" ", strip=True) for p in body_el.find_all("p", recursive=False)]
        body = "\n".join(p for p in paragraphs if p)

    return title, body


def main():
    config = load_config()
    session = PoliteSession(config)
    store = JsonlStore("cnbc_indonesia", config=config)

    queue = load_queue_for_site("cnbc_indonesia", config)
    existing = store.load_existing_urls()
    queue = queue[~queue["url"].isin(existing)]
    print(f"[cnbc_indonesia] {len(queue)} queued URLs to fetch ({len(existing)} already done)")

    for _, row in tqdm(queue.iterrows(), total=len(queue), desc="cnbc_indonesia"):
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
            "source": "cnbc_indonesia",
        })


if __name__ == "__main__":
    main()
