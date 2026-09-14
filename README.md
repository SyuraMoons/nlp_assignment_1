# Global Geopolitical Event Prediction: Impact on USD/IDR

NLP course project testing whether global geopolitical news predicts USD/IDR exchange
rate movements. This repo covers **Task 1: news data acquisition & preprocessing**
(Person A's scope) -- pulling Indonesian financial/geopolitical news and global
geopolitical headlines, scraping full article text for a filtered subset, cleaning
everything, and scoring daily sentiment, ready to merge with Person B's USD/IDR series
(Person C's job).

## Scope (current)

- **Indonesian sources**: Kontan.co.id, Bisnis.com, CNBC Indonesia
- **Global sources**: a whitelist of major English outlets (Reuters, AP, BBC, Al
  Jazeera, CNBC, The Guardian, NYT, FT, WSJ, Bloomberg), filtered to geopolitical
  GDELT themes
- **Date range**: ~5 years, Sept 2021 - Sept 2026 (see `config.yaml`)
- **Sentiment**: IndoBERT (`mdhugol/indonesia-bert-sentiment-classification`) on
  Indonesian titles and scraped bodies; FinBERT (`ProsusAI/finbert`) on global titles

## Why GDELT + targeted scraping, not scrape-everything

An earlier version of this pipeline discovered article URLs by querying the Internet
Archive's Wayback Machine CDX API (native sitemaps on these sites only cover the last
~2 days, and site search/full sitemaps are Cloudflare-protected). That approach broke
down for real use: domain-wide CDX queries return 504 timeouts, fetching every
archived article across 5 years would be roughly 1M requests (multiple weeks even at
polite delays), and Kontan's URLs carry no date, so articles filed under the wrong
month.

Instead, **GDELT's translingual GKG feed (pulled via BigQuery) is the discovery index
and the primary dataset**: it already has each article's URL, original-language
headline, timestamp, GDELT topic themes, and tone score, across the whole 5-year
window, without fetching a single article page. Headline sentiment is a real,
complete deliverable on its own if scraping produces nothing else.

Full article text is still valuable (this is a "strategic preprocessing" task, and a
10-word headline gives little to preprocess or analyze), so a filtered subset of that
GDELT index -- sections relevant to macro/markets/international news, GDELT themes
relevant to geopolitics/economy, capped per site per day -- gets scraped for its full
body, reusing per-site parsers that were built and tested against each site's real
HTML. This keeps the scrape to roughly 50-80k articles instead of ~1M.

**Open risk, not yet confirmed from inside this session (no BigQuery access here):**
whether GDELT's `gkg_partitioned` BigQuery table actually carries translingual rows
for these three sites, and whether English-original rows populate the same
`<PAGE_TITLE>` tag used to extract titles. Both are flagged in `gdelt/queries/*.sql`
and checked by `gdelt/fetch.py` (`coverage.sql` first, and a null-title-rate warning
on the global pull) -- **run those checks before trusting the rest of the pipeline.**

## Pipeline

```
gdelt/fetch.py --query coverage           # sanity check: do these domains have GKG rows?
gdelt/fetch.py --query id --year YYYY     # -> data/raw/gdelt_id_<year>.parquet
gdelt/fetch.py --query global --year YYYY # -> data/raw/gdelt_global_<year>.parquet

selection/build_queue.py
    -> data/processed/id_headlines_filtered.parquet   (section + theme filtered, deduped)
    -> data/interim/scrape_queue.parquet              (above, capped per site/day)
    -> data/processed/filter_report.json              (drop counts per stage)

scrapers/kontan.py, bisnis.py, cnbc_indonesia.py  (or scrapers/run_all.py for all three)
    -> data/raw/articles/<site>.jsonl                 (full text for queued URLs only)

preprocessing/clean_text.py
    -> data/interim/id_headlines_clean.parquet
    -> data/interim/global_headlines_clean.parquet
    -> data/interim/bodies_clean.parquet

preprocessing/align_dates.py
    -> data/processed/id_articles.parquet       (headlines + bodies merged, WIB dates)
    -> data/processed/global_headlines.parquet

sentiment/run_indobert.py --check-labels  # confirm the label mapping first
sentiment/run_indobert.py
    -> data/processed/id_articles_scored.parquet
sentiment/run_finbert.py
    -> data/processed/global_headlines_scored.parquet

sentiment/build_daily.py
    -> data/processed/daily_news_features.csv   # one row per calendar day, gap days included
```

`data/raw`, `data/interim`, and `data/processed` are gitignored (large, reproducible
from the scripts).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
gcloud auth application-default login   # BigQuery auth; you run this yourself
```

## Running the pipeline

```bash
# 1. Confirm GDELT has what we need before pulling anything real
python -m gdelt.fetch --query coverage
python -m gdelt.fetch --query id --year 2024 --dry-run

# 2. Pull one pilot year, build the queue, and spot-check before scaling to all 5 years
python -m gdelt.fetch --query id --year 2024
python -m gdelt.fetch --query global --year 2024
python -m selection.build_queue

# 3. Scrape the queued articles (can run all three sites in parallel)
python -m scrapers.run_all

# 4. Preprocess
python -m preprocessing.clean_text
python -m preprocessing.align_dates

# 5. Score sentiment
python -m sentiment.run_indobert --check-labels   # confirm label mapping first
python -m sentiment.run_indobert
python -m sentiment.run_finbert

# 6. Build the daily feature table
python -m sentiment.build_daily
```

Repeat steps 1-2 for every year in the range once the pilot year looks right (or
adjust `gdelt/fetch.py` to loop over years). Each scraper is resumable: re-running it
skips URLs already saved in `data/raw/articles/<site>.jsonl`.

## Config

`config.yaml` holds the date range, timezone, GDELT domains/themes, per-site sections
considered in scope, the relevance theme list, the daily scrape cap, and scraping
politeness settings -- shared by every script in the pipeline.

## Design decisions

*(To be filled in by the team in your own words before evaluation -- you need to be
able to justify each of these. Starting points, from what's implemented:)*

- **Why GDELT as the discovery index, not scrape-everything** -- see "Why GDELT +
  targeted scraping" above.
- **Why these three Indonesian sources / this outlet whitelist for global** -- ?
- **Why these sections** (`config.yaml` → `sections`) **and not the full site** -- ?
- **Why this theme-prefix list** (`config.yaml` → `relevant_theme_prefixes`) and how
  it was checked against real filter output (see the EDA notebook once built) -- ?
- **Why a daily cap of 15 per site**, and what tradeoff that represents between
  coverage and scrape time -- ?
- **Why WIB (UTC+7) for date alignment** -- IDR trades in Jakarta.
- **Why a probability-based sentiment score** (P(pos) − P(neg)) instead of a hard
  label -- preserves model confidence instead of collapsing it to -1/0/1.
- **Why score both title and body**, and what the title/body sentiment agreement rate
  says about whether headlines alone would have been sufficient -- ?

## Handoff to Person C

`data/processed/daily_news_features.csv` is the Task 1 deliverable: one row per
**WIB calendar day**, every day in the range present (zero counts / null scores on
days without news), ready to merge with the daily USD/IDR series on `date`.
