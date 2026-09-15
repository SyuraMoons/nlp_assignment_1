# Global Geopolitical Event Prediction: Impact on USD/IDR

NLP course project testing whether global geopolitical news predicts USD/IDR exchange
rate movements. This repo covers **Task 1: news data acquisition & preprocessing**
(Person A's scope) -- pulling Indonesian financial/geopolitical news and global
geopolitical headlines, scraping full article text for a filtered subset, cleaning
everything, and scoring daily sentiment, ready to merge with Person B's USD/IDR series
(Person C's job).

## Scope (current)

- **Indonesian sources**: Kontan.co.id (GDELT coverage from 2023-03 only) and
  Bisnis.com (full range). CNBC Indonesia was in the original target list but has
  zero GDELT translingual coverage and was dropped -- see "Resolved risk" below.
- **Global sources**: a whitelist of major English outlets (Reuters, AP, BBC, Al
  Jazeera, CNBC, The Guardian, NYT, FT, WSJ, Bloomberg), filtered to geopolitical
  GDELT themes. **Not yet pulled** -- blocked on BigQuery billing, see "Known
  limitation" below; current deliverable is Indonesian-side only.
- **Date range**: ~5 years, Sept 2021 - Sept 2026 (see `config.yaml`)
- **Sentiment**: IndoBERT (`mdhugol/indonesia-bert-sentiment-classification`) on
  Indonesian titles and scraped bodies; FinBERT (`ProsusAI/finbert`) on global titles
  (pending global data)

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

**Resolved risk (coverage.sql run 2026-09-15):** GDELT's `gkg_partitioned` table does
carry translingual rows for two of the three configured Indonesian sites, but **not
all**:
- `bisnis.com` -- full coverage, every month from 2021-09 through 2026-09.
- `kontan.co.id` -- coverage only starts **2023-03**; nothing before that. The 2021-09
  to 2023-02 window for Kontan is a real gap in GDELT's index, not a scraping or
  query bug.
- `cnbcindonesia.com` -- **zero rows across the entire 5-year window.** GDELT's
  translingual GKG feed does not index this domain at all. It was dropped from the
  scraped/scored dataset as a result; `sites`/`id_domains` in `config.yaml` still list
  it for traceability, but no pipeline output contains CNBC Indonesia data. This is a
  documented data-availability constraint, not an oversight.

**Known limitation, not yet resolved:** the global (English-outlet) GDELT pull has not
been run -- the project's GCP project has no billing account attached, and BigQuery's
free-tier sandbox quota (1 TiB/month) was exhausted by the Indonesian pulls alone. As a
result, `data/processed/global_headlines_scored.parquet` and every downstream "global"
column in `daily_news_features.csv` are currently empty (0 rows). The daily feature
table is still a valid, complete deliverable for the **Indonesian domestic-reaction**
half of the hypothesis; the global/FinBERT half is pending billing being enabled on
the BigQuery project.

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

- **Why GDELT as the discovery index, not scrape-everything** -- see "Why GDELT +
  targeted scraping" above.
- **Why these three Indonesian sources / this outlet whitelist for global** -- Kontan,
  Bisnis, and CNBC Indonesia are the three largest Indonesian-language financial/macro
  news outlets, giving broad domestic coverage of exactly the kind of reporting
  (rate moves, inflation, trade policy) expected to move USD/IDR. In practice GDELT
  only indexes two of the three (see "Resolved risk" above) -- CNBC Indonesia was
  dropped after the coverage check came back empty for it, leaving Kontan + Bisnis as
  the actual domestic source set. The global whitelist (Reuters, AP, BBC, Al Jazeera,
  CNBC, The Guardian, NYT, FT, WSJ, Bloomberg) was chosen as major English-language
  outlets with both broad geopolitical reach and financial-market desks, so the same
  set plausibly covers both the "shock" (geopolitical event) and "market read-through"
  (financial-press framing) sides of the global hypothesis half.
- **Why these sections** (`config.yaml` → `sections`) **and not the full site** -- each
  site's section filter keeps only channels covering national/international news,
  finance, and markets/investment (e.g. Kontan's `nasional`, `internasional`,
  `keuangan`, `investasi`; Bisnis's `ekonomi`, `market`, `kabar24`), and drops sections
  irrelevant to the hypothesis. On the real filter output this dropped roughly half
  the raw pull (118,459 → 56,917 rows) -- the largest dropped categories were
  `momsmoney` (11,014, a separate personal-finance vertical), `finansial` (7,667,
  overlapping/duplicate financial content), `insight` (7,372, opinion/analysis rather
  than news), plus regional/lifestyle sections (`bandung`, `sumatra`, `bola`,
  `lifestyle`, etc.) that are off-topic for a macro/FX hypothesis.
- **Why this theme-prefix list** (`config.yaml` → `relevant_theme_prefixes`) -- the
  list (`ECON_`, `EPU_`, `TAX_FNCACT`, `ARMEDCONFLICT`, `SANCTIONS`, `MILITARY`,
  `TRADE`, `CRISISLEX`) targets GDELT's own topic taxonomy for economic policy,
  conflict, and trade themes -- the categories most directly tied to currency-moving
  events. Checked against real output: the theme filter is comparatively light-touch
  (56,917 → 55,272 rows, ~3% dropped) since the section filter already narrows to
  finance/macro/international channels where these themes dominate; most of the actual
  precision comes from section filtering, not theme filtering.
- **Why a daily cap of 15 per site**, and what tradeoff that represents -- full-text
  scraping is the slowest, most failure-prone stage (network I/O, per-site parsing,
  politeness delays), while GDELT headline-level sentiment already covers every
  filtered row regardless of the cap. Capping at 15/site/day bounds scrape volume to a
  tractable ~10-11k articles across the full date range (the real queue came out to
  10,633) instead of the full 55,272 filtered headlines, trading full-body coverage
  for scrape time/cost while keeping headline-level coverage complete for all of them.
  Articles are selected deterministically by sorted URL hash so cap choices are
  reproducible across reruns.
- **Why WIB (UTC+7) for date alignment** -- IDR trades in Jakarta.
- **Why a probability-based sentiment score** (P(pos) − P(neg)) instead of a hard
  label -- preserves model confidence instead of collapsing it to -1/0/1.
- **Why score both title and body**, and what the title/body sentiment agreement rate
  says -- of the 10,633 queued articles, 10,560 got both title and body scored; their
  title/body sentiment **sign agreement is 91.9%**. That's high enough that headline-only
  sentiment would likely have captured most of the same signal, but the ~8% disagreement
  (cases where a headline reads one way and the body's overall tone reads the other --
  e.g. a neutral/negative-sounding headline over an ultimately reassuring article) is
  exactly the kind of nuance full-text scraping was meant to capture, justifying the
  scrape effort for the capped subset.

## Person B: USD/IDR exchange rate

`fx/fetch_usdidr.py` pulls the daily USD/IDR exchange rate for Person B's slice of
Task 1, producing the other half of the dataset Person C merges with
`daily_news_features.csv`.

```
fx/fetch_usdidr.py
    -> data/raw/usd_idr_raw.parquet    (raw daily bars, trading days only)
    -> data/processed/usd_idr_daily.csv  (date, usd_idr_close, usd_idr_pct_change,
                                           is_trading_day)
```

Run with: `python -m fx.fetch_usdidr`

**Why Yahoo Finance (`USDIDR=X`) over Bank Indonesia's official JISDOR rate** -- BI's
rate has no clean historical bulk API (page-scraping only), while `yfinance` gives a
free, no-auth daily series covering the full project date range in one call --
important after the BigQuery billing wall hit on the news side. Good enough for a
course-level hypothesis test; not necessarily the rate a trading desk would use.

**Why forward-fill weekends/holidays instead of leaving gaps** -- forex trades ~24/5,
so there's no traded close on the ~530 weekend/holiday days in the range. Every
calendar day in the range still gets a row (`is_trading_day=False` on filled days) so
the merge with the news table on `date` never drops a date, matching how
`daily_news_features.csv` already includes every day, news or not.

## Handoff to Person C

`data/processed/daily_news_features.csv` (Person A) and `data/processed/usd_idr_daily.csv`
(Person B) are the two Task 1 deliverables: one row per calendar day each, every day
in the range present in both, ready to `merge(..., on="date")` for the hypothesis test.
