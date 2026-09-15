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

## Final Task 1 dataset

`analysis/merge_dataset.py` joins the news-sentiment features and the USD/IDR rate
into one dataset:

```
analysis/merge_dataset.py
    -> data/processed/merged_dataset.csv
```

Run with: `python -m analysis.merge_dataset`

`data/processed/merged_dataset.csv` is the Task 1 deliverable: one row per calendar
day (1840 rows, 2021-09-01 to 2026-09-14), combining `daily_news_features.csv`'s
sentiment/count columns with `usd_idr_daily.csv`'s exchange-rate columns, ready for
the modeling/hypothesis-testing tasks later in the course.

## Phase 2: Pipeline Proposal & Baseline Experimentation

`modeling/train_baseline.py` is the first modeling pass on top of the Task 1
dataset -- a naive baseline plus two standard simple/interpretable classifiers
(Gaussian Naive Bayes, Logistic Regression), evaluated on a proper chronological
(no-shuffle) split, before any more advanced modeling.

```
modeling/train_baseline.py
    -> data/processed/baseline_results.json
```

Run with: `python -m modeling.train_baseline`

**Task**: binary classification of next-trading-day USD/IDR direction (up vs.
down), from same-day Indonesian news-sentiment features. Exact-zero "flat"
trading days (9 of them) are dropped as a rare, ambiguous third class;
non-trading days are excluded entirely since their `usd_idr_pct_change` is a
forward-fill artifact, not a real market move.

**Known limitation that scopes this experiment:** inspecting `merged_dataset.csv`
by month shows the Indonesian news features (`id_title_count`, etc.) are nonzero
**only for 2024-01 through 2025-01** -- every other month across the full 5-year
range has zero articles. This lines up with the pipeline's "pull one pilot year
first" instructions (see "Running the pipeline" above) never having been scaled
to the other years. Training on the full 5-year range would therefore mostly
train on "no news that day" rows, so baseline modeling here is restricted to the
2024-01-01 -- 2025-01-07 window where news coverage is real. Extending training
to the full range is future work, pending the rest of the years being pulled/
scraped/scored the same way 2024 was.

**Features**: `id_title_sent_mean`, `id_title_count`, `id_gdelt_tone_mean`,
`id_body_sent_mean`, `id_body_count`, `id_bisnis_count`, `id_kontan_count`, plus a
binary `has_news` indicator (news is missing on ~80% of days even within the 2024
window, so missing sentiment is filled with 0/neutral rather than dropped).
`global_*` columns are excluded -- they're all-NaN in this window (see "Known
limitation" under Person B/global scope above).

**Split**: chronological, last 20% of dates in the window held out as test (train
on earlier dates, test on later ones) to avoid look-ahead leakage.

**Results** (2024-01-01 to 2025-01-07 window, train=212, test=53,
test set positive rate=0.566):

| model | accuracy | f1_macro |
|---|---|---|
| majority_baseline | 0.566 | 0.361 |
| gaussian_naive_bayes | 0.415 | 0.356 |
| logistic_regression | 0.396 | 0.378 |

**Reading the result**: neither classifier beats the trivial majority-class
baseline on this held-out window -- both actually score below it on accuracy.
With only 212 training rows (and news present on just ~20% of them even inside
the "good" year), this is an honest negative baseline result rather than a bug:
same-day sentiment alone, on this little data, doesn't show a predictive edge
over "always guess up." That's a legitimate Phase 2 finding to carry into Phase 3
(more data/years, richer features, or different modeling choices) and Phase 4
(hypothesis testing/error analysis should address directly why the baseline
underperforms majority-class here).

## Phase 3: Model Refinement & Multimodal Integration

Phase 3 addresses the three concrete gaps behind the Phase 2 negative result: too
little data (news features existed for one year only), a single modality (same-day
text sentiment, nothing else), and a noisy single-split evaluation.

### Data extension: news features across the full 5-year range

`selection/build_queue.py`, `preprocessing/clean_text.py`,
`preprocessing/align_dates.py`, `sentiment/run_indobert.py`, `sentiment/build_daily.py`,
and `analysis/merge_dataset.py` were re-run against all six already-downloaded
`data/raw/gdelt_id_<year>.parquet` files (2021-2026, 492,777 raw GDELT rows) instead
of just 2024. This is a pipeline-scope fix, not a new data source -- the raw pulls
already covered the full range; only the filtering/scraping/scoring steps had been
run for one pilot year (see Phase 2's "Known limitation").

- Filtered headlines: 55,272 (2024 only) -> **245,162** (2021-09 to 2026-09).
- `sentiment/run_indobert.py` now caches previously-scored rows by URL (`--no-cache`
  to force a full re-score) and scores on `mps` where available, since re-scoring
  the full set on every rerun would otherwise dominate iteration time.
- Article **bodies stay 2024-only** (10,633 scraped articles) -- scraping all 5 years
  of the larger 39,837-URL queue is a multi-day job and out of scope here. Body
  features (`id_body_sent_mean`, `id_body_count`) are therefore sparse outside 2024;
  they're kept as optional features (filled 0 when absent) rather than blocking the
  rest of the range.
- The Kontan GDELT-coverage gap before 2023-03 (see Phase 1's "Resolved risk") is a
  real structural break in source mix, not new noise -- `features/text_features.py`'s
  `id_kontan_count_share` / `id_bisnis_count_share` features isolate it from a genuine
  change in sentiment or volume.

### Modality 2: market data (`fx/fetch_market.py`)

News-only models have no numeric time-series signal to compare against, so Phase 3
adds one: daily closes for the US Dollar Index (DXY), Brent crude, gold, VIX, US10Y,
IHSG (Jakarta Composite), and three regional-peer FX pairs (USD/MYR, USD/THB,
USD/INR), via the same `yfinance` source as `fx/fetch_usdidr.py`. Output:
`data/processed/market_daily.csv`, joined into `merged_dataset.csv`.

**Leakage handling**: DXY/oil/gold/VIX/US10Y trade on US/London exchanges that close
hours after Jakarta's WIB trading day ends, so their same-day close is not actually
observable yet at WIB close-of-day. Every such ticker (`lag_utc_close: true` in
`config.yaml` -> `market.tickers`) is shifted forward one calendar day before being
stored, so the value under date `t` is what was genuinely knowable by the end of WIB
day `t`. IHSG and the regional FX pairs trade during the Asian day and are left
unshifted. The model also gets yesterday's own USD/IDR return and direction
(`usd_idr_pct_change_lag1`, `usd_idr_direction_lag1`) as an autocorrelation baseline
signal, and a "persistence" baseline (predict tomorrow = today's direction) alongside
the majority-class one.

### Modality 1 extended: richer text features

`features/text_features.py` adds three feature families on top of Phase 2's same-day
sentiment mean/count:
- **Temporal dynamics** -- rolling 3/7/14-day sentiment and GDELT-tone means, a
  "sentiment surprise" (today vs. trailing 30-day mean), and an article-volume
  z-score, so a news shock still registers on days with zero articles.
- **Theme-group counts** -- daily counts per `config.yaml` -> `theme_groups`
  (econ, epu, trade, conflict, sanctions, crisis, politics), parsed from each
  article's raw GDELT `V2Themes` string -- the most direct geopolitical-shock signal,
  separate from general tone.
- **Source-mix share** -- `id_bisnis_count_share` / `id_kontan_count_share`, isolating
  the Kontan-coverage structural break noted above.

`features/embed_titles.py` adds a semantic-embedding feature: headlines encoded with
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim), mean-pooled
per WIB day. The raw daily vectors are cached as-is and **not** PCA-reduced in this
script -- fitting a PCA on the full dataset before any train/test split would leak
future structure into the training fold, so the reduction (to 16 dims) is fit inside
`modeling/train_refined.py`'s cross-validation loop, on the training fold only, every
fold.

### Model refinement (`modeling/train_refined.py`)

- **Target**: same as Phase 2 (next-trading-day USD/IDR direction), so results are
  comparable, now over the full range instead of the 2024 window.
- **Evaluation**: walk-forward `TimeSeriesSplit` (5 folds, 2-day gap between train and
  validation to keep rolling-window features from leaking across the fold boundary),
  plus an untouched final holdout (last 15% of dates). Metrics: accuracy, macro-F1,
  MCC, ROC-AUC.
- **Models**: regularized logistic regression (scaled features) and LightGBM, run
  over both.
- **Ablation grid** -- the key Phase 3 output, answering "does news add anything
  beyond market data": `market_only`, `text_sentiment_only`, `text_full`,
  `text_full_emb`, `market_text_sentiment`, `market_text_full`, `market_text_full_emb`.
- Outputs: `data/processed/refined_results.json` (full ablation grid, CV means and
  holdout metrics per feature-set/model combination) and
  `data/processed/predictions_refined.parquet` (per-row holdout predictions for every
  combination plus both baselines, for Phase 4 error analysis).

Run the full Phase 3 pipeline with:

```bash
python -m selection.build_queue
python -m preprocessing.clean_text
python -m preprocessing.align_dates
python -m sentiment.run_indobert            # caches by URL; only new rows re-scored
python -m sentiment.build_daily
python -m fx.fetch_market
python -m analysis.merge_dataset
python -m features.text_features
python -m features.embed_titles             # optional but required for *_emb feature sets
python -m modeling.train_refined
```

**Results**: _pending -- `modeling/train_refined.py` is running against the
newly-extended 5-year dataset; the ablation table and holdout metrics will be filled
in here once that run completes._

## Phase 4: Hypothesis Testing, Error Analysis & Final Report

Phase 4 turns Phase 3's ablation grid and holdout predictions into a formal answer to
the project's hypothesis ("global geopolitical news significantly influences and
helps predict USD/IDR"), plus an honest look at where the models succeed or fail.
The write-up itself is delivered as a separate document (Google Docs); this repo's
side is the analysis code and the numbers/figures that document draws on.

`analysis/hypothesis_testing.py`
```
Inputs: data/processed/predictions_refined.parquet, refined_results.json,
        merged_dataset.csv, model_features.parquet
Output: data/processed/hypothesis_test_results.json
```
- **McNemar's test** (paired, same holdout days) on `market_only` vs
  `market_text_full`, `text_sentiment_only` vs `text_full_emb`, and
  `majority_baseline` vs `market_text_full` -- tests whether adding news changes
  which specific days the model gets right/wrong by more than chance would predict.
- **Bootstrap CI** (2,000 resamples) on the accuracy/F1 gap for each pair -- an effect
  size with uncertainty, complementing McNemar's binary significant/not-significant.
- **Logistic-regression coefficient significance** on the news features
  (`statsmodels`, so real p-values), fit on the full pre-holdout window -- tests
  whether news features carry a statistically distinguishable relationship with
  next-day direction on their own, independent of any single classifier's holdout
  score.

Run with: `python -m analysis.hypothesis_testing`

`analysis/error_analysis.py`
```
Inputs: data/processed/predictions_refined.parquet, merged_dataset.csv,
        model_features.parquet
Outputs: data/processed/error_analysis.json
         data/processed/figures/{confusion_matrix_market_text,feature_importance,
                                  error_rate_vs_news_volume}.png
```
- **Contingency breakdown** -- of holdout days, how many does `market_text_full` get
  right that `market_only` gets wrong (and vice versa) -- the concrete version of
  what McNemar's test above scores statistically.
- **Misclassification profile** -- compares news volume, sentiment surprise, and
  theme-group counts on days the best model gets wrong vs. right, and whether errors
  cluster on unusually large moves.
- **Feature importance** -- LightGBM gain-based importance for `market_text_full`,
  refit once on the full pre-holdout window (not a new evaluation).
- **Case studies** -- the 5 largest single-day USD/IDR moves in the holdout, with
  what was predicted and what the news looked like that day.

Run with: `python -m analysis.error_analysis`

**Results and report text**: _pending -- both scripts depend on Phase 3's
`refined_results.json` / `predictions_refined.parquet`, which are still being
produced. Once they exist, this section will carry the hypothesis-test verdict, the
error-analysis findings, and the final report text will be drafted for copy-paste
into the Google Doc._

**Known limitations carried into the final report** (see earlier phases for detail):
the global/FinBERT half of the hypothesis was never pulled (BigQuery billing never
enabled), article bodies were only scraped for 2024 (headline-level sentiment covers
the full 2021-2026 range), Kontan has no GDELT translingual coverage before 2023-03,
and CNBC Indonesia has no GDELT coverage at all.
