# Global Geopolitical Event Prediction: Analyzing Impact on Dollar Exchange Rates

**Hypothesis:** Global geopolitical news significantly influences and helps predict fluctuations in the US Dollar (USD) exchange rate.

This report documents our end-to-end NLP and analytical pipeline testing this hypothesis using Indonesian and global news coverage against the USD/IDR exchange rate, across four project tasks.

---

## Task 1: Data Acquisition & Strategic Preprocessing

### News Data Source

We used **GDELT's translingual Global Knowledge Graph (GKG) feed**, queried via Google BigQuery, as our news discovery layer rather than attempting to scrape entire news domains directly. A domain-wide crawl was tested and found infeasible (~1M requests, systematic timeouts, and many undated URLs), so GDELT was used to identify and date relevant articles, which were then scraped for full text from the original publishers.

**Indonesian sources:** Kontan.co.id and Bisnis.com. A third planned source, CNBC Indonesia, was dropped entirely — it has zero GDELT translingual coverage, a genuine data-availability constraint rather than a scraping failure. Kontan's coverage also only begins in March 2023 (a real gap from September 2021 to February 2023); Bisnis has full coverage across the whole window.

**Global sources:** A whitelist of major outlets (Reuters, AP, BBC, Al Jazeera, CNBC, The Guardian, NYT, FT, WSJ, Bloomberg) was configured and filtered to geopolitically relevant GDELT themes (armed conflict, sanctions, military activity, trade/economic policy uncertainty). However, this pull was **never executed** — it was blocked by BigQuery billing limits (no billing account attached to the project; the free-tier 1 TiB/month sandbox quota was exhausted by the Indonesian pulls alone). As a result, all global-source fields in the dataset are currently empty, and the "global" half of the hypothesis has only been tested indirectly through Indonesian domestic reporting on global events.

**Date range:** September 1, 2021 – September 14, 2026 (approximately 5 years).

### Filtering Pipeline

Starting from 492,777 raw GDELT rows pulled for Indonesian sources:

| Stage | Rows remaining |
|---|---|
| Raw GDELT pull | 492,777 |
| After URL dedupe | 492,750 |
| After section filter (drop non-news sections: finance-listicle, lifestyle, sports, etc.) | 252,302 |
| After geopolitical theme filter | 245,162 |
| After empty-title drop | 245,162 |
| After cross-site title dedupe | 244,823 |
| Scrape queue (capped 15 articles/site/day) | 39,837 |
| Full article bodies successfully scraped and cleaned | 10,594 |

Article bodies were fully scraped only for a 2024 pilot window (10,594 of the ~39,837-URL five-year queue); scraping the full five years of bodies was judged out of scope for the course timeline, so body-level features are sparse outside 2024.

### Preprocessing

- **Cleaning:** HTML-unescaping and whitespace normalization on titles; regex-based boilerplate stripping on bodies (bylines, "read also" links, subscription prompts) per source site, followed by a minimum length filter (200 characters).
- **Deduplication:** Cross-site title-key matching (lowercased, whitespace-normalized), keeping the earliest occurrence by timestamp.
- **Timezone alignment:** All timestamps converted to WIB (Asia/Jakarta) and bucketed into calendar days, since IDR trading follows the Jakarta session.

### Sentiment Scoring

We used **IndoBERT** (mdhugol/indonesia-bert-sentiment-classification) to score Indonesian-language titles and bodies. Rather than using the hard positive/neutral/negative label, we computed a continuous signed sentiment score, P(positive) minus P(negative), ranging from -1 to 1, to preserve confidence information rather than collapsing it into three buckets. Label indices from the model were not self-descriptive, so we validated the mapping with a small sanity-check pass over hand-labeled Indonesian sentences before scoring the full corpus.

To validate that headline-only sentiment was a reasonable proxy (given full-text scraping was only available for 2024), we compared title vs. body sentiment sign on the 10,633-article 2024 subset and found 91.9% agreement, with the remaining ~8% disagreement treated as the nuance full-text scraping captures.

FinBERT (ProsusAI/finbert) was implemented for global-source titles but never run, since the global pull returned zero rows.

### Exchange Rate & Feature Table

USD/IDR daily closes were sourced from **Yahoo Finance (USDIDR=X)** via yfinance, chosen over Bank Indonesia's official JISDOR rate because JISDOR has no bulk historical API (page-scraping only). Non-trading days (weekends/holidays, ~530 days) were forward-filled rather than dropped, and flagged with an is_trading_day indicator, so every calendar day has a row and no dates are lost when merging with news data.

The final Task 1 deliverable, merged_dataset.csv, contains **1,840 daily rows** (2021-09-01 to 2026-09-14) with per-day news sentiment aggregates (title/body sentiment means, article counts, GDELT tone) merged against USD/IDR close price and percent change.

---

## Task 2: Pipeline Proposal & Baseline Experimentation

**Task formulation:** Binary classification of next-trading-day USD/IDR direction (up/down) from same-day Indonesian news-sentiment features.

**Scope constraint:** At this stage, Indonesian news features were only populated for a 2024 pilot run, so training was restricted to 2024-01-01 through 2025-01-07 (212 training rows, 53 held-out test rows via a chronological 80/20 split — no shuffling, to avoid look-ahead leakage). Nine exact-zero "flat" days were dropped as an ambiguous third class.

**Features:** title/body sentiment means, article counts (overall and per-source), GDELT tone, and a binary has_news flag (news was present on only ~20% of days even within this window; missing days were filled as neutral).

**Models compared:** majority-class baseline, Gaussian Naive Bayes, Logistic Regression.

**Results:**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority-class baseline | 0.566 | 0.361 |
| Gaussian Naive Bayes | 0.415 | 0.356 |
| Logistic Regression | 0.396 | 0.378 |

Test set: n = 53, positive-class rate 56.6%.

**Finding:** Neither classifier beat the majority-class baseline on accuracy. We interpret this as an honest negative result driven by data scarcity — only 212 training rows, with usable news signal present on roughly one day in five — rather than a flaw in the modeling approach. This motivated Task 3's two main interventions: (1) extending the training window to the full five years, and (2) adding market-derived features that don't depend on sparse news coverage.

---

## Task 3: Model Refinement & Multimodal Integration

*Status: methodology finalized and implemented; results pending completion of the current pipeline run.*

**Data extension:** The filtering/scoring pipeline was re-run against the full five years of already-downloaded raw GDELT data instead of just the 2024 window, growing the filtered headline count from 55,272 (2024-only) to 245,162 (2021–2026).

**Modality 1 — Expanded text features:** rolling 3/7/14-day sentiment and GDELT-tone means, a "sentiment surprise" feature (today vs. trailing 30-day mean), article-volume z-score, per-theme-group daily counts (economy, policy uncertainty, trade, conflict, sanctions, crisis, politics), and source-mix share (to isolate the effect of Kontan's coverage starting only in 2023).

**Modality 1b — Semantic embeddings:** multilingual sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2, 384-dim), mean-pooled per day, with dimensionality reduction (PCA to 16 components) fit only within each cross-validation fold's training data to avoid leakage.

**Modality 2 — Market data:** daily closes for the US Dollar Index (DXY), Brent crude, gold, VIX, US 10-year yield, the Jakarta Composite Index (IHSG), and regional FX peers (USD/MYR, USD/THB, USD/INR), all via yfinance. Series that close outside the Jakarta trading session (DXY, Brent, gold, VIX, US10Y) are lagged by one calendar day to prevent look-ahead leakage; IHSG and regional FX are left unshifted. We also added the exchange rate's own lagged direction as an autocorrelation feature, alongside a naive "persistence" baseline (predict tomorrow equals today).

**Model architectures:** Regularized Logistic Regression and LightGBM, evaluated with walk-forward time-series cross-validation (5 splits, 2-day gap) plus an untouched final 15% holdout, reporting accuracy, macro-F1, Matthews correlation coefficient, and ROC-AUC.

**Ablation design:** Seven feature-set combinations were evaluated per model — market-only, text-sentiment-only, text-full, text-full plus embeddings, market plus text-sentiment, market plus text-full, and market plus text-full plus embeddings — against the majority-class and persistence baselines, to isolate the marginal contribution of news features over market signal alone.

**Results table (to be completed):**

| Feature set | Model | Accuracy | Macro-F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|
| Market only | LightGBM | | | | |
| Market only | Logistic Regression | | | | |
| Text sentiment only | LightGBM | | | | |
| Text full | LightGBM | | | | |
| Text full + embeddings | LightGBM | | | | |
| Market + text sentiment | LightGBM | | | | |
| Market + text full | LightGBM | | | | |
| Market + text full + embeddings | LightGBM | | | | |

---

## Task 4: Hypothesis Testing, Error Analysis & Final Report

*Status: methodology finalized and implemented; requires Task 3 results to run.*

**Statistical hypothesis testing:**

- McNemar's paired test comparing market-only vs. market+text models (both LightGBM and Logistic Regression), and text-sentiment-only vs. text-full+embeddings, to test whether adding news features produces a statistically significant change in per-day correctness.
- Bootstrap confidence intervals (2,000 resamples) on the accuracy and macro-F1 differences between the best multimodal model and the market-only/majority baselines.
- Logistic regression coefficient significance testing on the news-derived features, to assess which specific signals (sentiment, sentiment surprise, article volume, theme categories) are statistically associated with exchange-rate movement, independent of predictive accuracy.

**Error analysis:**

- Contingency analysis (both-correct / both-wrong / market-only-correct / market+text-correct) comparing the best market-only and multimodal models.
- Misclassification profiling: comparing news volume, sentiment surprise, and theme activity on correctly vs. incorrectly classified days.
- Feature importance (LightGBM gain-based) for the best-performing multimodal model.
- Case studies of the five largest single-day USD/IDR moves in the holdout period, examining whether news signal was present and whether the model captured the move.

**Conclusion:** *To be written once Task 3/4 results are available. Will directly address whether the data supports or fails to support the hypothesis, and will explicitly note the limitation that global-source news was never collected due to BigQuery billing constraints, meaning our test is really of Indonesian domestic reaction to geopolitical events rather than global coverage directly.*

---

## Limitations

- **Global news was never collected.** BigQuery billing was not enabled on the project, and the free sandbox quota was exhausted by the Indonesian pulls. All global-source features are empty throughout the project — the hypothesis has only been tested via Indonesian-language coverage of global events, not the global outlets themselves.
- **CNBC Indonesia has zero GDELT translingual coverage** and was excluded entirely.
- **Kontan has no coverage before March 2023**, a structural gap addressed via source-mix-share features rather than imputation.
- **Full-text article bodies were only scraped for 2024**; outside that window, models rely on headline-level sentiment and GDELT tone only.
- **The Task 2 baseline was trained on a narrow 2024 window** (212 rows) due to news sparsity at the time — directly motivating Task 3's full five-year re-run.
- **USD/IDR rates come from Yahoo Finance, not Bank Indonesia's official JISDOR rate** — adequate for a course-level hypothesis test, but not necessarily the rate a trading desk would rely on.
