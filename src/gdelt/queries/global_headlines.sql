-- Global English-language geopolitical headlines from a whitelist of major outlets,
-- filtered by GDELT theme at query time (not downstream) because the full English GKG
-- is enormous -- pulling it unfiltered would blow past a reasonable BigQuery scan
-- budget. This is what feeds FinBERT, covering the "global" half of the hypothesis
-- (the three Indonesian sites only cover the domestic/reaction side).
--
-- Params (substituted by gdelt/fetch.py): @start_date, @end_date ('YYYY-MM-DD'),
-- @source_regex (built from config.yaml's global_sources),
-- @theme_regex (built from config.yaml's global_theme_prefixes)
--
-- RISK (unconfirmed from inside this session, no BigQuery access here): the
-- <PAGE_TITLE> tag in Extras was observed on *translingual* GKG rows (where GDELT
-- machine-translates the title). It is not confirmed that English-original GKG rows
-- populate the same tag. gdelt/fetch.py checks the null-title rate on English rows
-- after the first pull and prints a warning if it's high -- if so, titles for those
-- rows need a lightweight follow-up fetch of just the page's <title> tag (not a full
-- scrape) rather than assuming Extras has them.

SELECT
  GKGRECORDID,
  DATE AS gdelt_timestamp,
  SourceCommonName,
  DocumentIdentifier AS url,
  REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
  V2Themes,
  V2Tone
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME BETWEEN TIMESTAMP(@start_date) AND TIMESTAMP(@end_date)
  AND (TranslationInfo IS NULL OR TranslationInfo = '')  -- English-original articles only
  AND REGEXP_CONTAINS(SourceCommonName, @source_regex)
  AND REGEXP_CONTAINS(V2Themes, @theme_regex)
  AND REGEXP_CONTAINS(Extras, r'<PAGE_TITLE>');
