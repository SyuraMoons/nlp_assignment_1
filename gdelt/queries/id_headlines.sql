-- Indonesian-language headlines from the three target sites, with URL, timestamp,
-- theme tags, and GDELT's own tone score. Filtering to sections/themes and building
-- the scrape queue happens downstream in selection/build_queue.py, not here, so this
-- query stays simple and cheap to rerun.
--
-- Params (substituted by gdelt/fetch.py): @start_date, @end_date (inclusive, 'YYYY-MM-DD')

SELECT
  GKGRECORDID,
  DATE AS gdelt_timestamp,          -- GDELT capture time, UTC (YYYYMMDDHHMMSS as INT64)
  SourceCommonName,
  DocumentIdentifier AS url,
  REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
  V2Themes,
  V2Tone
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME BETWEEN TIMESTAMP(@start_date) AND TIMESTAMP(@end_date)
  AND TranslationInfo IS NOT NULL AND TranslationInfo != ''
  AND (
    SourceCommonName LIKE '%kontan.co.id%'
    OR SourceCommonName LIKE '%bisnis.com%'
    OR SourceCommonName LIKE '%cnbcindonesia.com%'
  )
  AND REGEXP_CONTAINS(Extras, r'<PAGE_TITLE>');
