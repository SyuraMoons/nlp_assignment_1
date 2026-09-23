-- Cheap sanity check before pulling any real data: confirms all three Indonesian
-- domains actually have translingual GKG coverage across the full 5-year window, and
-- roughly how it's distributed by month. Only scans SourceCommonName, TranslationInfo,
-- and the partition column, so this should be a small, cheap query.
--
-- NOTE: the standard `gdelt-bq.gdeltv2.gkg_partitioned` table is documented as loading
-- GDELT's main (English-language) GKG feed. Whether translingual (TranslationInfo-
-- populated) rows for non-English sources like Kontan/Bisnis/CNBC Indonesia are present
-- in this exact table -- versus only in the separately-published *.translation.gkg.csv.zip
-- files at data.gdeltproject.org -- has NOT been confirmed from inside this session
-- (no BigQuery credentials available here). Run this query first and check the row
-- counts before trusting the rest of the pipeline. If domains come back empty, the
-- fallback is to stream the free translingual GKG files directly (see README for
-- the alternative, gdelt/fetch.py --source=stream).

SELECT
  SourceCommonName,
  FORMAT_DATE('%Y-%m', DATE(_PARTITIONTIME)) AS year_month,
  COUNT(*) AS row_count
FROM `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE _PARTITIONTIME BETWEEN TIMESTAMP('2021-09-01') AND TIMESTAMP('2026-09-15')
  AND TranslationInfo IS NOT NULL AND TranslationInfo != ''
  AND (
    SourceCommonName LIKE '%kontan.co.id%'
    OR SourceCommonName LIKE '%bisnis.com%'
    OR SourceCommonName LIKE '%cnbcindonesia.com%'
  )
GROUP BY SourceCommonName, year_month
ORDER BY SourceCommonName, year_month;
