"""Pull headline data from GDELT via BigQuery.

Auth: run `gcloud auth application-default login` yourself first (Application Default
Credentials) -- this script does not handle authentication.

Usage:
    python -m gdelt.fetch --query coverage
    python -m gdelt.fetch --query id --year 2024 --dry-run
    python -m gdelt.fetch --query id --year 2024
    python -m gdelt.fetch --query global --year 2024

Writes data/raw/gdelt_<id|global>_<year>.parquet (or prints results for --query coverage).
"""
import argparse
import os
import sys
from pathlib import Path

from google.cloud import bigquery

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config

QUERIES_DIR = Path(__file__).resolve().parent / "queries"


def read_sql(name):
    return (QUERIES_DIR / name).read_text()


def build_regex(terms):
    escaped = [t.replace(".", r"\.") for t in terms]
    return "(" + "|".join(escaped) + ")"


def year_bounds(year, config_start, config_end):
    start = max(f"{year}-01-01", config_start)
    end = min(f"{year}-12-31", config_end)
    return start, end


def resolve_project(config):
    project = config.get("gcp_project") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        sys.exit(
            "No GCP project configured for BigQuery. Set one of:\n"
            "  - `gcp_project: <your-project-id>` in config.yaml\n"
            "  - the GOOGLE_CLOUD_PROJECT environment variable\n"
            "The project must have the BigQuery API enabled (billing is free for "
            "GDELT's public dataset, but a project is still required to run queries)."
        )
    return project


def run_query(client, sql, params, dry_run):
    job_config = bigquery.QueryJobConfig(
        query_parameters=params,
        dry_run=dry_run,
        use_query_cache=not dry_run,
    )
    job = client.query(sql, job_config=job_config)
    if dry_run:
        gb = job.total_bytes_processed / 1e9
        print(f"[dry-run] would scan {gb:.2f} GB")
        return None
    return job.result().to_dataframe()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", choices=["coverage", "id", "global"], required=True)
    parser.add_argument("--year", type=int, help="required for --query id/global")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config()
    client = bigquery.Client(project=resolve_project(config))
    date_range = config["date_range"]

    if args.query == "coverage":
        sql = read_sql("coverage.sql")
        df = run_query(client, sql, [], args.dry_run)
        if df is not None:
            print(df.to_string(index=False))
        return

    if args.year is None:
        parser.error("--year is required for --query id/global")

    start_date, end_date = year_bounds(args.year, date_range["start"], date_range["end"])
    params = [
        bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
        bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
    ]

    if args.query == "id":
        sql = read_sql("id_headlines.sql")
        out_name = f"gdelt_id_{args.year}.parquet"
    else:
        gdelt_cfg = config["gdelt"]
        params += [
            bigquery.ScalarQueryParameter(
                "source_regex", "STRING", build_regex(gdelt_cfg["global_sources"])
            ),
            bigquery.ScalarQueryParameter(
                "theme_regex", "STRING", build_regex(gdelt_cfg["global_theme_prefixes"])
            ),
        ]
        sql = read_sql("global_headlines.sql")
        out_name = f"gdelt_global_{args.year}.parquet"

    df = run_query(client, sql, params, args.dry_run)
    if df is None:
        return

    if args.query == "global":
        null_title_rate = df["title"].isna().mean()
        if null_title_rate > 0.3:
            print(
                f"[WARNING] {null_title_rate:.0%} of English-source rows have no title "
                f"extracted from Extras -- see the RISK note in "
                f"gdelt/queries/global_headlines.sql before trusting this data."
            )

    raw_dir = Path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / out_name
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
