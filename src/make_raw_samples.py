"""Write small, committed samples of every raw source to data/raw_sample/, so the
repository shows what the raw inputs look like without committing the full
(multi-hundred-MB, reproducible) pulls in data/raw/.

  gdelt_id_sample.csv        -- 40 raw GDELT GKG rows from each yearly pull
  articles_<site>_sample.jsonl -- first 20 scraped articles per site
  jisdor_raw.xlsx            -- the Bank Indonesia JISDOR export (small, kept whole)

Run with: python -m src.make_raw_samples
"""
import shutil

import pandas as pd

from src.common.config import PROJECT_ROOT, data_path, load_config

ROWS_PER_YEAR = 40
ARTICLES_PER_SITE = 20


def main():
    config = load_config()
    raw_dir = data_path(config, "raw_dir")
    out_dir = data_path(config, "sample_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_parquet(p).head(ROWS_PER_YEAR)
              for p in sorted(raw_dir.glob("gdelt_id_*.parquet"))]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_dir / "gdelt_id_sample.csv", index=False)

    for path in sorted((raw_dir / "articles").glob("*.jsonl")):
        if path.name.startswith("_"):
            continue
        with open(path, encoding="utf-8") as src, \
             open(out_dir / f"articles_{path.stem}_sample.jsonl", "w", encoding="utf-8") as dst:
            for i, line in enumerate(src):
                if i >= ARTICLES_PER_SITE:
                    break
                dst.write(line)

    jisdor = PROJECT_ROOT / config["fx"]["raw_file"]
    if jisdor.exists():
        shutil.copy(jisdor, out_dir / "jisdor_raw.xlsx")

    print(f"[make_raw_samples] wrote samples to {out_dir}: "
          f"{sorted(p.name for p in out_dir.iterdir())}")


if __name__ == "__main__":
    main()
