"""Score global English geopolitical headlines with ProsusAI/finbert.

Input: data/processed/global_headlines.parquet (from preprocessing/align_dates.py)
Output: data/processed/global_headlines_scored.parquet, same rows plus
  title_sentiment (P(positive) - P(negative), -1..1), title_label

FinBERT's labels are already human-readable ("positive"/"neutral"/"negative"), unlike
the IndoBERT checkpoint used in run_indobert.py, so no label-mapping inference is
needed here.

Run with: python -m sentiment.run_finbert
"""
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config

MODEL_NAME = "ProsusAI/finbert"
MAX_CHARS = 2000


def score_texts(classifier, texts, batch_size=16):
    signed_scores, hard_labels = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="scoring"):
        batch = [t[:MAX_CHARS] for t in texts[i:i + batch_size]]
        results = classifier(batch)
        for scores in results:
            by_label = {s["label"]: s["score"] for s in scores}
            pos = by_label.get("positive", 0.0)
            neg = by_label.get("negative", 0.0)
            signed_scores.append(pos - neg)
            hard_labels.append(max(scores, key=lambda s: s["score"])["label"])
    return signed_scores, hard_labels


def main():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    headlines_path = processed_dir / "global_headlines.parquet"
    out_path = processed_dir / "global_headlines_scored.parquet"

    if not headlines_path.exists():
        raise SystemExit(f"{headlines_path} not found -- run preprocessing/align_dates.py first")

    df = pd.read_parquet(headlines_path)
    if df.empty:
        df["title_sentiment"] = []
        df["title_label"] = []
        df.to_parquet(out_path, index=False)
        print(f"[run_finbert] no global headlines to score -- wrote empty {out_path}")
        return

    classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)
    scores, labels = score_texts(classifier, df["title"].tolist())
    df["title_sentiment"] = scores
    df["title_label"] = labels

    df.to_parquet(out_path, index=False)
    print(f"[run_finbert] wrote {len(df)} scored rows to {out_path}")


if __name__ == "__main__":
    main()
