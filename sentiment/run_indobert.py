"""Score Indonesian headlines and (where scraped) article bodies with a pretrained
IndoBERT sentiment model.

Uses "mdhugol/indonesia-bert-sentiment-classification" (negative/neutral/positive) via
the Hugging Face pipeline API with top_k=None, so the score is a continuous
P(positive) - P(negative) in [-1, 1] rather than a hard label (the hard label is kept
alongside it).

Input: data/processed/id_articles.parquet (from preprocessing/align_dates.py)
Output: data/processed/id_articles_scored.parquet, same rows plus
  title_sentiment, title_label, body_sentiment (nullable), body_label (nullable)

Rows already scored in an existing id_articles_scored.parquet are reused by URL
instead of re-run through the model (Phase 3 extends the headline set from ~55k
2024-only rows to ~245k rows across 2021-2026, and re-scoring everything already
scored on every rerun would waste most of the runtime). Pass --no-cache to force a
full re-score (e.g. after changing the model or the label mapping).

Run with:
  python -m sentiment.run_indobert --check-labels   # sanity-check the label mapping first
  python -m sentiment.run_indobert
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config

MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
MAX_CHARS = 2000  # transformer truncation still applies at the tokenizer level (512 tokens)
SCORE_COLUMNS = ["title_sentiment", "title_label", "body_sentiment", "body_label"]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

SANITY_POSITIVE = [
    "Ekonomi Indonesia Tumbuh Pesat, Investor Optimistis",
    "Rupiah Menguat Signifikan Terhadap Dolar AS",
    "Laba Perusahaan Melonjak Tajam Tahun Ini",
    "IHSG Ditutup Menguat, Sentimen Pasar Positif",
    "Pemerintah Berhasil Tekan Inflasi ke Level Rendah",
]
SANITY_NEGATIVE = [
    "Ekonomi Indonesia Terancam Resesi Parah",
    "Rupiah Anjlok Tajam Akibat Krisis Global",
    "Perusahaan Merugi Besar, Ribuan Karyawan Terkena PHK",
    "IHSG Anjlok, Investor Panik Jual Saham",
    "Inflasi Melonjak Tinggi, Daya Beli Masyarakat Turun",
]


def build_classifier():
    device = pick_device()
    print(f"[run_indobert] using device={device}")
    return pipeline(
        "text-classification", model=MODEL_NAME, top_k=None, truncation=True, device=device,
    )


def score_texts(classifier, texts, batch_size=64):
    """Returns (signed_scores, hard_labels) using P(positive) - P(negative)."""
    signed_scores, hard_labels = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="scoring"):
        batch = [t[:MAX_CHARS] for t in texts[i:i + batch_size]]
        results = classifier(batch, truncation=True, max_length=512)
        for scores in results:
            by_label = {s["label"]: s["score"] for s in scores}
            pos = by_label.get(LABEL_POSITIVE, 0.0)
            neg = by_label.get(LABEL_NEGATIVE, 0.0)
            signed_scores.append(pos - neg)
            hard_labels.append(max(scores, key=lambda s: s["score"])["label"])
    return signed_scores, hard_labels


# Populated by check_label_mapping() before the first real run; this model's label ids
# are not self-descriptive (LABEL_0/1/2), so we infer which is positive/negative from
# the sanity examples rather than hardcoding an assumption from the model card alone.
LABEL_POSITIVE = None
LABEL_NEGATIVE = None


def check_label_mapping():
    global LABEL_POSITIVE, LABEL_NEGATIVE
    classifier = build_classifier()
    pos_results = classifier(SANITY_POSITIVE)
    neg_results = classifier(SANITY_NEGATIVE)

    def top_label(scores):
        return max(scores, key=lambda s: s["score"])["label"]

    pos_votes = [top_label(r) for r in pos_results]
    neg_votes = [top_label(r) for r in neg_results]
    print("Positive examples predicted as:", pos_votes)
    print("Negative examples predicted as:", neg_votes)

    from collections import Counter
    pos_majority = Counter(pos_votes).most_common(1)[0][0]
    neg_majority = Counter(neg_votes).most_common(1)[0][0]
    if pos_majority == neg_majority:
        raise SystemExit(
            f"Label mapping check inconclusive: both positive and negative sanity "
            f"examples were mostly predicted as '{pos_majority}'. Inspect the model's "
            f"output manually before trusting scores."
        )

    LABEL_POSITIVE, LABEL_NEGATIVE = pos_majority, neg_majority
    print(f"Inferred mapping: positive='{LABEL_POSITIVE}', negative='{LABEL_NEGATIVE}'")
    return LABEL_POSITIVE, LABEL_NEGATIVE


def load_cache(out_path):
    """Previously scored rows, keyed by url, reused instead of re-running the model."""
    if not out_path.exists():
        return None
    cached = pd.read_parquet(out_path)
    if not {"url", *SCORE_COLUMNS}.issubset(cached.columns):
        return None
    return cached[["url", *SCORE_COLUMNS]].drop_duplicates(subset="url", keep="last")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-labels", action="store_true",
                         help="only run the sanity check and print the inferred label mapping")
    parser.add_argument("--no-cache", action="store_true",
                         help="re-score every row instead of reusing an existing scored file")
    args = parser.parse_args()

    global LABEL_POSITIVE, LABEL_NEGATIVE
    LABEL_POSITIVE, LABEL_NEGATIVE = check_label_mapping()
    if args.check_labels:
        return

    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])
    articles_path = processed_dir / "id_articles.parquet"
    if not articles_path.exists():
        raise SystemExit(f"{articles_path} not found -- run preprocessing/align_dates.py first")

    df = pd.read_parquet(articles_path)
    if df.empty:
        raise SystemExit("id_articles.parquet is empty -- nothing to score.")

    out_path = processed_dir / "id_articles_scored.parquet"
    cache = None if args.no_cache else load_cache(out_path)

    if cache is not None:
        df = df.merge(cache, on="url", how="left")
        to_score = df["title_sentiment"].isna()
        print(f"[run_indobert] reusing {(~to_score).sum()} cached rows, "
              f"scoring {to_score.sum()} new rows")
    else:
        to_score = pd.Series(True, index=df.index)
        for col in SCORE_COLUMNS:
            df[col] = pd.NA

    classifier = build_classifier()

    if to_score.any():
        title_scores, title_labels = score_texts(classifier, df.loc[to_score, "title"].tolist())
        df.loc[to_score, "title_sentiment"] = title_scores
        df.loc[to_score, "title_label"] = title_labels

        needs_body = to_score & df["body"].notna()
        if needs_body.any():
            body_scores, body_labels = score_texts(classifier, df.loc[needs_body, "body"].tolist())
            df.loc[needs_body, "body_sentiment"] = body_scores
            df.loc[needs_body, "body_label"] = body_labels

    has_body = df["body_sentiment"].notna()
    df["title_sentiment"] = df["title_sentiment"].astype(float)
    df.to_parquet(out_path, index=False)
    print(f"[run_indobert] wrote {len(df)} scored rows to {out_path}")
    if has_body.any():
        agree = (
            (df.loc[has_body, "title_sentiment"] > 0) ==
            (df.loc[has_body, "body_sentiment"].astype(float) > 0)
        ).mean()
        print(f"[run_indobert] title/body sentiment sign agreement: {agree:.1%}")


if __name__ == "__main__":
    main()
