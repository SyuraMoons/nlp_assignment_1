"""Phase 3 semantic-embedding feature: encode Indonesian headlines with a
multilingual sentence encoder and mean-pool per WIB day.

The raw per-day embedding (384-dim, one vector per day) is written as-is -- it is
NOT PCA-reduced here. Fitting a PCA/dimensionality reduction on the full dataset
before the train/test split would leak future-day structure into the training fold,
so that reduction is fit inside modeling/train_refined.py's cross-validation loop,
on the training fold only, each time.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 -- small, fast on
CPU/MPS, and trained for cross-lingual semantic similarity (works fine on Indonesian
despite the name), so this doubles as a text representation that Phase 3's global
English headlines could reuse later without retraining anything.

Input (data/processed/): id_articles_scored.parquet
Output (data/processed/): title_embeddings_daily.parquet -- date, emb_0..emb_383

Run with: python -m features.embed_titles
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMB_DIM = 384


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])

    articles_path = processed_dir / "id_articles_scored.parquet"
    if not articles_path.exists():
        raise SystemExit(f"{articles_path} not found -- run sentiment/run_indobert.py first")

    df = pd.read_parquet(articles_path, columns=["date", "title", "url"])

    out_path = processed_dir / "title_embeddings_daily.parquet"
    cache_path = processed_dir / "title_embeddings_raw.parquet"
    emb_cols = [f"emb_{i}" for i in range(EMB_DIM)]

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        df = df.merge(cached[["url", *emb_cols]], on="url", how="left")
        to_embed = df[emb_cols[0]].isna()
    else:
        to_embed = pd.Series(True, index=df.index)
        for c in emb_cols:
            df[c] = np.nan

    if to_embed.any():
        device = pick_device()
        print(f"[embed_titles] embedding {to_embed.sum()} titles on device={device}")
        model = SentenceTransformer(MODEL_NAME, device=device)
        vectors = model.encode(
            df.loc[to_embed, "title"].tolist(),
            batch_size=128, show_progress_bar=True, convert_to_numpy=True,
        )
        df.loc[to_embed, emb_cols] = vectors

    df[["url", *emb_cols]].to_parquet(cache_path, index=False)

    daily = df.groupby("date")[emb_cols].mean().reset_index()
    daily.to_parquet(out_path, index=False)
    print(f"[embed_titles] wrote {len(daily)} daily rows ({EMB_DIM}-dim mean embedding) to {out_path}")


if __name__ == "__main__":
    main()
