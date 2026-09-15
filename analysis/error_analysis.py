"""Phase 4: error analysis on top of Phase 3's holdout predictions -- where does
adding text help or hurt relative to market-only, what characterizes the days the
best model still gets wrong, which features actually drive it, and what do the
biggest single-day moves in the holdout look like.

Inputs (data/processed/, from Phase 3):
  predictions_refined.parquet, merged_dataset.csv, model_features.parquet

Outputs (data/processed/):
  error_analysis.json
  figures/confusion_matrix_market_text.png
  figures/feature_importance.png
  figures/error_rate_vs_news_volume.png

Run with: python -m analysis.error_analysis
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config

BEST_MODEL_COL = "market_text_full__lightgbm"
MARKET_ONLY_COL = "market_only__lightgbm"
N_CASE_STUDIES = 5


def load_joined(processed_dir):
    pred_path = processed_dir / "predictions_refined.parquet"
    merged_path = processed_dir / "merged_dataset.csv"
    features_path = processed_dir / "model_features.parquet"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found -- run modeling/train_refined.py first")

    preds = pd.read_parquet(pred_path)
    preds = preds.rename(columns={c: c[len("pred_"):] for c in preds.columns if c.startswith("pred_")})
    preds["date"] = pd.to_datetime(preds["date"])

    context_cols = ["date", "usd_idr_pct_change", "id_title_count", "id_title_sent_mean"]
    if merged_path.exists():
        merged = pd.read_csv(merged_path, parse_dates=["date"])
        preds = preds.merge(merged[[c for c in context_cols if c in merged.columns]],
                             on="date", how="left")
    theme_cols = []
    if features_path.exists():
        features = pd.read_parquet(features_path)
        features["date"] = pd.to_datetime(features["date"])
        theme_cols = [c for c in features.columns if c.startswith("id_theme_")]
        extra = ["id_volume_zscore", "id_sentiment_surprise"] + theme_cols
        extra = [c for c in extra if c in features.columns]
        preds = preds.merge(features[["date", *extra]], on="date", how="left")

    return preds, theme_cols


def contingency_breakdown(df, col_a, col_b):
    correct_a = df[col_a] == df["target"]
    correct_b = df[col_b] == df["target"]
    return {
        "both_correct": int((correct_a & correct_b).sum()),
        "both_wrong": int((~correct_a & ~correct_b).sum()),
        "only_a_correct": int((correct_a & ~correct_b).sum()),
        "only_b_correct": int((~correct_a & correct_b).sum()),
    }


def misclassification_profile(df, model_col, theme_cols):
    correct = df[model_col] == df["target"]
    profile_cols = ["id_title_count", "id_volume_zscore", "id_sentiment_surprise"] + theme_cols
    profile_cols = [c for c in profile_cols if c in df.columns]

    wrong_stats = df.loc[~correct, profile_cols].mean(numeric_only=True)
    right_stats = df.loc[correct, profile_cols].mean(numeric_only=True)
    move_wrong = df.loc[~correct, "usd_idr_pct_change"].abs().mean() if "usd_idr_pct_change" in df else None
    move_right = df.loc[correct, "usd_idr_pct_change"].abs().mean() if "usd_idr_pct_change" in df else None

    return {
        "n_wrong": int((~correct).sum()),
        "n_correct": int(correct.sum()),
        "mean_abs_move_wrong_days": float(move_wrong) if move_wrong is not None else None,
        "mean_abs_move_correct_days": float(move_right) if move_right is not None else None,
        "feature_means_wrong_days": wrong_stats.to_dict(),
        "feature_means_correct_days": right_stats.to_dict(),
    }


def feature_importance(processed_dir, theme_cols):
    """Refit market_text_full's LightGBM once on the full CV window (everything
    before the holdout) purely to read off feature importances -- not a new holdout
    evaluation, so this doesn't affect any reported metric.
    """
    merged_path = processed_dir / "merged_dataset.csv"
    features_path = processed_dir / "model_features.parquet"
    if not merged_path.exists() or not features_path.exists():
        return None

    merged = pd.read_csv(merged_path, parse_dates=["date"])
    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"])
    overlap = [c for c in features.columns if c != "date" and c in merged.columns]
    df = merged.merge(features.drop(columns=overlap), on="date", how="left").sort_values("date")

    df["target"] = (df["usd_idr_pct_change"].shift(-1) > 0).astype(float)
    trading = df[df["is_trading_day"] & (df["usd_idr_pct_change"] != 0)].dropna(subset=["target"])
    trading = trading.iloc[:int(len(trading) * 0.85)]  # same holdout split as train_refined.py

    market_cols = [
        "dxy_pct_change", "brent_pct_change", "gold_pct_change", "vix_pct_change",
        "us10y_pct_change", "ihsg_pct_change", "usdmyr_pct_change", "usdthb_pct_change",
        "usdinr_pct_change",
    ]
    text_cols = [
        "id_title_sent_mean", "id_title_count", "id_gdelt_tone_mean",
        "id_bisnis_count", "id_kontan_count", "id_title_sent_roll3", "id_title_sent_roll7",
        "id_title_sent_roll14", "id_sentiment_surprise", "id_volume_zscore",
    ] + theme_cols
    feature_cols = [c for c in market_cols + text_cols if c in trading.columns]

    X = trading[feature_cols].fillna(0.0)
    y = trading["target"].astype(int)

    model = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            random_state=42, verbosity=-1)
    model.fit(X, y)
    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    return dict(sorted(importances.items(), key=lambda kv: -kv[1]))


def case_studies(df, model_col, n=N_CASE_STUDIES):
    if "usd_idr_pct_change" not in df.columns:
        return []
    top = df.reindex(df["usd_idr_pct_change"].abs().sort_values(ascending=False).index).head(n)
    cases = []
    for _, row in top.iterrows():
        cases.append({
            "date": str(row["date"].date()),
            "usd_idr_pct_change": float(row["usd_idr_pct_change"]),
            "actual_direction": int(row["target"]),
            "predicted_direction": int(row[model_col]) if model_col in row and pd.notna(row[model_col]) else None,
            "correct": bool(row[model_col] == row["target"]) if model_col in row else None,
            "id_title_count": float(row["id_title_count"]) if "id_title_count" in row and pd.notna(row["id_title_count"]) else None,
            "id_title_sent_mean": float(row["id_title_sent_mean"]) if "id_title_sent_mean" in row and pd.notna(row["id_title_sent_mean"]) else None,
        })
    return cases


def save_figures(df, importances, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)

    if BEST_MODEL_COL in df.columns:
        cm = confusion_matrix(df["target"], df[BEST_MODEL_COL])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("market_text_full (LightGBM) -- holdout confusion matrix")
        fig.tight_layout()
        fig.savefig(figures_dir / "confusion_matrix_market_text.png", dpi=150)
        plt.close(fig)

    if importances:
        top = list(importances.items())[:15]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh([k for k, _ in reversed(top)], [v for _, v in reversed(top)])
        ax.set_title("LightGBM feature importance (market_text_full)")
        fig.tight_layout()
        fig.savefig(figures_dir / "feature_importance.png", dpi=150)
        plt.close(fig)

    if BEST_MODEL_COL in df.columns and "id_title_count" in df.columns:
        correct = df[BEST_MODEL_COL] == df["target"]
        has_news = df["id_title_count"] > 0
        rates = {
            "no news": 1 - correct[~has_news].mean() if (~has_news).any() else np.nan,
            "has news": 1 - correct[has_news].mean() if has_news.any() else np.nan,
        }
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(rates.keys(), rates.values())
        ax.set_ylabel("Error rate")
        ax.set_title("Error rate: no-news vs. has-news days")
        fig.tight_layout()
        fig.savefig(figures_dir / "error_rate_vs_news_volume.png", dpi=150)
        plt.close(fig)


def main():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])

    df, theme_cols = load_joined(processed_dir)

    result = {}
    if MARKET_ONLY_COL in df.columns and BEST_MODEL_COL in df.columns:
        result["contingency_market_vs_market_text"] = contingency_breakdown(
            df, MARKET_ONLY_COL, BEST_MODEL_COL
        )
    if BEST_MODEL_COL in df.columns:
        result["misclassification_profile"] = misclassification_profile(df, BEST_MODEL_COL, theme_cols)
        result["case_studies_largest_moves"] = case_studies(df, BEST_MODEL_COL)

    importances = feature_importance(processed_dir, theme_cols)
    if importances:
        result["feature_importance"] = importances

    out_path = processed_dir / "error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    save_figures(df, importances, processed_dir / "figures")

    print(f"[error_analysis] wrote {out_path}")
    if "contingency_market_vs_market_text" in result:
        print(f"[error_analysis] market vs market+text: {result['contingency_market_vs_market_text']}")
    if importances:
        top5 = list(importances.items())[:5]
        print(f"[error_analysis] top 5 features: {top5}")
    print(f"[error_analysis] wrote figures to {processed_dir / 'figures'}")


if __name__ == "__main__":
    main()
