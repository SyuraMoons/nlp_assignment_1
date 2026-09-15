"""Phase 2 baseline experiments: predict next-day USD/IDR direction (up/down) from
Indonesian news-sentiment features.

News coverage is only real for 2024-01 through 2025-01 (see README's "Phase 2"
section) -- every other month in the 5-year merged dataset has zero articles, so
this restricts training/evaluation to that window rather than the full range.

Input (data/processed/):
  merged_dataset.csv

Output (data/processed/):
  baseline_results.json -- accuracy/F1/confusion matrix per model

Run with: python -m modeling.train_baseline
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import GaussianNB

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config

NEWS_WINDOW_START = "2024-01-01"
NEWS_WINDOW_END = "2025-01-07"  # last date with any article in this dataset

FEATURE_COLUMNS = [
    "id_title_sent_mean",
    "id_title_count",
    "id_gdelt_tone_mean",
    "id_body_sent_mean",
    "id_body_count",
    "id_bisnis_count",
    "id_kontan_count",
]

TEST_FRACTION = 0.2


def load_dataset(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "merged_dataset.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run analysis/merge_dataset.py first")
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def build_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    window = df[
        (df["date"] >= NEWS_WINDOW_START) & (df["date"] <= NEWS_WINDOW_END)
    ].copy()

    trading = window[window["is_trading_day"]].copy()
    trading = trading[trading["usd_idr_pct_change"] != 0]  # drop the rare exact-flat days

    trading["has_news"] = (trading["id_title_count"] > 0).astype(int)
    for col in FEATURE_COLUMNS:
        trading[col] = trading[col].fillna(0.0)

    trading["target"] = (trading["usd_idr_pct_change"] > 0).astype(int)
    return trading


def chronological_split(df: pd.DataFrame):
    df = df.sort_values("date")
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    return train, test


def evaluate(name: str, y_true, y_pred) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])

    df = load_dataset(processed_dir)
    data = build_features_and_target(df)
    train, test = chronological_split(data)

    feature_cols = FEATURE_COLUMNS + ["has_news"]
    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    results = []

    # 1. Naive baseline: always predict the train set's majority class.
    majority_class = y_train.mode()[0]
    y_pred_majority = pd.Series(majority_class, index=y_test.index)
    results.append(evaluate("majority_baseline", y_test, y_pred_majority))

    # 2. Gaussian Naive Bayes.
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    results.append(evaluate("gaussian_naive_bayes", y_test, nb.predict(X_test)))

    # 3. Logistic Regression.
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    results.append(evaluate("logistic_regression", y_test, logreg.predict(X_test)))

    summary = {
        "window": {"start": NEWS_WINDOW_START, "end": NEWS_WINDOW_END},
        "n_train": len(train),
        "n_test": len(test),
        "test_positive_rate": y_test.mean(),
        "results": results,
    }

    out_path = processed_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[train_baseline] window {NEWS_WINDOW_START}..{NEWS_WINDOW_END}, "
          f"train={len(train)} test={len(test)} test_positive_rate={y_test.mean():.3f}")
    for r in results:
        print(f"[train_baseline] {r['model']:<22} accuracy={r['accuracy']:.3f} "
              f"f1_macro={r['f1_macro']:.3f} confusion={r['confusion_matrix']}")
    print(f"[train_baseline] wrote {out_path}")


if __name__ == "__main__":
    main()
