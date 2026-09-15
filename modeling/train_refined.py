"""Phase 3: model refinement + multimodal integration.

Extends the Phase 2 baseline (modeling/train_baseline.py) three ways:
  1. Uses the full 2021-09..2026-09 date range instead of the 2024-01..2025-01
     window -- news features are no longer sparse-outside-2024 once
     selection/build_queue.py etc. have been re-run for every year (see README).
  2. Adds a market-data modality (fx/fetch_market.py) alongside text.
  3. Uses walk-forward TimeSeriesSplit CV (+ an untouched final holdout) instead of
     one 80/20 split, and runs an ablation grid over feature-set combinations so the
     "does news add anything beyond market data" question has a direct answer.

Inputs (data/processed/):
  merged_dataset.csv        -- news + FX + market, from analysis/merge_dataset.py
  model_features.parquet    -- rolling/theme/source-mix text features, from
                                features/text_features.py
  title_embeddings_daily.parquet (optional) -- raw per-day title embeddings, from
                                features/embed_titles.py; PCA-reduced per CV fold here

Outputs (data/processed/):
  refined_results.json      -- ablation grid: metrics per (feature_set, model)
  predictions_refined.parquet -- per-row holdout predictions for Phase 4 error analysis

Run with: python -m modeling.train_refined
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config

N_SPLITS = 5
CV_GAP = 2                 # days skipped between train and validation fold, avoids
                            # rolling-window features leaking across the boundary
HOLDOUT_FRACTION = 0.15    # untouched final slice, evaluated once at the end
EMBED_PCA_DIMS = 16
RANDOM_STATE = 42

MARKET_COLS = [
    "dxy_pct_change", "brent_pct_change", "gold_pct_change", "vix_pct_change",
    "us10y_pct_change", "ihsg_pct_change", "usdmyr_pct_change", "usdthb_pct_change",
    "usdinr_pct_change",
]
SENTIMENT_COLS = [
    "id_title_sent_mean", "id_title_count", "id_gdelt_tone_mean",
    "id_bisnis_count", "id_kontan_count",
]
DYNAMICS_COLS = [
    "id_title_sent_roll3", "id_title_sent_roll7", "id_title_sent_roll14",
    "id_gdelt_tone_roll3", "id_gdelt_tone_roll7", "id_gdelt_tone_roll14",
    "id_sentiment_surprise", "id_volume_zscore",
    "id_bisnis_count_share", "id_kontan_count_share",
]
THEME_COLS_PREFIX = "id_theme_"
OWN_LAG_COLS = ["usd_idr_pct_change_lag1", "usd_idr_direction_lag1"]

FEATURE_SETS = {
    "market_only": MARKET_COLS + OWN_LAG_COLS,
    "text_sentiment_only": SENTIMENT_COLS + ["has_news"],
    "text_full": SENTIMENT_COLS + DYNAMICS_COLS + ["has_news", "__themes__"],
    "text_full_emb": SENTIMENT_COLS + DYNAMICS_COLS + ["has_news", "__themes__", "__emb__"],
    "market_text_sentiment": MARKET_COLS + OWN_LAG_COLS + SENTIMENT_COLS + ["has_news"],
    "market_text_full": MARKET_COLS + OWN_LAG_COLS + SENTIMENT_COLS + DYNAMICS_COLS
                         + ["has_news", "__themes__"],
    "market_text_full_emb": MARKET_COLS + OWN_LAG_COLS + SENTIMENT_COLS + DYNAMICS_COLS
                             + ["has_news", "__themes__", "__emb__"],
}


def load_data(processed_dir):
    merged_path = processed_dir / "merged_dataset.csv"
    features_path = processed_dir / "model_features.parquet"
    if not merged_path.exists():
        raise SystemExit(f"{merged_path} not found -- run analysis/merge_dataset.py first")
    if not features_path.exists():
        raise SystemExit(f"{features_path} not found -- run features/text_features.py first")

    merged = pd.read_csv(merged_path, parse_dates=["date"])
    text_features = pd.read_parquet(features_path)
    text_features["date"] = pd.to_datetime(text_features["date"])
    overlap = [c for c in text_features.columns if c != "date" and c in merged.columns]
    df = merged.merge(text_features.drop(columns=overlap), on="date", how="left")

    emb_path = processed_dir / "title_embeddings_daily.parquet"
    emb_cols = []
    if emb_path.exists():
        emb = pd.read_parquet(emb_path)
        emb["date"] = pd.to_datetime(emb["date"])
        emb_cols = [c for c in emb.columns if c.startswith("emb_")]
        df = df.merge(emb, on="date", how="left")
    else:
        print("[train_refined] title_embeddings_daily.parquet not found -- "
              "skipping embedding feature set (run features/embed_titles.py)")

    return df, emb_cols


def build_target(df):
    df = df.sort_values("date").reset_index(drop=True)

    df["usd_idr_direction"] = (df["usd_idr_pct_change"] > 0).astype("Int64")
    df.loc[df["usd_idr_pct_change"] == 0, "usd_idr_direction"] = pd.NA

    df["usd_idr_pct_change_lag1"] = df["usd_idr_pct_change"].shift(1)
    df["usd_idr_direction_lag1"] = df["usd_idr_direction"].shift(1).fillna(0).astype(int)

    df["has_news"] = (df["id_title_count"] > 0).astype(int)

    # target: NEXT trading day's direction, from features known at end of day t
    trading = df[df["is_trading_day"]].sort_values("date").reset_index(drop=True)
    trading = trading[trading["usd_idr_pct_change"] != 0]  # drop rare exact-flat days
    trading["target"] = trading["usd_idr_direction"].shift(-1)
    trading = trading.dropna(subset=["target"]).reset_index(drop=True)
    trading["target"] = trading["target"].astype(int)
    return trading


def resolve_columns(feature_set, df, theme_cols, emb_cols):
    cols = []
    for c in feature_set:
        if c == "__themes__":
            cols.extend(theme_cols)
        elif c == "__emb__":
            cols.extend(emb_cols)
        else:
            cols.append(c)
    return [c for c in cols if c in df.columns]


def evaluate(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def fit_predict(model_name, X_train, y_train, X_test, emb_train_raw=None, emb_test_raw=None):
    """Fits PCA (embeddings only, train-fold-only) then the requested model."""
    if emb_train_raw is not None and emb_train_raw.shape[1] > 0:
        n_components = min(EMBED_PCA_DIMS, emb_train_raw.shape[0], emb_train_raw.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        emb_train_reduced = pca.fit_transform(emb_train_raw)
        emb_test_reduced = pca.transform(emb_test_raw)
        X_train = np.hstack([X_train, emb_train_reduced])
        X_test = np.hstack([X_test, emb_test_reduced])

    if model_name == "logistic_regression":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
        ])
    elif model_name == "lightgbm":
        model = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE, verbosity=-1,
        )
    else:
        raise ValueError(model_name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba


def run_ablation(data, theme_cols, emb_cols):
    non_emb_cols = sorted(set(c for cols in FEATURE_SETS.values() for c in cols
                               if c not in ("__themes__", "__emb__")) | set(theme_cols))
    for c in non_emb_cols:
        if c in data.columns:
            data[c] = data[c].fillna(0.0)

    n = len(data)
    holdout_start = int(n * (1 - HOLDOUT_FRACTION))
    cv_data = data.iloc[:holdout_start].reset_index(drop=True)
    holdout = data.iloc[holdout_start:].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=CV_GAP)
    results = []
    holdout_predictions = {}

    for set_name, spec in FEATURE_SETS.items():
        cols = resolve_columns(spec, data, theme_cols, emb_cols)
        use_emb = "__emb__" in spec and emb_cols
        plain_cols = [c for c in cols if c not in emb_cols]

        for model_name in ["logistic_regression", "lightgbm"]:
            fold_metrics = []
            for train_idx, val_idx in tscv.split(cv_data):
                train, val = cv_data.iloc[train_idx], cv_data.iloc[val_idx]
                X_train, X_val = train[plain_cols].values, val[plain_cols].values
                emb_train = train[emb_cols].values if use_emb else None
                emb_val = val[emb_cols].values if use_emb else None
                y_pred, y_proba = fit_predict(
                    model_name, X_train, train["target"].values, X_val, emb_train, emb_val,
                )
                fold_metrics.append(evaluate(val["target"].values, y_pred, y_proba))

            cv_summary = {
                metric: float(np.mean([m[metric] for m in fold_metrics if metric in m]))
                for metric in ["accuracy", "f1_macro", "mcc", "roc_auc"]
                if all(metric in m for m in fold_metrics)
            }

            X_train_full = cv_data[plain_cols].values
            X_holdout = holdout[plain_cols].values
            emb_train_full = cv_data[emb_cols].values if use_emb else None
            emb_holdout = holdout[emb_cols].values if use_emb else None
            y_pred_h, y_proba_h = fit_predict(
                model_name, X_train_full, cv_data["target"].values, X_holdout,
                emb_train_full, emb_holdout,
            )
            holdout_metrics = evaluate(holdout["target"].values, y_pred_h, y_proba_h)

            results.append({
                "feature_set": set_name, "model": model_name, "n_features": len(cols),
                "cv_mean": cv_summary, "holdout": holdout_metrics,
            })
            holdout_predictions[f"{set_name}__{model_name}"] = y_pred_h

    return results, holdout, holdout_predictions


def baseline_metrics(cv_data, holdout):
    results = []

    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(cv_data[["usd_idr_direction_lag1"]], cv_data["target"])
    pred = dummy.predict(holdout[["usd_idr_direction_lag1"]])
    results.append({"feature_set": "majority_baseline", "model": "-", "n_features": 0,
                     "cv_mean": {}, "holdout": evaluate(holdout["target"].values, pred)})

    persistence_pred = holdout["usd_idr_direction_lag1"].values
    results.append({"feature_set": "persistence_baseline", "model": "-", "n_features": 0,
                     "cv_mean": {},
                     "holdout": evaluate(holdout["target"].values, persistence_pred)})
    return results, {"persistence_baseline": persistence_pred}


def main():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])

    df, emb_cols = load_data(processed_dir)
    theme_cols = [c for c in df.columns if c.startswith(THEME_COLS_PREFIX)]
    data = build_target(df)

    print(f"[train_refined] {len(data)} labeled trading-day rows, "
          f"{data['date'].min().date()} to {data['date'].max().date()}")

    results, holdout, holdout_predictions = run_ablation(data, theme_cols, emb_cols)
    cv_data = data.iloc[:len(data) - len(holdout)]
    baseline_results, baseline_predictions = baseline_metrics(cv_data, holdout)
    holdout_predictions.update(baseline_predictions)

    summary = {
        "n_total": len(data), "n_cv": len(cv_data), "n_holdout": len(holdout),
        "holdout_positive_rate": float(holdout["target"].mean()),
        "date_range": {
            "start": str(data["date"].min().date()), "end": str(data["date"].max().date()),
            "holdout_start": str(holdout["date"].min().date()),
        },
        "results": baseline_results + results,
    }

    out_path = processed_dir / "refined_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    pred_df = holdout[["date", "target"]].copy()
    for name, preds in holdout_predictions.items():
        pred_df[f"pred_{name}"] = preds
    pred_df.to_parquet(processed_dir / "predictions_refined.parquet", index=False)

    print(f"[train_refined] holdout: {len(holdout)} rows, "
          f"positive_rate={holdout['target'].mean():.3f}")
    for r in baseline_results + results:
        h = r["holdout"]
        print(f"[train_refined] {r['feature_set']:<24} {r['model']:<20} "
              f"holdout_acc={h['accuracy']:.3f} f1_macro={h['f1_macro']:.3f} "
              f"mcc={h.get('mcc', float('nan')):.3f}")
    print(f"[train_refined] wrote {out_path}")


if __name__ == "__main__":
    main()
