"""Phase 4: formal test of the project's core hypothesis -- "global geopolitical news
significantly influences and helps predict USD/IDR" -- on top of Phase 3's ablation
grid and holdout predictions.

Three independent angles, so the conclusion doesn't rest on one test:
  1. McNemar's test -- are two models' *specific* right/wrong holdout calls
     different enough to not be chance, for market-only vs market+text pairs.
  2. Bootstrap CI -- how big is the accuracy/F1/MCC gap between those pairs, with
     uncertainty (McNemar answers "is there a difference"; this answers "how much").
  3. Logistic-regression coefficient significance -- do the news features themselves
     carry a statistically distinguishable relationship with next-day direction,
     independent of any one classifier's holdout performance.

Inputs (data/processed/, from Phase 3):
  predictions_refined.parquet, refined_results.json, merged_dataset.csv,
  model_features.parquet

Output (data/processed/):
  hypothesis_test_results.json

Run with: python -m analysis.hypothesis_testing
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config

N_BOOTSTRAP = 2000
RANDOM_STATE = 42

# (baseline, treatment) pairs: does adding the treatment's ingredient change the
# model's specific right/wrong calls in a way too large to be chance?
COMPARISON_PAIRS = [
    ("market_only__lightgbm", "market_text_full__lightgbm", "market -> market+text (LightGBM)"),
    ("market_only__logistic_regression", "market_text_full__logistic_regression",
     "market -> market+text (LogReg)"),
    ("text_sentiment_only__lightgbm", "text_full_emb__lightgbm",
     "sentiment-only -> full text+embeddings (LightGBM)"),
    ("majority_baseline", "market_text_full__lightgbm", "majority baseline -> market+text (LightGBM)"),
]


def load_predictions(processed_dir):
    path = processed_dir / "predictions_refined.parquet"
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run modeling/train_refined.py first (Phase 3 must "
            "finish before Phase 4 hypothesis testing can run)"
        )
    return pd.read_parquet(path)


def run_mcnemar(df, col_a, col_b):
    if col_a not in df.columns or col_b not in df.columns:
        return None
    correct_a = (df[col_a] == df["target"])
    correct_b = (df[col_b] == df["target"])
    # 2x2: rows = a correct?, cols = b correct?
    table = pd.crosstab(correct_a, correct_b).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    ).values
    result = mcnemar(table, exact=(table[0, 1] + table[1, 0]) < 25, correction=True)
    return {
        "n_holdout": len(df),
        "a_correct_b_wrong": int(table[1, 0]),
        "a_wrong_b_correct": int(table[0, 1]),
        "both_correct": int(table[1, 1]),
        "both_wrong": int(table[0, 0]),
        "accuracy_a": float(correct_a.mean()),
        "accuracy_b": float(correct_b.mean()),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_at_0.05": bool(result.pvalue < 0.05),
    }


def bootstrap_ci(df, col_a, col_b, metric_fn, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(df)
    y = df["target"].values
    pred_a, pred_b = df[col_a].values, df[col_b].values
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(metric_fn(y[idx], pred_b[idx]) - metric_fn(y[idx], pred_a[idx]))
    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lower_2.5": float(np.percentile(diffs, 2.5)),
        "ci_upper_97.5": float(np.percentile(diffs, 97.5)),
        "excludes_zero": bool((diffs > 0).all() or (diffs < 0).all()),
    }


def accuracy_fn(y_true, y_pred):
    return (y_true == y_pred).mean()


def f1_macro_fn(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro")


def regression_significance(processed_dir):
    """Does each news feature carry a statistically distinguishable relationship with
    next-day direction, controlling for the others? Uses the same feature set as
    market_text_full (minus embeddings, which aren't individually interpretable),
    fit on the full CV window (all data before the holdout) with statsmodels so we
    get proper p-values, not just a scikit-learn point estimate.
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

    feature_cols = [
        "id_title_sent_mean", "id_title_count", "id_gdelt_tone_mean",
        "id_sentiment_surprise", "id_volume_zscore",
    ] + [c for c in trading.columns if c.startswith("id_theme_")]
    feature_cols = [c for c in feature_cols if c in trading.columns]

    X = trading[feature_cols].fillna(0.0)
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    X = sm.add_constant(X)
    y = trading["target"]

    model = sm.Logit(y, X).fit(disp=0)
    return {
        "n_obs": int(model.nobs),
        "coefficients": {
            name: {"coef": float(model.params[name]), "p_value": float(model.pvalues[name])}
            for name in X.columns
        },
        "significant_features_at_0.05": [
            name for name in feature_cols
            if name in model.pvalues.index and model.pvalues[name] < 0.05
        ],
        "pseudo_r_squared": float(model.prsquared),
    }


def main():
    config = load_config()
    processed_dir = Path(config["paths"]["processed_dir"])

    predictions = load_predictions(processed_dir)
    pred_cols = [c for c in predictions.columns if c.startswith("pred_")]
    predictions = predictions.rename(columns={c: c[len("pred_"):] for c in pred_cols})

    mcnemar_results = {}
    bootstrap_results = {}
    for col_a, col_b, label in COMPARISON_PAIRS:
        mc = run_mcnemar(predictions, col_a, col_b)
        if mc is None:
            print(f"[hypothesis_testing] skipping '{label}': columns not found "
                  f"(check refined_results.json ran the expected feature sets)")
            continue
        mcnemar_results[label] = mc
        bootstrap_results[label] = {
            "accuracy_diff": bootstrap_ci(predictions, col_a, col_b, accuracy_fn),
            "f1_macro_diff": bootstrap_ci(predictions, col_a, col_b, f1_macro_fn),
        }

    regression = regression_significance(processed_dir)

    summary = {
        "mcnemar": mcnemar_results,
        "bootstrap_ci": bootstrap_results,
        "logistic_regression_significance": regression,
    }

    out_path = processed_dir / "hypothesis_test_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[hypothesis_testing] McNemar's test results:")
    for label, mc in mcnemar_results.items():
        sig = "SIGNIFICANT" if mc["significant_at_0.05"] else "not significant"
        print(f"  {label}: p={mc['p_value']:.4f} ({sig}), "
              f"acc {mc['accuracy_a']:.3f} -> {mc['accuracy_b']:.3f}")
    if regression:
        print(f"[hypothesis_testing] significant news features (p<0.05): "
              f"{regression['significant_features_at_0.05']}")
    print(f"[hypothesis_testing] wrote {out_path}")


if __name__ == "__main__":
    main()
