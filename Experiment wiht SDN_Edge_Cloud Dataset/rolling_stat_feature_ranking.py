#!/usr/bin/env python3
"""
rolling_stat_feature_ranking.py

Compute three feature-ranking diagnostics for deciding which raw metrics are
worth rolling-window statistics:

1) Variance ratio:
   mean(rolling std) / global std

2) Autocorrelation:
   lag-1, lag-5, lag-15 autocorrelation and a summary score

3) Mutual information:
   mutual information between each raw feature and target columns
   (auto-detects *_bin, *_ttf, *_cause if present)

Outputs CSV files into --outdir.

Usage:
    python rolling_stat_feature_ranking.py --csv data.csv --outdir out_rank
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

try:
    from sklearn.feature_selection import mutual_info_classif
except ImportError as e:
    raise SystemExit(
        "scikit-learn is required for option 3. Install it with: pip install scikit-learn"
    ) from e


EPS = 1e-12


def infer_candidate_features(df: pd.DataFrame) -> List[str]:
    """
    Infer raw candidate features from a dataframe by excluding:
    - target columns: *_ttf, *_bin, *_cause
    - time column: t
    - precomputed rolling features: *_mean_*, *_std_*
    - non-numeric columns
    """
    excluded_substrings = ("_ttf", "_bin", "_cause", "_mean_", "_std_")
    candidates = []

    for col in df.columns:
        if col == "t":
            continue
        if any(s in col for s in excluded_substrings):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)

    return candidates


def infer_target_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect target columns for option 3.
    Preference: *_bin, *_cause.
    *_ttf is numeric too, but MI classification is most natural for binary targets.
    """
    target_cols = []
    for col in df.columns:
        if col.endswith("_bin") or col.endswith("_cause"):
            target_cols.append(col)
    return target_cols


def clean_series(s: pd.Series) -> pd.Series:
    """Convert to numeric and replace inf with nan."""
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return x


def option1_variance_ratio(df: pd.DataFrame, features: List[str], windows: List[int]) -> pd.DataFrame:
    """
    For each feature and each rolling window:
      global_std
      mean_rolling_std
      ratio = mean_rolling_std / global_std

    Higher ratio indicates more short-term local variation relative to overall scale.
    """
    rows = []
    for feat in features:
        x = clean_series(df[feat])
        global_std = float(np.nanstd(x.values, ddof=0))

        for w in windows:
            roll_std = x.rolling(window=w, min_periods=1).std(ddof=0)
            mean_roll_std = float(np.nanmean(roll_std.values))
            ratio = mean_roll_std / (global_std + EPS)

            rows.append({
                "feature": feat,
                "window": w,
                "global_std": global_std,
                "mean_rolling_std": mean_roll_std,
                "variance_ratio": ratio,
            })

    out = pd.DataFrame(rows)
    # Largest ratio first
    out = out.sort_values(["window", "variance_ratio"], ascending=[True, False]).reset_index(drop=True)
    return out


def autocorr_safe(x: pd.Series, lag: int) -> float:
    """Safe autocorrelation for a pandas series."""
    try:
        v = x.autocorr(lag=lag)
        if pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def option2_autocorrelation(df: pd.DataFrame, features: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Compute lag autocorrelation for each feature.
    Summary score:
      short_term_dynamics = mean(1 - abs(acf_lag)) across lags
    Larger score means less persistence / more short-term change.
    """
    rows = []
    for feat in features:
        x = clean_series(df[feat]).astype(float)

        acfs = {}
        abs_vals = []
        for lag in lags:
            acf = autocorr_safe(x, lag)
            acfs[f"acf_lag_{lag}"] = acf
            if not pd.isna(acf):
                abs_vals.append(abs(acf))

        short_term_dynamics = float(np.mean([1.0 - v for v in abs_vals])) if abs_vals else np.nan
        mean_abs_acf = float(np.mean(abs_vals)) if abs_vals else np.nan

        row = {
            "feature": feat,
            "mean_abs_acf": mean_abs_acf,
            "short_term_dynamics_score": short_term_dynamics,
        }
        row.update(acfs)
        rows.append(row)

    out = pd.DataFrame(rows)
    # Largest short-term dynamics first
    out = out.sort_values("short_term_dynamics_score", ascending=False).reset_index(drop=True)
    return out


def mutual_info_for_feature_target(
    x: pd.Series,
    y: pd.Series,
    random_state: int = 0,
) -> float:
    """
    Mutual information between one numeric feature and one discrete target.
    Rows with NaN are dropped.
    """
    x_clean = clean_series(x)
    y_clean = clean_series(y)

    mask = x_clean.notna() & y_clean.notna()
    if mask.sum() < 5:
        return np.nan

    X = x_clean.loc[mask].to_numpy().reshape(-1, 1)
    yy = y_clean.loc[mask].to_numpy()

    # If target is not binary/integer-like, discretize by rounding if it is very close.
    # Otherwise, let sklearn treat it as discrete labels for classification MI.
    try:
        # mutual_info_classif expects discrete target labels for classification
        mi = mutual_info_classif(X, yy, random_state=random_state, discrete_features=False)
        return float(mi[0])
    except Exception:
        return np.nan


def option3_mutual_information(
    df: pd.DataFrame,
    features: List[str],
    target_cols: List[str],
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Compute MI(feature, target) for each feature-target pair.
    Also report:
      mi_mean_over_targets
      mi_max_over_targets
    """
    if not target_cols:
        return pd.DataFrame()

    rows = []
    for feat in features:
        mi_vals = {}
        vals = []
        for tgt in target_cols:
            mi = mutual_info_for_feature_target(df[feat], df[tgt], random_state=random_state)
            mi_vals[f"mi_{tgt}"] = mi
            if not pd.isna(mi):
                vals.append(mi)

        row = {
            "feature": feat,
            "mi_mean_over_targets": float(np.mean(vals)) if vals else np.nan,
            "mi_max_over_targets": float(np.max(vals)) if vals else np.nan,
        }
        row.update(mi_vals)
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values("mi_max_over_targets", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to the dataset CSV")
    ap.add_argument("--outdir", required=True, help="Output directory for ranking CSVs")
    ap.add_argument("--windows", nargs="+", type=int, default=[5, 15], help="Rolling windows")
    ap.add_argument("--lags", nargs="+", type=int, default=[1, 5, 15], help="Autocorrelation lags")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    features = infer_candidate_features(df)
    if not features:
        raise SystemExit("No numeric candidate features found after filtering.")

    target_cols = infer_target_columns(df)

    print(f"[INFO] Loaded: {args.csv}")
    print(f"[INFO] Candidate features: {len(features)}")
    print(f"[INFO] Targets detected for MI: {target_cols if target_cols else 'None'}")

    # Option 1
    opt1 = option1_variance_ratio(df, features, args.windows)
    opt1_path = os.path.join(args.outdir, "option1_variance_ratio.csv")
    opt1.to_csv(opt1_path, index=False)
    print(f"[OK] Wrote {opt1_path}")

    # Option 2
    opt2 = option2_autocorrelation(df, features, args.lags)
    opt2_path = os.path.join(args.outdir, "option2_autocorrelation.csv")
    opt2.to_csv(opt2_path, index=False)
    print(f"[OK] Wrote {opt2_path}")

    # Option 3
    if target_cols:
        opt3 = option3_mutual_information(df, features, target_cols, random_state=args.seed)
        opt3_path = os.path.join(args.outdir, "option3_mutual_information.csv")
        opt3.to_csv(opt3_path, index=False)
        print(f"[OK] Wrote {opt3_path}")
    else:
        print("[WARN] No *_bin or *_cause target columns found. Skipping option 3.")

    # Convenience: merge top rankings into a single summary table
    summary = pd.DataFrame({"feature": features})

    # Rank 1 = best
    opt1_rank = opt1.groupby("feature")["variance_ratio"].mean().rank(ascending=False, method="min")
    opt2_rank = opt2.set_index("feature")["short_term_dynamics_score"].rank(ascending=False, method="min")

    summary["rank_option1"] = summary["feature"].map(opt1_rank)
    summary["rank_option2"] = summary["feature"].map(opt2_rank)

    if target_cols:
        opt3_rank = opt3.set_index("feature")["mi_max_over_targets"].rank(ascending=False, method="min")
        summary["rank_option3"] = summary["feature"].map(opt3_rank)
        summary["avg_rank"] = summary[["rank_option1", "rank_option2", "rank_option3"]].mean(axis=1)
    else:
        summary["avg_rank"] = summary[["rank_option1", "rank_option2"]].mean(axis=1)

    summary = summary.sort_values("avg_rank", ascending=True).reset_index(drop=True)
    summary_path = os.path.join(args.outdir, "combined_feature_ranking.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Wrote {summary_path}")

    print("\nTop features by combined rank:")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()