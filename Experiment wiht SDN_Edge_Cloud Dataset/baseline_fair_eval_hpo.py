#!/usr/bin/env python3
"""Fair, resumable baseline evaluation and tuning for MILD comparisons.

This script follows the same core idea as the MILD HPO runner:
1) chronological outer split into train/validation pool and held-out test,
2) hyperparameter selection on validation only,
3) final evaluation on the untouched test set,
4) per-dataset and per-baseline output folders,
5) resumable state and log files.

Baselines covered:
- weighted_kpi
- distance
- logistic
- mlp
- lstm

The search spaces are intentionally narrow and centered around the values you
already used successfully.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    tf = None
    layers = None
    models = None

try:
    from determinism import set_global_determinism
except Exception:
    def set_global_determinism(seed: int) -> None:
        np.random.seed(seed)
        try:
            import random
            random.seed(seed)
        except Exception:
            pass


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

FEATURES = [
    "cpu_pct",
    "ram_pct",
    "storage_pct",
    "snet",
    "sri",
    "cpu_delta",
    "sri_delta",
    "api_latency",
    "analytics_tput",
    "telemetry_queue",
]

LOWER_IS_BAD = {"sri", "snet", "analytics_tput"}

DATASETS = {
    "synthetic": {
        "data": "data/hard/dataset.csv",
        "events": "data/hard/events_by_intent.json",
        "outdir": "out_fair_baselines/synthetic",
        "test_frac": 0.20,
        "base_cfg": {
            "folds": 11,
            "horizon": 120,
            "fp_budget": 1.0,
            "epochs": 30,
            "batch": 512,
            "lookback": 60,
            "lr_c": 1.0,
            "buffer_minutes": 30,
        },
        "candidates": {
            "folds": [5, 7, 11],
            "horizon": [100, 120, 140],
            "fp_budget": [0.8, 1.0, 1.2],
            "epochs": [20, 30, 40],
            "batch": [256, 512],
            "lookback": [40, 60],
            "lr_c": [0.3, 1.0, 3.0],
            "buffer_minutes": [20, 30, 40],
        },
        "acceptance": {"min_fdr": 97.0, "min_lt": 90.0, "max_fp": 6.5, "min_da": 87.0},
        "refs": {"fdr": 98.89, "lt": 100.09, "fp": 4.97, "da": 89.67},
    },
    "microservices": {
        "data": "data_micro/data/hard/dataset.csv",
        "events": "data_micro/data/hard/events_by_intent.json",
        "outdir": "out_fair_baselines/microservices",
        "test_frac": 0.20,
        "base_cfg": {
            "folds": 3,
            "horizon": 50,
            "fp_budget": 3.0,
            "epochs": 20,
            "batch": 128,
            "lookback": 40,
            "lr_c": 1.0,
            "buffer_minutes": 30,
        },
        "candidates": {
            "folds": [3, 5],
            "horizon": [40, 50, 60],
            "fp_budget": [2.0, 3.0, 4.0],
            "epochs": [15, 20, 25],
            "batch": [64, 128, 256],
            "lookback": [30, 40, 50],
            "lr_c": [0.3, 1.0, 3.0],
            "buffer_minutes": [20, 30, 40],
        },
        "acceptance": {"min_fdr": 90.0, "min_lt": 28.0, "max_fp": 10.0, "min_da": 65.0},
        "refs": {"fdr": 91.32, "lt": 31.76, "fp": 8.17, "da": 66.80},
    },
    "ryu": {
        "data": "data_ryu/data/hard/dataset.csv",
        "events": "data_ryu/data/hard/events_by_intent.json",
        "outdir": "out_fair_baselines/ryu",
        "test_frac": 0.20,
        "base_cfg": {
            "folds": 3,
            "horizon": 15,
            "fp_budget": 3.0,
            "epochs": 20,
            "batch": 256,
            "lookback": 40,
            "lr_c": 1.0,
            "buffer_minutes": 30,
        },
        "candidates": {
            "folds": [3, 5],
            "horizon": [10, 15, 20],
            "fp_budget": [2.0, 3.0, 4.0],
            "epochs": [15, 20, 25],
            "batch": [128, 256],
            "lookback": [20, 40, 60],
            "lr_c": [0.3, 1.0, 3.0],
            "buffer_minutes": [15, 30],
        },
        "acceptance": {"min_fdr": 90.0, "min_lt": 7.0, "max_fp": 10.0, "min_da": 87.0},
        "refs": {"fdr": 92.35, "lt": 8.38, "fp": 8.59, "da": 88.72},
    },
}


@dataclass
class State:
    best_cfg: Dict[str, Any]
    best_score: float
    round_idx: int
    param_idx: int
    candidate_idx: int
    last_completed_key: str


def read_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def to_py(v: Any) -> Any:
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def filter_events(events_by_intent: Dict[str, list], t_min: int, t_max: int) -> Dict[str, list]:
    subset = {k: [] for k in events_by_intent}
    for k, evs in events_by_intent.items():
        for e in evs:
            if (e["start"] <= t_max) and (e["end"] >= t_min):
                e2 = dict(e)
                e2["start"] = max(e["start"], int(t_min))
                e2["end"] = min(e["end"], int(t_max))
                e2["failure_time"] = min(max(e["failure_time"], e2["start"] + 1), e2["end"])
                subset[k].append(e2)
    return subset


def make_outer_split(df: pd.DataFrame, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("t").reset_index(drop=True)
    n_test = max(1, int(round(len(df) * test_frac)))
    split = max(1, len(df) - n_test)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


def make_inner_split(df: pd.DataFrame, folds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    n = len(df)
    if n < folds * 20:
        folds = max(2, min(folds, max(2, n // 20)))
    block = max(1, n // folds)
    split = max(1, n - block)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


def build_binary_labels(df_slice: pd.DataFrame, evs: Dict[str, list], horizon: int, intent_ids: List[str]) -> np.ndarray:
    Y = np.zeros((len(df_slice), len(intent_ids)), dtype=np.float32)
    t = df_slice["t"].values
    for j, intent in enumerate(intent_ids):
        for ev in evs.get(intent, []):
            start = max(ev["failure_time"] - horizon, int(t[0]))
            end = min(ev["failure_time"], int(t[-1]) + 1)
            if end > start:
                Y[(t >= start) & (t < end), j] = 1.0
    return Y


def grid_search_params(scores: np.ndarray, tvals: np.ndarray, evs_by_intent: Dict[str, list], intent: str, fp_budget_per_day: float = 1.0) -> Dict[str, float]:
    Ws = [3, 5, 8, 13, 21, 34]
    best = {"W": 5, "tau": 0.5, "lead": -1.0}
    evs = evs_by_intent.get(intent, [])
    if not evs:
        return best
    all_windows = [(e["start"], e["end"]) for v in evs_by_intent.values() for e in v]

    for W in Ws:
        ema = pd.Series(scores, index=tvals).ewm(span=W, adjust=False).mean().values
        smin, smax = float(np.min(ema)), float(np.max(ema))
        if smax == smin:
            continue
        for tau in np.linspace(smin, smax, 60):
            alerts = tvals[ema >= tau]
            fp = sum(1 for a in alerts if not any((s <= a <= e) for (s, e) in all_windows))
            days = (tvals[-1] - tvals[0]) / 1440.0 if len(tvals) > 1 else 1.0
            fp_rate = fp / max(days, 1e-9)
            if fp_rate > fp_budget_per_day:
                continue
            tp, lt = 0, []
            for e in evs:
                in_win = alerts[(alerts >= e["start"]) & (alerts <= e["failure_time"])]
                if in_win.size > 0:
                    tp += 1
                    lt.append(e["failure_time"] - in_win[0])
            avg_lead = float(np.mean(lt)) if lt else 0.0
            if avg_lead > best["lead"]:
                best = {"W": W, "tau": float(tau), "lead": avg_lead}
    return best


def evaluate(per_intent_alerts: Dict[str, np.ndarray], te_events: Dict[str, list], Tte: np.ndarray, score_by_intent: Dict[str, np.ndarray]):
    tp = {k: 0 for k in te_events}
    fn = {k: 0 for k in te_events}
    lt = {k: [] for k in te_events}
    all_windows = [(e_["start"], e_["end"]) for v in te_events.values() for e_ in v]
    fp_total = 0

    for name, alerts in per_intent_alerts.items():
        for eobj in te_events.get(name, []):
            in_win = alerts[(alerts >= eobj["start"]) & (alerts <= eobj["failure_time"])]
            if in_win.size > 0:
                tp[name] += 1
                lt[name].append(int(eobj["failure_time"] - in_win[0]))
            else:
                fn[name] += 1
        for a in alerts:
            if not any((s <= a <= e) for (s, e) in all_windows):
                fp_total += 1

    time_to_idx = {int(t): idx for idx, t in enumerate(Tte)}
    total_events_with_cause = 0
    total_correctly_identified = 0
    conf: Dict[Tuple[str, str], int] = {}

    for true_name, evs in te_events.items():
        for eobj in evs:
            event_type = eobj.get("type", "")
            is_root_cause_event = event_type.endswith("_cause") or event_type.startswith("independent")
            if not is_root_cause_event:
                continue

            total_events_with_cause += 1
            earliest = None
            s_pred = set()
            for name, arr in per_intent_alerts.items():
                arr2 = arr[(arr >= eobj["start"]) & (arr <= eobj["failure_time"])]
                if arr2.size > 0:
                    t0 = arr2[0]
                    if (earliest is None) or (t0 < earliest):
                        earliest = t0
                        s_pred = {name}
                    elif t0 == earliest:
                        s_pred.add(name)

            if earliest is None:
                continue
            idx = time_to_idx.get(int(earliest), None)
            if idx is None:
                continue

            if len(s_pred) == 1:
                pred = next(iter(s_pred))
            else:
                pred = max(s_pred, key=lambda name: score_by_intent[name][idx])

            if pred == true_name:
                total_correctly_identified += 1
            conf[(true_name, pred)] = conf.get((true_name, pred), 0) + 1

    return tp, fn, lt, fp_total, total_correctly_identified, total_events_with_cause, conf


def compute_metrics(tp, fn, lt, fp_total, correct_cause, total_cause, Tte: np.ndarray) -> Dict[str, Any]:
    per_intent = {}
    for k in tp:
        per_intent[k] = {
            "detection_rate": float(tp[k] / max(tp[k] + fn[k], 1e-9)),
            "avg_lead_time": float(np.mean(lt[k])) if lt[k] else 0.0,
        }
    days = (Tte[-1] - Tte[0]) / 1440.0 if len(Tte) > 1 else 1.0
    return {
        "per_intent": per_intent,
        "overall": {
            "fdr": float(np.mean([v["detection_rate"] for v in per_intent.values()]) * 100.0),
            "lt": float(np.mean([v["avg_lead_time"] for v in per_intent.values()])),
            "fp": float(fp_total / max(days, 1e-9)),
            "da": float((correct_cause / max(total_cause, 1)) * 100.0 if total_cause > 0 else 0.0),
            "num_cause_events": int(total_cause),
        },
    }


def compute_score(metrics: Dict[str, float], refs: Dict[str, float]) -> float:
    fdr = metrics["fdr"]
    lt = metrics["lt"]
    fp = metrics["fp"]
    da = metrics["da"]

    fdr_n = min(max(fdr / max(refs["fdr"], 1e-6), 0.0), 1.2) / 1.2
    lt_n = min(max(lt / max(refs["lt"], 1e-6), 0.0), 1.2) / 1.2
    da_n = min(max(da / max(refs["da"], 1e-6), 0.0), 1.2) / 1.2
    fp_n = min(max(fp / max(refs["fp"], 1e-6), 0.0), 1.5) / 1.5
    return 0.35 * fdr_n + 0.25 * lt_n + 0.25 * da_n + 0.15 * (1.0 - fp_n)


def is_acceptable(metrics: Dict[str, float], acceptance: Dict[str, float]) -> bool:
    return (
        metrics["fdr"] >= acceptance["min_fdr"]
        and metrics["lt"] >= acceptance["min_lt"]
        and metrics["fp"] <= acceptance["max_fp"]
        and metrics["da"] >= acceptance["min_da"]
    )


def create_mlp(num_features: int, intent_ids: List[str], widths=(96, 96), head_units: int = 16, dropout: float = 0.2):
    if tf is None:
        raise ImportError("TensorFlow is required for the MLP baseline")
    x_in = layers.Input(shape=(num_features,), name="x")
    x = x_in
    for u in widths:
        x = layers.Dense(u, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
    outs = []
    for intent in intent_ids:
        h = layers.Dense(head_units, activation="relu", name=f"head_{intent}")(x)
        out = layers.Dense(1, activation="sigmoid", name=f"{intent}_out")(h)
        outs.append(out)
    return models.Model(x_in, outs, name="MLP_Baseline")


def create_lstm(input_shape: Tuple[int, int], intent_ids: List[str], units: int = 64, dropout: float = 0.2):
    if tf is None:
        raise ImportError("TensorFlow is required for the LSTM baseline")
    x_in = layers.Input(shape=input_shape, name="x_seq")
    x = layers.LSTM(units, return_sequences=False)(x_in)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    outs = []
    for intent in intent_ids:
        h = layers.Dense(16, activation="relu", name=f"head_{intent}")(x)
        out = layers.Dense(1, activation="sigmoid", name=f"{intent}_out")(h)
        outs.append(out)
    return models.Model(x_in, outs, name="LSTM_Baseline")


def make_seq_data(df: pd.DataFrame, X_raw: np.ndarray, Y_raw: np.ndarray, lookback: int):
    X_seq, Y_seq, T_seq = [], [], []
    t = df["t"].values
    for end in range(lookback - 1, len(df)):
        X_seq.append(X_raw[end - lookback + 1 : end + 1])
        Y_seq.append(Y_raw[end])
        T_seq.append(t[end])
    return np.asarray(X_seq, dtype=np.float32), np.asarray(Y_seq, dtype=np.float32), np.asarray(T_seq, dtype=np.int64)


def predict_thresholds(scores_val: Dict[str, np.ndarray], Tval: np.ndarray, val_events: Dict[str, list], fp_budget: float):
    params = {k: grid_search_params(scores_val[k], Tval, val_events, k, fp_budget) for k in scores_val}
    score_by_intent, per_intent_alerts = {}, {}
    for k in scores_val:
        W, tau = params[k]["W"], params[k]["tau"]
        ema = pd.Series(scores_val[k], index=Tval).ewm(span=W, adjust=False).mean().values
        score_by_intent[k] = ema
        per_intent_alerts[k] = Tval[ema >= tau]
    return params, score_by_intent, per_intent_alerts


def train_eval_weighted_kpi(train_df, val_df, events_by_intent, intent_ids, cfg, outdir):
    tr_events = filter_events(events_by_intent, train_df["t"].min(), train_df["t"].max())
    va_events = filter_events(events_by_intent, val_df["t"].min(), val_df["t"].max())

    scaler = StandardScaler().fit(train_df[FEATURES].values)
    Xtr = scaler.transform(train_df[FEATURES].values)
    Xva = scaler.transform(val_df[FEATURES].values)
    Xtr_df = pd.DataFrame(Xtr, columns=FEATURES)
    Xva_df = pd.DataFrame(Xva, columns=FEATURES)

    Ytr = build_binary_labels(train_df, tr_events, int(cfg["horizon"]), intent_ids)
    coefs = {}
    for j, name in enumerate(intent_ids):
        y = Ytr[:, j]
        if np.unique(y).size < 2:
            coefs[name] = np.zeros(len(FEATURES))
            continue
        lr = LogisticRegression(max_iter=300, random_state=int(cfg.get("seed", 42)), class_weight="balanced", C=float(cfg["lr_c"]))
        lr.fit(Xtr, y)
        coefs[name] = np.abs(lr.coef_[0])

    def risk_weighted(frame: pd.DataFrame, weights: np.ndarray):
        total = float(np.sum(weights)) if float(np.sum(weights)) > 1e-9 else 1.0
        w = weights / total
        score = np.zeros(len(frame), dtype=np.float32)
        for i, feat in enumerate(FEATURES):
            s = frame[feat].values
            if feat in LOWER_IS_BAD:
                s = -s
            score += float(w[i]) * s
        if score.max() == score.min():
            return np.zeros_like(score)
        return (score - score.min()) / (score.max() - score.min() + 1e-9)

    train_scores = {k: risk_weighted(Xtr_df, coefs[k]) for k in intent_ids}
    val_scores = {k: risk_weighted(Xva_df, coefs[k]) for k in intent_ids}
    params, score_by_intent, per_intent_alerts = predict_thresholds(train_scores, train_df["t"].values, tr_events, float(cfg["fp_budget"]))
    # use validation thresholds selected on train signal, then validate on val set with same settings
    params, val_score_by_intent, val_alerts = predict_thresholds(val_scores, val_df["t"].values, va_events, float(cfg["fp_budget"]))
    tp, fn, lt, fp, correct_cause, total_cause, conf = evaluate(val_alerts, va_events, val_df["t"].values, val_score_by_intent)
    metrics = compute_metrics(tp, fn, lt, fp, correct_cause, total_cause, val_df["t"].values)
    return {
        "scaler": scaler,
        "coefs": coefs,
        "params": params,
        "metrics": metrics,
        "artifact": {"type": "weighted_kpi"},
    }


def train_eval_distance(train_df, val_df, events_by_intent, intent_ids, cfg, outdir):
    tr_events = filter_events(events_by_intent, train_df["t"].min(), train_df["t"].max())
    va_events = filter_events(events_by_intent, val_df["t"].min(), val_df["t"].max())

    scaler = StandardScaler().fit(train_df[FEATURES].values)
    Xtr = scaler.transform(train_df[FEATURES].values)
    Xva = scaler.transform(val_df[FEATURES].values)
    Xtr_df = pd.DataFrame(Xtr, columns=FEATURES)
    Xva_df = pd.DataFrame(Xva, columns=FEATURES)

    all_event_times_tr = set()
    buffer_minutes = int(cfg["buffer_minutes"])
    for intent_events in tr_events.values():
        for e in intent_events:
            start_exclude = max(int(train_df["t"].min()), e["start"] - buffer_minutes)
            end_exclude = min(int(train_df["t"].max()), e["end"] + buffer_minutes)
            all_event_times_tr.update(range(start_exclude, end_exclude + 1))

    golden_indices = train_df[~train_df["t"].isin(all_event_times_tr)].index
    if len(golden_indices) < 100:
        target_unscaled = train_df[FEATURES].mean()
    else:
        target_unscaled = train_df.loc[golden_indices, FEATURES].mean()
    target_scaled = scaler.transform(target_unscaled.values.reshape(1, -1))[0]
    target_dict = dict(zip(FEATURES, target_scaled))

    feat_sel = {
        "telemetry": ["sri", "snet", "ram_pct", "cpu_pct", "telemetry_queue"],
        "api": ["api_latency", "cpu_pct", "sri"],
        "analytics": ["snet", "analytics_tput", "cpu_pct"],
    }

    def score_frame(frame: pd.DataFrame, intent: str):
        sub_feats = feat_sel[intent]
        target_subset = np.array([target_dict[f] for f in sub_feats])
        s = np.sqrt(np.sum((frame[sub_feats].values - target_subset) ** 2, axis=1))
        smin, smax = s.min(), s.max()
        return (s - smin) / (smax - smin + 1e-9) if (smax - smin) > 1e-9 else np.zeros_like(s)

    train_scores = {k: score_frame(Xtr_df, k) for k in intent_ids}
    val_scores = {k: score_frame(Xva_df, k) for k in intent_ids}
    params, val_score_by_intent, val_alerts = predict_thresholds(train_scores, train_df["t"].values, tr_events, float(cfg["fp_budget"]))
    params, val_score_by_intent, val_alerts = predict_thresholds(val_scores, val_df["t"].values, va_events, float(cfg["fp_budget"]))
    tp, fn, lt, fp, correct_cause, total_cause, conf = evaluate(val_alerts, va_events, val_df["t"].values, val_score_by_intent)
    metrics = compute_metrics(tp, fn, lt, fp, correct_cause, total_cause, val_df["t"].values)
    return {
        "scaler": scaler,
        "target": target_dict,
        "params": params,
        "metrics": metrics,
        "artifact": {"type": "distance", "buffer_minutes": buffer_minutes},
    }


def train_eval_logistic(train_df, val_df, events_by_intent, intent_ids, cfg, outdir):
    tr_events = filter_events(events_by_intent, train_df["t"].min(), train_df["t"].max())
    va_events = filter_events(events_by_intent, val_df["t"].min(), val_df["t"].max())

    scaler = StandardScaler().fit(train_df[FEATURES].values)
    Xtr = scaler.transform(train_df[FEATURES].values)
    Xva = scaler.transform(val_df[FEATURES].values)

    Ytr = build_binary_labels(train_df, tr_events, int(cfg["horizon"]), intent_ids)
    scores_train, scores_val = {}, {}
    models_lr = {}
    for j, name in enumerate(intent_ids):
        y = Ytr[:, j]
        if np.unique(y).size < 2:
            scores_train[name] = np.zeros(len(Xtr), dtype=np.float32)
            scores_val[name] = np.zeros(len(Xva), dtype=np.float32)
            models_lr[name] = None
            continue
        lr = LogisticRegression(max_iter=300, random_state=int(cfg.get("seed", 42)), class_weight="balanced", C=float(cfg["lr_c"]))
        lr.fit(Xtr, y)
        models_lr[name] = lr
        scores_train[name] = lr.predict_proba(Xtr)[:, 1]
        scores_val[name] = lr.predict_proba(Xva)[:, 1]

    params, val_score_by_intent, val_alerts = predict_thresholds(scores_train, train_df["t"].values, tr_events, float(cfg["fp_budget"]))
    params, val_score_by_intent, val_alerts = predict_thresholds(scores_val, val_df["t"].values, va_events, float(cfg["fp_budget"]))
    tp, fn, lt, fp, correct_cause, total_cause, conf = evaluate(val_alerts, va_events, val_df["t"].values, val_score_by_intent)
    metrics = compute_metrics(tp, fn, lt, fp, correct_cause, total_cause, val_df["t"].values)
    return {
        "scaler": scaler,
        "models": models_lr,
        "params": params,
        "metrics": metrics,
        "artifact": {"type": "logistic"},
    }


def train_eval_mlp(train_df, val_df, events_by_intent, intent_ids, cfg, outdir):
    tr_events = filter_events(events_by_intent, train_df["t"].min(), train_df["t"].max())
    va_events = filter_events(events_by_intent, val_df["t"].min(), val_df["t"].max())

    scaler = StandardScaler().fit(train_df[FEATURES].values)
    Xtr = scaler.transform(train_df[FEATURES].values)
    Xva = scaler.transform(val_df[FEATURES].values)
    Ytr = build_binary_labels(train_df, tr_events, int(cfg["horizon"]), intent_ids)
    Yva = build_binary_labels(val_df, va_events, int(cfg["horizon"]), intent_ids)

    ytr = {f"{name}_out": Ytr[:, j] for j, name in enumerate(intent_ids)}
    yva = {f"{name}_out": Yva[:, j] for j, name in enumerate(intent_ids)}

    if tf is None:
        raise ImportError("TensorFlow is required for the MLP baseline")
    tf.keras.backend.clear_session()
    model = create_mlp(Xtr.shape[1], intent_ids)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    model.fit(Xtr, ytr, epochs=int(cfg["epochs"]), batch_size=int(cfg["batch"]), validation_data=(Xva, yva), callbacks=cb, verbose=0)

    preds_val = model.predict(Xva, verbose=0)
    scores_val = {name: preds_val[j].reshape(-1) for j, name in enumerate(intent_ids)}
    params, val_score_by_intent, val_alerts = predict_thresholds(scores_val, val_df["t"].values, va_events, float(cfg["fp_budget"]))
    tp, fn, lt, fp, correct_cause, total_cause, conf = evaluate(val_alerts, va_events, val_df["t"].values, val_score_by_intent)
    metrics = compute_metrics(tp, fn, lt, fp, correct_cause, total_cause, val_df["t"].values)
    return {
        "scaler": scaler,
        "model": model,
        "params": params,
        "metrics": metrics,
        "artifact": {"type": "mlp"},
    }


def train_eval_lstm(train_df, val_df, events_by_intent, intent_ids, cfg, outdir):
    tr_events = filter_events(events_by_intent, train_df["t"].min(), train_df["t"].max())
    va_events = filter_events(events_by_intent, val_df["t"].min(), val_df["t"].max())

    scaler = StandardScaler().fit(train_df[FEATURES].values)
    Xtr_raw = scaler.transform(train_df[FEATURES].values)
    Xva_raw = scaler.transform(val_df[FEATURES].values)
    Ytr_raw = build_binary_labels(train_df, tr_events, int(cfg["horizon"]), intent_ids)
    Yva_raw = build_binary_labels(val_df, va_events, int(cfg["horizon"]), intent_ids)

    lookback = int(cfg["lookback"])
    Xtr, Ytr, Ttr = make_seq_data(train_df, Xtr_raw, Ytr_raw, lookback)
    Xva, Yva, Tva = make_seq_data(val_df, Xva_raw, Yva_raw, lookback)
    if len(Xtr) < 20 or len(Xva) < 20:
        raise ValueError("not enough sequence data after lookback")

    ytr = {f"{name}_out": Ytr[:, j] for j, name in enumerate(intent_ids)}
    yva = {f"{name}_out": Yva[:, j] for j, name in enumerate(intent_ids)}

    if tf is None:
        raise ImportError("TensorFlow is required for the LSTM baseline")
    tf.keras.backend.clear_session()
    model = create_lstm((lookback, len(FEATURES)), intent_ids)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)]
    model.fit(Xtr, ytr, epochs=int(cfg["epochs"]), batch_size=int(cfg["batch"]), validation_data=(Xva, yva), callbacks=cb, verbose=0)

    preds_val = model.predict(Xva, verbose=0)
    scores_val = {name: preds_val[j].reshape(-1) for j, name in enumerate(intent_ids)}
    val_events_seq = filter_events(events_by_intent, val_df["t"].iloc[lookback - 1], val_df["t"].iloc[-1])
    params, val_score_by_intent, val_alerts = predict_thresholds(scores_val, Tva, val_events_seq, float(cfg["fp_budget"]))
    tp, fn, lt, fp, correct_cause, total_cause, conf = evaluate(val_alerts, val_events_seq, Tva, val_score_by_intent)
    metrics = compute_metrics(tp, fn, lt, fp, correct_cause, total_cause, Tva)
    return {
        "scaler": scaler,
        "model": model,
        "params": params,
        "metrics": metrics,
        "artifact": {"type": "lstm", "lookback": lookback},
    }

def evaluate_test_from_artifact(baseline: str, bundle, cfg: Dict[str, Any], train_df, val_df, test_df, events_by_intent, intent_ids):
    te_events = filter_events(events_by_intent, test_df["t"].min(), test_df["t"].max())
    Tte = test_df["t"].values

    if baseline == "weighted_kpi":
        scaler = bundle["scaler"]
        coefs = bundle["coefs"]
        Xte = scaler.transform(test_df[FEATURES].values)
        Xte_df = pd.DataFrame(Xte, columns=FEATURES)

        def risk_weighted(frame: pd.DataFrame, weights: np.ndarray):
            total = float(np.sum(weights)) if float(np.sum(weights)) > 1e-9 else 1.0
            w = weights / total
            score = np.zeros(len(frame), dtype=np.float32)
            for i, feat in enumerate(FEATURES):
                s = frame[feat].values
                if feat in LOWER_IS_BAD:
                    s = -s
                score += float(w[i]) * s
            if score.max() == score.min():
                return np.zeros_like(score)
            return (score - score.min()) / (score.max() - score.min() + 1e-9)

        scores = {k: risk_weighted(Xte_df, coefs[k]) for k in intent_ids}
        params = bundle["params"]
        score_by_intent, alerts = {}, {}
        for k in intent_ids:
            W, tau = params[k]["W"], params[k]["tau"]
            ema = pd.Series(scores[k], index=Tte).ewm(span=W, adjust=False).mean().values
            score_by_intent[k] = ema
            alerts[k] = Tte[ema >= tau]

    elif baseline == "distance":
        scaler = bundle["scaler"]
        target = bundle["target"]
        Xte = scaler.transform(test_df[FEATURES].values)
        Xte_df = pd.DataFrame(Xte, columns=FEATURES)
        feat_sel = {
            "telemetry": ["sri", "snet", "ram_pct", "cpu_pct", "telemetry_queue"],
            "api": ["api_latency", "cpu_pct", "sri"],
            "analytics": ["snet", "analytics_tput", "cpu_pct"],
        }
        def score_frame(frame: pd.DataFrame, intent: str):
            sub_feats = feat_sel[intent]
            target_subset = np.array([target[f] for f in sub_feats])
            s = np.sqrt(np.sum((frame[sub_feats].values - target_subset) ** 2, axis=1))
            smin, smax = s.min(), s.max()
            return (s - smin) / (smax - smin + 1e-9) if (smax - smin) > 1e-9 else np.zeros_like(s)
        scores = {k: score_frame(Xte_df, k) for k in intent_ids}
        params = bundle["params"]
        score_by_intent, alerts = {}, {}
        for k in intent_ids:
            W, tau = params[k]["W"], params[k]["tau"]
            ema = pd.Series(scores[k], index=Tte).ewm(span=W, adjust=False).mean().values
            score_by_intent[k] = ema
            alerts[k] = Tte[ema >= tau]

    elif baseline == "logistic":
        scaler = bundle["scaler"]
        models_lr = bundle["models"]
        Xte = scaler.transform(test_df[FEATURES].values)
        scores = {}
        for name in intent_ids:
            lr = models_lr.get(name)
            if lr is None:
                scores[name] = np.zeros(len(Xte), dtype=np.float32)
            else:
                scores[name] = lr.predict_proba(Xte)[:, 1]
        params = bundle["params"]
        score_by_intent, alerts = {}, {}
        for k in intent_ids:
            W, tau = params[k]["W"], params[k]["tau"]
            ema = pd.Series(scores[k], index=Tte).ewm(span=W, adjust=False).mean().values
            score_by_intent[k] = ema
            alerts[k] = Tte[ema >= tau]

    elif baseline == "mlp":
        scaler = bundle["scaler"]
        model = bundle["model"]
        Xte = scaler.transform(test_df[FEATURES].values)
        preds = model.predict(Xte, verbose=0)
        scores = {name: preds[j].reshape(-1) for j, name in enumerate(intent_ids)}
        params = bundle["params"]
        score_by_intent, alerts = {}, {}
        for k in intent_ids:
            W, tau = params[k]["W"], params[k]["tau"]
            ema = pd.Series(scores[k], index=Tte).ewm(span=W, adjust=False).mean().values
            score_by_intent[k] = ema
            alerts[k] = Tte[ema >= tau]

    elif baseline == "lstm":
        scaler = bundle["scaler"]
        model = bundle["model"]
        #lookback = int(bundle["lookback"])
        lookback = int(bundle.get("lookback", cfg["lookback"]))
        Xte_raw = scaler.transform(test_df[FEATURES].values)
        Yte_raw = build_binary_labels(test_df, te_events, int(cfg["horizon"]), intent_ids)
        Xte, Yte, Tte_seq = make_seq_data(test_df, Xte_raw, Yte_raw, lookback)
        te_events_seq = filter_events(events_by_intent, int(Tte_seq[0]), int(Tte_seq[-1]))
        preds = model.predict(Xte, verbose=0)
        scores = {name: preds[j].reshape(-1) for j, name in enumerate(intent_ids)}
        params = bundle["params"]
        score_by_intent, alerts = {}, {}
        for k in intent_ids:
            W, tau = params[k]["W"], params[k]["tau"]
            ema = pd.Series(scores[k], index=Tte_seq).ewm(span=W, adjust=False).mean().values
            score_by_intent[k] = ema
            alerts[k] = Tte_seq[ema >= tau]
        tp, fn, lt, fp, correct_cause, total_cause, conf = evaluate(alerts, te_events_seq, Tte_seq, score_by_intent)
        metrics = compute_metrics(tp, fn, lt, fp, correct_cause, total_cause, Tte_seq)
        return metrics, {f"{a}->{b}": c for (a, b), c in conf.items()}, pd.DataFrame({"t": Tte_seq, **{f"ema_{k}": v for k, v in score_by_intent.items()}})
    else:
        raise ValueError(f"unknown baseline: {baseline}")

    tp, fn, lt, fp, correct_cause, total_cause, conf = evaluate(alerts, te_events, Tte, score_by_intent)
    metrics = compute_metrics(tp, fn, lt, fp, correct_cause, total_cause, Tte)
    return metrics, {f"{a}->{b}": c for (a, b), c in conf.items()}, pd.DataFrame({"t": Tte, **{f"ema_{k}": v for k, v in score_by_intent.items()}})


def baseline_runner(baseline: str):
    if baseline == "weighted_kpi":
        return train_eval_weighted_kpi
    if baseline == "distance":
        return train_eval_distance
    if baseline == "logistic":
        return train_eval_logistic
    if baseline == "mlp":
        return train_eval_mlp
    if baseline == "lstm":
        return train_eval_lstm
    raise ValueError(baseline)


def candidate_space(dataset_name: str, baseline: str, spec: Dict[str, Any]) -> Dict[str, List[Any]]:
    c = spec["candidates"]
    if baseline == "weighted_kpi":
        return {k: c[k] for k in ["folds", "horizon", "fp_budget", "lr_c"] if k in c}
    if baseline == "distance":
        return {k: c[k] for k in ["folds", "fp_budget", "buffer_minutes"] if k in c}
    if baseline == "logistic":
        return {k: c[k] for k in ["folds", "horizon", "fp_budget", "lr_c"] if k in c}
    if baseline == "mlp":
        return {k: c[k] for k in ["folds", "horizon", "fp_budget", "epochs", "batch"] if k in c}
    if baseline == "lstm":
        return {k: c[k] for k in ["folds", "horizon", "fp_budget", "epochs", "batch", "lookback"] if k in c}
    raise ValueError(baseline)


def eval_cfg(dataset_name: str, baseline: str, cfg: Dict[str, Any], df: pd.DataFrame, events_by_intent: Dict[str, list], outdir: Path):
    intent_ids = sorted(events_by_intent.keys())
    trainval_df, test_df = make_outer_split(df, DATASETS[dataset_name]["test_frac"])
    train_df, val_df = make_inner_split(trainval_df, int(cfg["folds"]))
    runner = baseline_runner(baseline)
    val_art = runner(train_df, val_df, events_by_intent, intent_ids, cfg, outdir)
    val_metrics = val_art["metrics"]
    return val_art, val_metrics, train_df, val_df, test_df


def cfg_key(cfg: Dict[str, Any]) -> str:
    items = sorted((k, to_py(v)) for k, v in cfg.items())
    return json.dumps(items, sort_keys=True)


def save_state(path: Path, state: State):
    write_json(path, {
        "best_cfg": {k: to_py(v) for k, v in state.best_cfg.items()},
        "best_score": state.best_score,
        "round_idx": state.round_idx,
        "param_idx": state.param_idx,
        "candidate_idx": state.candidate_idx,
        "last_completed_key": state.last_completed_key,
    })


def load_state(path: Path, default_cfg: Dict[str, Any]) -> State:
    if not path.exists():
        return State(best_cfg=dict(default_cfg), best_score=-1e9, round_idx=0, param_idx=0, candidate_idx=0, last_completed_key="")
    obj = read_json(path)
    return State(
        best_cfg=dict(obj.get("best_cfg", default_cfg)),
        best_score=float(obj.get("best_score", -1e9)),
        round_idx=int(obj.get("round_idx", 0)),
        param_idx=int(obj.get("param_idx", 0)),
        candidate_idx=int(obj.get("candidate_idx", 0)),
        last_completed_key=str(obj.get("last_completed_key", "")),
    )


def run_search(dataset_name: str, baseline: str, df: pd.DataFrame, events_by_intent: Dict[str, list], spec: Dict[str, Any], root_outdir: Path, rounds: int = 2):
    baseline_outdir = root_outdir / dataset_name / baseline
    baseline_outdir.mkdir(parents=True, exist_ok=True)

    log_path = baseline_outdir / "run.log"
    log_file = open(log_path, "a", buffering=1)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    try:
        intent_ids = sorted(events_by_intent.keys())
        base_cfg = dict(spec["base_cfg"])
        base_cfg["seed"] = int(spec.get("seed", 42))
        search_params = candidate_space(dataset_name, baseline, spec)
        state_path = baseline_outdir / "search_state.json"
        state = load_state(state_path, base_cfg)
        completed_path = baseline_outdir / "candidate_results.csv"
        if completed_path.exists():
            completed_df = pd.read_csv(completed_path)
        else:
            completed_df = pd.DataFrame()

        print(f"\n=== Dataset: {dataset_name} | Baseline: {baseline} ===")
        print(f"Data: {spec['data']}")
        print(f"Events: {spec['events']}")
        print(f"Output: {baseline_outdir}")
        print(f"Base config: {json.dumps(base_cfg, indent=2)}")
        print(f"Search params: {json.dumps(search_params, indent=2)}")

        all_params = list(search_params.keys())
        best_cfg = dict(state.best_cfg)
        best_score = float(state.best_score)

        # coordinate-ascent style search with restartable state
        for round_idx in range(state.round_idx, rounds):
            changed = False
            start_param_idx = state.param_idx if round_idx == state.round_idx else 0
            for pidx in range(start_param_idx, len(all_params)):
                param = all_params[pidx]
                values = search_params[param]
                best_for_param = None
                best_for_param_score = -1e9
                start_cand = state.candidate_idx if (round_idx == state.round_idx and pidx == state.param_idx) else 0
                print(f"\n[round {round_idx+1}] tuning {param} starting from candidate {start_cand}")

                for cidx in range(start_cand, len(values)):
                    cand = values[cidx]
                    cfg = dict(best_cfg)
                    cfg[param] = cand
                    key = cfg_key(cfg)
                    if not completed_df.empty and (completed_df["cfg_key"] == key).any():
                        row = completed_df[completed_df["cfg_key"] == key].iloc[0].to_dict()
                        score = float(row.get("score", -1e9))
                        accepted = bool(row.get("accepted", False))
                        val_metrics = {"fdr": row.get("val_fdr", np.nan), "lt": row.get("val_lt", np.nan), "fp": row.get("val_fp", np.nan), "da": row.get("val_da", np.nan)}
                        print(f"  skip existing {param}={cand} score={score:.4f}")
                    else:
                        try:
                            val_art, val_metrics, train_df, val_df, test_df = eval_cfg(dataset_name, baseline, cfg, df, events_by_intent, baseline_outdir)
                            score = compute_score({
                                "fdr": val_metrics["overall"]["fdr"],
                                "lt": val_metrics["overall"]["lt"],
                                "fp": val_metrics["overall"]["fp"],
                                "da": val_metrics["overall"]["da"],
                            }, spec["refs"])
                            accepted = is_acceptable({
                                "fdr": val_metrics["overall"]["fdr"],
                                "lt": val_metrics["overall"]["lt"],
                                "fp": val_metrics["overall"]["fp"],
                                "da": val_metrics["overall"]["da"],
                            }, spec["acceptance"])
                            print(
                                f"  {param}={cand} -> val FDR={val_metrics['overall']['fdr']:.2f}, "
                                f"LT={val_metrics['overall']['lt']:.2f}, FP={val_metrics['overall']['fp']:.2f}, "
                                f"DA={val_metrics['overall']['da']:.2f}, score={score:.4f}, accepted={accepted}"
                            )
                            row = {
                                "dataset": dataset_name,
                                "baseline": baseline,
                                "cfg_key": key,
                                "round": round_idx + 1,
                                "param": param,
                                "candidate": cand,
                                "score": score,
                                "accepted": accepted,
                                **{f"cfg_{k}": to_py(v) for k, v in cfg.items()},
                                "val_fdr": val_metrics["overall"]["fdr"],
                                "val_lt": val_metrics["overall"]["lt"],
                                "val_fp": val_metrics["overall"]["fp"],
                                "val_da": val_metrics["overall"]["da"],
                                "val_num_cause_events": val_metrics["overall"]["num_cause_events"],
                            }
                            completed_df = pd.concat([completed_df, pd.DataFrame([row])], ignore_index=True)
                            completed_df.to_csv(completed_path, index=False)
                            save_state(state_path, State(best_cfg=best_cfg, best_score=best_score, round_idx=round_idx, param_idx=pidx, candidate_idx=cidx + 1, last_completed_key=key))
                        except Exception as e:
                            print(f"  {param}={cand} FAILED: {e}")
                            traceback.print_exc()
                            row = {
                                "dataset": dataset_name,
                                "baseline": baseline,
                                "cfg_key": key,
                                "round": round_idx + 1,
                                "param": param,
                                "candidate": cand,
                                "score": -1e9,
                                "accepted": False,
                                "error": repr(e),
                                **{f"cfg_{k}": to_py(v) for k, v in cfg.items()},
                            }
                            completed_df = pd.concat([completed_df, pd.DataFrame([row])], ignore_index=True)
                            completed_df.to_csv(completed_path, index=False)
                            save_state(state_path, State(best_cfg=best_cfg, best_score=best_score, round_idx=round_idx, param_idx=pidx, candidate_idx=cidx + 1, last_completed_key=key))
                            continue

                    if score > best_for_param_score:
                        best_for_param_score = score
                        best_for_param = cand
                    if score > best_score:
                        best_score = score

                if best_for_param is not None and best_cfg.get(param) != best_for_param:
                    best_cfg[param] = best_for_param
                    changed = True
                state = State(best_cfg=dict(best_cfg), best_score=float(best_score), round_idx=round_idx, param_idx=pidx + 1, candidate_idx=0, last_completed_key=state.last_completed_key)
                save_state(state_path, state)

            if not changed:
                break
            state = State(best_cfg=dict(best_cfg), best_score=float(best_score), round_idx=round_idx + 1, param_idx=0, candidate_idx=0, last_completed_key=state.last_completed_key)
            save_state(state_path, state)

        # final evaluation on held-out test
        trainval_df, test_df = make_outer_split(df, spec["test_frac"])
        train_df, val_df = make_inner_split(trainval_df, int(best_cfg["folds"]))
        final_art = baseline_runner(baseline)(train_df, val_df, events_by_intent, intent_ids, best_cfg, baseline_outdir)
        val_metrics = final_art["metrics"]
        #test_metrics, conf, ts = evaluate_test_from_artifact(baseline, final_art["artifact"], best_cfg, train_df, val_df, test_df, events_by_intent, intent_ids)
        test_metrics, conf, ts = evaluate_test_from_artifact(
            baseline, final_art, best_cfg, train_df, val_df, test_df, events_by_intent, intent_ids
        )

        write_json(baseline_outdir / "selected_config.json", {k: to_py(v) for k, v in best_cfg.items()})
        write_json(baseline_outdir / "validation_result.json", val_metrics)
        write_json(baseline_outdir / "test_result.json", {
            "test_overall": test_metrics["overall"],
            "test_per_intent": test_metrics["per_intent"],
            "confusion": conf,
        })
        ts.to_csv(baseline_outdir / "test_timeseries.csv", index=False)

        summary = pd.DataFrame([
            {"split": "validation", **{k: to_py(v) for k, v in best_cfg.items()}, **val_metrics["overall"], "score": compute_score(val_metrics["overall"], spec["refs"]), "accepted": is_acceptable(val_metrics["overall"], spec["acceptance"])},
            {"split": "test", **{k: to_py(v) for k, v in best_cfg.items()}, **test_metrics["overall"], "score": compute_score(test_metrics["overall"], spec["refs"]), "accepted": is_acceptable(test_metrics["overall"], spec["acceptance"])},
        ])
        summary.to_csv(baseline_outdir / "selected_summary.csv", index=False)

        per_intent_rows = []
        for intent, v in test_metrics["per_intent"].items():
            per_intent_rows.append({
                "intent": intent,
                "detection_rate": v["detection_rate"] * 100.0,
                "avg_lead_time": v["avg_lead_time"],
            })
        pd.DataFrame(per_intent_rows).to_csv(baseline_outdir / "test_per_intent.csv", index=False)

        print("\n===== FINAL SELECTED CONFIG =====")
        print(json.dumps({k: to_py(v) for k, v in best_cfg.items()}, indent=2))
        print("\n===== VALIDATION METRICS =====")
        print(json.dumps(val_metrics["overall"], indent=2))
        print("\n===== TEST METRICS =====")
        print(json.dumps(test_metrics["overall"], indent=2))

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=["synthetic", "microservices", "ryu"], choices=list(DATASETS.keys()))
    parser.add_argument("--baselines", nargs="*", default=["weighted_kpi", "distance", "logistic", "mlp", "lstm"], choices=["weighted_kpi", "distance", "logistic", "mlp", "lstm"])
    parser.add_argument("--data-root", default=None, help="Optional override root for dataset files")
    parser.add_argument("--events-root", default=None, help="Optional override root for events files")
    parser.add_argument("--outdir", default="out_fair_baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    set_global_determinism(args.seed)
    root_outdir = Path(args.outdir)
    root_outdir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        spec = dict(DATASETS[dataset_name])
        spec["seed"] = args.seed
        if args.data_root:
            spec["data"] = str(Path(args.data_root) / Path(spec["data"]).name)
        if args.events_root:
            spec["events"] = str(Path(args.events_root) / Path(spec["events"]).name)
        df = pd.read_csv(spec["data"])
        events_by_intent = read_json(spec["events"])
        for baseline in args.baselines:
            run_search(dataset_name, baseline, df, events_by_intent, spec, root_outdir, rounds=args.rounds)


if __name__ == "__main__":
    main()
