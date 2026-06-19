
import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from determinism import set_global_determinism
from make_forward_looking_dataset import make_forward_looking_dataset
from custom_loss import head_decorrelation
from hybrid_mild_model import create_mild_moe_with_teacher

# Reuse helper functions from your existing runner.
from run_hybrid_experiment import (
    read_json,
    filter_events,
    _logit,
    distillation_focal_with_temperature,
    gate_kldiv_with_teacher,
    grid_search_params,
    evaluate,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@dataclass
class DatasetSpec:
    name: str
    data_path: str
    events_path: str
    outdir: str
    test_frac: float
    baseline: Dict[str, Any]
    candidates: Dict[str, List[Any]]
    acceptance: Dict[str, float]
    scoring_refs: Dict[str, float]


def prob_to_dist(P: np.ndarray, T: float = 2.0, eps: float = 1e-6) -> np.ndarray:
    """Convert per-intent probabilities into a normalized distribution over intents."""
    logits = np.log(np.clip(P, eps, 1.0 - eps)) - np.log(np.clip(1.0 - P, eps, 1.0 - eps))
    logits = logits / T
    m = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - m)
    z = np.sum(e, axis=1, keepdims=True) + eps
    return e / z


def make_inner_split(df: pd.DataFrame, folds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/val split inside a trainval pool.
    folds=3 => first 2/3 train, last 1/3 val.
    folds=11 => first 10/11 train, last 1/11 val.
    """
    if folds < 2:
        raise ValueError("folds must be >= 2")
    n = len(df)
    if n < folds * 20:
        # keep a minimum amount of data per block
        folds = max(2, min(folds, max(2, n // 20)))
    block = max(1, n // folds)
    split = max(1, n - block)
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df = df.iloc[split:].reset_index(drop=True)
    return train_df, val_df


def make_outer_split(df: pd.DataFrame, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("t").reset_index(drop=True)
    n_test = max(1, int(round(len(df) * test_frac)))
    split = max(1, len(df) - n_test)
    trainval = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return trainval, test


def compute_score(metrics: Dict[str, float], refs: Dict[str, float]) -> float:
    """
    A conservative composite score used only for hyperparameter selection.
    Values are scaled relative to the dataset's current good operating point.
    """
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


def build_binary_labels(df_slice, evs, horizon, intent_ids):
    Y = np.zeros((len(df_slice), len(intent_ids)), dtype=np.float32)
    t = df_slice["t"].values
    for j, intent in enumerate(intent_ids):
        for ev in evs.get(intent, []):
            start = max(ev["failure_time"] - horizon, int(t[0]))
            end = min(ev["failure_time"], int(t[-1]) + 1)
            if end > start:
                Y[(t >= start) & (t < end), j] = 1.0
    return Y


def train_eval_once(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    events_by_intent: Dict[str, list],
    intent_ids: List[str],
    cfg: Dict[str, Any],
    outdir: Path,
    run_tag: str,
    save_timeseries: bool = False,
):
    """
    Train on train_df, tune alert thresholds on val_df, and compute validation metrics.
    Returns metrics + chosen threshold params + (optionally) artifacts for final evaluation.
    """
    # Split events
    tr_events = filter_events(events_by_intent, train_df["t"].min(), train_df["t"].max())
    va_events = filter_events(events_by_intent, val_df["t"].min(), val_df["t"].max())

    # Build datasets
    Xtr, Ytr_ttf, Ytr_bin, Ytr_cause, Ttr, _ = make_forward_looking_dataset(
        train_df,
        tr_events,
        intent_ids,
        cfg["horizon"],
        base_features=[
            "cpu_pct", "ram_pct", "storage_pct", "snet", "sri",
            "cpu_delta", "sri_delta", "api_latency", "analytics_tput", "telemetry_queue",
        ],
    )
    Xva, Yva_ttf, Yva_bin, Yva_cause, Tva, _ = make_forward_looking_dataset(
        val_df,
        va_events,
        intent_ids,
        cfg["horizon"],
        base_features=[
            "cpu_pct", "ram_pct", "storage_pct", "snet", "sri",
            "cpu_delta", "sri_delta", "api_latency", "analytics_tput", "telemetry_queue",
        ],
    )

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xva = scaler.transform(Xva)

    # Teacher: one logistic regressor per intent, trained only on the train split.
    P_tr = np.zeros((len(Xtr), len(intent_ids)), dtype="float32")
    P_va = np.zeros((len(Xva), len(intent_ids)), dtype="float32")

    Ytr_bin_fold = build_binary_labels(train_df, tr_events, cfg["horizon"], intent_ids)
    for j, name in enumerate(intent_ids):
        if np.unique(Ytr_bin_fold[:, j]).size < 2:
            # no positive examples in this training split
            continue
        lr = LogisticRegression(max_iter=600, random_state=cfg["seed"], class_weight="balanced")
        lr.fit(Xtr, Ytr_bin_fold[:, j])
        P_tr[:, j] = lr.predict_proba(Xtr)[:, 1]
        P_va[:, j] = lr.predict_proba(Xva)[:, 1]

    D_tr = prob_to_dist(P_tr, T=cfg["temp"])
    D_va = prob_to_dist(P_va, T=cfg["temp"])

    # Targets
    ytr, yva = {}, {}
    for j, name in enumerate(intent_ids):
        ytr[f"{name}_out"] = np.concatenate([Ytr_ttf[:, j:j+1], P_tr[:, j:j+1]], axis=1)
        yva[f"{name}_out"] = np.concatenate([Yva_ttf[:, j:j+1], P_va[:, j:j+1]], axis=1)

    gate_tr = Ytr_cause.astype("float32").copy()
    gate_va = Yva_cause.astype("float32").copy()
    ytr["gate"] = np.concatenate([gate_tr, D_tr], axis=1)
    yva["gate"] = np.concatenate([gate_va, D_va], axis=1)

    ytr["head_concat"] = np.zeros((len(Xtr), len(intent_ids) * 16), dtype="float32")
    yva["head_concat"] = np.zeros((len(Xva), len(intent_ids) * 16), dtype="float32")

    # Model
    tf.keras.backend.clear_session()
    model = create_mild_moe_with_teacher(
        num_features=Xtr.shape[1],
        num_teacher=len(intent_ids),
        intent_ids=intent_ids,
        use_teacher_gate=cfg["use_teacher_gate"],
    )

    losses = {
        f"{name}_out": distillation_focal_with_temperature(
            alpha=cfg["distill_alpha"],
            horizon_min=cfg["horizon"],
            temperature=cfg["temp"],
        )
        for name in intent_ids
    }
    losses.update({
        "gate": gate_kldiv_with_teacher(
            supervise_weight=cfg["gate_supervise"],
            teacher_weight=cfg["gate_teacher"],
            sparsity_weight=cfg["gate_sparsity"],
        ),
        "head_concat": head_decorrelation(cfg["lambda_decorr"]),
    })

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=losses)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]

    if cfg["use_teacher_gate"]:
        train_inputs = [Xtr, D_tr]
        val_inputs = [Xva, D_va]
    else:
        train_inputs = Xtr
        val_inputs = Xva

    hist = model.fit(
        train_inputs,
        ytr,
        epochs=cfg["epochs"],
        batch_size=cfg["batch"],
        validation_data=(val_inputs, yva),
        callbacks=callbacks,
        verbose=0,
    )

    preds_va = model.predict(val_inputs, verbose=0)
    per_intent_va = preds_va[:len(intent_ids)]
    gate_probs_va = preds_va[-2]

    params = {}
    for j, name in enumerate(intent_ids):
        params[name] = grid_search_params(
            per_intent_va[j].reshape(-1),
            val_df["t"].values,
            va_events,
            name,
            fp_budget_per_day=cfg["fp_budget"],
        )

    # Apply on validation
    Tva_vals = val_df["t"].values
    ema_by_intent = {}
    per_intent_alerts = {}
    for j, name in enumerate(intent_ids):
        W = params[name]["W"]
        tau = params[name]["tau"]
        ema = pd.Series(per_intent_va[j].reshape(-1), index=Tva_vals).ewm(span=W, adjust=False).mean().values
        ema_by_intent[name] = ema
        per_intent_alerts[name] = Tva_vals[ema >= tau]

    tp, fn, lt, fp_total, correct_cause, total_cause, _, _, conf = evaluate(
        per_intent_alerts,
        va_events,
        Tva_vals,
        ema_by_intent,
        gate_probs_va,
    )

    per_intent_metrics = {
        name: {
            "detection_rate": float(tp[name] / max(tp[name] + fn[name], 1e-9)),
            "avg_lead_time": float(np.mean(lt[name])) if lt[name] else 0.0,
        }
        for name in intent_ids
    }
    val_metrics = {
        "fdr": float(np.mean([v["detection_rate"] for v in per_intent_metrics.values()]) * 100.0),
        "lt": float(np.mean([v["avg_lead_time"] for v in per_intent_metrics.values()])),
        "fp": float(fp_total / max((Tva_vals[-1] - Tva_vals[0]) / 1440.0 if len(Tva_vals) > 1 else 1.0, 1e-9)),
        "da": float((correct_cause / max(total_cause, 1)) * 100.0 if total_cause > 0 else 0.0),
    }

    # optional timeseries export
    if save_timeseries:
        cols = {"t": Tva_vals}
        for name in intent_ids:
            cols[f"ema_{name}"] = ema_by_intent[name]
        pd.DataFrame(cols).to_csv(outdir / f"{run_tag}_val_timeseries.csv", index=False)

    return {
        "model": model,
        "scaler": scaler,
        "params": params,
        "val_metrics": val_metrics,
        "per_intent_metrics": per_intent_metrics,
        "gate_probs_va": gate_probs_va,
        "ema_by_intent": ema_by_intent,
        "preds_va": preds_va,
        "history": hist.history,
    }


def evaluate_test(
    model,
    scaler: StandardScaler,
    params: Dict[str, Dict[str, float]],
    cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    events_by_intent: Dict[str, list],
    intent_ids: List[str],
):
    # Rebuild train+val? No: test evaluation must use the selected training/validation split.
    tr_events = filter_events(events_by_intent, train_df["t"].min(), train_df["t"].max())
    va_events = filter_events(events_by_intent, val_df["t"].min(), val_df["t"].max())
    te_events = filter_events(events_by_intent, test_df["t"].min(), test_df["t"].max())

    Xte, Yte_ttf, Yte_bin, Yte_cause, Tte, _ = make_forward_looking_dataset(
        test_df,
        te_events,
        intent_ids,
        cfg["horizon"],
        base_features=[
            "cpu_pct", "ram_pct", "storage_pct", "snet", "sri",
            "cpu_delta", "sri_delta", "api_latency", "analytics_tput", "telemetry_queue",
        ],
    )
    Xte = scaler.transform(Xte)

    # teacher inputs for test are only used if architecture includes them
    # We rebuild train+val teacher distributions only for the gate input during prediction
    # by reusing the model's second input when present.
    # The model itself was trained already in train_eval_once; here we only predict.
    if cfg["use_teacher_gate"]:
        # Recreate train/val teacher distribution on test for gate conditioning:
        # Since the gate input is only used during model prediction, we derive it from
        # a teacher trained on train+val to keep the inference pipeline consistent.
        trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        trainval_events = filter_events(events_by_intent, trainval_df["t"].min(), trainval_df["t"].max())
        Xtv, Ytv_ttf, Ytv_bin, Ytv_cause, Ttv, _ = make_forward_looking_dataset(
            trainval_df,
            trainval_events,
            intent_ids,
            cfg["horizon"],
            base_features=[
                "cpu_pct", "ram_pct", "storage_pct", "snet", "sri",
                "cpu_delta", "sri_delta", "api_latency", "analytics_tput", "telemetry_queue",
            ],
        )
        Xtv = scaler.transform(Xtv)

        # Train teacher models on train+val only, then predict on test.
        # This mirrors the signal available in the original implementation.
        Ytv_bin_fold = build_binary_labels(trainval_df, trainval_events, cfg["horizon"], intent_ids)
        P_tv = np.zeros((len(Xtv), len(intent_ids)), dtype="float32")
        P_te = np.zeros((len(Xte), len(intent_ids)), dtype="float32")
        for j, name in enumerate(intent_ids):
            if np.unique(Ytv_bin_fold[:, j]).size < 2:
                continue
            lr = LogisticRegression(max_iter=600, random_state=cfg["seed"], class_weight="balanced")
            lr.fit(Xtv, Ytv_bin_fold[:, j])
            P_te[:, j] = lr.predict_proba(Xte)[:, 1]
        D_te = prob_to_dist(P_te, T=cfg["temp"])
        test_inputs = [Xte, D_te]
    else:
        test_inputs = Xte

    preds_te = model.predict(test_inputs, verbose=0)
    per_intent_te = preds_te[:len(intent_ids)]
    gate_probs_te = preds_te[-2]

    Tte_vals = test_df["t"].values
    ema_by_intent = {}
    per_intent_alerts = {}
    for j, name in enumerate(intent_ids):
        W = params[name]["W"]
        tau = params[name]["tau"]
        ema = pd.Series(per_intent_te[j].reshape(-1), index=Tte_vals).ewm(span=W, adjust=False).mean().values
        ema_by_intent[name] = ema
        per_intent_alerts[name] = Tte_vals[ema >= tau]

    tp, fn, lt, fp_total, correct_cause, total_cause, _, _, conf = evaluate(
        per_intent_alerts,
        te_events,
        Tte_vals,
        ema_by_intent,
        gate_probs_te,
    )

    per_intent_metrics = {
        name: {
            "detection_rate": float(tp[name] / max(tp[name] + fn[name], 1e-9)),
            "avg_lead_time": float(np.mean(lt[name])) if lt[name] else 0.0,
        }
        for name in intent_ids
    }
    days = (Tte_vals[-1] - Tte_vals[0]) / 1440.0 if len(Tte_vals) > 1 else 1.0
    overall = {
        "fp_per_day": float(fp_total / max(days, 1e-9)),
        "disamb_accuracy": float((correct_cause / max(total_cause, 1)) * 100.0 if total_cause > 0 else 0.0),
        "num_cause_events": int(total_cause),
    }

    return {
        "per_intent": per_intent_metrics,
        "overall": overall,
        "confusion": {f"{a}->{b}": c for (a, b), c in conf.items()},
        "ema_by_intent": ema_by_intent,
        "gate_probs": gate_probs_te,
        "timeseries": pd.DataFrame({"t": Tte_vals, **{f"ema_{k}": v for k, v in ema_by_intent.items()}}),
    }


def candidate_space_for_dataset(spec: DatasetSpec):
    # Narrow, dataset-aware search spaces centered around the values you already found useful.
    # folds is used as the train/validation split ratio inside the pre-test pool.
    if spec.name == "synthetic":
        return {
            "horizon": [100, 120, 140],
            "folds": [5, 7, 11],
            "epochs": [20, 30, 40],
            "batch": [256, 512],
            "fp_budget": [0.5, 1.0, 1.5],
            "gate_supervise": [0.5, 0.7, 0.9],
            "gate_sparsity": [0.001, 0.005, 0.01],
            "gate_teacher": [0.0, 0.3, 0.7],
            "distill_alpha": [0.7, 0.9, 1.0],
            "temp": [1.5, 2.0],
            "use_teacher_gate": [True, False],
            "lambda_decorr": [0.0, 1e-4, 5e-4],
        }
    if spec.name == "microservices":
        return {
            "horizon": [40, 50, 60],
            "folds": [3, 5],
            "epochs": [15, 20, 25],
            "batch": [64, 128, 256],
            "fp_budget": [2.0, 3.0, 4.0],
            "gate_supervise": [0.5, 0.6, 0.7],
            "gate_sparsity": [0.002, 0.005],
            "gate_teacher": [0.0, 0.3, 0.7],
            "distill_alpha": [0.5, 0.6, 0.7, 1.0],
            "temp": [2.0],
            "use_teacher_gate": [True, False],
            "lambda_decorr": [0.0, 1e-4, 5e-4],
        }
    # ryu
    return {
        "horizon": [10, 15, 20],
        "folds": [3, 5],
        "epochs": [15, 20, 25],
        "batch": [128, 256],
        "fp_budget": [2.0, 3.0, 4.0],
        "gate_supervise": [0.5, 0.6, 0.7],
        "gate_sparsity": [0.001, 0.002, 0.005],
        "gate_teacher": [0.0, 0.3, 0.7],
        "distill_alpha": [0.6, 0.9, 1.0],
        "temp": [2.0],
        "use_teacher_gate": [True, False],
        "lambda_decorr": [0.0, 1e-4, 5e-4],
    }


def build_dataset_specs(args) -> DatasetSpec:
    # Defaults are your current selected settings.
    if args.dataset == "synthetic":
        baseline = {
            "seed": args.seed,
            "horizon": 120,
            "folds": 11,
            "epochs": 30,
            "batch": 512,
            "fp_budget": 1.0,
            "gate_supervise": 0.7,
            "gate_sparsity": 0.005,
            "gate_teacher": 0.7,
            "distill_alpha": 0.9,
            "temp": 2.0,
            "use_teacher_gate": True,
            "lambda_decorr": 1e-4,
        }
        acceptance = {"min_fdr": 97.0, "min_lt": 90.0, "max_fp": 6.5, "min_da": 87.0}
        refs = {"fdr": 98.89, "lt": 100.09, "fp": 4.97, "da": 89.67}
        test_frac = args.test_frac if args.test_frac is not None else 0.20
    elif args.dataset == "microservices":
        baseline = {
            "seed": args.seed,
            "horizon": 50,
            "folds": 3,
            "epochs": 20,
            "batch": 128,
            "fp_budget": 3.0,
            "gate_supervise": 0.6,
            "gate_sparsity": 0.005,
            "gate_teacher": 0.3,
            "distill_alpha": 0.6,
            "temp": 2.0,
            "use_teacher_gate": True,
            "lambda_decorr": 1e-4,
        }
        acceptance = {"min_fdr": 90.0, "min_lt": 28.0, "max_fp": 10.0, "min_da": 65.0}
        refs = {"fdr": 91.32, "lt": 31.76, "fp": 8.17, "da": 66.80}
        test_frac = args.test_frac if args.test_frac is not None else 0.20
    else:  # ryu
        baseline = {
            "seed": args.seed,
            "horizon": 15,
            "folds": 3,
            "epochs": 20,
            "batch": 256,
            "fp_budget": 3.0,
            "gate_supervise": 0.6,
            "gate_sparsity": 0.002,
            "gate_teacher": 0.3,
            "distill_alpha": 0.6,
            "temp": 2.0,
            "use_teacher_gate": True,
            "lambda_decorr": 1e-4,
        }
        acceptance = {"min_fdr": 90.0, "min_lt": 7.0, "max_fp": 10.0, "min_da": 87.0}
        refs = {"fdr": 92.35, "lt": 8.38, "fp": 8.59, "da": 88.72}
        test_frac = args.test_frac if args.test_frac is not None else 0.20

    return DatasetSpec(
        name=args.dataset,
        data_path=args.data,
        events_path=args.events,
        outdir=str(Path(args.outdir) / args.dataset),
        test_frac=test_frac,
        baseline=baseline,
        candidates=candidate_space_for_dataset(DatasetSpec(
            name=args.dataset,
            data_path=args.data,
            events_path=args.events,
            outdir=str(Path(args.outdir) / args.dataset),
            test_frac=test_frac,
            baseline=baseline,
            candidates={},
            acceptance=acceptance,
            scoring_refs=refs,
        )),
        acceptance=acceptance,
        scoring_refs=refs,
    )


def search_hyperparameters(spec: DatasetSpec, df: pd.DataFrame, events_by_intent: Dict[str, list], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    best_cfg = dict(spec.baseline)

    # Pre-sort by time and split into trainval/test.
    trainval_df, test_df = make_outer_split(df, spec.test_frac)

    # Coordinate ascent over narrow candidate sets.
    search_order = [
        "folds",
        "horizon",
        "batch",
        "epochs",
        "fp_budget",
        "gate_sparsity",
        "gate_supervise",
        "gate_teacher",
        "distill_alpha",
        "temp",
        "use_teacher_gate",
        "lambda_decorr",
    ]

    for round_idx in range(2):  # two passes is usually enough
        changed = False
        for param in search_order:
            cand_values = spec.candidates[param]
            candidate_rows = []

            for cand in cand_values:
                cfg = dict(best_cfg)
                cfg[param] = cand

                train_df, val_df = make_inner_split(trainval_df, int(cfg["folds"]))
                intent_ids = sorted(events_by_intent.keys())

                try:
                    result = train_eval_once(
                        train_df=train_df,
                        val_df=val_df,
                        events_by_intent=events_by_intent,
                        intent_ids=intent_ids,
                        cfg=cfg,
                        outdir=outdir,
                        run_tag=f"r{round_idx+1}_{param}_{str(cand).replace('.','p')}",
                        save_timeseries=False,
                    )
                    val_metrics = result["val_metrics"]
                    score = compute_score(val_metrics, spec.scoring_refs)
                    accepted = is_acceptable(val_metrics, spec.acceptance)
                    candidate_rows.append({
                        "round": round_idx + 1,
                        "param": param,
                        "candidate": cand,
                        "score": score,
                        "accepted": accepted,
                        **{f"cfg_{k}": v for k, v in cfg.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    })
                    print(
                        f"[{spec.name}][round {round_idx+1}] {param}={cand} "
                        f"-> val FDR={val_metrics['fdr']:.2f}, LT={val_metrics['lt']:.2f}, "
                        f"FP={val_metrics['fp']:.2f}, DA={val_metrics['da']:.2f}, "
                        f"score={score:.4f}, accepted={accepted}"
                    )
                except Exception as e:
                    candidate_rows.append({
                        "round": round_idx + 1,
                        "param": param,
                        "candidate": cand,
                        "score": -1e9,
                        "accepted": False,
                        "error": repr(e),
                        **{f"cfg_{k}": v for k, v in cfg.items()},
                    })
                    print(f"[{spec.name}][round {round_idx+1}] {param}={cand} FAILED: {e}")

            # pick best acceptable; fallback to best score
            df_candidates = pd.DataFrame(candidate_rows)
            # append to global log
            log_rows.extend(candidate_rows)
            df_candidates.to_csv(outdir / f"stage_{round_idx+1}_{param}_candidates.csv", index=False)

            if df_candidates.empty:
                continue
            acceptable = df_candidates[df_candidates["accepted"] == True].copy()
            if not acceptable.empty:
                best_row = acceptable.sort_values("score", ascending=False).iloc[0]
            else:
                best_row = df_candidates.sort_values("score", ascending=False).iloc[0]

            chosen_val = best_row["candidate"]
            if best_cfg[param] != chosen_val:
                best_cfg[param] = chosen_val
                changed = True

        if not changed:
            break

    # Final run with selected config
    train_df, val_df = make_inner_split(trainval_df, int(best_cfg["folds"]))
    intent_ids = sorted(events_by_intent.keys())
    final_result = train_eval_once(
        train_df=train_df,
        val_df=val_df,
        events_by_intent=events_by_intent,
        intent_ids=intent_ids,
        cfg=best_cfg,
        outdir=outdir,
        run_tag="selected_final",
        save_timeseries=True,
    )

    test_result = evaluate_test(
        model=final_result["model"],
        scaler=final_result["scaler"],
        params=final_result["params"],
        cfg=best_cfg,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        events_by_intent=events_by_intent,
        intent_ids=intent_ids,
    )

    # Save outputs
    search_df = pd.DataFrame(log_rows)
    search_df.to_csv(outdir / "search_candidates.csv", index=False)

    with open(outdir / "selected_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)

    with open(outdir / "validation_result.json", "w") as f:
        json.dump({
            "val_metrics": final_result["val_metrics"],
            "per_intent_metrics": final_result["per_intent_metrics"],
            "threshold_params": final_result["params"],
        }, f, indent=2)

    with open(outdir / "test_result.json", "w") as f:
        json.dump({
            "test_overall": test_result["overall"],
            "test_per_intent": test_result["per_intent"],
            "confusion": test_result["confusion"],
        }, f, indent=2)

    # Final summary table
    summary_rows = []
    summary_rows.append({
        "split": "validation",
        **best_cfg,
        **final_result["val_metrics"],
        "score": compute_score(final_result["val_metrics"], spec.scoring_refs),
        "accepted": is_acceptable(final_result["val_metrics"], spec.acceptance),
    })
    summary_rows.append({
        "split": "test",
        **best_cfg,
        "fdr": np.mean([v["detection_rate"] for v in test_result["per_intent"].values()]) * 100.0,
        "lt": np.mean([v["avg_lead_time"] for v in test_result["per_intent"].values()]),
        "fp": test_result["overall"]["fp_per_day"],
        "da": test_result["overall"]["disamb_accuracy"],
        "score": compute_score({
            "fdr": np.mean([v["detection_rate"] for v in test_result["per_intent"].values()]) * 100.0,
            "lt": np.mean([v["avg_lead_time"] for v in test_result["per_intent"].values()]),
            "fp": test_result["overall"]["fp_per_day"],
            "da": test_result["overall"]["disamb_accuracy"],
        }, spec.scoring_refs),
        "accepted": is_acceptable({
            "fdr": np.mean([v["detection_rate"] for v in test_result["per_intent"].values()]) * 100.0,
            "lt": np.mean([v["avg_lead_time"] for v in test_result["per_intent"].values()]),
            "fp": test_result["overall"]["fp_per_day"],
            "da": test_result["overall"]["disamb_accuracy"],
        }, spec.acceptance),
    })
    pd.DataFrame(summary_rows).to_csv(outdir / "selected_summary.csv", index=False)

    # Per-intent test table
    per_intent_rows = []
    for intent, v in test_result["per_intent"].items():
        per_intent_rows.append({
            "intent": intent,
            "detection_rate": v["detection_rate"] * 100.0,
            "avg_lead_time": v["avg_lead_time"],
        })
    pd.DataFrame(per_intent_rows).to_csv(outdir / "test_per_intent.csv", index=False)

    # A compact human-readable text log of the selected configuration.
    with open(outdir / "selected_config_readable.txt", "w") as f:
        f.write("Selected configuration\\n")
        f.write(json.dumps(best_cfg, indent=2))
        f.write("\\n\\nValidation metrics\\n")
        f.write(json.dumps(final_result["val_metrics"], indent=2))
        f.write("\\n\\nTest metrics\\n")
        f.write(json.dumps({
            "overall": test_result["overall"],
            "per_intent": test_result["per_intent"],
        }, indent=2))

    return best_cfg, final_result, test_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["synthetic", "microservices", "ryu"])
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--events", required=True, help="Path to events_by_intent.json")
    parser.add_argument("--outdir", required=True, help="Base output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_frac", type=float, default=None, help="Outer chronological test fraction")
    args = parser.parse_args()

    out_base = Path(args.outdir)
    out_base.mkdir(parents=True, exist_ok=True)
    outdir = out_base / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    # Tee stdout to a log file.
    log_path = outdir / "run.log"
    log_file = open(log_path, "w", buffering=1)
    import sys as _sys

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

    _sys.stdout = Tee(_sys.stdout, log_file)
    _sys.stderr = Tee(_sys.stderr, log_file)

    set_global_determinism(args.seed)

    spec = build_dataset_specs(args)
    df = pd.read_csv(args.data)
    events_by_intent = read_json(args.events)

    print(f"Dataset: {spec.name}")
    print(f"Data: {args.data}")
    print(f"Events: {args.events}")
    print(f"Output: {outdir}")
    print(f"Baseline config: {json.dumps(spec.baseline, indent=2)}")
    print(f"Acceptance thresholds: {json.dumps(spec.acceptance, indent=2)}")
    print(f"Scoring refs: {json.dumps(spec.scoring_refs, indent=2)}")

    best_cfg, val_result, test_result = search_hyperparameters(spec, df, events_by_intent, outdir)

    print("\\n===== FINAL SELECTED CONFIG =====")
    print(json.dumps(best_cfg, indent=2))
    print("\\n===== VALIDATION METRICS =====")
    print(json.dumps(val_result["val_metrics"], indent=2))
    print("\\n===== TEST METRICS =====")
    print(json.dumps(test_result["overall"], indent=2))

    log_file.close()


if __name__ == "__main__":
    main()
