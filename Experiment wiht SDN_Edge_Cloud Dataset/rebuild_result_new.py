from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd

from mild_hpo_train_val_test import (
    make_outer_split,
    make_inner_split,
    train_eval_once,
    evaluate_test,
    read_json,
)

SEARCH_ORDER = [
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

BASELINES = {
    "synthetic": {
        "seed": 42,
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
    },
    "microservices": {
        "seed": 42,
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
    },
    "ryu": {
        "seed": 42,
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
    },
}

def _to_py(v):
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    return v

def _parse_value(v):
    if pd.isna(v):
        return v
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return _to_py(v)
    if isinstance(v, str):
        s = v.strip()
        if s in {"True", "False"}:
            return s == "True"
        try:
            return ast.literal_eval(s)
        except Exception:
            return s
    return v

def recover_cfg_from_stage_csvs(outdir: Path, baseline: dict) -> dict:
    """
    Rebuild the final config by replaying the already-saved coordinate search
    from stage_*_candidates.csv. No retraining, no resweeping.
    """
    cfg = dict(baseline)

    for round_idx in (1, 2):
        for param in SEARCH_ORDER:
            fp = outdir / f"stage_{round_idx}_{param}_candidates.csv"
            if not fp.exists():
                continue

            df = pd.read_csv(fp)

            if "accepted" in df.columns:
                acc = df[df["accepted"].astype(str).str.lower().isin(["true", "1"])]
                use = acc if not acc.empty else df
            else:
                use = df

            if "score" in use.columns:
                use = use.sort_values("score", ascending=False)

            row = use.iloc[0]
            cfg[param] = _parse_value(row["candidate"])

    return {k: _parse_value(v) for k, v in cfg.items()}

def finish_one_dataset(dataset_name, data_path, events_path, out_root):
    outdir = Path(out_root) / dataset_name
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = recover_cfg_from_stage_csvs(outdir, BASELINES[dataset_name])
    print(f"\n=== {dataset_name} recovered config ===")
    print(json.dumps(cfg, indent=2, default=_to_py))

    df = pd.read_csv(data_path)
    events_by_intent = read_json(events_path)
    intent_ids = sorted(events_by_intent.keys())

    # Same outer split used by the HPO script
    trainval_df, test_df = make_outer_split(df, 0.20)
    train_df, val_df = make_inner_split(trainval_df, int(cfg["folds"]))

    final_result = train_eval_once(
        train_df=train_df,
        val_df=val_df,
        events_by_intent=events_by_intent,
        intent_ids=intent_ids,
        cfg=cfg,
        outdir=outdir,
        run_tag="final",
        save_timeseries=True,
    )

    test_result = evaluate_test(
        model=final_result["model"],
        scaler=final_result["scaler"],
        params=final_result["params"],
        cfg=cfg,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        events_by_intent=events_by_intent,
        intent_ids=intent_ids,
    )

    # Save clean final artifacts
    with open(outdir / "final_selected_config.json", "w") as f:
        json.dump({k: _to_py(v) for k, v in cfg.items()}, f, indent=2)

    with open(outdir / "final_validation_metrics.json", "w") as f:
        json.dump(final_result["val_metrics"], f, indent=2)

    with open(outdir / "final_test_metrics.json", "w") as f:
        json.dump(test_result["overall"], f, indent=2)

    pd.DataFrame([{
        "split": "validation",
        **final_result["val_metrics"],
    }, {
        "split": "test",
        **test_result["overall"],
    }]).to_csv(outdir / "final_summary.csv", index=False)

    pd.DataFrame([
        {
            "intent": intent,
            "detection_rate": v["detection_rate"] * 100.0,
            "avg_lead_time": v["avg_lead_time"],
        }
        for intent, v in test_result["per_intent"].items()
    ]).to_csv(outdir / "test_per_intent.csv", index=False)

    test_result["timeseries"].to_csv(outdir / "final_test_timeseries.csv", index=False)

    print("Saved final outputs to:", outdir)

# Run only the final pass, reusing already explored candidates
finish_one_dataset(
    "synthetic",
    "data/hard/dataset.csv",
    "data/hard/events_by_intent.json",
    "out_hpo_synthetic",
)

finish_one_dataset(
    "microservices",
    "data_micro/data/hard/dataset.csv",
    "data_micro/data/hard/events_by_intent.json",
    "out_hpo_micro",
)

finish_one_dataset(
    "ryu",
    "data_ryu/data/hard/dataset.csv",
    "data_ryu/data/hard/events_by_intent.json",
    "out_hpo_ryu",
)