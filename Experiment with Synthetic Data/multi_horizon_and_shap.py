"""
multi_horizon_and_shap.py  (updated)
------------------------------------
Trains the selected model (HYBRID-MILD distilled student by default) for multiple horizons,
generates dynamic time-to-failure (TTF) visuals, and computes per-alert SHAP explanations
with REAL feature names (engineered by make_forward_looking_dataset).

Outputs (under --outdir):
  - metrics_{H}.json
  - fold{f}_timeseries_{H}.csv
  - multihorizon_event_{fold}_{intent}.png
  - shap_{fold}_{intent}_{H}.png
  - shap_{fold}_{intent}_{H}.csv
  - ttf_table_{fold}_{intent}.csv
  - feature_names_fold{fold}_{H}.csv    <-- NEW: saved feature-name mapping per horizon/fold

Example:
  python multi_horizon_and_shap.py \
    --data data/hard/dataset.csv \
    --outdir out/hard/multihorizon_shap \
    --horizons 120 60 30 \
    --fold 1 --folds 5 \
    --intent telemetry \
    --epochs 30 --batch 512 \
    --fp_budget 0.25 \
    --gate_supervise 0.2 --gate_sparsity 0.005 \
    --gate_teacher 0.5 --distill_alpha 0.5 --temp 2.0
"""
import os, json, argparse, numpy as np, pandas as pd, tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from determinism import set_global_determinism
from make_forward_looking_dataset import make_forward_looking_dataset
from hybrid_mild_model import create_mild_moe_with_teacher
from custom_loss import head_decorrelation
# reuse helpers & losses from hybrid runner
from run_hybrid_experiment import (
    distillation_focal_with_temperature,
    gate_kldiv_with_teacher,
    grid_search_params,
    filter_events,
    evaluate,
)

# ---------- SHAP (optional) ----------
try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False

BASE_FEATURES = [
    'cpu_pct','ram_pct','storage_pct','snet','sri',
    'cpu_delta','sri_delta','api_latency','analytics_tput','telemetry_queue'
]

def _to_scalar(x):
    """Best-effort conversion of SHAP/TF expected_value into a Python float."""
    import numpy as _np
    try:
        if hasattr(x, "numpy"):
            x = x.numpy()
        x = _np.asarray(x).reshape(-1)[0]
        return float(x)
    except Exception:
        return float(x)

def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def _prob_to_dist(P, T=2.0, eps=1e-6):
    logits = np.log(np.clip(P, eps, 1-eps)) - np.log(np.clip(1-P, eps, 1-eps))
    logits = logits / T
    m = np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits - m); Z = np.sum(e, axis=1, keepdims=True) + eps
    return e / Z

def _save_feature_names(names, out_csv):
    pd.DataFrame({'index': list(range(len(names))), 'feature': names}).to_csv(out_csv, index=False)

def _train_one_horizon(df, events_by_intent, intent_ids, H, args, fold_idx=1, folds=5):
    """Return:
       metrics, ts_path, model, scaler, (df_tr, df_va, df_te),
       (tr_ev, va_ev, te_ev), (Xtr, Xva, Xte), (Ttr, Tva, Tte),
       params, ema_by_intent, feats
    """
    # blocked CV folds
    N = len(df)
    fold_sizes = [N//folds]*(folds-1) + [N - (folds-1)*(N//folds)]
    starts = [sum(fold_sizes[:i]) for i in range(folds)]
    tests = [(s, s+fold_sizes[i]) for i,s in enumerate(starts)]

    s,e = tests[fold_idx-1]
    df_test = df.iloc[s:e].reset_index(drop=True)
    df_train_full = df.iloc[:s].reset_index(drop=True)
    assert len(df_train_full) >= 1000, "Not enough train history for this fold; try a later fold or reduce --folds."

    split = int(0.8 * len(df_train_full))
    df_train = df_train_full.iloc[:split].reset_index(drop=True)
    df_val   = df_train_full.iloc[split:].reset_index(drop=True)

    tr_events = filter_events(events_by_intent, df_train['t'].min(), df_train['t'].max())
    va_events = filter_events(events_by_intent, df_val['t'].min(),   df_val['t'].max())
    te_events = filter_events(events_by_intent, df_test['t'].min(),  df_test['t'].max())

    # get features + REAL feature names
    Xtr, Ytr_ttf, Ytr_bin, Ytr_cause, Ttr, feats     = make_forward_looking_dataset(df_train, tr_events, intent_ids, H, BASE_FEATURES)
    Xva, Yva_ttf, Yva_bin, Yva_cause, Tva, feats_val = make_forward_looking_dataset(df_val,   va_events, intent_ids, H, BASE_FEATURES)
    Xte, Yte_ttf, Yte_bin, Yte_cause, Tte, feats_te  = make_forward_looking_dataset(df_test,  te_events, intent_ids, H, BASE_FEATURES)
    assert feats == feats_val == feats_te, "Feature order mismatch across splits."

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr); Xva = scaler.transform(Xva); Xte = scaler.transform(Xte)

    # Teacher: Logistic OvR (per-intent)
    P_tr = np.zeros((len(Xtr), len(intent_ids)), dtype='float32')
    P_va = np.zeros((len(Xva), len(intent_ids)), dtype='float32')
    P_te = np.zeros((len(Xte), len(intent_ids)), dtype='float32')

    def build_binary_labels(df_slice, evs, horizon):
        Y=np.zeros((len(df_slice), len(intent_ids)), dtype=np.float32); t=df_slice['t'].values
        for j, intent in enumerate(intent_ids):
            for ev in evs.get(intent, []):
                start = max(ev['failure_time']-horizon, int(t[0])); end=min(ev['failure_time'], int(t[-1])+1)
                if end > start: Y[(t>=start)&(t<end), j] = 1.0
        return Y
    Ytr_bin_fold = build_binary_labels(df_train, tr_events, H)

    for j, name in enumerate(intent_ids):
        if np.unique(Ytr_bin_fold[:, j]).size < 2:
            # leave zero probs for this head
            continue
        lr = LogisticRegression(max_iter=200, random_state=args.seed, class_weight='balanced')
        lr.fit(Xtr, Ytr_bin_fold[:, j])
        P_tr[:, j] = lr.predict_proba(Xtr)[:,1]
        P_va[:, j] = lr.predict_proba(Xva)[:,1]
        P_te[:, j] = lr.predict_proba(Xte)[:,1]

    # Teacher distributions for gate (temperature on logits -> softmax)
    D_tr = _prob_to_dist(P_tr, T=args.temp)
    D_va = _prob_to_dist(P_va, T=args.temp)
    D_te = _prob_to_dist(P_te, T=args.temp)

    # ----- Targets -----
    ytr, yva = {}, {}
    for j, name in enumerate(intent_ids):
        ytr[f'{name}_out'] = np.concatenate([Ytr_ttf[:, j:j+1], P_tr[:, j:j+1]], axis=1)
        yva[f'{name}_out'] = np.concatenate([Yva_ttf[:, j:j+1], P_va[:, j:j+1]], axis=1)

    # gate targets: cause-aware + teacher distribution
    gate_tr = Ytr_cause.astype('float32').copy()
    gate_va = Yva_cause.astype('float32').copy()
    mask_tr = (gate_tr.sum(axis=1) == 0); mask_va = (gate_va.sum(axis=1) == 0)
    gate_tr[mask_tr, :] = Ytr_bin[mask_tr, :]
    gate_va[mask_va, :] = Yva_bin[mask_va, :]
    ytr['gate'] = np.concatenate([gate_tr, D_tr], axis=1)
    yva['gate'] = np.concatenate([gate_va, D_va], axis=1)

    # decorrelation target (dummy)
    ytr['head_concat'] = np.zeros((len(Xtr), len(intent_ids)*16), dtype='float32')
    yva['head_concat'] = np.zeros((len(Xva), len(intent_ids)*16), dtype='float32')

    # teacher-as-feature toggle
    Ttr_in = D_tr if args.use_teacher_inputs else np.zeros_like(D_tr)
    Tva_in = D_va if args.use_teacher_inputs else np.zeros_like(D_va)
    Tte_in = D_te if args.use_teacher_inputs else np.zeros_like(D_te)

    # ----- Model -----
    tf.keras.backend.clear_session()
    model = create_mild_moe_with_teacher(num_features=Xtr.shape[1], num_teacher=len(intent_ids), intent_ids=intent_ids)

    losses = {f'{name}_out': distillation_focal_with_temperature(alpha=args.distill_alpha, horizon_min=H, temperature=args.temp)
              for name in intent_ids}
    losses.update({
        'gate': gate_kldiv_with_teacher(supervise_weight=args.gate_supervise, teacher_weight=args.gate_teacher, sparsity_weight=args.gate_sparsity),
        'head_concat': head_decorrelation(1e-4)
    })

    cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)]
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=losses)
    model.fit([Xtr, Ttr_in], ytr, epochs=args.epochs, batch_size=args.batch,
              validation_data=([Xva, Tva_in], yva), callbacks=cb, verbose=0)

    # ----- Validation threshold tuning & Test eval -----
    preds_va = model.predict([Xva, Tva_in], verbose=0)
    preds_te = model.predict([Xte, Tte_in], verbose=0)

    per_intent_va = preds_va[:len(intent_ids)]
    per_intent_te = preds_te[:len(intent_ids)]
    gate_probs_te  = preds_te[len(intent_ids)]

    params = {
        name: grid_search_params(per_intent_va[j].reshape(-1), df_val['t'].values, va_events, name, fp_budget_per_day=args.fp_budget)
        for j,name in enumerate(intent_ids)
    }

    # Apply on test
    Tte_vals = df_test['t'].values
    ema_by_intent = {}; per_intent_alerts = {}
    for j, name in enumerate(intent_ids):
        W = params[name]['W']; tau = params[name]['tau']
        ema = pd.Series(per_intent_te[j].reshape(-1), index=Tte_vals).ewm(span=W, adjust=False).mean().values
        ema_by_intent[name] = ema
        per_intent_alerts[name] = Tte_vals[ema >= tau]

    #tp, fn, lt, fp_total, ex, ns, jh, nm, conf = evaluate(per_intent_alerts, te_events, Tte_vals, ema_by_intent)
    tp, fn, lt, fp_total, ex, ns, jh, nm, conf = evaluate(per_intent_alerts, te_events, Tte_vals, ema_by_intent, gate_probs_te)
    days = (Tte_vals[-1]-Tte_vals[0])/1440.0 if len(Tte_vals)>1 else 1.0
    fold_metrics = {
        'per_intent': {name: {'detection_rate': (tp[name]/max(tp[name]+fn[name],1e-9)),
                              'avg_lead_time': (float(np.mean(lt[name])) if lt[name] else 0.0)} for name in intent_ids},
        'overall': {'fp_per_day': fp_total/max(days,1e-9),
                    'disamb_exact_singleton': (ex/max(ns,1)) if ns>0 else 0.0,
                    'disamb_jaccard>=0.5_co': (jh/max(nm,1)) if nm>0 else 0.0,
                    'num_singleton': int(ns), 'num_co_drift': int(nm)},
        'params': params
    }

    # export time-series for this horizon
    ts_path = os.path.join(args.outdir, f'fold{fold_idx}_timeseries_{H}.csv')
    cols = {'t': Tte_vals}
    for name in intent_ids:
        cols[f'ema_{name}'] = ema_by_intent[name]
        cols[f'tau_{name}'] = np.full_like(Tte_vals, params[name]['tau'], dtype=float)
    for j,name in enumerate(intent_ids):
        cols[f'gate_{name}'] = gate_probs_te[:, j].reshape(-1)
    pd.DataFrame(cols).to_csv(ts_path, index=False)

    # save feature-name mapping for this horizon/fold
    _save_feature_names(feats, os.path.join(args.outdir, f'feature_names_fold{fold_idx}_{H}.csv'))

    return (fold_metrics, ts_path, model, scaler,
            (df_train, df_val, df_test), (tr_events, va_events, te_events),
            (Xtr, Xva, Xte), (Ttr, Tva, Tte_vals), params, ema_by_intent, feats)

def _pick_event_for_intent(te_events, intent, prefer='singleton'):
    evs = te_events.get(intent, [])
    if not evs: return None
    if prefer == 'singleton':
        # keep those not overlapping >50% with others
        S = []
        for e in evs:
            a1,b1 = e['start'], e['failure_time']
            overlap = False
            for k,L in te_events.items():
                if k == intent: continue
                for e2 in L:
                    a2,b2 = e2['start'], e2['failure_time']
                    inter = max(0, min(b1,b2)-max(a1,a2))
                    dur = min(b1-a1, b2-a2)
                    if dur>0 and inter >= 0.5*dur:
                        overlap=True; break
                if overlap: break
            if not overlap: S.append(e)
        if S: return S[0]
    return evs[0]

def _plot_multihorizon_event(df_test, intent_ids, HLIST, ts_paths, te_events, intent, out_path, title):
    e = _pick_event_for_intent(te_events, intent, 'singleton')
    if e is None:
        print(f"[WARN] No test event for intent {intent}.")
        return
    s0, f0, e0 = e['start'], e['failure_time'], e['end']

    series = {}; threshold = {}; times = None

    for H, path in ts_paths.items():
        df = pd.read_csv(path)
        times = df['t'].values
        series[H] = df[f'ema_{intent}'].values
        threshold[H] = df[f'tau_{intent}'].values[0]

    pad = 120
    m = (times >= (s0-pad)) & (times <= (e0+pad))
    tt = times[m]

    plt.figure(figsize=(10, 4.5))
    for H in sorted(HLIST, reverse=True):
        line, = plt.plot(tt, series[H][m], label=f'Risk (H={H})')  # Get the line handle
        color = line.get_color()  # Extract the automatically assigned color
        plt.axhline(threshold[H], linestyle='--', linewidth=1.2, color=color)

    # plt.figure(figsize=(10,4.5))
    # for H in sorted(HLIST, reverse=True):
    #     plt.plot(tt, series[H][m], label=f'Risk (H={H})')
    #     plt.axhline(threshold[H], linestyle='--', linewidth=1.0)
    plt.axvspan(s0, e0, color='orange', alpha=0.2, label='Drift window')
    plt.axvline(f0, color='red', linewidth=2, label='Failure')
    #plt.title(title)

    plt.xlabel('Time (min)', fontsize=18)      # Axis label font size
    plt.ylabel('Smoothed risk score', fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=15)  # Major tick labels

    plt.legend(ncol=1, fontsize=15, loc = 'upper right')
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def _first_cross(tt, rr, tau, s0, f0):
    mask = (tt >= s0) & (tt <= f0)
    idx = np.where(mask & (rr >= tau))[0]
    return (tt[idx[0]] if len(idx)>0 else None)

def _write_ttf_table(df_test, HLIST, ts_paths, te_events, intent, out_csv):
    e = _pick_event_for_intent(te_events, intent, 'singleton')
    if e is None: return
    s0, f0, e0 = e['start'], e['failure_time'], e['end']

    rows = []
    for H, path in ts_paths.items():
        df = pd.read_csv(path)
        t = df['t'].values; r = df[f'ema_{intent}'].values; tau = df[f'tau_{intent}'].values[0]
        talert = _first_cross(t, r, tau, s0, f0)
        est = (f0 - talert) if talert is not None else np.nan
        rows.append({'H': H, 't_alert': talert, 'threshold': tau,
                     'risk_at_alert': (np.interp(talert, t, r) if talert is not None else np.nan),
                     'est_TTF_min': est})
    pd.DataFrame(rows).sort_values('H', ascending=False).to_csv(out_csv, index=False)

# --- add this helper near the other small helpers (e.g., below BASE_FEATURES) ---
def _to_scalar(x):
    """Best-effort conversion of SHAP/TF expected_value into a Python float."""
    import numpy as _np
    try:
        if hasattr(x, "numpy"):
            x = x.numpy()
        x = _np.asarray(x).reshape(-1)[0]
        return float(x)
    except Exception:
        return float(x)

# --- replace the entire _shap_for_alert(...) function with this version ---
def _shap_for_alert(model, scaler, Xte, Tte, df_test, ts_path, intent, feature_names,
                    out_png, out_csv, bg_size=200, dpi=200):
    """
    Create SHAP explanation for the first alert of `intent` and export:
      - CSV of sorted feature contributions (|shap| desc)
      - Interactive force plot HTML (same style as explain_alert.py)
      - Bar plot PNG using shap.plots.bar with matching typography

    out_png: path for the bar plot (.png). A matching force-plot HTML will be
             saved next to it by replacing '.png' with '_force.html'.
    """
    if not _HAS_SHAP:
        print("[WARN] SHAP not installed. Skipping explainability figure.")
        return

    import numpy as np
    import pandas as pd
    import shap
    import matplotlib.pyplot as plt
    # ------------- locate first alert time -------------
    df_ts = pd.read_csv(ts_path)
    tvals = df_ts['t'].values
    r = df_ts[f'ema_{intent}'].values
    tau = float(df_ts[f'tau_{intent}'].values[0])

    idx = np.where(r >= tau)[0]
    if len(idx) == 0:
        print(f"[WARN] No alert for intent {intent} in this horizon; skipping SHAP.")
        return
    t_alert = tvals[idx[0]]
    i_te = np.where(Tte == t_alert)[0]
    if len(i_te) == 0:
        print("[WARN] Could not locate alert index in test T; skipping SHAP.")
        return
    i0 = int(i_te[0])

    # ------------- isolate single-head model -------------
    head = model.get_layer(f'{intent}_out').output
    model_intent = tf.keras.Model(inputs=model.inputs, outputs=head)

    # ------------- background & instance (multi-input safe) -------------
    n_bg = min(bg_size, Xte.shape[0])
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(np.arange(Xte.shape[0]), n_bg, replace=False)
    bg = Xte[bg_idx]                   # already scaled
    x0 = Xte[i0:i0+1]                  # already scaled

    # teacher input width (second input); use zeros for explanations
    t_dim = model.inputs[1].shape[-1]
    if t_dim is None:
        # fallback: infer from exported gate_ columns if needed
        t_dim = int(df_ts.filter(like="gate_").shape[1])
    t_bg = np.zeros((n_bg, int(t_dim)), dtype=bg.dtype)
    t0   = np.zeros((1,   int(t_dim)), dtype=bg.dtype)

    # For display, de-scale the feature vector so axis “features” show original values
    try:
        data_display = scaler.inverse_transform(x0).reshape(-1)
    except Exception:
        data_display = x0.reshape(-1)

    # ------------- explain: prefer DeepExplainer -> fallback to Explainer -> Kernel -------------
    explanation = None
    base_value = None
    values_1d = None

    # try DeepExplainer with multi-input list
    try:
        dexp = shap.DeepExplainer(model_intent, [bg, t_bg])
        sv = dexp.shap_values([x0, t0])
        # DeepExplainer on single-output returns a list per input; take the first (features)
        if isinstance(sv, list):
            sv_feat = sv[0]
            if isinstance(sv_feat, list):  # in some versions it nests again per-output
                sv_feat = sv_feat[0]
        else:
            sv_feat = sv
        values_1d = np.asarray(sv_feat).reshape(1, -1)[0]
        base_value = _to_scalar(dexp.expected_value)
    except Exception as e_deep:
        # generic model-aware Explainer
        try:
            exp = shap.Explainer(model_intent, [bg, t_bg], feature_names=feature_names)
            e = exp([x0, t0])
            values_1d = np.array(e.values).reshape(-1)
            base_value = _to_scalar(e.base_values)
        except Exception as e_expl:
            # model-agnostic fallback
            def f(X):
                return model_intent([X, np.zeros((X.shape[0], int(t_dim)), dtype=X.dtype)]).numpy()
            kexp = shap.KernelExplainer(f, bg)
            sv = kexp.shap_values(x0, nsamples=100)
            values_1d = sv[0].flatten() if isinstance(sv, list) else np.array(sv).flatten()
            base_value = 0.0  # KernelExplainer doesn’t always expose expected_value; OK for plotting.

    # Defensive naming
    feat_names = feature_names if (feature_names is not None and len(feature_names) == len(values_1d)) \
                 else [f'f{i}' for i in range(len(values_1d))]

    # Build a clean single-sample Explanation (values/data 1D)
    explanation = shap.Explanation(
        values=values_1d,
        base_values=float(base_value) if base_value is not None else 0.0,
        data=data_display.astype(np.float32),
        feature_names=feat_names,
    )

    # ------------- export CSV (|shap| sorted) -------------
    dfv = pd.DataFrame({'feature': explanation.feature_names, 'shap': explanation.values})
    dfv.sort_values('shap', key=np.abs, ascending=False, inplace=True)
    dfv.to_csv(out_csv, index=False)

    # ------------- force plot (interactive HTML) -------------
    html_path = out_png.replace('.png', '_force.html')
    force_fig = shap.force_plot(
        base_value=float(explanation.base_values),
        shap_values=explanation.values,     # 1D
        features=explanation.data,          # 1D
        feature_names=explanation.feature_names,
    )
    shap.save_html(html_path, force_fig)

    # ------------- bar plot (same typography as your reference) -------------
    shap.plots.bar(explanation, show=False)
    ax = plt.gca()
    ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_fontsize(12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved SHAP: {out_csv}, {out_png}, {html_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--horizons', type=int, nargs='+', default=[120,60,30])
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--intent', type=str, default='telemetry', help='Which intent to explain in the multihorizon figure')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fp_budget', type=float, default=0.25)
    ap.add_argument('--gate_supervise', type=float, default=0.2)
    ap.add_argument('--gate_sparsity', type=float, default=0.005)
    ap.add_argument('--gate_teacher', type=float, default=0.5)
    ap.add_argument('--distill_alpha', type=float, default=0.5)
    ap.add_argument('--temp', type=float, default=2.0)
    ap.add_argument('--use_teacher_inputs', action='store_true')
    args = ap.parse_args()

    set_global_determinism(args.seed)
    _ensure_dir(args.outdir)
    df = pd.read_csv(args.data)
    with open(os.path.join(os.path.dirname(args.data), 'events_by_intent.json'),'r') as f:
        events_by_intent = json.load(f)
    intent_ids = sorted(events_by_intent.keys())

    all_metrics = {}
    ts_paths = {}

    # capture feature names per H and save
    for H in args.horizons:
        print(f"\n=== Training HYBRID-MILD for H={H} ===")
        (metrics, ts_path, model, scaler,
         (df_tr, df_va, df_te), (tr_ev, va_ev, te_ev),
         (Xtr, Xva, Xte), (Ttr, Tva, Tte),
         params, ema, feats) = _train_one_horizon(df, events_by_intent, intent_ids, H, args, fold_idx=args.fold, folds=args.folds)

        all_metrics[H] = metrics
        with open(os.path.join(args.outdir, f'metrics_{H}.json'),'w') as f:
            json.dump(metrics, f, indent=2)
        ts_paths[H] = ts_path

        # SHAP for this horizon at first alert of selected intent (with feature names)
        _shap_for_alert(model, scaler, Xte, Tte, df_te, ts_path, args.intent, feats,
                        os.path.join(args.outdir, f'shap_{args.fold}_{args.intent}_{H}.png'),
                        os.path.join(args.outdir, f'shap_{args.fold}_{args.intent}_{H}.csv'))

    # multi-horizon zoom event plot + TTF table
    _plot_multihorizon_event(df_te, intent_ids, args.horizons, ts_paths, te_ev, args.intent,
                             os.path.join(args.outdir, f'multihorizon_event_{args.fold}_{args.intent}.png'),
                             f'Multi-horizon risk (Fold {args.fold}, intent={args.intent})')
    _write_ttf_table(df_te, args.horizons, ts_paths, te_ev, args.intent,
                     os.path.join(args.outdir, f'ttf_table_{args.fold}_{args.intent}.csv'))

    print("[OK] Wrote multihorizon risks, SHAP (with real feature names), and TTF tables in:", args.outdir)

if __name__=='__main__':
    main()
