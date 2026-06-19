#!/usr/bin/env python
"""
baseline_mlp.py
---------------
Evaluates a "Standard Neural Network" (MLP) baseline.

This model is the most direct and fair comparison to MILD:
  - It is a non-linear neural network.
  - It uses a shared "body" (like MILD's Encoder) to find complex patterns.
  - It has separate "heads" (one per intent) for multi-task prediction.

This script uses the same cross-validation, evaluation, and reporting
logic as your other baseline scripts.

Usage:
  python baseline_mlp.py --data data/final/dataset.csv --outdir out/baseline_mlp_final \
      --epochs 30 --batch 256 --fp_budget 0.25
"""
import os, json, argparse, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from determinism import set_global_determinism

# TensorFlow / Keras imports for the new NN baseline
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# ------------------ Helper Functions ------------------
# (Copied directly from your baseline scripts for consistency)

def read_json(p):
    with open(p,'r') as f: return json.load(f)

def filter_events(events_by_intent, t_min, t_max):
    subset = {k: [] for k in events_by_intent}
    for k, evs in events_by_intent.items():
        for e in evs:
            if (e['start'] <= t_max) and (e['end'] >= t_min):
                e2 = dict(e)
                e2['start'] = max(e['start'], int(t_min))
                e2['end'] = min(e['end'], int(t_max))
                e2['failure_time'] = min(max(e['failure_time'], e2['start']+1), e2['end'])
                subset[k].append(e2)
    return subset

def grid_search_params(scores, tvals, evs_by_intent, intent, fp_budget_per_day=1.0):
    Ws=[3,5,8,13,21,34]
    best = {'W':5, 'tau':0.5, 'lead':-1.0}
    evs = evs_by_intent.get(intent, [])
    if not evs: return best
    all_windows = [(e['start'], e['end']) for v in evs_by_intent.values() for e in v]

    for W in Ws:
        ema = pd.Series(scores, index=tvals).ewm(span=W, adjust=False).mean().values
        smin, smax = float(np.min(ema)), float(np.max(ema))
        if smax == smin: continue
        for tau in np.linspace(smin, smax, 60):
            alerts = tvals[ema >= tau]
            fp = sum(1 for a in alerts if not any((s <= a <= e) for (s,e) in all_windows))
            days = (tvals[-1]-tvals[0]) / 1440.0 if len(tvals)>1 else 1.0
            fp_rate = fp / max(days, 1e-9)
            if fp_rate > fp_budget_per_day: continue
            tp, lt = 0, []
            for e in evs:
                in_win = alerts[(alerts >= e['start']) & (alerts <= e['failure_time'])]
                if in_win.size>0:
                    tp += 1; lt.append(e['failure_time'] - in_win[0])
            avg_lead = float(np.mean(lt)) if lt else 0.0
            if avg_lead > best['lead']:
                best = {'W': W, 'tau': float(tau), 'lead': avg_lead}
    return best

# [IN: baseline_NEW_MLP.py]
# REPLACE the existing 'evaluate' function with this:

def evaluate(per_intent_alerts, te_events, Tte, ema_by_intent):
    # Note: No 'gate_probs' here, as the baseline doesn't have one
    import numpy as np
    tp = {k:0 for k in te_events}; fn = {k:0 for k in te_events}; lt = {k:[] for k in te_events}
    all_windows = [(e_['start'], e_['end']) for v in te_events.values() for e_ in v] # Renamed 'e'
    fp_total = 0
    for name, alerts in per_intent_alerts.items():
        # ... [TP/FN/FP logic remains unchanged] ...
        for eobj in te_events.get(name, []):
            in_win = alerts[(alerts >= eobj['start']) & (alerts <= eobj['failure_time'])]
            if in_win.size>0:
                tp[name]+=1; lt[name].append(int(eobj['failure_time'] - in_win[0]))
            else:
                fn[name]+=1
        for a in alerts:
            if not any((s <= a <= e) for (s,e) in all_windows): fp_total += 1
    
    # --- NEW UNIFIED DISAMBIGUATION LOGIC ---
    time_to_idx = {int(t): idx for idx,t in enumerate(Tte)}
    intent_ids = list(ema_by_intent.keys()) # Get sorted intent names
    
    total_events_with_cause = 0
    total_correctly_identified = 0
    conf = {} # Confusion matrix

    for true_name, evs in te_events.items():
        for eobj in evs:
            # 1. Find the True Root Cause for this event
            event_type = eobj.get('type', '')
            is_root_cause_event = event_type.endswith('_cause') or event_type.startswith('independent')
            
            if not is_root_cause_event:
                continue

            true_root_cause = true_name
            total_events_with_cause += 1

            # 2. Find the Earliest Alert Time for this event
            earliest = None
            S_pred_set = set() # Store all intents that alerted at the earliest time
            for name, arr in per_intent_alerts.items():
                arr2 = arr[(arr >= eobj['start']) & (arr <= eobj['failure_time'])]
                if arr2.size>0:
                    t0 = arr2[0]
                    if (earliest is None) or (t0 < earliest): 
                        earliest = t0
                        S_pred_set = {name} # Reset the set
                    elif t0 == earliest:
                        S_pred_set.add(name) # Add to the set if tied
            
            if earliest is None:
                continue # No alert fired

            # 3. Find MLP's Predicted Root Cause (using highest EMA score heuristic)
            idx_at_alert = time_to_idx.get(int(earliest), None)
            if idx_at_alert is None:
                continue
            
            if len(S_pred_set) == 1:
                predicted_root_cause = list(S_pred_set)[0]
            else:
                # Tie-breaking: find intent with max risk score at alert time
                best_score = -np.inf
                predicted_root_cause = list(S_pred_set)[0] # Default
                for name in S_pred_set:
                    score = ema_by_intent[name][idx_at_alert]
                    if score > best_score:
                        best_score = score
                        predicted_root_cause = name
            
            # 4. Compare True vs. Predicted
            if predicted_root_cause == true_root_cause:
                total_correctly_identified += 1
            
            conf_key = (true_root_cause, predicted_root_cause)
            conf[conf_key] = conf.get(conf_key, 0) + 1

    # We are replacing the old (ex, ns, jh, nm) with (total_correctly_identified, total_events_with_cause)
    # To keep the function signature, we'll return 0 for the old Jaccard metrics
    return tp, fn, lt, fp_total, total_correctly_identified, total_events_with_cause, 0, 0, conf

# Label generation function (from your baseline script)
def build_labels(df_slice, evs_by_intent, intents, horizon=120):
    K=len(intents); Y=np.zeros((len(df_slice),K),dtype=np.float32); t=df_slice['t'].values
    for j,intent in enumerate(intents):
        for e in evs_by_intent.get(intent, []):
            start = max(e['failure_time']-horizon, int(t[0])); end=min(e['failure_time'], int(t[-1])+1)
            if end<=start: continue
            m=(t>=start)&(t<end); Y[m,j]=1.0
    return Y

# ------------------ New MLP Baseline Model ------------------

def create_mlp_baseline(num_features, intent_ids, enc_units=(96, 96), head_units=16, l2=1e-5, dropout=0.2):
    """
    Creates a standard MLP baseline with a shared body and per-intent heads.
    This is the fairest NN comparison to MILD's architecture.
    """
    x_in = layers.Input(shape=(num_features,), name='x')
    
    # Shared Body (like MILD's encoder)
    # This learns complex, non-linear patterns from all features
    x = x_in
    for u in enc_units:
        x = layers.Dense(u, activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout(dropout)(x)
    shared_body = x
    
    outputs = []
    for intent in intent_ids:
        # Per-intent head (like MILD's heads, but without expert input)
        # Each head learns to predict risk for its intent from the shared patterns
        h = layers.Dense(head_units, activation='relu', name=f'head_{intent}')(shared_body)
        out = layers.Dense(1, activation='sigmoid', name=f'{intent}_out')(h)
        outputs.append(out)
        
    model = models.Model(inputs=x_in, outputs=outputs, name='MLP_Baseline')
    return model

# ------------------ Main Execution ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--fp_budget', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--horizon', type=int, default=120)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=512)
    args = ap.parse_args()

    set_global_determinism(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    events_by_intent = read_json(os.path.join(os.path.dirname(args.data), 'events_by_intent.json'))
    intents = sorted(events_by_intent.keys())
    # Use the same features as MILD
    feats = ['cpu_pct','ram_pct','storage_pct','snet','sri','cpu_delta','sri_delta','api_latency','analytics_tput','telemetry_queue']

    # Blocked cross-validation logic
    N = len(df)
    fold_sizes = [N//args.folds]*(args.folds-1) + [N - (args.folds-1)*(N//args.folds)]
    starts = [sum(fold_sizes[:i]) for i in range(args.folds)]
    tests = [(s, s+fold_sizes[i]) for i,s in enumerate(starts)]

    # Aggregators for the new MLP baseline
    aggs = {
        'mlp': {'dr': defaultdict(list), 'lt': defaultdict(list), 'fp': [], 'd_exact': [], 'd_jacc': []}
    }

    # Main cross-validation loop
    for i, (s, e) in enumerate(tests):
        print(f"\n--- Fold {i+1}/{args.folds}: test [{s}:{e}) ---")
        df_test = df.iloc[s:e].reset_index(drop=True)
        df_train_full = df.iloc[:s].reset_index(drop=True) # All data before test

        if len(df_train_full) < 1000:
            print("  [skip] not enough train history before this test block.")
            continue
            
        # Create a train/validation split for Keras EarlyStopping
        # This is more robust than tuning on the training set
        split = int(0.8 * len(df_train_full))
        df_train = df_train_full.iloc[:split].reset_index(drop=True)
        df_val = df_train_full.iloc[split:].reset_index(drop=True)

        # Filter events for each split
        tr_events = filter_events(events_by_intent, df_train['t'].min(), df_train['t'].max())
        va_events = filter_events(events_by_intent, df_val['t'].min(),   df_val['t'].max())
        te_events = filter_events(events_by_intent, df_test['t'].min(),  df_test['t'].max())
        
        Ttr = df_train['t'].values
        Tva = df_val['t'].values
        Tte = df_test['t'].values
        days = (Tte[-1]-Tte[0]) / 1440.0 if len(Tte)>1 else 1.0

        # Scaler (fit on train, transform all)
        scaler = StandardScaler().fit(df_train[feats].values)
        Xtr = scaler.transform(df_train[feats].values)
        Xva = scaler.transform(df_val[feats].values)
        Xte = scaler.transform(df_test[feats].values)

        # Build labels
        Ytr = build_labels(df_train, tr_events, intents, args.horizon)
        Yva = build_labels(df_val, va_events, intents, args.horizon)
        
        # Format labels for Keras (a dictionary of outputs)
        ytr_keras = {f'{name}_out': Ytr[:, j] for j, name in enumerate(intents)}
        yva_keras = {f'{name}_out': Yva[:, j] for j, name in enumerate(intents)}
        
        # ---------- 1. Create and Train MLP Model ----------
        print("  Training MLP Baseline Model...")
        tf.keras.backend.clear_session()
        model = create_mlp_baseline(Xtr.shape[1], intents)
        
        # A simple loss is fine, as this is a standard baseline
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss='binary_crossentropy')
        
        cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)]
        
        model.fit(Xtr, ytr_keras,
                  epochs=args.epochs,
                  batch_size=args.batch,
                  validation_data=(Xva, yva_keras),
                  callbacks=cb,
                  verbose=0)

        # ---------- 2. Tune Thresholds on Validation Set ----------
        print("  Tuning thresholds on validation set...")
        # model.predict() returns a list of arrays, one for each output
        preds_va = model.predict(Xva, verbose=0)
        
        # Convert list to dict for grid_search_params
        scores_val = {name: preds_va[j].reshape(-1) for j, name in enumerate(intents)}
        
        params = {k: grid_search_params(scores_val[k], Tva, va_events, k, args.fp_budget) for k in intents}

        # ---------- 3. Evaluate on Test Set ----------
        print("  Evaluating on test set...")
        preds_te = model.predict(Xte, verbose=0)
        scores_test = {name: preds_te[j].reshape(-1) for j, name in enumerate(intents)}

        per_intent_alerts, ema_by_intent = {}, {}
        for k in intents:
            W, tau = params[k]['W'], params[k]['tau']
            ema = pd.Series(scores_test[k], index=Tte).ewm(span=W, adjust=False).mean().values
            ema_by_intent[k] = ema
            per_intent_alerts[k] = Tte[ema >= tau]

        #tp, fn, lt, fp, ex, ns, jh, nm, _ = evaluate(per_intent_alerts, te_events, Tte, ema_by_intent)
        tp, fn, lt, fp, correct_cause, total_cause, _, _, _ = evaluate(
            per_intent_alerts, te_events, Tte, ema_by_intent
            )
        
        # Aggregate results
        key = 'mlp'
        for k in intents:
            aggs[key]['dr'][k].append(tp[k]/max(tp[k]+fn[k],1e-9))
            aggs[key]['lt'][k].append(float(np.mean(lt[k])) if lt[k] else 0.0)
        aggs[key]['fp'].append(fp/max(days,1e-9))
        #aggs[key]['d_exact'].append((ex/max(ns,1)) if ns>0 else 0.0)
        #aggs[key]['d_jacc'].append((jh/max(nm,1)) if nm>0 else 0.0)
        aggs[key]['d_acc'] = aggs[key].get('d_acc', []) # Create new list
        aggs[key]['d_acc'].append((correct_cause/max(total_cause,1)) if total_cause > 0 else 0.0)

    # ---------- Final Reporting ----------
    if not aggs['mlp']['fp']:
        print("\nNo folds were evaluated. Increase dataset size or decrease number of folds.")
        return

    baselines_map = {
        'MLP Baseline': 'mlp',
    }

    for baseline_name, key in baselines_map.items():
        print(f"\n--- Baseline: {baseline_name} ({args.folds}-Fold CV Results) ---")
        print(f"{'Intent':<15} | {'Detection Rate':<20} | {'Avg Lead Time (min)':<22} | {'FP Rate/Day':<18} | {'Disambiguation Acc.':<20}")
        print('-'*115)
        for name in intents:
            dr_m, dr_s = np.mean(aggs[key]['dr'][name]), np.std(aggs[key]['dr'][name])
            lt_m, lt_s = np.mean(aggs[key]['lt'][name]), np.std(aggs[key]['lt'][name])
            print(f"{name:<15} | {dr_m:.2%} ± {dr_s:.2%}      | {lt_m:.2f} ± {lt_s:.2f}        | {' ':<18} | {' ':<20}")
        print('-'*115)

        fp_m, fp_s = np.mean(aggs[key]['fp']), np.std(aggs[key]['fp'])
        da_m, da_s = np.mean(aggs[key]['d_acc']), np.std(aggs[key]['d_acc']) # <-- NEW METRIC
        print(f"{'Overall':<15} | {' ':<20} | {' ':<22} | {fp_m:.2f} ± {fp_s:.2f}       | {da_m:.2%} ± {da_s:.2%} (Root Cause Acc.)")

        # fp_m, fp_s = np.mean(aggs[key]['fp']), np.std(aggs[key]['fp'])
        # de_m, de_s = np.mean(aggs[key]['d_exact']), np.std(aggs[key]['d_exact'])
        # dj_m, dj_s = np.mean(aggs[key]['d_jacc']), np.std(aggs[key]['d_jacc'])
        # print(f"{'Overall':<15} | {' ':<20} | {' ':<22} | {fp_m:.2f} ± {fp_s:.2f}       | {de_m:.2%} (Exact singletons), {dj_m:.2%} (Jacc≥0.5 co-drifts)")
        
    # Save raw metrics to the outdir
    with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
        json.dump({'folds': aggs['mlp']}, f, indent=2)

if __name__=='__main__':
    main()