#!/usr/bin/env python
"""
baseline_LSTM.py
----------------
Evaluates a Sequence-based LSTM baseline.

Unlike the MLP baseline which looks at a single timestamp t,
this model looks at a window [t - lookback, t].

Architecture:
  - Input: Sequence of shape (lookback, num_features)
  - Shared Body: LSTM layer(s) to extract temporal features.
  - Heads: Per-intent Dense layers for multi-task prediction.

Usage:
  python baseline_LSTM.py --data data/hard/dataset.csv --outdir out_final/hard/baseline_LSTM \
      --fp_budget 1 --folds 11 --lookback 60
"""
import os, json, argparse, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from determinism import set_global_determinism

# ------------------ Helper Functions ------------------

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

def create_sequences(X, Y, lookback):
    """
    Converts 2D arrays into 3D sequences.
    X: (samples, features) -> (samples - lookback, lookback, features)
    Y: (samples, intents)  -> (samples - lookback, intents)
    """
    Xs, Ys = [], []
    # We start at 'lookback' so the first window is [0 : lookback]
    # In a real efficient impl, use np.lib.stride_tricks.sliding_window_view
    # But for ~200k rows, a loop or simple list comp is acceptable/clear.
    # Actually, let's use a vectorized approach for speed.
    
    n_samples, n_feats = X.shape
    # Resulting shape: (n_samples - lookback, lookback, n_feats)
    # This might be heavy on RAM for very large lookbacks/datasets.
    # If OOM occurs, use tf.keras.utils.timeseries_dataset_from_array
    
    # Fast numpy stride trick
    # shape: (num_windows, lookback, n_feats)
    # strides: (feat_stride, feat_stride, elem_size) -> tricky for 2D.
    # Let's stick to a simple loop with pre-allocation for safety.
    
    num_seq = n_samples - lookback
    if num_seq <= 0:
        return np.array([]), np.array([])
        
    X_seq = np.zeros((num_seq, lookback, n_feats), dtype=np.float32)
    
    # Fill X
    # There is a faster way using stride_tricks
    from numpy.lib.stride_tricks import sliding_window_view
    # sliding_window_view(X, lookback, axis=0) -> (N-L+1, features, lookback)
    # We need (N-L, lookback, features)
    # Let's trust the standard Keras generator approach which is memory efficient
    return None, None # handled inside main via generator if needed, 
                      # but let's just do the simple slice loop for X since 200k is small.

    # Actually, 200,000 * 60 * 10 * 4 bytes ~= 480MB. It fits in RAM easily.
    X_seq = []
    for i in range(lookback, n_samples):
        X_seq.append(X[i-lookback:i])
    X_seq = np.array(X_seq, dtype=np.float32)
    
    # Y is just the label at the END of the window
    Y_seq = Y[lookback:]
    
    return X_seq, Y_seq

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

def evaluate(per_intent_alerts, te_events, Tte, ema_by_intent):
    import numpy as np
    tp = {k:0 for k in te_events}; fn = {k:0 for k in te_events}; lt = {k:[] for k in te_events}
    all_windows = [(e_['start'], e_['end']) for v in te_events.values() for e_ in v]
    fp_total = 0
    for name, alerts in per_intent_alerts.items():
        for eobj in te_events.get(name, []):
            in_win = alerts[(alerts >= eobj['start']) & (alerts <= eobj['failure_time'])]
            if in_win.size>0:
                tp[name]+=1; lt[name].append(int(eobj['failure_time'] - in_win[0]))
            else:
                fn[name]+=1
        for a in alerts:
            if not any((s <= a <= e) for (s,e) in all_windows): fp_total += 1
    
    # Unified Disambiguation Logic
    time_to_idx = {int(t): idx for idx,t in enumerate(Tte)}
    # Note: Tte might be shorter than original due to lookback, but time_to_idx handles mapping correctly
    intent_ids = list(ema_by_intent.keys())
    
    total_events_with_cause = 0
    total_correctly_identified = 0
    conf = {}

    for true_name, evs in te_events.items():
        for eobj in evs:
            event_type = eobj.get('type', '')
            is_root_cause_event = event_type.endswith('_cause') or event_type.startswith('independent')
            
            if not is_root_cause_event: continue

            true_root_cause = true_name
            total_events_with_cause += 1

            earliest = None
            S_pred_set = set()
            for name, arr in per_intent_alerts.items():
                arr2 = arr[(arr >= eobj['start']) & (arr <= eobj['failure_time'])]
                if arr2.size>0:
                    t0 = arr2[0]
                    if (earliest is None) or (t0 < earliest): 
                        earliest = t0
                        S_pred_set = {name}
                    elif t0 == earliest:
                        S_pred_set.add(name)
            
            if earliest is None: continue

            idx_at_alert = time_to_idx.get(int(earliest), None)
            if idx_at_alert is None: continue
            
            if len(S_pred_set) == 1:
                predicted_root_cause = list(S_pred_set)[0]
            else:
                best_score = -np.inf
                predicted_root_cause = list(S_pred_set)[0]
                for name in S_pred_set:
                    # Look up score in the aligned EMA arrays
                    # Careful: EMA arrays are length (N-lookback). 
                    # idx_at_alert is 0..(N-lookback).
                    score = ema_by_intent[name][idx_at_alert]
                    if score > best_score:
                        best_score = score
                        predicted_root_cause = name
            
            if predicted_root_cause == true_root_cause:
                total_correctly_identified += 1
            
            conf_key = (true_root_cause, predicted_root_cause)
            conf[conf_key] = conf.get(conf_key, 0) + 1

    return tp, fn, lt, fp_total, total_correctly_identified, total_events_with_cause, 0, 0, conf

def build_labels(df_slice, evs_by_intent, intents, horizon=120):
    K=len(intents); Y=np.zeros((len(df_slice),K),dtype=np.float32); t=df_slice['t'].values
    for j,intent in enumerate(intents):
        for e in evs_by_intent.get(intent, []):
            start = max(e['failure_time']-horizon, int(t[0])); end=min(e['failure_time'], int(t[-1])+1)
            if end<=start: continue
            m=(t>=start)&(t<end); Y[m,j]=1.0
    return Y

# ------------------ LSTM Model ------------------

def create_lstm_baseline(input_shape, intent_ids, lstm_units=64, dropout=0.2):
    """
    input_shape: (lookback, num_features)
    """
    x_in = layers.Input(shape=input_shape, name='x_seq')
    
    # Shared Body: LSTM
    # We can stack LSTMs or just use one. Let's use one strong LSTM layer followed by a Dense bottleneck.
    x = layers.LSTM(lstm_units, return_sequences=False)(x_in)
    x = layers.Dropout(dropout)(x)
    
    # A shared dense layer to mix temporal features before heads
    shared_body = layers.Dense(64, activation='relu')(x)
    
    outputs = []
    for intent in intent_ids:
        # Per-intent head
        h = layers.Dense(16, activation='relu', name=f'head_{intent}')(shared_body)
        out = layers.Dense(1, activation='sigmoid', name=f'{intent}_out')(h)
        outputs.append(out)
        
    model = models.Model(inputs=x_in, outputs=outputs, name='LSTM_Baseline')
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
    ap.add_argument('--lookback', type=int, default=60, help='Window size for LSTM')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=512)
    args = ap.parse_args()

    set_global_determinism(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    events_by_intent = read_json(os.path.join(os.path.dirname(args.data), 'events_by_intent.json'))
    intents = sorted(events_by_intent.keys())
    feats = ['cpu_pct','ram_pct','storage_pct','snet','sri','cpu_delta','sri_delta','api_latency','analytics_tput','telemetry_queue']

    # Blocked cross-validation logic
    N = len(df)
    fold_sizes = [N//args.folds]*(args.folds-1) + [N - (args.folds-1)*(N//args.folds)]
    starts = [sum(fold_sizes[:i]) for i in range(args.folds)]
    tests = [(s, s+fold_sizes[i]) for i,s in enumerate(starts)]

    aggs = {'lstm': {'dr': defaultdict(list), 'lt': defaultdict(list), 'fp': [], 'd_acc': []}}

    for i, (s, e) in enumerate(tests):
        print(f"\n--- Fold {i+1}/{args.folds}: test [{s}:{e}) ---")
        df_test = df.iloc[s:e].reset_index(drop=True)
        df_train_full = df.iloc[:s].reset_index(drop=True)

        # LSTM requires history. If train set is smaller than lookback, skip.
        if len(df_train_full) < args.lookback + 200:
            print("  [skip] not enough train history.")
            continue
            
        split = int(0.8 * len(df_train_full))
        df_train = df_train_full.iloc[:split].reset_index(drop=True)
        df_val = df_train_full.iloc[split:].reset_index(drop=True)

        # Scale based on training data
        scaler = StandardScaler().fit(df_train[feats].values)

        # Prepare Data helper
        def prepare_fold_data(df_part, evs_part):
            # 1. Scale
            X_raw = scaler.transform(df_part[feats].values)
            # 2. Labels (aligned with raw df)
            Y_raw = build_labels(df_part, evs_part, intents, args.horizon)
            
            # 3. Create Sequences
            X_seq = []
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                # sliding_window_view(X, lookback, axis=0) returns shape (N-W+1, Features, Lookback)
                X_seq = sliding_window_view(X_raw, window_shape=args.lookback, axis=0)
                
                # *** CRITICAL FIX: Swap axes to get (N, Lookback, Features) ***
                X_seq = np.swapaxes(X_seq, 1, 2)
                
                # Truncate to align with valid history
                # We usually want the prediction at time t, using history [t-L, t)
                X_seq = X_seq[:-1] if len(X_seq) > (len(X_raw)-args.lookback) else X_seq
                
            except ImportError:
                # Fallback for older numpy: list comprehension creates (N, Lookback, Features) correctly
                X_seq = np.array([X_raw[i-args.lookback:i] for i in range(args.lookback, len(X_raw))])
                
            # Align Y and T
            # If window is [0..59], it predicts for index 59 (or 60).
            # We align so the label is the one at the END of the window.
            Y_seq = Y_raw[args.lookback-1 : ]
            T_seq = df_part['t'].values[args.lookback-1 : ]
            
            # Ensure lengths match exactly
            min_len = min(len(X_seq), len(Y_seq), len(T_seq))
            return X_seq[:min_len], Y_seq[:min_len], T_seq[:min_len]
        
        # Prepare Data helper
        def prepare_fold_data_old(df_part, evs_part):
            # 1. Scale
            X_raw = scaler.transform(df_part[feats].values)
            # 2. Labels (aligned with raw df)
            Y_raw = build_labels(df_part, evs_part, intents, args.horizon)
            # 3. Create Sequences
            # Note: We must be careful. create_sequences returns len = N - lookback.
            # We must truncate T (timestamps) similarly to align evaluation.
            X_seq = []
            # Optimized simple loop for "small" data (<500k)
            # sliding_window_view is available in numpy 1.20+
            try:
                from numpy.lib.stride_tricks import sliding_window_view
                # shape (N, W, F)
                X_seq = sliding_window_view(X_raw, window_shape=args.lookback, axis=0)
                # The output of sliding_window_view is (N - W + 1, W, F)
                # We need to drop the last one if we want exactly N-lookback? 
                # Actually usually we want prediction at t, using [t-L, t).
                # If window [0..L-1] predicts label at L-1.
                X_seq = X_seq[:-1] if len(X_seq) > (len(X_raw)-args.lookback) else X_seq
                
            except ImportError:
                # Fallback for older numpy
                X_seq = np.array([X_raw[i-args.lookback:i] for i in range(args.lookback, len(X_raw))])
                
            # Align Y and T
            # If X_seq[0] uses indices 0..59, it predicts for index 59 (or 60? usually 59 is the last observed step).
            # The label should be Y[59].
            # So we slice Y and T starting from `lookback-1`.
            # Let's assume prediction is for the *last* timestamp in the window.
            Y_seq = Y_raw[args.lookback-1 : ]
            T_seq = df_part['t'].values[args.lookback-1 : ]
            
            # Ensure lengths match exactly (sliding_window_view edge cases)
            min_len = min(len(X_seq), len(Y_seq), len(T_seq))
            return X_seq[:min_len], Y_seq[:min_len], T_seq[:min_len]

        tr_events = filter_events(events_by_intent, df_train['t'].min(), df_train['t'].max())
        va_events = filter_events(events_by_intent, df_val['t'].min(),   df_val['t'].max())
        te_events = filter_events(events_by_intent, df_test['t'].min(),  df_test['t'].max())
        
        # Prepare datasets
        Xtr, Ytr, Ttr = prepare_fold_data(df_train, tr_events)
        Xva, Yva, Tva = prepare_fold_data(df_val, va_events)
        Xte, Yte, Tte = prepare_fold_data(df_test, te_events)

        if len(Xtr) < 100: 
            print("  [skip] Sequence generation resulted in empty train set."); continue

        # Format for Keras
        ytr_keras = {f'{name}_out': Ytr[:, j] for j, name in enumerate(intents)}
        yva_keras = {f'{name}_out': Yva[:, j] for j, name in enumerate(intents)}
        
        # ---------- Train ----------
        print("  Training LSTM Baseline Model...")
        tf.keras.backend.clear_session()
        model = create_lstm_baseline(input_shape=(args.lookback, len(feats)), intent_ids=intents)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy')
        
        cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)]
        model.fit(Xtr, ytr_keras, epochs=args.epochs, batch_size=args.batch, 
                  validation_data=(Xva, yva_keras), callbacks=cb, verbose=0)

        # ---------- Tune ----------
        print("  Tuning thresholds on validation set...")
        preds_va = model.predict(Xva, verbose=0)
        scores_val = {name: preds_va[j].reshape(-1) for j, name in enumerate(intents)}
        params = {k: grid_search_params(scores_val[k], Tva, va_events, k, args.fp_budget) for k in intents}

        # ---------- Eval ----------
        print("  Evaluating on test set...")
        preds_te = model.predict(Xte, verbose=0)
        scores_test = {name: preds_te[j].reshape(-1) for j, name in enumerate(intents)}

        per_intent_alerts, ema_by_intent = {}, {}
        for k in intents:
            W, tau = params[k]['W'], params[k]['tau']
            ema = pd.Series(scores_test[k], index=Tte).ewm(span=W, adjust=False).mean().values
            ema_by_intent[k] = ema
            per_intent_alerts[k] = Tte[ema >= tau]

        tp, fn, lt, fp, correct_cause, total_cause, _, _, _ = evaluate(
            per_intent_alerts, te_events, Tte, ema_by_intent
        )
        
        # Aggregation
        days = (Tte[-1]-Tte[0]) / 1440.0 if len(Tte)>1 else 1.0
        for k in intents:
            aggs['lstm']['dr'][k].append(tp[k]/max(tp[k]+fn[k],1e-9))
            aggs['lstm']['lt'][k].append(float(np.mean(lt[k])) if lt[k] else 0.0)
        aggs['lstm']['fp'].append(fp/max(days,1e-9))
        aggs['lstm']['d_acc'].append((correct_cause/max(total_cause,1)) if total_cause > 0 else 0.0)

    # ---------- Report ----------
    if not aggs['lstm']['fp']: return

    print(f"\n--- Baseline: LSTM (Window={args.lookback}) ({args.folds}-Fold CV) ---")
    print(f"{'Intent':<15} | {'Detection Rate':<20} | {'Avg Lead Time (min)':<22} | {'FP Rate/Day':<18} | {'Disambiguation Acc.':<20}")
    print('-'*115)
    for name in intents:
        dr_m, dr_s = np.mean(aggs['lstm']['dr'][name]), np.std(aggs['lstm']['dr'][name])
        lt_m, lt_s = np.mean(aggs['lstm']['lt'][name]), np.std(aggs['lstm']['lt'][name])
        print(f"{name:<15} | {dr_m:.2%} ± {dr_s:.2%}      | {lt_m:.2f} ± {lt_s:.2f}        | {' ':<18} | {' ':<20}")
    print('-'*115)
    fp_m, fp_s = np.mean(aggs['lstm']['fp']), np.std(aggs['lstm']['fp'])
    da_m, da_s = np.mean(aggs['lstm']['d_acc']), np.std(aggs['lstm']['d_acc'])
    print(f"{'Overall':<15} | {' ':<20} | {' ':<22} | {fp_m:.2f} ± {fp_s:.2f}       | {da_m:.2%} ± {da_s:.2%} (Root Cause Acc.)")

    with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
        json.dump({'folds': aggs['lstm']}, f, indent=2)

if __name__=='__main__':
    main()