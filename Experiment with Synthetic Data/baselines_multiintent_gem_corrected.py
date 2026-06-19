import os, json, argparse, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from determinism import set_global_determinism
from collections import defaultdict

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
    import numpy as np, pandas as pd
    #Ws = [3,5,8,13,21]
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

# [NEW]
def evaluate(per_intent_alerts, te_events, Tte, ema_by_intent):
    """
    Evaluates detection, lead time, and NEW UNIFIED root-cause accuracy.
    The root cause is predicted using the heuristic:
    "Intent with the highest EMA score at the time of the first alert."
    """
    import numpy as np
    tp = {k:0 for k in te_events}; fn = {k:0 for k in te_events}; lt = {k:[] for k in te_events}
    all_windows = [(e_['start'], e_['end']) for v in te_events.values() for e_ in v]
    fp_total = 0
    
    # 1. Calculate Detection Rate (TP/FN) and Lead Time
    for name, alerts in per_intent_alerts.items():
        for eobj in te_events.get(name, []):
            in_win = alerts[(alerts >= eobj['start']) & (alerts <= eobj['failure_time'])]
            if in_win.size>0:
                tp[name]+=1; lt[name].append(int(eobj['failure_time'] - in_win[0]))
            else:
                fn[name]+=1
        # false positives
        for a in alerts:
            if not any((s <= a <= e) for (s,e) in all_windows): fp_total += 1

    # 2. Calculate NEW Unified Disambiguation Accuracy
    time_to_idx = {int(t): idx for idx,t in enumerate(Tte)}
    intent_ids = list(ema_by_intent.keys()) # Get sorted intent names
    
    total_events_with_cause = 0
    total_correctly_identified = 0
    conf = {} # Confusion matrix

    for true_name, evs in te_events.items():
        for eobj in evs:
            # 2a. Find the True Root Cause for this event
            # Check if this event 'eobj' is the designated root cause
            event_type = eobj.get('type', '')
            is_root_cause_event = event_type.endswith('_cause') or event_type.startswith('independent')
            
            # Only evaluate disambiguation if this 'eobj' is the true root cause
            if not is_root_cause_event:
                continue

            true_root_cause = true_name
            total_events_with_cause += 1

            # 2b. Find the Earliest Alert Time for this event
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

            # 2c. Find Baseline's Predicted Root Cause (using highest EMA score heuristic)
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
                    # Get the specific EMA score for this intent at the alert time
                    score = ema_by_intent[name][idx_at_alert] 
                    if score > best_score:
                        best_score = score
                        predicted_root_cause = name
            
            # 2d. Compare True vs. Predicted
            if predicted_root_cause == true_root_cause:
                total_correctly_identified += 1
            
            conf_key = (true_root_cause, predicted_root_cause)
            conf[conf_key] = conf.get(conf_key, 0) + 1

    # Return new unified metrics
    return tp, fn, lt, fp_total, total_correctly_identified, total_events_with_cause, 0, 0, conf

# [NEW]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--fp_budget', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    set_global_determinism(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.data)
    events_by_intent = read_json(os.path.join(os.path.dirname(args.data), 'events_by_intent.json'))
    intents = sorted(events_by_intent.keys())
    feats = ['cpu_pct','ram_pct','storage_pct','snet','sri','cpu_delta','sri_delta','api_latency','analytics_tput','telemetry_queue']

    N = len(df)
    fold_sizes = [N//args.folds]*(args.folds-1) + [N - (args.folds-1)*(N//args.folds)]
    starts = [sum(fold_sizes[:i]) for i in range(args.folds)]
    tests = [(s, s+fold_sizes[i]) for i,s in enumerate(starts)]

    # UPDATED: Aggregators to store metrics from each fold for each baseline
    aggs = {
        'kpi': {'dr': defaultdict(list), 'lt': defaultdict(list), 'fp': [], 'd_acc': []},
        'dist': {'dr': defaultdict(list), 'lt': defaultdict(list), 'fp': [], 'd_acc': []},
        'log': {'dr': defaultdict(list), 'lt': defaultdict(list), 'fp': [], 'd_acc': []}
    }

    for i, (s, e) in enumerate(tests):
        print(f"\n--- Fold {i+1}/{args.folds}: test [{s}:{e}) ---")
        df_test = df.iloc[s:e].reset_index(drop=True)
        df_train = df.iloc[:s].reset_index(drop=True)

        if len(df_train) < 1000:
            print("  [skip] not enough train history before this test block.")
            continue

        tr_events = filter_events(events_by_intent, df_train['t'].min(), df_train['t'].max())
        te_events = filter_events(events_by_intent, df_test['t'].min(), df_test['t'].max())
        Tte = df_test['t'].values
        days = (Tte[-1]-Tte[0]) / 1440.0 if len(Tte)>1 else 1.0

        scaler = StandardScaler().fit(df_train[feats].values)
        Xtr = scaler.transform(df_train[feats].values); Xte = scaler.transform(df_test[feats].values)
        Xtr_df = pd.DataFrame(Xtr, columns=feats); Xte_df = pd.DataFrame(Xte, columns=feats)

        # ----------Tuned Baseline 1: Tuned Weighted‑KPI (using LR Coeffs) ----------
        print("  Pre-training Logistic Regression for Baseline 1 weights...")
        def build_labels_for_lr(df_slice, evs_by_intent, horizon=120):
            K=len(intents); Y=np.zeros((len(df_slice),K),dtype=np.float32); t=df_slice['t'].values
            for j,intent in enumerate(intents):
                for e_ in evs_by_intent.get(intent, []): # Renamed e -> e_
                    start = max(e_['failure_time']-horizon, int(t[0])); end=min(e_['failure_time'], int(t[-1])+1)
                    if end<=start: continue
                    m=(t>=start)&(t<end); Y[m,j]=1.0
            return Y
            
        Ytr_lr = build_labels_for_lr(df_train, tr_events)
        lr_models = {}
        lr_coefs = {}
        for j,k in enumerate(intents):
            if np.sum(Ytr_lr[:, j]) > 0 and np.sum(Ytr_lr[:, j]) < len(Ytr_lr):
                lr = LogisticRegression(max_iter=200, random_state=args.seed, class_weight='balanced')
                lr.fit(Xtr, Ytr_lr[:,j])
                lr_models[k] = lr 
                lr_coefs[k] = np.abs(lr.coef_[0]) 
            else:
                print(f"  [WARN] Baseline 1: Skipping LR training for '{k}' due to lack of positive/negative samples in fold.")
                lr_models[k] = None
                lr_coefs[k] = np.zeros(len(feats)) 

        print("  Calculating scores for Baseline 1 (Tuned Weighted-KPI)...")
        def risk_weighted(series, weights, lower_is_bad):
            z = series.copy(); r = 0
            total_weight = sum(weights.values()) if sum(weights.values()) > 1e-9 else 1.0
            normalized_weights = {k: w / total_weight for k, w in weights.items()}
            
            for k,w in normalized_weights.items():
                if k not in z.columns: continue
                s = z[k]
                if k in lower_is_bad: s = -s 
                r += w * s
            if r.max() == r.min(): return pd.Series(0.0, index=series.index)
            return (r - r.min()) / (r.max() - r.min() + 1e-9)

        feature_weights_by_intent = {}
        for k in intents:
            feature_weights_by_intent[k] = {feat_name: lr_coefs[k][feat_idx] for feat_idx, feat_name in enumerate(feats)}
        lower_is_bad = set(['sri','snet','analytics_tput']) 
        scores_train = {k: risk_weighted(Xtr_df, feature_weights_by_intent[k], lower_is_bad).values for k in intents}
        scores_test  = {k: risk_weighted(Xte_df, feature_weights_by_intent[k], lower_is_bad).values for k in intents}

        print("  Tuning thresholds and evaluating Baseline 1...")
        params = {k: grid_search_params(scores_train[k], df_train['t'].values, tr_events, k, args.fp_budget) for k in intents}
        per_intent_alerts, ema_by_intent = {}, {}
        for k in intents:
            W, tau = params[k]['W'], params[k]['tau']
            ema = pd.Series(scores_test[k], index=Tte).ewm(span=W, adjust=False).mean().values 
            ema_by_intent[k] = ema; per_intent_alerts[k] = Tte[ema >= tau] 

        # UPDATED: Calling new evaluate function
        tp, fn, lt, fp, correct_cause, total_cause, _, _, _ = evaluate(per_intent_alerts, te_events, Tte, ema_by_intent)
        
        agg_key = 'kpi'
        for k in intents:
            aggs[agg_key]['dr'][k].append(tp[k]/max(tp[k]+fn[k],1e-9))
            aggs[agg_key]['lt'][k].append(float(np.mean(lt[k])) if lt[k] else 0.0)
        aggs[agg_key]['fp'].append(fp/max(days,1e-9))
        # UPDATED: Storing new metric
        aggs[agg_key]['d_acc'].append((correct_cause/max(total_cause,1)) if total_cause > 0 else 0.0)


        # ---------- Corrected Baseline 2: Target-Based Distance (Paper's Method) ----------
        print("  Calculating target vector from golden period...")
        all_event_times_tr = set()
        buffer_minutes = 30
        for intent_events in tr_events.values():
            for e_ in intent_events: # Renamed e -> e_
                start_exclude = max(int(df_train['t'].min()), e_['start'] - buffer_minutes)
                end_exclude = min(int(df_train['t'].max()), e_['end'] + buffer_minutes)
                all_event_times_tr.update(range(start_exclude, end_exclude + 1))

        golden_indices = df_train[~df_train['t'].isin(all_event_times_tr)].index
        
        if len(golden_indices) < 100:
             print(f"  [WARN] Fold {i+1}: Insufficient golden period data ({len(golden_indices)} points). Using full train mean as fallback target.")
             target_vector_fold_unscaled = df_train[feats].mean()
        else:
             print(f"  [INFO] Fold {i+1}: Found golden period ({len(golden_indices)} points).")
             target_vector_fold_unscaled = df_train.loc[golden_indices, feats].mean()

        target_vector_fold_scaled = scaler.transform(target_vector_fold_unscaled.values.reshape(1, -1))[0]
        target_dict_scaled = dict(zip(feats, target_vector_fold_scaled))
        
        feat_sel = {'telemetry': ['sri','snet','ram_pct','cpu_pct','telemetry_queue'], 
                    'api': ['api_latency','cpu_pct','sri'], 
                    'analytics': ['snet','analytics_tput','cpu_pct']}
        scores_train2, scores_test2 = {},{}
        
        for k in intents:
            sub_feats = feat_sel[k]
            target_subset_scaled = np.array([target_dict_scaled[f] for f in sub_feats])
            scores_train2[k] = np.sqrt(np.sum((Xtr_df[sub_feats].values - target_subset_scaled)**2, axis=1))
            scores_test2[k]  = np.sqrt(np.sum((Xte_df[sub_feats].values  - target_subset_scaled)**2, axis=1))
            
            for S in [scores_train2, scores_test2]:
                s = S[k]; 
                s_min, s_max = s.min(), s.max()
                S[k] = (s - s_min) / (s_max - s_min + 1e-9) if (s_max - s_min) > 1e-9 else np.zeros_like(s)

        params2 = {k: grid_search_params(scores_train2[k], df_train['t'].values, tr_events, k, args.fp_budget) for k in intents}
        per_intent_alerts2, ema_by_intent2 = {}, {}
        for k in intents:
            W,tau = params2[k]['W'], params2[k]['tau']
            ema = pd.Series(scores_test2[k], index=Tte).ewm(span=W, adjust=False).mean().values 
            ema_by_intent2[k] = ema; per_intent_alerts2[k] = Tte[ema >= tau]
        
        # UPDATED: Calling new evaluate function
        tp2, fn2, lt2, fp2, correct_cause2, total_cause2, _, _, _ = evaluate(per_intent_alerts2, te_events, Tte, ema_by_intent2)
        
        agg_key = 'dist'
        for k in intents:
            aggs[agg_key]['dr'][k].append(tp2[k]/max(tp2[k]+fn2[k],1e-9))
            aggs[agg_key]['lt'][k].append(float(np.mean(lt2[k])) if lt2[k] else 0.0)
        aggs[agg_key]['fp'].append(fp2/max(days,1e-9))
        # UPDATED: Storing new metric
        aggs[agg_key]['d_acc'].append((correct_cause2/max(total_cause2,1)) if total_cause2 > 0 else 0.0)


        # ---------- Baseline 3: 1‑vs‑rest Logistic Regression (per intent) ----------
        def build_labels(df_slice, evs_by_intent, horizon=120):
            K=len(intents); Y=np.zeros((len(df_slice),K),dtype=np.float32); t=df_slice['t'].values
            for j,intent in enumerate(intents):
                for e_ in evs_by_intent.get(intent, []): # Renamed e -> e_
                    start = max(e_['failure_time']-horizon, int(t[0])); end=min(e_['failure_time'], int(t[-1])+1)
                    if end<=start: continue
                    m=(t>=start)&(t<end); Y[m,j]=1.0
            return Y
        
        Ytr = build_labels(df_train, tr_events)
        scores_train3, scores_test3 = {}, {}
        for j,k in enumerate(intents):
            # Check for valid labels again, as this is separate from Baseline 1's LR
            if np.sum(Ytr[:, j]) > 0 and np.sum(Ytr[:, j]) < len(Ytr):
                lr = LogisticRegression(max_iter=200, random_state=args.seed, class_weight='balanced')
                lr.fit(Xtr, Ytr[:,j])
                scores_train3[k] = lr.predict_proba(Xtr)[:,1]
                scores_test3[k] = lr.predict_proba(Xte)[:,1]
            else:
                print(f"  [WARN] Baseline 3: Skipping LR training for '{k}' due to lack of positive/negative samples in fold.")
                scores_train3[k] = np.zeros(len(Xtr))
                scores_test3[k] = np.zeros(len(Xte))


        params3 = {k: grid_search_params(scores_train3[k], df_train['t'].values, tr_events, k, args.fp_budget) for k in intents}
        per_intent_alerts3, ema_by_intent3 = {}, {}
        for k in intents:
            W,tau = params3[k]['W'], params3[k]['tau']
            ema = pd.Series(scores_test3[k], index=Tte).ewm(span=W, adjust=False).mean().values
            ema_by_intent3[k]=ema; per_intent_alerts3[k]=Tte[ema>=tau]
        
        # UPDATED: Calling new evaluate function
        tp3, fn3, lt3, fp3, correct_cause3, total_cause3, _, _, _ = evaluate(per_intent_alerts3, te_events, Tte, ema_by_intent3)
        
        agg_key = 'log'
        for k in intents:
            aggs[agg_key]['dr'][k].append(tp3[k]/max(tp3[k]+fn3[k],1e-9))
            aggs[agg_key]['lt'][k].append(float(np.mean(lt3[k])) if lt3[k] else 0.0)
        aggs[agg_key]['fp'].append(fp3/max(days,1e-9))
        # UPDATED: Storing new metric
        aggs[agg_key]['d_acc'].append((correct_cause3/max(total_cause3,1)) if total_cause3 > 0 else 0.0)


    if not aggs['kpi']['fp']:
        print("\nNo folds were evaluated. Increase dataset size or decrease number of folds.")
        return

    baselines_map = {
        'Weighted-KPI': 'kpi',
        'Distance': 'dist',
        'Logistic (OvR)': 'log'
    }

    # UPDATED: Final reporting section
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
        # UPDATED: Reporting new metric
        da_m, da_s = np.mean(aggs[key]['d_acc']), np.std(aggs[key]['d_acc'])
        print(f"{'Overall':<15} | {' ':<20} | {' ':<22} | {fp_m:.2f} ± {fp_s:.2f}       | {da_m:.2%} ± {da_s:.2%} (Root Cause Acc.)")


if __name__=='__main__':
    main()