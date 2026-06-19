
"""
run_hybrid_experiment.py
------------------------
Train/evaluate two variants:
  (A) Distilled student (no teacher at inference): --use_teacher_inputs False
  (B) Teacher-augmented student (teacher-as-feature at inference): --use_teacher_inputs True

Both use: per‑intent distillation to match Logistic (OvR) and *gate* KL to a teacher distribution,
plus cause‑aware gate supervision and head decorrelation.

"""
import os, argparse, json, numpy as np, pandas as pd, tensorflow as tf
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from determinism import set_global_determinism
from make_forward_looking_dataset import make_forward_looking_dataset
from custom_loss import head_decorrelation  # reuse existing decorrelation
from hybrid_mild_model import create_mild_moe_with_teacher
'''
import os, argparse, json, numpy as np, pandas as pd

# (Optional) pick which GPU to use (0 = first GPU). Comment out to use all visible GPUs.
# IMPORTANT: must be set BEFORE importing tensorflow.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# (Optional) reduce TF log spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

# ---------------- CUDA / GPU setup (MUST run right after importing TF) ----------------
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError(
        "TensorFlow does not see any CUDA GPU. "
        "Install a GPU-enabled TensorFlow build + NVIDIA driver/CUDA, "
        "and make sure CUDA_VISIBLE_DEVICES is not hiding the GPU."
    )

# Prevent TF from grabbing all VRAM at once
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Force using only the first GPU (optional, but makes behavior explicit)
tf.config.set_visible_devices(gpus[0], "GPU")
print("[TF] Using GPU:", gpus[0].name)
print("[TF] Built with CUDA:", tf.test.is_built_with_cuda())

from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from determinism import set_global_determinism
from make_forward_looking_dataset import make_forward_looking_dataset
from custom_loss import head_decorrelation  # reuse existing decorrelation
from hybrid_mild_model import create_mild_moe_with_teacher
'''

# ------------------ helpers (mirroring your existing scripts) ------------------
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
    #import pandas as pd, numpy as np
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

# ------------------ new KD and gate losses ------------------
def _logit(p, eps=1e-6):
    p = tf.clip_by_value(p, eps, 1.0-eps)
    return tf.math.log(p) - tf.math.log(1.0 - p)

def distillation_focal_with_temperature(alpha=0.2, gamma=2.0, horizon_min=120, pos_weight=3.0,
                                        min_time_weight=0.3, temperature=2.0):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
    eps = tf.constant(1e-6, tf.float32)

    def base_focal(y_ttf, y_pred):
        is_pos = tf.cast(y_ttf > 0.0, tf.float32)
        ttf = tf.maximum(y_ttf, eps)
        time_w = min_time_weight + (1.0 - min_time_weight) * tf.clip_by_value(ttf / float(horizon_min), 0.0, 1.0)
        cls_w = 1.0 + (pos_weight - 1.0) * is_pos
        w = cls_w * time_w
        per_bce = bce(is_pos, y_pred)
        p_t = y_pred * is_pos + (1.0 - y_pred) * (1.0 - is_pos)
        mod = tf.pow(1.0 - p_t, gamma)
        return tf.reduce_mean(mod * per_bce * tf.stop_gradient(tf.squeeze(w, axis=-1)))

    kld = tf.keras.losses.KLDivergence()

    def loss(y_true_and_soft, y_pred):
        y_ttf = tf.slice(y_true_and_soft, [0, 0], [-1, 1])
        y_soft = tf.slice(y_true_and_soft, [0, 1], [-1, 1])

        # Ground-truth focal component
        gt_loss = base_focal(y_ttf, y_pred)

        # Temperature-scaled distillation: convert to logits then soften
        y_soft_T = tf.sigmoid(_logit(y_soft) / temperature)
        y_pred_T = tf.sigmoid(_logit(y_pred) / temperature)
        y_soft_dist = tf.concat([y_soft_T, 1.0 - y_soft_T], axis=1)
        y_pred_dist = tf.concat([y_pred_T, 1.0 - y_pred_T], axis=1)
        distill = kld(y_soft_dist, y_pred_dist)

        return alpha * gt_loss + (1.0 - alpha) * distill

    return loss

def gate_kldiv_with_teacher_new(supervise_weight=0.3, teacher_weight=0.15, sparsity_weight=0.005):
    kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)

    def loss(y_true_concat, p_gate):
        K = tf.shape(p_gate)[-1]
        y_true_concat = tf.cast(y_true_concat, tf.float32)
        p_gate = tf.cast(p_gate, tf.float32)

        y_gate_true    = y_true_concat[:, :K]
        y_gate_teacher = y_true_concat[:, K:]

        s_true = tf.reduce_sum(y_gate_true, axis=-1, keepdims=True)
        has_cause = tf.cast(s_true > 0, tf.float32)

        y_dist_true = tf.where(
            s_true > 0,
            y_gate_true / tf.maximum(s_true, 1e-6),
            y_gate_true
        )

        s_teacher = tf.reduce_sum(y_gate_teacher, axis=-1, keepdims=True)
        y_dist_teacher = tf.where(
            s_teacher > 0,
            y_gate_teacher / tf.maximum(s_teacher, 1e-6),
            tf.ones_like(y_gate_teacher) / tf.cast(K, tf.float32)
        )

        kl_true_raw = kld(y_dist_true, p_gate)       # shape: (batch,)
        kl_teacher_raw = kld(y_dist_teacher, p_gate) # shape: (batch,)

        mask = tf.squeeze(has_cause, axis=-1)
        denom = tf.maximum(tf.reduce_sum(mask), 1.0)

        kl_true = supervise_weight * tf.reduce_sum(kl_true_raw * mask) / denom
        kl_teacher = teacher_weight * tf.reduce_sum(kl_teacher_raw * mask) / denom

        sparsity = sparsity_weight * tf.reduce_mean(
            tf.reduce_sum(p_gate * (1.0 - p_gate), axis=-1)
        )

        return kl_true + kl_teacher + sparsity

    return loss

def gate_kldiv_with_teacher(supervise_weight=0.3, teacher_weight=0.15, sparsity_weight=0.005):
    kld = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
    def loss(y_true_concat, p_gate):
        K = tf.shape(p_gate)[-1]
        y_true_concat = tf.cast(y_true_concat, tf.float32)
        p_gate = tf.cast(p_gate, tf.float32)

        y_gate_true    = y_true_concat[:, :K]
        y_gate_teacher = y_true_concat[:, K:]
        s_true = tf.reduce_sum(y_gate_true, axis=-1, keepdims=True)
        # normalize over active intents (cause supervision)
        y_dist_true = tf.where(s_true > 0, y_gate_true / s_true, y_gate_true)
        mask = tf.cast(s_true > 0, tf.float32)

        # teacher distribution (already a distribution but re-normalize for safety)
        s_teacher = tf.reduce_sum(y_gate_teacher, axis=-1, keepdims=True)
        y_dist_teacher = tf.where(s_teacher > 0, y_gate_teacher / s_teacher, tf.ones_like(y_gate_teacher)/tf.cast(K, tf.float32))

        kl_true = supervise_weight * tf.reduce_mean(kld(y_dist_true, p_gate) * tf.squeeze(mask, -1))
        kl_teacher = teacher_weight * tf.reduce_mean(kld(y_dist_teacher, p_gate))
        sparsity = sparsity_weight * tf.reduce_mean(tf.reduce_sum(p_gate * (1.0 - p_gate), axis=-1))
        return kl_true + kl_teacher + sparsity
    return loss

# [IN: run_hybrid_experiment.py]
# REPLACE the existing 'evaluate' function with this:

def evaluate(per_intent_alerts, te_events, Tte, ema_by_intent, gate_probs):
    import numpy as np
    tp = {k:0 for k in te_events}; fn = {k:0 for k in te_events}; lt = {k:[] for k in te_events}
    #all_windows = [(e['start'], e_['end']) for v in te_events.values() for e_ in v] # Renamed 'e'
    all_windows = [(e_['start'], e_['end']) for v in te_events.values() for e_ in v]
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
            # Check if this event 'eobj' is the designated root cause
            event_type = eobj.get('type', '')
            is_root_cause_event = event_type.endswith('_cause') or event_type.startswith('independent')
            
            # Only evaluate disambiguation if this 'eobj' is the true root cause
            if not is_root_cause_event:
                continue

            true_root_cause = true_name
            total_events_with_cause += 1

            # 2. Find the Earliest Alert Time for this event
            earliest = None
            for name, arr in per_intent_alerts.items():
                arr2 = arr[(arr >= eobj['start']) & (arr <= eobj['failure_time'])]
                if arr2.size>0:
                    t0 = arr2[0]
                    if (earliest is None) or (t0 < earliest): 
                        earliest = t0
            
            if earliest is None:
                continue # No alert fired, so it's a miss

            # 3. Find MILD's Predicted Root Cause (using the Gating Network)
            idx_at_alert = time_to_idx.get(int(earliest), None)
            if idx_at_alert is None:
                continue

            gate_distribution = gate_probs[idx_at_alert]
            predicted_cause_idx = np.argmax(gate_distribution)
            predicted_root_cause = intent_ids[predicted_cause_idx]
            
            # 4. Compare True vs. Predicted
            if predicted_root_cause == true_root_cause:
                total_correctly_identified += 1
            
            conf_key = (true_root_cause, predicted_root_cause)
            conf[conf_key] = conf.get(conf_key, 0) + 1

    # We are replacing the old (ex, ns, jh, nm) with (total_correctly_identified, total_events_with_cause)
    # To keep the function signature, we'll return 0 for the old Jaccard metrics
    return tp, fn, lt, fp_total, total_correctly_identified, total_events_with_cause, 0, 0, conf

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--horizon', type=int, default=120)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fp_budget', type=float, default=0.25)
    ap.add_argument('--gate_supervise', type=float, default=0.2)
    ap.add_argument('--gate_sparsity', type=float, default=0.005)
    ap.add_argument('--gate_teacher', type=float, default=0.15, help='Weight of KL to teacher distribution in the gate loss')
    ap.add_argument('--distill_alpha', type=float, default=0.25, help='Weight for GT in distillation (1=no distill)')
    ap.add_argument('--temp', type=float, default=2.0, help='Distillation temperature')
    #ap.add_argument('--use_teacher_inputs', action='store_true', help='If set, feed teacher distribution as model input at inference')
    ap.add_argument('--use_teacher_gate', action='store_true')
    ap.add_argument('--lambda_decorr', type=float, default=1e-4)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_global_determinism(args.seed)

    df = pd.read_csv(args.data)
    events_by_intent = read_json(os.path.join(os.path.dirname(args.data), 'events_by_intent.json'))
    intent_ids = sorted(events_by_intent.keys())

    # --- blocked CV as in your scripts
    N = len(df); fold_sizes = [N//args.folds]*(args.folds-1) + [N - (args.folds-1)*(N//args.folds)]
    starts = [sum(fold_sizes[:i]) for i in range(args.folds)]
    tests = [(s, s+fold_sizes[i]) for i,s in enumerate(starts)]
    per_fold = []

    for i,(s,e) in enumerate(tests):
        print(f"\n--- Fold {i+1}/{args.folds}: test [{s}:{e}) ---")
        df_test = df.iloc[s:e].reset_index(drop=True)
        df_train_full = df.iloc[:s].reset_index(drop=True)
        if len(df_train_full) < 1000:
            print("  [skip] not enough train history before this test block."); continue

        split = int(0.8 * len(df_train_full))
        df_train = df_train_full.iloc[:split].reset_index(drop=True)
        df_val = df_train_full.iloc[split:].reset_index(drop=True)

        tr_events = filter_events(events_by_intent, df_train['t'].min(), df_train['t'].max())
        va_events = filter_events(events_by_intent, df_val['t'].min(),   df_val['t'].max())
        te_events = filter_events(events_by_intent, df_test['t'].min(),  df_test['t'].max())

        from make_forward_looking_dataset import make_forward_looking_dataset
        Xtr, Ytr_ttf, Ytr_bin, Ytr_cause, Ttr, feats = make_forward_looking_dataset(df_train, tr_events, intent_ids, args.horizon,
                                                                                    base_features=['cpu_pct','ram_pct','storage_pct','snet','sri','cpu_delta','sri_delta','api_latency','analytics_tput','telemetry_queue'])
        
        from make_forward_looking_dataset import _engineer

        Xtr_df, feats = _engineer(
            df_train,
            base_features=[
                'cpu_pct','ram_pct','storage_pct',
                'snet','sri','cpu_delta','sri_delta',
                'api_latency','analytics_tput',
                'telemetry_queue'
            ]
        )

        df_save = Xtr_df.copy()

        for j, intent in enumerate(intent_ids):
            df_save[f'{intent}_ttf'] = Ytr_ttf[:, j]
            df_save[f'{intent}_bin'] = Ytr_bin[:, j]
            df_save[f'{intent}_cause'] = Ytr_cause[:, j]

        df_save['t'] = Ttr

        df_save.to_csv(
            os.path.join(args.outdir, f'fold{i+1}_train_with_targets.csv'),
            index=False
        )


        Xva, Yva_ttf, Yva_bin, Yva_cause, Tva, _    = make_forward_looking_dataset(df_val,   va_events, intent_ids, args.horizon,
                                                                                    base_features=['cpu_pct','ram_pct','storage_pct','snet','sri','cpu_delta','sri_delta','api_latency','analytics_tput','telemetry_queue'])
        Xte, Yte_ttf, Yte_bin, Yte_cause, Tte, _    = make_forward_looking_dataset(df_test,  te_events, intent_ids, args.horizon,
                                                                                    base_features=['cpu_pct','ram_pct','storage_pct','snet','sri','cpu_delta','sri_delta','api_latency','analytics_tput','telemetry_queue'])

        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr); Xva = scaler.transform(Xva); Xte = scaler.transform(Xte)

        # ---------------- teacher: per‑intent Logistic (OvR) ----------------
        print("  Training teacher (Logistic OvR) ...")
        teachers = {}
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

        Ytr_bin_fold = build_binary_labels(df_train, tr_events, args.horizon)
        for j, name in enumerate(intent_ids):
            if np.unique(Ytr_bin_fold[:, j]).size < 2:
                teachers[name] = None
                print(f"  [WARN] No positives for intent '{name}' in this fold.")
                continue
            lr = LogisticRegression(max_iter=600, random_state=args.seed, class_weight='balanced')
            lr.fit(Xtr, Ytr_bin_fold[:, j])
            teachers[name] = lr
            P_tr[:, j] = lr.predict_proba(Xtr)[:,1]
            P_va[:, j] = lr.predict_proba(Xva)[:,1]
            P_te[:, j] = lr.predict_proba(Xte)[:,1]

        # Teacher distribution for gating (use temperature on logits -> softmax)
        def prob_to_dist(P, T=2.0, eps=1e-6):
            # convert each row of P (K probs) into a distribution over intents
            logits = np.log(np.clip(P, eps, 1-eps)) - np.log(np.clip(1-P, eps, 1-eps))
            logits = logits / T
            m = np.max(logits, axis=1, keepdims=True)
            e = np.exp(logits - m)
            Z = np.sum(e, axis=1, keepdims=True) + eps
            return e / Z

        D_tr = prob_to_dist(P_tr, T=args.temp)
        D_va = prob_to_dist(P_va, T=args.temp)
        D_te = prob_to_dist(P_te, T=args.temp)

        # ---------------- Targets ----------------
        ytr, yva = {}, {}
        for j, name in enumerate(intent_ids):
            ytr[f'{name}_out'] = np.concatenate([Ytr_ttf[:, j:j+1], P_tr[:, j:j+1]], axis=1)
            yva[f'{name}_out'] = np.concatenate([Yva_ttf[:, j:j+1], P_va[:, j:j+1]], axis=1)

        # Gate target: concat[cause-aware, teacher dist]
        gate_tr = Ytr_cause.astype('float32').copy()
        gate_va = Yva_cause.astype('float32').copy()
        # fill zeros with binary positives as in your script
        #mask_tr = (gate_tr.sum(axis=1) == 0)
        #mask_va = (gate_va.sum(axis=1) == 0)
        #gate_tr[mask_tr, :] = Ytr_bin[mask_tr, :]
        #gate_va[mask_va, :] = Yva_bin[mask_va, :]
        
        ytr['gate'] = np.concatenate([gate_tr, D_tr], axis=1)
        yva['gate'] = np.concatenate([gate_va, D_va], axis=1)

        # head decorrelation dummy
        ytr['head_concat'] = np.zeros((len(Xtr), len(intent_ids)*16), dtype='float32')
        yva['head_concat'] = np.zeros((len(Xva), len(intent_ids)*16), dtype='float32')

        # Teacher inputs (if enabled) else zeros
        # Ttr_in = D_tr if args.use_teacher_inputs else np.zeros_like(D_tr)
        # Tva_in = D_va if args.use_teacher_inputs else np.zeros_like(D_va)
        # Tte_in = D_te if args.use_teacher_inputs else np.zeros_like(D_te)

        # ---------------- Model ----------------
        tf.keras.backend.clear_session()
        #model = create_mild_moe_with_teacher(num_features=Xtr.shape[1], num_teacher=len(intent_ids), intent_ids=intent_ids, use_teacher_gate=args.use_teacher_gate,)

        model = create_mild_moe_with_teacher(
            num_features=Xtr.shape[1],
            num_teacher=len(intent_ids),
            intent_ids=intent_ids,
            use_teacher_gate=args.use_teacher_gate,
        )

        losses = {f'{name}_out': distillation_focal_with_temperature(alpha=args.distill_alpha, horizon_min=args.horizon, temperature=args.temp)
                  for name in intent_ids}
        losses.update({
            'gate': gate_kldiv_with_teacher(supervise_weight=args.gate_supervise, teacher_weight=args.gate_teacher, sparsity_weight=args.gate_sparsity),
            'head_concat': head_decorrelation(args.lambda_decorr)
        })

        #cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)]
        #model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=losses)
        #print("  Training student (HYBRID MILD) ...")
        
        cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)]
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=losses)
        print("  Training student (HYBRID MILD) ...")

        if args.use_teacher_gate:
            train_inputs = [Xtr, D_tr]
            val_inputs = [Xva, D_va]
            test_inputs = [Xte, D_te]
        else:
            train_inputs = Xtr
            val_inputs = Xva
            test_inputs = Xte
        
        #model.fit([Xtr, Ttr_in], ytr, epochs=args.epochs, batch_size=args.batch, validation_data=([Xva, Tva_in], yva), callbacks=cb, verbose=0)
        model.fit(train_inputs, ytr, epochs=args.epochs, batch_size=args.batch, validation_data=(val_inputs, yva), callbacks=cb, verbose=0)

        # ---------------- Validation threshold tuning & Test eval ----------------
        #preds_va = model.predict([Xva, Tva_in], verbose=0)
        #preds_te = model.predict([Xte, Tte_in], verbose=0)

        preds_va = model.predict(val_inputs, verbose=0)
        preds_te = model.predict(test_inputs, verbose=0)

        # last two outputs are gate, head_concat; first K are per‑intent risks
        per_intent_va = preds_va[:len(intent_ids)]
        per_intent_te = preds_te[:len(intent_ids)]

        gate_probs_te = preds_te[-2]
        #np.save(os.path.join(args.outdir, f'gate_probs_fold{i+1}.npy'), gate_probs_te)

        params = {}
        for j, name in enumerate(intent_ids):
            params[name] = grid_search_params(per_intent_va[j].reshape(-1), df_val['t'].values, va_events, name, fp_budget_per_day=args.fp_budget)

        # Apply on test
        Tte_vals = df_test['t'].values
        ema_by_intent = {}; per_intent_alerts = {}
        #import pandas as pd
        for j, name in enumerate(intent_ids):
            W = params[name]['W']; tau = params[name]['tau']
            ema = pd.Series(per_intent_te[j].reshape(-1), index=Tte_vals).ewm(span=W, adjust=False).mean().values
            ema_by_intent[name] = ema
            per_intent_alerts[name] = Tte_vals[ema >= tau]

        tp, fn, lt, fp_total, correct_cause, total_cause, _, _, conf = evaluate(
                                            per_intent_alerts, 
                                            te_events, Tte_vals, 
                                            ema_by_intent, 
                                            gate_probs_te )

        #days = (Tte_vals[-1]-Tte_vals[0]) / 1440.0 if len(Tte_vals)>1 else 1.0
        # --------------------------
        days = (Tte_vals[-1]-Tte_vals[0]) / 1440.0 if len(Tte_vals)>1 else 1.0
        fold_metrics = {
            'per_intent': {
                name: {
                    'detection_rate': (tp[name]/max(tp[name]+fn[name],1e-9)),
                    'avg_lead_time': (float(np.mean(lt[name])) if lt[name] else 0.0),
                } for name in intent_ids
            },
            'overall': {
                'fp_per_day': fp_total / max(days, 1e-9),
                'disamb_accuracy': (correct_cause / max(total_cause, 1)) if total_cause > 0 else 0.0, # <-- NEW METRIC
                'num_cause_events': int(total_cause) # <-- NEW METRIC

            },
            'confusion': {f'{a}->{b}': c for (a,b),c in conf.items()},
            'params': params
        }
        per_fold.append(fold_metrics)

        # Export fold time series
        ts_path = os.path.join(args.outdir, f'fold{i+1}_timeseries.csv')
        cols = {'t': Tte_vals}
        for name in intent_ids:
            cols[f'ema_{name}'] = ema_by_intent[name]
        pd.DataFrame(cols).to_csv(ts_path, index=False)

        # Export fold time series
        ts_path = os.path.join(args.outdir, f'fold{i+1}_timeseries.csv')
        cols = {'t': Tte_vals}
        for j, name in enumerate(intent_ids):
            cols[f'ema_{name}'] = ema_by_intent[name]
            cols[f'gate_{name}'] = gate_probs_te[:, j].reshape(-1)

        pd.DataFrame(cols).to_csv(ts_path, index=False)

        # Optional fallback for the plotting script
        np.save(os.path.join(args.outdir, f'gate_probs_fold{i+1}.npy'), gate_probs_te)

    if not per_fold:
        print("No folds evaluated."); return
    # aggregate
    from collections import defaultdict
    #dr = defaultdict(list); lt = defaultdict(list); fp = []; d_exact = []; d_jacc = []
    dr = defaultdict(list); lt = defaultdict(list); fp = []; d_acc = [] # <-- Change d_exact to d_acc
    for fm in per_fold:
        for name,v in fm['per_intent'].items():
            dr[name].append(v['detection_rate']); lt[name].append(v['avg_lead_time'])
        fp.append(fm['overall']['fp_per_day'])
        d_acc.append(fm['overall']['disamb_accuracy']) # <-- NEW METRIC


    print("\n--- HYBRID‑MILD Final Definitive Experiment Results ---")
    print(f"{'Intent':<15} | {'Detection Rate':<20} | {'Avg Lead Time (min)':<22} | {'FP Rate/Day':<18} | {'Disambiguation Acc.':<20}")
    print('-'*115)
    for name in intent_ids:
        dr_m, dr_s = np.mean(dr[name]), np.std(dr[name])
        lt_m, lt_s = np.mean(lt[name]), np.std(lt[name])
        print(f"{name:<15} | {dr_m:.2%} ± {dr_s:.2%}      | {lt_m:.2f} ± {lt_s:.2f}        | {' ':<18} | {' ':<20}")
    print('-'*115)
    fp_m, fp_s = np.mean(fp), np.std(fp)
    da_m, da_s = np.mean(d_acc), np.std(d_acc) # <-- NEW METRIC
    print(f"{'Overall':<15} | {' ':<20} | {' ':<22} | {fp_m:.2f} ± {fp_s:.2f}       | {da_m:.2%} ± {da_s:.2%} (Root Cause Acc.)")

    # dump raw per‑fold metrics
    with open(os.path.join(args.outdir, 'metrics.json'), 'w') as f:
        json.dump({'folds': per_fold}, f, indent=2)

if __name__=='__main__':
    main()
