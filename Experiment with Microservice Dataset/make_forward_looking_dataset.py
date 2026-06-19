
import numpy as np, pandas as pd
from typing import Dict, List, Tuple

def _engineer(df: pd.DataFrame, base_features: List[str]):
    df_e = df.copy()
    for k in ['cpu_pct','ram_pct','sri','snet','api_latency','analytics_tput','telemetry_queue']:
        for w in [5, 15]:
            df_e[f'{k}_mean_{w}'] = df_e[k].rolling(window=w, min_periods=1).mean()
            df_e[f'{k}_std_{w}']  = df_e[k].rolling(window=w, min_periods=1).std()
    df_e.fillna(0, inplace=True)
    eng_cols = [c for c in df_e.columns if c.endswith(('_mean_5','_std_5','_mean_15','_std_15'))]
    feats = base_features + eng_cols
    return df_e[feats], feats

def build_forward_labels(df: pd.DataFrame,
                         events_by_intent: Dict[str, List[dict]],
                         intent_ids: List[str],
                         horizon_min: int):
    N, K = len(df), len(intent_ids)
    y_ttf = np.zeros((N, K), dtype='float32')
    y_bin = np.zeros((N, K), dtype='float32')
    y_cause = np.zeros((N, K), dtype='float32')  # cause-aware mask for gating supervision

    tvals = df['t'].values
    intent_to_idx = {name:i for i,name in enumerate(intent_ids)}

    def add_pos(j, failure_time, start):
        start = max(start, int(tvals[0]))
        end   = min(failure_time, int(tvals[-1])+1)
        if end <= start: return
        mask = (tvals >= start) & (tvals < end)
        y_ttf[mask, j] = (failure_time - tvals[mask]).astype('float32')
        y_bin[mask, j] = 1.0

    for intent, evs in events_by_intent.items():
        j = intent_to_idx[intent]
        for e in evs:
            failure = e['failure_time']; start = failure - horizon_min
            add_pos(j, failure, start)
            typ = e.get('type','')
            # cause-aware: if event is a conflicting cause for this intent OR independent/correlated
            if typ.startswith('conflicting') and typ.endswith('cause'):
                y_cause[(tvals >= start) & (tvals < failure), j] = 1.0
            #elif typ.startswith('independent') or typ=='correlated_hw':
            #    y_cause[(tvals >= start) & (tvals < failure), j] = 1.0
            elif typ.startswith('independent') or typ.startswith('nonlinear') or typ == 'correlated_hw':
                y_cause[(tvals >= start) & (tvals < failure), j] = 1.0

            # victims are not marked in y_cause

    return y_ttf, y_bin, y_cause

def make_forward_looking_dataset(df: pd.DataFrame,
                                 events_by_intent: Dict[str, List[dict]],
                                 intent_ids: List[str],
                                 horizon_min: int,
                                 base_features: List[str]):
    Xdf, feats = _engineer(df, base_features)
    Y_ttf, Y_bin, Y_cause = build_forward_labels(df, events_by_intent, intent_ids, horizon_min)
    T = df['t'].values.astype('int64')
    X = Xdf.values.astype('float32')
    return X, Y_ttf, Y_bin, Y_cause, T, feats
