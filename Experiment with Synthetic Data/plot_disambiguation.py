# plot_fold_disambiguation.py
import os, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _load_gate_probs(outdir, fold, ema_intents, df_times):
    gate = {}
    # 1) use gate_<intent> columns if present
    for name in ema_intents:
        col = f'gate_{name}'
        if col in df_times.columns:
            gate[name] = df_times[col].values
    if gate:
        return gate
    # 2) else try a numpy file gate_probs_fold{fold}.npy of shape (T, K)
    npy_path = os.path.join(outdir, f'gate_probs_fold{fold}.npy')
    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        if arr.ndim == 2 and arr.shape[0] == len(df_times) and arr.shape[1] == len(ema_intents):
            for j, name in enumerate(ema_intents):
                gate[name] = arr[:, j]
            return gate
        else:
            print(f"[WARN] {os.path.basename(npy_path)} has shape {arr.shape}, expected ({len(df_times)}, {len(ema_intents)}). Ignoring.")
    return {}

def _load_thresholds(outdir, fold):
    mpath = os.path.join(outdir, 'metrics.json')
    if not os.path.exists(mpath): return {}
    with open(mpath, 'r') as f:
        M = json.load(f)
    folds = M.get('folds', [])
    if not (1 <= fold <= len(folds)): return {}
    params = folds[fold-1].get('params', {})
    return {k: {'W': v.get('W'), 'tau': v.get('tau')} for k, v in params.items()}

def _filter_events(events_by_intent, t_min, t_max):
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

def _pick_event(te_events):
    # Prefer a co-drift cause+victim pair
    for root in te_events:
        for e in te_events[root]:
            typ = e.get('type', '')
            if typ.startswith('conflicting_') and typ.endswith('_cause'):
                for victim, evs_v in te_events.items():
                    for v in evs_v:
                        if v.get('type','') == 'conflicting_victim':
                            if not (v['end'] < e['start'] or v['start'] > e['end']):
                                return root, victim, e
    # Fallback: any independent event
    for root in te_events:
        for e in te_events[root]:
            if e.get('type','').startswith('independent'):
                victim = next((v for v in te_events.keys() if v != root), None)
                return root, victim, e
    return None, None, None

def plot_fold(outdir, fold, dataset_events_json, out_png=None, margin=40, title=None):
    ts_path = os.path.join(outdir, f'fold{fold}_timeseries.csv')
    if not os.path.exists(ts_path):
        raise FileNotFoundError(ts_path)
    df = pd.read_csv(ts_path)
    intents = [re.sub(r'^ema_', '', c) for c in df.columns if c.startswith('ema_')]

    th   = _load_thresholds(outdir, fold)
    gate = _load_gate_probs(outdir, fold, intents, df)
    with open(dataset_events_json, 'r') as f:
        events_by_intent = json.load(f)
    te_events = _filter_events(events_by_intent, int(df['t'].min()), int(df['t'].max()))

    cause, victim, e = _pick_event(te_events)
    if e is None:
        raise RuntimeError("No event found in this fold to plot.")

    # Crop around the event with a small margin
    t0 = max(int(df['t'].min()), e['start'] - margin)
    t1 = min(int(df['t'].max()), e['end'] + margin)
    d = df[(df['t'] >= t0) & (df['t'] <= t1)].copy()

    # Colors/styles to match your figure
    C_RED  = "#b22222"
    C_BLUE = "#1f77b4"
    GREY   = (0.85, 0.85, 0.85)

    fig, ax1 = plt.subplots(figsize=(8, 9))
    ax1.plot(d['t'], d[f'ema_{cause}'],  color=C_RED,  lw=2.5, label='Risk (Cause)')
    if victim:
        ax1.plot(d['t'], d[f'ema_{victim}'], color=C_BLUE, lw=2.5, label='Risk (Victim)')
    ax1.set_ylabel('Smoothed Risk Score'); ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, ls='--', lw=0.5, alpha=0.6)

    if th.get(cause, {}).get('tau') is not None:
        ax1.hlines(th[cause]['tau'], d['t'].iloc[0], d['t'].iloc[-1], colors=C_RED,  linestyles=':', lw=2, label='Alarm Threshold (Cause)')
    if victim and th.get(victim, {}).get('tau') is not None:
        ax1.hlines(th[victim]['tau'], d['t'].iloc[0], d['t'].iloc[-1], colors=C_BLUE, linestyles=':', lw=2, label='Alarm Threshold (Victim)')

    ax1.axvspan(e['start'], e['end'], color=GREY, alpha=0.6, label='Drift Window')
    ax1.axvline(e['failure_time'], color='k', lw=2.5, ls='--', label='Failure Time')

    if gate:
        ax2 = ax1.twinx()
        if cause in gate:
            ax2.plot(d['t'], gate[cause][(df['t'] >= t0) & (df['t'] <= t1)],  color=C_RED,  ls='--', lw=2, label='Gating Probability (Cause)')
        if victim and victim in gate:
            ax2.plot(d['t'], gate[victim][(df['t'] >= t0) & (df['t'] <= t1)], color=C_BLUE, ls='--', lw=2, label='Gating Probability (Victim)')
        ax2.set_ylabel('Gating Probability'); ax2.set_ylim(-0.02, 1.05)

    ax1.set_xlabel('Time (minutes)')
    ax1.set_title(title or f"Fold {fold}: {e.get('type','event')}")
    handles, labels = ax1.get_legend_handles_labels()
    if 'ax2' in locals():
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2; labels += l2
    ax1.legend(handles, labels, loc='upper left', frameon=True)

    fig.tight_layout()
    out_png = out_png or os.path.join(outdir, f'fold{fold}_disamb.png')
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"[OK] saved {out_png}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True, help='same --outdir used by training')
    ap.add_argument('--fold',   required=True, type=int, help='1-based fold index')
    ap.add_argument('--events', required=True, help='path to dataset events_by_intent.json')
    ap.add_argument('--png',    default=None)
    ap.add_argument('--title',  default=None)
    args = ap.parse_args()
    plot_fold(args.outdir, args.fold, args.events, out_png=args.png, title=args.title)
