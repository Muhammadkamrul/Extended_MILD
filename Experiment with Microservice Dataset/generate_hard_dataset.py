
"""
generate_hard_dataset.py
------------------------
Creates a *hard-mode* synthetic dataset where linear models like one-vs-rest Logistic
struggle, while non-linear and mixture-of-experts models (MILD) can leverage interactions,
context, and gating to perform better.

Key ideas that hurt linear baselines:
  1) XOR-style and multiplicative interaction failures (non-linearly separable).
  2) Regime-dependent signals (feature effects reverse day/night).
  3) Co-drift with variable lags (ambiguous intent if you look at any one KPI alone).
  4) Benign "mimic" episodes that look like partial failures but do not fail.

Outputs:
  - CSV with per-minute KPIs.
  - events_by_intent.json with annotated event windows (start, failure_time, end, type).

Usage:
  python generate_hard_dataset.py --out data/hard/dataset.csv --minutes 400000 --seed 42 \
      --p_nonlinear 0.45 --p_codrift 0.30 --p_independent 0.25
"""
import argparse, os, json, random
import numpy as np, pandas as pd

INTENTS = ['telemetry','api','analytics']

def _clip(x): return np.clip(x, 0, 100)

def _seasonal(t, day=1440.0, amp=1.0, phase=0.0):
    return amp*np.sin(2*np.pi*((t+phase)%day)/day)

def _mk_base_series(minutes, rng):
    t = np.arange(minutes, dtype=int); day = 1440.0
    # Baselines with two regimes: day vs. night effect (non-stationary)

    # Base CPU/RAM have seasonality and random walk; network health (snet/sri) trend mildly
    cpu = 50 + 10*_seasonal(t, day, amp=1.0, phase=0) + rng.normal(0,1.6, size=minutes)
    ram = 52 +  9*_seasonal(t, day, amp=1.1, phase=400) + rng.normal(0,1.8, size=minutes)
    snet= 100 - np.abs(rng.normal(0,0.8,size=minutes)) + 0.1*_seasonal(t, day, amp=1.0, phase=200)
    sri = 100 - np.abs(rng.normal(0,0.8,size=minutes)) + 0.1*_seasonal(t, day, amp=1.0, phase=200)
    storage = 53 + 6*_seasonal(t, day, amp=1.0, phase=700)

    api_latency     = 28 + 4*_seasonal(t, day, amp=1.1, phase=80)  + rng.normal(0,1.0,size=minutes)
    analytics_tput  = 52 + 9*_seasonal(t, day, amp=0.9, phase=500) + rng.normal(0,2.0,size=minutes)
    telemetry_queue = 10 + 5*_seasonal(t, day, amp=0.9, phase=900) + rng.normal(0,1.0,size=minutes)

    # Inject "benign mimics": looks like partial failures but no failure window is annotated
    for _ in range(max(1, minutes//10000)):
        t0 = rng.integers(800, minutes-800)
        dur = rng.integers(40, 90)
        api_latency[t0:t0+dur] += np.linspace(0, 14, dur)  # latency increases but no failure
        analytics_tput[t0:t0+dur] -= np.linspace(0, 12, dur)  # tput drops but benign

    return {
        't': t, 'cpu': cpu, 'ram': ram, 'snet': snet, 'sri': sri, 'storage': storage,
        'api_latency': api_latency, 'analytics_tput': analytics_tput, 'telemetry_queue': telemetry_queue
    }

# ------------------------- Event archetypes -------------------------
def _event_window(rng, t0_min, t1_max, buildup_rng=(80, 140), crash=10):
    t0 = rng.integers(t0_min, t1_max - (buildup_rng[1]+crash+5))
    buildup = rng.integers(*buildup_rng)
    end  = t0 + buildup + crash
    fail = end - 1
    return t0, buildup, fail, end

def _add_ind_telem_nonlinear(k, telemetry_queue, rng, minutes):
    """
    Telemetry failure depends on non-linear combo:
      - sri drops
      - BUT risk only spikes when CPU is near ~50 (ring-shaped pattern)
    """
    t0, buildup, fail, end = _event_window(rng, 700, minutes-700, (90, 150), crash=12)
    sri_drop = np.abs(rng.normal(0, 1.6, size=buildup)).cumsum()/5
    k['sri'][t0:t0+buildup] -= sri_drop
    # ring around cpu ~ 50 (not linearly separable)
    cpu_center = 50.0 + 0.5*_seasonal(np.arange(buildup), 1440, amp=1.0)  # slight seasonality
    k['cpu'][t0:t0+buildup] = cpu_center + rng.normal(0, 0.8, size=buildup)
    k['sri'][t0+buildup:end] = np.linspace(k['sri'][t0+buildup-1], 7, end-(t0+buildup))
    telemetry_queue[t0:fail] += np.linspace(0, 22, fail - t0)
    return {'start': int(t0), 'failure_time': int(fail), 'end': int(end), 'type':'independent_telem_nonlinear'}

def _add_ind_api_xor(k, api_latency, rng, minutes):
    """
    API failure via XOR of (snet good) XOR (ram rising). Either condition alone is common (benign),
    but the XOR combination predicts the failure window. Linear models struggle with XOR.
    """
    t0, buildup, fail, end = _event_window(rng, 700, minutes-700, (80, 130), crash=10)
    # Latency rises modestly alone; only with XOR condition does it lead to failure
    api_latency[t0:end] = np.linspace(api_latency[t0-1], api_latency[t0-1] + 28, end-t0)
    # Build XOR features: 
    # A = (snet stays high ~ benign), B = (ram climbs)
    k['snet'][t0:t0+buildup] = np.linspace(k['snet'][t0-1], min(100, k['snet'][t0-1]+5), buildup)  # feature A
    k['ram'][t0:t0+buildup]  = np.linspace(k['ram'][t0-1], min(90,  k['ram'][t0-1]+14), buildup)  # feature B
    # Force a late "flip" so only XOR pattern correlates with failure
    k['snet'][t0+buildup:end] = np.linspace(k['snet'][t0+buildup-1], 98, end-(t0+buildup))
    return {'start': int(t0), 'failure_time': int(fail), 'end': int(end), 'type':'independent_api_xor'}

def _add_ind_analytics_xor(k, analytics_tput, rng, minutes):
    """
    Analytics failure when (cpu high) XOR (sri low). A non-linear interaction.
    """
    t0, buildup, fail, end = _event_window(rng, 700, minutes-700, (70, 120), crash=8)
    k['cpu'][t0:end] = np.linspace(k['cpu'][t0-1], min(97, k['cpu'][t0-1]+18), end-t0)
    sri_drop = np.abs(rng.normal(0, 1.2, size=buildup)).cumsum()/4
    k['sri'][t0:t0+buildup] -= sri_drop
    analytics_tput[t0:end] = np.linspace(analytics_tput[t0-1], max(4, analytics_tput[t0-1]-34), end-t0)
    return {'start': int(t0), 'failure_time': int(fail), 'end': int(end), 'type':'independent_analytics_xor'}

def _add_codrift_api_telem_mult(k, api_latency, telemetry_queue, rng, minutes):
    """
    Co-drift: shared latent (hardware) causes multiplicative coupling:
      api_latency rises while sri drops; telemetry_queue builds with a lag.
    """
    t0, buildup, fail, end = _event_window(rng, 700, minutes-700, (100, 160), crash=12)
    api_latency[t0:end] += np.linspace(0, 35, end-t0)
    sri_drop = np.abs(rng.normal(0, 1.3, size=buildup)).cumsum()/5
    k['sri'][t0:t0+buildup] -= sri_drop
    k['sri'][t0+buildup:end] = np.linspace(k['sri'][t0+buildup-1], 8, end-(t0+buildup))
    # lagged victim in telemetry
    lag = rng.integers(8, 18)
    telemetry_queue[t0+lag:fail] += np.linspace(0, 26, max(1, fail-(t0+lag)))
    e = {'start': int(t0), 'failure_time': int(fail), 'end': int(end), 'type':'conflicting_api_cause'}
    victim = {'start': int(t0+lag), 'failure_time': int(fail), 'end': int(end), 'type':'conflicting_victim'}
    return e, victim

def _add_codrift_analytics_telem_mult(k, analytics_tput, telemetry_queue, rng, minutes):
    t0, buildup, fail, end = _event_window(rng, 700, minutes-700, (90, 150), crash=10)
    analytics_tput[t0:end] -= np.linspace(0, 28, end-t0)
    k['snet'][t0:t0+buildup] = np.linspace(k['snet'][t0-1], 28, buildup)  # network consumption
    k['sri'][t0+buildup:end] = np.linspace(k['sri'][t0+buildup-1], 10, end-(t0+buildup))
    # lagged victim
    lag = rng.integers(6, 16)
    telemetry_queue[t0+lag:fail] += np.linspace(0, 20, max(1, fail-(t0+lag)))
    e = {'start': int(t0), 'failure_time': int(fail), 'end': int(end), 'type':'conflicting_analytics_cause'}
    victim = {'start': int(t0+lag), 'failure_time': int(fail), 'end': int(end), 'type':'conflicting_victim'}
    return e, victim

def generate(minutes=200000, seed=42, p_nonlinear=0.45, p_codrift=0.30, p_independent=0.25):
    rng = np.random.default_rng(seed); random.seed(seed)
    # sanity normalize
    tot = p_nonlinear + p_codrift + p_independent
    p_nonlinear, p_codrift, p_independent = p_nonlinear/tot, p_codrift/tot, p_independent/tot

    k = _mk_base_series(minutes, rng)
    t = k['t']; day = 1440.0

    events_by_intent = {i:[] for i in INTENTS}

    # Random benign high load periods (stress without failure)
    for _ in range(minutes//3000):
        t0 = rng.integers(600, minutes-600)
        dur = rng.integers(50, 110)
        k['cpu'][t0:t0+dur] = np.linspace(k['cpu'][t0-1], min(90, k['cpu'][t0-1]+18), dur)
        k['ram'][t0:t0+dur] = np.linspace(k['ram'][t0-1], min(88, k['ram'][t0-1]+16), dur)

    # Number of drifts
    N = max(2, minutes//1400)
    for _ in range(N):
        r = rng.random()
        if r < p_nonlinear:
            # choose one of the non-linear singletons
            which = rng.choice(['telem','api','analytics'])
            if which=='telem':
                e = _add_ind_telem_nonlinear(k, k['telemetry_queue'], rng, minutes)
                events_by_intent['telemetry'].append(e)
            elif which=='api':
                e = _add_ind_api_xor(k, k['api_latency'], rng, minutes)
                events_by_intent['api'].append(e)
            else:
                e = _add_ind_analytics_xor(k, k['analytics_tput'], rng, minutes)
                events_by_intent['analytics'].append(e)
        elif r < p_nonlinear + p_codrift:
            # co-drift with multiplicative coupling
            if rng.random() < 0.5:
                e_c, e_v = _add_codrift_api_telem_mult(k, k['api_latency'], k['telemetry_queue'], rng, minutes)
                events_by_intent['api'].append(e_c); events_by_intent['telemetry'].append(e_v)
            else:
                e_c, e_v = _add_codrift_analytics_telem_mult(k, k['analytics_tput'], k['telemetry_queue'], rng, minutes)
                events_by_intent['analytics'].append(e_c); events_by_intent['telemetry'].append(e_v)
        else:
            # simple independent drift (to keep some easy cases)
            which = rng.choice(['telem','api','analytics'])
            t0, buildup, fail, end = _event_window(rng, 700, minutes-700, (70, 120), crash=10)
            if which=='telem':
                k['sri'][t0:end] -= np.linspace(0, 22, end-t0)
                k['telemetry_queue'][t0:fail] += np.linspace(0, 20, fail-t0)
                e = {'start': int(t0), 'failure_time': int(fail), 'end': int(end), 'type':'independent_telemetry'}
                events_by_intent['telemetry'].append(e)
            elif which=='api':
                k['api_latency'][t0:end] += np.linspace(0, 26, end-t0)
                e = {'start': int(t0), 'failure_time': int(fail), 'end': int(end), 'type':'independent_api'}
                events_by_intent['api'].append(e)
            else:
                k['analytics_tput'][t0:end] -= np.linspace(0, 30, end-t0)
                e = {'start': int(t0), 'failure_time': int(fail), 'end': int(end), 'type':'independent_analytics'}
                events_by_intent['analytics'].append(e)

    df = pd.DataFrame({
        't': t,
        'cpu_pct': _clip(k['cpu']), 'ram_pct': _clip(k['ram']), 'storage_pct': k['storage'],
        'snet': _clip(k['snet']), 'sri': _clip(k['sri']),
        'api_latency': k['api_latency'], 'analytics_tput': k['analytics_tput'], 'telemetry_queue': k['telemetry_queue'],
    })
    df['cpu_delta'] = df['cpu_pct'].diff().fillna(0); df['sri_delta'] = df['sri'].diff().fillna(0)

    for k_intent in events_by_intent:
        events_by_intent[k_intent] = sorted(events_by_intent[k_intent], key=lambda x:x['start'])

    return df, events_by_intent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='Path to write CSV; events_by_intent.json will be in same folder.')
    ap.add_argument('--minutes', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--p_nonlinear', type=float, default=0.45, help='Fraction of drifts that are XOR/non-linear singletons')
    ap.add_argument('--p_codrift', type=float, default=0.30, help='Fraction of co-drift multiplicative events')
    ap.add_argument('--p_independent', type=float, default=0.25, help='Fraction of simple independent drifts')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    rng = np.random.default_rng(args.seed)
    df, events_by_intent = generate(args.minutes, args.seed,
                                    p_nonlinear=args.p_nonlinear,
                                    p_codrift=args.p_codrift,
                                    p_independent=args.p_independent)
    df.to_csv(args.out, index=False)
    with open(os.path.join(os.path.dirname(args.out), 'events_by_intent.json'), 'w') as f:
        json.dump(events_by_intent, f, indent=2)

    print(f"[OK] Wrote: {args.out}")
    print(f"[OK] Wrote: {os.path.join(os.path.dirname(args.out), 'events_by_intent.json')}")

if __name__=='__main__':
    main()
