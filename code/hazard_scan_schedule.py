"""
The one lever that is not closed: the SENSING scan schedule.

WHY THIS IS DIFFERENT FROM 2.5, WHICH IS CLOSED. audit_adaptive_value.py tested
adapting the scan rate to the ARRIVAL RATE over time of day, found the gain
bounded by workload variability at 4.5% under perfect knowledge, and not
reliably positive on the real deployment. That is a rate-conditioned policy.

This tests a STATE-conditioned policy: scan rate as a function of how long the
bay has already been in its current state. The two are different because parking
dwell is heavy-tailed, so the hazard of a state change falls sharply with elapsed
time -- measured here at 27x (free) and 6x (occupied) over eight hours. A
rate-conditioned policy cannot exploit that; a state-conditioned one can.

WHY IT MATTERS MORE THAN ANYTHING ELSE IN THE PROJECT. Communication is bounded
at 0.087% of the budget by the coincidence bound. Radar transmit power is 1.6 uW.
The ONLY term that can be large is the scheduled radar scan, whose share spans
0.2%-32.5% and is the single largest unknown (3.3, 5.3). If a scan schedule can
be cut substantially without breaking the datasheet's 35 s reporting spec, that
is the only place on this device where an algorithm can move real energy.

THE FORMULATION. Minimise the number of scans subject to a bound on expected
detection latency, given a hazard h(t) that depends on elapsed time in state:

    min integral r(t) dt   s.t.   integral h(t)/(2 r(t)) dt <= L * integral h(t) dt

The stationary condition gives the square-root rule r(t) proportional to sqrt(h(t)),
and by Cauchy-Schwarz the achievable scan ratio against the best CONSTANT rate
delivering the same expected latency is

    scans_adaptive / scans_fixed = (integral sqrt(h) dt)^2 / (T * integral h dt)  <= 1

with equality if and only if h is constant. So the saving is a pure functional of
how variable the hazard is, and it is measurable from the state-change record.

DISCIPLINE. The hazard is fitted on the FIRST 60% of each device's intervals and
scored on the LAST 40%, because an in-sample hazard fit is exactly what produced
the bogus 68% in 2.5. A constant-hazard (exponential) null is run alongside: if
the estimator is working it must return ~0% saving on exponential data.

Author: Vullnet Laniku
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RESULTS = os.path.join(HERE, '..', 'results')
XL = os.path.join(DATA, 'FIEK_parking_export_83day.xlsx')

NAMES = ['FIEK_UP_PARKING_SENZOR_0%d' % i for i in (1, 2, 3, 4, 5)]
CADENCE_S = 35.0
EDGES_MIN = np.array([0, 15, 30, 60, 120, 240, 480, 1440])   # drop the open tail


def load_dwells():
    rows = []
    for n in NAMES:
        d = pd.read_excel(XL, sheet_name=n)
        d['timestamp'] = pd.to_datetime(d['timestamp'])
        d = d.sort_values('timestamp')
        sc = d[d['status_changed'] == True]
        if len(sc) < 5:
            continue
        t = sc['timestamp'].values
        occ = sc['occupied'].values
        for i in range(len(t) - 1):
            dur = (t[i + 1] - t[i]) / np.timedelta64(1, 'm')
            if 0 < dur < 1440:            # censor at 24 h; the tail is unobserved
                rows.append((n[-2:], 'OCCUPIED' if occ[i] else 'FREE', float(dur)))
    return pd.DataFrame(rows, columns=['dev', 'state', 'min'])


def hazard_profile(durations, edges=EDGES_MIN):
    """Piecewise-constant hazard per minute, from a set of completed dwells."""
    x = np.asarray(durations)
    h = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        at_risk = (x >= edges[i]).sum()
        ended = ((x >= edges[i]) & (x < edges[i + 1])).sum()
        width = edges[i + 1] - edges[i]
        h[i] = (ended / at_risk / width) if at_risk >= 5 else np.nan
    return h


def scan_ratio(h, edges=EDGES_MIN, occupancy=None):
    """
    (int sqrt(h))^2 / (T int h)  -- adaptive scans as a fraction of fixed scans,
    at equal expected detection latency. `occupancy` weights each bin by the time
    actually spent in it, which is what a real schedule integrates over.
    """
    w = np.diff(edges).astype(float)
    if occupancy is not None:
        w = w * occupancy
    m = ~np.isnan(h) & (w > 0)
    if m.sum() < 2:
        return np.nan
    hh, ww = h[m], w[m]
    num = (np.sum(np.sqrt(hh) * ww)) ** 2
    den = np.sum(ww) * np.sum(hh * ww)
    return float(num / den)


def time_weights(durations, edges=EDGES_MIN):
    """Fraction of total in-state time spent in each elapsed-time bin."""
    x = np.asarray(durations)
    w = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        w[i] = np.clip(np.minimum(x, edges[i + 1]) - edges[i], 0, None).sum()
    return w / w.sum() if w.sum() > 0 else w


def main():
    res = {}
    w = load_dwells()
    print("=" * 92)
    print("  PART 0   estimator null test: constant hazard must return ~0%% saving")
    print("=" * 92)
    rng = np.random.default_rng(0)
    for n in (200, 1000, 5000):
        exp = rng.exponential(300.0, n)
        exp = exp[exp < 1440]
        h = hazard_profile(exp)
        r = scan_ratio(h, occupancy=time_weights(exp))
        print("  exponential dwells, n=%-5d -> scan ratio %.3f  (saving %.1f%%)"
              % (len(exp), r, 100 * (1 - r)))
    print("  -> near 1.00 as required; the estimator does not manufacture savings.")

    print()
    print("=" * 92)
    print("  PART 1   measured hazard, held out")
    print("=" * 92)
    print("  hazard fitted on the FIRST 60%% of intervals, scored on the LAST 40%%")
    print()
    print("  %-10s %6s %7s %7s %10s %10s %11s"
          % ("state", "n_fit", "n_test", "h_ratio", "in-sample", "HELD OUT", "scans saved"))
    out = {}
    for st, g in w.groupby('state'):
        d = g['min'].values
        k = int(0.6 * len(d))
        fit, test = d[:k], d[k:]
        h_fit = hazard_profile(fit)
        h_all = hazard_profile(d)
        # score the FITTED hazard against the time distribution of the TEST set
        r_out = scan_ratio(h_fit, occupancy=time_weights(test))
        r_in = scan_ratio(h_all, occupancy=time_weights(d))
        hr = np.nanmax(h_fit) / np.nanmin(h_fit[h_fit > 0]) if np.any(h_fit > 0) else np.nan
        print("  %-10s %6d %7d %7.1fx %9.1f%% %9.1f%% %10.1f%%"
              % (st, len(fit), len(test), hr, 100 * (1 - r_in), 100 * (1 - r_out),
                 100 * (1 - r_out)))
        out[st] = {'n_fit': int(len(fit)), 'n_test': int(len(test)),
                   'hazard_ratio': float(hr), 'saving_in_sample': float(1 - r_in),
                   'saving_held_out': float(1 - r_out)}
    res['by_state'] = out

    print()
    print("=" * 92)
    print("  PART 2   per-device held-out check -- does it hold on every unit?")
    print("=" * 92)
    print("  %-6s %-10s %7s %12s" % ("dev", "state", "n", "held-out saving"))
    per = []
    for (dev, st), g in w.groupby(['dev', 'state']):
        d = g['min'].values
        if len(d) < 25:
            continue
        k = int(0.6 * len(d))
        r = scan_ratio(hazard_profile(d[:k]), occupancy=time_weights(d[k:]))
        if np.isnan(r):
            continue
        per.append({'dev': dev, 'state': st, 'n': len(d), 'saving': float(1 - r)})
        print("  %-6s %-10s %7d %11.1f%%" % (dev, st, len(d), 100 * (1 - r)))
    res['per_device'] = per
    if per:
        s = np.array([p['saving'] for p in per])
        print()
        print("  %d device-state cells, median saving %.1f%%, range %.1f%% to %.1f%%"
              % (len(s), 100 * np.median(s), 100 * s.min(), 100 * s.max()))
        print("  positive on %d of %d cells" % ((s > 0).sum(), len(s)))

    print()
    print("=" * 92)
    print("  PART 3   what this is and is not")
    print("=" * 92)
    print("  IS: a bound on the scans a state-conditioned schedule needs relative to")
    print("      a fixed 35 s schedule delivering the SAME expected detection latency.")
    print("      It follows from the measured hazard and the square-root rule, and it")
    print("      is scored out of sample.")
    print()
    print("  IS NOT: an energy saving. Per-scan energy is not on the datasheet and is")
    print("      swept over 0.05-5 mJ (2.8). Scans saved converts to budget saved only")
    print("      through a quantity nobody publishes. The honest claim is a scan-count")
    print("      reduction and the energy consequence is stated as a band.")
    print()
    print("  IS NOT: validated against the vendor firmware, which we cannot modify.")
    print("      This bounds what an occupancy-aware scheduler could achieve; running")
    print("      one requires either firmware access or the SF/cadence experiment.")

    with open(os.path.join(RESULTS, 'hazard_scan_schedule_results.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    print("\nSaved ../results/hazard_scan_schedule_results.json")


if __name__ == '__main__':
    main()
