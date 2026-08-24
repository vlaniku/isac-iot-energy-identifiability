"""
Does the identifiability criterion predict what it claims? Test it by injection.

WHY THIS EXISTS. Sec. VII derives a closed form for the smallest energy change a
campaign can resolve. Everything downstream rests on it, and until now it had
been checked only against a Monte-Carlo that shared its assumptions. The prior
work on this same deployment faced the same problem from the other side -- no
labelled attacks existed -- and solved it by injecting synthetic anomalies into
real event data at a known rate and measuring what the detectors recovered. The
same move works here.

THE DESIGN. Inject a known fractional change in depletion rate into real
telemetry and ask whether the criterion predicted it would be detected.

  1. Fit each device's daily battery series, keep the RESIDUALS. They carry
     whatever the real noise is: the serial correlation measured in
     noise_autocorrelation.py, the quantisation staircase, any non-normality,
     and any structure nobody has thought to look for.
  2. Build a synthetic crossover from those residuals. A block is a contiguous
     window of one device's own residuals plus a linear ramp; the high arm's
     ramp is steeper by a known epsilon. Contiguous windows are used rather
     than resampled points precisely so the serial correlation survives.
  3. Run the analysis the paper proposes: pairs within device, device means,
     a one-sample t-test across devices.
  4. Sweep epsilon, find where empirical power reaches 80%, and compare that
     against what the closed form predicted.

WHAT WOULD FALSIFY THE CRITERION. If the injected epsilon needed for 80% power
is materially larger than eps_min predicts, the criterion is optimistic and
every campaign figure in the paper is too short. This is the check that can
say so.

Author: Vullnet Laniku
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.path.join(HERE, '..', 'results')
EXPORT = os.path.join(HERE, '..', 'data', 'FIEK_parking_export_83day.xlsx')

from identifiability import eps_min                      # noqa: E402

V_QUANT = 0.01          # observed quantisation of battery_v, volts
BLOCK_D = 25            # days per block; the daily series supports 29-46
N_PAIRS = 2             # per device, as the design specifies
N_SIM = 1200
SEED = 20260823
SIGMA_BYTE = {'01': 0.714, '03': 0.436, '05': 0.301}      # the three live units


def daily_residuals():
    ev = pd.read_excel(EXPORT, sheet_name='All Events')
    ev['ts'] = pd.to_datetime(ev['timestamp_local'])
    ev = ev.dropna(subset=['battery_v']).sort_values('ts')
    out = {}
    for d, g in ev.groupby('device_name'):
        key = d[-2:]
        if key not in SIGMA_BYTE:
            continue
        dd = g.set_index('ts').battery_v.resample('1D').mean().dropna()
        if len(dd) < BLOCK_D + 4:
            continue
        t = (dd.index - dd.index[0]).total_seconds().values / 86400.0
        v = dd.values.astype(float)
        lr = stats.linregress(t, v)
        out[key] = {'resid': v - (lr.intercept + lr.slope * t),
                    'slope': float(lr.slope), 'n': len(v)}
    return out


def one_block(res, slope, eps, rng):
    """A block of real residuals plus a ramp, quantised as the device reports."""
    r = res['resid']
    i = rng.integers(0, len(r) - BLOCK_D)
    t = np.arange(BLOCK_D, dtype=float)
    v = 3.0 + slope * t + r[i:i + BLOCK_D]
    return stats.linregress(t, np.round(v / V_QUANT) * V_QUANT).slope


def power_at(eps, devs, rng, n_sim=N_SIM, alpha=0.05):
    hits = 0
    for _ in range(n_sim):
        means = []
        for res in devs.values():
            s0 = res['slope']
            s1 = s0 * (1.0 + eps)                # the injected effect
            diffs = [abs(one_block(res, s1, eps, rng))
                     - abs(one_block(res, s0, eps, rng))
                     for _ in range(N_PAIRS)]
            means.append(np.mean(diffs))
        t, p = stats.ttest_1samp(means, 0.0)
        if p < alpha and np.mean(means) > 0:
            hits += 1
    return hits / n_sim


def main():
    rng = np.random.default_rng(SEED)
    devs = daily_residuals()
    sig_rms = float(np.sqrt(np.mean([SIGMA_BYTE[k] ** 2 for k in devs])))
    predicted = eps_min(len(devs) * N_PAIRS, BLOCK_D, sig_rms,
                        n_devices=len(devs))

    print("=" * 92)
    print("  INJECTION TEST OF THE IDENTIFIABILITY CRITERION")
    print("=" * 92)
    print("  %d devices, %d pairs each, %d-day blocks, real residual windows"
          % (len(devs), N_PAIRS, BLOCK_D))
    print("  devices: %s" % ", ".join("%s (n=%d daily)" % (k, v['n'])
                                      for k, v in sorted(devs.items())))
    print("  sigma RMS = %.3f byte  ->  criterion predicts eps_min = %.2f%%"
          % (sig_rms, 100 * predicted))
    print()
    print("  %-14s %10s" % ("injected eps", "power"))
    grid = sorted({round(x, 4) for x in
                   np.concatenate([np.linspace(0.2 * predicted, 2.4 * predicted, 10),
                                   [predicted]])})
    curve = []
    for e in grid:
        pw = power_at(e, devs, rng)
        curve.append({'eps': float(e), 'power': pw})
        mark = '   <- criterion' if abs(e - predicted) < 1e-9 else ''
        print("  %-13.3f%% %9.2f%s" % (100 * e, pw, mark))

    xs = [c['eps'] for c in curve]
    ys = [c['power'] for c in curve]
    emp = float('nan')
    for a, b in zip(curve, curve[1:]):
        if a['power'] < 0.80 <= b['power']:
            f = (0.80 - a['power']) / (b['power'] - a['power'])
            emp = a['eps'] + f * (b['eps'] - a['eps'])
            break

    print()
    print("=" * 92)
    print("  READING")
    print("=" * 92)
    print("  criterion predicted 80%% power at eps = %.3f%%" % (100 * predicted))
    if np.isfinite(emp):
        rel = (emp - predicted) / predicted
        print("  injection reaches 80%% power at eps = %.3f%%" % (100 * emp))
        print("  discrepancy %+.0f%%  ->  %s"
              % (100 * rel,
                 "criterion CONFIRMED on real noise" if abs(rel) < 0.25 else
                 "criterion is OPTIMISTIC; campaigns are too short"))
    else:
        print("  80% power not bracketed by the sweep; widen the grid")

    out = {'_method': {
        'block_days': BLOCK_D, 'n_pairs': N_PAIRS, 'n_sim': N_SIM,
        'seed': SEED, 'devices': sorted(devs),
        'note': ('Residual windows are contiguous, so the real serial '
                 'correlation, quantisation and any non-normality are carried '
                 'into the test rather than modelled.')},
        'sigma_rms_byte': sig_rms,
        'eps_predicted': predicted,
        'eps_empirical_80pct': emp,
        'relative_discrepancy': (emp - predicted) / predicted if np.isfinite(emp) else None,
        'power_curve': curve}
    path = os.path.join(RESULTS, 'identifiability_injection.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print()
    print("Saved %s" % path)


if __name__ == '__main__':
    main()
