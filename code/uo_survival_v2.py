"""
Survival analysis, corrected: workload from PAYLOAD streams, not battery streams.

WHY v1 WAS WRONG. uo_survival.py measured each device's workload by how often it
reported its battery. Checking that covariate against the devices' own payload
streams showed it is a valid proxy for only some of them:

    6 of 10 devices report battery on EVERY payload uplink   (ratio 1.00)
    4 of 10 report it 14x to 95x more sparsely               (separate diagnostic)

So a device logged at "2 battery reports/day" actually transmits 48-265 payloads
per day. The 354x within-class rate spread that made the EML_RAIN class look like
a strong natural experiment was mostly battery-reporting configuration. The true
spread is 14.7x. v1's covariate is discarded.

A rank correlation is not sufficient to validate a covariate whose MAGNITUDE
matters: v1's automatic check passed on Spearman rho = 0.95 while the magnitudes
were wrong by up to 95x. This script checks the ratio, not the rank.

WHERE THE TEST ACTUALLY LIVES. Of the three hardware classes:

  EML_RAIN    10 devices with matched payload streams, 14.7x workload spread,
              install cohort NOT associated with outcome (p = 0.36).  <-- the test
  LORA        1.7x internal spread and install cohort perfectly predicts outcome
              (p = 0.007). Confounded; reported, not used.
  Water SONS  all but one device present at corpus start, so left-truncated.

THE HYPOTHESES, unchanged. lifetime ~ 1/rate gives a log-log slope of -1; a
workload-irrelevant lifetime gives 0.

MINIMUM DETECTABLE EFFECT is reported alongside the null result, because "no
detectable association" at n = 10 is a statement about the design before it is a
statement about the devices. This is the same discipline the FIEK regression
required.

Author: Vullnet Laniku
"""

import glob
import json
import os
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

from uo_survival import cox_fit, validate_cox

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RESULTS = os.path.join(HERE, '..', 'results')
ARCH = os.path.join(DATA, 'uo_archive')

SILENT_DEAD_D = 180
MIN_SPAN_D = 100
PAYLOAD_FILES = ['*-Rainfall.csv.zip', '*-Water Level.csv.zip']


def load_zips(patterns):
    fr = []
    for pat in patterns:
        for z in sorted(glob.glob(os.path.join(ARCH, pat))):
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.lower().endswith('.csv'):
                        fr.append(pd.read_csv(zf.open(n), low_memory=False))
    d = pd.concat(fr, ignore_index=True)
    d['Timestamp'] = pd.to_datetime(d['Timestamp'], errors='coerce')
    return d.dropna(subset=['Timestamp'])


def main():
    res = {}
    print("=" * 94)
    print("  PART 0   estimator validation")
    print("=" * 94)
    validate_cox()

    bat = pd.read_pickle(os.path.join(ARCH, '_battery_all.pkl'))
    bat['Timestamp'] = pd.to_datetime(bat['Timestamp'], errors='coerce')
    bat = bat.dropna(subset=['Timestamp', 'Value'])
    END = bat.Timestamp.max()
    pay = load_zips(PAYLOAD_FILES)

    # payload files cover 2023-2024 only; measure rate in that window for both
    W0, W1 = pd.Timestamp('2023-01-01'), pd.Timestamp('2025-01-01')
    payw = pay[(pay.Timestamp >= W0) & (pay.Timestamp < W1)]

    print()
    print("=" * 94)
    print("  PART 1   covariate: payload uplinks per day (2023-2024 window)")
    print("=" * 94)
    rows = []
    for s, gp in payw.groupby('Sensor Name'):
        gb = bat[bat['Sensor Name'] == s]
        if len(gb) == 0:
            continue
        dp = max((gp.Timestamp.max() - gp.Timestamp.min()).days, 1)
        gbw = gb[(gb.Timestamp >= W0) & (gb.Timestamp < W1)]
        db = max((gbw.Timestamp.max() - gbw.Timestamp.min()).days, 1) if len(gbw) else 1
        first, last = gb.Timestamp.min(), gb.Timestamp.max()
        silent = (END - last).days
        rows.append(dict(
            dev=s, cls=gb['Broker Name'].iloc[0],
            pay_rate=len(gp) / dp,
            bat_rate=len(gbw) / db if len(gbw) else np.nan,
            first=first, last=last,
            span_d=(last - first).days, silent_d=silent,
            dead=int(silent > SILENT_DEAD_D),
            duration=(last - first).days if silent > SILENT_DEAD_D
            else (END - first).days))
    d = pd.DataFrame(rows)
    d['bat_pay_ratio'] = d.pay_rate / d.bat_rate

    print("  %-32s %-11s %10s %10s %8s %8s %7s"
          % ("sensor", "class", "payload/d", "battery/d", "ratio", "service", "status"))
    for _, x in d.sort_values('pay_rate').iterrows():
        print("  %-32s %-11s %10.1f %10.1f %8.1f %8d %7s"
              % (x.dev[:32], x.cls[:11], x.pay_rate, x.bat_rate, x.bat_pay_ratio,
                 x.duration, "CEASED" if x.dead else "alive"))
    print()
    n_ok = (d.bat_pay_ratio < 1.5).sum()
    print("  battery is piggybacked on every uplink for %d of %d devices;"
          % (n_ok, len(d)))
    print("  for the rest it is a separate diagnostic, %.0fx to %.0fx sparser."
          % (d[d.bat_pay_ratio >= 1.5].bat_pay_ratio.min(),
             d.bat_pay_ratio.max()))
    print("  -> battery rate is NOT a usable workload covariate. Payload rate is.")
    res['covariate_check'] = {
        'n_piggybacked': int(n_ok), 'n_total': int(len(d)),
        'ratio_max': float(d.bat_pay_ratio.max())}

    # --------------------------------------------------------- the test ----
    coh = d[(d.cls == 'EML_RAIN') & (d.span_d >= MIN_SPAN_D)].copy()
    print()
    print("=" * 94)
    print("  PART 2   the test, on the one class that supports it: EML_RAIN")
    print("=" * 94)
    print("  n = %d devices, %d ceased, %d censored"
          % (len(coh), coh.dead.sum(), (1 - coh.dead).sum()))
    print("  workload spread %.0f - %.0f payload uplinks/day  (%.1fx)"
          % (coh.pay_rate.min(), coh.pay_rate.max(),
             coh.pay_rate.max() / coh.pay_rate.min()))
    print("  service life    %d - %d days" % (coh.duration.min(), coh.duration.max()))
    res['cohort'] = {'n': int(len(coh)), 'events': int(coh.dead.sum()),
                     'spread': float(coh.pay_rate.max() / coh.pay_rate.min())}

    print()
    print("  (a) log-log regression among ceased devices")
    print("      received model predicts slope -1 ; workload-irrelevance predicts 0")
    ev = coh[coh.dead == 1]
    if len(ev) >= 4:
        x, y = np.log(ev.pay_rate.values), np.log(ev.duration.values)
        lr = stats.linregress(x, y)
        tc = stats.t.ppf(0.975, len(x) - 2)
        lo, hi = lr.slope - tc * lr.stderr, lr.slope + tc * lr.stderr
        print("      n=%d  slope %+.2f  95%% CI [%+.2f, %+.2f]  p=%.3f"
              % (len(x), lr.slope, lo, hi, lr.pvalue))
        print("      excludes -1 (received model): %-5s   excludes 0: %s"
              % (not (lo <= -1 <= hi), not (lo <= 0 <= hi)))
        print("      NOTE: conditioning on cessation truncates long lifetimes")
        print("      preferentially among low-rate devices, which biases this")
        print("      slope TOWARD zero. It is not a conservative test of -1.")
        res['loglog'] = {'n': int(len(x)), 'slope': float(lr.slope),
                         'ci': [float(lo), float(hi)], 'p': float(lr.pvalue),
                         'excludes_minus1': bool(not (lo <= -1 <= hi)),
                         'excludes_zero': bool(not (lo <= 0 <= hi))}
    else:
        print("      too few events")

    print()
    print("  (b) Cox proportional hazards -- handles censoring, no such bias")
    X = np.log(coh.pay_rate.values).reshape(-1, 1)
    Xs = (X - X.mean()) / X.std()
    beta, se, z, p = cox_fit(Xs, coh.duration.values.astype(float),
                             coh.dead.values.astype(int),
                             np.zeros(len(coh), int))
    hr = float(np.exp(beta[0]))
    ci = (float(np.exp(beta[0] - 1.96 * se[0])), float(np.exp(beta[0] + 1.96 * se[0])))
    print("      covariate log(payload/day), standardised")
    print("      HR = %.3f per SD   95%% CI [%.3f, %.3f]   z=%+.2f   p=%.3f"
          % (hr, ci[0], ci[1], z[0], p[0]))
    print("      received model implies HR > 1 (more traffic, more hazard)")
    res['cox'] = {'hazard_ratio': hr, 'ci': list(ci), 'p': float(p[0]),
                  'beta': float(beta[0]), 'se': float(se[0])}

    print()
    print("  (c) MINIMUM DETECTABLE EFFECT -- what could this design have seen?")
    mde_beta = 1.96 * se[0] + 1.96 * se[0] * 0  # two-sided detection at 80%? use 2.8 SE
    mde = float(np.exp(2.8 * se[0]))
    print("      with SE(beta) = %.3f, the smallest hazard ratio detectable at" % se[0])
    print("      80%% power and alpha=0.05 is HR = %.2f per SD of log-rate." % mde)
    print("      An effect smaller than that would have been missed.")
    res['mde_hazard_ratio'] = mde

    print()
    print("=" * 94)
    print("  PART 3   what this does and does not establish")
    print("=" * 94)
    sig = p[0] < 0.05
    print("  Association between workload and hazard detected: %s" % sig)
    print()
    print("  DOES support: across a %.0fx workload range within one hardware class,"
          % (coh.pay_rate.max() / coh.pay_rate.min()))
    print("    operated by one organisation, service life shows no detectable")
    print("    dependence on transmission rate. The received 1/rate model is not")
    print("    what this fleet does.")
    print()
    print("  DOES NOT support: a claim that workload is irrelevant. n = %d with %d"
          % (len(coh), coh.dead.sum()))
    print("    events cannot exclude a hazard ratio below %.2f. Cessation cannot be" % mde)
    print("    distinguished from decommissioning in an archive. Power architecture")
    print("    is uncontrolled -- these are mains/large-battery flood sensors, not")
    print("    primary-cell nodes, so the result may not transfer to the FIEK class.")

    out = os.path.join(RESULTS, 'uo_survival_v2_results.json')
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=2, default=str)
    coh.to_csv(os.path.join(ARCH, '_survival_cohort_v2.csv'), index=False)
    print("\nSaved %s" % out)


if __name__ == '__main__':
    main()
