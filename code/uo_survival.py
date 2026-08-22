"""
Does reporting workload predict service life in a deployed IoT sensor fleet?

THE CLAIM UNDER TEST. The IoT/WSN energy literature models lifetime as set by
sensing and transmission workload -- "the more they sense and transmit, the more
the energy is depleted" -- and the entire adaptive-allocation literature
optimises that term. FIEK says otherwise at n=5, and the coincidence bound
(coincidence_bound.py) says so very sharply at n=2. Neither can carry the claim.

This tests it on the Newcastle Urban Observatory archive: 38 devices, three
independent hardware populations, six years, with real cessations.

THE SHARP FORM OF THE TEST. If lifetime is workload-driven, lifetime ~ 1/rate,
so a regression of log(service life) on log(reports per day) has slope -1. If
workload is irrelevant, the slope is 0. The two hypotheses make opposite
point predictions and the data can separate them.

DESIGN DECISIONS, and why.

  Left truncation. Devices whose first record coincides with the corpus start
  were already installed and their true age is unknown. They are EXCLUDED, not
  treated as age zero. This removes most of the Water SONS class and is the
  single biggest cost of using an archive rather than a commissioning record.

  Workload measured EARLY, not over the whole span. A device that dies early has
  fewer days to average over, and terminal behaviour would contaminate the rate.
  Rate is taken over each device's first 90 days only, so the covariate cannot
  be contaminated by the outcome.

  Total lifetime transmissions is NOT used as a predictor. It is rate x span and
  span is the outcome, so it would be circular.

  Cessation is silence > 180 d at corpus end. This CANNOT be distinguished from
  decommissioning, and that limitation is fatal to a causal reading. It is
  reported as such rather than argued away.

  Class is a stratum, never a pooled covariate. Reporting rate differs ~8x
  between classes (720 / 144 / 96 per day) and so does hardware, so a pooled
  regression would measure the class, not the workload.

The Cox estimator is implemented here (no lifelines available) and is validated
against synthetic data with a known hazard ratio before being run on the fleet.

Author: Vullnet Laniku
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RESULTS = os.path.join(HERE, '..', 'results')
PKL = os.path.join(DATA, 'uo_archive', '_battery_all.pkl')

SILENT_DEAD_D = 180          # silence at corpus end that counts as cessation
MIN_SPAN_D = 100
MIN_SAMPLES = 500
EARLY_WINDOW_D = 90          # window for the workload covariate
TRUNC_MARGIN_D = 30          # first-seen must exceed corpus start by this much


# ------------------------------------------------------------------ Cox ----
def cox_neg_loglik(beta, X, T, E, strata):
    """Stratified Cox partial likelihood, Breslow ties. Returns -loglik."""
    beta = np.atleast_1d(beta)
    total = 0.0
    for s in np.unique(strata):
        m = strata == s
        x, t, e = X[m], T[m], E[m]
        if e.sum() == 0:
            continue
        eta = x @ beta
        for i in np.where(e == 1)[0]:
            risk = t >= t[i]                     # at risk at this event time
            total += eta[i] - np.log(np.exp(eta[risk]).sum())
    return -total


def cox_fit(X, T, E, strata):
    X = np.asarray(X, float).reshape(len(T), -1)
    p = X.shape[1]
    r = optimize.minimize(cox_neg_loglik, np.zeros(p), args=(X, T, E, strata),
                          method='BFGS')
    beta = r.x
    # observed information by central differences
    h, n = 1e-4, p
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            bpp, bpm, bmp, bmm = (beta.copy() for _ in range(4))
            bpp[i] += h; bpp[j] += h
            bpm[i] += h; bpm[j] -= h
            bmp[i] -= h; bmp[j] += h
            bmm[i] -= h; bmm[j] -= h
            H[i, j] = ((cox_neg_loglik(bpp, X, T, E, strata)
                        - cox_neg_loglik(bpm, X, T, E, strata)
                        - cox_neg_loglik(bmp, X, T, E, strata)
                        + cox_neg_loglik(bmm, X, T, E, strata)) / (4 * h * h))
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    z = beta / se
    pval = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return beta, se, z, pval


def validate_cox():
    """Recover a known hazard ratio from synthetic data before trusting the fit."""
    rng = np.random.default_rng(0)
    n, true_beta = 400, 0.8
    x = rng.normal(size=n)
    t = rng.exponential(1.0 / np.exp(true_beta * x))
    c = rng.exponential(2.0, n)
    T, E = np.minimum(t, c), (t <= c).astype(int)
    b, se, z, p = cox_fit(x, T, E, np.zeros(n, int))
    ok = abs(b[0] - true_beta) < 3 * se[0]
    print("  estimator validation on synthetic data (n=400, %d events):"
          % int(E.sum()))
    print("    true beta %.3f   estimated %.3f +/- %.3f   within 3 SE: %s"
          % (true_beta, b[0], se[0], ok))
    if not ok:
        raise SystemExit("Cox estimator failed validation -- do not trust output")
    return ok


# ------------------------------------------------------------- cohort ----
def build_cohort():
    b = pd.read_pickle(PKL)
    b['Timestamp'] = pd.to_datetime(b['Timestamp'], errors='coerce')
    b = b.dropna(subset=['Timestamp', 'Value'])
    start, end = b.Timestamp.min(), b.Timestamp.max()

    rows = []
    for s, g in b.groupby('Sensor Name'):
        g = g.sort_values('Timestamp')
        first, last = g.Timestamp.min(), g.Timestamp.max()
        span = (last - first).days
        silent = (end - last).days
        early = g[g.Timestamp < first + pd.Timedelta(days=EARLY_WINDOW_D)]
        early_days = max((early.Timestamp.max() - first).days, 1)
        rows.append(dict(
            dev=s, cls=g['Broker Name'].iloc[0], n=len(g),
            first=first, last=last, span_d=span, silent_d=silent,
            rate=len(early) / early_days,
            truncated=(first - start).days <= TRUNC_MARGIN_D,
            dead=int(silent > SILENT_DEAD_D),
            duration=span if silent > SILENT_DEAD_D else (end - first).days))
    return pd.DataFrame(rows), start, end


def main():
    res = {}
    print("=" * 94)
    print("  PART 0   estimator validation")
    print("=" * 94)
    validate_cox()

    raw, start, end = build_cohort()
    print()
    print("=" * 94)
    print("  PART 1   cohort construction  (corpus %s -> %s)"
          % (start.date(), end.date()))
    print("=" * 94)
    print("  all devices in archive                     %3d" % len(raw))
    d = raw[~raw.truncated]
    print("  drop left-truncated (present at corpus start) -%3d  -> %3d"
          % ((raw.truncated).sum(), len(d)))
    d2 = d[(d.span_d >= MIN_SPAN_D) & (d.n >= MIN_SAMPLES)]
    print("  drop span < %d d or samples < %d           -%3d  -> %3d"
          % (MIN_SPAN_D, MIN_SAMPLES, len(d) - len(d2), len(d2)))
    d = d2.copy()
    print()
    print("  analysis cohort: %d devices, %d cessations, %d censored"
          % (len(d), d.dead.sum(), (1 - d.dead).sum()))
    print("  by class:")
    for c, g in d.groupby('cls'):
        print("    %-11s n=%2d  ceased %d  censored %d  rate %.0f-%.0f/day  "
              "service %d-%d d"
              % (c, len(g), g.dead.sum(), (1 - g.dead).sum(),
                 g.rate.min(), g.rate.max(), g.duration.min(), g.duration.max()))
    res['cohort'] = {'n': int(len(d)), 'events': int(d.dead.sum()),
                     'excluded_truncated': int(raw.truncated.sum())}

    if d.dead.sum() < 3:
        print("\n  too few events. Stopping.")
        return

    print()
    print("=" * 94)
    print("  PART 2   the sharp test: log(service life) vs log(reports/day)")
    print("=" * 94)
    print("  received model predicts slope = -1 ; workload-irrelevance predicts 0")
    print()
    ev = d[d.dead == 1]
    for lab, sub in [("all ceased devices, pooled", ev)] + \
                    [("ceased, %s only" % c, g) for c, g in ev.groupby('cls')
                     if len(g) >= 4]:
        x, y = np.log(sub.rate.values), np.log(sub.duration.values)
        if len(x) < 4 or np.ptp(x) == 0:
            print("  %-32s n=%d  insufficient spread" % (lab, len(x)))
            continue
        lr = stats.linregress(x, y)
        tcrit = stats.t.ppf(0.975, len(x) - 2)
        lo, hi = lr.slope - tcrit * lr.stderr, lr.slope + tcrit * lr.stderr
        rej_recv = not (lo <= -1.0 <= hi)
        rej_null = not (lo <= 0.0 <= hi)
        print("  %-32s n=%2d  slope %+.2f  95%% CI [%+.2f, %+.2f]"
              % (lab, len(x), lr.slope, lo, hi))
        print("  %-32s       excludes -1 (received model): %-5s   "
              "excludes 0: %s" % ("", rej_recv, rej_null))
        res['loglog_' + lab.replace(' ', '_')] = {
            'n': int(len(x)), 'slope': float(lr.slope), 'ci': [float(lo), float(hi)],
            'p': float(lr.pvalue), 'excludes_minus1': bool(rej_recv),
            'excludes_zero': bool(rej_null)}

    print()
    print("=" * 94)
    print("  PART 3   Cox proportional hazards, stratified by hardware class")
    print("=" * 94)
    X = np.log(d.rate.values).reshape(-1, 1)
    X = (X - X.mean()) / X.std()
    strata = pd.Categorical(d.cls).codes
    beta, se, z, p = cox_fit(X, d.duration.values.astype(float),
                             d.dead.values.astype(int), strata)
    hr = np.exp(beta[0])
    print("  covariate: log(reports per day), standardised")
    print("  beta = %+.3f  (SE %.3f)   hazard ratio %.3f per SD   z = %+.2f   p = %.3f"
          % (beta[0], se[0], hr, z[0], p[0]))
    print("  95%% CI on HR: %.3f - %.3f"
          % (np.exp(beta[0] - 1.96 * se[0]), np.exp(beta[0] + 1.96 * se[0])))
    res['cox'] = {'beta': float(beta[0]), 'se': float(se[0]),
                  'hazard_ratio': float(hr), 'p': float(p[0]),
                  'ci': [float(np.exp(beta[0] - 1.96 * se[0])),
                         float(np.exp(beta[0] + 1.96 * se[0]))]}
    print()
    if p[0] < 0.05:
        print("  -> workload IS associated with hazard.")
    else:
        print("  -> no detectable association between workload and hazard.")
        print("     A hazard ratio of 1.0 sits inside the interval.")

    print()
    print("=" * 94)
    print("  PART 4   controls -- what else could produce this?")
    print("=" * 94)
    print("  (a) install cohort vs outcome")
    for c, g in d.groupby('cls'):
        if g.dead.nunique() < 2:
            print("      %-11s single outcome class, cannot separate" % c)
            continue
        dd = g[g.dead == 1].first.map(pd.Timestamp.toordinal)
        aa = g[g.dead == 0].first.map(pd.Timestamp.toordinal)
        u = stats.mannwhitneyu(dd, aa) if len(dd) and len(aa) else None
        print("      %-11s ceased installed %s | censored installed %s%s"
              % (c,
                 g[g.dead == 1].first.dt.date.min(),
                 g[g.dead == 0].first.dt.date.min(),
                 "  (p=%.3f)" % u.pvalue if u else ""))
    print()
    print("  (b) does rate separate the classes rather than the devices?")
    for c, g in d.groupby('cls'):
        print("      %-11s rate %.0f-%.0f/day (%.1fx within class)"
              % (c, g.rate.min(), g.rate.max(),
                 g.rate.max() / max(g.rate.min(), 1e-9)))
    print()
    print("  (c) cessation cannot be distinguished from decommissioning in an")
    print("      archive. Any causal reading of PART 3 is unavailable.")

    out = os.path.join(RESULTS, 'uo_survival_results.json')
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=2, default=str)
    d.to_csv(os.path.join(DATA, 'uo_archive', '_survival_cohort.csv'), index=False)
    print("\nSaved %s" % out)


if __name__ == '__main__':
    main()
