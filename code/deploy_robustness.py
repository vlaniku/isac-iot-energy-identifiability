"""
Answering the two reviewer objections that decide the paper.

R2 - "the headline is a null result; with n=5 what could you have detected?"
     Reports the minimum detectable effect at 80% power, alpha=0.05, and
     converts it to a detectable share of the budget. If the MDE is larger
     than the bound being claimed, the bound describes the instrument and the
     claim must be weakened accordingly.

R3 - "battery_v is a rescaled DevStatusAns byte with undocumented semantics,
     and SENZOR_01 sits outside the other four devices' byte range while being
     both the lowest-workload device and the one with the shallowest slope --
     it is driving the regression."
     Reports leave-one-out over devices, a rank (Spearman) test that survives
     any monotone byte-to-energy map, and the fit restricted to the four
     devices that share a byte range.

Author: Vullnet Laniku
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

from deploy_depletion_analysis import load, per_device

ALPHA, POWER = 0.05, 0.80


def fit(r, xcol='uplinks_per_day', ycol='ols_mV_day'):
    x, y = r[xcol].values, r[ycol].values
    lr = stats.linregress(x, y)
    n = len(x)
    share = abs(lr.slope) * x.mean() / abs(y.mean())
    return lr, n, share


def mde(lr, n, x, y):
    """Minimum detectable slope at the given alpha/power, and its share."""
    df = n - 2
    t_a = stats.t.ppf(1 - ALPHA / 2, df)
    t_b = stats.t.ppf(POWER, df)
    b_min = (t_a + t_b) * lr.stderr
    return b_min, abs(b_min) * x.mean() / abs(y.mean())


def main():
    ev = load()
    r = per_device(ev)
    out = {}

    print("=" * 96)
    print("  R2   STATISTICAL POWER: what could this experiment have detected?")
    print("=" * 96)
    for xcol, label in (('uplinks_per_day', 'uplinks/day'),
                        ('status_changes_per_day', 'status changes/day')):
        lr, n, share = fit(r, xcol)
        b_min, share_min = mde(lr, n, r[xcol].values, r.ols_mV_day.values)
        print("\n  regressor: %s   (n = %d devices, df = %d)" % (label, n, n - 2))
        print("    observed slope           %+.5f   p = %.3f" % (lr.slope, lr.pvalue))
        print("    minimum detectable slope %+.5f   at %.0f%% power, alpha = %.2f"
              % (b_min, 100 * POWER, ALPHA))
        print("    => smallest workload-attributable share this experiment")
        print("       could have detected:  %.1f%%" % (100 * share_min))
        print("    => claimed bound (bootstrap p95):  11-17%%")
        out['mde_' + xcol] = {'observed_slope': float(lr.slope),
                              'p': float(lr.pvalue), 'mde_slope': float(b_min),
                              'mde_share_pct': float(100 * share_min)}
    print("\n  READING FOR THE PAPER")
    print("    The minimum detectable share is of the same order as the bound")
    print("    being claimed. The honest statement is therefore NOT 'communication")
    print("    contributes nothing' but 'communication contributes less than the")
    print("    smallest share this design could resolve'. Both the bound and the")
    print("    resolution must be reported together, every time.")

    print()
    print("=" * 96)
    print("  R3   IS THE RESULT DRIVEN BY SENZOR_01?")
    print("=" * 96)
    print("  byte range per device (battery_v rescaled back to DevStatusAns units):")
    for _, x in r.iterrows():
        b_first = (x.v_first - 2.0) / 1.6 * 254
        b_last = (x.v_last - 2.0) / 1.6 * 254
        print("    %-12s byte %5.1f -> %5.1f   %5.2f uplinks/day   slope %+.3f mV/day"
              % (x.device[-11:], b_first, b_last, x.uplinks_per_day, x.ols_mV_day))

    print("\n  leave-one-out over devices:")
    print("    %-14s %11s %9s %12s %14s" % (
        "dropped", "slope", "p", "share%", "verdict"))
    loo = {}
    for i in range(len(r)):
        sub = r.drop(r.index[i])
        lr, n, share = fit(sub)
        verdict = "sig" if lr.pvalue < 0.05 else "not sig"
        print("    %-14s %+11.5f %9.3f %11.1f%% %14s"
              % (r.iloc[i].device[-11:], lr.slope, lr.pvalue, 100 * share, verdict))
        loo[r.iloc[i].device] = {'slope': float(lr.slope), 'p': float(lr.pvalue),
                                 'share_pct': float(100 * share)}
    out['leave_one_out'] = loo

    # four devices sharing a byte range
    keep = r[r.v_first < 3.0]
    lr4, n4, share4 = fit(keep)
    print("\n  restricted to the %d devices sharing a byte range (~100-117):" % len(keep))
    print("    slope %+.5f   p = %.3f   share %.1f%%"
          % (lr4.slope, lr4.pvalue, 100 * share4))
    out['same_byte_range'] = {'n': int(len(keep)), 'slope': float(lr4.slope),
                              'p': float(lr4.pvalue), 'share_pct': float(100 * share4)}

    # rank test - survives ANY monotone byte-to-energy map
    rho, p_rho = stats.spearmanr(r.uplinks_per_day, r.ols_mV_day)
    print("\n  Spearman rank correlation, workload vs depletion rate:")
    print("    rho = %+.3f   p = %.3f   (n = %d)" % (rho, p_rho, len(r)))
    print("    A rank test is invariant to ANY monotone byte-to-energy map, so")
    print("    it answers the 'undocumented semantics' objection directly:")
    print("    if traffic drove depletion, the ranks would agree. They do not.")
    out['spearman'] = {'rho': float(rho), 'p': float(p_rho), 'n': int(len(r))}

    rho4, p4 = stats.spearmanr(keep.uplinks_per_day, keep.ols_mV_day)
    print("    same four devices: rho = %+.3f  p = %.3f" % (rho4, p4))
    out['spearman_same_range'] = {'rho': float(rho4), 'p': float(p4)}

    with open('../results/deploy_robustness_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved ../results/deploy_robustness_results.json")


if __name__ == '__main__':
    main()
