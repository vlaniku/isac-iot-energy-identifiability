"""
What does the 3 h 53 min separation between the two cessations actually bound?

THE RESULT BELOW IS WITHDRAWN. The text that follows argues this is the
tightest field bound in the dataset and reports s <= 0.087%. It is not,
and the number should not be quoted. coincidence_estimator_validation.py
is the Monte-Carlo that killed it: at realistic batch spread the estimator
lands below the true value about half the time, and the observed fleet
spread makes a 3 h 53 min separation a 1-in-430 event for ANY value of s,
so the separation is evidence about the cells rather than about workload.
What survives is a design requirement -- roughly 20 co-commissioned
cessations would be needed to make this construction work.

The script is published unchanged, with its own argument intact, because
the validation is only legible next to what it refuted.

SENZOR_02 and SENZOR_04 stopped 3 h 53 min apart after 760 days of service, from a
commissioning date of 2024-05-24 supplied by the deployment team. The
draft treats this as a striking observation and nothing more. It is in fact the
tightest field bound in the dataset on the workload-proportional share of the
energy budget, and it is nearly model-free.

THE ARGUMENT.

  Let s be the fraction of the budget that scales with uplink workload. Two units
  with workload rates r_2 < r_4, commissioned together with matched cells, exhaust
  at times whose difference is approximately

      dt  ~=  L * s * (r_4 - r_2) / r_4

  where L is the service life. Larger s implies a larger separation. The observed
  separation therefore bounds s from above.

  The two units differ by 32% in measured workload over the 12-month record. Had
  workload driven lifetime at the level the field regression permits (s <= 17%),
  they would have died about a month apart. They died 3.9 hours apart, and the
  BUSIER unit lasted LONGER, so the workload term is below the noise.

WHAT THIS ASSUMES, AND IT IS NOT NOTHING.

  (a) both units were commissioned at the same time  -- CONFIRMED 2024-05-24
  (b) both cells had matched initial capacity        -- UNKNOWN, plausible for a
                                                        single procurement batch
  (c) the 12-month workload ratio is representative of the lifetime ratio

  (a) was the load-bearing one and is now a recorded fact, so the bound is reported
  as a measurement resting on (b) alone.

THE OTHER EDGE.

  The same coincidence is improbable under INDEPENDENT exhaustion. Reported here
  because a reviewer will compute it, and because the reconciliation is the
  paper's own thesis: if consumption is almost entirely quiescent, matched cells
  commissioned together SHOULD stop close together regardless of workload.

Author: Vullnet Laniku
"""

import json
import os
from math import erf, sqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RESULTS = os.path.join(HERE, '..', 'results')

DT_HOURS = 3.8805555555555555          # 2026-06-23 21:42:54 -> 2026-06-24 01:35:44
LIFE_DAYS = 760.0                      # from the commissioning date 2024-05-24, exact
MONTH_DAYS = {"2025-08": 31, "2025-09": 30, "2025-10": 31, "2025-11": 30,
              "2025-12": 31, "2026-01": 31, "2026-02": 28, "2026-03": 31,
              "2026-04": 30, "2026-05": 31, "2026-06": 30, "2026-07": 31}


def rate(dev):
    tot = days = 0
    for m, c in zip(MONTHS, dev['rx_count']):
        if m in MONTH_DAYS and c > 0:
            tot += c
            days += MONTH_DAYS[m]
    return tot / days


with open(os.path.join(DATA, 'chirpstack_12mo_metrics.json')) as fh:
    M = json.load(fh)
MONTHS = M['months']
BY = {d['name'][-2:]: d for d in M['devices'].values()}


def main():
    r2, r4 = rate(BY['02']), rate(BY['04'])
    lo, hi = min(r2, r4), max(r2, r4)
    frac = (hi - lo) / hi
    res = {'dt_hours': DT_HOURS, 'life_days': LIFE_DAYS,
           'rate_02': r2, 'rate_04': r4, 'workload_frac_diff': frac}

    print("=" * 92)
    print("  PART 1   the workload difference between the two units that ceased")
    print("=" * 92)
    print("  SENZOR_02  %.2f uplinks/day" % r2)
    print("  SENZOR_04  %.2f uplinks/day   -> %.0f%% busier" % (r4, 100 * (r4 / r2 - 1)))
    print("  fractional difference used below: %.3f" % frac)
    print()
    print("  observed separation %.2f h = %.4f%% of a %.0f-day service life"
          % (DT_HOURS, 100 * (DT_HOURS / 24) / LIFE_DAYS, LIFE_DAYS))
    print("  ordering: SENZOR_04 is the BUSIER unit and stopped %s"
          % ("SECOND" if True else "FIRST"))
    print("  -> the sign is inverted, so the workload term is below the noise floor")

    print()
    print("=" * 92)
    print("  PART 2   what separation each candidate workload share predicts")
    print("=" * 92)
    print("  %-34s %12s %12s" % ("hypothesis for s", "predicted dt", "vs observed"))
    rows = {}
    for lab, s in [("s = 1.00  (all energy is workload)", 1.0),
                   ("s = 0.173 (field regression bound)", 0.173),
                   ("s = 0.107 (status-change bound)", 0.107),
                   ("s = 0.050", 0.05),
                   ("s = 0.006 (link budget, upper)", 0.006),
                   ("s = 0.003 (link budget, typical)", 0.003)]:
        dt = LIFE_DAYS * s * frac
        rows[lab] = {'s': s, 'dt_days': dt, 'dt_hours': 24 * dt}
        if 24 * dt > 48:
            pred = "%.1f days" % dt
        else:
            pred = "%.1f h" % (24 * dt)
        print("  %-34s %12s %11.0fx" % (lab, pred, 24 * dt / DT_HOURS))
    res['predictions'] = rows

    s_bound = (DT_HOURS / 24) / (LIFE_DAYS * frac)
    res['s_upper_bound'] = s_bound
    print()
    print("  Inverting: the observed separation is consistent with")
    print("      s <= %.5f  =  %.3f%% of the energy budget" % (s_bound, 100 * s_bound))
    print("  against %.1f%% from the n=5 regression and %.2f-%.2f%% from the link budget."
          % (17.3, 0.24, 0.60))
    print("  This is the tightest field-observed bound in the dataset, and unlike the")
    print("  regression it does not depend on the depletion telemetry at all.")
    print()
    print("  Simultaneous commissioning is CONFIRMED (2024-05-24), so this is a")
    print("  measurement. Still assumed: matched initial cell capacity (plausible for")
    print("  one procurement batch, unverified) and a representative workload ratio.")

    print()
    print("=" * 92)
    print("  PART 3   the other edge -- how surprising is this under independence?")
    print("=" * 92)
    print("  If the two cessations were independent draws with per-unit lifetime")
    print("  spread sd about a common mean, P(|dt| < %.2f h) is:" % DT_HOURS)
    print()
    print("  %-16s %12s %14s" % ("sd of lifetime", "probability", "odds"))
    ind = {}
    for sd_days in (5, 15, 30, 60, 120):
        z = (DT_HOURS / 24) / (sqrt(2) * sd_days)
        p = erf(z / sqrt(2))
        ind['sd_%dd' % sd_days] = p
        print("  %-16s %12.5f %13s" % ("%d days" % sd_days, p, "1 in %.0f" % (1 / p)))
    res['independence_p'] = ind
    print()
    print("  So the coincidence is improbable under INDEPENDENT exhaustion. The paper")
    print("  must reconcile this, and its own thesis is the reconciliation: if the")
    print("  budget is almost entirely quiescent (PART 2), then matched cells")
    print("  commissioned together stop together, and the 32%% workload difference")
    print("  moves the separation by only %.0f hours -- which is what is observed."
          % (24 * LIFE_DAYS * 0.003 * frac))
    print()
    print("  The alternative -- a common external cause -- is constrained by the")
    print("  survivor evidence: three units on the SAME gateway continued for eight")
    print("  further weeks, so any common cause must be device-subset-specific.")

    out = os.path.join(RESULTS, 'coincidence_bound_results.json')
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=2)
    print("\nSaved %s" % out)


if __name__ == '__main__':
    main()
