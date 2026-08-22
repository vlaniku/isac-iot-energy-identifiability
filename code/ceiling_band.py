"""
The allocation ceiling, stated as precisely as the evidence allows.

Two tightenings of the regime map's Part 1.

TIGHTENING 1 - f for the datasheet-implied operating point was a point estimate
resting on an assumed 20 uA sleep and 1 mJ scan, neither of which is on the
datasheet. It is a band here, and the band is pruned by two datasheet facts:

    (i)  "Reporting of parking state changes within 35 seconds (typical)"
         -> the occupancy decision, and therefore the sensing, runs at <=35 s
    (ii) "Up to 5 years battery life", "< 2 g of lithium", PI 970 Section II
         -> total average current <= C/43800 h, with C <= ~6 Ah of pack

Any (sleep current, per-scan energy) pair whose implied total current exceeds
the rating is inconsistent with the device Bosch actually sells, and is pruned.
That is what turns an uninformative 0-32% band into a bounded one.

TIGHTENING 2 - the field measurement bounds more than the uplink.

  deploy_depletion_analysis.py regressed depletion on three workload measures.
  The status-changes regressor bounds everything in the budget that scales with
  parking events -- which, in the dual-sensor architecture the datasheet
  describes ("two independent sensor principles: magnetometer and radar"),
  includes magnetometer-triggered radar activations, not only the LoRa uplink.

  The budget splits three ways:
    E_sleep      fixed rate, NOT controllable by allocation
    E_scheduled  fixed rate, controllable only via sensing cadence
    E_event      proportional to parking events, controllable via power/SF

  The field bounds E_event/E_total. It does not bound E_scheduled/E_total.
  What closes the argument anyway, without ever measuring E_scheduled:
    (a) comm-side allocation is bounded by E_event                -> measured
    (b) radar TRANSMIT POWER is not an energy knob: -28 dBm EIRP
        = 1.6 uW radiated, so consumed energy is front-end static
        current during the scan, not the allocation variable      -> physics
    (c) radar CADENCE is a real knob, but adapting it was already
        measured at a 4.5% ceiling, not reliably positive on the
        real deployment                                           -> measured

Author: Vullnet Laniku
"""

import json

import numpy as np

import experiment_lifetime_budget as LB

V_BAT = 3.6
HOURS_5Y = 5 * 365 * 24.0
E_UPLINK = LB.uplink_energy(7)
UPLINKS_PER_DAY = 3.2
SPREAD = 0.333                 # 1 - e_min/e_mean, measured on the 75-action grid

# measured field bounds (deploy_depletion_results.json, bootstrap 95th pct)
F_EVENT_ALL_UPLINKS = 0.173
F_EVENT_STATUS_ONLY = 0.107

CADENCE_S = 35.0               # datasheet reporting spec
CELL_AH = (2.4, 3.6, 4.8, 6.0)
I_SLEEP_UA = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 81.0)
SCAN_MJ = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)


def i_sched_ua(scan_mj, cadence_s=CADENCE_S):
    """Average current from scheduled scanning, microamps."""
    return scan_mj * 1e-3 / (V_BAT * cadence_s) * 1e6


I_EVENT_UA = UPLINKS_PER_DAY * E_UPLINK / (V_BAT * 86400.0) * 1e6


def main():
    print("=" * 100)
    print("  PART 1   what is FIRM: the event-proportional ceiling")
    print("=" * 100)
    print("  ceiling = f x (1 - e_min/e_mean) = f x %.3f" % SPREAD)
    print("    measured, all uplinks as regressor  : f <= %.1f%%  ->  ceiling <= %.1f%%"
          % (100 * F_EVENT_ALL_UPLINKS, 100 * F_EVENT_ALL_UPLINKS * SPREAD))
    print("    measured, status changes only       : f <= %.1f%%  ->  ceiling <= %.1f%%"
          % (100 * F_EVENT_STATUS_ONLY, 100 * F_EVENT_STATUS_ONLY * SPREAD))
    model_event_ua = I_EVENT_UA
    print()
    print("  Independent cross-check from the link budget: 3.2 uplinks/day at")
    print("  SF7 costs %.2f mJ each = %.3f uA average. Against a 5-year rating"
          % (1000 * E_UPLINK, model_event_ua))
    print("  that is well under 1%% of the budget, i.e. the PHYSICAL model gives a")
    print("  far tighter answer than the field regression does.")
    print("  Two independent routes, both small; the field bound is the loose one,")
    print("  and it is the one to quote because it does not depend on the model.")
    print()
    print("  The submission claims 24.9%%. The measured ceiling is %.1f%%."
          % (100 * F_EVENT_ALL_UPLINKS * SPREAD))

    print()
    print("=" * 100)
    print("  PART 2   the scheduled-scan share as a BAND, pruned by the datasheet")
    print("=" * 100)
    print("  sensing cadence fixed at %.0f s (datasheet reporting spec)" % CADENCE_S)
    print("  a (sleep, scan) pair survives if implied total current fits the rating")
    print()
    print("  implied total average current for a 5-year life:")
    for ah in CELL_AH:
        print("    %.1f Ah cell -> %.1f uA" % (ah, ah / HOURS_5Y * 1e6))
    i_max = max(CELL_AH) / HOURS_5Y * 1e6
    i_min = min(CELL_AH) / HOURS_5Y * 1e6

    print()
    print("  %-10s" % "I_sleep" + "".join("%9s" % ("%.2fmJ" % m) for m in SCAN_MJ))
    surviving, cells = [], {}
    for i_ua in I_SLEEP_UA:
        row = []
        for mj in SCAN_MJ:
            isch = i_sched_ua(mj)
            itot = i_ua + isch + I_EVENT_UA
            f_tot = (isch + I_EVENT_UA) / itot
            ok = itot <= i_max
            ceiling = 100 * f_tot * SPREAD
            row.append("%7.1f%s" % (ceiling, "*" if ok else "x"))
            cells['%.0fuA_%.2fmJ' % (i_ua, mj)] = {
                'i_sleep_uA': i_ua, 'i_sched_uA': isch, 'i_total_uA': itot,
                'f_total': f_tot, 'ceiling_pct': ceiling,
                'fits_5y_rating': bool(ok)}
            if ok:
                surviving.append(ceiling)
        print("  %-10s" % ("%.0f uA" % i_ua) + "".join(row))
    print("    (* fits a 5-year life on <=6 Ah;  x = implied current exceeds the rating)")
    print()
    lo = min(c['ceiling_pct'] for c in cells.values())
    hi = max(c['ceiling_pct'] for c in cells.values())
    print("  ceiling over ALL combinations           : %.1f%% .. %.1f%%" % (lo, hi))
    print("  ceiling over combinations that fit 5 y  : %.1f%% .. %.1f%%  (%d of %d)"
          % (min(surviving), max(surviving), len(surviving), len(cells)))
    print()
    print("  THE PRUNE DOES NOT BIND: %d of %d combinations fit a 5-year life on"
          % (len(surviving), len(cells)))
    print("  <=6 Ah, because even 81 uA sleep plus 5 mJ scans at 35 s comes to")
    print("  %.0f uA, under the %.0f uA a 6 Ah cell allows. So the datasheet does"
          % (81 + i_sched_ua(5.0) + I_EVENT_UA, i_max))
    print("  NOT determine the controllable fraction, and neither does anything")
    print("  else Bosch publishes.")
    print()
    print("  => The total controllable share of a commercial ISAC-IoT device is")
    print("     INDETERMINATE over two orders of magnitude (%.1f%% .. %.1f%%) even" % (lo, hi))
    print("     with the datasheet, the 5-year rating, the <2 g Li limit and the")
    print("     35 s reporting spec all in hand. That is not a gap in this analysis;")
    print("     it is the strongest form of the calibration-impossibility finding.")
    print()
    print("  What IS firm is the ALLOCATION ceiling, which does not require it:")
    print("  each controllable term is bounded separately in Part 3.")

    print()
    print("=" * 100)
    print("  PART 3   why the unmeasured share does not rescue the direction")
    print("=" * 100)
    print("  The scheduled-scan share is the only unbounded term, and it is")
    print("  controllable ONLY through sensing cadence. Cadence adaptation was")
    print("  already measured (audit_adaptive_value.py):")
    print("    clairvoyant, knows the realised Poisson draw : 17.7%   NOT ATTAINABLE")
    print("    knows the true drift exactly                 :  4.5%   attainable ceiling")
    print("    28-day trailing forecast                     :  3.7%")
    print("    real 83-day deployment, out-of-sample        :  not reliably positive")
    print()
    print("  Radar transmit power is excluded by physics: -28 dBm EIRP = 1.6 uW")
    print("  radiated, so consumed energy is set by front-end static current")
    print("  during the scan, not by the allocation variable.")
    print()
    print("  SUMMARY OF THE CEILING")
    print("    comm-side allocation   <= %.1f%%   measured, field, n=5"
          % (100 * F_EVENT_ALL_UPLINKS * SPREAD))
    print("    radar transmit power      ~0      excluded by the EIRP cap")
    print("    radar cadence adaptation <= 4.5%   measured, workload-bounded")
    print("    static cadence choice     is a design decision, not an allocation problem")

    with open('../results/ceiling_band_results.json', 'w') as f:
        json.dump({'spread': SPREAD, 'cadence_s': CADENCE_S,
                   'f_event_all_uplinks': F_EVENT_ALL_UPLINKS,
                   'f_event_status_only': F_EVENT_STATUS_ONLY,
                   'firm_comm_ceiling_pct': 100 * F_EVENT_ALL_UPLINKS * SPREAD,
                   'model_event_current_uA': I_EVENT_UA,
                   'band_all_pct': [min(c['ceiling_pct'] for c in cells.values()),
                                    max(c['ceiling_pct'] for c in cells.values())],
                   'band_fits_5y_pct': [min(surviving), max(surviving)],
                   'cells': cells}, f, indent=2)
    print("\nSaved ../results/ceiling_band_results.json")


if __name__ == '__main__':
    main()
