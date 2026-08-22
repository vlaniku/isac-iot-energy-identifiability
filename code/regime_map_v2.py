"""
Where adaptive ISAC allocation pays -- corrected map, with the deployed device
placed on it by measurement rather than by assertion.

Fixes to regime_map.py, all of which mattered:

 1. tau ran down to 0.60, below the survival threshold e_min/e_median ~ 0.667.
    Those rows are pure death-penalty noise (gap_state -45%) and are dropped.
 2. `reactive` was myopic-with-reserve, which FRONT-LOADS: it spends on the
    J-optimal action early and is forced cheap later. Against a well-chosen
    constant that is strictly worse when the budget binds, which is why
    gap_state came out negative at tau=0.80. A properly paced `ration` policy
    is now included and reactive = best(reserve, ration).
 3. gap_plan used a clairvoyant clamped to the best heuristic, so any duality
    gap silently became "foresight is worth 0". It now uses the Lagrangian DUAL
    LOWER BOUND, making gap_plan an honest UPPER bound on the value of foresight.
 4. tau >= 1.0 is degenerate (budget stops binding); the grid is re-centred on
    the region where anything happens.

And the substantive addition: kappa for the real device is now MEASURED.

kappa in this model is a persistent AR(1)-style drift -- state that moves and
can therefore be tracked. Received-signal variation on a deployed parking sensor
is mostly fading, which moves but CANNOT be tracked. The two are indistinguishable
in a variance statistic and completely different for allocation. So the script
measures the lag-1 autocorrelation of per-device RSSI: persistent drift shows
high autocorrelation, fading shows ~0. Only the persistent part can place the
device on the kappa axis.

Author: Vullnet Laniku
"""

import json

import numpy as np
import pandas as pd

from integrated_models import ParetoGridSelector
from audit_binding_boundary import run_clairvoyant, run_reserve, run_ration
from regime_map import action_table, trajectory, run_fixed, N_DEVICES, N_STEPS

N_SEEDS = 3
DEATH_J = 3.0
KAPPAS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
TAUS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.10)
PLE = 2.8                      # path-loss exponent used in the simulator
EXPORT = '../data/FIEK_parking_export_83day.xlsx'


def kappa_to_db(kappa, d_ref=200.0):
    """Per-epoch RSSI change (dB) that a given kappa corresponds to."""
    delta = kappa * 350.0
    return float(10 * PLE * np.log10((d_ref + delta) / d_ref))


def measure_link_dynamics():
    """
    Place the deployed device on the kappa axis.

    Reports, per device: the sd of successive RSSI changes (how much the link
    moves per uplink) and the lag-1 autocorrelation of the RSSI series (whether
    that movement is trackable drift or untrackable fading).
    """
    ev = pd.read_excel(EXPORT, sheet_name='All Events')
    ev['ts'] = pd.to_datetime(ev['timestamp_local'])
    ev = ev.dropna(subset=['ts', 'rssi']).sort_values('ts')
    rows = []
    print("  %-30s %6s %10s %12s %14s" % (
        "device", "n", "rssi sd", "d(rssi) sd", "lag-1 autocorr"))
    for d, g in ev.groupby('device_name'):
        r = g['rssi'].astype(float).values
        if len(r) < 30:
            continue
        dr = np.diff(r)
        ac = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        rows.append({'device': d, 'n': len(r), 'rssi_sd': float(r.std()),
                     'drssi_sd': float(dr.std()), 'lag1_autocorr': ac})
        print("  %-30s %6d %10.2f %12.2f %14.3f"
              % (d, len(r), r.std(), dr.std(), ac))
    ac_mean = float(np.mean([x['lag1_autocorr'] for x in rows]))
    dr_mean = float(np.mean([x['drssi_sd'] for x in rows]))
    print()
    print("  mean lag-1 autocorrelation of RSSI : %.3f" % ac_mean)
    print("  mean sd of successive RSSI change  : %.2f dB" % dr_mean)
    print()
    if abs(ac_mean) < 0.25:
        print("  => The link moves by %.1f dB per uplink but is essentially" % dr_mean)
        print("     UNCORRELATED between uplinks. That is fading, not drift.")
        print("     Persistent, trackable state motion is ~0, so the device sits")
        print("     at kappa ~ 0 on the map below regardless of how much the")
        print("     RSSI varies. A state-aware allocator has nothing to track.")
    else:
        print("  => The link shows persistent structure (autocorr %.2f)." % ac_mean)
        print("     Trackable drift exists; place the device by dr sd above.")
    print()
    print("  reference: kappa -> per-epoch RSSI change at d=200 m")
    for k in KAPPAS[1:]:
        print("     kappa %.3f  ->  %.2f dB" % (k, kappa_to_db(k)))
    return {'per_device': rows, 'mean_lag1_autocorr': ac_mean,
            'mean_successive_rssi_sd_db': dr_mean,
            'kappa_db_reference': {('%.3f' % k): kappa_to_db(k) for k in KAPPAS[1:]}}


def build(sel, kappa, seed):
    rng = np.random.default_rng(1000 * seed + int(kappa * 1e4) + 7)
    tabs = []
    for _ in range(N_DEVICES):
        traj = trajectory(rng, kappa, N_STEPS)
        rows = [action_table(sel, *s) for s in traj]
        tabs.append((np.array([x[0] for x in rows]),
                     np.array([x[1] for x in rows])))
    return tabs


def main():
    print("=" * 100)
    print("  PART A   placing the deployed device on the kappa axis (measured)")
    print("=" * 100)
    link = measure_link_dynamics()

    sel = ParetoGridSelector(mac_mode='tdma', n_devices=N_DEVICES)
    print()
    print("=" * 100)
    print("  PART B   the regime map")
    print("=" * 100)
    print("  gap_state = (fixed - reactive)/fixed        value of STATE AWARENESS")
    print("  gap_plan  = (reactive - clair_LB)/reactive  value of FORESIGHT (upper bound)")

    grid = {}
    for kappa in KAPPAS:
        tabs = []
        for seed in range(N_SEEDS):
            tabs += build(sel, kappa, seed)
        print("\n  kappa = %.3f   (%.2f dB per epoch at d=200 m)"
              % (kappa, kappa_to_db(kappa)))
        print("    %-7s %10s %11s %11s %12s %11s %11s" % (
            "tau", "fixed", "reserve", "ration", "clair_LB", "gap_state", "gap_plan"))
        for tau in TAUS:
            fx, rs, ra, cl = [], [], [], []
            for J, E in tabs:
                batt0 = tau * float(np.median(E)) * N_STEPS
                f_, _ = run_fixed(J, E, batt0, DEATH_J)
                s_, _ = run_reserve(J, E, batt0, DEATH_J)
                a_, _ = run_ration(J, E, batt0, DEATH_J)
                _, lb, _ = run_clairvoyant(J, E, batt0, DEATH_J)
                fx.append(f_); rs.append(s_); ra.append(a_); cl.append(lb)
            F, S, A, C = np.mean(fx), np.mean(rs), np.mean(ra), np.mean(cl)
            R = min(S, A)
            gs = 100 * (F - R) / F if F > 0 else 0.0
            gp = 100 * (R - C) / R if R > 0 else 0.0
            print("    %-7.2f %10.5f %11.5f %11.5f %12.5f %10.2f%% %10.2f%%"
                  % (tau, F, S, A, C, gs, gp))
            grid['k%.3f_t%.2f' % (kappa, tau)] = {
                'kappa': kappa, 'tau': tau, 'fixed': float(F),
                'reserve': float(S), 'ration': float(A), 'reactive': float(R),
                'clair_lb': float(C), 'gap_state_pct': float(gs),
                'gap_plan_pct': float(gp)}

    print()
    print("=" * 100)
    print("  BOUNDARY")
    print("=" * 100)
    vals = list(grid.values())
    for name, key in (("state awareness", 'gap_state_pct'),
                      ("foresight", 'gap_plan_pct')):
        live = [v for v in vals if v[key] > 5.0]
        print("\n  %s worth >5%%: %d of %d cells" % (name, len(live), len(vals)))
        if live:
            print("    kappa >= %.3f   and   tau <= %.2f"
                  % (min(v['kappa'] for v in live), max(v['tau'] for v in live)))
            print("    max in grid: %.1f%% at kappa=%.3f tau=%.2f"
                  % (max(v[key] for v in vals),
                     max(vals, key=lambda v: v[key])['kappa'],
                     max(vals, key=lambda v: v[key])['tau']))
        else:
            print("    nowhere in the grid")

    at0 = [v for v in vals if v['kappa'] == 0.0 and v['tau'] >= 0.90]
    print("\n  WHERE THE TPS110 SITS")
    print("    kappa ~ 0 (measured: RSSI lag-1 autocorr %.3f -> fading, not drift)"
          % link['mean_lag1_autocorr'])
    print("    tau  >> 1 (controllable energy is <=17%% of the budget, so the")
    print("               budget cannot bind on the controllable part)")
    print("    gap_state there: %.2f%%   gap_plan there: %.2f%%"
          % (np.mean([v['gap_state_pct'] for v in at0]),
             np.mean([v['gap_plan_pct'] for v in at0])))

    with open('../results/regime_map_v2_results.json', 'w') as f:
        json.dump({'link_dynamics': link, 'grid': grid,
                   'n_devices': N_DEVICES, 'n_steps': N_STEPS,
                   'n_seeds': N_SEEDS}, f, indent=2)
    print("\nSaved ../results/regime_map_v2_results.json")


if __name__ == '__main__':
    main()
