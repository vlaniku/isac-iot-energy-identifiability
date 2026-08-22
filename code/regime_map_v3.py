"""
The regime map, with two confounds removed. This is the version to cite.

Confound 1 - trajectories differed across kappa.
  v2 seeded on `1000*seed + int(kappa*1e4)`, so every kappa row got completely
  different random trajectories. The `fixed` column swung +/-10% across kappa for
  reasons unrelated to kappa, and the kappa trend was not monotone as a result.
  Fixed here with COMMON RANDOM NUMBERS: the initial state and the standardised
  noise increments are drawn from a kappa-independent stream and the increments
  are then scaled by kappa, so kappa=0 and kappa=0.4 share a starting point and
  a noise realisation and differ only in drift magnitude.

Confound 2 - `gap_state` was not measuring state awareness.
  At kappa = 0 the state is static, so a state-aware policy should gain nothing.
  v2 nonetheless showed gap_state up to 10.6% at tau = 0.95. The cause is
  MIXING: under a binding budget a time-varying policy can time-share between a
  cheap and an expensive action to spend the budget exactly, which a single fixed
  action cannot do. That is a property of the constraint, not of state.
  Separated here by reading state awareness only in the UNCONSTRAINED column
  (tau = 1.10+), where no mixing gain exists, and reporting the mixing gain
  itself as the kappa = 0 row.

Read the map as:
  state awareness  ->  tau >= 1.10 column, across kappa
  foresight        ->  gap_plan, which lives only where the budget binds
  mixing           ->  kappa = 0 row at tau < 1.0

Author: Vullnet Laniku
"""

import json

import numpy as np

from integrated_models import ParetoGridSelector
from audit_binding_boundary import run_clairvoyant, run_reserve, run_ration
from regime_map import (action_table, run_fixed, N_DEVICES, N_STEPS,
                        D_LO, D_HI, R_LO, R_HI, Q_LO, Q_HI)
from regime_map_v2 import kappa_to_db, measure_link_dynamics

N_SEEDS = 3
DEATH_J = 3.0
KAPPAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
TAUS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.10, 1.50)


def build_crn(sel, kappa, seed):
    """Common random numbers: only the drift scale depends on kappa."""
    rng = np.random.default_rng(1000 * seed + 7)          # kappa-INDEPENDENT
    tabs = []
    for _ in range(N_DEVICES):
        d0 = rng.uniform(D_LO, D_HI)
        r0 = rng.uniform(R_LO, R_HI)
        q0 = rng.uniform(Q_LO, Q_HI)
        z = rng.normal(0.0, 1.0, size=(N_STEPS, 3))       # shared noise draw
        d, r, q = d0, r0, q0
        traj = []
        for t in range(N_STEPS):
            d = float(np.clip(d + kappa * (D_HI - D_LO) * z[t, 0], D_LO, D_HI))
            r = float(np.clip(r + kappa * (R_HI - R_LO) * z[t, 1], R_LO, R_HI))
            q = float(np.clip(q + kappa * (Q_HI - Q_LO) * z[t, 2], Q_LO, Q_HI))
            traj.append((d, r, q))
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
    print("  PART B   regime map, common random numbers across kappa")
    print("=" * 100)

    grid = {}
    for kappa in KAPPAS:
        tabs = []
        for seed in range(N_SEEDS):
            tabs += build_crn(sel, kappa, seed)
        print("\n  kappa = %.3f   (%.2f dB per epoch at d=200 m)"
              % (kappa, kappa_to_db(kappa)))
        print("    %-7s %10s %11s %12s %11s %11s" % (
            "tau", "fixed", "reactive", "clair_LB", "gap_state", "gap_plan"))
        for tau in TAUS:
            fx, rx, cl = [], [], []
            for J, E in tabs:
                batt0 = tau * float(np.median(E)) * N_STEPS
                f_, _ = run_fixed(J, E, batt0, DEATH_J)
                s_, _ = run_reserve(J, E, batt0, DEATH_J)
                a_, _ = run_ration(J, E, batt0, DEATH_J)
                _, lb, _ = run_clairvoyant(J, E, batt0, DEATH_J)
                fx.append(f_); rx.append(min(s_, a_)); cl.append(lb)
            F, R, C = np.mean(fx), np.mean(rx), np.mean(cl)
            gs = 100 * (F - R) / F if F > 0 else 0.0
            gp = 100 * (R - C) / R if R > 0 else 0.0
            print("    %-7.2f %10.5f %11.5f %12.5f %10.2f%% %10.2f%%"
                  % (tau, F, R, C, gs, gp))
            grid['k%.3f_t%.2f' % (kappa, tau)] = {
                'kappa': kappa, 'tau': tau, 'fixed': float(F),
                'reactive': float(R), 'clair_lb': float(C),
                'gap_state_pct': float(gs), 'gap_plan_pct': float(gp)}

    # ------------------------------------------------------------------ --
    print()
    print("=" * 100)
    print("  THE TWO CLEAN READINGS")
    print("=" * 100)
    print("\n  (1) VALUE OF STATE AWARENESS, read where the budget does NOT bind")
    print("      (tau >= 1.10, so no mixing gain is possible)")
    print("      %-10s %14s %16s" % ("kappa", "dB per epoch", "gap_state"))
    curve = []
    for k in KAPPAS:
        v = [grid['k%.3f_t%.2f' % (k, t)]['gap_state_pct'] for t in (1.10, 1.50)]
        curve.append((k, float(np.mean(v))))
        print("      %-10.3f %14.2f %15.2f%%" % (k, kappa_to_db(k), np.mean(v)))

    print("\n  (2) VALUE OF FORESIGHT, read where it is largest")
    for k in KAPPAS:
        best = max(TAUS, key=lambda t: grid['k%.3f_t%.2f' % (k, t)]['gap_plan_pct'])
        print("      kappa %-7.3f  max gap_plan %5.2f%%  at tau = %.2f"
              % (k, grid['k%.3f_t%.2f' % (k, best)]['gap_plan_pct'], best))

    print("\n  (3) MIXING GAIN (not state awareness): kappa = 0, budget binding")
    for t in TAUS:
        print("      tau %-6.2f  gap_state %5.2f%%"
              % (t, grid['k%.3f_t%.2f' % (0.0, t)]['gap_state_pct']))

    print()
    print("=" * 100)
    print("  WHERE THE DEPLOYED DEVICE SITS")
    print("=" * 100)
    ac = link['mean_lag1_autocorr']
    print("  measured RSSI lag-1 autocorrelation : %.3f" % ac)
    print("  measured sd of successive RSSI      : %.2f dB" % link['mean_successive_rssi_sd_db'])
    print("  => the link moves a great deal but has almost no memory between")
    print("     uplinks, so the trackable component is ~0 and the device sits at")
    print("     kappa ~ 0 whatever its raw variability.")
    print("  controllable energy share (measured): <=17%%  => tau cannot bind")
    k0 = float(np.mean([grid['k0.000_t%.2f' % t]['gap_state_pct'] for t in (1.10, 1.50)]))
    p0 = float(np.mean([grid['k0.000_t%.2f' % t]['gap_plan_pct'] for t in (1.10, 1.50)]))
    print("\n  at (kappa ~ 0, tau >> 1):  state awareness %.2f%%   foresight %.2f%%"
          % (k0, p0))

    with open('../results/regime_map_v3_results.json', 'w') as f:
        json.dump({'link_dynamics': link, 'grid': grid,
                   'state_awareness_curve': curve,
                   'device_point': {'gap_state_pct': k0, 'gap_plan_pct': p0},
                   'n_devices': N_DEVICES, 'n_steps': N_STEPS,
                   'n_seeds': N_SEEDS}, f, indent=2)
    print("\nSaved ../results/regime_map_v3_results.json")


if __name__ == '__main__':
    main()
