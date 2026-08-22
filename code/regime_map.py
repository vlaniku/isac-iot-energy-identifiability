"""
Where does adaptive ISAC resource allocation actually pay?

Six formulations of the allocation problem have now been measured and closed on
this system. Rather than try a seventh, this maps the REGIME: it sweeps the
conditions that have to hold for allocation to be worth anything, finds the
boundary, and places real devices on the map by datasheet and by measurement.

The diagnosis being tested is that allocation pays only when three things hold:

  1. the knob moves real energy      -> controllable fraction  f
  2. the budget actually binds       -> effective tightness    tau
  3. the state moves, but trackably  -> drift per epoch        kappa

PART 1 - the analytic ceiling.

Sleep energy is a constant added to every action, so it cannot change which
action is optimal. It can only change feasibility. That gives a hard first-order
bound on what ANY allocation policy can save, with no simulation at all:

    max energy saving  =  f * (1 - e_min / e_mean)

where f is the controllable share of the budget and (e_min/e_mean) is how much
cheaper the cheapest action is than the average one. A perfect allocator that
always picked the cheapest action would save exactly this and no more.

PART 2 - the two gaps that matter, swept over (tau, kappa).

  fixed        one action for the whole horizon, chosen with hindsight. The best
               possible DESIGN CHOICE - no allocator, no state, no adaptation.
  reactive     per-step argmin J given current state, with a reserve rule so it
               does not die. This is what the grid search does.
  clairvoyant  exact Lagrangian optimum over a variable horizon. Upper bound.

  gap_state = (fixed - reactive) / fixed        value of STATE AWARENESS
  gap_plan  = (reactive - clairvoyant) / reactive  value of FORESIGHT

If gap_state is ~0 the allocation problem does not exist: pick a constant at
deployment and stop. If gap_plan is ~0 no planner, learner or scheduler can add
anything on top of a reactive optimiser.

PART 3 - device placement. Real hardware and published models located on the f
axis, with their ceilings computed from Part 1.

Physics comes from the corrected ISACPhysicalModels grid; only the regime
parameters are swept.

Author: Vullnet Laniku
"""

import json

import numpy as np

from integrated_models import ParetoGridSelector
from closed_loop_simulator import SimulatorDeviceState
from experiment_nonstationary import objective_from_perf
from audit_binding_boundary import run_clairvoyant, run_reserve

N_DEVICES = 6
N_STEPS = 400
N_SEEDS = 3
DEATH_J = 3.0

KAPPAS = (0.0, 0.005, 0.02, 0.05, 0.15, 0.40)
TAUS = (0.60, 0.70, 0.80, 0.90, 1.00, 1.30, 2.00)

D_LO, D_HI = 50.0, 400.0
R_LO, R_HI = 5.0, 50.0
Q_LO, Q_HI = 0.0, 12.0


def action_table(sel, dist, trange, queue):
    st = SimulatorDeviceState(
        device_id='x', battery_level=1.0, battery_capacity_joules=7400.0,
        location=(0.0, 0.0), queue_length=int(queue),
        distance_to_base=float(dist), target_range=float(trange))
    js, es = [], []
    for sp in sel.POWER_LEVELS:
        for cp in sel.POWER_LEVELS:
            for bw in sel.BW_RATIOS:
                p = sel._evaluate_action(sp, cp, bw, st)
                js.append(objective_from_perf(p))
                es.append(p['energy_consumption'])
    return np.array(js), np.array(es)


def trajectory(rng, kappa, T):
    """Exogenous state drifting at `kappa` of its own range per epoch."""
    d = rng.uniform(D_LO, D_HI)
    r = rng.uniform(R_LO, R_HI)
    q = rng.uniform(Q_LO, Q_HI)
    out = []
    for _ in range(T):
        d = float(np.clip(d + rng.normal(0, kappa * (D_HI - D_LO)), D_LO, D_HI))
        r = float(np.clip(r + rng.normal(0, kappa * (R_HI - R_LO)), R_LO, R_HI))
        q = float(np.clip(q + rng.normal(0, kappa * (Q_HI - Q_LO)), Q_LO, Q_HI))
        out.append((d, r, q))
    return out


def build(sel, kappa, seed):
    rng = np.random.default_rng(1000 * seed + int(kappa * 1e4))
    tabs = []
    for _ in range(N_DEVICES):
        traj = trajectory(rng, kappa, N_STEPS)
        rows = [action_table(sel, *s) for s in traj]
        tabs.append((np.array([x[0] for x in rows]),
                     np.array([x[1] for x in rows])))
    return tabs


def run_fixed(J, E, batt0, death):
    """Best single action held for the whole horizon, chosen with hindsight."""
    T, NA = J.shape
    tot_e = E.sum(axis=0)
    mean_j = J.mean(axis=0)
    feas = tot_e <= batt0
    if feas.any():
        return float(np.min(np.where(feas, mean_j, np.inf))), T
    # nothing survives the horizon: best action run until the budget dies
    best = np.inf
    for a in range(NA):
        c = np.cumsum(E[:, a])
        tau = int(np.searchsorted(c, batt0, side='right'))
        sc = (J[:tau, a].sum() + (T - tau) * death) / T
        best = min(best, sc)
    return float(best), 0


def main():
    sel = ParetoGridSelector(mac_mode='tdma', n_devices=N_DEVICES)

    # ---------------------------------------------------------- part 1 ----
    print("=" * 100)
    print("  PART 1   analytic ceiling on energy saving from allocation")
    print("=" * 100)
    probe = build(sel, 0.02, 0)
    Eall = np.concatenate([E.ravel() for _, E in probe])
    e_min_frac = float(np.mean([E.min(axis=1).mean() / E.mean() for _, E in probe]))
    spread = 1.0 - e_min_frac
    print("  action-set energy geometry (corrected physical model, 75 actions):")
    print("    e_min / e_mean = %.3f   =>  cheapest action is %.1f%% below average"
          % (e_min_frac, 100 * spread))
    print()
    print("  max saving from a PERFECT allocator  =  f * %.3f" % spread)
    print()
    print("  %-46s %8s %14s" % ("device / model", "f", "max saving"))
    devices = [
        ("Bosch TPS110, measured comm share (this work)", 0.17,
         "field, n=5, upper bound"),
        ("Bosch TPS110, datasheet-implied 35 s cadence", 0.287,
         "20 uA sleep, 1 mJ scan"),
        ("Newcastle LoRa air-quality node", 0.02,
         "no depletion in 30 d; fixed 10 min schedule"),
        ("Bai 2026 (IEEE Access) system model", 0.97,
         "P_cir 5 mW of 200 mW, 1 ms frames, no sleep"),
        ("TGCN submission as written", 1.00,
         "0.083 J per 1 s epoch, no sleep term"),
    ]
    place = {}
    for name, f, note in devices:
        print("  %-46s %8.3f %13.1f%%   (%s)" % (name, f, 100 * f * spread, note))
        place[name] = {'f': f, 'max_saving_pct': 100 * f * spread, 'basis': note}
    print()
    print("  The submission claims 24.9%% energy saving. The measured device's")
    print("  ceiling is %.1f%%. The claim is only reachable in the f~1 regime that"
          % (100 * 0.17 * spread))
    print("  the modelling literature assumes and the hardware does not occupy.")

    # ---------------------------------------------------------- part 2 ----
    print()
    print("=" * 100)
    print("  PART 2   where do the two gaps become material?")
    print("=" * 100)
    print("  gap_state = value of state awareness (fixed -> reactive)")
    print("  gap_plan  = value of foresight      (reactive -> clairvoyant)")

    grid = {}
    for kappa in KAPPAS:
        tabs = []
        for seed in range(N_SEEDS):
            tabs += build(sel, kappa, seed)
        print("\n  kappa = %.3f  (state moves %.1f%% of its range per epoch)"
              % (kappa, 100 * kappa))
        print("    %-8s %10s %11s %13s %11s %11s" % (
            "tau", "fixed", "reactive", "clairvoyant", "gap_state", "gap_plan"))
        for tau in TAUS:
            fx, rx, cx = [], [], []
            for J, E in tabs:
                batt0 = tau * float(np.median(E)) * N_STEPS
                f_, _ = run_fixed(J, E, batt0, DEATH_J)
                r_, _ = run_reserve(J, E, batt0, DEATH_J)
                c_, clb, _ = run_clairvoyant(J, E, batt0, DEATH_J)
                c_ = min(c_, r_, f_)
                fx.append(f_); rx.append(r_); cx.append(c_)
            F, R, C = np.mean(fx), np.mean(rx), np.mean(cx)
            gs = 100 * (F - R) / F if F > 0 else 0.0
            gp = 100 * (R - C) / R if R > 0 else 0.0
            print("    %-8.2f %10.5f %11.5f %13.5f %10.2f%% %10.2f%%"
                  % (tau, F, R, C, gs, gp))
            grid['k%.3f_t%.2f' % (kappa, tau)] = {
                'kappa': kappa, 'tau': tau, 'fixed': float(F),
                'reactive': float(R), 'clairvoyant': float(C),
                'gap_state_pct': float(gs), 'gap_plan_pct': float(gp)}

    # ---------------------------------------------------------- summary ---
    print()
    print("=" * 100)
    print("  BOUNDARY")
    print("=" * 100)
    mat = [(v['kappa'], v['tau'], v['gap_state_pct'], v['gap_plan_pct'])
           for v in grid.values()]
    live_state = [(k, t) for k, t, gs, gp in mat if gs > 5.0]
    live_plan = [(k, t) for k, t, gs, gp in mat if gp > 5.0]
    print("  cells where state awareness is worth >5%%: %d of %d"
          % (len(live_state), len(mat)))
    if live_state:
        print("    kappa >= %.3f, tau <= %.2f"
              % (min(k for k, t in live_state), max(t for k, t in live_state)))
    print("  cells where foresight is worth >5%%: %d of %d"
          % (len(live_plan), len(mat)))
    if live_plan:
        print("    kappa in [%.3f, %.3f], tau <= %.2f"
              % (min(k for k, t in live_plan), max(k for k, t in live_plan),
                 max(t for k, t in live_plan)))

    print()
    print("  TPS110 sits at kappa ~ 0 (197.5 min between decisions, state")
    print("  effectively re-randomised but decisions are too rare to track it)")
    print("  and tau >> 1 (the budget does not bind on controllable energy,")
    print("  because controllable energy is <=17%% of it).")

    with open('../results/regime_map_results.json', 'w') as f:
        json.dump({'n_devices': N_DEVICES, 'n_steps': N_STEPS, 'n_seeds': N_SEEDS,
                   'e_min_over_e_mean': e_min_frac, 'saving_spread': spread,
                   'device_placement': place, 'grid': grid}, f, indent=2)
    print("\nSaved ../results/regime_map_results.json")


if __name__ == '__main__':
    main()
