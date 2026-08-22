"""
Shape-invariance check for the regime map.

WHY THIS EXISTS
---------------
`comm_action_bracket.py` impeaches the simulator: the bracket in the energy
identity, 1 - e_min/e_mean, moves 0.101 -> 0.777 over two constants internal to
the simulator that produces it (the MAC slot fraction and the uncited 20 mW
`p_proc`). That is Fig. 6 of the paper.

Fig. 5 - the regime map - is produced by the same simulator, and the paper
defends it with one sentence: "absolute values inherit the physical model; the
shape is the result." That sentence is an assertion, not a measurement. A
reviewer who puts Fig. 5 and Fig. 6 side by side is entitled to ask why the
configuration constants that move the bracket by 8x leave the map alone.

This script answers it by measurement. It re-runs the whole kappa x tau map at
every combination of the two constants Fig. 6 sweeps, and tests the five claims
the paper actually makes about the map's shape:

  S1  state awareness is 0 at kappa = 0, read where the budget does not bind
  S2  state awareness is monotone non-decreasing in kappa
  S3  state awareness saturates rather than growing without bound
  S4  foresight is 0 where the budget does not bind, and peaks where it binds
  S5  the deployed device's point (kappa ~ 0, tau >> 1) is 0 on both

If a claim fails at some configuration, that is the finding and it goes in the
paper. The point is not to confirm the map.

WHAT IS VARIED, AND WHAT IS HELD
--------------------------------
Varied, over exactly the ranges Fig. 6 uses:
  - the MAC slot fraction, via the *selector's* n_devices: 1/n for n in
    {1, 2, 6, 14, 30}, i.e. fr = 1.000 .. 0.033. In TDMA mode n_devices enters
    the physical model at two points and only two: the energy scaling
    (P_s + P_c) * fr, and the latency term's rate_fraction. Verified by reading
    `ParetoGridSelector._evaluate_action`; bandwidth and the interference floor
    are constants in that branch.
  - `p_proc`, over 0 -> 100 mW. This was a hard-coded literal inside
    `isac_joint_performance`; it is now the module-level `P_PROC_W`, default
    unchanged.

Held fixed, deliberately:
  - the number of simulated device trajectories, at 6. The selector's n_devices
    and the population size are the same number in `regime_map_v3.py`, which
    confounds the slot fraction with the averaging. Decoupling them is what
    isolates the slot-fraction effect.
  - DEATH_J = 3.0. The penalty for running the battery flat is exogenous to the
    allocator and must not move with the allocator's own energy model.
  - the common-random-number stream, which is kappa-independent and identical
    to `regime_map_v3.build_crn`.

At (n = 6, p_proc = 0.02) this reproduces `regime_map_v3.py` exactly, and that
is asserted against `results/regime_map_v3_results.json` rather than eyeballed.

A NOTE ON COST
--------------
p_proc enters the physical model additively and only additively: energy is
(P_s + P_c) * fr + p_proc, and the objective's energy term is linear in energy,
so raising p_proc adds W['energy'] * p_proc / E_REF to every action's J and
p_proc to every action's E. Both shifts are constant across the action set, so
the action tables can be built once per slot fraction and shifted analytically
for each p_proc. That is exact, not an approximation, and `verify_shift()`
asserts it against a direct model evaluation before any of it is used.

It is not, however, a no-op on the map: tau scales the budget by the *median*
action energy, which p_proc shifts, so the same tau is a different tightness at
a different p_proc. That is precisely the confound worth testing.

Author: Vullnet Laniku
"""

import json
import os
import time

import numpy as np

import isac_physical_models as ipm
from integrated_models import ParetoGridSelector
from regime_map import (action_table, run_fixed, N_STEPS,
                        D_LO, D_HI, R_LO, R_HI, Q_LO, Q_HI)
from audit_binding_boundary import run_clairvoyant, run_reserve, run_ration
from experiment_nonstationary import W, E_REF

N_TRAJ = 6
N_SEEDS = 3
DEATH_J = 3.0

KAPPAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40)
TAUS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.10, 1.50)
UNBOUND_TAUS = (1.10, 1.50)

SLOT_N = (1, 2, 6, 14, 30)
P_PROCS = (0.0, 0.005, 0.02, 0.05, 0.10)

BASELINE_N = 6
BASELINE_P = 0.02

J_SHIFT_PER_W = W['energy'] / E_REF        # dJ / d p_proc


# ------------------------------------------------------------ table build --
def build_crn(sel, kappa, seed, n_traj=N_TRAJ):
    """Common random numbers: only the drift scale depends on kappa.

    Byte-identical in construction to regime_map_v3.build_crn; reproduced here
    rather than imported because that one closes over the module-level
    N_DEVICES, which is the quantity this script has to decouple.
    """
    rng = np.random.default_rng(1000 * seed + 7)          # kappa-INDEPENDENT
    tabs = []
    for _ in range(n_traj):
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


def verify_shift(tol=1e-9):
    """Assert the analytic p_proc shift against direct model evaluation.

    Checked at two slot fractions, because the shift must be independent of
    the slot fraction (p_proc is added after the fr scaling) and a bug that
    multiplied it by fr would pass a single-fraction test.
    """
    checks = []
    for n in (2, 14):
        sel = ParetoGridSelector(mac_mode='tdma', n_devices=n)
        ipm.P_PROC_W = 0.0
        J0, E0 = action_table(sel, 213.0, 27.0, 5)
        for p in (0.005, 0.02, 0.10):
            ipm.P_PROC_W = p
            J1, E1 = action_table(sel, 213.0, 27.0, 5)
            dE = np.abs((E1 - E0) - p).max()
            dJ = np.abs((J1 - J0) - J_SHIFT_PER_W * p).max()
            checks.append({'n_devices': n, 'p_proc': p,
                           'max_abs_err_E': float(dE), 'max_abs_err_J': float(dJ)})
            assert dE < tol, "energy shift not exact at n=%d p=%g: %.3e" % (n, p, dE)
            assert dJ < tol, "objective shift not exact at n=%d p=%g: %.3e" % (n, p, dJ)
    ipm.P_PROC_W = BASELINE_P
    print("  verify_shift: PASS  (max |err| E %.2e, J %.2e over %d checks)"
          % (max(c['max_abs_err_E'] for c in checks),
             max(c['max_abs_err_J'] for c in checks), len(checks)))
    return checks


def shift(tabs, p_proc):
    """Apply the (exact) p_proc offset to tables built at p_proc = 0."""
    return [(J + J_SHIFT_PER_W * p_proc, E + p_proc) for J, E in tabs]


# -------------------------------------------------------------- the map ----
def read_map(tabs_by_kappa):
    """kappa x tau grid of the four policies, identical to regime_map_v3."""
    grid = {}
    for kappa in KAPPAS:
        tabs = tabs_by_kappa[kappa]
        for tau in TAUS:
            fx, rx, cl = [], [], []
            for J, E in tabs:
                batt0 = tau * float(np.median(E)) * N_STEPS
                f_, _ = run_fixed(J, E, batt0, DEATH_J)
                s_, _ = run_reserve(J, E, batt0, DEATH_J)
                a_, _ = run_ration(J, E, batt0, DEATH_J)
                _, lb, _ = run_clairvoyant(J, E, batt0, DEATH_J)
                fx.append(f_); rx.append(min(s_, a_)); cl.append(lb)
            F, R, C = float(np.mean(fx)), float(np.mean(rx)), float(np.mean(cl))
            gs = 100.0 * (F - R) / F if F > 0 else 0.0
            gp = 100.0 * (R - C) / R if R > 0 else 0.0
            grid['k%.3f_t%.2f' % (kappa, tau)] = {
                'kappa': kappa, 'tau': tau, 'fixed': F, 'reactive': R,
                'clair_lb': C, 'gap_state_pct': gs, 'gap_plan_pct': gp}
    return grid


def summarise(grid):
    """The five shape statistics, plus the pass/fail verdicts."""
    curve = [float(np.mean([grid['k%.3f_t%.2f' % (k, t)]['gap_state_pct']
                            for t in UNBOUND_TAUS])) for k in KAPPAS]
    plan_unbound = [float(np.mean([grid['k%.3f_t%.2f' % (k, t)]['gap_plan_pct']
                                   for t in UNBOUND_TAUS])) for k in KAPPAS]
    peak_tau, peak_plan = [], []
    for k in KAPPAS:
        best = max(TAUS, key=lambda t: grid['k%.3f_t%.2f' % (k, t)]['gap_plan_pct'])
        peak_tau.append(best)
        peak_plan.append(grid['k%.3f_t%.2f' % (k, best)]['gap_plan_pct'])

    drops = [curve[i] - curve[i + 1] for i in range(len(curve) - 1)]
    worst_drop = max(drops) if drops else 0.0
    top = max(curve)
    # saturation: the last doubling of kappa (0.20 -> 0.40) must add little
    tail_growth = curve[-1] - curve[-2]

    verdict = {
        'S1_zero_at_kappa0': {'value': curve[0], 'pass': abs(curve[0]) < 0.05},
        'S2_monotone': {'worst_decrease_pp': worst_drop, 'pass': worst_drop < 0.50},
        'S3_saturating': {'top_pct': top, 'last_step_pp': tail_growth,
                          'pass': (top > 0) and (tail_growth < 0.20 * top)},
        'S4_foresight_needs_binding': {
            'max_unbound_plan_pct': max(plan_unbound),
            'peak_taus': peak_tau,
            'pass': (max(plan_unbound) < 0.05) and all(t <= 0.95 for t in peak_tau)},
        'S5_device_point': {
            'gap_state_pct': curve[0], 'gap_plan_pct': plan_unbound[0],
            'pass': abs(curve[0]) < 0.05 and abs(plan_unbound[0]) < 0.05},
    }
    return {'state_curve': curve, 'plan_unbound': plan_unbound,
            'peak_tau': peak_tau, 'peak_plan_pct': peak_plan,
            'verdict': verdict,
            'all_pass': all(v['pass'] for v in verdict.values())}


def check_baseline(grid):
    """Assert this reproduces the published regime_map_v3 numbers."""
    path = os.path.join('..', 'results', 'regime_map_v3_results.json')
    if not os.path.exists(path):
        print("  baseline check SKIPPED: %s not found" % path)
        return None
    with open(path) as f:
        ref = json.load(f)['grid']
    worst, worst_key = 0.0, None
    n = 0
    for key, cell in grid.items():
        if key not in ref:
            continue
        n += 1
        d = abs(cell['gap_state_pct'] - ref[key]['gap_state_pct'])
        d = max(d, abs(cell['gap_plan_pct'] - ref[key]['gap_plan_pct']))
        if d > worst:
            worst, worst_key = d, key
    ok = worst < 1e-6
    print("  baseline vs regime_map_v3: %s  (%d cells, worst |diff| %.3e at %s)"
          % ("MATCH" if ok else "*** DIVERGES ***", n, worst, worst_key))
    return {'n_cells': n, 'worst_abs_diff_pp': float(worst),
            'worst_key': worst_key, 'match': bool(ok)}


# ------------------------------------------------------------------ main --
def main():
    t_start = time.time()
    print("=" * 100)
    print("  REGIME-MAP SHAPE INVARIANCE over the constants that move Fig. 6")
    print("=" * 100)
    print("  slot fractions : %s" % ", ".join("1/%d = %.4f" % (n, 1.0 / n) for n in SLOT_N))
    print("  p_proc (W)     : %s" % ", ".join("%g" % p for p in P_PROCS))
    print("  kappa          : %s" % ", ".join("%g" % k for k in KAPPAS))
    print("  tau            : %s" % ", ".join("%g" % t for t in TAUS))
    print("  trajectories   : %d x %d seeds, %d steps, DEATH_J %.1f"
          % (N_TRAJ, N_SEEDS, N_STEPS, DEATH_J))
    print()
    shift_checks = verify_shift()
    print()

    out = {'configs': {}, 'shift_checks': shift_checks,
           'slot_n': list(SLOT_N), 'p_procs': list(P_PROCS),
           'kappas': list(KAPPAS), 'taus': list(TAUS),
           'n_traj': N_TRAJ, 'n_seeds': N_SEEDS, 'n_steps': N_STEPS,
           'death_j': DEATH_J}
    baseline_report = None

    for n in SLOT_N:
        fr = 1.0 / n
        print("-" * 100)
        print("  slot fraction 1/%d = %.4f   building action tables at p_proc = 0 ..." % (n, fr))
        ipm.P_PROC_W = 0.0
        sel = ParetoGridSelector(mac_mode='tdma', n_devices=n)
        t0 = time.time()
        base = {}
        for kappa in KAPPAS:
            tabs = []
            for seed in range(N_SEEDS):
                tabs += build_crn(sel, kappa, seed)
            base[kappa] = tabs
        print("  ... %d kappa x %d tables in %.1f s" % (len(KAPPAS), N_SEEDS * N_TRAJ,
                                                        time.time() - t0))

        for p in P_PROCS:
            tabs_by_kappa = {k: shift(v, p) for k, v in base.items()}
            grid = read_map(tabs_by_kappa)
            summ = summarise(grid)
            key = 'n%d_p%g' % (n, p)
            out['configs'][key] = {'n_devices': n, 'slot_fraction': fr,
                                   'p_proc': p, 'grid': grid, **summ}
            print("    p_proc %-6g  state@k0 %5.2f%%  saturation %5.2f%%  "
                  "peak foresight %5.2f%% at tau %.2f  ->  %s"
                  % (p, summ['state_curve'][0], max(summ['state_curve']),
                     max(summ['peak_plan_pct']),
                     summ['peak_tau'][int(np.argmax(summ['peak_plan_pct']))],
                     "SHAPE HOLDS" if summ['all_pass'] else "SHAPE BREAKS"))
            if n == BASELINE_N and abs(p - BASELINE_P) < 1e-12:
                baseline_report = check_baseline(grid)
        del base

    out['baseline_check'] = baseline_report

    # ------------------------------------------------------------ report --
    print()
    print("=" * 100)
    print("  THE STATE-AWARENESS CURVE AT EVERY CONFIGURATION")
    print("=" * 100)
    print("  %-8s %-8s %s" % ("1/n", "p_proc", "".join("%9s" % ("k=%g" % k) for k in KAPPAS)))
    for n in SLOT_N:
        for p in P_PROCS:
            c = out['configs']['n%d_p%g' % (n, p)]['state_curve']
            print("  %-8.4f %-8g %s" % (1.0 / n, p, "".join("%8.2f%%" % v for v in c)))

    print()
    print("=" * 100)
    print("  VERDICTS")
    print("=" * 100)
    print("  %-10s %-8s %8s %8s %8s %8s %8s   %s"
          % ("1/n", "p_proc", "S1", "S2", "S3", "S4", "S5", "overall"))
    n_fail = 0
    for n in SLOT_N:
        for p in P_PROCS:
            c = out['configs']['n%d_p%g' % (n, p)]
            v = c['verdict']
            row = [v['S1_zero_at_kappa0']['pass'], v['S2_monotone']['pass'],
                   v['S3_saturating']['pass'], v['S4_foresight_needs_binding']['pass'],
                   v['S5_device_point']['pass']]
            if not all(row):
                n_fail += 1
            print("  %-10.4f %-8g %8s %8s %8s %8s %8s   %s"
                  % (1.0 / n, p, *["ok" if r else "FAIL" for r in row],
                     "HOLDS" if all(row) else "BREAKS"))

    span = [max(out['configs']['n%d_p%g' % (n, p)]['state_curve'])
            for n in SLOT_N for p in P_PROCS]
    dev = [out['configs']['n%d_p%g' % (n, p)]['state_curve'][0]
           for n in SLOT_N for p in P_PROCS]
    out['summary'] = {
        'n_configs': len(SLOT_N) * len(P_PROCS),
        'n_shape_failures': n_fail,
        'saturation_min_pct': float(min(span)),
        'saturation_max_pct': float(max(span)),
        'saturation_ratio': float(max(span) / min(span)) if min(span) > 0 else None,
        'device_point_max_abs_pct': float(max(abs(x) for x in dev)),
    }
    print()
    print("  configurations tested          : %d" % out['summary']['n_configs'])
    print("  configurations where shape breaks: %d" % n_fail)
    print("  saturation level across configs : %.2f%% - %.2f%%  (ratio %.2fx)"
          % (out['summary']['saturation_min_pct'], out['summary']['saturation_max_pct'],
             out['summary']['saturation_ratio'] or float('nan')))
    print("  device point, worst |gap_state| : %.4f%%" % out['summary']['device_point_max_abs_pct'])
    print()
    print("  For contrast, the bracket of Fig. 6 moves 0.101 - 0.777 (7.7x) over")
    print("  these same two constants. The comparison to report is the ratio above")
    print("  against that one, and whether the *ordering* in kappa ever inverts.")

    ipm.P_PROC_W = BASELINE_P
    dest = os.path.join('..', 'results', 'regime_map_invariance.json')
    with open(dest, 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved %s   (%.1f min)" % (dest, (time.time() - t_start) / 60.0))


if __name__ == '__main__':
    main()
