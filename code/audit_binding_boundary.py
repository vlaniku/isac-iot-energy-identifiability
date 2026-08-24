"""
AUDIT of experiment_binding_boundary.py.

Defect being fixed
------------------
The heuristics and the Lagrangian clairvoyant were scored on different
objectives.

  myopic / ration : run until no action is affordable at step tau, then pay
                    DEATH_J for each of the remaining (T - tau) steps.
                    score = [ sum_{t<tau} J_t + (T-tau)*DEATH_J ] / T

  clairvoyant     : solve for a policy that survives ALL T steps. If no such
                    policy exists, return DEATH_J outright -- i.e. it pays the
                    death penalty for all T steps, never for a tail.

So at any budget where full-horizon survival is infeasible, the clairvoyant is
charged T*DEATH_J while the heuristics are charged only (T-tau)*DEATH_J. That
makes the "optimum" worse than the heuristics (80 invariant violations in the
recorded run) and makes gap_pct negative at 30-60% budget.

Fix: the clairvoyant now optimises the SAME objective, over a variable horizon.

  minimise over tau and over actions:
      [ min sum_{t<tau} J_t  s.t.  sum_{t<tau} e_t <= batt0 ] + (T-tau)*DEATH_J
  all divided by T.

tau is bounded above by tau_max, the largest horizon affordable if the cheapest
action is taken at every step. For each tau the inner problem is a
multiple-choice knapsack solved by Lagrangian bisection, as before.

Second defect, not previously noted
-----------------------------------
Lagrangian bisection returns a FEASIBLE primal solution, not the exact optimum:
integrality leaves a duality gap. The reported clairvoyant is therefore an
UPPER bound on the true optimum, which makes the gap (myopic - clairvoyant) an
UNDER-estimate and the rationing heuristic's capture share an OVER-estimate.
The 90-98% capture claim is exactly the quantity this biases. This script
reports the Lagrangian dual lower bound alongside the primal, and recomputes
the capture share against the lower bound, which is the conservative reading.

Third thing this checks
-----------------------
Whether the gap is an OPTIMISATION gap or merely a SURVIVAL gap. The recorded
run shows myopic dying in 20/20 cases at 70% budget and 1/20 at 90%, so the
"gap" may be nothing more than the death penalty. Reported here restricted to
the subset of runs where every policy survives.

Extra policy: `reserve` -- myopic, except it refuses any action that would leave
less than the cheapest-action cost for each remaining step. One extra line on
top of myopic, no pacing, no lookahead.

Author: audit pass
"""

import json
import os
import time

import numpy as np

N_DEVICES = 5
N_STEPS = 300
N_SEEDS = 4
FRACTIONS = tuple(round(float(x), 3) for x in
                  list(np.arange(0.30, 0.66, 0.05)) +
                  list(np.arange(0.66, 1.005, 0.02)) + [1.20])
DEATH_LEVELS = (1.0, 3.0, 10.0)
CACHE = 'audit_action_tables.npz'


# ------------------------------------------------------------------ tables --
def build_tables():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return [(z['J_%d' % i], z['E_%d' % i]) for i in range(int(z['n']))]
    from integrated_models import ParetoGridSelector
    import experiment_temporal_ceiling as TC
    TC.N_DEVICES, TC.N_STEPS = N_DEVICES, N_STEPS
    sel = ParetoGridSelector(mac_mode='tdma', n_devices=N_DEVICES)
    out = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        devs, traj = TC.build_trajectory(seed)
        for d in devs:
            tabs = [TC.action_table(sel, *s) for s in traj[d.device_id]]
            out.append((np.array([t[0] for t in tabs]),
                        np.array([t[1] for t in tabs])))
        print("  seed %d  (%.1fs)" % (seed, time.time() - t0))
    payload = {'n': len(out)}
    for i, (j, e) in enumerate(out):
        payload['J_%d' % i] = j
        payload['E_%d' % i] = e
    np.savez_compressed(CACHE, **payload)
    return out


# ---------------------------------------------------------------- policies --
def run_myopic(J, E, batt0, death):
    T = len(J)
    e, tot = batt0, 0.0
    for t in range(T):
        feas = E[t] <= e
        if not feas.any():
            return (tot + (T - t) * death) / T, t
        i = int(np.argmin(np.where(feas, J[t], np.inf)))
        tot += J[t, i]
        e -= E[t, i]
    return tot / T, T


def run_reserve(J, E, batt0, death):
    """Myopic, but never spend into the reserve needed to reach the horizon."""
    T = len(J)
    floor = np.concatenate([np.cumsum(E.min(axis=1)[::-1])[::-1], [0.0]])
    e, tot = batt0, 0.0
    for t in range(T):
        need_after = floor[t + 1]
        ok = (E[t] <= e - need_after)
        if not ok.any():
            ok = E[t] <= e
            if not ok.any():
                return (tot + (T - t) * death) / T, t
        i = int(np.argmin(np.where(ok, J[t], np.inf)))
        tot += J[t, i]
        e -= E[t, i]
    return tot / T, T


def run_ration(J, E, batt0, death):
    T = len(J)
    e, tot = batt0, 0.0
    for t in range(T):
        feas = E[t] <= e
        if not feas.any():
            return (tot + (T - t) * death) / T, t
        pace = e / max(T - t, 1)
        on_pace = feas & (E[t] <= pace)
        if on_pace.any():
            i = int(np.argmin(np.where(on_pace, J[t], np.inf)))
        else:
            i = int(np.argmin(np.where(feas, E[t], np.inf)))
        tot += J[t, i]
        e -= E[t, i]
    return tot / T, T


# ------------------------------------------------------------- clairvoyant --
def _lagrangian(J, E, batt0):
    """(primal feasible total J, dual lower bound) for a fixed horizon."""
    tau = J.shape[0]
    if tau == 0:
        return 0.0, 0.0
    rows = np.arange(tau)

    def solve(lam):
        idx = np.argmin(J + lam * E, axis=1)
        return float(J[rows, idx].sum()), float(E[rows, idx].sum())

    j0, e0 = solve(0.0)
    if e0 <= batt0:
        return j0, j0                      # unconstrained optimum, exact
    lo, hi = 0.0, 1.0
    while solve(hi)[1] > batt0 and hi < 1e12:
        hi *= 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if solve(mid)[1] > batt0:
            lo = mid
        else:
            hi = mid
    jf, ef = solve(hi)
    if ef > batt0 + 1e-12:
        return np.inf, np.inf              # infeasible at this horizon
    jl, el = solve(lo)
    dual = max(jl + lo * (el - batt0), jf + hi * (ef - batt0), 0.0)
    return jf, min(dual, jf)


def run_clairvoyant(J, E, batt0, death):
    """
    Exact optimum of the same variable-horizon objective the heuristics face.
    Returns (primal upper bound, dual lower bound, best tau).
    """
    T = len(J)
    cum_min = np.cumsum(E.min(axis=1))
    tau_max = int(np.searchsorted(cum_min, batt0, side='right'))
    best, best_lb, best_tau = death, death, 0          # tau = 0: die immediately
    for tau in range(tau_max, -1, -1):
        if (T - tau) * death / T >= best:              # J >= 0, so this prunes
            break
        F, Flb = _lagrangian(J[:tau], E[:tau], batt0)
        if not np.isfinite(F):
            continue
        s = (F + (T - tau) * death) / T
        slb = (Flb + (T - tau) * death) / T
        if s < best:
            best, best_tau = s, tau
        best_lb = min(best_lb, slb)
    return best, best_lb, best_tau


# -------------------------------------------------------------------- main --
def main():
    print("building action tables ...")
    tabs = build_tables()
    print("  %d device-trajectories, %d steps, %d actions"
          % (len(tabs), tabs[0][0].shape[0], tabs[0][0].shape[1]))
    assert min(J.min() for J, _ in tabs) >= 0.0, "J must be non-negative for the prune"

    out = {}
    violations = 0
    for death in DEATH_LEVELS:
        print("\n" + "=" * 118)
        print("  DEATH_J = %.1f   (worst single-step J in the tables = %.3f)"
              % (death, max(J.max() for J, _ in tabs)))
        print("=" * 118)
        print("  %-9s %8s %8s %8s %9s %9s %8s %8s %8s %10s %10s" % (
            "budget", "myopic", "reserve", "ration", "clair_UB", "clair_LB",
            "surv_m", "surv_rs", "surv_ra", "cap_UB%", "cap_LB%"))
        rows = {}
        for frac in FRACTIONS:
            ms, rss, ras, cu, cl = [], [], [], [], []
            sm = sr = sa = 0
            allsurv = []
            for J, E in tabs:
                batt0 = frac * float(np.median(E[0])) * N_STEPS
                m, tm = run_myopic(J, E, batt0, death)
                rs, tr = run_reserve(J, E, batt0, death)
                ra, ta = run_ration(J, E, batt0, death)
                cub, clb, tc = run_clairvoyant(J, E, batt0, death)
                if cub > min(m, rs, ra) + 1e-9:
                    violations += 1
                ms.append(m); rss.append(rs); ras.append(ra)
                cu.append(cub); cl.append(clb)
                sm += (tm == N_STEPS); sr += (tr == N_STEPS); sa += (ta == N_STEPS)
                allsurv.append(tm == N_STEPS and tr == N_STEPS and ta == N_STEPS
                               and tc == N_STEPS)
            n = len(tabs)
            m, rs, ra = np.mean(ms), np.mean(rss), np.mean(ras)
            cub, clb = np.mean(cu), np.mean(cl)
            best_h = min(rs, ra)
            cap_ub = 100 * (m - best_h) / (m - cub) if (m - cub) > 1e-9 else float('nan')
            cap_lb = 100 * (m - best_h) / (m - clb) if (m - clb) > 1e-9 else float('nan')
            idx = [i for i, ok in enumerate(allsurv) if ok]
            if idx:
                ms2 = float(np.mean([ms[i] for i in idx]))
                ra2 = float(np.mean([ras[i] for i in idx]))
                rs2 = float(np.mean([rss[i] for i in idx]))
                cu2 = float(np.mean([cu[i] for i in idx]))
                cl2 = float(np.mean([cl[i] for i in idx]))
                surv_gap_ub = 100 * (ms2 - cu2) / ms2
                surv_gap_lb = 100 * (ms2 - cl2) / ms2
            else:
                ms2 = ra2 = rs2 = cu2 = cl2 = surv_gap_ub = surv_gap_lb = float('nan')
            rows['%.2f' % frac] = {
                'myopic': float(m), 'reserve': float(rs), 'ration': float(ra),
                'clairvoyant_ub': float(cub), 'clairvoyant_lb': float(clb),
                'surv_myopic': sm, 'surv_reserve': sr, 'surv_ration': sa, 'n': n,
                'gap_pct_ub': float(100 * (m - cub) / m) if m else 0.0,
                'gap_pct_lb': float(100 * (m - clb) / m) if m else 0.0,
                'capture_pct_ub': float(cap_ub), 'capture_pct_lb': float(cap_lb),
                'n_all_survive': len(idx),
                'survivors_only': {'myopic': ms2, 'reserve': rs2, 'ration': ra2,
                                   'clair_ub': cu2, 'clair_lb': cl2,
                                   'gap_pct_ub': float(surv_gap_ub),
                                   'gap_pct_lb': float(surv_gap_lb)},
            }
            print("  %-9s %8.5f %8.5f %8.5f %9.5f %9.5f %5d/%-2d %5d/%-2d %5d/%-2d %10s %10s"
                  % ("%.0f%%" % (100 * frac), m, rs, ra, cub, clb,
                     sm, n, sr, n, sa, n,
                     ("%.1f" % cap_ub) if np.isfinite(cap_ub) else "n/a",
                     ("%.1f" % cap_lb) if np.isfinite(cap_lb) else "n/a"))
        out['death_%.1f' % death] = rows

    print("\n  invariant check (clairvoyant <= every heuristic): %s"
          % ("PASS" if violations == 0 else "FAILED in %d cases" % violations))

    print("\n" + "=" * 118)
    print("  SURVIVORS-ONLY VIEW  (runs where myopic, reserve, ration and the")
    print("  clairvoyant all reach the horizon -- no death penalty anywhere)")
    print("=" * 118)
    print("  %-9s %6s %10s %10s %10s %11s %11s" % (
        "budget", "n", "myopic", "ration", "clair_LB", "gap_UB%", "gap_LB%"))
    for frac, r in out['death_3.0'].items():
        s = r['survivors_only']
        if r['n_all_survive'] == 0:
            continue
        print("  %-9s %6d %10.5f %10.5f %10.5f %10.3f%% %10.3f%%"
              % (frac, r['n_all_survive'], s['myopic'], s['ration'],
                 s['clair_lb'], s['gap_pct_ub'], s['gap_pct_lb']))

    with open('../results/audit_binding_boundary_results.json', 'w') as f:
        json.dump({'n_devices': N_DEVICES, 'n_steps': N_STEPS, 'n_seeds': N_SEEDS,
                   'fractions': list(FRACTIONS), 'death_levels': list(DEATH_LEVELS),
                   'invariant_violations': violations, 'results': out}, f, indent=2)
    print("\nSaved ../results/audit_binding_boundary_results.json")


if __name__ == '__main__':
    main()
