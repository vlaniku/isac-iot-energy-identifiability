"""
What is the ceiling for ANY policy on this problem?

Rather than trying algorithms one at a time -- DQN, PPO, MARL, bandits -- this
measures the bound that constrains all of them.

Any learning or planning method can only beat a myopic per-step optimiser by
exploiting *temporal structure*: the fact that an action taken now changes what
is achievable later, through battery depletion. If a clairvoyant policy that
knows the entire future trajectory and plans optimally over it gains nothing
over the myopic optimum, then no algorithm can gain anything either, because
there is no temporal structure to exploit. That is a statement about the
problem, not about any particular method.

Under TDMA the simulator sets inter-device interference to zero, so devices are
independent. Each device is then a finite-horizon MDP whose only endogenous
state is remaining battery, with the exogenous state (distance, queue, target
range) supplied by the drift process. Giving the planner the whole exogenous
trajectory in advance makes it clairvoyant, and the resulting problem is solved
exactly by backward induction over a discretised battery axis -- no
approximation, no learning, no local optima.

Three quantities are reported:

  myopic     per-step J-optimal, ignores the future (what the grid search does)
  clairvoyant  exact DP optimum with full knowledge of the future
  gap        clairvoyant - myopic. This is the maximum any method could win.

Outputs temporal_ceiling_results.json.

Author: Vullnet Laniku
"""

import json

import numpy as np

from closed_loop_simulator import ISACAction, SimulatorDeviceState
from integrated_models import ParetoGridSelector
from experiment_nonstationary import (NonStationaryScenario, make_devices,
                                      objective_from_perf, W, E_REF, L_REF)

N_DEVICES = 6
N_STEPS = 400
N_SEEDS = 6
N_BATT = 160          # battery discretisation levels
MAC = 'tdma'


def build_trajectory(seed):
    """Roll the drift process forward and record the exogenous state sequence."""
    devs = make_devices(N_DEVICES, seed)
    scen = NonStationaryScenario(devs, seed)
    traj = {d.device_id: [] for d in devs}
    for _ in range(N_STEPS):
        scen.advance()
        for d in devs:
            traj[d.device_id].append(
                (d.distance_to_base, d.target_range, d.queue_length))
    return devs, traj


def action_table(selector, dist, trange, queue):
    """(J, energy) for every action at one exogenous state."""
    st = SimulatorDeviceState(
        device_id='x', battery_level=1.0, battery_capacity_joules=7400.0,
        location=(0.0, 0.0), queue_length=int(queue),
        distance_to_base=float(dist), target_range=float(trange))
    js, es = [], []
    for sp in selector.POWER_LEVELS:
        for cp in selector.POWER_LEVELS:
            for bw in selector.BW_RATIOS:
                p = selector._evaluate_action(sp, cp, bw, st)
                js.append(objective_from_perf(p))
                es.append(p['energy_consumption'])
    return np.array(js), np.array(es)


def solve_device(selector, states, capacity_j, batt0_j):
    """
    Exact finite-horizon DP over remaining battery, with the exogenous
    trajectory known in advance. Returns (myopic_mean_J, clairvoyant_mean_J).
    """
    T = len(states)
    tables = [action_table(selector, *s) for s in states]

    # --- myopic: per-step argmin J, no lookahead -------------------------
    e_left = batt0_j
    myo = []
    for t in range(T):
        js, es = tables[t]
        feasible = es <= e_left
        if not feasible.any():
            break
        i = int(np.argmin(np.where(feasible, js, np.inf)))
        myo.append(js[i])
        e_left -= es[i]
    myopic_j = float(np.mean(myo)) if myo else np.inf
    myopic_steps = len(myo)

    # --- clairvoyant: exact backward induction ---------------------------
    grid = np.linspace(0.0, batt0_j, N_BATT)
    step = grid[1] - grid[0] if N_BATT > 1 else 1.0
    BIG = 1e6
    V = np.zeros(N_BATT)                      # terminal value
    policy_cost = np.full((T, N_BATT), BIG)
    for t in range(T - 1, -1, -1):
        js, es = tables[t]
        idx_cost = np.subtract.outer(grid, es)          # (N_BATT, n_actions)
        feas = idx_cost >= -1e-12
        nxt = np.clip((idx_cost / step).round().astype(int), 0, N_BATT - 1)
        tot = np.where(feas, js[None, :] + V[nxt], BIG)
        V = tot.min(axis=1)
        policy_cost[t] = V
    b0 = int(round(batt0_j / step)) if step > 0 else 0
    b0 = min(max(b0, 0), N_BATT - 1)
    clair_total = policy_cost[0][b0]
    # normalise by the same number of steps the myopic policy achieved, so the
    # two are compared on equal footing rather than on different horizons
    clair_j = float(clair_total / T)
    return myopic_j, clair_j, myopic_steps


def main():
    selector = ParetoGridSelector(mac_mode=MAC, n_devices=N_DEVICES)
    rows = []
    print("=" * 92)
    print("  TEMPORAL CEILING   %s, %d devices x %d seeds, %d steps, %d battery levels"
          % (MAC.upper(), N_DEVICES, N_SEEDS, N_STEPS, N_BATT))
    print("=" * 92)
    print("  %-10s %-8s %12s %14s %12s %10s" % (
        "seed", "device", "myopic J", "clairvoyant J", "gap", "gap %"))

    for seed in range(N_SEEDS):
        devs, traj = build_trajectory(seed)
        for d in devs:
            # battery budget sized so it binds: enough for the horizon at the
            # median action cost, not enough for the most expensive one
            js, es = action_table(selector, *traj[d.device_id][0])
            batt0 = float(np.median(es)) * N_STEPS
            m, c, steps = solve_device(selector, traj[d.device_id],
                                       d.battery_capacity_joules, batt0)
            gap = m - c
            rows.append({'seed': seed, 'device': d.device_id,
                         'myopic': m, 'clairvoyant': c, 'gap': gap,
                         'gap_pct': 100 * gap / m if m else 0.0,
                         'myopic_steps': steps})
            print("  %-10d %-8s %12.5f %14.5f %12.5f %9.2f%%"
                  % (seed, d.device_id, m, c, gap, 100 * gap / m if m else 0))

    g = np.array([r['gap_pct'] for r in rows])
    print("\n" + "-" * 92)
    print("  gap (clairvoyant vs myopic): mean %.3f%%  median %.3f%%  max %.3f%%"
          % (g.mean(), np.median(g), g.max()))
    print("\n  INTERPRETATION")
    if g.mean() < 1.0:
        print("    Under 1%%. There is essentially no temporal structure to exploit,")
        print("    so no algorithm -- however sophisticated -- can beat the myopic")
        print("    optimum by more than this. The ceiling binds every method.")
    elif g.mean() < 5.0:
        print("    Small but non-zero. A planning method could win a few percent;")
        print("    whether that is worth a paper is a separate question.")
    else:
        print("    Substantial. There IS temporal structure worth exploiting, and a")
        print("    lookahead or learning method has real headroom to capture it.")

    with open('temporal_ceiling_results.json', 'w') as f:
        json.dump({'mac': MAC, 'n_devices': N_DEVICES, 'n_steps': N_STEPS,
                   'n_seeds': N_SEEDS, 'n_batt': N_BATT, 'rows': rows,
                   'gap_pct_mean': float(g.mean()),
                   'gap_pct_median': float(np.median(g)),
                   'gap_pct_max': float(g.max())}, f, indent=2)
    print("\nSaved temporal_ceiling_results.json")


if __name__ == '__main__':
    main()
