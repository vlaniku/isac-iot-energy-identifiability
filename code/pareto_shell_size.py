"""
How much does the Pareto shell actually prune?

WHY THIS EXISTS. The superseded submission's first contribution claimed the
non-dominated filter "reduces 27 actions to 5-12", p. 11 said "median six", and
Fig. 7(e) plotted it flat at 6. The claim mattered: the argument that a learner
"cannot select a dominated action" is only as strong as the fraction of actions
the shell removes. Nobody had measured it. This script does, on the same
`ParetoGridSelector` the regime map is built on, so the number is checkable
rather than asserted.

WHAT IT MEASURES. For each sampled exogenous state (distance, target range,
queue length, drawn over the ranges `regime_map.py` uses), the full
|POWER_LEVELS|^2 x |BW_RATIOS| = 75-point action grid is evaluated and passed
through `_pareto_filter`. Reported: the mean and the distribution of the number
of actions that survive, under both MAC modes.

The four objectives are energy, sensing error, communication unreliability and
latency, minimised. A shell that removes little is not a bug in the filter --
the filter is a textbook non-dominated sort and it is correct. It is a fact
about this action grid under this objective set: the actions genuinely trade
off against one another, so few are dominated. The defect was in the claim, not
in the code.

Author: Vullnet Laniku
"""

import json
import os

import numpy as np

from integrated_models import ParetoGridSelector, SimulatorDeviceState
from regime_map import D_LO, D_HI, R_LO, R_HI, Q_LO, Q_HI, N_DEVICES

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')

N_STATES = 400
SEED = 20260822


def shell_size(sel, dist, trange, queue):
    st = SimulatorDeviceState(
        device_id='x', battery_level=1.0, battery_capacity_joules=7400.0,
        location=(0.0, 0.0), queue_length=int(queue),
        distance_to_base=float(dist), target_range=float(trange))
    cands = []
    for sp in sel.POWER_LEVELS:
        for cp in sel.POWER_LEVELS:
            for bw in sel.BW_RATIOS:
                p = sel._evaluate_action(sp, cp, bw, st)
                cands.append({'obj': np.array([
                    p['energy_consumption'],
                    1.0 - p['sensing_accuracy'],
                    1.0 - p['communication_reliability'],
                    p['latency']])})
    return len(cands), len(sel._pareto_filter(cands))


def main():
    rng = np.random.default_rng(SEED)
    states = [(rng.uniform(D_LO, D_HI), rng.uniform(R_LO, R_HI),
               rng.uniform(Q_LO, Q_HI)) for _ in range(N_STATES)]

    out = {'_method': {
        'n_states': N_STATES, 'seed': SEED, 'n_devices': N_DEVICES,
        'state_ranges': {'distance_m': [D_LO, D_HI],
                         'target_range_m': [R_LO, R_HI],
                         'queue_length': [Q_LO, Q_HI]},
        'objectives': ['energy', '1 - sensing accuracy',
                       '1 - communication reliability', 'latency'],
        'note': ('The prior submission claimed the shell reduces the action '
                 'set to 5-12, median 6. This measures it.')}}

    print("=" * 84)
    print("  PARETO SHELL SIZE   %d sampled states, %d-point action grid"
          % (N_STATES, 75))
    print("=" * 84)
    print("  %-8s %8s %8s %8s %8s %8s %10s"
          % ("mac", "grid", "mean", "median", "min", "max", "% removed"))

    for mac in ('tdma', 'ofdma'):
        sel = ParetoGridSelector(mac_mode=mac, n_devices=N_DEVICES)
        sizes, grid_n = [], None
        for d, r, q in states:
            grid_n, k = shell_size(sel, d, r, q)
            sizes.append(k)
        a = np.array(sizes, dtype=float)
        rec = {'grid_size': int(grid_n), 'mean': float(a.mean()),
               'median': float(np.median(a)), 'min': int(a.min()),
               'max': int(a.max()), 'std': float(a.std(ddof=1)),
               'pct_surviving': float(100.0 * a.mean() / grid_n),
               'pct_removed': float(100.0 * (1.0 - a.mean() / grid_n))}
        out[mac] = rec
        print("  %-8s %8d %8.1f %8.0f %8d %8d %9.1f%%"
              % (mac, grid_n, rec['mean'], rec['median'], rec['min'],
                 rec['max'], rec['pct_removed']))

    print()
    print("  The prior submission claimed 5-12 surviving of 27, median 6.")
    print("  Measured on the 75-point grid the code actually enumerates:")
    for mac in ('tdma', 'ofdma'):
        print("    %-6s %.1f survive (%.0f%% of the grid); the shell removes %.0f%%."
              % (mac.upper(), out[mac]['mean'], out[mac]['pct_surviving'],
                 out[mac]['pct_removed']))
    print("  A filter that leaves most of the grid standing does not license")
    print("  the claim that a learner cannot choose a dominated action.")

    path = os.path.join(RESULTS, 'pareto_shell_size.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print()
    print("Saved %s" % path)


if __name__ == '__main__':
    main()
