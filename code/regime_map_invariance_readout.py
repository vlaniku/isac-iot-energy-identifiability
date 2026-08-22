"""
Post-processing of the shape-invariance sweep. Reads the saved JSON; runs nothing.

WHY THIS EXISTS. `regime_map_invariance.py` read "state awareness" in the
tau >= 1.10 column on the assumption that a budget set to 1.10x the median action
energy cannot bind. In 4 of 25 configurations the zero-at-kappa-zero claim came
back non-zero (0.04% to 0.35%), and always at the two smallest slot fractions.

The assumption is the suspect, not the map. tau scales the budget by the MEDIAN
action energy, but the fixed policy is free to choose an action ABOVE the median,
so at a sufficiently skewed energy distribution tau = 1.10 does not unbind
anything. If that is what happened, it is visible without re-running: where the
budget genuinely does not bind, the clairvoyant lower bound must EQUAL the fixed
policy, because with a static state and no constraint the best fixed action is
already optimal.

So the test is: in the cells that failed, is clair_lb < fixed? If yes, the cell
was never unconstrained and the reading, not the result, was wrong. This script
answers that, then re-reads state awareness using the measured condition
(fixed == clair_lb at kappa = 0) instead of a tau threshold.

This is a better definition on its own terms. It does not depend on a normalising
constant, and it is checkable per cell.

Author: Vullnet Laniku
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'results', 'regime_map_invariance.json')
DEST = os.path.join(HERE, '..', 'results', 'regime_map_invariance_readout.json')

TOL = 1e-9          # fixed == clair_lb to this tolerance means "budget not binding"


def main():
    with open(SRC) as fh:
        d = json.load(fh)
    kappas, taus = d['kappas'], d['taus']

    print("=" * 104)
    print("  DIAGNOSIS   were the failing cells ever unconstrained?")
    print("=" * 104)
    print("  %-12s %10s %12s %14s %14s %10s"
          % ("config", "state@k0", "plan@k0", "fixed(k0,1.10)", "clair(k0,1.10)", "binding?"))
    diag = {}
    for key, c in d['configs'].items():
        cell = c['grid']['k0.000_t1.10']
        binding = (cell['fixed'] - cell['clair_lb']) > TOL
        failed = not c['verdict']['S1_zero_at_kappa0']['pass']
        diag[key] = {'binding_at_tau110': bool(binding), 'failed_S1': bool(failed),
                     'state_k0': c['state_curve'][0]}
        if failed or binding:
            print("  %-12s %9.3f%% %11.3f%% %14.5f %14.5f %10s"
                  % (key, c['state_curve'][0],
                     c['verdict']['S4_foresight_needs_binding']['max_unbound_plan_pct'],
                     cell['fixed'], cell['clair_lb'], "YES" if binding else "no"))

    failed = {k for k, v in diag.items() if v['failed_S1']}
    binding = {k for k, v in diag.items() if v['binding_at_tau110']}
    print()
    print("  configurations failing the zero-at-kappa-0 claim : %d  %s"
          % (len(failed), sorted(failed)))
    print("  configurations where tau = 1.10 still binds      : %d  %s"
          % (len(binding), sorted(binding)))
    print("  every failure is a binding cell?                 : %s"
          % ("YES" if failed <= binding else "NO - the assumption is not the cause"))

    # ------------------------------------------------- the corrected reading --
    print()
    print("=" * 104)
    print("  RE-READ   state awareness at the MEASURED unconstrained condition")
    print("=" * 104)
    print("  For each configuration, take the smallest tau at which fixed == clair_lb")
    print("  at kappa = 0, and read the whole kappa curve in that column and above.")
    print()
    print("  %-12s %8s %s" % ("config", "tau*", "".join("%9s" % ("k=%g" % k) for k in kappas)))
    out = {}
    for key, c in d['configs'].items():
        g = c['grid']
        tstar = None
        for t in taus:
            cell = g['k%.3f_t%.2f' % (0.0, t)]
            if (cell['fixed'] - cell['clair_lb']) <= TOL:
                tstar = t
                break
        if tstar is None:
            out[key] = {'tau_star': None}
            print("  %-12s %8s  (never unbinds within the tau grid)" % (key, "-"))
            continue
        cols = [t for t in taus if t >= tstar]
        curve = [float(np.mean([g['k%.3f_t%.2f' % (k, t)]['gap_state_pct'] for t in cols]))
                 for k in kappas]
        drops = [curve[i] - curve[i + 1] for i in range(len(curve) - 1)]
        out[key] = {'tau_star': tstar, 'curve': curve,
                    'zero_at_k0': abs(curve[0]) < 0.05,
                    'monotone': (max(drops) if drops else 0.0) < 0.50,
                    'saturation_pct': float(max(curve))}
        print("  %-12s %8.2f %s" % (key, tstar, "".join("%8.2f%%" % v for v in curve)))

    ok0 = sum(1 for v in out.values() if v.get('zero_at_k0'))
    okm = sum(1 for v in out.values() if v.get('monotone'))
    sats = [v['saturation_pct'] for v in out.values() if 'saturation_pct' in v]
    n = len(out)
    print()
    print("  zero at kappa = 0, all %d configurations : %d/%d" % (n, ok0, n))
    print("  monotone in kappa, all %d configurations : %d/%d" % (n, okm, n))
    print("  saturation level across configurations   : %.2f%% - %.2f%%  (%.1fx)"
          % (min(sats), max(sats), max(sats) / min(sats)))
    print()
    print("  READ THIS AS: the SHAPE of the regime map is invariant to the two")
    print("  constants that move the Fig. 6 bracket by 7.7x; its LEVEL is not, and")
    print("  moves by %.1fx over the same constants. The paper may claim the shape" % (max(sats) / min(sats)))
    print("  and must not claim the level.")

    with open(DEST, 'w') as fh:
        json.dump({'diagnosis': diag, 'reread': out,
                   'n_configs': n, 'zero_at_k0': ok0, 'monotone': okm,
                   'saturation_min_pct': float(min(sats)),
                   'saturation_max_pct': float(max(sats)),
                   'saturation_ratio': float(max(sats) / min(sats)),
                   'tolerance': TOL}, fh, indent=2)
    print("\nSaved %s" % DEST)


if __name__ == '__main__':
    main()
