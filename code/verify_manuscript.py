"""
Does the manuscript still say what the result files say?

WHY THIS EXISTS. Across thirteen revisions the single most persistent class of
defect has not been a wrong analysis. It has been a right analysis whose number
changed while the manuscript kept the old one -- a fleet table recomputed but
not retyped, an abstract quoting a figure the body had superseded, a claim
refuted by its own next sentence. Every one was found by a human reading
carefully, which is the wrong tool: reading does not scale over 2,100 lines and
does not run on every build.

This checks it mechanically. Each entry below names a number the manuscript
states, where that number comes from, and how it is rendered in LaTeX. The
check fails if the source moves and the prose does not follow.

WHAT IT DOES NOT DO. It cannot tell you a claim is wrong, only that the paper
and its evidence disagree. Assumptions, framing and inference are not
checkable this way and remain a reading problem.

Author: Vullnet Laniku
"""

import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
TEX = os.path.join(ROOT, 'paper_v4', 'main.tex')
RESULTS = os.path.join(ROOT, 'results')


def jload(name):
    p = os.path.join(RESULTS, name)
    try:
        return json.load(io.open(p, encoding='utf-8'))
    except UnicodeDecodeError:
        return json.load(io.open(p, encoding='cp1252'))
    except FileNotFoundError:
        return None


def pct(x, nd=3):
    return ('%.*f' % (nd, 100 * x)).rstrip('0').rstrip('.') if nd else str(x)


# Each claim: (label, callable -> list of strings that must ALL appear in the tex)
def build_claims():
    C = []

    d = jload('corpus_iot_subset.json')
    if d:
        A = d['populations']['A_full']
        B = d['populations']['B_iot']
        C.append(('corpus denominator', ['%d ISAC energy-efficiency papers' % A['n']]))
        C.append(('IoT-facing subset', ['%d name IoT' % B['n']]))
        C.append(('transmit-side count', ['%d (%.1f\\%%)' % (A['tx_any'], 100 * A['tx_share'])]))
        C.append(('standing-charge count', ['%d (%.1f\\%%)' % (A['sc_any'], 100 * A['sc_share'])]))

    d = jload('deploy_robustness_results.json')
    if d:
        a = d['mde_status_changes_per_day']['mde_share_pct']
        b = d['mde_uplinks_per_day']['mde_share_pct']
        C.append(('deployment MDE', ['%.1f--%.1f\\%%' % (a, b)]))

    d = jload('uo_survival_v2_results.json')
    if d:
        C.append(('Cox hazard ratio', ['%.3f' % d['cox']['hazard_ratio']]))
        C.append(('Cox MDE', ['%.2f' % d['mde_hazard_ratio']]))

    d = jload('noise_autocorrelation.json')
    if d:
        C.append(('daily rho', ['%.2f' % d['rho_daily_median']]))
        C.append(('SE inflation', ['%.2f$\\times$' % d['se_inflation_at_120d']]))

    d = jload('sf_energy_ratio.json')
    if d:
        # The ceiling on any allocator. This was wrong in the conclusion for a
        # full revision because nothing was watching it.
        cs = d.get('controllable_split')
        if cs:
            lo, hi = cs['f_comm_ctrl_pct']
            C.append(('controllable communication ceiling',
                      ['%.2f--%.2f\\%%' % (lo, hi)]))
        cf = d['confounds']
        C.append(('airtime ratio', ['$%.1f\\times$' % cf['airtime_ratio']]))
        C.append(('k with RX', ['$%.1f\\times$' % cf['k_with_rx']]))
        C.append(('MDE penalty from RX', ['%.2f' % cf['mde_share_penalty']]))

    d = jload('identifiability.json')
    if d:
        f = d.get('observational_by_fleet', {})
        for lab, key in (('this deployment (5)', 'this deployment (5)'),
                         ('20 devices', '20 devices'),
                         ('100 devices', '100 devices')):
            v = f.get(key)
            if v:
                lo, hi = v['eps_min_device']
                C.append(('fleet row: %s' % lab,
                          ['%.3f--%.3f\\%%' % (100 * lo, 100 * hi)]))

    d = jload('channel_coherence_control.json')
    if d:
        cen = d['within_device_sf_control'].get('120-240 min')
        if cen:
            C.append(('centred correlation at the decision interval',
                      ['%.3f' % cen['r']]))
        C.append(('median inter-uplink gap', ['%.1f~min' % d['median_gap_min']]))

    d = jload('identifiability_injection.json')
    if d:
        C.append(('injection: predicted eps',
                  [r'$\varepsilon = %.1f\%%$' % (100 * d['eps_predicted'])]))
        C.append(('injection: empirical eps',
                  [r'$\varepsilon = %.1f\%%$' % (100 * d['eps_empirical_80pct'])]))

    # The Pareto shell is a defect in the SUPERSEDED submission, documented in
    # the release README and not in this manuscript. It was in this manifest by
    # mistake, and the check correctly refused to pass until it came out.

    return C


def main():
    tex = io.open(TEX, encoding='utf-8').read()
    claims = build_claims()
    bad = []
    print("=" * 88)
    print("  MANUSCRIPT AGAINST ITS RESULT FILES   %d checked claims" % len(claims))
    print("=" * 88)
    for label, needles in claims:
        missing = [n for n in needles if n not in tex]
        if missing:
            bad.append((label, missing))
            print("  FAIL  %-44s expected %s" % (label, missing))
        else:
            print("  ok    %-44s %s" % (label, needles[0][:34]))
    print()
    if bad:
        print("  *** %d CLAIM(S) DO NOT MATCH THE RESULT FILES ***" % len(bad))
        print("  Either the manuscript is stale or the manifest here is. Fix one.")
        sys.exit(1)
    print("  every checked claim matches its source")


if __name__ == '__main__':
    main()
