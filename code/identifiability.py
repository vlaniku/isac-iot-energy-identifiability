"""
What size of energy claim can this instrument resolve?

WHY THIS EXISTS. `experiment_protocol_power.py` answers a design question: given
the crossover we propose, what is the smallest COMMUNICATION SHARE it detects?
This inverts it into the question a reader of the literature actually has: given
a published claim of an X% energy improvement, what campaign would be needed to
check it, and is that campaign feasible on the hardware in question?

THE QUANTITY. Everything the crossover measures enters through the fractional
change in the depletion rate,

    eps = delta_d / d0,

which is the fractional change in the device's total energy consumption rate --
exactly what an energy-efficiency claim asserts. Rearranging the paired-t
expression already validated against Monte-Carlo in
`experiment_protocol_power.check_agreement`,

    eps_min(n, D, sigma) = (t_{alpha/2,n-1} + t_{beta,n-1})
                           * sqrt(2) * sigma_eff * sqrt(12 / (D^3 - D))
                           / (sqrt(n) * d0)

with sigma_eff^2 = sigma^2 + q^2/12 for a reading quantised to steps of q. Note
what is ABSENT: the spreading-factor ratio k. k converts eps into a share of the
budget; it does not affect what change in total energy is detectable. So eps_min
is a property of the instrument and the campaign alone, and it is the right axis
on which to ask whether a published claim is checkable at all.

THREE THINGS THIS PRODUCES.

  1. The inverse map: for a claim of size eps, the campaign duration required at
     a given fleet size.
  2. The quantisation floor. The daily one-byte battery reading contributes
     q^2/12 to sigma_eff regardless of how quiet the cell is, so part of the
     requirement is imposed by the telemetry rather than by the device.
  3. What finer instrumentation buys, priced in block-days saved, which is what
     the instrumentation specification in the paper is asking for.

CONTROL. `verify_against_published()` recomputes every row of the published MDE
table from this expression and requires each to agree to better than 2%. The
manuscript asserts that the two are the same expression, so the control covers
the whole table rather than the one row quoted in the text. If any row
disagrees, this file is wrong and refuses to print the rest.

Author: Vullnet Laniku
"""

import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')

D0 = 0.176              # byte/day, measured baseline depletion
SIGMA_LO = 0.3014       # byte, quietest live device (SENZOR_05)
SIGMA_HI = 0.7138       # byte, noisiest live device (SENZOR_01)
K_SF7_SF12 = 17.42      # energy ratio, used by the control only
ALPHA, POWER = 0.05, 0.80

# Service life bounds the campaign. 760 d is the measured service of the two
# units that ceased; the deployed fleet is five devices.
MAX_CAMPAIGN_DAYS = 760
FLEET_N = 5

# SENZOR_01 is a replacement unit, first reporting in 2025-09, and it is the
# noisiest of the three live devices. Excluding it the bracket is much tighter,
# so every power figure is reported both ways rather than resting silently on
# the unit with the shortest service history.
SIGMA_ORIG_HI = 0.4364      # byte, noisiest ORIGINAL unit (SENZOR_03)

# The design Sec. IX recommends, and the campaign Sec. VII prices against.
BLOCK_DAYS = 120
CAMPAIGN_DAYS = 2 * BLOCK_DAYS * 2

# The measured communication share, which upper-bounds any communication-side
# intervention: removing the term entirely saves exactly f_comm.
F_COMM_LO, F_COMM_HI = 0.0024, 0.0060

# The residuals about a block's depletion slope are NOT independent. Measured
# lag-1 rho at the daily analysis cadence is 0.464, it is not explained by the
# quantisation staircase and not removed by a quadratic trend, and simulating
# AR(1) residuals at that rho inflates the spread of fitted slopes by the factor
# below (1.62-1.74 over 60-190 day blocks). See noise_autocorrelation.py.
#
# This is applied to sigma_eff, so it multiplies every minimum detectable effect
# in this file. Setting it to 1.0 recovers the iid figures the earlier version
# of this work reported, which is how the two can be compared.
RHO_DAILY = 0.464
SE_INFLATION = 1.68


def eps_min(n_pairs, block_days, sigma, q=1.0, alpha=ALPHA, power=POWER,
            n_devices=None, inflate=SE_INFLATION):
    """Smallest detectable fractional change in total energy consumption rate.

    `n_devices` selects the unit of analysis, and the choice is not cosmetic.

      None (default) -- the PAIR is the unit: all n_pairs blocks enter one
        one-sample t-test with n_pairs - 1 degrees of freedom. This is correct
        only if the treatment effect is identical across devices, because pairs
        from the same device then differ only by within-device noise.

      an integer -- the DEVICE is the unit: pairs are averaged within a device
        and the t-test runs on the device means, with n_devices - 1 degrees of
        freedom. This is the summary-measures analysis, and it is the honest
        default when the treatment effect may vary by device.

    We cannot estimate the device-by-treatment variance from three devices, so
    the two are reported as bounds rather than one being chosen. The truth lies
    between them, at the pair-level figure if the effect is homogeneous and at
    the device-level figure if it is not.
    """
    if n_pairs < 2 or block_days < 3:
        return float('nan')
    sigma_eff = inflate * np.sqrt(sigma ** 2 + (q ** 2) / 12.0)
    se_slope = sigma_eff * np.sqrt(12.0 / (block_days ** 3 - block_days))
    sd_diff = np.sqrt(2.0) * se_slope
    if n_devices is None:
        n, sd = n_pairs, sd_diff
    else:
        if n_devices < 2:
            return float('nan')
        per_dev = n_pairs / float(n_devices)      # pairs averaged per device
        n, sd = n_devices, sd_diff / np.sqrt(per_dev)
    tcrit = (stats.t.ppf(1 - alpha / 2.0, n - 1)
             + stats.t.ppf(power, n - 1))
    return float(tcrit * sd / (np.sqrt(n) * D0))


def block_days_required(eps, n_pairs, sigma, q=1.0, cap=1000000,
                        n_devices=None, inflate=SE_INFLATION):
    """Invert eps_min for the block length. Monotone decreasing in block_days."""
    lo, hi = 3, 64
    while hi < cap and eps_min(n_pairs, hi, sigma, q, n_devices=n_devices,
                               inflate=inflate) > eps:
        lo, hi = hi, hi * 2
    if hi >= cap:
        return float('inf')
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if eps_min(n_pairs, mid, sigma, q, n_devices=n_devices,
                   inflate=inflate) > eps:
            lo = mid
        else:
            hi = mid
    return int(hi)


def campaign_days(block_days, n_pairs_total, n_devices):
    """ABBA: a pair is one high block and one low block, run back to back.

    Pairs are shared out across devices, which run in parallel, so the wall
    clock is set by the pairs one device has to carry.
    """
    pairs_per_device = max(1, int(np.ceil(n_pairs_total / max(n_devices, 1))))
    return 2 * block_days * pairs_per_device


def verify_against_published():
    """Every row of the published MDE table must fall out of this expression.

    Pinned to inflate=1.0 because the published table was computed under the iid
    assumption. The control checks the arithmetic, not the assumption; the
    assumption is checked in noise_autocorrelation.py and fails.

    The manuscript says eps_min reproduces that table, so the control is over
    the whole table rather than the one row that happens to be quoted in the
    text. Each entry is recomputed from eps_min and converted to a share by
    dividing by (k - 1); agreement is required to 2%, which is finer than the
    three significant figures the table prints.
    """
    published = [
        # (label, k, block_days, share_lo, share_hi)
        ('SF7 vs SF12  60 d', 17.42, 60, 0.00217, 0.00400),
        ('SF7 vs SF12  90 d', 17.42, 90, 0.001181, 0.002178),
        ('SF9 vs SF12  90 d', 6.90, 90, 0.00328, 0.00605),
        ('SF7 vs SF10  90 d', 4.65, 90, 0.00531, 0.00980),
    ]
    print("=" * 92)
    print("  CONTROL   does this reproduce every row of the published MDE table?")
    print("=" * 92)
    print("  %-20s %20s %20s %10s" % ("contrast", "published", "from here", "worst"))
    rows, ok_all = [], True
    for label, k, bd, lo, hi in published:
        got = [eps_min(6, bd, sig, inflate=1.0) / (k - 1.0)
               for sig in (SIGMA_LO, SIGMA_HI)]
        worst = max(abs(got[0] - lo) / lo, abs(got[1] - hi) / hi)
        ok = worst < 0.02
        ok_all = ok_all and ok
        print("  %-20s %8.4f%% - %.4f%% %8.4f%% - %.4f%% %9.1f%%  %s"
              % (label, 100 * lo, 100 * hi, 100 * got[0], 100 * got[1],
                 100 * worst, "" if ok else "*** DISAGREES ***"))
        rows.append({'contrast': label, 'k': k, 'block_days': bd,
                     'published': [lo, hi], 'recomputed': got,
                     'worst_rel': worst, 'agrees': bool(ok)})
    if not ok_all:
        raise AssertionError("the identifiability expression does not reproduce "
                             "the published MDE table; do not use its output")
    print("  all four rows agree to better than 2%.")
    return rows


def main():
    out = {'_method': {
        'd0_byte_per_day': D0, 'sigma_bracket': [SIGMA_LO, SIGMA_HI],
        'sigma_bracket_original_units_only': [SIGMA_LO, SIGMA_ORIG_HI],
        'alpha': ALPHA, 'power': POWER,
        'campaign_days': CAMPAIGN_DAYS, 'block_days': BLOCK_DAYS,
        'quantisation': 'daily reading, integer steps of 1 byte-unit',
        'note': ('eps is the fractional change in the total energy consumption '
                 'rate, which is what an efficiency claim asserts and what an '
                 'OBSERVATIONAL design must resolve. An AMPLIFIED design '
                 'manipulates a term by a known ratio k and resolves a share '
                 's = eps/(k-1) instead, which is why the two thresholds in '
                 'this paper differ and are not comparable directly.')}}
    out['control'] = verify_against_published()

    # ------------------------------------------------- the inverse map ------
    claims = [0.30, 0.10, 0.03, 0.01, 0.003, 0.001]
    print()
    print("=" * 96)
    print("  1. OBSERVATIONAL: what it takes to check a claim of a given size")
    print("     3 devices x 2 pairs, daily one-byte telemetry, 80% power, "
          "alpha=0.05")
    print("=" * 96)
    print("  %-9s %17s %17s %16s   %s"
          % ("claim", "block d (pair)", "block d (device)", "campaign d",
             "within 760 d?"))
    rows = []
    for c in claims:
        bp = [block_days_required(c, 6, s) for s in (SIGMA_LO, SIGMA_HI)]
        bd = [block_days_required(c, 6, s, n_devices=3) for s in (SIGMA_LO, SIGMA_HI)]
        camp = [4 * bd[0], 4 * bd[1]]
        ok = camp[1] <= MAX_CAMPAIGN_DAYS
        print("  %-9s %8d-%-8d %8d-%-8d %7d-%-8d   %s"
              % ("%.2f%%" % (100 * c), bp[0], bp[1], bd[0], bd[1],
                 camp[0], camp[1], "yes" if ok else "NO"))
        rows.append({'claim': c, 'block_days_pair': bp,
                     'block_days_device': bd, 'campaign_days_device': camp,
                     'within_service_life': bool(ok)})
    out['inverse_map'] = rows

    # ------------------------------------------ observational, by fleet -----
    print()
    print("  2. OBSERVATIONAL: smallest resolvable claim at the design Sec. IX")
    print("     recommends (%d d blocks, 2 pairs, %d d campaign)"
          % (BLOCK_DAYS, CAMPAIGN_DAYS))
    print("=" * 96)
    print("  %-22s %-18s %16s %16s" % ("fleet", "design", "pair-level",
                                       "device-level"))
    fleet = {}
    for lab, nd in (("this deployment (5)", FLEET_N), ("20 devices", 20),
                    ("100 devices", 100)):
        nb = nd * 2
        a = [eps_min(nb, BLOCK_DAYS, s) for s in (SIGMA_LO, SIGMA_HI)]
        b = [eps_min(nb, BLOCK_DAYS, s, n_devices=nd) for s in (SIGMA_LO, SIGMA_HI)]
        print("  %-22s %-18s %6.3f-%.3f%% %8.3f-%.3f%%"
              % (lab, "%d pairs x %d d" % (nb, BLOCK_DAYS),
                 100 * a[0], 100 * a[1], 100 * b[0], 100 * b[1]))
        fleet[lab] = {'n_devices': nd, 'n_pairs': nb, 'block_days': BLOCK_DAYS,
                      'eps_min_pair': a, 'eps_min_device': b}
    out['observational_by_fleet'] = fleet

    print()
    print("  Against f_comm = %.2f-%.2f%%, which upper-bounds any"
          % (100 * F_COMM_LO, 100 * F_COMM_HI))
    print("  communication-side intervention (removing the term entirely saves")
    print("  exactly f_comm):")
    for lab, v in fleet.items():
        worst = 100 * v['eps_min_device'][1]
        print("    %-22s eps_min = %6.3f%%  ->  %s"
              % (lab, worst,
                 "resolvable" if worst < 100 * F_COMM_LO else "NOT resolvable"))
    out['verdict_observational'] = {
        lab: bool(v['eps_min_device'][1] < F_COMM_LO) for lab, v in fleet.items()}

    # --------------------------------------------- amplified, by design -----
    print()
    print("  3. AMPLIFIED: the same telemetry, manipulating SF by a known ratio")
    print("     Detects a SHARE s = eps/(k-1), which is a different quantity")
    print("=" * 96)
    print("  %-26s %9s %16s %16s  %s"
          % ("design (3 devices)", "campaign", "pair-level", "device-level",
             "clears %.2f%%?" % (100 * F_COMM_LO)))
    amp = {}
    for pairs, bdays in ((2, 60), (2, 90), (2, BLOCK_DAYS), (3, 90)):
        nb, camp = 3 * pairs, 2 * bdays * pairs
        a = [eps_min(nb, bdays, s) / (K_SF7_SF12 - 1.0) for s in (SIGMA_LO, SIGMA_HI)]
        b = [eps_min(nb, bdays, s, n_devices=3) / (K_SF7_SF12 - 1.0)
             for s in (SIGMA_LO, SIGMA_HI)]
        ok = b[1] < F_COMM_LO
        print("  %-26s %7d d %6.3f-%.3f%% %8.3f-%.3f%%  %s"
              % ("%d pairs/device x %d d" % (pairs, bdays), camp,
                 100 * a[0], 100 * a[1], 100 * b[0], 100 * b[1],
                 "YES" if ok else "no"))
        amp['%dx%d' % (pairs, bdays)] = {
            'pairs_per_device': pairs, 'block_days': bdays, 'campaign_days': camp,
            'share_pair': a, 'share_device': b, 'clears_f_comm_lo': bool(ok)}
    out['amplified_by_design'] = amp

    # ----------------------------------- what the replacement unit costs ----
    print()
    print("  4. The noise bracket, with and without the replacement unit")
    print("=" * 96)
    print("  SENZOR_01 entered service in 2025-09 as a replacement and is the")
    print("  noisiest of the three (sigma = %.3f against %.3f and %.3f)."
          % (SIGMA_HI, SIGMA_LO, SIGMA_ORIG_HI))
    print("  %-26s %16s %16s" % ("sigma bracket", "pair-level", "device-level"))
    repl = {}
    for lab, shi in (("all three live units", SIGMA_HI),
                     ("two original units only", SIGMA_ORIG_HI)):
        a = [eps_min(6, 90, s) / (K_SF7_SF12 - 1.0) for s in (SIGMA_LO, shi)]
        b = [eps_min(6, 90, s, n_devices=3) / (K_SF7_SF12 - 1.0)
             for s in (SIGMA_LO, shi)]
        print("  %-26s %6.3f-%.3f%% %8.3f-%.3f%%"
              % (lab, 100 * a[0], 100 * a[1], 100 * b[0], 100 * b[1]))
        repl[lab] = {'sigma_hi': shi, 'share_pair': a, 'share_device': b}
    out['replacement_unit_sensitivity'] = repl

    # ------------------------------------------------ what finer buys -------
    print()
    print("  5. What finer telemetry buys: block days for a 1% claim, 6 pairs")
    print("=" * 96)
    print("  %-30s %10s %14s %14s"
          % ("reading", "q", "noisy device", "quiet device"))
    q_cases = [('1-byte daily (as deployed)', 1.0),
               ('4x finer (2 extra bits)', 0.25),
               ('16x finer (4 extra bits)', 0.0625),
               ('continuous (no quantisation)', 1e-6)]
    quant = []
    for lab, q in q_cases:
        b_hi = block_days_required(0.01, 6, SIGMA_HI, q=q)
        b_lo = block_days_required(0.01, 6, SIGMA_LO, q=q)
        print("  %-30s %10.4g %14d %14d" % (lab, q, b_hi, b_lo))
        quant.append({'label': lab, 'q': q, 'block_days_noisy': b_hi,
                      'block_days_quiet': b_lo})
    out['quantisation'] = quant
    ref_hi, inf_hi = quant[0]['block_days_noisy'], quant[-1]['block_days_noisy']
    ref_lo, inf_lo = quant[0]['block_days_quiet'], quant[-1]['block_days_quiet']
    save_hi = 100.0 * (ref_hi - inf_hi) / ref_hi
    save_lo = 100.0 * (ref_lo - inf_lo) / ref_lo
    print()
    print("  Removing quantisation entirely saves %.0f%% of the campaign on the"
          % save_hi)
    print("  noisy device and %.0f%% on the quiet one; cell variability binds."
          % save_lo)
    out['quantisation_verdict'] = {'noisy_saving_pct': save_hi,
                                   'quiet_saving_pct': save_lo}

    path = os.path.join(RESULTS, 'identifiability.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print()
    print("Saved %s" % path)


if __name__ == '__main__':
    main()
