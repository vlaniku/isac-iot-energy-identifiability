"""
Power analysis for the controlled spreading-factor experiment.

WHY THIS EXPERIMENT AND NOT ANOTHER. Every observational route in this project is
now closed, and the reason is always the same: cell-to-cell variation swamps the
workload effect (§3.10 validation) or the sample is too small to resolve it
(§3.5 MDE 15.6-22.2%, §4.8 MDE hazard ratio 3.23). A WITHIN-DEVICE CROSSOVER
removes cell variation exactly, because each device is its own control.

WHAT CAN ACTUALLY BE CONTROLLED. The Bosch manual documents no configurable
device parameters -- only OTAA credentials. Sensing cadence is therefore not a
lever without vendor cooperation. The LoRaWAN data rate IS a lever: ADR can be
disabled and the data rate pinned by LinkADRReq from ChirpStack, network-server
side, no device configuration and no vendor involvement.

That constrains what the experiment can measure: the COMMUNICATION share of the
energy budget, not the sensing share. The sensing share remains unmeasurable
without a current probe or vendor support, and this protocol does not pretend
otherwise.

THE CONTRAST. Uplink energy by spreading factor (standard 125 kHz airtime,
SX1276-class currents, 12-byte payload):

    SF7   9.82 mJ   1.00x        SF10   45.67 mJ   4.65x
    SF9  24.76 mJ   2.52x        SF12  171.13 mJ  17.42x

so a device moved from SF7 to SF12 multiplies its communication energy by 17.4
while everything else about it is unchanged. If the communication share at SF7
is s7, the depletion rate changes by

    delta_d  =  d0 * s7 * (k - 1),      d0 = 0.176 byte/day measured

which is a 16.4x amplification of s7 at the SF7/SF12 contrast. That is what makes
a share of a fraction of a percent detectable at all.

NOISE MODEL, CALIBRATED NOT INVENTED. The response is the DevStatusAns byte,
reported once per day and quantised to integers. The simulation reproduces that
staircase and is checked against the real deployment: deploy_depletion_analysis
measured OLS slope CIs of +/-0.031 to +/-0.065 mV/day over 47-83 day windows,
i.e. +/-0.005 to +/-0.010 byte/day. If the simulated SE does not land in that
range, the noise model is wrong and the power numbers are worthless.

Author: Vullnet Laniku
"""

import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')

D0 = 0.176                 # byte/day, measured baseline depletion
U0_PER_DAY = 3.68          # uplinks/day, fleet mean from the network server
MV_PER_BYTE = 6.3          # from pct = 62.52 V - 125.08 on a 1-254 scale
K = {7: 1.00, 8: 1.61, 9: 2.52, 10: 4.65, 11: 8.91, 12: 17.42}


def block_slope(days, d_true, rng, sigma_extra=0.0, b0=110.0):
    """OLS slope (byte/day) from one block of daily, integer-quantised readings."""
    t = np.arange(days, dtype=float)
    true = b0 - d_true * t + rng.normal(0, sigma_extra, days)
    obs = np.round(true)
    return stats.linregress(t, obs).slope


def calibrate_per_device(daily_convention=True):
    """
    Invert each device's own slope SE for its own noise sigma.

    SUPERSEDES the previous calibrate(), which swept sigma over an arbitrary
    grid (0, 0.25, 0.5, 1.0) and adopted the FIRST value whose simulated SE fell
    anywhere inside the pooled field range 0.0025-0.0053 byte/day. Two things
    were wrong with that. It compared a simulated 60- or 83-day block against a
    range pooled over windows of 47-83 days, so a sigma could "match" a window
    it was never run at; and taking the first match made the answer depend on
    the order of the grid. It reported the bracket 0.25-0.50; the devices
    themselves give a wider one.

    Each device contributes (n readings, span S days, OLS slope CI). For an OLS
    slope on points spread evenly over S days, sum (t - tbar)^2 = n S^2 / 12, so

        SE = sigma_eff * sqrt(12 / (n S^2))   =>   sigma_eff = SE * S * sqrt(n/12)

    and sigma_extra^2 = sigma_eff^2 - 1/12, the 1/12 removing the variance that
    integer rounding contributes on its own.

    `daily_convention` decides what n is. The experiment's power model treats a
    block as one reading per day, so the calibration must use the same
    convention: n = S. Setting it False uses each device's actual reading count,
    which is 1.1-1.8 per day, and yields larger sigmas. The daily convention is
    the one that is consistent with how the blocks are modelled, and it is the
    conservative choice for the design because it does not claim power from
    intra-day readings whose errors are dominated by quantisation and therefore
    are not independent.
    """
    path = os.path.join(RESULTS, 'deploy_depletion_results.json')
    with open(path) as fh:
        devs = json.load(fh)['per_device']
    print("=" * 92)
    print("  PART 0   noise model, inverted from each device's own slope SE")
    print("=" * 92)
    print("  convention: %s" % ("one reading per day" if daily_convention
                                else "each device's actual reading count"))
    print()
    print("  %-14s %6s %7s %10s %12s %10s"
          % ("device", "n", "span d", "CI (mV)", "SE (byte/d)", "sigma_x"))
    out = {}
    for r in devs:
        S = float(r['span_days'])
        n = S if daily_convention else float(r['n_batt'])
        se = r['ols_ci_mV'] / 1.96 / MV_PER_BYTE
        sig_eff = se * S * np.sqrt(n / 12.0)
        sx = float(np.sqrt(max(sig_eff ** 2 - 1.0 / 12.0, 0.0)))
        short = r['device'].replace('FIEK_UP_PARKING_', '')
        out[short] = sx
        print("  %-14s %6.0f %7.1f %10.5f %12.6f %10.3f"
              % (short, n, S, r['ols_ci_mV'], se, sx))
    print()
    return out


def calibrate(rng):
    """Kept so the old grid sweep can still be reproduced; not used for power."""
    print("  legacy grid sweep (superseded by calibrate_per_device):")
    for se_x in (0.0, 0.25, 0.5, 1.0):
        for days in (60, 83):
            sl = [block_slope(days, D0, rng, se_x) for _ in range(1500)]
            se = float(np.std(sl, ddof=1))
            lo, hi = 0.031 / MV_PER_BYTE / 1.96, 0.065 / MV_PER_BYTE / 1.96
            print("    %-4d d  sigma %.2f  sim SE %.5f  %s"
                  % (days, se_x, se, "in field range" if lo <= se <= hi else "-"))
    return 0.5


def crossover_power(s7, k, block_days, n_dev, n_pairs, sigma_extra, rng,
                    nsim=3000, alpha=0.05):
    """
    ABBA crossover, arm order randomised per device so calendar effects cancel.
    Unit of analysis: one (high-arm block, low-arm block) pair. Paired t-test.

    `k` is the ENERGY RATIO of the high arm to the low arm, passed as a float
    rather than looked up from an SF index. That is deliberate: the previous
    version took an SF index, and the SF10-vs-SF12 row was called with index 12,
    silently reproducing the SF7-vs-SF12 numbers. It is also what lets the same
    function price a contrast that is not an SF change at all -- see
    forced_uplink_ratio() -- since every such design enters only through k.
    """
    d_lo = D0                                  # low arm
    d_hi = D0 * (1 - s7 + s7 * k)              # high arm
    hits = 0
    for _ in range(nsim):
        diffs = []
        for _dev in range(n_dev):
            for _p in range(n_pairs):
                a = block_slope(block_days, d_hi, rng, sigma_extra)
                b = block_slope(block_days, d_lo, rng, sigma_extra)
                diffs.append(abs(a) - abs(b))   # more negative slope = faster depletion
        t, p = stats.ttest_1samp(diffs, 0.0)
        if p < alpha and np.mean(diffs) > 0:
            hits += 1
    return hits / nsim


def mde(k, block_days, n_dev, n_pairs, sigma_extra, rng, target=0.80, nsim=700):
    lo, hi = 1e-5, 0.30
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        pw = crossover_power(mid, k, block_days, n_dev, n_pairs, sigma_extra,
                             rng, nsim=nsim)
        if pw < target:
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------ closed form ----
def mde_closed(k, block_days, n_pairs_total, sigma_extra, alpha=0.05, power=0.80):
    """
    Minimum detectable share from the paired-t expression, no simulation.

    The response is a daily integer reading over a block of D days, so the OLS
    slope SE is sigma_eff * sqrt(12 / (D^3 - D)) with sigma_eff^2 = sigma_extra^2
    + 1/12, the 1/12 being the variance of rounding to integers. A pair is the
    difference of two independent block slopes, so its SD is sqrt(2) times that.
    The paired t-test on n pairs then detects

        delta_min = (t_{alpha/2, n-1} + t_{beta, n-1}) * SD_diff / sqrt(n)

    and delta = D0 * s * (k - 1) converts that to a share.

    The header block of this file asserted that Monte-Carlo and the closed form
    "agree to three digits". That was a claim in a comment; check_agreement()
    below makes it a check that runs.
    """
    n = n_pairs_total
    sigma_eff = np.sqrt(sigma_extra ** 2 + 1.0 / 12.0)
    se_slope = sigma_eff * np.sqrt(12.0 / (block_days ** 3 - block_days))
    sd_diff = np.sqrt(2.0) * se_slope
    tcrit = stats.t.ppf(1 - alpha / 2.0, n - 1) + stats.t.ppf(power, n - 1)
    delta_min = tcrit * sd_diff / np.sqrt(n)
    return float(delta_min / (D0 * (k - 1.0)))


def check_agreement(rng, tol=0.08, nsim=2500):
    """Closed form vs Monte-Carlo. Fails loudly rather than being asserted in prose.

    The residual disagreement is one-sided and explainable: the closed form adds
    1/12 for integer rounding, which is the variance of a uniform error, whereas
    rounding a smooth ramp produces a structured, serially correlated error that
    OLS absorbs better than white noise. So the closed form is mildly
    CONSERVATIVE, and most so when quantisation dominates sigma_extra. At
    sigma = 0.25 the two terms are comparable (0.0625 vs 0.0833) and the gap is
    ~5%; at sigma = 0.50 it is ~1%. The tolerance is set to 8% to admit that
    known bias while still failing on a real error, and the direction is printed
    so a one-sided drift cannot hide inside a two-sided tolerance.

    An earlier version ran the bisection at nsim = 700, where Monte-Carlo error
    alone moved the ratio by several percent between seeds. That is why nsim is
    raised here rather than the tolerance alone.
    """
    print("=" * 92)
    print("  PART 0b  closed form against Monte-Carlo")
    print("=" * 92)
    print("  %-8s %-8s %12s %12s %10s" % ("block d", "sigma", "closed", "monte-carlo", "ratio"))
    worst, ratios = 0.0, []
    for bd in (60, 90):
        for sx in (0.25, 0.50):
            cf = mde_closed(K[12], bd, 6, sx)
            mc = mde(K[12], bd, 3, 2, sx, rng, nsim=nsim)
            r = mc / cf
            ratios.append(r)
            worst = max(worst, abs(r - 1.0))
            print("  %-8d %-8.2f %11.4f%% %11.4f%% %10.3f"
                  % (bd, sx, 100 * cf, 100 * mc, r))
    ok = worst <= tol
    print("  worst relative disagreement %.1f%%  ->  %s"
          % (100 * worst, "OK" if ok else "*** THE TWO ROUTES DISAGREE ***"))
    print("  all ratios <= 1 (closed form conservative)? %s"
          % ("yes" if all(r <= 1.02 for r in ratios) else "NO - check for a real error"))
    if not ok:
        raise AssertionError("closed form and Monte-Carlo disagree by %.1f%%" % (100 * worst))
    return float(worst)


# ------------------------------------------------- the console-only fallback --
def forced_uplink_ratio(extra_per_day, u0=U0_PER_DAY):
    """
    Energy ratio for the contrast that needs no host access.

    Server-side pinning of the data rate needs a file on the ChirpStack host
    (see EXPERIMENT_PROTOCOL.md 3c). The one energy manipulation the web console
    alone can make is to enqueue CONFIRMED downlinks, each of which obliges the
    device to acknowledge, adding uplinks to a budget whose baseline is only
    U0_PER_DAY. The communication energy then scales as the uplink count, so

        k = (u0 + extra_per_day) / u0

    and everything else in this file applies unchanged. The catch is entirely in
    what `extra_per_day` can be: a Class A device is reachable only in the RX
    window following one of its own uplinks, so extra uplinks are available only
    if the device answers a confirmed downlink with an immediate uplink, which
    then opens the next window. Whether this Bosch firmware does that is not
    documented and is the pilot check in EXPERIMENT_PROTOCOL.md 3d.
    """
    return (u0 + float(extra_per_day)) / u0


def main():
    rng = np.random.default_rng(3)
    res = {}

    per_dev = calibrate_per_device(daily_convention=True)
    per_dev_actual = calibrate_per_device(daily_convention=False)
    LIVE = ('SENZOR_01', 'SENZOR_03', 'SENZOR_05')     # the three that would run it
    live_sig = [per_dev[d] for d in LIVE]
    SIGMAS = (min(live_sig), max(live_sig))
    sigma_extra = SIGMAS[1]           # conservative end, for the MC power grid
    res['sigma_per_device_daily'] = per_dev
    res['sigma_per_device_actual_n'] = per_dev_actual
    res['sigma_bracket_live_devices'] = list(SIGMAS)
    res['sigma_extra_conservative'] = sigma_extra
    print("  The three units that would run the experiment give sigma_x = %s,"
          % ", ".join("%.3f" % x for x in live_sig))
    print("  so every figure below is bracketed over %.2f - %.2f. The whole fleet"
          % SIGMAS)
    print("  spans %.2f - %.2f. The protocol previously reported 0.25 - 0.50, which"
          % (min(per_dev.values()), max(per_dev.values())))
    print("  came from a grid sweep rather than from the devices; see")
    print("  calibrate_per_device.__doc__.")

    print()
    res['closed_form_vs_mc_worst_rel'] = check_agreement(rng)

    print()
    print("=" * 92)
    print("  PART 1   power to detect a given communication share")
    print("=" * 92)
    print("  design: 3 devices, ABBA crossover (2 high/2 low blocks each),")
    print("          so 6 paired blocks; SF7 against SF12 (17.4x energy contrast)")
    print("          at sigma_extra = %.2f, the conservative end of the bracket" % sigma_extra)
    print()
    print("  %-12s" % "block days" + "".join("%12s" % ("s7=%.2f%%" % (100 * s))
                                             for s in (0.003, 0.01, 0.02, 0.05)))
    grid = {}
    for bd in (30, 45, 60, 90):
        row = []
        for s7 in (0.003, 0.01, 0.02, 0.05):
            pw = crossover_power(s7, K[12], bd, 3, 2, sigma_extra, rng, nsim=1200)
            row.append(pw)
            grid['bd%d_s%.3f' % (bd, s7)] = pw
        print("  %-12d" % bd + "".join("%11.0f%%" % (100 * x) for x in row))
    res['power_grid'] = grid

    print()
    print("=" * 92)
    print("  PART 2   minimum detectable communication share, by design choice")
    print("=" * 92)
    print("  %-24s %-8s %-8s %20s %12s"
          % ("contrast", "block d", "total d", "MDE (share)", "vs current"))
    rows = {}
    # DEFECT FIXED 20 Aug (second pass). The header comment claimed this row had
    # been corrected to use explicit energy ratios; the code still read an
    # undefined `sf_hi` and ignored `kval` entirely, so PART 2 raised NameError
    # and the published table was produced by hand instead. Both halves are now
    # real: kval is used, and the closed form that produced the published table
    # is the one called here.
    for kval, lab in [(17.42, "SF7 vs SF12 (17.4x)"), (6.91, "SF9 vs SF12 (6.9x)"),
                      (4.65, "SF7 vs SF10 (4.65x)"), (3.75, "SF10 vs SF12 (3.75x)")]:
        for bd in (45, 60, 90):
            band = [mde_closed(kval, bd, 6, sx) for sx in SIGMAS]
            total = bd * 4
            rows['%s_%d' % (lab, bd)] = [float(x) for x in band]
            print("  %-24s %-8d %-8d %9.3f%% - %.3f%% %8.0f-%.0fx"
                  % (lab, bd, total, 100 * band[0], 100 * band[1],
                     0.156 / band[1], 0.156 / band[0]))
        print()
    res['mde'] = rows
    print("  Each MDE is a range over sigma = %.2f - %.2f, the bracket the three live" % SIGMAS)
    print("  devices give when their own slope SEs are inverted. 'vs current' compares")
    print("  against the n=5 observational MDE of 15.6%.")

    print()
    print("=" * 92)
    print("  PART 3   what each outcome would license")
    print("=" * 92)
    band = [mde_closed(K[12], 90, 6, sx) for sx in SIGMAS]
    print("  Recommended design: SF7/SF12, 90-day blocks, ABBA, 3 devices")
    print("  -> total duration 360 days, MDE = %.3f%% - %.3f%%"
          % (100 * band[0], 100 * band[1]))
    print("  The link-budget prediction is 0.24 - 0.60%. The MDE range sits BELOW")
    print("  that whole interval at either end of the calibration bracket, so the")
    print("  design tests the prediction rather than bounding it.")
    print()
    print("  If the experiment DETECTS an effect at share s:")
    print("    the communication share is measured, not bounded, for the first")
    print("    time on deployed commercial ISAC hardware.")
    print()
    print("  If the experiment finds NOTHING:")
    print("    communication is bounded below %.3f%% BY MEASUREMENT, which is"
          % (100 * band[1]))
    print("    %.0fx tighter than the current observational bound and is a"
          % (0.156 / band[1]))
    print("    measurement of the device rather than of the instrument.")
    print("    Either outcome is publishable; that is the point of the design.")
    res['recommended_mde'] = [float(x) for x in band]

    # -------------------------------------------------------------- part 4 --
    print()
    print("=" * 92)
    print("  PART 4   the console-only fallback, priced")
    print("=" * 92)
    print("  Pinning the data rate needs a file on the ChirpStack host, which is")
    print("  not available (EXPERIMENT_PROTOCOL.md 3c). Without it the only energy")
    print("  manipulation the web console can make is to add uplinks, by enqueuing")
    print("  confirmed downlinks that the device must acknowledge. Baseline is")
    print("  %.2f uplinks/day, so the contrast is bounded by how many extra uplinks" % U0_PER_DAY)
    print("  can be forced per day, which is bounded in turn by whether the device")
    print("  answers a confirmed downlink immediately. Both cases are priced here.")
    print()
    print("  %-34s %8s %20s %12s" % ("extra uplinks/day", "k", "MDE at 90-d blocks", "vs current"))
    fb = {}
    cases = [(3.68, "no cascade: 1 per uplink"),
             (11.0, "3 per uplink"),
             (32.0, "cascade, ~1/hour sustained"),
             (60.4, "cascade, matching SF7/SF12")]
    for extra, lab in cases:
        k = forced_uplink_ratio(extra)
        band = [mde_closed(k, 90, 6, sx) for sx in SIGMAS]
        fb['extra%.1f' % extra] = {'k': float(k), 'label': lab,
                                   'mde_90d': [float(x) for x in band]}
        print("  %-34s %8.2f %8.3f%% - %.3f%% %8.0f-%.0fx"
              % ("+%.1f/day (%s)" % (extra, lab), k, 100 * band[0], 100 * band[1],
                 0.156 / band[1], 0.156 / band[0]))
    res['console_only_fallback'] = fb
    print()
    print("  Read this against the 0.24 - 0.60% link-budget prediction. Only the")
    print("  cascading cases reach it. Whether the cascade exists is one afternoon's")
    print("  pilot on the console and is the single fact that decides this design.")

    with open(os.path.join(RESULTS, 'experiment_protocol_power.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    print("\nSaved ../results/experiment_protocol_power.json")


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------- RESULTS ----
# Run 20 Aug 2026. Calibration: sigma_extra 0.25 and 0.50 byte both reproduce the
# field SE of 0.0025-0.0053 byte/day; 0.50 adopted as the conservative choice.
#
# POWER, 3 devices x ABBA = 6 paired blocks, SF7 vs SF12 (k = 17.4):
#
#   block days   s=0.30%   s=0.50%   s=1.00%   s=2.00%
#   30               17%       37%       90%      100%
#   45               46%       87%      100%      100%
#   60               80%       99%      100%      100%
#   90              100%      100%      100%      100%
#
# MINIMUM DETECTABLE COMMUNICATION SHARE (80% power, alpha = 0.05):
#
#   SF7  vs SF12 (17.4x)  60-day blocks, 240 d total -> 0.315%   50x better
#   SF7  vs SF12 (17.4x)  90-day blocks, 360 d total -> 0.166%   94x better
#   SF9  vs SF12 ( 6.9x)  60-day blocks, 240 d total -> 0.837%   19x better
#   SF9  vs SF12 ( 6.9x)  90-day blocks, 360 d total -> 0.473%   33x better
#   SF10 vs SF12 ( 3.75x) 60-day blocks, 240 d total -> 1.793%    9x better
#
#   "better" is against the current observational MDE of 15.6% (STATE.md 3.5).
#
# THE DECISIVE COMPARISON. The link-budget model predicts a communication share of
# 0.24-0.60% (STATE.md 2.8). The 240-day design has MDE 0.315%, which sits INSIDE
# that interval; the 360-day design has MDE 0.166%, which sits BELOW it. So the
# 360-day design can test the model's prediction directly, and the 240-day design
# can test its upper half. This is the first design in the project capable of
# testing a quantitative energy prediction rather than bounding it.


# ------------------------------------------------------- CORRECTED RESULTS ----
# Superseding the block above, which used a single calibration constant.
#
# TWO DEFECTS FOUND when the full run was compared against a fast re-derivation:
#   (1) calibrate() returned the FIRST sigma matching the field SE (0.25), but
#       0.50 matches equally well. The calibration is bracketed, not pinned, and
#       reporting one value overstates what the field data determines.
#   (2) the SF10-vs-SF12 row passed sf_hi=12 and silently reproduced the
#       SF7-vs-SF12 numbers. Contrasts are now explicit energy ratios.
#
# Monte-Carlo and the closed-form paired-t expression agree to three digits
# (0.192% both, 60-day blocks, sigma = 0.25), so the table below is analytic.
#
# MDE (80% power, alpha = 0.05, 6 paired blocks), BRACKETED over sigma 0.25-0.50:
#
#   contrast                block   total    MDE               vs 15.6%
#   SF7  vs SF12 (17.42x)   60 d    240 d    0.194 - 0.299%    52-80x
#   SF7  vs SF12 (17.42x)   90 d    360 d    0.107 - 0.162%    96-146x
#   SF9  vs SF12 ( 6.91x)   60 d    240 d    0.537 - 0.836%    19-29x
#   SF9  vs SF12 ( 6.91x)   90 d    360 d    0.297 - 0.454%    34-53x
#   SF10 vs SF12 ( 3.75x)   60 d    240 d    1.152 - 1.796%     9-14x
#   SF7  vs SF10 ( 4.65x)   60 d    240 d    0.874 - 1.351%    12-18x
#
# AGAINST THE LINK-BUDGET PREDICTION OF 0.24-0.60%:
#   240-day design  MDE 0.194-0.299%  -> INSIDE the interval (tests its upper half)
#   360-day design  MDE 0.107-0.162%  -> BELOW the whole interval (tests it outright)
# The verdict holds at both ends of the calibration bracket, which is why the
# bracket is reported rather than a point value.
