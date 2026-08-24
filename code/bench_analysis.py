"""
Turn the bench session into the two constants the paper currently models.

WHAT THIS REPLACES. Two numbers in this paper are modelled, and the manuscript
says so in both places rather than hiding it:

  k, the SF12/SF7 amplification factor.  sf_energy_ratio.py derives it from the
  LoRa airtime formula and an SX1276-class radio model -- V = 3.3, I_tx =
  43.9 mA, I_rx = 11.0 mA, a 3.66 mJ fixed overhead. Transmit-only it is 17.4x;
  including the class-A receive windows it is 11.6x. That 1.58x swing lands
  directly on the minimum detectable share through s = eps/(k-1), and no amount
  of further modelling settles it. A current trace of one device through a
  complete uplink cycle does.

  The battery response.  The whole identifiability criterion rests on the
  reported battery byte being proportional to energy drawn. Sec. VII-G states
  plainly that this cannot presently be verified. Discharging one cell through
  a known load while logging the byte measures it directly, as byte against
  coulombs.

WHAT THIS SCRIPT IS FOR. It consumes the two capture files and produces
results/bench_uplink_energy.json and results/bench_response.json. It ships
BEFORE the measurements exist, and exits cleanly saying so, because the
analysis being written first is what stops the bench session from being
repeated: every quantity the analysis needs is named in docs/BENCH_PROTOCOL.md,
so the capture can be checked against it at the bench rather than at the desk.

  python bench_analysis.py --selftest    validate the analysis on synthetic
                                         captures with known answers
  python bench_analysis.py               run it on the real captures

THE SELF-TEST IS NOT OPTIONAL. This project's recurring defect is a check that
agrees with itself: check_agreement validated a closed form against a
Monte-Carlo that shared its independence assumption, and could not see rho =
0.46. So the self-test here does not merely confirm the estimators recover a
known answer -- it also builds a case they must REJECT (a curved response) and
fails if they pass it. An estimator that cannot fail cannot certify.

Author: Vullnet Laniku
"""

import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results')

UPLINK_CSV = os.path.join(DATA, 'bench_uplink_traces.csv')
DISCHARGE_CSV = os.path.join(DATA, 'bench_discharge.csv')

# What the model currently says, so the measurement is reported against it
# rather than in isolation. From sf_energy_ratio.py / results/sf_energy_ratio.json.
MODELLED_K_TX_ONLY = 17.42
MODELLED_K_WITH_RX = 11.60
MODELLED_E_SF7_MJ = 9.82
MODELLED_E_SF12_MJ = 171.13

# Fraction of the peak excess current above which a sample counts as transmit
# rather than receive. The transmit draw of an SX1276 at +14 dBm is about 4x the
# receive draw, so anything from 0.3 to 0.7 separates them; 0.5 is the midpoint
# and the phase table is printed so the split can be eyeballed, not trusted.
TX_FRACTION = 0.50
# Above this fraction of peak excess, a sample counts as active rather than sleep.
ACTIVE_FRACTION = 0.05


# --------------------------------------------------------------- loading ---
def read_csv(path):
    """Minimal CSV reader: header row, then floats, with '' and 'nan' as NaN."""
    with io.open(path, encoding='utf-8-sig') as fh:
        header = [c.strip() for c in fh.readline().strip().split(',')]
        rows = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            row = []
            for p in parts:
                p = p.strip()
                row.append(np.nan if p == '' or p.lower() == 'nan' else float(p))
            rows.append(row)
    a = np.array(rows, dtype=float)
    return {c: a[:, i] for i, c in enumerate(header)}


# ------------------------------------------------------- uplink energetics --
def estimate_sleep(i):
    """Sleep draw, from the quiet window BEFORE the radio wakes.

    The obvious estimator -- the median of the lowest decile of the trace -- is
    wrong, and wrong in a way that biases exactly the number this session
    exists to settle. Sleep occupies about 20% of an SF7 capture but only 3% of
    an SF12 one, because the airtime is 28x longer while the fixed windows are
    not. So at SF12 the lowest decile is mostly RECEIVE current, the baseline
    comes out at the RX level, RX is subtracted from every phase, and k is
    understated. The self-test caught this at 12.0 against a true 16.5.

    Instead: find the first sample that rises clearly off the floor, and take
    the median of everything before it. That is the pre-trigger sleep window
    the capture protocol requires, and it does not depend on what fraction of
    the trace the radio is awake for. Falls back to the 1st percentile if the
    capture begins mid-activity, which the protocol tells the operator not to do.
    """
    lo, hi = float(i.min()), float(i.max())
    if hi <= lo:
        return lo
    rising = np.flatnonzero(i > lo + ACTIVE_FRACTION * (hi - lo))
    if len(rising) and rising[0] >= 5:
        return float(np.median(i[:rising[0]]))
    return float(np.percentile(i, 1))


def analyse_one_trace(t, i, v_rail):
    """Energy of one uplink cycle, split into transmit and receive.

    Everything is measured as EXCESS over the sleep baseline. The sleep draw is
    a standing cost of the device existing, not of sending, and charging it to
    the uplink would make k depend on how long the capture window was.
    """
    order = np.argsort(t)
    t, i = t[order], i[order]
    if len(t) < 10:
        return None
    dt = np.gradient(t)

    i_sleep = estimate_sleep(i)
    excess = i - i_sleep
    peak = float(excess.max())
    if peak <= 0:
        return None

    active = excess > ACTIVE_FRACTION * peak
    tx = excess > TX_FRACTION * peak
    rx = active & ~tx

    def energy(mask):
        return float(v_rail * np.sum(excess[mask] * dt[mask]))

    def charge(mask):
        return float(np.sum(excess[mask] * dt[mask]))

    def duration(mask):
        return float(np.sum(dt[mask]))

    return {
        'i_sleep_A': i_sleep,
        'i_peak_A': float(i.max()),
        'E_total_J': energy(active),
        'E_tx_J': energy(tx),
        'E_rx_J': energy(rx),
        'Q_total_C': charge(active),
        't_active_s': duration(active),
        't_tx_s': duration(tx),
        't_rx_s': duration(rx),
    }


def analyse_uplinks(d, v_rail):
    """Per-SF energy, and k as the SF12/SF7 ratio of measured uplink energy."""
    out, per_sf = {}, {}
    sfs = sorted(set(int(s) for s in d['sf']))
    for sf in sfs:
        reps = sorted(set(int(r) for r in d['rep'][d['sf'] == sf]))
        traces = []
        for rep in reps:
            m = (d['sf'] == sf) & (d['rep'] == rep)
            r = analyse_one_trace(d['t_s'][m], d['i_a'][m], v_rail)
            if r:
                traces.append(r)
        if not traces:
            continue
        agg = {}
        for key in ('E_total_J', 'E_tx_J', 'E_rx_J', 't_tx_s', 'i_sleep_A'):
            vals = np.array([tr[key] for tr in traces], dtype=float)
            agg[key] = float(vals.mean())
            agg[key + '_sd'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg['n_reps'] = len(traces)
        per_sf[sf] = agg

    out['per_sf'] = {str(k): v for k, v in per_sf.items()}
    if 7 in per_sf and 12 in per_sf:
        e7, e12 = per_sf[7]['E_total_J'], per_sf[12]['E_total_J']
        k_meas = e12 / e7 if e7 > 0 else float('nan')
        tx7, tx12 = per_sf[7]['E_tx_J'], per_sf[12]['E_tx_J']
        out['k_measured'] = float(k_meas)
        out['k_measured_tx_only'] = float(tx12 / tx7) if tx7 > 0 else float('nan')
        out['k_modelled_tx_only'] = MODELLED_K_TX_ONLY
        out['k_modelled_with_rx'] = MODELLED_K_WITH_RX
        # Which modelled value the measurement lands nearer is the whole point
        # of the session, so it is stated rather than left to the reader.
        d_tx = abs(k_meas - MODELLED_K_TX_ONLY)
        d_rx = abs(k_meas - MODELLED_K_WITH_RX)
        out['nearer'] = 'transmit-only (17.4x)' if d_tx < d_rx else 'with-RX (11.6x)'
        # s = eps/(k-1): what the measurement does to the detectable share.
        out['share_factor_vs_tx_only'] = float(
            (MODELLED_K_TX_ONLY - 1.0) / (k_meas - 1.0)) if k_meas > 1 else None
    return out


# ------------------------------------------------------ battery response ---
def analyse_response(d):
    """Is the reported byte proportional to charge drawn?

    Returns the slope in byte per coulomb, and a curvature test. Linearity is
    what the criterion assumes; a significant quadratic term means the byte is
    not a linear proxy for energy and every depletion-rate comparison in the
    paper needs a transform first.
    """
    t, i, byte = d['t_s'], d['i_a'], d['byte']
    order = np.argsort(t)
    t, i, byte = t[order], i[order], byte[order]

    # Cumulative charge drawn, in coulombs, by the trapezoid rule.
    q = np.concatenate([[0.0], np.cumsum(np.diff(t) * (i[1:] + i[:-1]) / 2.0)])

    m = ~np.isnan(byte)
    if m.sum() < 4:
        return {'error': 'fewer than four byte readings paired with charge'}
    qq, bb = q[m], byte[m]

    lin = np.polyfit(qq, bb, 1)
    pred = np.polyval(lin, qq)
    resid = bb - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((bb - bb.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # Curvature: is the quadratic coefficient distinguishable from zero? Its
    # standard error comes from the usual OLS covariance, so this is a t-test.
    X = np.column_stack([np.ones_like(qq), qq, qq ** 2])
    beta, *_ = np.linalg.lstsq(X, bb, rcond=None)
    r2nd = bb - X @ beta
    dof = len(qq) - 3
    t_quad = float('nan')
    if dof > 0:
        s2 = float(r2nd @ r2nd) / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(max(cov[2, 2], 0.0)))
        if se > 0:
            t_quad = float(beta[2] / se)

    span = float(bb.max() - bb.min())
    return {
        'n_readings': int(m.sum()),
        'charge_span_C': float(qq.max() - qq.min()),
        'byte_span': span,
        'slope_byte_per_C': float(lin[0]),
        'intercept_byte': float(lin[1]),
        'r2_linear': float(r2),
        'max_abs_resid_byte': float(np.max(np.abs(resid))),
        'max_resid_pct_of_span': float(100.0 * np.max(np.abs(resid)) / span)
        if span > 0 else float('nan'),
        'quadratic_coef': float(beta[2]),
        'quadratic_t': t_quad,
        # |t| > 2 is the conventional line and is used only as a flag; the
        # residual plot is what a reader should believe.
        'curvature_flagged': bool(abs(t_quad) > 2.0) if t_quad == t_quad else False,
    }


# ------------------------------------------------------------- self-test ---
def _synth_uplink(v_rail=3.3, i_sleep=2e-6, i_tx=0.0439, i_rx=0.0110,
                  fs=20000.0, seed=0):
    """Synthetic captures with a KNOWN energy ratio between SF7 and SF12."""
    rng = np.random.default_rng(seed)
    # Airtimes for PL=12, CR=4/5, BW=125k, explicit header -- the deployed config.
    airtime = {7: 0.061696, 12: 1.712512}
    rows = []
    for sf, at in airtime.items():
        for rep in range(5):
            pre, post, rx1, gap = 0.02, 0.02, 0.0164, 0.01
            total = pre + at + gap + rx1 + gap + 0.1638 + post
            n = int(total * fs)
            t = np.arange(n) / fs
            i = np.full(n, i_sleep)

            def fill(t0, dur, amp):
                a, b = int(t0 * fs), int((t0 + dur) * fs)
                i[a:b] = amp

            fill(pre, at, i_tx)
            fill(pre + at + gap, rx1, i_rx)
            fill(pre + at + gap + rx1 + gap, 0.1638, i_rx)
            i = i + rng.normal(0, 2e-5, n)
            for k in range(n):
                rows.append((sf, rep, t[k], i[k]))
    a = np.array(rows, dtype=float)
    return {'sf': a[:, 0], 'rep': a[:, 1], 't_s': a[:, 2], 'i_a': a[:, 3]}, v_rail


def _synth_discharge(slope=-0.85, curve=0.0, seed=1, n_pts=400):
    """Synthetic discharge with a KNOWN byte-per-coulomb slope and curvature."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 3600 * 24 * 40, n_pts)
    i = np.full(n_pts, 0.02)
    q = np.concatenate([[0.0], np.cumsum(np.diff(t) * (i[1:] + i[:-1]) / 2.0)])
    byte = 255.0 + slope * (q / 1000.0) + curve * (q / 1000.0) ** 2
    byte = np.round(byte + rng.normal(0, 0.3, n_pts))   # quantised to integers
    return {'t_s': t, 'i_a': i, 'byte': byte}, slope


def selftest():
    ok = True
    print("=" * 92)
    print("  SELF-TEST -- the estimators against captures whose answer is known")
    print("=" * 92)

    d, v = _synth_uplink()
    r = analyse_uplinks(d, v)
    # Ground truth for this synthetic device: energy is I*V*duration summed over
    # transmit and both receive windows, identical apart from airtime.
    e7 = 3.3 * (0.0439 * 0.061696 + 0.0110 * (0.0164 + 0.1638))
    e12 = 3.3 * (0.0439 * 1.712512 + 0.0110 * (0.0164 + 0.1638))
    truth = e12 / e7
    got = r.get('k_measured', float('nan'))
    err = abs(got - truth) / truth * 100
    print("  k: recovered %.3f, true %.3f, error %.2f%%" % (got, truth, err))
    if err > 2.0:
        print("  SELF-TEST FAILED: k not recovered within 2%")
        ok = False

    d2, slope = _synth_discharge(slope=-0.85, curve=0.0)
    r2 = analyse_response(d2)
    got_s = r2['slope_byte_per_C'] * 1000.0
    err_s = abs(got_s - slope) / abs(slope) * 100
    print("  response slope: recovered %.4f, true %.4f byte/kC, error %.2f%%"
          % (got_s, slope, err_s))
    if err_s > 2.0:
        print("  SELF-TEST FAILED: response slope not recovered within 2%")
        ok = False
    if r2['curvature_flagged']:
        print("  SELF-TEST FAILED: curvature flagged on a linear response")
        ok = False

    # The half that matters: the test must also be able to FAIL. A response with
    # real curvature has to be caught, or a clean bill of health means nothing.
    d3, _ = _synth_discharge(slope=-0.85, curve=-0.25)
    r3 = analyse_response(d3)
    print("  curved case: quadratic t = %.1f, flagged = %s"
          % (r3['quadratic_t'], r3['curvature_flagged']))
    if not r3['curvature_flagged']:
        print("  SELF-TEST FAILED: curvature NOT flagged on a curved response")
        ok = False

    # How much curvature can this session actually resolve? A test that only
    # catches a gross departure is not evidence of linearity, and this paper of
    # all papers should say what its instrument resolves rather than report a
    # pass. Reported as the departure from the straight line, in byte, at the
    # midpoint of the discharge -- which is the quantity an operator can judge.
    smallest = None
    for c in np.logspace(0, -6, 25):
        d4, _ = _synth_discharge(slope=-0.85, curve=-float(c))
        if analyse_response(d4)['curvature_flagged']:
            smallest = float(c)
    if smallest is not None:
        # byte deviation at mid-span for coefficient c over the charge range.
        q_span = _synth_discharge()[0]
        qmax = float(np.trapezoid(np.full(400, 0.02),
                                  np.linspace(0, 3600 * 24 * 40, 400))) / 1000.0
        dev = abs(smallest) * (qmax ** 2) / 4.0
        print("  resolution: departures from linearity as small as %.2f byte off "
              "the line\n              at mid-discharge are detected "
              "(quadratic coefficient %.1e byte/kC^2)" % (dev, smallest))
    else:
        print("  SELF-TEST WARNING: no curvature level in the sweep was detected")
        ok = False

    print()
    print("  self-test: %s" % ("PASS" if ok else "FAIL"))
    return ok


# ------------------------------------------------------------------ main ---
def main():
    if '--selftest' in sys.argv:
        sys.exit(0 if selftest() else 1)

    if not selftest():
        print("\n  Refusing to analyse real captures with a failing self-test.")
        sys.exit(1)
    print()

    v_rail = 3.3
    for name, path in (('uplink traces', UPLINK_CSV), ('discharge', DISCHARGE_CSV)):
        if not os.path.exists(path):
            print("  not captured yet: %s  (%s)" % (name, os.path.basename(path)))

    if os.path.exists(UPLINK_CSV):
        d = read_csv(UPLINK_CSV)
        r = analyse_uplinks(d, v_rail)
        print("=" * 92)
        print("  UPLINK ENERGY")
        print("=" * 92)
        for sf, a in sorted(r['per_sf'].items(), key=lambda kv: int(kv[0])):
            print("    SF%-3s n=%d  E=%.2f mJ (sd %.2f)  tx %.2f  rx %.2f"
                  % (sf, a['n_reps'], a['E_total_J'] * 1e3, a['E_total_J_sd'] * 1e3,
                     a['E_tx_J'] * 1e3, a['E_rx_J'] * 1e3))
        if 'k_measured' in r:
            print("    k measured        %.2fx" % r['k_measured'])
            print("    k modelled        %.2fx transmit-only, %.2fx with RX"
                  % (r['k_modelled_tx_only'], r['k_modelled_with_rx']))
            print("    nearer            %s" % r['nearer'])
        with io.open(os.path.join(RESULTS, 'bench_uplink_energy.json'), 'w',
                     encoding='utf-8') as fh:
            json.dump(r, fh, indent=2)
        print("    saved results/bench_uplink_energy.json")

    if os.path.exists(DISCHARGE_CSV):
        d = read_csv(DISCHARGE_CSV)
        r = analyse_response(d)
        print("=" * 92)
        print("  BATTERY RESPONSE")
        print("=" * 92)
        for k in sorted(r):
            print("    %-26s %s" % (k, r[k]))
        with io.open(os.path.join(RESULTS, 'bench_response.json'), 'w',
                     encoding='utf-8') as fh:
            json.dump(r, fh, indent=2)
        print("    saved results/bench_response.json")


if __name__ == '__main__':
    main()
