"""
Is the long-gap RSSI correlation real, or an artefact of overlapping pairs?

WHY. `channel_coherence.py` forms, for every uplink i, the pairs (i, j) for the
next eleven j. Each RSSI value therefore enters up to 22 pairs, and the pairs
within a bin share observations. The Fisher-z interval it then reports assumes
n independent pairs. It is not a small effect: the >24 h bin claims n = 4784
from a few thousand uplinks.

That matters for the paper's conclusion rather than only for its error bars.
The regime-map placement rests on the correlation being indistinguishable from
zero at the device's own decision interval. In the published table the
correlation is null in the 120-240 min bin and SIGNIFICANTLY POSITIVE at
240-480 min, >8 h and >24 h. A decaying channel does not do that. Two
explanations are available -- a diurnal component, or intervals that are too
narrow because the pairs are not independent -- and they are distinguishable by
measurement.

WHAT THIS DOES. Recomputes every bin three ways:

  overlapping     the published construction, reproduced as a baseline
  disjoint        greedy non-overlapping pairs: each uplink used at most once
                  per bin, so Fisher-z applies
  device block    bootstrap resampling whole devices, which also absorbs any
                  between-device heterogeneity

and, separately, removes the time of day from RSSI before repeating the
disjoint estimate, which is the diurnal control.

Author: Vullnet Laniku
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
EXPORT = os.path.join(HERE, '..', 'data', 'FIEK_parking_export_83day.xlsx')

BINS_MIN = [(0, 10), (10, 30), (30, 60), (60, 120), (120, 240),
            (240, 480), (480, 1440), (1440, 100000)]
NEIGHBOURS = 12          # the published construction's forward window
N_BOOT = 2000
SEED = 20260822


def label(lo, hi):
    return "%d-%d min" % (lo, hi) if hi < 1440 else ">%d h" % (lo / 60)


def load():
    ev = pd.read_excel(EXPORT, sheet_name='All Events')
    ev['ts'] = pd.to_datetime(ev['timestamp_local'])
    return ev.dropna(subset=['ts', 'rssi']).sort_values('ts')


def overlapping_pairs(ev, col='rssi'):
    """The published construction: every anchor paired with its next 11."""
    out = []
    for d, g in ev.groupby('device_name'):
        g = g.sort_values('ts')
        r = g[col].astype(float).values
        t = g['ts'].values
        for i in range(len(r) - 1):
            for j in range(i + 1, min(i + NEIGHBOURS, len(r))):
                gap = (t[j] - t[i]) / np.timedelta64(1, 'm')
                out.append((d, gap, r[i], r[j]))
    return pd.DataFrame(out, columns=['dev', 'gap_min', 'a', 'b'])


def disjoint_pairs(ev, lo, hi, col='rssi'):
    """Greedy non-overlapping pairs whose gap falls in [lo, hi).

    Walk each device's series; on finding a partner in the window, emit the
    pair and restart past the partner, so no uplink is used twice.
    """
    out = []
    for d, g in ev.groupby('device_name'):
        g = g.sort_values('ts')
        r = g[col].astype(float).values
        t = g['ts'].values
        i, n = 0, len(r)
        while i < n - 1:
            hit = None
            for j in range(i + 1, min(i + NEIGHBOURS, n)):
                gap = (t[j] - t[i]) / np.timedelta64(1, 'm')
                if gap >= hi:
                    break
                if gap >= lo:
                    hit = j
                    break
            if hit is None:
                i += 1
            else:
                out.append((d, r[i], r[hit]))
                i = hit + 1
    return pd.DataFrame(out, columns=['dev', 'a', 'b'])


def fisher_ci(r, n):
    if n < 5 or not np.isfinite(r) or abs(r) >= 1:
        return (float('nan'), float('nan'))
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))


def device_block_ci(df, rng, n_boot=N_BOOT):
    """Resample whole devices with replacement; the CI then respects clustering."""
    devs = df.dev.unique()
    if len(devs) < 2:
        return (float('nan'), float('nan'))
    by = {d: df[df.dev == d] for d in devs}
    est = []
    for _ in range(n_boot):
        pick = rng.choice(devs, size=len(devs), replace=True)
        s = pd.concat([by[d] for d in pick])
        if len(s) < 5 or s.a.std() == 0 or s.b.std() == 0:
            continue
        est.append(float(np.corrcoef(s.a, s.b)[0, 1]))
    if len(est) < 100:
        return (float('nan'), float('nan'))
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def deseasonalise(ev):
    """Remove each device's hour-of-day mean RSSI. The diurnal control."""
    ev = ev.copy()
    ev['rssi'] = ev['rssi'].astype(float)
    ev['hour'] = ev['ts'].dt.hour
    ev['rssi_resid'] = ev.groupby(['device_name', 'hour'])['rssi'].transform(
        lambda x: x - x.mean())
    return ev


def main():
    rng = np.random.default_rng(SEED)
    ev = load()
    ov = overlapping_pairs(ev)
    evd = deseasonalise(ev)

    out = {'_method': {
        'n_uplinks': int(len(ev)), 'forward_window': NEIGHBOURS,
        'n_boot': N_BOOT, 'seed': SEED,
        'note': ('The published construction reuses each uplink in up to '
                 '2*(NEIGHBOURS-1) pairs. Disjoint pairs use each uplink once '
                 'per bin; the device bootstrap additionally respects '
                 'clustering. The diurnal control repeats the disjoint '
                 'estimate on RSSI with each device hour-of-day mean removed.')}}

    print("=" * 100)
    print("  RSSI CORRELATION BY GAP, THREE WAYS   %d uplinks" % len(ev))
    print("=" * 100)
    print("  %-12s %26s %26s %22s" % ("", "overlapping (published)",
                                      "disjoint pairs", "device bootstrap"))
    print("  %-12s %7s %7s %11s %7s %7s %11s %11s"
          % ("gap", "n", "r", "95% CI", "n", "r", "95% CI", "95% CI"))

    rows = {}
    for lo, hi in BINS_MIN:
        lab = label(lo, hi)
        s = ov[(ov.gap_min >= lo) & (ov.gap_min < hi)]
        if len(s) < 25:
            continue
        r_ov = float(np.corrcoef(s.a, s.b)[0, 1])
        ci_ov = fisher_ci(r_ov, len(s))

        dj = disjoint_pairs(ev, lo, hi)
        if len(dj) < 10:
            continue
        r_dj = float(np.corrcoef(dj.a, dj.b)[0, 1])
        ci_dj = fisher_ci(r_dj, len(dj))
        ci_bs = device_block_ci(dj, rng)

        print("  %-12s %7d %+7.3f %5.3f..%.3f %7d %+7.3f %5.3f..%.3f %5.3f..%.3f"
              % (lab, len(s), r_ov, ci_ov[0], ci_ov[1],
                 len(dj), r_dj, ci_dj[0], ci_dj[1], ci_bs[0], ci_bs[1]))

        rows[lab] = {
            'overlapping': {'n': int(len(s)), 'r': r_ov, 'ci': list(ci_ov),
                            'excludes_zero': bool(ci_ov[0] > 0 or ci_ov[1] < 0)},
            'disjoint': {'n': int(len(dj)), 'r': r_dj, 'ci': list(ci_dj),
                         'excludes_zero': bool(ci_dj[0] > 0 or ci_dj[1] < 0)},
            'device_bootstrap': {'ci': list(ci_bs),
                                 'excludes_zero': bool(
                                     np.isfinite(ci_bs[0]) and
                                     (ci_bs[0] > 0 or ci_bs[1] < 0))}}
    out['bins'] = rows

    # ------------------------------------------------ the diurnal control ---
    print()
    print("=" * 100)
    print("  DIURNAL CONTROL   disjoint pairs, each device's hour-of-day mean "
          "removed from RSSI")
    print("=" * 100)
    print("  %-12s %8s %9s %16s %16s" % ("gap", "n", "r", "95% CI", "r before"))
    diur = {}
    for lo, hi in BINS_MIN:
        lab = label(lo, hi)
        if lab not in rows:
            continue
        dj = disjoint_pairs(evd, lo, hi, col='rssi_resid')
        if len(dj) < 10:
            continue
        r = float(np.corrcoef(dj.a, dj.b)[0, 1])
        ci = fisher_ci(r, len(dj))
        before = rows[lab]['disjoint']['r']
        print("  %-12s %8d %+9.3f %7.3f..%.3f %15.3f"
              % (lab, len(dj), r, ci[0], ci[1], before))
        diur[lab] = {'n': int(len(dj)), 'r': r, 'ci': list(ci),
                     'r_before': before}
    out['diurnal_control'] = diur

    # ------------------------------------------- the SF stratum control ----
    # Sec. V-D of the manuscript reports the lag-1 coefficient WITHIN a
    # spreading-factor stratum and states that pooling strata dilutes it. This
    # gap-binned analysis pools them, which is the same mistake in the other
    # direction: ADR moves the device between strata, RSSI differs
    # systematically by stratum, and two uplinks that happen to share a stratum
    # will look correlated for that reason alone. Restricting to pairs whose
    # two ends share a spreading factor removes it.
    print()
    print("=" * 100)
    print("  SPREADING-FACTOR CONTROL   disjoint pairs whose two ends share an SF")
    print("=" * 100)
    print("  %-12s %8s %9s %16s %14s" % ("gap", "n", "r", "95% CI", "r pooled"))
    sfc = {}
    for lo, hi in BINS_MIN:
        lab = label(lo, hi)
        if lab not in rows:
            continue
        parts = []
        for sf, g in ev.groupby('spreading_factor'):
            if len(g) < 30:
                continue
            parts.append(disjoint_pairs(g, lo, hi))
        if not parts:
            continue
        dj = pd.concat(parts)
        if len(dj) < 10 or dj.a.std() == 0 or dj.b.std() == 0:
            continue
        r = float(np.corrcoef(dj.a, dj.b)[0, 1])
        ci = fisher_ci(r, len(dj))
        print("  %-12s %8d %+9.3f %7.3f..%.3f %13.3f"
              % (lab, len(dj), r, ci[0], ci[1], rows[lab]['disjoint']['r']))
        sfc[lab] = {'n': int(len(dj)), 'r': r, 'ci': list(ci),
                    'r_pooled': rows[lab]['disjoint']['r'],
                    'excludes_zero': bool(ci[0] > 0 or ci[1] < 0)}
    out['sf_stratum_control'] = sfc

    # -------------------------------------- the within-device/SF control ---
    # Every estimate above correlates pairs POOLED ACROSS DEVICES. Pairs are
    # formed within a device, but the correlation is taken over the pooled set,
    # so devices sitting at different mean RSSI -- which they do, being at
    # different distances from the one gateway -- manufacture correlation that
    # has nothing to do with persistence over time. The spreading factor does
    # the same thing within a device, because RSSI differs systematically by
    # stratum and ADR moves the device between strata.
    #
    # The quantity the regime map needs is how much of a device's own deviation
    # from its own mean, in its own stratum, persists across a decision
    # interval. That means centring on (device, SF) before pairing.
    ev_c = ev.copy()
    ev_c['rssi'] = ev_c['rssi'].astype(float)
    ev_c['rssi'] = ev_c.groupby(['device_name', 'spreading_factor'])['rssi'] \
                       .transform(lambda x: x - x.mean())
    print()
    print("=" * 100)
    print("  WITHIN-DEVICE, WITHIN-SF CONTROL   disjoint pairs on RSSI centred "
          "on its own (device, SF) mean")
    print("=" * 100)
    print("  %-12s %8s %9s %16s %12s %12s"
          % ("gap", "n", "r", "95% CI", "r pooled", "r same-SF"))
    cen = {}
    for lo, hi in BINS_MIN:
        lab = label(lo, hi)
        if lab not in rows:
            continue
        dj = disjoint_pairs(ev_c, lo, hi)
        if len(dj) < 10 or dj.a.std() == 0 or dj.b.std() == 0:
            continue
        r = float(np.corrcoef(dj.a, dj.b)[0, 1])
        ci = fisher_ci(r, len(dj))
        ci_bs = device_block_ci(dj, rng)
        sf_r = out.get('sf_stratum_control', {}).get(lab, {}).get('r', float('nan'))
        print("  %-12s %8d %+9.3f %7.3f..%.3f %11.3f %11.3f"
              % (lab, len(dj), r, ci[0], ci[1],
                 rows[lab]['disjoint']['r'], sf_r))
        cen[lab] = {'n': int(len(dj)), 'r': r, 'ci': list(ci),
                    'ci_device_bootstrap': list(ci_bs),
                    'r_pooled': rows[lab]['disjoint']['r'], 'r_same_sf': sf_r,
                    'excludes_zero': bool(ci[0] > 0 or ci[1] < 0)}
    out['within_device_sf_control'] = cen

    # ------------------------------------------------------- the reading ---
    print()
    print("=" * 100)
    print("  READING")
    print("=" * 100)
    ov_sig = [k for k, v in rows.items() if v['overlapping']['excludes_zero']]
    dj_sig = [k for k, v in rows.items() if v['disjoint']['excludes_zero']]
    bs_sig = [k for k, v in rows.items() if v['device_bootstrap']['excludes_zero']]
    print("  bins whose CI excludes zero:")
    print("    overlapping (published) : %d  %s" % (len(ov_sig), ov_sig))
    print("    disjoint pairs          : %d  %s" % (len(dj_sig), dj_sig))
    print("    device bootstrap        : %d  %s" % (len(bs_sig), bs_sig))
    sf_sig = [k for k, v in out.get('sf_stratum_control', {}).items()
              if v['excludes_zero']]
    cen_sig = [k for k, v in out.get('within_device_sf_control', {}).items()
               if v['excludes_zero']]
    print("    SF-stratified disjoint  : %d  %s" % (len(sf_sig), sf_sig))
    print("    within device and SF    : %d  %s" % (len(cen_sig), cen_sig))
    out['summary'] = {'sig_overlapping': ov_sig, 'sig_disjoint': dj_sig,
                      'sig_device_bootstrap': bs_sig,
                      'sig_sf_stratified': sf_sig,
                      'sig_within_device_sf': cen_sig}

    med = ev.groupby('device_name')['ts'].apply(
        lambda x: x.sort_values().diff().dt.total_seconds().median() / 60).median()
    out['median_gap_min'] = float(med)
    print()
    print("  median inter-uplink gap, median over devices: %.1f min" % med)

    path = os.path.join(RESULTS, 'channel_coherence_control.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print()
    print("Saved %s" % path)


if __name__ == '__main__':
    main()
