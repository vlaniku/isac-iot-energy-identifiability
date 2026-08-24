"""
Two-year workload comparison on the same physical devices.

INVALID AS A RATE COMPARISON, and kept for that reason. The 2026 side
reads the application export, which `deploy_workload_control.py` shows
captures only 44-88% of traffic and does so unevenly across devices. Every
per-day rate below is therefore biased downward by an unknown, device-
specific factor, and the 2024-to-2026 decline this script prints cannot be
separated from the ingestion loss. Do not quote the rates or the decline.

What survives the bias is the RANK comparison in part D: capture loss
shifts levels, and the Spearman correlation across devices is the one
reading that does not depend on the levels being right. That is the
assumption the natural experiment rests on, which is why the script is
still here.

The public dataset behind Kadriu et al. (2024-06-15 -> 2024-07-16) and our 2026
export (2026-04-21 -> 2026-07-13) share four DevEUIs. That gives a two-year
trajectory of parking-event workload on identical hardware in identical bays,
which bears on three things:

  1. whether the workload used in the depletion analysis is stable over the
     device's life, or whether 2026 is unrepresentative;
  2. whether bay turnover is a stable device property (rank stability), which
     is the assumption underlying the natural experiment;
  3. whether event rate DECLINED before a device fell silent -- a possible
     degradation signature, though confounded, since a lost uplink and an
     absent car look identical in a receive-only log. RSSI and SNR trends are
     reported alongside to separate them where possible.

Comparability care:
  * 2024 records are state changes only; the 2026 comparison therefore uses
    event_type == 'status_change' and excludes heartbeats.
  * windows differ in length and season, so rates are per-day and three
    windows are reported: full, season-matched, and all-devices-alive.

Author: Vullnet Laniku
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

# The 2024 window is the public dataset released with Kadriu et al. (2024).
# It was read from a temporary working directory when this analysis was first
# run, which made the script unrunnable once that directory was cleared; it
# now lives in data/ under a name that says what it is.
OLD = '../data/kadriu2024_public_events.xlsx'
NEW = '../data/FIEK_parking_export_83day.xlsx'


def load_old():
    d = pd.read_excel(OLD)
    d['eui'] = d.name.str.extract(r'_([0-9A-Fa-f]{16})')[0].str.lower()
    d['ts'] = pd.to_datetime(d.timestamp)
    return d


def load_new():
    ev = pd.read_excel(NEW, sheet_name='All Events')
    fs = pd.read_excel(NEW, sheet_name='Fleet Summary')
    name2eui = dict(zip(fs.device_name, fs.dev_eui.str.lower()))
    ev['eui'] = ev.device_name.map(name2eui)
    ev['ts'] = pd.to_datetime(ev.timestamp_local)
    return ev.dropna(subset=['ts'])


def rate(df, eui, t0, t1):
    s = df[(df.eui == eui) & (df.ts >= t0) & (df.ts <= t1)]
    days = (t1 - t0).total_seconds() / 86400.0
    return len(s), len(s) / days if days > 0 else np.nan


def main():
    old, new = load_old(), load_new()
    new_sc = new[new.event_type == 'status_change']
    common = sorted(set(old.eui.unique()) & set(new.eui.dropna().unique()))
    label = {e: new[new.eui == e].device_name.iloc[0][-11:] for e in common}
    last = {e: new[new.eui == e].ts.max() for e in common}

    o0, o1 = old.ts.min(), old.ts.max()
    n0, n1 = new_sc.ts.min(), new_sc.ts.max()
    print("=" * 100)
    print("  TWO-YEAR WORKLOAD COMPARISON   %d devices present in both records" % len(common))
    print("=" * 100)
    print("  2024 window: %s .. %s  (%.0f days, %d state changes)"
          % (o0.date(), o1.date(), (o1 - o0).total_seconds() / 86400, len(old)))
    print("  2026 window: %s .. %s  (%.0f days, %d state changes)"
          % (n0.date(), n1.date(), (n1 - n0).total_seconds() / 86400, len(new_sc)))

    # ---- window A: full records ------------------------------------------
    print("\n  A. FULL WINDOWS (per-device, events/day)")
    print("     %-12s %10s %10s %9s %14s" % ("device", "2024", "2026", "ratio", "last report"))
    rows = {}
    for e in common:
        _, r24 = rate(old, e, o0, o1)
        _, r26 = rate(new_sc, e, n0, last[e])
        rows[e] = {'label': label[e], 'rate_2024': r24, 'rate_2026_full': r26,
                   'ratio_full': r26 / r24 if r24 else np.nan,
                   'last': str(last[e].date())}
        print("     %-12s %10.2f %10.2f %9.2f %14s"
              % (label[e], r24, r26, r26 / r24, last[e].date()))

    # ---- window B: all devices alive -------------------------------------
    alive_end = min(last.values())
    print("\n  B. ALL-ALIVE WINDOW 2026 (%s .. %s, %.0f days) vs 2024"
          % (n0.date(), alive_end.date(), (alive_end - n0).total_seconds() / 86400))
    print("     %-12s %10s %10s %9s" % ("device", "2024", "2026", "ratio"))
    for e in common:
        _, r24 = rate(old, e, o0, o1)
        _, r26 = rate(new_sc, e, n0, alive_end)
        rows[e]['rate_2026_alive'] = r26
        rows[e]['ratio_alive'] = r26 / r24 if r24 else np.nan
        print("     %-12s %10.2f %10.2f %9.2f" % (label[e], r24, r26, r26 / r24))

    # ---- window C: season matched ----------------------------------------
    s0 = pd.Timestamp('2026-06-15'); s1 = pd.Timestamp('2026-07-16')
    print("\n  C. SEASON-MATCHED (15 Jun - 16 Jul, both years)")
    print("     %-12s %10s %10s %9s %s" % ("device", "2024", "2026", "ratio", "note"))
    for e in common:
        _, r24 = rate(old, e, o0, o1)
        cap = min(s1, last[e])
        n_ev, r26 = rate(new_sc, e, s0, cap)
        note = "" if last[e] >= s1 else "device silent from %s" % last[e].date()
        rows[e]['rate_2026_season'] = r26
        print("     %-12s %10.2f %10.2f %9s %s"
              % (label[e], r24, r26, ("%.2f" % (r26 / r24)) if r24 else "n/a", note))

    # ---- rank stability ---------------------------------------------------
    a = np.array([rows[e]['rate_2024'] for e in common])
    b = np.array([rows[e]['rate_2026_alive'] for e in common])
    rho, p = stats.spearmanr(a, b)
    pr, pp = stats.pearsonr(a, b)
    print("\n  D. IS BAY TURNOVER A STABLE DEVICE PROPERTY?")
    print("     Spearman rho = %+.3f (p = %.3f), Pearson r = %+.3f (p = %.3f), n = %d"
          % (rho, p, pr, pp, len(common)))
    print("     2024 rate range %.2f-%.2f /day ; 2026 %.2f-%.2f /day"
          % (a.min(), a.max(), b.min(), b.max()))
    if rho > 0.5:
        print("     => ordering broadly preserved: turnover is a bay property, which is")
        print("        the assumption the natural experiment rests on.")
    else:
        print("     => ordering NOT preserved. The natural experiment's assumption that")
        print("        workload is a stable device property does not hold across years.")

    # ---- pre-silence trend ------------------------------------------------
    print("\n  E. DID EVENT RATE OR LINK QUALITY DECLINE BEFORE SILENCE?")
    print("     %-12s %12s %12s %11s %11s" % (
        "device", "rate first½", "rate last½", "RSSI first½", "RSSI last½"))
    for e in common:
        g = new[(new.eui == e)].sort_values('ts')
        gs = new_sc[new_sc.eui == e].sort_values('ts')
        if len(gs) < 20:
            continue
        mid = gs.ts.min() + (gs.ts.max() - gs.ts.min()) / 2
        d1 = (mid - gs.ts.min()).total_seconds() / 86400
        d2 = (gs.ts.max() - mid).total_seconds() / 86400
        r1 = (gs.ts <= mid).sum() / d1
        r2 = (gs.ts > mid).sum() / d2
        q1 = g[g.ts <= mid].rssi.astype(float).mean()
        q2 = g[g.ts > mid].rssi.astype(float).mean()
        rows[e]['rate_first_half'] = float(r1); rows[e]['rate_last_half'] = float(r2)
        rows[e]['rssi_first_half'] = float(q1); rows[e]['rssi_last_half'] = float(q2)
        print("     %-12s %12.2f %12.2f %11.2f %11.2f" % (label[e], r1, r2, q1, q2))
    print("     A decline in received events could be fewer cars OR more lost uplinks;")
    print("     a simultaneous RSSI decline would favour the second. Read both columns.")

    with open('../results/two_year_workload_results.json', 'w') as f:
        json.dump({'devices': {label[e]: rows[e] for e in common},
                   'rank_spearman': {'rho': float(rho), 'p': float(p)},
                   'rank_pearson': {'r': float(pr), 'p': float(pp)}}, f, indent=2)
    print("\nSaved ../results/two_year_workload_results.json")


if __name__ == '__main__':
    main()
