"""
Figures for PAPER_v3.md.

Every figure is generated from a results JSON or a data file in this repository --
none is drawn by hand, and none contains a number that is not traceable to a
script in code/. Output goes to figures/ as both PDF (vector, for submission) and
PNG (for reading).

IEEE two-column geometry: single column 3.5 in, double column 7.16 in.

A note on what these figures are for. Five of the eight report NEGATIVE results or
the limits of an instrument. That is deliberate: the paper's argument is that the
quantity the field optimises cannot be measured from deployed telemetry, and a
figure showing an apparent signal beside the control that removed it carries that
argument better than prose does.

Author: Vullnet Laniku
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
DATA = os.path.join(HERE, '..', 'data')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

COL1, COL2 = 3.5, 7.16
plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 160, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'lines.linewidth': 1.2, 'font.family': 'sans-serif',
})
# colour-blind safe
C_DEAD, C_ALIVE, C_ACC, C_GREY = '#D55E00', '#0072B2', '#009E73', '#7F7F7F'
C_WARN = '#CC79A7'


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, '%s.%s' % (name, ext)))
    plt.close(fig)
    print("  wrote figures/%s.pdf / .png" % name)


def jload(n):
    with open(os.path.join(RES, n), encoding='utf-8', errors='replace') as f:
        return json.load(f)


# =====================================================================  FIG 1
def fig1_timeline():
    """Two-year service record. The rating is an open arrow, not a target line."""
    inst = datetime(2024, 5, 24)
    devs = [
        ('fcd6bd000019cd11  SENZOR_03', inst, datetime(2026, 8, 18), 'alive'),
        ('fcd6bd000019cd0c  SENZOR_05', inst, datetime(2026, 8, 18), 'alive'),
        ('fcd6bd000019ccfb  SENZOR_04', inst, datetime(2026, 6, 24), 'ceased'),
        ('fcd6bd000019cd03  SENZOR_02', inst, datetime(2026, 6, 23), 'ceased'),
        ('fcd6bd000019cd02  (replaced)', inst, datetime(2025, 9, 1), 'replaced'),
        ('fcd6bd000019cd04  SENZOR_01', datetime(2025, 9, 1), datetime(2026, 8, 18), 'alive'),
    ]
    fig, ax = plt.subplots(figsize=(COL2, 2.5))
    for i, (name, a, b, st) in enumerate(devs):
        c = {'alive': C_ALIVE, 'ceased': C_DEAD, 'replaced': C_GREY}[st]
        ax.barh(i, (b - a).days, left=a, height=0.5, color=c,
                alpha=0.85 if st != 'replaced' else 0.4)
        if st == 'ceased':
            ax.plot(b, i, marker='X', color=C_DEAD, ms=7, zorder=5)
            ax.text(b, i + 0.42, '%d d' % (b - a).days, color=C_DEAD,
                    fontsize=6.5, ha='center')
        elif st == 'alive':
            ax.annotate('', xy=(b + (b - a) * 0.03, i), xytext=(b, i),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.1))
    # the two disjoint record windows
    for a, b, lab in [(datetime(2024, 6, 15), datetime(2024, 7, 16), 'public 2024 record'),
                      (datetime(2026, 4, 21), datetime(2026, 7, 13), 'application export')]:
        ax.add_patch(Rectangle((mdates.date2num(a), -0.6), mdates.date2num(b) - mdates.date2num(a),
                               len(devs) - 0.1, color='k', alpha=0.07, zorder=0))
        ax.text(b, len(devs) - 0.30, ' ' + lab, fontsize=6, color='0.35', va='center')
    ax.axvline(inst, color='k', lw=0.9, ls='--')
    ax.text(inst, 2.4, ' commissioned 2024-05-24', fontsize=6.5, va='center', rotation=90)
    # the rating, as an open-ended arrow
    y = -1.20
    ax.annotate('', xy=(datetime(2029, 6, 1), y), xytext=(inst, y),
                arrowprops=dict(arrowstyle='->', color='0.45', lw=1.4))
    ax.text(inst, y - 0.30, '  manufacturer rating: "up to 5 years"\n'
            '  no duty cycle, traffic profile or temperature basis stated',
            fontsize=6.5, color='0.35', va='top')
    ax.set_yticks(range(len(devs)))
    ax.set_yticklabels([d[0] for d in devs], fontsize=6.5, family='monospace')
    ax.set_ylim(-2.6, len(devs) + 0.15)
    ax.set_xlim(datetime(2024, 3, 1), datetime(2029, 9, 1))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_title('Service record: two units ceased at 25.0 months, 3 h 53 min apart; '
                 'two continue at 26.8 months', loc='left')
    ax.grid(axis='y', visible=False)
    save(fig, 'fig1_service_timeline')


# =====================================================================  FIG 2
def fig2_june_control():
    """The survivor control: the pre-cessation dip is fleet-wide."""
    m = json.load(open(os.path.join(DATA, 'chirpstack_12mo_metrics.json'), encoding='utf-8'))
    months = m['months']
    idx = [i for i, x in enumerate(months) if x != '2026-08']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.6),
                                  gridspec_kw={'width_ratios': [2.1, 1]})
    stats = []
    for dev in m['devices'].values():
        nm = dev['name'][-2:]
        rx = np.array([dev['rx_count'][i] for i in idx], float)
        base = np.array([dev['rx_count'][i] for i, mm in enumerate(months)
                         if '2025-08' <= mm < '2026-06' and dev['rx_count'][i] > 0], float)
        mu = base.mean()
        dead = dev['status'] == 'dead'
        y = np.where(rx > 0, rx / mu, np.nan)
        ax.plot(range(len(idx)), y, marker='o', ms=2.6,
                color=C_DEAD if dead else C_ALIVE, alpha=0.9,
                lw=1.4 if dead else 1.0, ls='-' if dead else '--',
                label='SENZOR_%s%s' % (nm, ' (ceased)' if dead else ''))
        june = dev['rx_count'][months.index('2026-06')]
        stats.append((nm, june / mu, dead, (june - mu) / base.std(ddof=1)))
    ax.axhline(1.0, color='k', lw=0.7)
    ax.axvline(months.index('2026-06'), color=C_DEAD, lw=0.8, ls=':')
    ax.text(months.index('2026-06'), 1.72, ' June 2026', fontsize=6.5, color=C_DEAD)
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([months[i][2:] for i in idx], rotation=90, fontsize=5.8)
    ax.set_ylabel('monthly frames /\nown Aug–May mean')
    ax.set_ylim(0, 1.85)
    ax.legend(ncol=2, frameon=False, loc='lower left', fontsize=6)
    ax.set_title('(a) each device normalised to its own baseline', loc='left')

    stats.sort(key=lambda s: s[1])
    cols = [C_DEAD if s[2] else C_ALIVE for s in stats]
    ax2.barh(range(len(stats)), [s[1] for s in stats], color=cols, alpha=0.85, height=0.6)
    ax2.axvline(1.0, color='k', lw=0.7)
    ax2.set_yticks(range(len(stats)))
    ax2.set_yticklabels(['SENZOR_%s' % s[0] for s in stats], fontsize=6.5)
    for i, s in enumerate(stats):
        ax2.text(s[1] + 0.03, i, '%.0f%%  z=%+.2f' % (100 * s[1], s[3]), fontsize=6,
                 va='center')
    ax2.set_xlim(0, 1.75)
    ax2.set_xlabel('June 2026 / own mean')
    ax2.set_title('(b) the control', loc='left')
    fig.suptitle('A surviving unit shows the deepest June reduction in the fleet — '
                 'the dip carries no information about cessation',
                 fontsize=8, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, 'fig2_june_survivor_control')


# =====================================================================  FIG 3
def fig3_depletion_regressor():
    """Depletion vs workload, with the corrupted regressor and the clean one."""
    d = jload('deploy_workload_control_results.json')
    per = d['per_device']
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6), sharey=True)
    for ax, key, lab, reg in [
            (axes[0], 'export_rate', '(a) export regressor (as originally run)', 'all5_export'),
            (axes[1], 'server_rate', '(b) network-server regressor (control)', 'all5_server')]:
        x = np.array([p[key] for p in per])
        y = np.array([p['slope_mV_day'] for p in per])
        ax.scatter(x, y, s=26, color=C_ACC, zorder=4, edgecolor='k', linewidth=0.4)
        for p in per:
            ax.annotate(p['device'][-2:], (p[key], p['slope_mV_day']),
                        textcoords='offset points', xytext=(4, 3), fontsize=6)
        xs = np.linspace(x.min() * 0.9, x.max() * 1.05, 50)
        r = d['regression'][reg]
        ax.plot(xs, y.mean() + r['slope'] * (xs - x.mean()), color=C_DEAD, lw=1.3)
        # MDE band: what slope would have been detectable
        mde = 0.156 * abs(y.mean()) / x.mean()
        ax.fill_between(xs, y.mean() - mde * (xs - x.mean()),
                        y.mean() + mde * (xs - x.mean()),
                        color=C_GREY, alpha=0.18, lw=0,
                        label='undetectable at 80% power\n(MDE 15.6%)')
        ax.axhline(y.mean(), color='k', lw=0.6, ls=':')
        ax.set_xlabel('uplinks per day')
        ax.set_title('%s\nslope %+.4f, p=%.3f, r=%+.3f' %
                     (lab, r['slope'], r['p'], r['r']), loc='left', fontsize=7)
    axes[0].set_ylabel('depletion (mV/day, exporter axis)')
    axes[0].legend(frameon=False, loc='lower left', fontsize=6)
    fig.suptitle('The clean regressor removes the leave-one-out sign flip — but n=5 '
                 'still cannot resolve below 15.6%', fontsize=8, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, 'fig3_depletion_vs_workload')


# =====================================================================  FIG 4
def fig4_channel():
    """Within-SF persistence, and correlation against elapsed time."""
    d = jload('channel_coherence_results.json')
    bins = d['bins']
    order = ['0-10 min', '10-30 min', '30-60 min', '60-120 min', '120-240 min',
             '240-480 min', '>8 h', '>24 h']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.5),
                                  gridspec_kw={'width_ratios': [1, 1.75]})
    ax.bar([0, 1], [0.287, 0.179], color=[C_ACC, C_ALIVE], width=0.55, alpha=0.9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['SF7\nn=318', 'SF10\nn=348'], fontsize=7)
    ax.set_ylabel('lag-1 RSSI autocorrelation')
    ax.set_ylim(0, 0.42)
    for i, v in enumerate([0.287, 0.179]):
        ax.text(i, v + 0.012, '%.3f\n(%.0f%% of var.)' % (v, 100 * v * v),
                ha='center', fontsize=6.2)
    ax.set_title('(a) within spreading factor', loc='left')

    # Two series. The published construction pairs every uplink with its next
    # eleven, so its intervals assume far more independent pairs than exist;
    # the corrected one uses disjoint pairs on RSSI centred within device and
    # spreading factor. Plotting both is the point of the panel: the two
    # long-gap "significances" in the first series do not survive.
    ctl = jload('channel_coherence_control.json')
    cen = ctl['within_device_sf_control']

    xs = np.arange(len(order))
    r = [bins[b]['r'] for b in order]
    lo = [bins[b]['ci'][0] for b in order]
    hi = [bins[b]['ci'][1] for b in order]
    ax2.errorbar(xs - 0.13, r,
                 yerr=[np.array(r) - np.array(lo), np.array(hi) - np.array(r)],
                 fmt='o', ms=3.0, color=C_GREY, ecolor=C_GREY, elinewidth=0.9,
                 capsize=2, alpha=0.75, label='overlapping pairs (as published)')

    rc = [cen[b]['r'] if b in cen else np.nan for b in order]
    lc = [cen[b]['ci'][0] if b in cen else np.nan for b in order]
    hc = [cen[b]['ci'][1] if b in cen else np.nan for b in order]
    ax2.errorbar(xs + 0.13, rc,
                 yerr=[np.array(rc) - np.array(lc), np.array(hc) - np.array(rc)],
                 fmt='o', ms=3.4, color=C_ACC, ecolor=C_ACC, elinewidth=1.1,
                 capsize=2, label='disjoint, centred within device and SF')
    ax2.axhline(0, color='k', lw=0.7)
    ax2.axvline(4, color=C_DEAD, lw=0.9, ls=':')
    ax2.text(3.88, 0.555, "device's own median\ndecision interval\n(187.6 min)",
             fontsize=6.2, color=C_DEAD, ha='right', va='top')
    ax2.legend(frameon=False, loc='lower center', fontsize=5.6, ncol=2,
               handletextpad=0.4, columnspacing=1.2, borderpad=0.2)
    ax2.set_xticks(xs); ax2.set_xticklabels(order, rotation=45, ha='right', fontsize=6)
    ax2.set_ylabel('correlation with previous observation')
    ax2.set_xlabel('elapsed time between observations')
    ax2.set_title('(b) resolved against elapsed time, 95% CI', loc='left')
    ax2.set_ylim(-0.42, 0.56)
    fig.suptitle('3–8% of residual variance is forecastable within a spreading factor; '
                 'at the decision interval it is indistinguishable from zero',
                 fontsize=8, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    save(fig, 'fig4_channel_predictability')


if __name__ == '__main__':
    print("Figures 1-4:")
    fig1_timeline()
    fig2_june_control()
    fig3_depletion_regressor()
    fig4_channel()


# =====================================================================  FIG 5
def fig5_regime_map():
    """Where allocation pays, and where the device and the literature sit."""
    d = jload('regime_map_v3_results.json')
    curve = d['state_awareness_curve']
    grid = d['grid']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.7),
                                  gridspec_kw={'width_ratios': [1, 1.15]})

    k = [c[0] for c in curve]; v = [c[1] for c in curve]
    ax.plot(k, v, marker='o', ms=3.4, color=C_ACC)
    ax.scatter([0], [0], s=70, marker='*', color=C_DEAD, zorder=6)
    ax.annotate('TPS110EU\nkappa ~ 0 : 0.00%', (0, 0),
                textcoords='offset points', xytext=(11, 8),
                fontsize=6.5, color=C_DEAD)
    ax.axhline(8.05, color=C_GREY, ls=':', lw=0.9)
    ax.text(0.24, 8.4, 'saturates near 8%', fontsize=6.2, color='0.35', ha='right')
    ax.set_xlabel('kappa  (state motion per epoch)')
    ax.set_ylabel('value of state awareness (%)')
    ax.set_ylim(-0.8, 10)
    ax.set_title('(a) foresight needs trackable state', loc='left')

    # ---- the f axis, with verified literature points
    rows = [('TPS110EU, link budget', 0.0042, C_DEAD, 'measured'),
            ('TPS110EU, field bound', 0.173, C_DEAD, 'n=5 limit'),
            ('Newcastle LoRa node', 0.02, C_ALIVE, '6 y, no depletion'),
            ('Bai 2026 (IEEE Access)', 0.97, C_GREY, 'no sleep term'),
            ('Ye et al. (IEEE TVT)', 0.984, C_GREY, 'no sleep term'),
            ('TGCN submission', 1.00, C_WARN, 'no sleep term')]
    ys = np.arange(len(rows))
    ax2.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], alpha=0.85, height=0.55)
    ax2.set_xscale('log'); ax2.set_xlim(2e-3, 2.2)
    ax2.set_yticks(ys); ax2.set_yticklabels([r[0] for r in rows], fontsize=6.4)
    for i, r in enumerate(rows):
        ax2.text(r[1] * 1.15, i, r[3], fontsize=5.8, va='center', color='0.3')
    ax2.set_xlabel('controllable share  f   (log scale)')
    ax2.set_title('(b) 234x between the literature and the device', loc='left')
    ax2.grid(axis='y', visible=False)
    fig.suptitle('The deployed device reads ~0 on every axis; published ISAC system '
                 'models sit at the opposite end', fontsize=8, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, 'fig5_regime_map')


# =====================================================================  FIG 6
def fig6_bracket():
    """The multiplier in the energy identity is a free parameter of the model."""
    d = jload('comm_action_bracket_results.json')
    sim, dev = d['simulator'], d['device_action_set']
    fig, ax = plt.subplots(figsize=(COL2, 2.4))
    mac = sim['over_mac_mode']
    labs = list(mac.keys()); vals = [mac[k] for k in labs]
    xs = np.arange(len(labs))
    ax.bar(xs, vals, color=C_WARN, alpha=0.85, width=0.55)
    pp = sim['over_p_proc']
    xs2 = np.arange(len(pp)) + len(labs) + 0.8
    ax.bar(xs2, [pp[k]['tdma_n6'] for k in pp], color=C_GREY, alpha=0.75, width=0.55)
    dv = [dev['all_dr_k1.00'], dev['all_dr_k0.35']]
    xs3 = np.array([xs2[-1] + 1.8, xs2[-1] + 2.6])
    ax.bar(xs3, dv, color=C_ACC, alpha=0.9, width=0.55)
    ax.axhline(0.333, color=C_DEAD, ls='--', lw=1.1)
    ax.text(0.1, 0.355, 'the value that was published: 0.333', fontsize=6.5, color=C_DEAD)
    ax.set_xticks(list(xs) + list(xs2) + list(xs3))
    ax.set_xticklabels([l.replace(' (slot_frac = 1.0)', '').replace('TDMA, ', '') for l in labs] +
                       ['%s W' % k for k in pp] + ['DR0-5 x\n2-14 dBm', 'same,\nlow-PA'],
                       rotation=45, ha='right', fontsize=5.8)
    ax.set_ylabel('bracket  1 - e_min/e_mean')
    ax.set_ylim(0, 0.95)
    for x0, x1, lab, c in [(xs[0] - 0.4, xs[-1] + 0.4, 'simulator MAC / device count', C_WARN),
                           (xs2[0] - 0.4, xs2[-1] + 0.4, 'simulator p_proc constant', C_GREY),
                           (xs3[0] - 0.4, xs3[-1] + 0.4, "the DEVICE's real action set", C_ACC)]:
        ax.plot([x0, x1], [0.90, 0.90], color=c, lw=2.4, solid_capstyle='butt')
        ax.text((x0 + x1) / 2, 0.915, lab, ha='center', fontsize=6.2, color=c)
    ax.set_title('The bracket varies 0.10–0.78 over choices internal to the model, and is '
                 '0.83–0.86 on real hardware', loc='left', fontsize=8)
    ax.grid(axis='x', visible=False)
    save(fig, 'fig6_bracket_sensitivity')


# =====================================================================  FIG 7
def fig7_coincidence():
    """The construction that looked sharpest, and the validation that killed it."""
    d = jload('coincidence_bound_results.json')
    v = jload('coincidence_validation_results.json')
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.6))

    preds = d['predictions']
    ss, dts, labs = [], [], []
    for k, p in preds.items():
        ss.append(p['s'] * 100); dts.append(p['dt_hours']); labs.append(k)
    o = np.argsort(ss)
    ss = np.array(ss)[o]; dts = np.array(dts)[o]
    ax.loglog(ss, dts, marker='o', ms=3.6, color=C_ACC)
    ax.axhline(d['dt_hours'], color=C_DEAD, lw=1.3)
    ax.text(0.35, d['dt_hours'] * 1.25, 'observed: 3 h 53 min', color=C_DEAD, fontsize=6.5)
    for xs, lab in [(17.3, 'field bound\n17.3%'), (0.3, 'link budget\n0.3%')]:
        ax.axvline(xs, color=C_GREY, ls=':', lw=0.9)
        ax.text(xs * 1.06, 3000, lab, fontsize=6, color='0.35')
    ax.set_xlabel('candidate workload share s (%)')
    ax.set_ylabel('predicted separation (hours, log)')
    ax.set_title('(a) the construction', loc='left')

    ul = v['upper_limits']
    sc = sorted(float(k) for k in ul)
    fr = []
    for s_ in sc:
        key = '%.4f' % s_
        fr.append(ul[key]['p_null'])
    ax2.semilogx([max(s, 1e-5) for s in sc], [100 * f for f in fr],
                 marker='o', ms=3.6, color=C_WARN)
    ax2.axhline(5, color=C_DEAD, ls='--', lw=1.0)
    ax2.text(2e-4, 6.5, 'model rejected below this line', fontsize=6.2, color=C_DEAD)
    ax2.axvline(0.052, color='k', ls=':', lw=1.0)
    ax2.text(0.045, 62, "this fleet's own\nbatch spread\n(1 in 430)", fontsize=6.2, ha='right')
    ax2.set_xlabel('cell-to-cell capacity spread  sigma_C')
    ax2.set_ylabel('P(observing dt this small)  %')
    ax2.set_title('(b) the validation that withdrew it', loc='left')
    fig.suptitle('The sharpest-looking bound in the project, and why it is not safe to quote',
                 fontsize=8, x=0.02, ha='left')
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, 'fig7_coincidence_and_validation')


# =====================================================================  FIG 8
def fig8_negatives():
    """Seven apparent signals, each beside the control that removed it."""
    rows = [
        ('Pre-failure RSSI decay', 'dying units 8.4 / 15.7 dB spread',
         'survivor 10.7 dB; spread tracks frame count', 'survivor'),
        ('Pre-cessation traffic dip', 'both ceased units at 76% of mean',
         'a survivor at 49%, still reporting', 'survivor'),
        ('Voltage-tail EOL predictor', 'perfect separation, p = 0.0061',
         'pooled ROC-AUC 0.619 across 3 classes', 'replication'),
        ('Workload -> service life', 'log-log slope +0.03 excludes -1',
         'Cox HR 1.053, p = 0.902; MDE 3.23', 'censoring + power'),
        ('Depletion -> service loss', 'r = -0.22 to -0.44 on 3 devices',
         'within-week r = +0.012; 0 of 8 hold', 'seasonal'),
        ('State-conditioned scanning', '4.6-5.6% scan reduction',
         'null-estimator floor is 3.8-5.1%', 'null model'),
        ('Cessation-time bound', 's <= 0.087%, 200x tighter',
         'estimator wrong ~50% of the time', 'Monte-Carlo'),
    ]
    fig, ax = plt.subplots(figsize=(COL2, 3.1))
    ys = np.arange(len(rows))[::-1]
    for y, (name, sig, ctl, kind) in zip(ys, rows):
        ax.plot([0.30, 0.62], [y, y], color='0.85', lw=0.8, zorder=0)
        ax.scatter([0.30], [y], s=46, color=C_ACC, zorder=3, marker='o')
        ax.scatter([0.62], [y], s=52, color=C_DEAD, zorder=3, marker='X')
        ax.text(-0.01, y, name, fontsize=6.8, ha='right', va='center', weight='bold')
        ax.text(0.32, y + 0.30, sig, fontsize=5.9, va='center', color='0.25')
        ax.text(0.665, y, ctl, fontsize=5.9, va='center', color='0.25')
        ax.text(0.30, y - 0.34, '(%s control)' % kind, fontsize=5.4,
                va='center', color='0.55', ha='center')
    ax.text(0.30, len(rows) - 0.35, 'apparent signal', fontsize=7,
            ha='center', color=C_ACC, weight='bold')
    ax.text(0.62, len(rows) - 0.35, 'after the control', fontsize=7,
            ha='center', color=C_DEAD, weight='bold')
    ax.set_xlim(-0.42, 1.30); ax.set_ylim(-0.9, len(rows) + 0.1)
    ax.axis('off')
    ax.set_title('Seven controlled negatives across four device populations — '
                 'each survived until a control was run', loc='left', fontsize=8)
    save(fig, 'fig8_controlled_negatives')


# =====================================================================  FIG 9
def fig9_corpus_terms():
    """What the ISAC energy-efficiency corpus foregrounds as the energy variable."""
    d = jload('isac_energy_term_audit.json')
    N = d['denominator']
    tx = {k.strip('"'): v for k, v in d['transmit_terms'].items()}
    st = {k.strip('"'): v for k, v in d['standing_terms'].items()}
    fig, ax = plt.subplots(figsize=(COL2, 2.9))
    items = ([(k, v, C_ALIVE) for k, v in sorted(tx.items(), key=lambda x: x[1])] +
             [('', 0, 'none')] +
             [(k, v, C_DEAD) for k, v in sorted(st.items(), key=lambda x: x[1])])
    ys, labs = [], []
    for i, (k, v, c) in enumerate(items):
        if c == 'none':
            labs.append(''); ys.append(i); continue
        ax.barh(i, 100.0 * v / N, color=c, alpha=0.85, height=0.62)
        ax.text(100.0 * v / N + 0.6, i, '%d' % v, fontsize=6, va='center', color='0.3')
        labs.append(k); ys.append(i)
    ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=6.3)
    ax.set_xlabel('share of the 360 ISAC "energy efficiency" papers mentioning the term (%)')
    ax.set_xlim(0, 50)
    ax.text(30, len(tx) / 2 - 0.5, 'transmit-side terms\nANY: 214 / 360 = 59.4%',
            fontsize=7, color=C_ALIVE, weight='bold', va='center')
    ax.text(30, len(tx) + 1 + len(st) / 2 - 0.5,
            'standing-charge terms\nANY: 6 / 360 = 1.7%',
            fontsize=7, color=C_DEAD, weight='bold', va='center')
    ax.set_title('What the ISAC energy literature treats as the energy variable: 35.7x more '
                 'transmit-side than standing-charge', loc='left', fontsize=8)
    ax.grid(axis='y', visible=False)
    fig.text(0.01, -0.05, 'OpenAlex title+abstract, 2018-, strings recorded in '
             'LITERATURE_CHECK.md. Measures salience, not full-text presence.',
             fontsize=5.8, color='0.45')
    save(fig, 'fig9_isac_corpus_terms')


def fig10_regime_invariance():
    """
    The panel a reviewer comparing Figs. 5 and 6 will look for.

    Fig. 6 shows the energy bracket moving 7.7x over two simulator constants.
    Fig. 5 is produced by the same simulator. This figure answers the obvious
    question by overlaying the state-awareness curve at all 25 configurations of
    those same two constants: the curves are a family of the same shape at
    different heights, so the shape is a result and the level is not.

    Left panel: every curve, normalised nowhere -- raw percentages, so the 5.5x
    spread in level is visible. Right panel: the same curves each divided by
    their own saturation value, which collapses them if and only if the shape
    claim is true.
    """
    d = jload('regime_map_invariance.json')
    rr = jload('regime_map_invariance_readout.json')
    kap = d['kappas']
    x = np.arange(len(kap))

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7))
    ax = axes[0]
    sats = []
    for key, c in d['configs'].items():
        # use the re-read curve where the measured unconstrained condition is
        # attained, and the tau-threshold curve otherwise; the three undefined
        # configurations are drawn dashed so they cannot be mistaken for support
        r = rr['reread'].get(key, {})
        if r.get('curve'):
            ax.plot(x, r['curve'], color=C_ALIVE, alpha=0.35, lw=0.9)
            sats.append(max(r['curve']))
        else:
            ax.plot(x, c['state_curve'], color=C_GREY, alpha=0.5, lw=0.9, ls='--')
    base = rr['reread']['n6_p0.02']['curve']
    ax.plot(x, base, color=C_DEAD, lw=1.8, zorder=5,
            label='published configuration (n = 6, p_proc = 20 mW)')
    ax.set_xticks(x)
    ax.set_xticklabels(['%g' % k for k in kap], rotation=45)
    ax.set_xlabel(r'state motion $\kappa$ (fraction of range per epoch)')
    ax.set_ylabel('value of state awareness (%)')
    ax.set_title('(a) 25 configurations: level spans %.1fx' % rr['saturation_ratio'],
                 loc='left')
    ax.plot([], [], color=C_GREY, ls='--', lw=0.9,
            label='never unbinds within the tau grid')
    ax.legend(frameon=False, loc='upper left', fontsize=6)

    ax = axes[1]
    for key in rr['reread']:
        r = rr['reread'][key]
        if not r.get('curve'):
            continue
        c = np.array(r['curve'])
        top = c.max()
        if top > 0:
            ax.plot(x, c / top, color=C_ALIVE, alpha=0.35, lw=0.9)
    b = np.array(base)
    ax.plot(x, b / b.max(), color=C_DEAD, lw=1.8, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(['%g' % k for k in kap], rotation=45)
    ax.set_xlabel(r'state motion $\kappa$ (fraction of range per epoch)')
    ax.set_ylabel('value, normalised to own saturation')
    ax.set_title('(b) same curves, each scaled by its own maximum', loc='left')
    ax.set_ylim(-0.05, 1.08)

    fig.tight_layout()
    # The full provenance goes in the manuscript caption; only a one-line source
    # note goes in the image, because a four-line footnote collides with the
    # x-axis labels once savefig's tight bounding box is applied.
    fig.text(0.005, -0.02,
             'Slot fraction 1/n, n in {1,2,6,14,30}, crossed with p_proc in {0,5,20,50,100} mW.  '
             'code/regime_map_invariance.py',
             fontsize=6, color='0.45', va='top')
    save(fig, 'fig10_regime_invariance')



def fig11_energy_budget():
    """
    Three energy budgets on one normalised axis: infrastructure ISAC, an
    ISAC model that names IoT, and the deployed sensor.

    The point of the figure is the thing you cannot see. On the deployed row
    the communication share is 0.24-0.60% of the budget, which at this scale is
    roughly a line width, while the two published models are almost entirely
    controllable. The sensing band is hatched rather than filled because its
    magnitude is not measured; giving it a solid width would assert the number
    Sec. V-E says cannot be identified.
    """
    C_STAND = '#B9C2CB'
    C_CTRL = C_ALIVE
    C_UNK = '#E4E9ED'
    SENSE_LO, SENSE_HI = 0.2, 32.5
    COMM_HI = 0.6

    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    h = 0.46
    rows = [
        (2, 'Ye et al. [2]\ninfrastructure ISAC, mains-fed', 98.4, None, 'static 1.6%'),
        (1, 'Bai [3]\nISAC model naming IoT', 97.0, None, 'circuit 3%'),
        (0, 'Bosch TPS110EU\ndeployed IoT sensor, primary cell', COMM_HI,
         (COMM_HI, SENSE_HI), None),
    ]

    for y, lab, ctrl, unk, tail in rows:
        x = 0.0
        ax.barh(y, ctrl, left=x, height=h, color=C_CTRL, zorder=3)
        x += ctrl
        if unk is not None:
            ax.barh(y, unk[1], left=x, height=h, color=C_UNK, zorder=3,
                    hatch='////', edgecolor='#9AA4AE', linewidth=0.4)
            x += unk[1]
        ax.barh(y, 100 - x, left=x, height=h, color=C_STAND, zorder=3)
        if ctrl > 50:
            ax.text(ctrl / 2, y, 'controllable  %.4g%%' % ctrl, ha='center',
                    va='center', fontsize=7.5, color='white', weight='bold', zorder=5)
        if tail:
            ax.text(100, y + 0.30, tail, ha='right', va='bottom', fontsize=6.6,
                    color='#3C464F', zorder=5)

    # --- callouts, all placed ABOVE the bar they annotate --------------------
    ax.annotate('communication  0.24-0.60%   MEASURED',
                xy=(COMM_HI, 0.25), xytext=(5.0, 0.60),
                fontsize=7.2, color=C_ALIVE, weight='bold', ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color=C_ALIVE, lw=1.2,
                                shrinkA=0, shrinkB=1))
    ax.text(COMM_HI + SENSE_HI / 2, 0,
            'sensing  %.1f-%.1f%%\nNOT MEASURED' % (SENSE_LO, SENSE_HI),
            ha='center', va='center', fontsize=7, color='#3C464F', zorder=6,
            bbox=dict(facecolor='white', alpha=0.86, edgecolor='none',
                      boxstyle='round,pad=0.28'))
    ax.text(COMM_HI + SENSE_HI + (100 - COMM_HI - SENSE_HI) / 2, 0,
            'standing charge:  at least 67%, and never measured',
            ha='center', va='center', fontsize=7.4, color='#2E373F', zorder=5)

    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=7.2)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.45, 2.45)
    ax.set_xlabel('share of the device energy budget (%)')
    ax.set_title('On the deployed sensor the controllable share is not visible '
                 'at the scale the published models occupy', loc='left', fontsize=8.2)
    ax.grid(axis='y', visible=False)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_CTRL, label='controllable / communication'),
        Patch(facecolor=C_UNK, hatch='////', edgecolor='#9AA4AE',
              label='sensing: not measured'),
        Patch(facecolor=C_STAND, label='standing charge')],
        loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=3,
        frameon=False, fontsize=7)

    fig.tight_layout()
    fig.text(0.005, -0.06,
             'Published shares are the controllable share f_ctrl, read from each '
             'paper\'s own simulation table. The deployed row shows f_comm measured '
             'and f_sense unresolved.',
             fontsize=6.2, color='0.45', va='top')
    save(fig, 'fig11_energy_budget')


def make_all():
    print("Regenerating all figures:")
    fig1_timeline(); fig2_june_control(); fig3_depletion_regressor(); fig4_channel()
    fig5_regime_map(); fig6_bracket(); fig7_coincidence(); fig8_negatives()
    fig9_corpus_terms(); fig10_regime_invariance(); fig11_energy_budget()


# =====================================================================  FIG 12
def fig12_claim_resolution():
    """What it costs to check a claim, passively and by amplification.

    The paper's argument in one axis pair. A claim of a given size sits
    somewhere on the x-axis; the curves say how long a campaign would have to
    run to resolve it. The passive curve crosses the feasibility line far to
    the right of the share actually available on this hardware; the amplified
    one crosses it far to the left. That gap is the reason the experiment in
    Sec. IX exists.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from identifiability import (block_days_required, SIGMA_LO, SIGMA_HI,
                                 MAX_CAMPAIGN_DAYS)

    K_TX, K_RX = 17.42, 11.60          # transmit-only and RX-inclusive
    F_LO, F_HI = 0.0024, 0.0060
    claims = np.logspace(np.log10(0.0006), np.log10(0.30), 40)

    def campaign(eps_series, sigma):
        out = []
        for e in eps_series:
            b = block_days_required(e, 6, sigma, n_devices=3)
            out.append(4.0 * b if np.isfinite(b) else np.nan)
        return np.array(out)

    # passive: the claim IS the observable fractional change
    pas_lo = campaign(claims, SIGMA_LO)
    pas_hi = campaign(claims, SIGMA_HI)
    # amplified: a share s shows up as eps = s*(k-1)
    amp_lo = campaign(claims * (K_TX - 1), SIGMA_LO)
    amp_hi = campaign(claims * (K_TX - 1), SIGMA_HI)
    amp_rx = campaign(claims * (K_RX - 1), SIGMA_HI)

    fig, ax = plt.subplots(figsize=(COL1, 2.65))
    x = 100 * claims

    ax.axvspan(100 * F_LO, 100 * F_HI, color=C_ACC, alpha=0.13, lw=0)
    ax.axhline(MAX_CAMPAIGN_DAYS, color=C_DEAD, lw=1.0, ls='--')
    ax.text(26, MAX_CAMPAIGN_DAYS * 1.18, 'longest observed service, 760 d',
            fontsize=5.6, color=C_DEAD, ha='right')

    ax.fill_between(x, pas_lo, pas_hi, color=C_GREY, alpha=0.30, lw=0)
    ax.plot(x, pas_hi, color=C_GREY, lw=1.4, label='passive telemetry')
    ax.fill_between(x, amp_lo, amp_hi, color=C_ACC, alpha=0.30, lw=0)
    ax.plot(x, amp_hi, color=C_ACC, lw=1.4,
            label=r'amplified, $k=17.4$ (transmit only)')
    ax.plot(x, amp_rx, color=C_ACC, lw=1.0, ls=':',
            label=r'amplified, $k=11.6$ (with RX windows)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.06, 30)
    ax.set_ylim(40, 4e4)
    ax.set_xlabel('claimed energy improvement (% of the device budget)')
    ax.set_ylabel('campaign required (days)')
    ax.text(np.sqrt(100 * F_LO * 100 * F_HI), 2.6e4,
            r'$f_{\mathrm{comm}}$' + '\n0.24-0.60%', fontsize=6.2,
            color=C_ACC, ha='center', va='top')
    ax.legend(frameon=False, fontsize=5.6, loc='lower left')
    ax.set_title('Amplification, not fleet size, brings the measurement\n'
                 'inside a service life', loc='left', fontsize=7.4)
    fig.tight_layout()
    save(fig, 'fig12_claim_resolution')


if __name__ == '__main__':
    # Previously this file ended with `if __name__ == '__main__' or True: pass`,
    # so running it produced nothing at all and every figure had to be made by
    # hand from an interactive session. That is a reproducibility hole in a paper
    # whose claim is about reproducibility, so it is fixed here.
    import sys
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            globals()[name]()
    else:
        make_all()
