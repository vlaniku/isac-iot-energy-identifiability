"""
What is (1 - e_min/e_mean) actually a property of?

The regime map's Part 1 bounds any allocator by

    max energy saving = f * (1 - e_min/e_mean)

and puts the bracket at 0.333, measured on the corrected 75-action grid. Every
ceiling figure in the project inherits that number, including the device rows.
The deployed radar is capped at -28 dBm EIRP, 38-48 dB below the 10-20 dBm grid
the bracket was measured on, so the number needs checking before it is quoted
against hardware.

PART 1 - is 0.333 a property of the ACTION SET?

  Energy in the simulator is

      e = (P_s + P_c) * tdma_slot_fraction + p_proc        [isac_physical_models]

  with p_proc = 0.02 W hard-coded and uncited. Bandwidth ratio does not enter
  energy at all, so the "75-action grid" carries 25 distinct energies, not 75.
  Sweep the two quantities the bracket must NOT depend on if it describes an
  action set: the MAC slot fraction, and the processing constant.

PART 2 - the bracket on the action set the DEVICE actually has.

  A EU868 Class-A endpoint chooses over
      DR0-DR5   == SF12..SF7 at 125 kHz   (LoRaWAN Regional Parameters)
      TXPower   == 2..14 dBm ERP          (manual: max 14 dBm ERP)
  Energy per uplink uses the SX1276-class per-mode currents already calibrated
  in experiment_lifetime_budget.py and the standard 125 kHz airtime formula.
  The PA current-vs-output-power curve is not published for this device, so it
  is swept as a sensitivity k = I_tx(2 dBm) / I_tx(14 dBm).

  Reported three ways, because they are three different action sets:
    (a) everything the radio can do
    (b) the two spreading factors the fleet actually uses
    (c) what an allocator could actually choose -- power only, because SF is set
        by ADR and is 95.8%/95.6% determined by the sensed occupancy state

PART 3 - the link-budget share, recomputed on the network-server record.

  ceiling_band.py used 3.2 uplinks/day at SF7. Both inputs came from the
  application export, which is now known to be incomplete (44-88% capture,
  deploy_workload_control.py). ChirpStack gives the true rate, and the fleet
  runs ~48/52 SF7/SF10, where an SF10 uplink costs 4.6x an SF7 one.

Author: Vullnet Laniku
"""

import json
import os

import numpy as np

import experiment_lifetime_budget as LB

P_LEVELS = [10.0, 13.0, 15.0, 18.0, 20.0]      # the grid's dBm levels
P_PROC = 0.02                                   # W, hard-coded and uncited
BW_RATIOS = 3                                   # replicates energy, does not vary it

SFS_ALL = (7, 8, 9, 10, 11, 12)                 # DR5..DR0
SFS_FLEET = (7, 10)                             # measured: 46% SF7, 50% SF10
TXP = (2, 4, 6, 8, 10, 12, 14)                  # dBm ERP, capped by the manual
K_SWEEP = (1.00, 0.75, 0.50, 0.35)              # I_tx(2 dBm) / I_tx(14 dBm)

SF_MIX = {7: 0.48, 10: 0.52}                    # renormalised from 46/50
CELL_AH = (2.4, 3.6, 4.8, 6.0)
HOURS_5Y = 5 * 365 * 24.0

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
RESULTS = os.path.join(HERE, '..', 'results')


def sim_bracket(slot_frac, p_proc=P_PROC, levels=P_LEVELS):
    """1 - e_min/e_mean on the simulator's action grid."""
    w = [10 ** (p / 10.0) / 1000.0 for p in levels]
    e = np.repeat(np.array([(a + b) * slot_frac + p_proc for a in w for b in w]),
                  BW_RATIOS)
    return float(1.0 - e.min() / e.mean())


def uplink_j(sf, p_dbm, k):
    """Joules for one Class-A uplink at (SF, TX power), SX1276-class currents."""
    i_tx = LB.I_TX * (k + (1.0 - k) * (p_dbm - min(TXP)) / (max(TXP) - min(TXP)))
    fixed = LB.I_MCU * LB.T_MCU + LB.I_RX1 * LB.T_RX1 + LB.I_RX2 * LB.T_RX2
    return LB.V_BAT * (i_tx * LB.lora_airtime(sf) + fixed)


def dev_bracket(sfs, k):
    e = np.array([uplink_j(sf, p, k) for sf in sfs for p in TXP])
    return float(1.0 - e.min() / e.mean()), float(e.min()), float(e.mean())


def server_rates():
    """Per-device uplinks/day from the network server, whole months only."""
    with open(os.path.join(DATA, 'chirpstack_12mo_metrics.json')) as fh:
        m = json.load(fh)
    days = {"2025-08": 31, "2025-09": 30, "2025-10": 31, "2025-11": 30,
            "2025-12": 31, "2026-01": 31, "2026-02": 28, "2026-03": 31,
            "2026-04": 30, "2026-05": 31, "2026-06": 30, "2026-07": 31}
    out = {}
    for dev in m['devices'].values():
        tot = nd = 0
        for mo, c in zip(m['months'], dev['rx_count']):
            if mo not in days or c == 0:        # partial month, or not in service
                continue
            tot += c
            nd += days[mo]
        out[dev['name'][-10:]] = tot / nd
    return out


def main():
    res = {}

    print("=" * 96)
    print("  PART 1   is 0.333 a property of the action set?")
    print("=" * 96)
    base = sim_bracket(1.0 / 6)
    print("  regime_map.py config (TDMA, n=6)      bracket = %.4f   (reported: 0.333)"
          % base)
    print()
    print("  same action set, sweeping the MAC slot fraction:")
    mac = {}
    for lab, fr in [("OFDMA (slot_frac = 1.0)", 1.0), ("TDMA, n=2", 0.5),
                    ("TDMA, n=6", 1 / 6), ("TDMA, n=14", 1 / 14),
                    ("TDMA, n=30", 1 / 30)]:
        b = sim_bracket(fr)
        mac[lab] = b
        print("    %-34s %.3f" % (lab, b))
    print()
    print("  same action set, sweeping p_proc (the uncited 20 mW constant):")
    proc = {}
    for pp in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
        b6, bo = sim_bracket(1 / 6, pp), sim_bracket(1.0, pp)
        proc['%.3f' % pp] = {'tdma_n6': b6, 'ofdma': bo}
        print("    p_proc = %.3f W   TDMA n=6 %.3f   OFDMA %.3f" % (pp, b6, bo))
    print()
    print("  At p_proc = 0 the bracket is the pure power-grid geometry, %.3f,"
          % sim_bracket(1 / 6, 0.0))
    print("  independent of slot fraction as it must be. The reported 0.333 is the")
    print("  ratio between two simulator constants, not a property of an action set.")
    res['simulator'] = {'reported': 0.333, 'reproduced': base,
                        'over_mac_mode': mac, 'over_p_proc': proc,
                        'pure_grid_geometry': sim_bracket(1 / 6, 0.0)}

    print()
    print("=" * 96)
    print("  PART 2   the bracket on the action set the device actually has")
    print("=" * 96)
    print("  %-44s %9s %10s %9s" % ("action set", "e_min mJ", "e_mean mJ", "bracket"))
    dev = {}
    for k in K_SWEEP:
        b, lo, mu = dev_bracket(SFS_ALL, k)
        dev['all_dr_k%.2f' % k] = b
        print("  %-44s %9.2f %10.2f %9.3f"
              % ("DR0-DR5 x 2-14 dBm, k=%.2f" % k, 1e3 * lo, 1e3 * mu, b))
    print()
    for k in (1.00, 0.50):
        b, lo, mu = dev_bracket(SFS_FLEET, k)
        dev['fleet_sf_k%.2f' % k] = b
        print("  %-44s %9.2f %10.2f %9.3f"
              % ("SF7/SF10 only (the fleet mix), k=%.2f" % k, 1e3 * lo, 1e3 * mu, b))
    print()
    print("  SF is not a free variable: ADR sets it, and it is 95.8%/95.6%")
    print("  determined by occupancy. Power only, at fixed SF:")
    for sf in SFS_FLEET:
        for k in K_SWEEP:
            b, lo, mu = dev_bracket((sf,), k)
            dev['sf%d_power_only_k%.2f' % (sf, k)] = b
            print("    SF%-3d k=%.2f   bracket = %.3f" % (sf, k, b))
    print()
    print("  Applied to the field bound f <= 0.173 the honest comparison is:")
    b_dev, _, _ = dev_bracket(SFS_ALL, 1.00)
    print("    with the simulator bracket 0.333 : ceiling %.1f%%  <- as published"
          % (100 * 0.173 * 0.333))
    print("    with the device bracket   %.3f  : ceiling %.1f%%"
          % (b_dev, 100 * 0.173 * b_dev))
    print("    with no bracket at all (<= 1)    : ceiling %.1f%%  <- firm, model-free"
          % (100 * 0.173))
    print("  The published 5.8%% is not conservative; it understates by %.1fx."
          % (b_dev / 0.333))
    res['device_action_set'] = dev
    res['ceiling_comparison'] = {
        'f_field_all_uplinks': 0.173,
        'as_published_pct': 100 * 0.173 * 0.333,
        'device_bracket': b_dev,
        'device_bracket_pct': 100 * 0.173 * b_dev,
        'no_bracket_firm_pct': 100 * 0.173}

    print()
    print("=" * 96)
    print("  PART 3   link-budget share on the network-server record")
    print("=" * 96)
    rates = server_rates()
    for nm, r in sorted(rates.items(), key=lambda kv: kv[1]):
        print("    %-12s %.2f uplinks/day" % (nm, r))
    lo_r, hi_r = min(rates.values()), max(rates.values())
    mean_r = float(np.mean(list(rates.values())))
    print("    fleet %.2f - %.2f/day, mean %.2f   (ceiling_band.py used 3.2, from"
          % (lo_r, hi_r, mean_r))
    print("    the incomplete export, and SF7 only)")
    print()
    e_mix = sum(SF_MIX[sf] * LB.uplink_energy(sf) for sf in SF_MIX)
    print("    per-uplink: SF7 %.2f mJ, SF10 %.2f mJ, at the 48/52 mix %.2f mJ"
          % (1e3 * LB.uplink_energy(7), 1e3 * LB.uplink_energy(10), 1e3 * e_mix))

    def ua(rate, ej):
        return rate * ej / (LB.V_BAT * 86400.0) * 1e6

    i_lo = ua(lo_r, LB.uplink_energy(7))
    i_typ = ua(mean_r, e_mix)
    i_hi = ua(hi_r, LB.uplink_energy(10))
    print("    average communication current: %.3f uA (slowest device, all SF7)"
          % i_lo)
    print("                                   %.3f uA (fleet mean at the SF mix)"
          % i_typ)
    print("                                   %.3f uA (busiest device, all SF10)"
          % i_hi)
    print()
    print("    against the total implied by 'up to 5 years':")
    tot = {}
    for ah in CELL_AH:
        it = ah / HOURS_5Y * 1e6
        tot['%.1fAh' % ah] = it
        print("      %.1f Ah -> %5.1f uA   comm share %.2f%% - %.2f%%"
              % (ah, it, 100 * i_lo / it, 100 * i_hi / it))
    i_max, i_min = max(tot.values()), min(tot.values())
    print()
    print("    typical   : %.2f%% - %.2f%%   (fleet mean, 2.4-6.0 Ah)"
          % (100 * i_typ / i_max, 100 * i_typ / i_min))
    print("    full range: %.2f%% - %.2f%%"
          % (100 * i_lo / i_max, 100 * i_hi / i_min))
    print("    ceiling_band.py reported '0.101 uA, under 0.2%'. Same conclusion,")
    print("    but the number is ~3x larger once the rate and the SF mix are right.")
    res['link_budget'] = {
        'server_rates_per_day': rates, 'rate_min': lo_r, 'rate_max': hi_r,
        'rate_mean': mean_r, 'e_uplink_sf7_J': LB.uplink_energy(7),
        'e_uplink_sf10_J': LB.uplink_energy(10), 'e_uplink_mix_J': e_mix,
        'i_comm_uA_min': i_lo, 'i_comm_uA_typ': i_typ, 'i_comm_uA_max': i_hi,
        'total_uA_by_capacity': tot,
        'share_pct_typical': [100 * i_typ / i_max, 100 * i_typ / i_min],
        'share_pct_full_range': [100 * i_lo / i_max, 100 * i_hi / i_min],
        'as_published_uA': 0.10104936921810699, 'as_published_share_pct': 0.2}

    out = os.path.join(RESULTS, 'comm_action_bracket_results.json')
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=2)
    print("\nSaved %s" % out)


if __name__ == '__main__':
    main()
