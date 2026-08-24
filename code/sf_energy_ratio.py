"""
Where the SF7/SF12 amplification factor comes from, and what else it moves.

WHY THIS EXISTS. The crossover's whole power calculation runs through one
number: the ratio k of per-uplink energy at the high spreading factor to the
low one. Until now that number lived in a dict --

    K = {7: 1.00, 8: 1.61, 9: 2.52, 10: 4.65, 11: 8.91, 12: 17.42}

-- with the per-SF energies quoted in a docstring and computed by nothing. A
reviewer is entitled to ask what 17.42 is a ratio OF, and the honest answer was
not available from the release. This derives it from the LoRa airtime formula
and a stated radio model, checks it against the published constants, and then
asks the harder question: does changing the spreading factor move only the
transmit term?

THE AIRTIME FORMULA (Semtech AN1200.13, EU868, BW = 125 kHz):

    T_sym      = 2^SF / BW
    T_preamble = (n_preamble + 4.25) * T_sym
    n_payload  = 8 + max(ceil((8*PL - 4*SF + 28 + 16 - 20*H)
                              / (4*(SF - 2*DE))) * (CR + 4), 0)
    T_packet   = T_preamble + n_payload * T_sym

with the low-data-rate optimiser DE = 1 where T_sym > 16 ms, i.e. SF11 and
SF12 at 125 kHz.

THE RADIO MODEL. Energy per uplink is

    E = E_fixed + P_tx * T_air + P_rx * T_rx

where E_fixed covers wake, oscillator settling and the sleep-to-transmit
transition; P_tx is the transmit draw at a fixed conducted power, which does
NOT vary with SF; and the receive term exists because LoRaWAN class-A opens two
downlink windows after every uplink. RX1 uses the data rate of the uplink, so
its duration scales with the symbol time -- which is the one place a spreading
factor change reaches beyond the transmit term by construction.

WHAT THIS ESTABLISHES. k is an energy ratio under a stated model, not a
measurement. It is not the airtime ratio: airtime moves 28.0x from SF7 to SF12
while energy moves 17.4x, because the fixed per-uplink overhead dilutes it. A
bench measurement of the modem in the deployed firmware configuration would
replace the model, and Sec. IX says so.

Author: Vullnet Laniku
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')

BW = 125e3
PL = 12          # bytes, the deployed payload
CR = 1           # 4/5
H = 0            # explicit header
N_PREAMBLE = 8

# Radio model. SX1276-class at +14 dBm on a 3.3 V rail.
V = 3.3
I_TX_A = 0.0439          # A, transmit draw at +14 dBm
I_RX_A = 0.0110          # A, receive draw
E_FIXED_J = 3.66e-3      # J, wake + settle + sleep-to-TX transition
RX_SYMBOLS = 5.0         # symbols the receiver must stay open to detect preamble
RX2_S = 0.1638           # s, RX2 always at SF12/125 kHz in EU868

# The constants this file has to reproduce.
PUBLISHED_K = {7: 1.00, 8: 1.61, 9: 2.52, 10: 4.65, 11: 8.91, 12: 17.42}
PUBLISHED_MJ = {7: 9.82, 9: 24.76, 10: 45.67, 12: 171.13}

# EU868 duty cycle, per sub-band, and the deployment's measured uplink rate.
DUTY_CYCLE = 0.01
UPLINKS_PER_DAY = 3.68


def t_sym(sf):
    return (2.0 ** sf) / BW


def airtime(sf):
    ts = t_sym(sf)
    de = 1 if ts > 16e-3 else 0
    num = 8 * PL - 4 * sf + 28 + 16 - 20 * H
    den = 4 * (sf - 2 * de)
    n_payload = 8 + max(math.ceil(num / den) * (CR + 4), 0)
    return (N_PREAMBLE + 4.25) * ts + n_payload * ts


def energy(sf, with_rx=True):
    """Per-uplink energy, J."""
    e = E_FIXED_J + V * I_TX_A * airtime(sf)
    if with_rx:
        t_rx = RX_SYMBOLS * t_sym(sf) + RX_SYMBOLS * t_sym(12)
        e += V * I_RX_A * t_rx
    return e


def main():
    out = {'_method': {
        'airtime': 'Semtech AN1200.13, EU868, BW=125 kHz, PL=%d B, CR=4/5, '
                   'explicit header, %d-symbol preamble' % (PL, N_PREAMBLE),
        'radio_model': {'V': V, 'I_tx_A': I_TX_A, 'I_rx_A': I_RX_A,
                        'E_fixed_J': E_FIXED_J, 'rx_symbols': RX_SYMBOLS},
        'note': 'k is a modelled energy ratio, not a measurement.'}}

    print("=" * 94)
    print("  1. AIRTIME AND ENERGY BY SPREADING FACTOR   (BW 125 kHz, %d-byte payload)"
          % PL)
    print("=" * 94)
    print("  %-4s %10s %12s %12s %10s %10s %10s"
          % ("SF", "T_sym ms", "airtime ms", "E tx-only mJ", "E +RX mJ",
             "k tx-only", "k +RX"))
    e7_tx = energy(7, with_rx=False)
    e7_rx = energy(7, with_rx=True)
    rows = {}
    for sf in range(7, 13):
        a = airtime(sf)
        etx, erx = energy(sf, False), energy(sf, True)
        rows[sf] = {'t_sym_ms': 1e3 * t_sym(sf), 'airtime_ms': 1e3 * a,
                    'E_tx_only_mJ': 1e3 * etx, 'E_with_rx_mJ': 1e3 * erx,
                    'k_tx_only': etx / e7_tx, 'k_with_rx': erx / e7_rx}
        print("  %-4d %10.3f %12.1f %12.2f %10.2f %10.2f %10.2f"
              % (sf, 1e3 * t_sym(sf), 1e3 * a, 1e3 * etx, 1e3 * erx,
                 etx / e7_tx, erx / e7_rx))
    out['per_sf'] = rows

    # ------------------------------------------------- the control ---------
    print()
    print("=" * 94)
    print("  2. CONTROL   does this reproduce the constants the paper uses?")
    print("=" * 94)
    print("  %-4s %14s %14s %10s   %14s %14s %10s"
          % ("SF", "published mJ", "derived mJ", "rel", "published k",
             "derived k", "rel"))
    worst = 0.0
    for sf in sorted(PUBLISHED_K):
        dk = rows[sf]['k_tx_only']
        pk = PUBLISHED_K[sf]
        rk = abs(dk - pk) / pk
        pm = PUBLISHED_MJ.get(sf)
        dm = rows[sf]['E_tx_only_mJ']
        rm = abs(dm - pm) / pm if pm else 0.0
        worst = max(worst, rk, rm)
        print("  %-4d %14s %14.2f %9.1f%%   %14.2f %14.2f %9.1f%%"
              % (sf, ('%.2f' % pm) if pm else '--', dm, 100 * rm,
                 pk, dk, 100 * rk))
    ok = worst < 0.03
    print("  worst relative disagreement %.1f%%  ->  %s"
          % (100 * worst, "AGREES" if ok else "*** DISAGREES ***"))
    out['control'] = {'worst_rel': worst, 'agrees': bool(ok)}
    if not ok:
        raise AssertionError("the derived energies do not reproduce the "
                             "constants the paper uses")

    # ------------------------------- what else the manipulation moves ------
    print()
    print("=" * 94)
    print("  3. WHAT ELSE CHANGES BETWEEN SF7 AND SF12")
    print("=" * 94)
    a7, a12 = airtime(7), airtime(12)
    tx7 = V * I_TX_A * a7
    tx12 = V * I_TX_A * a12
    rx7 = V * I_RX_A * (RX_SYMBOLS * t_sym(7) + RX_SYMBOLS * t_sym(12))
    rx12 = V * I_RX_A * (RX_SYMBOLS * t_sym(12) + RX_SYMBOLS * t_sym(12))
    dc7 = a7 * UPLINKS_PER_DAY / 86400.0
    dc12 = a12 * UPLINKS_PER_DAY / 86400.0
    hourly7 = DUTY_CYCLE * 3600.0 / a7
    hourly12 = DUTY_CYCLE * 3600.0 / a12

    def line(q, v7, v12, note):
        print("  %-26s %12s %12s   %s" % (q, v7, v12, note))

    print("  %-26s %12s %12s   %s" % ("quantity", "SF7", "SF12", "status"))
    line("airtime", "%.1f ms" % (1e3 * a7), "%.0f ms" % (1e3 * a12),
         "ratio %.1fx -- the driver" % (a12 / a7))
    line("transmit energy", "%.2f mJ" % (1e3 * tx7), "%.1f mJ" % (1e3 * tx12),
         "ratio %.1fx" % (tx12 / tx7))
    line("RX-window energy", "%.2f mJ" % (1e3 * rx7), "%.2f mJ" % (1e3 * rx12),
         "RX1 tracks the uplink DR: NOT controlled")
    line("fixed overhead", "%.2f mJ" % (1e3 * E_FIXED_J),
         "%.2f mJ" % (1e3 * E_FIXED_J), "invariant by construction")
    line("duty-cycle use", "%.4f%%" % (100 * dc7), "%.3f%%" % (100 * dc12),
         "limit %.0f%%: %.0f vs %.0f uplinks/h allowed"
         % (100 * DUTY_CYCLE, hourly7, hourly12))
    line("payload, cadence", "12 B", "12 B", "unchanged: the device decides")
    line("sensing, scan rate", "same", "same", "no vendor-configurable knob")

    print()
    print("  RX windows are the term that is NOT controlled, and they cut the")
    print("  other way from the intuition. EU868 opens RX2 at SF12/125 kHz")
    print("  REGARDLESS of the uplink data rate, so the low arm pays almost the")
    print("  same receive cost as the high arm:")
    print("    RX energy   SF7 %.2f mJ    SF12 %.2f mJ   (ratio only %.2fx)"
          % (1e3 * rx7, 1e3 * rx12, rx12 / rx7))
    print("  Including it therefore SHRINKS the amplification, from")
    print("    k = %.2f (transmit only)  ->  k = %.2f (transmit + RX windows),"
          % (rows[12]['k_tx_only'], rows[12]['k_with_rx']))
    print("  a %.0f%% reduction. Since the crossover detects a share"
          % (100 * (1 - rows[12]['k_with_rx'] / rows[12]['k_tx_only'])))
    print("  s = eps/(k-1), that makes the minimum detectable share WORSE by")
    print("  a factor of %.2f. This is the unfavourable direction and the"
          % ((rows[12]['k_tx_only'] - 1) / (rows[12]['k_with_rx'] - 1)))
    print("  published k = 17.42 is optimistic for the experiment.")
    print()
    print("  The same omission runs the other way for f_comm. The deployed")
    print("  share is computed from airtime alone, so it counts transmit")
    print("  energy and no receive energy, and is therefore a LOWER BOUND on")
    print("  the communication share:")
    mix = {7: 0.46, 10: 0.50, 9: 0.04}          # measured fleet SF mix
    e_tx = sum(w * energy(sf, False) for sf, w in mix.items())
    e_rx = sum(w * energy(sf, True) for sf, w in mix.items())
    print("    per uplink at the measured SF mix:  %.2f mJ transmit only,"
          % (1e3 * e_tx))
    print("                                        %.2f mJ including RX windows"
          % (1e3 * e_rx))
    print("    so f_comm scales by %.2fx: 0.24-0.60%% -> %.2f-%.2f%%"
          % (e_rx / e_tx, 0.24 * e_rx / e_tx, 0.60 * e_rx / e_tx))
    print()
    print("  The two corrections partly cancel for the EXPERIMENT, because the")
    print("  detectable effect is d0*s*(k-1): s rises %.2fx while (k-1) falls"
          % (e_rx / e_tx))
    print("  %.2fx, a net %.2fx on the effect size."
          % ((rows[12]['k_with_rx'] - 1) / (rows[12]['k_tx_only'] - 1),
             (e_rx / e_tx) * (rows[12]['k_with_rx'] - 1) / (rows[12]['k_tx_only'] - 1)))
    print()
    print("  Retransmissions are zero by configuration: the deployed devices")
    print("  send unconfirmed uplinks, which are not retransmitted.")
    print()
    print("  Duty cycle does NOT bind at either arm: at the measured %.2f"
          % UPLINKS_PER_DAY)
    print("  uplinks/day the device uses %.3f%% of a %.0f%% allowance at SF12."
          % (100 * dc12, 100 * DUTY_CYCLE))

    out['confounds'] = {
        'airtime_ratio': a12 / a7,
        'tx_energy_mJ': [1e3 * tx7, 1e3 * tx12],
        'rx_energy_mJ': [1e3 * rx7, 1e3 * rx12],
        'k_tx_only': rows[12]['k_tx_only'],
        'k_with_rx': rows[12]['k_with_rx'],
        'k_change_pct': 100 * (rows[12]['k_with_rx'] / rows[12]['k_tx_only'] - 1),
        'mde_share_penalty': ((rows[12]['k_tx_only'] - 1)
                              / (rows[12]['k_with_rx'] - 1)),
        'f_comm_scale_if_rx_included': e_rx / e_tx,
        'per_uplink_mJ_tx_only': 1e3 * e_tx,
        'per_uplink_mJ_with_rx': 1e3 * e_rx,
        'duty_cycle_used': [dc7, dc12],
        'duty_cycle_limit': DUTY_CYCLE,
        'uplinks_per_hour_allowed': [hourly7, hourly12],
        'binds': bool(dc12 > DUTY_CYCLE)}

    # ------------------- which part of communication is controllable --------
    # The energy identity bounds an allocator by what it can VARY. Not all
    # communication energy qualifies: EU868 fixes the second receive window at
    # SF12 regardless of the uplink data rate, so RX2 is a standing cost of
    # being a class-A device rather than an allocation variable. RX1 does open
    # at the uplink data rate and is therefore controllable along with the
    # transmit term.
    print()
    print("=" * 94)
    print("  4. WHICH PART OF COMMUNICATION AN ALLOCATOR CAN ACTUALLY VARY")
    print("=" * 94)
    mix = {7: 0.46, 10: 0.50, 9: 0.04}          # measured fleet SF mix
    tx_m = sum(w * V * I_TX_A * airtime(sf) for sf, w in mix.items())
    rx1_m = sum(w * V * I_RX_A * RX_SYMBOLS * t_sym(sf) for sf, w in mix.items())
    rx2_m = V * I_RX_A * RX_SYMBOLS * t_sym(12)
    print("  %-24s %10s   %s" % ("term", "mJ/uplink", "status"))
    print("  %-24s %10.2f   invariant" % ("fixed overhead", 1e3 * E_FIXED_J))
    print("  %-24s %10.2f   varies with SF: CONTROLLABLE" % ("transmit", 1e3 * tx_m))
    print("  %-24s %10.2f   opens at the uplink DR: CONTROLLABLE"
          % ("RX1 window", 1e3 * rx1_m))
    print("  %-24s %10.2f   fixed at SF12 by EU868: NOT controllable"
          % ("RX2 window", 1e3 * rx2_m))
    e_txonly = E_FIXED_J + tx_m
    e_ctrl = E_FIXED_J + tx_m + rx1_m
    e_all = e_ctrl + rx2_m
    print()
    print("  transmit-only accounting  %6.2f mJ   (what the paper reports)"
          % (1e3 * e_txonly))
    print("  controllable communication %5.2f mJ   x%.3f" % (1e3 * e_ctrl,
                                                             e_ctrl / e_txonly))
    print("  all communication         %6.2f mJ   x%.3f" % (1e3 * e_all,
                                                            e_all / e_txonly))
    print()
    print("  So the ceiling on any allocator is f_comm,ctrl, not f_comm:")
    print("    reported f_comm      0.24-0.60%")
    print("    f_comm,ctrl          %.2f-%.2f%%   <- the correct quantity"
          % (0.24 * e_ctrl / e_txonly, 0.60 * e_ctrl / e_txonly))
    print("    f_comm (all)         %.2f-%.2f%%"
          % (0.24 * e_all / e_txonly, 0.60 * e_all / e_txonly))
    print("  The definition is now rigorous and the number moves by 3%, which is")
    print("  well inside its own bracket. RX2 is 17% of communication energy and")
    print("  no allocator on this device can touch it.")
    out['controllable_split'] = {
        'mJ_per_uplink': {'fixed': 1e3 * E_FIXED_J, 'tx': 1e3 * tx_m,
                          'rx1': 1e3 * rx1_m, 'rx2_not_controllable': 1e3 * rx2_m},
        'e_transmit_only_mJ': 1e3 * e_txonly,
        'e_controllable_mJ': 1e3 * e_ctrl,
        'e_all_comm_mJ': 1e3 * e_all,
        'ctrl_over_txonly': e_ctrl / e_txonly,
        'f_comm_ctrl_pct': [0.24 * e_ctrl / e_txonly, 0.60 * e_ctrl / e_txonly],
        'rx2_share_of_comm': rx2_m / (tx_m + rx1_m + rx2_m)}

    path = os.path.join(RESULTS, 'sf_energy_ratio.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print()
    print("Saved %s" % path)


if __name__ == '__main__':
    main()
