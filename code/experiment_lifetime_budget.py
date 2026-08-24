"""
Kill test: is a lifetime budget worth managing adaptively?

The Pareto-constrained RL direction failed because per-step allocation quality
barely mattered -- a 100-step-stale solution cost under 1% regret, so there was
nothing for adaptation to buy. This experiment asks whether the *long-horizon
energy budget* behaves differently.

The setting is the deployed device, not an abstraction. A Bosch TPS110 carries a
radar and a LoRaWAN radio on one non-replaceable lithium cell rated for five
years. Its radar detects vehicles; each detection generates an uplink. So the
sensing rate determines the traffic, which determines the energy, which
determines whether the device reaches its rated life. The controllable knob is
the sensing interval s: short s gives low detection latency but more scans and
more uplinks; long s saves energy but events are reported late.

Policies, all on the same arrival realisation and the same energy budget:

  myopic        always sense at the fastest rate. Ignores the budget entirely.
  fixed_ration  choose one s at t=0 so that predicted spend exactly exhausts the
                budget at the horizon, using the arrival rate observed initially.
                Never adapts.
  adr_only      nominal fixed sensing interval, SF adapted for link margin only
                -- i.e. what LoRaWAN ADR actually does. No battery awareness.
  budget_aware  adapt s from remaining budget versus remaining time and the
                recently observed arrival rate.
  oracle        knows the whole arrival sequence in advance and picks the best
                constant s offline. Upper bound for non-adaptive schedules.

If budget_aware does not beat fixed_ration and adr_only by a clear margin, the
budget is not worth managing and this direction dies here.

Calibration: per-mode currents are taken from published lab measurements of an
SX1276-class LPWAN module (Keysight N6705B, 3.7 V): sleep 0.081 mA, MCU active
7.79 mA, TX 39.14 mA, RX windows 11.24 / 10.49 mA. LoRa airtime uses the
standard 125 kHz formula. The radar scan energy is NOT on the Bosch datasheet
(it gives -28 dBm EIRP, not consumed power), so it is a swept parameter.
The budget is calibrated so that the measured operating point -- 197 min median
cadence at the measured SF mix -- lasts the rated five years.

Outputs lifetime_budget_results.json.

Author: Vullnet Laniku
"""

import json

import numpy as np

# ---- measured per-mode currents (A) at 3.7 V, published SX1276-class module --
V_BAT = 3.7
I_SLEEP = 0.081e-3
I_MCU = 7.79e-3
I_TX = 39.14e-3
I_RX1 = 11.24e-3
I_RX2 = 10.49e-3
T_MCU = 0.050          # s of MCU activity per uplink
T_RX1, T_RX2 = 0.030, 0.030

HORIZON_DAYS = 5 * 365
STEP_MIN = 60.0                     # simulation resolution
N_STEPS = int(HORIZON_DAYS * 24 * 60 / STEP_MIN)
PAYLOAD_BYTES = 12
SF_CHOICES = (7, 10)                # the two modes actually observed in the fleet
S_MIN, S_MAX = 5.0, 720.0           # sensing interval bounds, minutes
NOMINAL_S = 197.5                   # measured median inter-uplink gap
HEARTBEAT_PER_DAY = 1.0             # measured: ~28.5% of traffic


def lora_airtime(sf, payload=PAYLOAD_BYTES, bw=125e3, cr=1, preamble=8,
                 header=0, crc=1, low_dr=None):
    """Standard LoRa time-on-air, seconds."""
    if low_dr is None:
        low_dr = 1 if (bw == 125e3 and sf >= 11) else 0
    t_sym = (2.0 ** sf) / bw
    t_pre = (preamble + 4.25) * t_sym
    num = 8 * payload - 4 * sf + 28 + 16 * crc - 20 * header
    den = 4 * (sf - 2 * low_dr)
    n_pay = max(int(np.ceil(num / den)) * (cr + 4), 0) + 8
    return t_pre + n_pay * t_sym


AIRTIME = {sf: lora_airtime(sf) for sf in SF_CHOICES}


def uplink_energy(sf):
    """Joules for one Class-A uplink at the given SF."""
    return V_BAT * (I_TX * AIRTIME[sf] + I_MCU * T_MCU
                    + I_RX1 * T_RX1 + I_RX2 * T_RX2)


def sleep_energy(minutes):
    return V_BAT * I_SLEEP * minutes * 60.0


# ---------------------------------------------------------------- arrivals --
class ArrivalProcess:
    """
    Non-stationary vehicle-arrival process over five years.

    Base rate from the measured fleet (1.23-3.07 detections/device/day). Layered
    on top: a seasonal cycle and occasional regime shifts, both of which the
    device cannot forecast. This is the drift that a budget-aware policy has to
    survive and a fixed schedule cannot anticipate.
    """

    def __init__(self, seed, base_per_day=2.2, seasonal_amp=0.45,
                 shift_prob_per_day=1 / 180.0, shift_scale=(0.5, 1.8)):
        self.rng = np.random.default_rng(seed)
        self.base = base_per_day
        self.amp = seasonal_amp
        self.shift_p = shift_prob_per_day
        self.shift_scale = shift_scale
        self.regime = 1.0

    def rate_at(self, day):
        if self.rng.random() < self.shift_p * (STEP_MIN / (24 * 60)) * (24 * 60 / STEP_MIN):
            pass
        seasonal = 1.0 + self.amp * np.sin(2 * np.pi * day / 365.0)
        return self.base * seasonal * self.regime

    def maybe_shift(self, day_frac):
        if self.rng.random() < self.shift_p * day_frac:
            lo, hi = self.shift_scale
            self.regime = float(self.rng.uniform(lo, hi))


# ---------------------------------------------------------------- policies --
def policy_myopic(state):
    return S_MIN, 7


def policy_fixed_ration(state):
    return state['ration_s'], 7


def policy_adr_only(state):
    """Nominal cadence; SF chosen for link margin, with no battery awareness."""
    return NOMINAL_S, state['adr_sf']   # measured fleet cadence, battery-blind


def policy_budget_aware(state):
    """
    Scale the sensing interval by how far ahead or behind the budget we are.

    burn_ratio > 1 means we are spending faster than the remaining budget
    supports, so lengthen the interval; < 1 means we can afford to sense more
    often. Also folds in the recently observed arrival rate, since a busier
    period costs more per unit time.
    """
    rem_e, rem_steps = state['energy_left'], state['steps_left']
    if rem_steps <= 0 or rem_e <= 0:
        return S_MAX, 10
    afford_per_step = rem_e / rem_steps
    recent = max(state['recent_rate'], 1e-6)
    # energy per step at the current interval
    spend = state['last_spend'] if state['last_spend'] > 0 else afford_per_step
    ratio = spend / afford_per_step
    s = state['s'] * float(np.clip(ratio, 0.7, 1.4))
    s *= float(np.clip(recent / max(state['base_rate'], 1e-6), 0.8, 1.25))
    sf = 7 if state['energy_left'] / max(state['budget'], 1e-9) > 0.25 else 10
    return float(np.clip(s, S_MIN, S_MAX)), sf


POLICIES = {
    'myopic': policy_myopic,
    'fixed_ration': policy_fixed_ration,
    'adr_only': policy_adr_only,
    'budget_aware': policy_budget_aware,
}


# -------------------------------------------------------------- simulation --
# Daily resolution. Scans are periodic at interval s, so for Poisson arrivals the
# expected wait for the next scan is exactly s/2 -- computed analytically rather
# than sampled, which removes the discretisation error that broke the previous
# version. Every arriving event is reported at the next scan while the device is
# alive; nothing is dropped. Coverage is therefore the fraction of the horizon
# survived, and the quality metric is detection latency.

FIXED_BUDGET_J = 3.6 * 3600 * 3.6          # 3.6 Ah Li-MnO2 C-cell at 3.6 V


def simulate(policy_fn, budget, seed, radar_scan_j, const_s=None):
    ap = ArrivalProcess(seed)
    rng = np.random.default_rng(90_000 + seed)
    e_left = budget
    s = NOMINAL_S if const_s is None else const_s
    sf = 7
    detected = reported = 0
    lat_sum = 0.0
    recent = []
    days_alive = 0
    ration_s = None

    if policy_fn is policy_fixed_ration:
        def cost(si):
            return ((24 * 60) / si * radar_scan_j
                    + (ap.base + HEARTBEAT_PER_DAY) * uplink_energy(7)
                    + sleep_energy(24 * 60))
        lo, hi = S_MIN, S_MAX
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if cost(mid) * HORIZON_DAYS > budget:
                lo = mid
            else:
                hi = mid
        ration_s = hi

    for day in range(HORIZON_DAYS):
        if e_left <= 0:
            break
        ap.maybe_shift(1.0)
        lam = ap.rate_at(day)
        n_ev = rng.poisson(max(lam, 0.0))
        detected += n_ev
        recent.append(n_ev)
        if len(recent) > 28:
            recent.pop(0)

        state = {'energy_left': e_left, 'budget': budget,
                 'steps_left': HORIZON_DAYS - day, 's': s, 'adr_sf': sf,
                 'recent_rate': float(np.mean(recent)), 'base_rate': ap.base,
                 'last_spend': 0.0,
                 'ration_s': ration_s if ration_s else NOMINAL_S}
        if const_s is None:
            s_new, sf = policy_fn(state)
            s = float(np.clip(s_new, S_MIN, S_MAX))
        else:
            s, sf = const_s, 7

        scans = (24 * 60) / s
        spend = (sleep_energy(24 * 60) + scans * radar_scan_j
                 + (n_ev + HEARTBEAT_PER_DAY) * uplink_energy(sf))
        if spend > e_left:
            break
        e_left -= spend
        state['last_spend'] = spend

        reported += n_ev
        lat_sum += n_ev * (s / 2.0)     # exact mean wait for periodic scans
        days_alive += 1

    return {
        'survived_days': float(days_alive),
        'survived_frac': float(days_alive / HORIZON_DAYS),
        'coverage': float(reported / max(detected, 1)),
        'mean_latency_min': float(lat_sum / max(reported, 1)),
        'energy_used_frac': float((budget - max(e_left, 0.0)) / budget),
    }


def calibrate_budget(radar_scan_j):
    """
    Budget is a property of the cell, not of the radar cost.

    An earlier version sized the budget from the radar cost, which made every
    radar scenario equivalent by construction and tested nothing.
    """
    return FIXED_BUDGET_J


def main():
    print("LoRa airtime: " + ", ".join("SF%d %.1f ms" % (sf, 1000 * AIRTIME[sf])
                                       for sf in SF_CHOICES))
    print("uplink energy: " + ", ".join("SF%d %.2f mJ" % (sf, 1000 * uplink_energy(sf))
                                        for sf in SF_CHOICES))
    print("sleep floor  : %.2f J/day\n" % sleep_energy(24 * 60))

    results = {}
    for radar_scan_j in (0.5e-3, 2.0e-3, 10.0e-3):
        budget = calibrate_budget(radar_scan_j)
        print("=" * 96)
        print("  radar scan = %.1f mJ   -> calibrated budget %.0f J (%.0f mAh @3.7V)"
              % (1000 * radar_scan_j, budget, budget / 3.7 / 3.6))
        print("=" * 96)
        print("  %-16s %12s %10s %14s %12s" % (
            "policy", "survived", "coverage", "latency(min)", "energy used"))
        row = {}
        for name, fn in POLICIES.items():
            runs = [simulate(fn, budget, s, radar_scan_j) for s in range(8)]
            agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            row[name] = agg
            print("  %-16s %11.1f%% %9.3f %14.1f %11.1f%%" % (
                name, 100 * agg['survived_frac'], agg['coverage'],
                agg['mean_latency_min'], 100 * agg['energy_used_frac']))

        # oracle: best constant interval given full knowledge of the horizon
        best = None
        for s_const in np.linspace(S_MIN, S_MAX, 40):
            runs = [simulate(None, budget, sd, radar_scan_j, const_s=float(s_const))
                    for sd in range(8)]
            agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            if agg['survived_frac'] >= 0.999:
                if best is None or agg['mean_latency_min'] < best['mean_latency_min']:
                    best = agg
                    best['s'] = float(s_const)
        if best:
            row['oracle_const'] = best
            print("  %-16s %11.1f%% %9.3f %14.1f %11.1f%%   (s=%.0f min)" % (
                'oracle_const', 100 * best['survived_frac'], best['coverage'],
                best['mean_latency_min'], 100 * best['energy_used_frac'], best['s']))

        ba, fr = row['budget_aware'], row['fixed_ration']
        if ba['survived_frac'] >= 0.999 and fr['survived_frac'] >= 0.999:
            gain = 100 * (fr['mean_latency_min'] - ba['mean_latency_min']) / fr['mean_latency_min']
            print("\n  budget_aware vs fixed_ration, latency: %+.1f%%" % gain)
        else:
            print("\n  survival differs -- compare survival first: budget_aware %.1f%%, "
                  "fixed_ration %.1f%%, adr_only %.1f%%"
                  % (100 * ba['survived_frac'], 100 * fr['survived_frac'],
                     100 * row['adr_only']['survived_frac']))
        results['radar_%.1fmJ' % (1000 * radar_scan_j)] = row
        print()

    with open('lifetime_budget_results.json', 'w') as f:
        json.dump({'horizon_days': HORIZON_DAYS, 'step_min': STEP_MIN,
                   'nominal_s_min': NOMINAL_S, 'results': results}, f, indent=2)
    print("Saved lifetime_budget_results.json")


if __name__ == '__main__':
    main()
