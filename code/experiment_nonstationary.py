"""
Non-stationary evaluation: does online adaptation beat periodic re-optimisation?

This is the experiment the manuscript never ran. Q-learning exists in the
framework in order to adapt, but the reported evaluation was a stationary
500-step run, in which an offline optimiser should win -- and did. Here the
environment drifts continuously and the question becomes the one that actually
matters for a deployed system:

    given that re-solving the allocation costs something, how stale is a
    solution allowed to get before online adaptation is worth more?

Policies compared, all on the same drifting environment and the same yardstick:

  oracle          Pareto grid search re-solved every step (K=1). Upper bound.
  stale_K         the same optimiser re-solved every K steps, holding its action
                  in between. This is what a deployed one-shot optimiser does.
  hybrid          Pareto-constrained Q-learning (online adaptation).
  random_in_shell the P0-1 control: same candidate set as the hybrid, uniform
                  random selection, no learning. Separates "the shell" from
                  "the learning".

Non-stationarity is injected by evolving device state between steps: AR(1)
drift in effective distance (mobility and shadowing combined), regime switches
in traffic arrival, AR(1) drift in radar target range, device join/leave, and
battery depletion from the simulator itself. The physics is untouched; only the
state the physics acts on drifts.

Scoring uses one fixed scalarisation for every policy, with fixed normalisers so
that per-step values are comparable over time (a per-step min-max normalisation
would make regret meaningless). Regret at step t is J(policy) - J(oracle) on the
same seed and the same realised drift.

Outputs nonstationary_results.json.

Author: Vullnet Laniku
"""

import json

import numpy as np

from closed_loop_simulator import (ClosedLoopISACSimulator, ISACAction,
                                   SimulatorDeviceState)
from integrated_models import ParetoGridSelector, HybridModelWrapper
from adaptive_hybrid import AdaptiveHybrid

N_DEVICES = 14
N_STEPS = 2000
N_SEEDS = 5
STALE_PERIODS = (10, 50, 100)

# Fixed scalarisation. Weights match the Q-learning reward in the manuscript;
# their sensitivity is a separate experiment (P0-7). Normalisers are fixed
# constants rather than per-step statistics so J is comparable across time.
W = {'energy': 0.25, 'accuracy': 0.25, 'reliability': 0.25, 'latency': 0.25}
E_REF = 0.10   # J/step
L_REF = 0.01   # s


def objective(o):
    """Lower is better. One yardstick for every policy (simulator outcome)."""
    return (W['energy'] * (o['energy_joules'] / E_REF)
            + W['accuracy'] * (1.0 - o['accuracy'])
            + W['reliability'] * (1.0 - o['reliability'])
            + W['latency'] * (o['latency'] / L_REF))


def objective_from_perf(p):
    """Same scalarisation, applied to a predicted performance dict."""
    return (W['energy'] * (p['energy_consumption'] / E_REF)
            + W['accuracy'] * (1.0 - p['sensing_accuracy'])
            + W['reliability'] * (1.0 - p['communication_reliability'])
            + W['latency'] * (p['latency'] / L_REF))


class JOracle:
    """
    Per-step optimum of the *evaluation* objective over the action grid.

    The grid selector optimises a battery-adaptive Chebyshev scalarisation,
    which is not the yardstick used here, so it cannot serve as the regret
    reference -- other policies routinely beat it on J. This class enumerates
    the same grid under the same physical-model assumptions and selects the
    action minimising J directly, making it the correct reference.

    It is myopic: it optimises the current step and ignores battery dynamics,
    so it is an instantaneous-allocation oracle rather than a globally optimal
    policy. Regret against it is therefore a lower bound on achievable quality
    per step, which is what the staleness comparison needs.
    """

    def __init__(self, mac, n_devices):
        self.sel = ParetoGridSelector(mac_mode=mac, n_devices=n_devices)

    def predict(self, states):
        out = {}
        for did, st in states.items():
            best, best_j = None, np.inf
            for sp in self.sel.POWER_LEVELS:
                for cp in self.sel.POWER_LEVELS:
                    for bw in self.sel.BW_RATIOS:
                        j = objective_from_perf(
                            self.sel._evaluate_action(sp, cp, bw, st))
                        if j < best_j:
                            best_j, best = j, ISACAction(sp, cp, bw, 1.0)
            out[did] = best
        return out


class NonStationaryScenario:
    """
    Drives continuous drift over the device population.

    Effective distance follows an AR(1) process, standing in for combined
    mobility and correlated shadowing; traffic switches between regimes at
    random times; radar target range drifts; devices join and leave. Time
    constants are long relative to a step so the drift is trackable -- pure
    noise would be untrackable by any policy and would make the comparison
    vacuous.
    """

    DIST_TAU = 200.0        # steps; AR(1) time constant for distance
    DIST_SIGMA = 0.18       # stationary sd as a fraction of nominal distance
    RANGE_TAU = 300.0
    RANGE_SIGMA = 0.15
    TRAFFIC_DWELL = 400.0   # mean steps between traffic regime switches
    LEAVE_P = 1.0 / 1500.0  # per-device per-step probability of toggling state
    REJOIN_P = 1.0 / 300.0

    def __init__(self, devices, seed):
        self.rng = np.random.default_rng(10_000 + seed)
        self.devices = devices
        self.nominal_d = {d.device_id: d.distance_to_base for d in devices}
        self.nominal_r = {d.device_id: d.target_range for d in devices}
        self.log_d = {d.device_id: 0.0 for d in devices}
        self.log_r = {d.device_id: 0.0 for d in devices}
        self.active = {d.device_id: True for d in devices}
        self.arrival = 0.05
        self.trace = []

    def advance(self):
        a_d = np.exp(-1.0 / self.DIST_TAU)
        a_r = np.exp(-1.0 / self.RANGE_TAU)
        s_d = self.DIST_SIGMA * np.sqrt(1 - a_d ** 2)
        s_r = self.RANGE_SIGMA * np.sqrt(1 - a_r ** 2)

        if self.rng.random() < 1.0 / self.TRAFFIC_DWELL:
            self.arrival = float(self.rng.choice([0.01, 0.05, 0.15, 0.30]))

        for d in self.devices:
            did = d.device_id
            self.log_d[did] = a_d * self.log_d[did] + s_d * self.rng.normal()
            self.log_r[did] = a_r * self.log_r[did] + s_r * self.rng.normal()
            d.distance_to_base = float(np.clip(
                self.nominal_d[did] * np.exp(self.log_d[did]), 20.0, 600.0))
            d.target_range = float(np.clip(
                self.nominal_r[did] * np.exp(self.log_r[did]), 20.0, 400.0))

            if self.active[did]:
                if self.rng.random() < self.LEAVE_P:
                    self.active[did] = False
            elif self.rng.random() < self.REJOIN_P:
                self.active[did] = True

            if self.rng.random() < self.arrival:
                d.queue_length = min(d.queue_length + 1, 20)

        self.trace.append({'arrival': self.arrival,
                           'n_active': int(sum(self.active.values()))})

    def active_states(self, sim):
        s = sim.get_current_states()
        return {k: v for k, v in s.items() if self.active[k]}


class StalePolicy:
    """Re-solve every K steps; hold the previous action in between (K=inf: never)."""

    def __init__(self, inner, period):
        self.inner = inner
        self.period = period
        self.t = 0
        self.cache = {}

    def predict(self, states):
        due = (self.period == 1) or (self.t % self.period == 0)
        if due or not self.cache:
            self.cache = self.inner.predict(states)
        self.t += 1
        # devices with no cached action (joined since last solve) get one now
        missing = {k: v for k, v in states.items() if k not in self.cache}
        if missing:
            self.cache.update(self.inner.predict(missing))
        return {k: self.cache[k] for k in states}


class RandomInShell:
    """Hybrid's candidate set, uniform random selection, no learning."""

    def __init__(self, mac, n_devices, seed):
        self.h = HybridModelWrapper(mac_mode=mac, n_devices=n_devices)
        self.rng = np.random.default_rng(50_000 + seed)

    def predict(self, states):
        self.h.epsilon = 1.0          # always explore == uniform over candidates
        return self.h.predict(states)


def make_devices(n, seed):
    rng = np.random.default_rng(seed)
    devs = []
    for i in range(n):
        x, y = rng.uniform(-300, 300), rng.uniform(-300, 300)
        devs.append(SimulatorDeviceState(
            device_id=f"d{i}", battery_level=float(rng.uniform(0.5, 1.0)),
            battery_capacity_joules=7400.0, location=(float(x), float(y)),
            queue_length=int(rng.integers(0, 5)),
            distance_to_base=float(np.hypot(x, y)),
            target_range=float(rng.uniform(40, 250))))
    return devs


def run_policy(make_policy, mac, seed, label):
    """One episode. Returns per-step objective and mean raw metrics."""
    sim = ClosedLoopISACSimulator(make_devices(N_DEVICES, seed), mac_mode=mac)
    scen = NonStationaryScenario(list(sim.devices.values()), seed)
    policy = make_policy(seed)
    if hasattr(policy, 'train'):
        try:
            policy.train(None, seed=seed)
        except Exception:
            pass

    js, raw = [], {'energy': 0.0, 'accuracy': 0.0, 'reliability': 0.0, 'n': 0}
    for t in range(N_STEPS):
        scen.advance()
        states = scen.active_states(sim)
        if not states:
            js.append(np.nan)
            continue
        actions = policy.predict(states)
        out = sim.step(actions)
        step_j = []
        for did, o in out.items():
            step_j.append(objective(o))
            raw['energy'] += o['energy_joules']
            raw['accuracy'] += o['accuracy']
            raw['reliability'] += o['reliability']
            raw['n'] += 1
            if hasattr(policy, 'update_q'):
                r = (-o['energy_joules'] / 0.1 + o['accuracy']
                     + o['reliability'] - o['latency'] / 0.01)
                policy.update_q(did, r, sim.get_device(did))
        js.append(float(np.mean(step_j)))
    n = max(raw['n'], 1)
    return (np.array(js),
            {k: raw[k] / n for k in ('energy', 'accuracy', 'reliability')})


def main(mac='ofdma'):
    builders = {
        'oracle_K1': lambda s: JOracle(mac, N_DEVICES),
        'chebyshev_K1': lambda s: ParetoGridSelector(mac_mode=mac, n_devices=N_DEVICES),
        'hybrid': lambda s: HybridModelWrapper(mac_mode=mac, n_devices=N_DEVICES),
        'hybrid_fixed': lambda s: AdaptiveHybrid(mac_mode=mac, n_devices=N_DEVICES),
        'random_in_shell': lambda s: RandomInShell(mac, N_DEVICES, s),
    }
    for K in STALE_PERIODS:
        builders[f'stale_K{K}'] = (
            lambda s, K=K: StalePolicy(JOracle(mac, N_DEVICES), K))

    print("=" * 96)
    print("  NON-STATIONARY EVALUATION   mac=%s  %d devices  %d steps  %d seeds"
          % (mac.upper(), N_DEVICES, N_STEPS, N_SEEDS))
    print("=" * 96)

    curves, summary = {}, {}
    for name, mk in builders.items():
        per_seed_j, per_seed_raw = [], []
        for s in range(N_SEEDS):
            j, raw = run_policy(mk, mac, s, name)
            per_seed_j.append(j)
            per_seed_raw.append(raw)
        curves[name] = np.vstack(per_seed_j)
        summary[name] = {k: float(np.mean([r[k] for r in per_seed_raw]))
                         for k in ('energy', 'accuracy', 'reliability')}
        summary[name]['J_mean'] = float(np.nanmean(curves[name]))
        print("  %-18s J=%.4f  energy=%.4f  acc=%.4f  rel=%.4f"
              % (name, summary[name]['J_mean'], summary[name]['energy'],
                 summary[name]['accuracy'], summary[name]['reliability']))

    base = curves['oracle_K1']
    print("\n" + "-" * 96)
    print("  TRACKING REGRET vs oracle (K=1).  positive = worse than oracle")
    print("  %-18s %12s %12s %14s %14s" % (
        "policy", "mean regret", "sd", "regret 1st half", "regret 2nd half"))
    half = N_STEPS // 2
    for name, c in curves.items():
        r = c - base
        summary[name]['regret_mean'] = float(np.nanmean(r))
        summary[name]['regret_sd'] = float(np.nanstd(np.nanmean(r, axis=1)))
        summary[name]['regret_h1'] = float(np.nanmean(r[:, :half]))
        summary[name]['regret_h2'] = float(np.nanmean(r[:, half:]))
        q = N_STEPS // 4
        summary[name]['regret_q'] = [float(np.nanmean(r[:, i * q:(i + 1) * q]))
                                     for i in range(4)]
        # downsampled mean regret curve, for the time-series figure
        blk = max(1, N_STEPS // 100)
        m = np.nanmean(r, axis=0)
        summary[name]['regret_curve'] = [
            float(np.nanmean(m[i:i + blk])) for i in range(0, N_STEPS, blk)]
        print("  %-18s %12.5f %12.5f %14.5f %14.5f   Q:%s" % (
            name, summary[name]['regret_mean'], summary[name]['regret_sd'],
            summary[name]['regret_h1'], summary[name]['regret_h2'],
            " ".join("%.4f" % x for x in summary[name]['regret_q'])))

    print("\n  DECISION:")
    hy, ri = summary['hybrid']['regret_mean'], summary['random_in_shell']['regret_mean']
    print("    hybrid vs random-in-shell : %.5f vs %.5f -> %s"
          % (hy, ri, "learning helps" if hy < ri else "LEARNING ADDS NOTHING"))
    beaten = [K for K in STALE_PERIODS
              if summary['hybrid']['regret_mean'] < summary[f'stale_K{K}']['regret_mean']]
    print("    hybrid beats stale re-solve at K in %s (of %s)"
          % (beaten if beaten else "NONE", list(STALE_PERIODS)))

    with open('nonstationary_results_%s.json' % mac, 'w') as f:
        json.dump({'mac': mac, 'n_devices': N_DEVICES, 'n_steps': N_STEPS,
                   'n_seeds': N_SEEDS, 'weights': W,
                   'e_ref': E_REF, 'l_ref': L_REF,
                   'stale_periods': list(STALE_PERIODS),
                   'summary': summary}, f, indent=2)
    print("\nSaved nonstationary_results_%s.json" % mac)


if __name__ == '__main__':
    import sys
    for m in (sys.argv[1:] or ['tdma', 'ofdma']):
        main(m)
