"""
Pareto-constrained Q-learning with the three diagnosed defects corrected.

Instrumenting the original HybridModelWrapper over 2000 drifting steps showed
why it never improves:

  1. State aliasing. 80 of 110 visited state keys mapped to more than one
     optimiser base action (up to ten). Since the *correct* deviation depends
     on which base action was chosen, the learner was averaging over
     contradictory experience. Fixed here by making the base action part of the
     state, so a Q-value means "deviate by (dp, dc) given this base", which is
     a well-posed question.

  2. Asymmetric action set. Under TDMA the original builds perturbations as
     `for dp in (0, 3): for dc in (0, 3)`, i.e. upward power offsets only, so
     every available deviation costs energy and the learner cannot improve on
     its own baseline -- only match it, minus exploration cost. Fixed here by
     making the offset set symmetric under both MAC schemes.

  3. Non-vanishing exploration and constant step size. Epsilon floored at
     0.03-0.05 forever and alpha was fixed at 0.15, so the policy kept paying
     for exploration it could no longer use and the estimates never settled.
     Fixed here with visit-count decayed alpha (Robbins-Monro) and epsilon
     decaying toward a negligible floor.

This is deliberately a separate class: the original is retained so the
before/after comparison is reproducible.

Author: Vullnet Laniku
"""

from typing import Dict, Optional, Sequence

import numpy as np

from closed_loop_simulator import ISACAction, SimulatorDeviceState
from integrated_models import ParetoGridSelector

# Symmetric offsets, applied to the optimiser's base action (dBm).
OFFSETS = tuple((dp, dc) for dp in (-3, 0, 3) for dc in (-3, 0, 3))

# State discretisation. Finer than the original, and the base action is folded
# in, which is what actually removes the aliasing.
BATTERY_STEP = 0.1
DISTANCE_STEP = 50.0
QUEUE_CAP = 5


class AdaptiveHybrid:
    """Grid search proposes; a correctly specified Q-learner adjusts."""

    def __init__(self,
                 mac_mode: str = 'ofdma',
                 n_devices: int = 14,
                 frequency_hz: float = 2.4e9,
                 gamma: float = 0.9,
                 eps0: float = 0.30,
                 eps_tau: float = 400.0,
                 eps_floor: float = 0.005,
                 alpha_exponent: float = 0.7,
                 selector_power_levels: Optional[Sequence[float]] = None):
        self.selector = ParetoGridSelector(
            mac_mode=mac_mode, n_devices=n_devices, frequency_hz=frequency_hz,
            power_levels=selector_power_levels)
        self.mac_mode = mac_mode.lower()
        self.gamma = float(gamma)
        self.eps0 = float(eps0)
        self.eps_tau = float(eps_tau)
        self.eps_floor = float(eps_floor)
        self.alpha_exponent = float(alpha_exponent)
        self.q = {}          # state_key -> np.array over OFFSETS
        self.n_sa = {}       # state_key -> visit counts per action
        self.prev = {}       # device_id -> (state_key, action_index)
        self.t = 0

    # ------------------------------------------------------------------ #
    def train(self, data=None, seed=None):
        self.q.clear()
        self.n_sa.clear()
        self.prev.clear()
        self.t = 0

    @property
    def epsilon(self) -> float:
        return max(self.eps_floor, self.eps0 / (1.0 + self.t / self.eps_tau))

    @staticmethod
    def _state_key(state: SimulatorDeviceState, base: ISACAction):
        """
        Device state *and* the base action being adjusted.

        Including the base is the fix for defect 1: without it, one key covers
        many different optimiser choices and the right deviation differs
        between them.
        """
        return (round(float(state.battery_level) / BATTERY_STEP),
                round(float(state.distance_to_base) / DISTANCE_STEP),
                min(int(state.queue_length), QUEUE_CAP),
                round(float(base.sensing_power_db)),
                round(float(base.comm_power_db)),
                round(float(base.sensing_bandwidth_ratio), 2))

    def _candidates(self, base: ISACAction):
        out = []
        for dp, dc in OFFSETS:
            out.append(ISACAction(
                float(np.clip(base.sensing_power_db + dp, 10, 20)),
                float(np.clip(base.comm_power_db + dc, 10, 20)),
                base.sensing_bandwidth_ratio, 1.0))
        return out

    # ------------------------------------------------------------------ #
    def predict(self, states: Dict[str, SimulatorDeviceState]) -> Dict[str, ISACAction]:
        actions = {}
        eps = self.epsilon
        for did, st in states.items():
            base = self.selector.predict({did: st})[did]
            sk = self._state_key(st, base)
            if sk not in self.q:
                self.q[sk] = np.zeros(len(OFFSETS))
                self.n_sa[sk] = np.zeros(len(OFFSETS))

            if np.random.random() < eps:
                ai = int(np.random.randint(len(OFFSETS)))
            else:
                ai = int(np.argmax(self.q[sk]))

            actions[did] = self._candidates(base)[ai]
            self.prev[did] = (sk, ai)
        self.t += 1
        return actions

    def update_q(self, device_id, reward, next_state):
        if device_id not in self.prev:
            return
        sk, ai = self.prev[device_id]
        self.n_sa[sk][ai] += 1
        # Robbins-Monro step size: sum(alpha) diverges, sum(alpha^2) converges.
        alpha = 1.0 / (1.0 + self.n_sa[sk][ai]) ** self.alpha_exponent
        # Bootstrap off the successor state under the action the selector would
        # propose there; if unseen its value is zero, which is the correct
        # optimistic-free initialisation for this reward scale.
        nxt = 0.0
        if next_state is not None:
            nb = self.selector.predict({device_id: next_state})[device_id]
            nk = self._state_key(next_state, nb)
            if nk in self.q:
                nxt = float(np.max(self.q[nk]))
        self.q[sk][ai] += alpha * (reward + self.gamma * nxt - self.q[sk][ai])

    # ------------------------------------------------------------------ #
    def diagnostics(self):
        if not self.q:
            return {}
        spreads = [float(v.max() - v.min()) for v in self.q.values() if len(v) > 1]
        greedy = [int(np.argmax(v)) for v in self.q.values()]
        return {
            'n_states': len(self.q),
            'epsilon': self.epsilon,
            'q_spread_median': float(np.median(spreads)) if spreads else 0.0,
            'greedy_is_base_pct': 100.0 * float(np.mean(
                [g == OFFSETS.index((0, 0)) for g in greedy])),
        }
