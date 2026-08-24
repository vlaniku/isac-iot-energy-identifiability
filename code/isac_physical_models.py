"""
ISAC Physical Models - Rigorous sensing and communication models
Author: Vullnet Laniku
Research: Energy-Efficient Dynamic Resource Allocation for IoT-Enabled ISAC Systems
"""

import numpy as np
import scipy.stats
import scipy.special
from typing import Tuple, Dict, Any


# ISAC sensing-communication coupling coefficient. Sensing accuracy is scaled
# by (1 - beta * P_c / (P_s + P_c)), so beta prices the conflict between the two
# functions sharing one front end. The value is empirical; reviewers asked
# whether conclusions depend on it, hence it is a module-level parameter that
# can be swept (see experiment_beta_sweep.py) rather than a literal.
ISAC_COUPLING_BETA = 0.15

# Static per-step processing power, added to every action's energy and so
# constant across the action set. It was a hard-coded 0.02 W literal inside
# isac_joint_performance with no citation, and comm_action_bracket.py showed
# the energy-identity bracket moves 0.777 -> 0.101 over 0 -> 100 mW of it.
# Promoted to a module-level parameter for the same reason as the beta above:
# an uncited constant that moves a reported number must be sweepable.
# Default unchanged, so every prior result reproduces.
P_PROC_W = 0.02


def _q(x):
    """Gaussian Q-function, Q(x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * scipy.special.erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def _mqam_ber(snr, m_order):
    """
    Square M-QAM bit error rate at received symbol SNR, standard approximation:

        BER ~= (4 / log2(M)) * (1 - 1/sqrt(M)) * Q(sqrt(3 * snr / (M - 1)))
    """
    k = np.log2(m_order)
    return ((4.0 / k) * (1.0 - 1.0 / np.sqrt(m_order))
            * _q(np.sqrt(3.0 * snr / (m_order - 1.0))))


class ISACPhysicalModels:
    """Physical models for ISAC sensing and communication (3GPP UMi-style path loss, IoT PER)."""

    def __init__(self):
        self.speed_of_light = 3.0e8
        self.boltzmann_constant = 1.38e-23
        self.temperature = 290.0
        self.noise_power_density = self.boltzmann_constant * self.temperature

    def path_loss_model(
        self, distance: float, frequency: float, environment: str = "urban"
    ) -> float:
        d0 = 1.0
        n = {"urban": 2.8, "suburban": 2.5, "rural": 2.2}[environment]
        wavelength = self.speed_of_light / frequency
        pl_d0 = 20 * np.log10(4 * np.pi * d0 / wavelength)
        pl = pl_d0 + 10 * n * np.log10(max(distance, d0) / d0)
        shadowing = 0.0
        return pl + shadowing

    def sensing_detection_probability(
        self,
        transmit_power_db: float,
        bandwidth: float,
        target_range: float,
        target_rcs: float = 1.0,
        frequency: float = 2.4e9,
        pulse_duration: float = 1e-6,
        desired_pfa: float = 1e-6,
        antenna_gain_db: float = 20.0,
    ) -> Tuple[float, float]:
        transmit_power_watt = 10 ** ((transmit_power_db - 30) / 10)
        wavelength = self.speed_of_light / frequency
        ag = 10 ** (antenna_gain_db / 10.0)
        numerator = transmit_power_watt * (ag**2) * (wavelength**2) * target_rcs
        denominator = ((4 * np.pi) ** 3) * (target_range**4) * self.noise_power_density * bandwidth
        if denominator == 0:
            return 0.0, desired_pfa
        snr_linear = numerator / denominator
        if snr_linear > 0:
            threshold = -2 * np.log(max(desired_pfa, 1e-20))
            pd = 1.0 - float(
                scipy.stats.norm.cdf(np.sqrt(threshold) - np.sqrt(2 * snr_linear))
            )
            pd = float(np.clip(pd, 0.0, 1.0))
        else:
            pd = 0.0
        pfa = desired_pfa
        return pd, pfa

    def sensing_accuracy_from_detection(
        self, pd: float, pfa: float, target_density: float = 0.5
    ) -> float:
        rho = target_density
        acc = rho * pd + (1.0 - rho) * (1.0 - pfa)
        return float(np.clip(acc, 0.0, 1.0))

    def communication_sinr(
        self,
        transmit_power_db: float,
        distance: float,
        frequency: float,
        interference_power_db: float,
        bandwidth: float,
        environment: str = "urban",
    ) -> float:
        path_loss_db = self.path_loss_model(max(distance, 1.0), frequency, environment)
        received_power_db = transmit_power_db - path_loss_db
        received_power_watt = 10 ** ((received_power_db - 30) / 10)
        if interference_power_db > -200:
            interference_watt = 10 ** ((interference_power_db - 30) / 10)
        else:
            interference_watt = 0.0
        noise_watt = self.noise_power_density * bandwidth
        tot = max(noise_watt + interference_watt, 1e-30)
        sinr = received_power_watt / tot
        return 10 * np.log10(sinr) if sinr > 0 else -100.0

    def packet_error_rate(
        self,
        sinr_db: float,
        modulation: str = "QAM16",
        coding_rate: float = 0.75,
        packet_length: int = 256,
    ) -> float:
        snr = 10 ** (sinr_db / 10.0)
        # All expressions are in terms of received symbol SNR.
        #
        # The M-QAM branches previously read
        #     QAM16: 1.5  * erfc(sqrt(0.1 *  5.0 * snr / 2)) * (1 - 1/16)
        #     QAM64: 3.5  * erfc(sqrt(0.1 * 21.0 * snr / 2)) * (1 - 1/64)
        # whose in-Q factor 0.1*(M-1)/3 (0.5 and 2.1) is the reciprocal of the
        # standard 3/(M-1) (0.2 and 0.0476), scaled by 0.1. That made the QAM64
        # argument ~44x too large in SNR terms, so QAM64 cleared 90% packet
        # success at 8.25 dB while QPSK needed 10.15 dB -- an impossible
        # ordering for a denser constellation, and the reason packet success
        # was non-monotone in SINR.
        ber_m = {
            "BPSK": lambda: _q(np.sqrt(2.0 * snr)),
            "QPSK": lambda: _q(np.sqrt(snr)),
            "QAM16": lambda: _mqam_ber(snr, 16),
            "QAM64": lambda: _mqam_ber(snr, 64),
        }
        ber = ber_m.get(modulation, ber_m["QAM16"])()
        coded_ber = max(0.0, min(0.5, ber * (1.0 - coding_rate * 0.5)))
        per = 1.0 - (1.0 - coded_ber) ** packet_length
        return float(np.clip(per, 0.0, 1.0))

    def communication_reliability(
        self,
        transmit_power_db: float,
        distance: float,
        frequency: float,
        interference_power_db: float,
        bandwidth: float,
        modulation: str = "QAM16",
        coding_rate: float = 0.75,
        packet_length: int = 256,
        environment: str = "urban",
    ) -> float:
        sinr = self.communication_sinr(
            transmit_power_db,
            distance,
            frequency,
            interference_power_db,
            bandwidth,
            environment,
        )
        per = self.packet_error_rate(
            sinr, modulation, coding_rate, packet_length=packet_length
        )
        return float(np.clip(1.0 - per, 0.0, 1.0))

    def latency_model(
        self,
        queue_length: int,
        packet_size: int,
        bandwidth: float,
        sinr_db: float,
        distance: float,
        arrival_rate: float = 10.0,
        rate_fraction: float = 1.0,
    ) -> float:
        snr = 10 ** (sinr_db / 10.0) if sinr_db > -100 else 0.0
        if snr > 0:
            data_rate = (
                float(bandwidth) * float(np.log2(1.0 + snr)) * float(rate_fraction)
            )
        else:
            data_rate = max(float(bandwidth) * 0.01 * float(rate_fraction), 1e-9)
        if data_rate <= 0:
            return 1.0
        prop = distance / self.speed_of_light
        if queue_length == 0:
            qd = 0.0
        else:
            qd = 0.001 * queue_length
        return min(
            float(qd + (packet_size / data_rate) + prop + 0.001) if data_rate > 0 else 1.0, 1.0
        )

    def isac_joint_performance(
        self,
        sensing_power_db: float,
        comm_power_db: float,
        shared_bandwidth: float,
        sensing_bandwidth_ratio: float,
        target_range: float = 100.0,
        target_rcs: float = 1.0,
        distance: float = 100.0,
        frequency: float = 2.4e9,
        queue_length: int = 0,
        packet_size: int = 256,
        interference_power_db: float = -100.0,
        environment: str = "urban",
        modulation: str = "QAM16",
        coding_rate: float = 0.75,
        tdma_slot_fraction: float = 1.0,
        beta: float = None,
    ) -> Dict[str, float]:
        sensing_bw = shared_bandwidth * sensing_bandwidth_ratio
        comm_bw = shared_bandwidth * (1.0 - sensing_bandwidth_ratio)
        pd, pfa = self.sensing_detection_probability(
            sensing_power_db,
            sensing_bw,
            target_range,
            target_rcs=target_rcs,
            frequency=frequency,
        )
        sensing_acc = self.sensing_accuracy_from_detection(pd, pfa, target_density=0.5)
        s_w = 10 ** ((sensing_power_db - 30) / 10)
        c_w = 10 ** ((comm_power_db - 30) / 10)
        ptot = s_w + c_w
        b = ISAC_COUPLING_BETA if beta is None else float(beta)
        if ptot > 0:
            sensing_acc = sensing_acc * max(0.1, 1.0 - b * (c_w / ptot))
        path = self.path_loss_model(max(distance, 1.0), frequency, environment)
        s_int_db = sensing_power_db - 20.0 - path
        s_w_r = 10 ** ((s_int_db - 30) / 10) if s_int_db > -200 else 0.0
        ext_w = 10 ** ((interference_power_db - 30) / 10) if interference_power_db > -200 else 0.0
        t_int = max(s_w_r, 0.0) + max(ext_w, 0.0)
        t_db = 10 * np.log10(t_int * 1000) if t_int > 0 else -200.0
        pre = self.communication_sinr(comm_power_db, distance, frequency, t_db, comm_bw, environment)
        # Link adaptation at a 10% block-error target, the convention used in
        # LTE/NR link adaptation. Switch points are the SINR at which each
        # constellation first clears 90% packet success for a 256-bit payload
        # at coding rate 0.75, measured against the BER model above.
        # Previous thresholds were 18 / 12 / 5 dB, i.e. 2-6 dB below the SINR
        # each constellation actually needs, so the model switched up too early
        # and packet success collapsed just above every threshold.
        if pre >= 22.95:
            m = "QAM64"
        elif pre >= 16.90:
            m = "QAM16"
        elif pre >= 10.15:
            m = "QPSK"
        else:
            m = "BPSK"
        psize = int(min(int(packet_size), 256))
        comm_rel = self.communication_reliability(
            comm_power_db, distance, frequency, t_db, comm_bw, m, coding_rate, psize, environment
        )
        snr = self.communication_sinr(comm_power_db, distance, frequency, t_db, comm_bw, environment)
        fr = float(tdma_slot_fraction)
        lat = self.latency_model(
            queue_length, psize, comm_bw, snr, distance, rate_fraction=fr
        )
        p_proc = P_PROC_W
        e = (s_w + c_w) * fr + p_proc
        totp = s_w + c_w + p_proc
        return {
            "sensing_accuracy": float(sensing_acc),
            "communication_reliability": comm_rel,
            "latency": lat,
            "energy_consumption": e,
            "sensing_pd": float(pd),
            "sensing_pfa": float(pfa),
            "comm_sinr_db": float(snr),
            "sensing_power_watt": float(s_w),
            "comm_power_watt": float(c_w),
            "total_power_watt": float(totp),
        }
