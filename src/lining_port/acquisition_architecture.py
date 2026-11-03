"""Input-referred white noise for passive and multiplexed strain combinations.

The digital architecture has four receivers and individually completed quarter
bridges. Only the four bridges being read are excited. Ideal switches, exact
calibration and no settling overhead favor the digital comparison. All networks
measure the same four unit strain rows of the representative frame.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bridge_network import Circuit
from .passive_acquisition_study import routed_candidate
from .passive_bridge import BOLTZMANN
from .passive_design_study import restore_network
from .passive_resources import acquisition_variances
from .passive_robustness_study import distances, source_data


def allocate_samples(variance_weights, total):
    """Minimize sum(a_i / N_i) over positive integer N_i with a fixed sum.

    Diminishing marginal reductions a_i/[N_i(N_i+1)] make greedy allocation
    exact for this separable integer problem. Zero-weight inputs are bypassed.
    """
    a = np.asarray(variance_weights, dtype=float)
    counts = (a > 0).astype(int)
    if total < counts.sum():
        raise ValueError("At least one sample is needed for each active input")
    for _ in range(total - counts.sum()):
        benefit = np.divide(a, counts * (counts + 1), out=np.zeros_like(a), where=counts > 0)
        counts[np.argmax(benefit)] += 1
    return counts


def run():
    config = json.loads(Path("inputs/passive_robustness_task.json").read_text())
    acquisition = json.loads(Path("inputs/passive_readout_task.json").read_text())
    data = next(d for d in source_data() if d["name"] == "frame")
    layout, channels = 6, 4
    with np.load("results/passive-robustness/hardware.npz") as saved, \
            np.load("results/passive-network/constrained.npz") as arrays:
        resistance_ratio = saved["frame_actual_nominal_ratio"][layout]
        prefix = "frame_l6_constrained"
        network = restore_network(arrays, prefix, resistance_ratio)
        circuit = Circuit(gauge_factor=50., voltage=float(arrays[prefix + "_supply_voltage_v"]) / 2)
        candidate = routed_candidate(network, circuit, distances(data, layout), data["depth"][layout], config)
        variance = sum(acquisition_variances(candidate["hardware"], config, acquisition))
        norms = np.array([np.linalg.norm(b.weight) for b in network])
        gains = np.array([h.nominal_linear_gain() for h in candidate["hardware"]]) * norms
        U = np.zeros((channels, len(resistance_ratio)))
        for q, bridge in enumerate(network):
            U[q, bridge.gauge_indices] = bridge.weight / norms[q]
        power_limit = float(np.max(arrays["frame_l6_baseline_design_gauge_peak_power_w"]))
        nominal_power = json.loads(Path("results/passive-network/constrained.json").read_text())["records"]
        source_limit = next(r["budget"]["source_power_w"] for r in nominal_power
                            if r["structure"] == "frame" and r["layout"] == layout)
        # Four equally balanced quarter bridges dissipate four times the power
        # of their one active gauge. Excitation is chosen at the tighter limit.
        gauge_power = min(power_limit, source_limit / (4 * channels))
        gauge_resistance = circuit.resistance * resistance_ratio
        excitation = np.sqrt(4 * gauge_power * gauge_resistance)
        digital_gain = excitation * circuit.gauge_factor / 4 / (
            1 + gauge_resistance / config["differential_input_resistance_ohm"])
        delta_t = acquisition["sample_period_s"]
        counts_per_average = int(round(config["average_time_s"] / delta_t))
        if acquisition["reference_average_time_s"] != config["average_time_s"]:
            raise ValueError("This comparison uses equal reference and service windows")
        # The high-impedance input sees the nominal quarter-bridge resistance.
        # Leads and switch resistance are zero in this idealized alternative.
        output_resistance = gauge_resistance / (1 + gauge_resistance / config["differential_input_resistance_ohm"])
        sample_variance = (2 * BOLTZMANN * acquisition["temperature_k"] * output_resistance / delta_t
                           + acquisition["independent_sample_sigma_v"]**2)
        input_sample_variance = sample_variance / digital_gain**2
        uniform_variance, optimal_variance, schedules = [], [], []
        for q, bridge in enumerate(network):
            group = bridge.gauge_indices
            a = U[q, group]**2 * input_sample_variance[group]
            uniform = np.full(len(group), counts_per_average // len(group), dtype=int)
            uniform[:counts_per_average % len(group)] += 1
            optimal = allocate_samples(a, counts_per_average)
            uniform_variance.append(2 * np.sum(a / uniform))
            optimal_variance.append(2 * np.divide(a, optimal, out=np.zeros_like(a), where=optimal > 0).sum())
            schedules.append({"gauge_indices": group.tolist(), "uniform_samples": uniform.tolist(),
                              "target_samples": optimal.tolist()})
        passive_variance = variance / gains**2
        target = data["design"][:, :, layout] - data["calibration"][:, layout]
        signal_energy = float(np.mean(np.sum((target @ U.T)**2, axis=-1)))
        models = []
        for label, value in (("passive", passive_variance), ("multiplexed_uniform", uniform_variance),
                             ("multiplexed_target_allocation", optimal_variance)):
            value = np.asarray(value)
            models.append({"architecture": label, "channel_noise_rms_microstrain": (1e6 * np.sqrt(value)).tolist(),
                           "noise_norm_rms_microstrain": float(1e6 * np.sqrt(value.sum())),
                           "design_signal_to_white_noise_energy": signal_energy / float(value.sum())})
        result = {
            "structure": "frame", "installation_seed": 8206, "receivers": channels,
            "nominal_source_power_limit_uW": source_limit * 1e6,
            "single_gauge_power_limit_uW": power_limit * 1e6,
            "digital_excited_source_power_uW": 4 * channels * gauge_power * 1e6,
            "digital_excitation_range_V": [float(excitation.min()), float(excitation.max())],
            "average_time_s": config["average_time_s"], "sample_period_s": delta_t,
            "samples_per_receiver_per_average": counts_per_average,
            "same_direction_signal_rms_microstrain": 1e6 * np.sqrt(signal_energy),
            "passive_unit_direction_gains": gains.tolist(), "models": models, "schedules": schedules,
            "scope": "Nominal linear response with Johnson and independent receiver white noise; "
                     "reference and current means have independent noise. Digital bridges are switched "
                     "at calibrated excitation levels, without settling loss, wire loss, switching error "
                     "or completion-resistor error. Receiver, switching and processing power are not modeled.",
        }
        path = Path("results/passive-network/acquisition_architecture.json")
        path.write_text(json.dumps(result, indent=2) + "\n")
        return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({k: v for k, v in result.items() if k != "schedules"}, indent=2))
