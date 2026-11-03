"""Continuous arm scales with a common source and matched excitation power."""
from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from .passive_bridge import BOLTZMANN
from .passive_hardware import BridgeHardware
from .passive_network import network_energies, regional_log_ratio


def minimum_arm_units(bridge, circuit, route):
    cells = np.sqrt(bridge.resistance_ratio * bridge.attenuation)
    return float(max(cells[bridge.arm == j].sum()
                     + route["arm_resistance_ohm"][j] / circuit.resistance
                     for j in range(4)))


def routed_hardware(network, circuit, routes, input_resistance):
    return [BridgeHardware.from_design(bridge, circuit, route["arm_resistance_ohm"],
                route["one_instrument_lead_ohm"], route["one_instrument_lead_ohm"],
                input_resistance) for bridge, route in zip(network, routes)]


def source_power(hardware, strain):
    return sum(h.operating_point(strain)["source_power_w"] for h in hardware)


def matched_power_network(network, units, circuit, routes, input_resistance, power_w):
    """Match nominal zero-strain source power, including every supply lead."""
    scaled = [replace(b, arm_units=float(m)) for b, m in zip(network, units)]
    hardware = routed_hardware(scaled, circuit, routes, input_resistance)
    zeros = np.zeros(1 + max(np.max(b.gauge_indices) for b in network))
    voltage_ratio = np.sqrt(power_w / source_power(hardware, zeros))
    supply = replace(circuit, voltage=circuit.voltage * voltage_ratio)
    return scaled, supply, [replace(h, circuit=supply) for h in hardware]


def acquisition_variances(hardware, config, acquisition):
    """White noise at the nominal balanced receiver, including leads and input."""
    zeros = np.zeros(1 + max(np.max(h.bridge.gauge_indices) for h in hardware))
    resistance = np.array([h.output_resistance(zeros) for h in hardware])
    return [4 * BOLTZMANN * acquisition["temperature_k"] * resistance / (2 * duration)
            + acquisition["independent_sample_sigma_v"]**2 * acquisition["sample_period_s"] / duration
            for duration in (config["average_time_s"], acquisition["reference_average_time_s"])]


def design_ratio(hardware, healthy, changed, config, acquisition):
    energies, y = network_energies(hardware, healthy, changed,
                                  acquisition["reference_calibration_index"])
    noise = sum(acquisition_variances(hardware, config, acquisition))
    log_ratio = regional_log_ratio(energies, y, noise)
    return float(log_ratio), energies, y


def optimize_arm_scales(network, circuit, routes, healthy, changed, config, acquisition):
    """Remove ballast within the original arm capacities using design data only."""
    receiver = config["differential_input_resistance_ohm"]
    original = np.array([b.arm_units for b in network])
    lower = np.array([minimum_arm_units(b, circuit, r) for b, r in zip(network, routes)])
    original_hardware = routed_hardware(network, circuit, routes, receiver)
    zeros = np.zeros(healthy.shape[-1])
    power = float(source_power(original_hardware, zeros))

    def candidate(fraction):
        return matched_power_network(network, original - fraction * (original - lower),
                                     circuit, routes, receiver, power)

    def objective(fraction):
        return -design_ratio(candidate(fraction)[2], healthy, changed, config, acquisition)[0]

    fit = minimize(objective, np.ones(len(network)), jac="3-point", method="L-BFGS-B",
                   bounds=[(0., 1.)] * len(network),
                   options={"ftol": 1e-9, "gtol": 1e-5, "finite_diff_rel_step": 1e-3})
    if not fit.success:
        raise RuntimeError(f"Arm-scale optimization failed: {fit.message}")
    candidates = {"integer": np.zeros(len(network)), "minimum": np.ones(len(network)),
                  "optimized": fit.x}
    objectives = {name: -objective(value) for name, value in candidates.items()}
    selected = max(candidates, key=objectives.get)
    scaled, supply, hardware = candidate(candidates[selected])
    return scaled, supply, hardware, {
        "minimum_arm_units": lower.tolist(), "original_arm_units": original.tolist(),
        "selected_arm_units": [b.arm_units for b in scaled], "selected_candidate": selected,
        "design_log_ratios": objectives, "iterations": int(fit.nit),
        "matched_source_power_w": power,
    }
