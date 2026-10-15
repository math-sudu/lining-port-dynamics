"""Four-ADC scanning and four mechanical bridges for the fixed installation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from .bridge_network import Circuit
from .installation import installed_network
from .installation_study import detector, residual, threshold_and_rates
from .mechanism_study import dump
from .network import receiver_matrices


def scan_operator(cycle=4., dt=.005, switching=.02):
    """Four simultaneous 16:1 banks, one bank per module, with filter memory.

    During switching the filter input is grounded and the ADC samples are
    excluded. All post-switch samples, including settling, remain available.
    Columns map 64 static divider voltages to retained time-major ADC samples.
    """
    a, b, c = receiver_matrices()
    transition = expm(a * dt)
    equilibrium = np.linalg.solve(a, -b)
    forcing = equilibrium - transition @ equilibrium
    slot_steps = round(cycle / (16 * dt))
    blank_steps = round(switching / dt)
    if slot_steps * 16 * dt != cycle or not 0 <= blank_steps < slot_steps:
        raise ValueError("The scan cycle must contain 16 slots with time for acquisition")
    def cycle_record(initial):
        state = initial.copy()
        records, indices = [], []
        for slot in range(16):
            selected = np.arange(4) * 16 + slot
            for tick in range(slot_steps):
                if tick >= blank_steps:
                    records.append(np.einsum("s,psn->pn", c, state))
                    indices.append(slot * slot_steps + tick)
                state = np.einsum("ij,pjn->pin", transition, state)
                if tick >= blank_steps:
                    state[np.arange(4), :, selected] += forcing
        return state, np.asarray(records).reshape(-1, 64), np.asarray(indices) * dt

    # Carry the receiver state across scan cycles, including the first switch.
    end, _, _ = cycle_record(np.zeros((4, 4, 64)))
    initial = np.linalg.solve(np.eye(4) - expm(a * cycle),
                              end.transpose(1, 0, 2).reshape(4, -1))
    initial = initial.reshape(4, 4, 64).transpose(1, 0, 2)
    _, records, times = cycle_record(initial)
    return records, times


def divider_voltage(strain, circuit):
    r = circuit.resistance * (1 + circuit.gauge_factor * np.asarray(strain))
    if np.any(r <= 0):
        raise ValueError("Gauge resistance must be positive")
    return circuit.voltage * (circuit.resistance - r) / (circuit.resistance + r)


def mechanical_bridge_voltage(strain, circuit):
    """Four active arms per module; four series gauges per arm.

    Opposite arms collect inner-layer gauges; the other two collect outer-layer
    gauges. Each gauge is used once and the output is the differential midpoint.
    """
    r = circuit.resistance * (1 + circuit.gauge_factor * np.asarray(strain).reshape(4, 8, 2))
    if np.any(r <= 0):
        raise ValueError("Gauge resistance must be positive")
    r1 = r[:, :4, 0].sum(axis=1)
    r2 = r[:, :4, 1].sum(axis=1)
    r3 = r[:, 4:, 1].sum(axis=1)
    r4 = r[:, 4:, 0].sum(axis=1)
    return 2 * circuit.voltage * (r2 / (r1 + r2) - r4 / (r3 + r4))


def bridge_operator(cycle=4., dt=.005):
    """A continuously excited bridge at the receiver's unit DC gain."""
    return np.tile(np.eye(4), (round(cycle / dt), 1))


def resources(config, geom, circuit):
    """Routed conductor lengths inside the four modules, excluding common trunks."""
    radius = config["nominal_mechanics"]["radius"]
    thickness = config["nominal_mechanics"]["thickness"]
    depth = config["depth_m"]
    # Each distributed bus follows the module mid-depth arc and radial stubs.
    bus = 4 * radius * 7 * np.pi / 16 + 64 * depth
    angle = (geom["theta"] - geom["module"] * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
    scan_returns = np.sum((radius + geom["z"]) * np.abs(angle) + thickness / 2 + geom["z"])
    links = geom["route_length"].sum() / 2
    leads = geom["port_lead_length"].sum()
    return {
        "scope": "Nominal resistive excitation energy and unshielded conductor length inside four modules; common external trunks and receiver/controller power excluded",
        "rc": {"adc_channels": 4, "record_samples": 1600, "cycle_s": 4., "reset_s": 2.,
               "active_gauges": 64, "reference_resistors": 64, "capacitors": 64, "leakage_resistors": 64,
               "link_resistors": 88, "signal_route_m": float(links + leads),
               "distributed_bus_conductors": 3, "local_conductor_m": float(3 * bus + links + leads),
               "nominal_excitation_energy_j": 64 * 2 * circuit.voltage**2 / circuit.resistance * 2},
        "scan": {"adc_channels": 4, "record_samples": 2944, "cycle_s": 4., "slots": 16,
                 "slot_s": .25, "switch_s": .02, "retained_samples_per_sensor": 46,
                 "active_gauges": 64, "completion_resistors": 4, "signal_mux_banks_16_to_1": 4, "blanking_ground_switches": 4,
                 "individual_returns": 64, "distributed_bus_conductors": 1,
                 "signal_route_m": float(scan_returns), "local_conductor_m": float(bus + scan_returns),
                 "controller": "Clock, reset and powered relay drivers at four accessible module heads",
                 "nominal_excitation_energy_j": 4 * 2 * circuit.voltage**2 / circuit.resistance * (4 - 16 * .02)},
        "mechanical_bridge": {"adc_channels": 4, "record_samples": 3200, "cycle_s": 4.,
                              "active_gauges": 64, "bridges": 4, "gauges_per_arm": 4,
                              "nominal_excitation_energy_j": 4 * circuit.voltage**2 / circuit.resistance * 4},
    }


def main():
    root = Path("results/fixed-installation")
    config = json.loads(Path("inputs/installation_task.json").read_text())
    device = json.loads(Path("inputs/device_node.json").read_text())
    circuit = Circuit(**device["circuit"])
    strains = np.load(root / "strains.npz")
    cases = json.loads((root / "cases.json").read_text())
    scan, times = scan_operator(switching=config["scan_switch_s"])
    bridge = bridge_operator()
    systems = {"scan": lambda e: scan @ divider_voltage(e, circuit),
               "mechanical_bridge": lambda e: bridge @ mechanical_bridge_voltage(e, circuit)}
    rc_rates = json.loads((root / "detection.json").read_text())
    rc_false_alarm = float(np.mean([r["false_positive_rate"] for r in rc_rates
                                    if r["mode"] == "residual_designed" and r["reduction"] == .1]))
    matched_rates = []
    records, scores, rates, examples = [], [], [], {"scan_time_s": times, "scan_operator": scan}
    for mode, response in systems.items():
        reference = response(strains["nominal"])
        nuisance = np.column_stack([(response(strains[f"{key}_1"]) - response(strains[f"{key}_-1"])) / (2 * step)
                                   for key, step in config["finite_difference_steps"].items()])
        targets = np.column_stack([response(e) - reference for e in strains["design_soft"]])
        _, background, signal = detector(reference, nuisance, targets)
        examples[mode + "_design_targets"] = targets
        examples[mode + "_design_nuisance"] = nuisance
        for split in ("calibration", "test"):
            selected = [r for r in cases if r["split"] == split]
            for ci, (case, eps) in enumerate(zip(selected, strains[f"{split}_healthy"])):
                healthy = response(eps)
                scores.append({"mode": mode, "family": case["scenario_family_id"], "split": split,
                               "reduction": 0., "score_v": float(np.linalg.norm(signal.T @ (healthy - reference)))})
                if split == "calibration":
                    continue
                for di, delta in enumerate(config["test_reductions"]):
                    soft = response(strains["test_soft"][ci, di])
                    scores.append({"mode": mode, "family": case["scenario_family_id"], "split": split,
                                   "reduction": delta, "score_v": float(np.linalg.norm(signal.T @ (soft - reference)))})
                    if delta == config["design_reduction"]:
                        records.append({"mode": mode, "family": case["scenario_family_id"],
                                        "contrast_l2_v": float(np.linalg.norm(soft - healthy)),
                                        "residual_l2_v": float(np.linalg.norm(residual(soft - healthy, background)))})
                        if ci == 0:
                            examples[mode + "_healthy"] = healthy
                            examples[mode + "_soft"] = soft
        selected = [r for r in scores if r["mode"] == mode]
        calibration = [r["score_v"] for r in selected if r["split"] == "calibration"]
        healthy = [r["score_v"] for r in selected if r["split"] == "test" and r["reduction"] == 0]
        for delta in config["test_reductions"]:
            soft = [r["score_v"] for r in selected if r["reduction"] == delta]
            rates.append({"mode": mode, "reduction": delta, **threshold_and_rates(
                calibration, healthy, soft, signal.shape[1], config["sample_noise_sigma_v"], config["healthy_quantile"])})
            matched_rates.append({"mode": mode, "reduction": delta, **threshold_and_rates(
                healthy, healthy, soft, signal.shape[1], config["sample_noise_sigma_v"], 1 - rc_false_alarm)})
    _, geom = installed_network(config["layout_seeds"][8])
    dump(root / "acquisition.json", {"resources": resources(config, geom, circuit), "detection": rates,
                                      "evaluation": records, "scores": scores,
                                      "roc_at_rc_false_alarm": matched_rates,
                                      "roc_definition": "Descriptive test-distribution ROC at the RC design's mean test false-positive rate; calibrated operating rates remain in detection"})
    np.savez_compressed(root / "acquisition_responses.npz", **examples)
    print(json.dumps(rates, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
