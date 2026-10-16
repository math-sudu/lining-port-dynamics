"""Scanning and physical bridge acquisition on the fixed 64-gauge installation."""
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


def bridge_netlists():
    """Zero-based gauge IDs, ordered by bridge, arm R1--R4 and series path."""
    local = np.array([[np.arange(i, i + 8, 2), np.arange(i + 1, i + 8, 2),
                       np.arange(i + 9, i + 16, 2), np.arange(i + 8, i + 16, 2)]
                      for i in range(0, 64, 16)])
    four = np.array([[np.arange(i, i + 8, 2), np.arange(i + 1, i + 8, 2),
                      np.arange(i + 32, i + 40, 2), np.arange(i + 33, i + 40, 2)]
                     for i in range(0, 32, 8)])
    two = np.array([[np.r_[a[0], a[3]], np.r_[a[1], a[2]],
                     np.r_[b[1], b[2]], np.r_[b[0], b[3]]]
                    for a, b in zip(four[::2], four[1::2])])
    return {"mechanical_bridge": local, "composite4": four, "composite2": two}


def bridge_voltage(strain, nodes, circuit):
    """Exact differential midpoint voltages for series-connected active arms.

    Leading strain dimensions are retained; the last dimension holds 64 gauges.
    Each netlist supplies its physical channels, including the two-bridge circuit.
    """
    r = circuit.resistance * (1 + circuit.gauge_factor * np.asarray(strain))
    if np.any(r <= 0):
        raise ValueError("Gauge resistance must be positive")
    arms = r[..., nodes].sum(axis=-1)
    return 2 * circuit.voltage * (
        arms[..., 1] / (arms[..., 0] + arms[..., 1])
        - arms[..., 3] / (arms[..., 2] + arms[..., 3]))


def mechanical_bridge_voltage(strain, circuit):
    """The original four local bridges, one per module."""
    return bridge_voltage(strain, bridge_netlists()["mechanical_bridge"], circuit)


def bridge_operator(cycle=4., dt=.005):
    """A continuously excited bridge at the receiver's unit DC gain."""
    return np.tile(np.eye(4), (round(cycle / dt), 1))


def resources(config, geom, circuit):
    """Physical receivers, routed conductors and nominal resistive excitation."""
    from .acquisition_routing import SCOPE, routed_layouts, route_totals

    result = {
        "scope": SCOPE,
        "rc": {"adc_channels": 4, "record_samples": 1600, "cycle_s": 4., "reset_s": 2.,
               "active_gauges": 64, "reference_resistors": 64, "capacitors": 64, "leakage_resistors": 64,
               "link_resistors": 88, "distributed_bus_conductors": 3,
               "nominal_excitation_energy_j": 64 * 2 * circuit.voltage**2 / circuit.resistance * 2},
        "scan": {"adc_channels": 4, "record_samples": 2944, "cycle_s": 4., "slots": 16,
                 "slot_s": .25, "switch_s": .02, "retained_samples_per_sensor": 46,
                 "active_gauges": 64, "completion_resistors": 4, "signal_mux_banks_16_to_1": 4, "blanking_ground_switches": 4,
                 "individual_returns": 64, "distributed_bus_conductors": 1,
                 "controller": "Clock, reset and powered relay drivers at four accessible module heads",
                 "nominal_excitation_energy_j": 4 * 2 * circuit.voltage**2 / circuit.resistance * (4 - 16 * .02)},
    }
    for name, nodes in bridge_netlists().items():
        channels, _, per_arm = nodes.shape
        result[name] = {"adc_channels": channels, "differential_inputs": channels,
                        "midpoint_leads": 2 * channels, "record_samples": channels * 800,
                        "cycle_s": 4., "active_gauges": 64, "bridges": channels,
                        "gauges_per_arm": per_arm,
                        "nominal_excitation_energy_j": channels * (2 * circuit.voltage)**2 / (per_arm * circuit.resistance) * 4}
    for name, layout in routed_layouts(geom).items():
        result[name].update(route_totals(layout))
    return result


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
