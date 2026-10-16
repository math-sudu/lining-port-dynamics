"""Fixed random installations: a healthy-load-only bridge grouping baseline."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .acquisition_comparison import bridge_voltage
from .bridge_network import Circuit
from .fe import RingParameters, recover_gauge_strain, solve_ring
from .installation_study import load_coefficients, strain_load_basis
from .network import make_layout


def random_gauge_geometry(seeds, radius, thickness):
    """Restore the original angular and annular-area draws, without moving gauges."""
    layouts = [make_layout(seed) for seed in seeds]
    theta = np.array([layout["theta"] for layout in layouts])
    quantile = np.array([layout["area_quantile"] for layout in layouts])
    radial = np.sqrt((radius - thickness / 2)**2 + 2 * radius * thickness * quantile)
    return {"theta": theta, "z": radial - radius}


def exchange_arm_membership(healthy_strain, nodes, circuit):
    """Minimize exact healthy voltage energy by best improving gauge exchanges.

    Positions and arm cardinalities stay fixed. Every gauge occurs once. The
    returned partition is a single-exchange local minimum; target strains are
    deliberately absent from this baseline objective.
    """
    nodes = np.asarray(nodes)
    channels, _, per_arm = nodes.shape
    permutation = nodes.ravel().copy()
    resistance = 1 + circuit.gauge_factor * np.asarray(healthy_strain)
    if np.any(resistance <= 0):
        raise ValueError("Gauge resistance must be positive")
    samples = len(resistance)
    first, second = np.triu_indices(len(permutation), 1)
    distinct = first // per_arm != second // per_arm
    first, second = first[distinct], second[distinct]
    arm_first, arm_second = first // per_arm, second // per_arm
    candidate_id = np.arange(len(first))

    def cost(arms):
        arms = arms.reshape(-1, samples, channels, 4)
        voltage = 2 * circuit.voltage * (
            arms[..., 1] / (arms[..., 0] + arms[..., 1])
            - arms[..., 3] / (arms[..., 2] + arms[..., 3]))
        return np.sum(voltage**2, axis=(1, 2))

    accepted = 0
    while True:
        arms = resistance[:, permutation].reshape(samples, channels * 4, per_arm).sum(axis=-1)
        current = cost(arms)[0]
        candidates = np.broadcast_to(arms, (len(first),) + arms.shape).copy()
        difference = (resistance[:, permutation[second]] - resistance[:, permutation[first]]).T
        candidates[candidate_id, :, arm_first] += difference
        candidates[candidate_id, :, arm_second] -= difference
        scores = cost(candidates)
        best = int(np.argmin(scores))
        if scores[best] >= current:
            return permutation.reshape(nodes.shape), accepted
        i, j = first[best], second[best]
        permutation[i], permutation[j] = permutation[j], permutation[i]
        accepted += 1


def main():
    config = json.loads(Path("inputs/installation_task.json").read_text())
    device = json.loads(Path("inputs/device_node.json").read_text())
    source_cases = Path("results/fixed-installation/cases.json")
    cases = json.loads(source_cases.read_text())
    calibration = [row for row in cases if row["split"] == "calibration"]
    test = [row for row in cases if row["split"] == "test"]
    nominal = RingParameters(**config["nominal_mechanics"])
    circuit = Circuit(**device["circuit"])
    seeds = config["layout_seeds"]
    geometry = random_gauge_geometry(seeds, nominal.radius, nominal.thickness)
    flat_geometry = {key: value.ravel() for key, value in geometry.items()}
    basis = strain_load_basis(nominal, flat_geometry, config["beam_elements"], device["gauge_length_m"])

    def healthy(rows):
        coefficients = np.array([load_coefficients(replace(nominal, **row["params"])) for row in rows])
        return (coefficients @ basis.T).reshape(len(rows), len(seeds), 64)

    calibration_strain, test_healthy = healthy(calibration), healthy(test)
    test_changed = []
    for row in test:
        field = solve_ring(replace(nominal, **row["params"], sector_center=row["sector_center"],
                                  reduction=.1), config["beam_elements"])
        test_changed.append(recover_gauge_strain(
            field, flat_geometry["theta"], flat_geometry["z"], device["gauge_length_m"]))
    test_changed = np.array(test_changed).reshape(len(test), len(seeds), 64)
    initial_nodes = np.arange(64).reshape(4, 4, 4)
    arrays = {
        "theta_rad": geometry["theta"], "depth_m": geometry["z"],
        "healthy_basis_per_pa": basis.reshape(len(seeds), 64, 3),
        "calibration_strain": calibration_strain, "test_healthy_strain": test_healthy,
        "test_changed_strain": test_changed, "initial_nodes_zero_based": initial_nodes,
    }
    netlists, initial_voltage, healthy_voltage, changed_voltage, rows = [], [], [], [], []
    for index, seed in enumerate(seeds):
        nodes, accepted = exchange_arm_membership(calibration_strain[:, index], initial_nodes, circuit)
        before = bridge_voltage(test_healthy[:, index], initial_nodes, circuit)
        after = bridge_voltage(test_healthy[:, index], nodes, circuit)
        changed = bridge_voltage(test_changed[:, index], nodes, circuit)
        initial_norm, healthy_norm = np.linalg.norm(before, axis=-1), np.linalg.norm(after, axis=-1)
        change_norm = np.linalg.norm(changed - after, axis=-1)
        rows.append({
            "layout_seed": seed, "accepted_swaps": accepted,
            "initial_healthy_norm_uv_median": float(np.median(initial_norm) * 1e6),
            "designed_healthy_norm_uv_median": float(np.median(healthy_norm) * 1e6),
            "support_change_norm_uv_median": float(np.median(change_norm) * 1e6),
        })
        netlists.append(nodes)
        initial_voltage.append(before)
        healthy_voltage.append(after)
        changed_voltage.append(changed)
    arrays.update(
        designed_nodes_zero_based=np.array(netlists),
        initial_test_healthy_voltage_v=np.stack(initial_voltage, axis=1),
        designed_test_healthy_voltage_v=np.stack(healthy_voltage, axis=1),
        designed_test_changed_voltage_v=np.stack(changed_voltage, axis=1),
    )
    summary = {
        "purpose": "Feasibility baseline for fixed random installations; healthy-load-only arm exchanges",
        "configuration_source": "inputs/installation_task.json",
        "device_source": "inputs/device_node.json", "source_cases": str(source_cases.as_posix()),
        "layout_role": "Original make_layout angular and area-quantile draws; each layout is fixed before wiring design",
        "circuit": "Four bridges, four series gauges per arm, every gauge used once; ideal wires",
        "design": "Best improving pairwise gauge exchange; exact bridge fractions on calibration healthy loads only",
        "evaluation": "Independent test loads and support-change sectors; no test strains enter design",
        "strain_array_axes": ["case", "layout", "gauge"],
        "voltage_array_axes": ["case", "layout", "channel"],
        "support_reduction": .1, "layout_count": len(seeds),
        "calibration_cases": len(calibration), "test_cases": len(test),
        "layouts": rows,
        "medians_of_layout_medians": {
            key: float(np.median([row[key] for row in rows]))
            for key in rows[0] if key.endswith("_median")
        },
    }
    output = Path("results/random-bridges")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "healthy_exchange.npz", **arrays)
    (output / "healthy_exchange.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["medians_of_layout_medians"], indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
