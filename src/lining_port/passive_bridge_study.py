"""Evaluate realizable passive null bridges on the retained random installations."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .bridge_network import Circuit
from .fe import RingParameters
from .installation_study import load_coefficients, strain_load_basis
from .passive_bridge import TrimmedBridge, optimize_exact_bridge, target_null_weight


def summarize_voltage(healthy, changed, noise_variance):
    healthy_norm = np.linalg.norm(healthy, axis=-1)
    increment_norm = np.linalg.norm(changed - healthy, axis=-1)
    return {
        "healthy_norm_uv_median": float(np.median(healthy_norm) * 1e6),
        "increment_norm_uv_median": float(np.median(increment_norm) * 1e6),
        "increment_norm_uv_min": float(increment_norm.min() * 1e6),
        "noise_norm_rms_uv": float(np.sqrt(np.sum(noise_variance)) * 1e6),
        "increment_to_residual_noise_rms": float(np.sqrt(np.mean(increment_norm**2)
            / (np.mean(healthy_norm**2) + np.sum(noise_variance)))),
    }


def run():
    config = json.loads(Path("inputs/passive_bridge_task.json").read_text())
    installation = json.loads(Path(config["source_installation"]).read_text())
    device = json.loads(Path(config["device_source"]).read_text())
    cases = json.loads(Path(config["source_cases"]).read_text())
    calibration_cases = [row for row in cases if row["split"] == "calibration"]
    source = np.load(config["source_strains"])
    circuit = Circuit(**device["circuit"])
    nominal = RingParameters(**installation["nominal_mechanics"])
    geometry = {"theta": source["theta_rad"].ravel(), "z": source["depth_m"].ravel()}
    layouts, gauges = source["theta_rad"].shape
    calibration = source["calibration_strain"]
    test_healthy, test_changed = source["test_healthy_strain"], source["test_changed_strain"]
    coefficients = np.array([load_coefficients(replace(nominal, **row["params"]))
                             for row in calibration_cases])
    sectors = np.arange(config["design_sector_count"]) * 2 * np.pi / config["design_sector_count"]
    target_bases = np.stack([strain_load_basis(
        replace(nominal, sector_center=center, reduction=config["design_support_reduction"]),
        geometry, installation["beam_elements"], device["gauge_length_m"])
        for center in sectors]).reshape(len(sectors), layouts, gauges, -1)
    design_changed = np.einsum("sp,dlgp->dslg", coefficients, target_bases)
    output = Path("results/passive-bridges")
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"design_changed_basis_per_pa": target_bases, "design_sector_center_rad": sectors}
    summaries = []
    circuit_records = []
    for channels in config["channels"]:
        modes = {mode: {"healthy": [], "changed": [], "attenuation": [], "arm": [], "units": [],
                        "bound": [], "linear_healthy": [], "noise_variance": []}
                 for mode in ("direct", "optimized")}
        rows, groups_by_layout = [], []
        for layout in range(layouts):
            groups = np.array_split(np.argsort(source["theta_rad"][layout]), channels)
            groups_by_layout.append(np.concatenate([np.full(len(group), j) for j, group in enumerate(groups)]))
            healthy_basis = source["healthy_basis_per_pa"][layout]
            targets = design_changed[:, :, layout] - calibration[:, layout]
            bridges = {"direct": [], "optimized": []}
            details = []
            for channel, group in enumerate(groups):
                weight, info = target_null_weight(healthy_basis[group], targets[..., group].reshape(-1, len(group)).T)
                direct = TrimmedBridge.from_weight(group, weight)
                variance = direct.mean_noise_variance(
                    circuit, config["average_time_s"], installation["sample_noise_sigma_v"],
                    installation["sample_period_s"], config["temperature_k"])
                optimized, fit = optimize_exact_bridge(direct, healthy_basis, calibration[:, layout],
                    design_changed[:, :, layout], circuit, variance)
                bridges["direct"].append(direct)
                bridges["optimized"].append(optimized)
                details.append({"channel": channel, "gauge_count": len(group), **info, **fit})
            row = {"layout_seed": installation["layout_seeds"][layout], "design": details}
            for mode, network in bridges.items():
                healthy = np.stack([bridge.voltage(test_healthy[:, layout], circuit) for bridge in network], axis=-1)
                changed = np.stack([bridge.voltage(test_changed[:, layout], circuit) for bridge in network], axis=-1)
                bound = np.stack([bridge.nonlinear_bound(test_healthy[:, layout], circuit) for bridge in network], axis=-1)
                linear = np.stack([bridge.linear_voltage(test_healthy[:, layout], circuit) for bridge in network], axis=-1)
                variance = np.array([bridge.mean_noise_variance(
                    circuit, config["average_time_s"], installation["sample_noise_sigma_v"],
                    installation["sample_period_s"], config["temperature_k"]) for bridge in network])
                attenuation, arm = np.empty(gauges), np.empty(gauges, dtype=int)
                for bridge in network:
                    attenuation[bridge.gauge_indices] = bridge.attenuation
                    arm[bridge.gauge_indices] = bridge.arm
                values = {"healthy": healthy, "changed": changed, "bound": bound,
                          "linear_healthy": linear, "attenuation": attenuation, "arm": arm,
                          "units": np.array([bridge.arm_units for bridge in network]), "noise_variance": variance}
                for key, value in values.items():
                    modes[mode][key].append(value)
                row[mode] = summarize_voltage(healthy, changed, variance)
                row[mode].update(
                    first_order_healthy_max_abs_v=float(np.max(np.abs(linear))),
                    healthy_nonlinear_bound_norm_uv_median=float(np.median(np.linalg.norm(bound, axis=-1)) * 1e6),
                    nominal_power_mw=float(sum(4 * circuit.voltage**2 / (bridge.arm_units * circuit.resistance)
                                               for bridge in network) * 1e3))
                netlists = [bridge.resistor_netlist(np.zeros(gauges), circuit) for bridge in network]
                components = [component for netlist in netlists for component in netlist]
                fixed_resistors = [component["resistance_ohm"] for component in components
                                   if component["kind"] != "gauge"]
                row[mode].update(
                    active_gauge_count=sum(component["kind"] == "gauge" for component in components),
                    added_resistor_count=len(fixed_resistors),
                    smallest_added_resistor_ohm=float(min(fixed_resistors)),
                    largest_added_resistor_ohm=float(max(fixed_resistors)))
                if mode == "optimized":
                    circuit_records.append({"channels": channels, "layout_seed": row["layout_seed"],
                        "rails_v": {"H": circuit.voltage, "L": -circuit.voltage},
                        "bridges": [{"gauge_indices_zero_based": bridge.gauge_indices.tolist(),
                            "arm_zero_based": bridge.arm.tolist(), "attenuation": bridge.attenuation.tolist(),
                            "arm_units": bridge.arm_units,
                            "zero_strain_netlist": netlist} for bridge, netlist in zip(network, netlists)]})
            rows.append(row)
            print(f"{channels} channels, seed {row['layout_seed']}: healthy "
                  f"{row['direct']['healthy_norm_uv_median']:.5f} -> "
                  f"{row['optimized']['healthy_norm_uv_median']:.5f} uV; target "
                  f"{row['optimized']['increment_norm_uv_median']:.3f} uV", flush=True)
        for mode, values in modes.items():
            for key, value in values.items():
                axis = 1 if key in ("healthy", "changed", "bound", "linear_healthy") else 0
                arrays[f"c{channels}_{mode}_{key}"] = np.stack(value, axis=axis)
        arrays[f"c{channels}_sorted_gauge_indices"] = np.argsort(source["theta_rad"], axis=-1)
        arrays[f"c{channels}_group_in_sorted_order"] = np.array(groups_by_layout)
        summaries.append({"channels": channels, "layouts": rows, "medians_of_layout_statistics": {
            mode: {key: float(np.median([row[mode][key] for row in rows])) for key in rows[0][mode]}
            for mode in modes}})
    summary = {"configuration": config, "assumptions": "Linear elastic mechanics; linear gauge law; matched "
        "nominal resistances; exact passive DC voltage; continuous ideal trim components and ideal wires. "
        "No post-digitization healthy-load projection or per-gauge observation in service.",
        "voltage_array_axes": ["test_case", "layout", "channel"],
        "design_basis_axes": ["sector", "layout", "gauge", "load_coefficient"],
        "evaluation": "Original independent test loads and sector centers; support reduction 0.1. "
        "Design uses 12 uniform sectors at reduction 0.45 and calibration loads only.",
        "channel_studies": summaries}
    np.savez_compressed(output / "synthesis.npz", **arrays)
    (output / "synthesis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "netlists.json").write_text(json.dumps(circuit_records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{ "channels": entry["channels"], **entry["medians_of_layout_statistics"]}
                      for entry in summaries], indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
