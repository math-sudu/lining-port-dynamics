"""Regional allocation with retained installations and loaded-reference readout."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .bridge_network import Circuit
from .passive_bridge import TrimmedBridge, distributed_target_weight
from .passive_hardware import BridgeHardware, route_bridge
from .passive_network import network_energies, optimize_exact_network, regional_projection
from .passive_readout import sample_fixed_reference
from .passive_readout_study import norm_statistics
from .passive_robustness_study import distances, source_data


def attenuation_cap(ratio, circuit, max_shunt):
    return float(np.min(np.minimum(1 / ratio, ratio * (
        max_shunt / (circuit.resistance * ratio + max_shunt))**2)))


def retained_network(saved, name, layout, channels, condition):
    key = f"{name}_l{layout}_c{channels}_{condition}"
    ratio = saved[f"{name}_actual_nominal_ratio"][layout]
    return [TrimmedBridge(group, arm, c, int(m), ratio[group])
            for group, arm, c, m in zip(
                np.array_split(saved[key + "_gauge_order"], channels),
                np.array_split(saved[key + "_arm"], channels),
                np.array_split(saved[key + "_attenuation"], channels),
                saved[key + "_arm_units"])]


def operational_variances(network, circuit, config, acquisition):
    return [np.array([bridge.mean_noise_variance(circuit, duration,
                    acquisition["independent_sample_sigma_v"], acquisition["sample_period_s"],
                    acquisition["temperature_k"]) for bridge in network])
            for duration in (config["average_time_s"], acquisition["reference_average_time_s"])]


def run():
    config = json.loads(Path("inputs/passive_robustness_task.json").read_text())
    acquisition = json.loads(Path("inputs/passive_readout_task.json").read_text())
    saved = np.load("results/passive-robustness/hardware.npz")
    circuit = Circuit(gauge_factor=50.)
    channels = 4
    reference_index = acquisition["reference_calibration_index"]
    arrays, records, netlists, selections = {}, [], [], []
    for data in source_data():
        name = data["name"]
        if name == "ring":
            cases = [r for r in json.loads(Path("results/fixed-installation/cases.json").read_text())
                     if r["split"] == "test"]
            positions = np.array([r["sector_center"] for r in cases])
            regions = np.floor((positions + np.pi / 12) % (2 * np.pi) / (np.pi / 6)).astype(int)
        else:
            patches = np.load("results/frame-bridges/synthesis.npz")["confirmation_patch"]
            positions = patches[:, 1:3].mean(axis=1)
            regions = patches[:, 0].astype(int) * 3 + np.argmin(abs(positions[:, None] - np.array([.2, .5, .8])), axis=1)
        arrays[name + "_case_region"] = regions
        arrays[name + "_case_position"] = positions
        for layout, seed in enumerate(data["seeds"]):
            key = f"{name}_l{layout}_c{channels}"
            response = saved[f"{key}_{acquisition['source_condition']}_response"]
            ratio = saved[name + "_actual_nominal_ratio"][layout]
            gain = saved[name + "_actual_gauge_factor_ratio"][layout]
            healthy = data["coefficients"] @ response.T
            increments = data["design"][:, :, layout] - data["calibration"][:, layout]
            changed = healthy + increments
            baseline = retained_network(saved, name, layout, channels, acquisition["source_condition"])
            caps = [attenuation_cap(b.resistance_ratio, circuit, config["max_shunt_ohm"]) for b in baseline]
            variances = sum(operational_variances(baseline, circuit, config, acquisition))
            fitted, fit = optimize_exact_network([BridgeHardware.from_design(b, circuit) for b in baseline],
                                                 response, healthy, changed,
                                                 variances, caps, reference_index)
            networks = {"bridgewise": baseline, "joint_fixed": fitted}
            fits = {"joint_fixed": fit}
            # Cyclic distribution in the existing spatial order gives each
            # bridge gauges spanning the array, with exclusive membership.
            order = np.concatenate([b.gauge_indices for b in baseline])
            initial = []
            for q in range(channels):
                group = order[q::channels]
                weight = distributed_target_weight(response[group],
                    increments[..., group].reshape(-1, len(group)).T, seed + q)
                bridge = TrimmedBridge.from_weight(group, weight)
                cap = attenuation_cap(ratio[group], circuit, config["max_shunt_ohm"])
                initial.append(replace(bridge, attenuation=cap * bridge.attenuation,
                                       nominal_ratio=ratio[group]))
            caps = [attenuation_cap(b.resistance_ratio, circuit, config["max_shunt_ohm"])
                    for b in initial]
            variances = sum(operational_variances(initial, circuit, config, acquisition))
            networks["joint_regrouped"], fits["joint_regrouped"] = optimize_exact_network(
                [BridgeHardware.from_design(b, circuit) for b in initial],
                response, healthy, changed, variances, caps, reference_index)
            for mode, network in networks.items():
                prefix = f"{name}_l{layout}_{mode}"
                current_variance, reference_variance = operational_variances(network, circuit, config, acquisition)
                energies, y = network_energies([BridgeHardware.from_design(b, circuit) for b in network],
                                               healthy, changed, reference_index)
                full, grouped = regional_projection(data["basis"][layout], increments,
                                                     [b.gauge_indices for b in network])
                for label, value in {"gauge_order": np.concatenate([b.gauge_indices for b in network]),
                    "arm": np.concatenate([b.arm for b in network]),
                    "attenuation": np.concatenate([b.attenuation for b in network]),
                    "arm_units": np.array([b.arm_units for b in network]),
                    "design_channel_energy_v2": energies,
                    "design_increment_v": np.stack([b.voltage(changed, circuit) - b.voltage(healthy, circuit)
                                                     for b in network], axis=-1),
                    "design_healthy_operational_v": y, "global_projected_energy": full,
                    "group_projected_energy": grouped,
                    "current_noise_variance_v2": current_variance,
                    "reference_noise_variance_v2": reference_variance}.items():
                    arrays[prefix + "_" + label] = value
                row = {"structure": name, "layout": layout, "layout_seed": seed, "mode": mode,
                       "design_region_rms_uv": (np.sqrt(energies.sum(axis=-1)) * 1e6).tolist(),
                       "group_projection_fraction": (grouped.sum(axis=-1) / full).tolist(),
                       "design_ratio": float(np.exp(np.log(energies.sum(axis=-1)).mean()) /
                           (np.mean(np.sum(y*y, axis=-1)) + current_variance.sum() + reference_variance.sum())),
                       "operational_noise_rms_uv": float(np.sqrt(current_variance.sum() + reference_variance.sum()) * 1e6),
                       "nominal_power_mw": float(sum(4 * circuit.voltage**2 /
                           (b.arm_units * circuit.resistance) for b in network) * 1e3)}
                if mode in fits:
                    row["fit"] = fits[mode]
                cell, instrument, center = distances(data, layout)
                raw_reference, raw_current, routes, components = [], [], [], []
                reference_strain = data["calibration"][reference_index, layout] * gain
                for q, bridge in enumerate(network):
                    group = bridge.gauge_indices
                    route = route_bridge(bridge, cell, center, instrument, data["depth"][layout],
                                         config["wire_resistance_ohm_per_m"])
                    hardware = BridgeHardware.from_design(bridge, circuit, route["arm_resistance_ohm"],
                        route["one_instrument_lead_ohm"], route["one_instrument_lead_ohm"],
                        config["differential_input_resistance_ohm"])
                    errors = {"shunt_error": acquisition["component_uniform_tolerance"] *
                              saved[key + "_shunt_error_draw"][:, None, group],
                              "ballast_error": acquisition["component_uniform_tolerance"] *
                              saved[key + "_ballast_error_draw"][:, None, q]}
                    raw_reference.append(hardware.voltage(reference_strain[None, None], **errors)[:, 0])
                    raw_current.append(np.stack([hardware.voltage(data[state][:, layout][None] * gain, **errors)
                                                 for state in ("healthy", "changed")]))
                    routes.append(route)
                    components.append(hardware.netlist(np.zeros(len(ratio)),
                        errors["shunt_error"][0, 0], errors["ballast_error"][0, 0]))
                raw_reference, raw_current = np.stack(raw_reference, -1), np.stack(raw_current, -1)
                rng = np.random.default_rng(acquisition["random_seed"] + 10 * seed + channels)
                measured, recorded, sampled = sample_fixed_reference(
                    raw_current, raw_reference, current_variance, reference_variance, rng)
                operational = raw_current - raw_reference[:, None]
                delta = raw_current[1] - raw_current[0]
                values = {"raw_reference_v": raw_reference, "recorded_reference_v": recorded,
                          "paired_increment_v": delta}
                for i, state in enumerate(("healthy", "changed")):
                    for label, value in (("raw", raw_current), ("sampled_raw", measured),
                                         ("operational", operational), ("sampled_operational", sampled)):
                        values[state + "_" + label + "_v"] = value[i]
                    row[state + "_sampled_norm_uv"] = norm_statistics(sampled[i])
                for label, value in values.items():
                    arrays[prefix + "_" + label] = value
                row["paired_increment_norm_uv"] = norm_statistics(delta)
                row["conductor_length_m"] = float(sum(r["total_conductor_m"] for r in routes))
                row["raw_receiver_peak_uv"] = float(max(np.max(abs(raw_reference)), np.max(abs(raw_current))) * 1e6)
                row["regions"] = [{"region": int(region), "case_count": int(np.count_nonzero(regions == region)),
                    "healthy_sampled_norm_uv": norm_statistics(sampled[0][:, regions == region]),
                    "changed_sampled_norm_uv": norm_statistics(sampled[1][:, regions == region]),
                    "paired_increment_norm_uv": norm_statistics(delta[:, regions == region])}
                    for region in np.unique(regions)]
                netlists.append({"structure": name, "layout_seed": seed, "mode": mode,
                    "rails_v": {"H": circuit.voltage, "L": -circuit.voltage}, "hardware_draw": 0,
                    "bridges": [{"route": r, "zero_strain_netlist": c} for r, c in zip(routes, components)]})
                records.append(row)
            local = [r for r in records if r["structure"] == name and r["layout_seed"] == seed]
            chosen = max((r for r in local if r["mode"] != "bridgewise"), key=lambda r: r["design_ratio"])
            selections.append({"structure": name, "layout": layout, "layout_seed": seed,
                               "mode": chosen["mode"]})
            print(name, seed, {r["mode"]: round(r["design_ratio"], 2) for r in local}, flush=True)
    output = Path("results/passive-network")
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "allocation.npz", **arrays)
    summaries = []
    for name in ("ring", "frame"):
        choices = [r for r in selections if r["structure"] == name]
        for mode in ("bridgewise", "joint_fixed", "selected"):
            chosen = [next(r for r in records if r["structure"] == name and r["layout"] == s["layout"]
                          and r["mode"] == (s["mode"] if mode == "selected" else mode)) for s in choices]
            summary = {"structure": name, "mode": mode,
                       "regrouped_installations": sum(s["mode"] == "joint_regrouped" for s in choices)
                                                  if mode == "selected" else 0}
            for key in ("nominal_power_mw", "operational_noise_rms_uv"):
                summary[key] = float(np.median([r[key] for r in chosen]))
            for key in ("healthy_sampled_norm_uv", "changed_sampled_norm_uv", "paired_increment_norm_uv"):
                summary[key] = {stat: float(np.median([r[key][stat] for r in chosen]))
                                for stat in ("median", "p10", "p95")}
            summary["conductor_length_m"] = float(np.median([r["conductor_length_m"] for r in chosen]))
            summary["raw_receiver_peak_uv"] = max(r["raw_receiver_peak_uv"] for r in chosen)
            values = np.concatenate([np.linalg.norm(arrays[f"{name}_l{r['layout']}_{r['mode']}_paired_increment_v"],
                                                   axis=-1).ravel() for r in chosen])
            summary["pooled_increment_p10_uv"] = float(np.quantile(values, .1) * 1e6)
            summaries.append(summary)
    metadata = {"channels": channels, "supply_voltage_v": 2 * circuit.voltage,
                "source_condition": acquisition["source_condition"], "acquisition": acquisition,
                "records": records, "selected_modes": selections,
                "summaries": summaries,
                "projection_basis": "Nominal mechanical healthy basis and nominal design increments; voltage design uses the commissioned effective basis.",
                "selection": "Larger design geometric regional energy divided by loaded-reference healthy energy "
                    "and both averaging variances, among fixed and cyclically interleaved memberships. "
                    "All signs and coefficients use design/calibration data only. Four channels and 2-V supply "
                    "are common; arm scales and actual conductor lengths are reported for each circuit.",
                "evaluation": {"ring": "Retained 32 independent test load-sector pairs per installation",
                    "frame": "Retained 40 independent confirmation load-patch pairs per installation; "
                        "not used in the current coefficient, grouping or seed selection"},
                "sampling": "Same gauge properties, calibration errors and 16 component draws as the hardware study. "
                    "One loaded 4-s reference and 4-s service averages. Identical standard-normal noise "
                    "draws across designs, scaled by each circuit's variance; one reference shared by all cases.",
                "array_axes": {"current": ["hardware_draw", "case", "channel"],
                    "design_increment": ["region", "design_load", "channel"],
                    "design_channel_energy": ["region", "channel"]}}
    (output / "allocation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "netlists.json").write_text(json.dumps(netlists, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
