"""Controlled bridge designs under one routed, recorded-reference acquisition."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .bridge_network import Circuit
from .passive_bridge import TrimmedBridge, distributed_target_weight, response_spaces
from .passive_export import network_export
from .passive_hardware import route_bridge
from .passive_network import optimize_exact_network, regional_projection
from .passive_network_study import attenuation_cap, retained_network
from .passive_readout import sample_fixed_reference
from .passive_readout_study import norm_statistics
from .passive_resources import (acquisition_variances, design_ratio, matched_power_network,
                                optimize_arm_scales, routed_hardware, source_power)
from .passive_robustness_study import distances, source_data


def routed_candidate(network, circuit, geometry, depth, config, power_w=None):
    cell, instrument, center = geometry
    routes = [route_bridge(b, cell, center, instrument, depth,
                           config["wire_resistance_ohm_per_m"]) for b in network]
    receiver = config["differential_input_resistance_ohm"]
    if power_w is None:
        hardware = routed_hardware(network, circuit, routes, receiver)
    else:
        network, circuit, hardware = matched_power_network(
            network, [b.arm_units for b in network], circuit, routes, receiver, power_w)
    return {"network": network, "circuit": circuit, "routes": routes, "hardware": hardware}


def refine_candidate(candidate, response, healthy, changed, config, acquisition,
                     independent=False, aggregation="geometric"):
    hardware = candidate["hardware"]
    circuit = candidate["circuit"]
    caps = [attenuation_cap(h.bridge.resistance_ratio, circuit, config["max_shunt_ohm"])
            for h in hardware]
    variance = sum(acquisition_variances(hardware, config, acquisition))
    reference = acquisition["reference_calibration_index"]
    if not independent:
        return optimize_exact_network(hardware, response, healthy, changed,
                                       variance, caps, reference, aggregation)
    network, fits = [], []
    for q, device in enumerate(hardware):
        resolved = np.ones(len(changed), dtype=bool)
        if aggregation == "geometric":
            group = device.bridge.gauge_indices
            null = response_spaces(response[group])[1]
            increments = (changed - healthy)[..., group]
            resolved = np.linalg.norm(increments @ null, axis=(1, 2)) > (
                100 * np.finfo(float).eps * np.linalg.norm(increments, axis=(1, 2)))
            if not np.any(resolved):
                raise ValueError("No prescribed region is separable in a bridge group")
        fitted, info = optimize_exact_network([device], response, healthy, changed[resolved],
                                              variance[q:q + 1], caps[q:q + 1], reference,
                                              aggregation)
        network.extend(fitted)
        fits.append(info | {"regions": np.flatnonzero(resolved).tolist()})
    return network, {"aggregation": aggregation, "bridges": fits}


def evaluate_candidate(candidate, data, layout, saved, healthy, changed, config, acquisition):
    """Evaluate a chosen design without changing its source, components or routes."""
    name, seed = data["name"], data["seeds"][layout]
    hardware, network = candidate["hardware"], candidate["network"]
    channels = len(network)
    key = f"{name}_l{layout}_c{channels}"
    gain = saved[name + "_actual_gauge_factor_ratio"][layout]
    current_variance, reference_variance = acquisition_variances(hardware, config, acquisition)
    log_ratio, energies, y = design_ratio(hardware, healthy, changed, config, acquisition)
    design_delta = np.stack([h.voltage(changed) - h.voltage(healthy) for h in hardware], -1)
    references, currents, reference_powers, powers = [], [], [], []
    design_peaks, service_peaks = [], []
    reference_strain = data["calibration"][acquisition["reference_calibration_index"], layout] * gain
    for q, device in enumerate(hardware):
        group = device.bridge.gauge_indices
        errors = {"shunt_error": acquisition["component_uniform_tolerance"]
                      * saved[key + "_shunt_error_draw"][:, None, group],
                  "ballast_error": acquisition["component_uniform_tolerance"]
                      * saved[key + "_ballast_error_draw"][:, None, q]}
        reference = device.operating_point(reference_strain[None, None], **errors)
        current = [device.operating_point(data[state][:, layout][None] * gain, **errors)
                   for state in ("healthy", "changed")]
        references.append(reference["voltage_v"][:, 0])
        currents.append(np.stack([p["voltage_v"] for p in current]))
        reference_powers.append(reference["source_power_w"][:, 0])
        powers.append(np.stack([p["source_power_w"] for p in current]))
        service_peaks.append(np.maximum(reference["gauge_power_w"].max(axis=(0, 1)),
            np.stack([p["gauge_power_w"].max(axis=(0, 1)) for p in current]).max(axis=0)))
        design_peaks.append(np.maximum(
            device.operating_point(healthy)["gauge_power_w"].max(axis=0),
            device.operating_point(changed)["gauge_power_w"].max(axis=(0, 1))))
    reference, current = np.stack(references, -1), np.stack(currents, -1)
    reference_power, power = np.stack(reference_powers, -1), np.stack(powers, -1)
    rng = np.random.default_rng(acquisition["random_seed"] + 10 * seed + channels)
    measured, recorded, sampled = sample_fixed_reference(
        current, reference, current_variance, reference_variance, rng)
    delta = current[1] - current[0]
    full, grouped = regional_projection(data["basis"][layout],
        data["design"][:, :, layout] - data["calibration"][:, layout],
        [b.gauge_indices for b in network])
    arrays = {
        "gauge_order": np.concatenate([b.gauge_indices for b in network]),
        "arm": np.concatenate([b.arm for b in network]),
        "attenuation": np.concatenate([b.attenuation for b in network]),
        "arm_units": np.array([b.arm_units for b in network]),
        "supply_voltage_v": 2 * candidate["circuit"].voltage,
        "design_channel_energy_v2": energies, "design_increment_v": design_delta,
        "design_healthy_operational_v": y, "global_projected_energy": full,
        "group_projected_energy": grouped,
        "current_noise_variance_v2": current_variance,
        "reference_noise_variance_v2": reference_variance,
        "raw_reference_v": reference, "recorded_reference_v": recorded,
        "raw_current_v": current, "sampled_raw_current_v": measured,
        "operational_v": current - reference[:, None], "sampled_operational_v": sampled,
        "paired_increment_v": delta, "reference_source_power_w": reference_power,
        "source_power_w": power, "design_gauge_peak_power_w": np.concatenate(design_peaks),
        "service_gauge_peak_power_w": np.concatenate(service_peaks),
    }
    row = {
        "structure": name, "layout": layout, "layout_seed": seed,
        "supply_voltage_v": 2 * candidate["circuit"].voltage,
        "nominal_source_power_mw": float(source_power(hardware, np.zeros(healthy.shape[-1])) * 1e3),
        "operating_power_min_max_mw": [float(min(power.sum(-1).min(), reference_power.sum(-1).min()) * 1e3),
                                        float(max(power.sum(-1).max(), reference_power.sum(-1).max()) * 1e3)],
        "conductor_length_m": float(sum(r["total_conductor_m"] for r in candidate["routes"])),
        "design_gauge_peak_power_uw": float(np.max(arrays["design_gauge_peak_power_w"]) * 1e6),
        "service_gauge_peak_power_uw": float(np.max(arrays["service_gauge_peak_power_w"]) * 1e6),
        "design_ratio": float(np.exp(log_ratio)),
        "design_region_rms_uv": (np.sqrt(energies.sum(-1)) * 1e6).tolist(),
        "operational_noise_rms_uv": float(np.sqrt((current_variance + reference_variance).sum()) * 1e6),
        "healthy_sampled_norm_uv": norm_statistics(sampled[0]),
        "changed_sampled_norm_uv": norm_statistics(sampled[1]),
        "paired_increment_norm_uv": norm_statistics(delta),
    }
    export = network_export(candidate, data, layout, gain, arrays["design_gauge_peak_power_w"],
                            arrays["service_gauge_peak_power_w"])
    return arrays, row, export


def summarize(records, arrays, selections):
    summaries = []
    for name in ("ring", "frame"):
        modes = ["original", *(["reference_matched"] if name == "ring" else []),
                 "independent", "joint_fixed", "selected", "scaled"]
        for mode in modes:
            chosen = []
            for selection in (s for s in selections if s["structure"] == name):
                key = selection["mode"] if mode == "selected" else mode
                chosen.append(next(r for r in records if r["structure"] == name
                    and r["layout"] == selection["layout"] and r["mode"] == key))
            row = {"structure": name, "mode": mode}
            for key in ("nominal_source_power_mw", "conductor_length_m", "operational_noise_rms_uv"):
                row[key] = float(np.median([r[key] for r in chosen]))
            for key in ("design_gauge_peak_power_uw", "service_gauge_peak_power_uw"):
                row[key] = max(r[key] for r in chosen)
            row["supply_voltage_min_max_v"] = [min(r["supply_voltage_v"] for r in chosen),
                                                max(r["supply_voltage_v"] for r in chosen)]
            for key in ("healthy_sampled_norm_uv", "changed_sampled_norm_uv", "paired_increment_norm_uv"):
                row[key] = {stat: float(np.median([r[key][stat] for r in chosen]))
                            for stat in ("median", "p10", "p95")}
            increments = np.concatenate([np.linalg.norm(
                arrays[f"{name}_l{r['layout']}_{r['mode']}_paired_increment_v"], axis=-1).ravel()
                for r in chosen])
            row["pooled_increment_p10_uv"] = float(np.quantile(increments, .1) * 1e6)
            summaries.append(row)
    return summaries


def run():
    output = Path("results/passive-network")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path("inputs/passive_robustness_task.json").read_text())
    acquisition = json.loads(Path("inputs/passive_readout_task.json").read_text())
    saved = np.load("results/passive-robustness/hardware.npz")
    circuit = Circuit(gauge_factor=50.)
    arrays, records, exports, selections = {}, [], [], []
    for data in source_data():
        name = data["name"]
        for layout, seed in enumerate(data["seeds"]):
            key = f"{name}_l{layout}_c4"
            response = saved[key + "_" + acquisition["source_condition"] + "_response"]
            ratio = saved[name + "_actual_nominal_ratio"][layout]
            healthy = data["coefficients"] @ response.T
            increments = data["design"][:, :, layout] - data["calibration"][:, layout]
            changed = healthy + increments
            geometry, depth = distances(data, layout), data["depth"][layout]
            baseline = retained_network(saved, name, layout, 4, acquisition["source_condition"])
            original = routed_candidate(baseline, circuit, geometry, depth, config)
            power = float(source_power(original["hardware"], np.zeros(healthy.shape[-1])))

            def candidate(network):
                return routed_candidate(network, circuit, geometry, depth, config, power)

            candidates, fits = {"original": original}, {}
            if name == "ring":
                network, fits["reference_matched"] = refine_candidate(
                    original, response, healthy, changed, config, acquisition, True, "mean")
                candidates["reference_matched"] = candidate(network)
            start = candidates.get("reference_matched", original)
            network, fits["independent"] = refine_candidate(
                start, response, healthy, changed, config, acquisition, True)
            candidates["independent"] = candidate(network)
            network, fits["joint_fixed"] = refine_candidate(
                candidates["independent"], response, healthy, changed, config, acquisition)
            candidates["joint_fixed"] = candidate(network)

            order = np.concatenate([b.gauge_indices for b in baseline])
            initial = []
            for q in range(4):
                group = order[q::4]
                weight = distributed_target_weight(response[group],
                    increments[..., group].reshape(-1, len(group)).T, seed + q)
                bridge = TrimmedBridge.from_weight(group, weight)
                cap = attenuation_cap(ratio[group], circuit, config["max_shunt_ohm"])
                initial.append(replace(bridge, attenuation=cap * bridge.attenuation,
                                       nominal_ratio=ratio[group]))
            network, fits["joint_regrouped"] = refine_candidate(
                candidate(initial), response, healthy, changed, config, acquisition)
            candidates["joint_regrouped"] = candidate(network)
            chosen = max(("joint_fixed", "joint_regrouped"), key=lambda mode:
                design_ratio(candidates[mode]["hardware"], healthy, changed, config, acquisition)[0])
            selection = candidates[chosen]
            selections.append({"structure": name, "layout": layout, "layout_seed": seed, "mode": chosen})
            scaled, supply, hardware, fits["scaled"] = optimize_arm_scales(
                selection["network"], selection["circuit"], selection["routes"],
                healthy, changed, config, acquisition)
            candidates["scaled"] = {"network": scaled, "circuit": supply,
                                    "routes": selection["routes"], "hardware": hardware}
            # All designs and the selection are now fixed; evaluate retained cases.
            for mode, current in candidates.items():
                values, row, export = evaluate_candidate(
                    current, data, layout, saved, healthy, changed, config, acquisition)
                arrays.update({f"{name}_l{layout}_{mode}_{label}": value for label, value in values.items()})
                if mode in fits:
                    row["fit"] = fits[mode]
                records.append(row | {"mode": mode})
                exports.append(export | {"mode": mode})
            print(name, seed, chosen, {r["mode"]: round(r["design_ratio"], 2)
                  for r in records if r["structure"] == name and r["layout"] == layout}, flush=True)
    np.savez_compressed(output / "acquisition.npz", **arrays)
    metadata = {
        "channels": 4, "acquisition": acquisition, "source_condition": acquisition["source_condition"],
        "records": records, "selected_modes": selections,
        "summaries": summarize(records, arrays, selections),
        "design": "Same commissioned effective healthy basis and nominal forward increments as the retained studies. "
                  "Every candidate uses the full routed receiver circuit, one loaded reference, both averaging variances "
                  "and the original installation's nominal source power including supply leads.",
        "modes": {
            "original": "Historical independently designed coefficients, evaluated under the common acquisition task.",
            "reference_matched": "Ring only: independent mean-energy refinement after unifying reference, noise and circuitry.",
            "independent": "Independent geometric regional energies for locally separable regions; fixed original arms and groups.",
            "joint_fixed": "Regional energies pooled over channels before geometric aggregation; same arms, groups and capacities.",
            "joint_regrouped": "Cyclic interleaving, new design-based signs and joint refinement, at matched source power.",
            "selected": "Larger full-circuit design objective between joint_fixed and joint_regrouped.",
            "scaled": "Continuous arm scales on selected coefficients, arms and routes, at the same nominal source power.",
        },
        "evaluation": {"ring": "32 retained test load-sector pairs per installation",
                       "frame": "40 retained six-mode confirmation load-patch pairs per installation"},
        "sampling": "Sixteen retained 100-ppm component-setting draws and 0.05-microstrain commissioning noise. "
                    "Same standard-normal noise draws across candidates; one reference shared by all cases in a draw. "
                    "No post-reference component drift in this controlled comparison.",
        "resources": "Nominal source power is evaluated at zero strain. Actual source power retains reference and "
                     "service axes. Design gauge peaks use calibration/design states only; service peaks use the retained "
                     "evaluation and component-setting draws. Neither is a device safety limit.",
        "projection_basis": "Nominal mechanical basis and nominal increments; voltage design uses commissioned effective coordinates.",
        "array_axes": {"current": ["state", "hardware_draw", "case", "channel"],
                       "reference": ["hardware_draw", "channel"],
                       "design_increment": ["region", "design_load", "channel"]},
    }
    (output / "acquisition.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "acquisition_netlists.json").write_text(json.dumps(exports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
