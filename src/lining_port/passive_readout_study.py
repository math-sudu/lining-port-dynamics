"""Operational readout of retained commissioned ring and frame circuits."""
import json
from pathlib import Path

import numpy as np

from .bridge_network import Circuit
from .passive_bridge import TrimmedBridge
from .passive_hardware import BridgeHardware
from .passive_readout import sample_fixed_reference
from .passive_robustness_study import source_data


def norm_statistics(voltage):
    values = np.linalg.norm(voltage, axis=-1) * 1e6
    return {"median": float(np.median(values)), "p10": float(np.quantile(values, .1)),
            "p95": float(np.quantile(values, .95))}


def run():
    config = json.loads(Path("inputs/passive_readout_task.json").read_text())
    hardware_config = json.loads(Path("inputs/passive_robustness_task.json").read_text())
    saved = np.load("results/passive-robustness/hardware.npz")
    netlists = json.loads(Path("results/passive-robustness/netlists.json").read_text())
    ring_cases = [row for row in json.loads(Path("results/fixed-installation/cases.json").read_text())
                  if row["split"] == "test"]
    frame_patches = np.load("results/frame-bridges/synthesis.npz")["confirmation_patch"]
    circuit = Circuit(gauge_factor=50.)
    drift_levels = [0., *hardware_config["drift_rms"]]
    arrays, records, layout_records = {}, [], []
    reference_index = config["reference_calibration_index"]
    for data in source_data():
        name = data["name"]
        # Ring regions use the twelve synthesis-sector centers; frame labels are
        # the physical member indices in the saved confirmation patches.
        regions = (np.floor((np.array([r["sector_center"] for r in ring_cases])
                             + np.pi / 12) % (2 * np.pi) / (np.pi / 6)).astype(int)
                   if name == "ring" else frame_patches[:, 0].astype(int))
        arrays[f"{name}_case_region"] = regions
        for layout, seed in enumerate(data["seeds"]):
            ratio = saved[name + "_actual_nominal_ratio"][layout]
            gain = saved[name + "_actual_gauge_factor_ratio"][layout]
            reference_strain = data["calibration"][reference_index, layout] * gain
            service_strains = (data["healthy"][:, layout] * gain,
                               data["changed"][:, layout] * gain)
            for channels in hardware_config["channels"]:
                prefix = f"{name}_l{layout}_c{channels}"
                key = f"{prefix}_{config['source_condition']}"
                groups = np.array_split(saved[key + "_gauge_order"], channels)
                coefficients = np.array_split(saved[key + "_attenuation"], channels)
                arms = np.array_split(saved[key + "_arm"], channels)
                units = saved[key + "_arm_units"]
                row = next(r for r in netlists if r["structure"] == name
                           and r["layout_seed"] == seed and r["channels"] == channels)
                reference, current, current_variance, reference_variance = [], [], [], []
                for channel, group in enumerate(groups):
                    bridge = TrimmedBridge(group, arms[channel], coefficients[channel],
                                           int(units[channel]), ratio[group])
                    route = row["bridges"][channel]["route"]
                    hardware = BridgeHardware.from_design(
                        bridge, circuit, route["arm_resistance_ohm"],
                        route["one_instrument_lead_ohm"], route["one_instrument_lead_ohm"],
                        hardware_config["differential_input_resistance_ohm"])
                    errors = {
                        "shunt_error": config["component_uniform_tolerance"]
                                       * saved[prefix + "_shunt_error_draw"][:, None, group],
                        "ballast_error": config["component_uniform_tolerance"]
                                         * saved[prefix + "_ballast_error_draw"][:, None, channel],
                    }
                    reference.append(hardware.voltage(reference_strain[None, None], **errors)[:, 0])
                    drift_currents = []
                    for drift in drift_levels:
                        deviations = {
                            "gauge_drift": drift * saved[prefix + "_gauge_drift_draw"][:, None, group],
                            "shunt_drift": drift * saved[prefix + "_shunt_drift_draw"][:, None, group],
                            "ballast_drift": drift * saved[prefix + "_ballast_drift_draw"][:, None, channel],
                        }
                        drift_currents.append(np.stack([
                            hardware.voltage(strain[None], **errors, **deviations)
                            for strain in service_strains]))
                    current.append(np.stack(drift_currents))
                    for duration, variances in (
                        (hardware_config["average_time_s"], current_variance),
                        (config["reference_average_time_s"], reference_variance),
                    ):
                        variances.append(bridge.mean_noise_variance(
                            circuit, duration, config["independent_sample_sigma_v"],
                            config["sample_period_s"], config["temperature_k"]))
                reference, current = np.stack(reference, -1), np.stack(current, -1)
                current_variance, reference_variance = map(np.array, (current_variance, reference_variance))
                rng = np.random.default_rng(config["random_seed"] + 10 * seed + channels)
                measured, recorded, sampled = sample_fixed_reference(
                    current, reference, current_variance, reference_variance, rng)
                operational = current - reference[:, None, :]
                for label, value in {
                    "raw_reference_v": reference, "recorded_reference_v": recorded,
                    "reference_effective_strain": reference_strain,
                    "current_noise_variance_v2": current_variance,
                    "reference_noise_variance_v2": reference_variance,
                }.items():
                    arrays[f"{prefix}_{label}"] = value
                layout_records.append({
                    "structure": name, "layout_seed": seed, "channels": channels,
                    "reference_load_coefficients": data["coefficients"][reference_index].tolist(),
                    "reference_peak_effective_strain_ue": float(np.max(abs(reference_strain)) * 1e6),
                    "raw_reference_min_max_uv": (np.array([reference.min(), reference.max()]) * 1e6).tolist(),
                    "current_noise_norm_rms_uv": float(np.sqrt(current_variance.sum()) * 1e6),
                    "reference_noise_norm_rms_uv": float(np.sqrt(reference_variance.sum()) * 1e6),
                    "operational_noise_norm_rms_uv": float(np.sqrt((current_variance + reference_variance).sum()) * 1e6),
                })
                for i, drift in enumerate(drift_levels):
                    base = f"{prefix}_drift{drift * 1e6:g}ppm"
                    delta = current[i, 1] - current[i, 0]
                    arrays[base + "_paired_increment_v"] = delta
                    record = {"structure": name, "layout_seed": seed, "channels": channels,
                              "independent_component_drift_rms": drift,
                              "paired_increment_norm_uv": norm_statistics(delta)}
                    for j, state in enumerate(("healthy", "changed")):
                        for label, values in (("raw", current), ("sampled_raw", measured),
                                              ("operational", operational), ("sampled_operational", sampled)):
                            arrays[f"{base}_{state}_{label}_v"] = values[i, j]
                        record[state + "_norm_uv"] = norm_statistics(operational[i, j])
                        record[state + "_sampled_norm_uv"] = norm_statistics(sampled[i, j])
                    raw_values = np.concatenate([reference.ravel(), current[i].ravel()])
                    record["raw_receiver_min_max_uv"] = (np.array([raw_values.min(), raw_values.max()]) * 1e6).tolist()
                    record["regions"] = [
                        {"region": int(region), "case_count": int(np.sum(regions == region)),
                         "healthy_sampled_norm_uv": norm_statistics(sampled[i, 0, :, regions == region]),
                         "changed_sampled_norm_uv": norm_statistics(sampled[i, 1, :, regions == region]),
                         "paired_increment_norm_uv": norm_statistics(delta[:, regions == region])}
                        for region in np.unique(regions)]
                    records.append(record)
                print(f"{name} seed {seed}, {channels} channels: recorded-reference readout", flush=True)
    summaries = []
    for name in ("ring", "frame"):
        for channels in hardware_config["channels"]:
            layouts = [r for r in layout_records if r["structure"] == name and r["channels"] == channels]
            for drift in drift_levels:
                selected = [r for r in records if r["structure"] == name and r["channels"] == channels
                            and r["independent_component_drift_rms"] == drift]
                summary = {"structure": name, "channels": channels, "independent_component_drift_rms": drift}
                for field in ("healthy_norm_uv", "changed_norm_uv", "paired_increment_norm_uv",
                              "healthy_sampled_norm_uv", "changed_sampled_norm_uv"):
                    summary[field] = {stat: float(np.median([r[field][stat] for r in selected]))
                                      for stat in ("median", "p10", "p95")}
                summary["operational_noise_norm_rms_uv"] = float(np.median([
                    r["operational_noise_norm_rms_uv"] for r in layouts]))
                summary["raw_receiver_min_max_uv"] = [
                    min(r["raw_receiver_min_max_uv"][0] for r in selected),
                    max(r["raw_receiver_min_max_uv"][1] for r in selected)]
                summaries.append(summary)
    metadata = {
        "configuration": config, "current_average_time_s": hardware_config["average_time_s"],
        "current_array_axes": ["hardware_draw", "case", "channel"],
        "reference_array_axes": ["hardware_draw", "channel"],
        "sampling": "One reference average after setting components at the first healthy commissioning load; "
                    "the same stored reference is subtracted from every subsequent current average. "
                    "Channels average simultaneously. Service intervals have independent white noise; "
                    "reference noise is shared across cases, states and drift scenarios of each hardware draw.",
        "noise_covariance": "Within a channel, variance(current-reference)=current_variance+reference_variance; "
                            "covariance between distinct service readings sharing that reference=reference_variance. "
                            "Channel noises are independent. Equal 4-s intervals give correlation 1/2.",
        "noise_model": "Existing balanced-arm Johnson resistance mR0 plus independent post-filter samples.",
        "case_sources": {"ring": "results/fixed-installation/cases.json: split=test, in source order",
                         "frame": "results/frame-bridges/synthesis.npz: confirmation_patch, in source order"},
        "region_labels": {"ring": "nearest design-sector center, 0..11 at multiples of pi/6",
                          "frame": "physical member index, 0..4"},
        "layout_records": layout_records, "records": records,
        "medians_of_layout_statistics": summaries,
    }
    output = Path("results/passive-robustness")
    np.savez_compressed(output / "readout.npz", **arrays)
    (output / "readout.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for row in summaries:
        print(row["structure"], row["channels"], f"drift={row['independent_component_drift_rms']:g}",
              "sampled H/C:", row["healthy_sampled_norm_uv"]["median"],
              row["changed_sampled_norm_uv"]["median"], flush=True)


if __name__ == "__main__":
    run()
