"""Commissioned passive bridges on the retained ring and asymmetric frame."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .bridge_network import Circuit
from .frame_response import asymmetric_frame
from .passive_bridge import TrimmedBridge, distributed_target_weight, optimize_exact_bridge, target_null_weight
from .passive_hardware import BridgeHardware, route_bridge


def source_data():
    ring = np.load("results/random-bridges/healthy_exchange.npz")
    synthesis = np.load("results/passive-bridges/synthesis.npz")
    # The physical coefficients are uniquely recovered from the saved healthy
    # strains and full-rank response basis, without refitting a mechanical model.
    coefficients = np.linalg.lstsq(ring["healthy_basis_per_pa"][0], ring["calibration_strain"][:, 0].T, rcond=None)[0].T
    design = np.einsum("cr,dlgr->dclg", coefficients, synthesis["design_changed_basis_per_pa"])
    yield {"name": "ring", "basis": ring["healthy_basis_per_pa"], "coefficients": coefficients,
           "calibration": ring["calibration_strain"], "design": design,
           "healthy": ring["test_healthy_strain"], "changed": ring["test_changed_strain"],
           "theta": ring["theta_rad"], "depth": ring["depth_m"], "seeds": list(range(7100, 7116))}
    frame = np.load("results/frame-bridges/synthesis.npz")
    yield {"name": "frame", "basis": frame["r6_healthy_basis"], "coefficients": frame["r6_calibration_coefficients"],
           "calibration": frame["r6_calibration_strain"], "design": frame["r6_design_changed_strain"],
           "healthy": frame["r6_confirmation_healthy_strain"], "changed": frame["r6_confirmation_changed_strain"],
           "member": frame["member"], "fraction": frame["fraction"], "depth": frame["depth_m"], "seeds": list(range(8200, 8208))}


def distances(data, layout):
    depth = data["depth"][layout]
    if data["name"] == "frame":
        return asymmetric_frame().path_distances(data["member"][layout], data["fraction"][layout], depth)
    theta = data["theta"][layout]
    center = 3. * abs((theta[:, None] - theta[None, :] + np.pi) % (2 * np.pi) - np.pi)
    cell = center + abs(depth[:, None]) + abs(depth[None, :])
    np.fill_diagonal(cell, 0.)
    instrument = 3. * np.minimum(theta, 2 * np.pi - theta)
    return cell, instrument, center


def synthesize(data, layout, groups, response, resistance_ratio, circuit, config):
    reference = data["coefficients"] @ response.T
    # Forward change increments remain the nominal model's prediction. Healthy
    # commissioning data do not supply actual damaged-gauge observations.
    change_prediction = data["design"][:, :, layout] - data["calibration"][:, layout]
    design = reference + change_prediction
    network = []
    for channel, group in enumerate(groups):
        target_columns = change_prediction[..., group].reshape(-1, len(group)).T
        if data["name"] == "frame":
            weight = distributed_target_weight(response[group], target_columns, data["seeds"][layout] + channel)
        else:
            weight, _ = target_null_weight(response[group], target_columns)
        bridge = TrimmedBridge.from_weight(group, weight)
        ratio = resistance_ratio[group]
        upper = float(np.min(np.minimum(1 / ratio, ratio * (
            config["max_shunt_ohm"] / (circuit.resistance * ratio + config["max_shunt_ohm"]))**2)))
        bridge = replace(bridge, attenuation=upper * bridge.attenuation, nominal_ratio=ratio)
        variance = bridge.mean_noise_variance(circuit, config["average_time_s"], 1e-6, .005)
        bridge, _ = optimize_exact_bridge(bridge, response, reference, design, circuit, variance, upper,
                                         "geometric" if data["name"] == "frame" else "mean")
        network.append(bridge)
    return network


def voltage_summary(healthy, changed, variance):
    # Arrays: hardware draw, test load, channel. One zero reference per draw.
    hn, dn = np.linalg.norm(healthy, axis=-1), np.linalg.norm(changed - healthy, axis=-1)
    return {"healthy_norm_uv_median": float(np.median(hn) * 1e6),
            "increment_norm_uv_median": float(np.median(dn) * 1e6),
            "healthy_norm_uv_p95": float(np.quantile(hn, .95) * 1e6),
            "noise_norm_rms_uv": float(np.sqrt(sum(variance)) * 1e6)}


def run():
    config = json.loads(Path("inputs/passive_robustness_task.json").read_text())
    circuit = Circuit(gauge_factor=50.)
    arrays, records, netlists = {}, [], []
    draws = config["hardware_draws"]
    for data in source_data():
        name = data["name"]
        layout_count, gauges, _ = data["basis"].shape
        actual_ratios, actual_gains, calibration_errors = [], [], []
        for layout, seed in enumerate(data["seeds"]):
            rng = np.random.default_rng(config["random_seed"] + seed)
            resistance = 1 + rng.uniform(-1., 1., gauges) * config["gauge_nominal_uniform_tolerance"]
            gain = 1 + rng.uniform(-1., 1., gauges) * config["gauge_factor_uniform_tolerance"]
            measurement_error = rng.normal(size=data["calibration"][:, layout].shape)
            actual_ratios.append(resistance)
            actual_gains.append(gain)
            calibration_errors.append(measurement_error)
            actual_healthy, actual_changed = data["healthy"][:, layout] * gain, data["changed"][:, layout] * gain
            cell_distance, instrument_distance, center_distance = distances(data, layout)
            ordering = np.argsort(data["theta"][layout]) if name == "ring" else np.lexsort((data["fraction"][layout], data["member"][layout]))
            for channels in config["channels"]:
                groups = np.array_split(ordering, channels)
                shunt_errors = rng.uniform(-1., 1., (draws, gauges))
                ballast_errors = rng.uniform(-1., 1., (draws, channels, 4))
                drift_gauge = rng.normal(size=(draws, gauges))
                drift_shunt = rng.normal(size=(draws, gauges))
                drift_ballast = rng.normal(size=(draws, channels, 4))
                prefix = f"{name}_l{layout}_c{channels}"
                for key, value in {"shunt_error_draw": shunt_errors, "ballast_error_draw": ballast_errors,
                                   "gauge_drift_draw": drift_gauge, "shunt_drift_draw": drift_shunt,
                                   "ballast_drift_draw": drift_ballast}.items():
                    arrays[f"{prefix}_{key}"] = value
                conditions = [("nominal", None), *zip(("commissioned", "calibration_0p05ue", "calibration_0p5ue"),
                                                       config["calibration_strain_noise_rms"])]
                for condition, sigma in conditions:
                    if sigma is None:
                        response, ratio = data["basis"][layout], np.ones(gauges)
                    else:
                        measured = data["calibration"][:, layout] * gain + sigma * measurement_error
                        response = np.linalg.lstsq(data["coefficients"], measured, rcond=None)[0].T
                        ratio = resistance
                    network = synthesize(data, layout, groups, response, ratio, circuit, config)
                    hardware, routes = [], []
                    for bridge in network:
                        route = route_bridge(bridge, cell_distance, center_distance, instrument_distance,
                                             data["depth"][layout], config["wire_resistance_ohm_per_m"])
                        hardware.append(BridgeHardware.from_design(bridge, circuit, route["arm_resistance_ohm"],
                            route["one_instrument_lead_ohm"], route["one_instrument_lead_ohm"],
                            config["differential_input_resistance_ohm"], resistance[bridge.gauge_indices]))
                        routes.append(route)
                    key = f"{prefix}_{condition}"
                    arrays[f"{key}_response"] = response
                    arrays[f"{key}_attenuation"] = np.concatenate([b.attenuation for b in network])
                    arrays[f"{key}_gauge_order"] = np.concatenate([b.gauge_indices for b in network])
                    arrays[f"{key}_arm"] = np.concatenate([b.arm for b in network])
                    arrays[f"{key}_arm_units"] = np.array([b.arm_units for b in network])
                    variance = [b.mean_noise_variance(circuit, config["average_time_s"], 1e-6, .005) for b in network]
                    tolerance_rows = []
                    for tolerance in config["component_uniform_tolerance"]:
                        h, d, zero = [], [], []
                        for channel, device in enumerate(hardware):
                            group = device.bridge.gauge_indices
                            errors = {"shunt_error": tolerance * shunt_errors[:, None, group],
                                      "ballast_error": tolerance * ballast_errors[:, None, channel]}
                            v0 = device.voltage(np.zeros((1, 1, gauges)), **errors)
                            h.append(device.voltage(actual_healthy[None], **errors) - v0)
                            d.append(device.voltage(actual_changed[None], **errors) - v0)
                            zero.append(v0[:, 0])
                        h, d, zero = np.stack(h, -1), np.stack(d, -1), np.stack(zero, -1)
                        suffix = f"trim{tolerance * 1e6:g}ppm"
                        arrays[f"{key}_{suffix}_healthy_v"], arrays[f"{key}_{suffix}_changed_v"] = h, d
                        arrays[f"{key}_{suffix}_zero_v"] = zero
                        tolerance_rows.append({"component_tolerance": tolerance, **voltage_summary(h, d, variance),
                                               "raw_zero_norm_uv_median": float(np.median(np.linalg.norm(zero, axis=-1)) * 1e6)})
                    row = {"structure": name, "layout_seed": seed, "channels": channels, "condition": condition,
                           "calibration_strain_noise_rms": sigma, "tolerances": tolerance_rows,
                           "conductor_length_m": sum(route["total_conductor_m"] for route in routes),
                           "active_gauges": int(sum(np.count_nonzero(b.attenuation) for b in network)),
                           "nominal_power_mw": float(sum(4 * circuit.voltage**2 / (b.arm_units * circuit.resistance) for b in network) * 1e3)}
                    if condition == "calibration_0p05ue":
                        row["drift"] = []
                        for drift in config["drift_rms"]:
                            healthy = []
                            for channel, device in enumerate(hardware):
                                group = device.bridge.gauge_indices
                                errors = {"shunt_error": 1e-4 * shunt_errors[:, None, group],
                                          "ballast_error": 1e-4 * ballast_errors[:, None, channel]}
                                zero = device.voltage(np.zeros((1, 1, gauges)), **errors)
                                healthy.append(device.voltage(actual_healthy[None], **errors,
                                    gauge_drift=drift * drift_gauge[:, None, group],
                                    shunt_drift=drift * drift_shunt[:, None, group],
                                    ballast_drift=drift * drift_ballast[:, None, channel]) - zero)
                            healthy = np.stack(healthy, -1)
                            arrays[f"{key}_drift{drift * 1e6:g}ppm_healthy_v"] = healthy
                            row["drift"].append({"independent_component_drift_rms": drift,
                                "healthy_norm_uv_median": float(np.median(np.linalg.norm(healthy, axis=-1)) * 1e6),
                                "healthy_norm_uv_p95": float(np.quantile(np.linalg.norm(healthy, axis=-1), .95) * 1e6)})
                        # The reference hardware includes the first 100-ppm trim
                        # draw and a fixed zero, with explicit component values.
                        netlists.append({"structure": name, "layout_seed": seed, "channels": channels,
                            "rails_v": {"H": 1., "L": -1.}, "hardware_draw": 0, "bridges": [
                                {"route": route, "zero_strain_netlist": device.netlist(np.zeros(gauges),
                                    1e-4 * shunt_errors[0, device.bridge.gauge_indices], 1e-4 * ballast_errors[0, channel])}
                                for channel, (device, route) in enumerate(zip(hardware, routes))]})
                    records.append(row)
                print(f"{name} seed {seed}, {channels} channels: commissioning and hardware scenarios complete", flush=True)
        arrays[f"{name}_actual_nominal_ratio"] = np.array(actual_ratios)
        arrays[f"{name}_actual_gauge_factor_ratio"] = np.array(actual_gains)
        arrays[f"{name}_unit_calibration_error"] = np.array(calibration_errors)
    summary = {"configuration": config, "voltage_array_axes": ["hardware_draw", "test_case", "channel"],
               "records": records, "medians_of_layout_statistics": []}
    for structure in ("ring", "frame"):
        for channels in config["channels"]:
            for condition, _ in conditions:
                selected = [row for row in records if row["structure"] == structure and row["channels"] == channels and row["condition"] == condition]
                result = {"structure": structure, "channels": channels, "condition": condition,
                    "conductor_length_m": float(np.median([row["conductor_length_m"] for row in selected])),
                    "active_gauges": float(np.median([row["active_gauges"] for row in selected])),
                    "nominal_power_mw": float(np.median([row["nominal_power_mw"] for row in selected])),
                    "tolerances": [{key: float(np.median([row["tolerances"][i][key] for row in selected]))
                                    for key in selected[0]["tolerances"][i]} for i in range(len(config["component_uniform_tolerance"]))]}
                if "drift" in selected[0]:
                    result["drift"] = [{key: float(np.median([row["drift"][i][key] for row in selected]))
                                        for key in selected[0]["drift"][i]} for i in range(len(config["drift_rms"]))]
                summary["medians_of_layout_statistics"].append(result)
    output = Path("results/passive-robustness")
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "hardware.npz", **arrays)
    (output / "hardware.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "netlists.json").write_text(json.dumps(netlists, indent=2) + "\n", encoding="utf-8")
    for row in summary["medians_of_layout_statistics"]:
        main = row["tolerances"][1]
        print(row["structure"], row["channels"], row["condition"], "100ppm:",
              round(main["healthy_norm_uv_median"], 4), round(main["increment_norm_uv_median"], 4), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
