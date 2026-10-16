"""Cross-structure validation using fixed random gauges on an asymmetric frame."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .bridge_network import Circuit
from .frame_response import asymmetric_frame, joint_load_basis, random_installation
from .passive_bridge import TrimmedBridge, distributed_target_weight, optimize_exact_bridge, response_spaces, target_null_weight
from .passive_bridge_study import summarize_voltage


def load_samples(rng, count, rank):
    if rank == 2:
        return np.column_stack((rng.uniform(.4, 1.6, count), rng.uniform(-1., 1., count)))
    coefficients = rng.uniform(-1., 1., (count, rank))
    coefficients[:, 1::2] = rng.uniform(.4, 1.6, (count, rank // 2))
    return coefficients


def run():
    config = json.loads(Path("inputs/frame_bridge_task.json").read_text())
    frame = asymmetric_frame()
    layouts = [random_installation(seed, config["gauges_per_member"]) for seed in config["layout_seeds"]]
    installed = {key: np.stack([layout[key] for layout in layouts]) for key in layouts[0]}
    combined = {key: value.ravel() for key, value in installed.items()}
    layout_count, gauge_count = installed["member"].shape
    design_patches = [(member, center - config["patch_fraction"] / 2,
                       center + config["patch_fraction"] / 2, config["design_modulus_reduction"])
                      for member in range(5) for center in config["design_centers"]]
    test_rng = np.random.default_rng(config["test_seed"])
    test_member = np.tile(np.arange(5), config["tests_per_member"])
    test_center = test_rng.uniform(.13, .87, len(test_member))
    test_patches = [(int(member), center - config["patch_fraction"] / 2,
                    center + config["patch_fraction"] / 2, config["test_modulus_reduction"])
                   for member, center in zip(test_member, test_center)]
    confirmation_rng = np.random.default_rng(config["confirmation_seed"])
    confirmation_centers = confirmation_rng.uniform(.13, .87, len(test_member))
    confirmation_patches = [(int(member), center - config["patch_fraction"] / 2,
                            center + config["patch_fraction"] / 2, config["test_modulus_reduction"])
                           for member, center in zip(test_member, confirmation_centers)]
    six_loads = joint_load_basis(6)
    healthy6 = frame.response(six_loads, combined).reshape(layout_count, gauge_count, 6)
    design6 = np.stack([asymmetric_frame([patch]).response(six_loads, combined)
                        for patch in design_patches]).reshape(len(design_patches), layout_count, gauge_count, 6)
    changed6 = np.stack([asymmetric_frame([patch]).response(six_loads, combined)
                         for patch in test_patches]).reshape(len(test_patches), layout_count, gauge_count, 6)
    confirmation6 = np.stack([asymmetric_frame([patch]).response(six_loads, combined)
                              for patch in confirmation_patches]).reshape(len(confirmation_patches), layout_count, gauge_count, 6)
    arrays = {**installed, "joint_xy_m": frame.joints, "member_nodes": frame.members,
              "design_patch": np.array(design_patches), "test_patch": np.array(test_patches),
              "confirmation_patch": np.array(confirmation_patches)}
    circuit = Circuit(gauge_factor=config["gauge_factor"], resistance=config["resistance_ohm"])
    cap = (config["max_shunt_ohm"] / (config["max_shunt_ohm"] + circuit.resistance))**2
    rows, netlists = [], []
    for rank in config["healthy_ranks"]:
        loads = joint_load_basis(rank)
        transform = np.linalg.lstsq(six_loads.reshape(-1, 6), loads.reshape(-1, rank), rcond=None)[0]
        healthy_basis = healthy6 @ transform
        design_basis = design6 @ transform
        changed_basis = changed6 @ transform
        calibration_coefficients = load_samples(np.random.default_rng(config["calibration_seed"] + rank),
                                                config["calibration_cases"], rank)
        test_coefficients = load_samples(np.random.default_rng(config["test_seed"] + rank), len(test_patches), rank)
        calibration = np.einsum("cr,lgr->clg", calibration_coefficients, healthy_basis)
        design = np.einsum("cr,dlgr->dclg", calibration_coefficients, design_basis)
        healthy = np.einsum("cr,lgr->clg", test_coefficients, healthy_basis)
        changed = np.einsum("cr,clgr->clg", test_coefficients, changed_basis)
        confirmation_coefficients = load_samples(np.random.default_rng(config["confirmation_seed"] + rank), len(confirmation_patches), rank)
        confirmation_healthy = np.einsum("cr,lgr->clg", confirmation_coefficients, healthy_basis)
        confirmation_changed = np.einsum("cr,clgr->clg", confirmation_coefficients, confirmation6 @ transform)
        for key, value in {"healthy_basis": healthy_basis, "design_basis": design_basis,
                           "test_changed_basis": changed_basis, "calibration_coefficients": calibration_coefficients,
                           "test_coefficients": test_coefficients, "calibration_strain": calibration,
                           "design_changed_strain": design, "test_healthy_strain": healthy,
                           "test_changed_strain": changed, "confirmation_coefficients": confirmation_coefficients,
                           "confirmation_healthy_strain": confirmation_healthy,
                           "confirmation_changed_strain": confirmation_changed}.items():
            arrays[f"r{rank}_{key}"] = value
        for channels in config["channels"]:
            modes = {mode: {key: [] for key in ("healthy", "changed", "confirmation_healthy", "confirmation_changed", "attenuation", "arm", "units", "noise_variance")}
                     for mode in ("direct", "energy", "optimized")}
            for layout in range(layout_count):
                groups = np.array_split(np.lexsort((installed["fraction"][layout], installed["member"][layout])), channels)
                targets = design[:, :, layout] - calibration[:, layout]
                networks = {"direct": [], "energy": [], "optimized": []}
                details = []
                for channel, group in enumerate(groups):
                    weight, info = target_null_weight(healthy_basis[layout, group], targets[..., group].reshape(-1, len(group)).T)
                    initial = TrimmedBridge.from_weight(group, weight)
                    initial = replace(initial, attenuation=cap * initial.attenuation)
                    variance = initial.mean_noise_variance(circuit, config["average_time_s"], 1e-6, .005)
                    fitted, fit = optimize_exact_bridge(initial, healthy_basis[layout], calibration[:, layout],
                                                        design[:, :, layout], circuit, variance, cap)
                    networks["direct"].append(initial)
                    networks["energy"].append(fitted)
                    weight = distributed_target_weight(healthy_basis[layout, group], targets[..., group].reshape(-1, len(group)).T,
                                                       config["layout_seeds"][layout] + channel)
                    balanced = TrimmedBridge.from_weight(group, weight)
                    balanced = replace(balanced, attenuation=cap * balanced.attenuation)
                    balanced_variance = balanced.mean_noise_variance(circuit, config["average_time_s"], 1e-6, .005)
                    balanced, balanced_fit = optimize_exact_bridge(balanced, healthy_basis[layout], calibration[:, layout],
                        design[:, :, layout], circuit, balanced_variance, cap, "geometric")
                    networks["optimized"].append(balanced)
                    info["energy_fit"] = fit
                    fit = balanced_fit
                    details.append({"channel": channel, **info, **fit})
                row = {"rank": rank, "channels": channels, "layout_seed": config["layout_seeds"][layout], "design": details}
                for mode, network in networks.items():
                    vh = np.stack([bridge.voltage(healthy[:, layout], circuit) for bridge in network], axis=-1)
                    vc = np.stack([bridge.voltage(changed[:, layout], circuit) for bridge in network], axis=-1)
                    variance = np.array([bridge.mean_noise_variance(circuit, config["average_time_s"], 1e-6, .005) for bridge in network])
                    ch = np.stack([bridge.voltage(confirmation_healthy[:, layout], circuit) for bridge in network], -1)
                    cc = np.stack([bridge.voltage(confirmation_changed[:, layout], circuit) for bridge in network], -1)
                    attenuation, arms = np.empty(gauge_count), np.empty(gauge_count, dtype=int)
                    for bridge in network:
                        attenuation[bridge.gauge_indices], arms[bridge.gauge_indices] = bridge.attenuation, bridge.arm
                    for key, value in {"healthy": vh, "changed": vc, "confirmation_healthy": ch,
                                       "confirmation_changed": cc, "attenuation": attenuation, "arm": arms,
                                       "units": [bridge.arm_units for bridge in network], "noise_variance": variance}.items():
                        modes[mode][key].append(value)
                    row[mode] = summarize_voltage(vh, vc, variance)
                    row[f"{mode}_confirmation"] = summarize_voltage(ch, cc, variance)
                    if mode == "optimized":
                        netlists.append({"rank": rank, "channels": channels, "layout_seed": row["layout_seed"],
                            "rails_v": {"H": circuit.voltage, "L": -circuit.voltage},
                            "bridges": [{"gauge_indices_zero_based": b.gauge_indices.tolist(), "arm_zero_based": b.arm.tolist(),
                                         "attenuation": b.attenuation.tolist(), "arm_units": b.arm_units,
                                         "zero_strain_netlist": b.resistor_netlist(np.zeros(gauge_count), circuit)} for b in network]})
                _, null = response_spaces(healthy_basis[layout])
                increments = changed[:, layout] - healthy[:, layout]
                row["global_target_projected_energy_fraction"] = float(np.linalg.norm(increments @ null)**2 / np.linalg.norm(increments)**2)
                rows.append(row)
                print(f"frame rank {rank}, channels {channels}, seed {row['layout_seed']}: "
                      f"healthy {row['optimized']['healthy_norm_uv_median']:.4f}, "
                      f"target {row['optimized']['increment_norm_uv_median']:.3f} uV", flush=True)
            for mode, values in modes.items():
                for key, value in values.items():
                    arrays[f"r{rank}_c{channels}_{mode}_{key}"] = np.stack(value, axis=1 if key in (
                        "healthy", "changed", "confirmation_healthy", "confirmation_changed") else 0)
    # A dense response trace illustrates the compatible strain redistribution.
    field = {"member": np.repeat(np.arange(5), 101), "fraction": np.tile(np.linspace(.001, .999, 101), 5),
             "depth_m": np.full(505, .175)}
    representative_patch = (3, .36, .54, config["test_modulus_reduction"])
    field_load = joint_load_basis(2)
    arrays.update({f"field_{k}": v for k, v in field.items()})
    arrays["field_healthy_strain"] = frame.response(field_load, field)
    arrays["field_changed_strain"] = asymmetric_frame([representative_patch]).response(field_load, field)
    arrays["field_joint_loads"] = field_load
    arrays["field_patch"] = np.array(representative_patch)
    uniform = asymmetric_frame(modulus_scale=.9).response(six_loads, combined).reshape(healthy6.shape)
    arrays["uniform_modulus_changed_basis"] = uniform
    null_projection = []
    for h, d in zip(healthy6, uniform - healthy6):
        _, null = response_spaces(h)
        null_projection.append(float(np.linalg.norm(null.T @ d) / np.linalg.norm(d)))
    # An omitted middle-joint lateral load probes the specified load-space
    # condition. Both networks see exactly the same independently sampled loads.
    extra = np.random.default_rng(config["test_seed"] + 99).uniform(-.1, .1, len(test_patches))
    extended_healthy = arrays["r2_test_healthy_strain"] + np.einsum("c,lg->clg", extra, healthy6[..., 2])
    extended_changed = arrays["r2_test_changed_strain"] + np.einsum("c,clg->clg", extra, changed6[..., 2])
    arrays["omitted_load_coefficients"] = extra
    arrays["extended_healthy_strain"] = extended_healthy
    arrays["extended_changed_strain"] = extended_changed
    omission = []
    for rank in config["healthy_ranks"]:
        for channels in config["channels"]:
            vh, vc = [], []
            for layout, seed in enumerate(config["layout_seeds"]):
                record = next(row for row in netlists if row["rank"] == rank and row["channels"] == channels and row["layout_seed"] == seed)
                bridges = [TrimmedBridge(np.array(row["gauge_indices_zero_based"]), np.array(row["arm_zero_based"]),
                    np.array(row["attenuation"]), row["arm_units"]) for row in record["bridges"]]
                h = np.stack([b.voltage(extended_healthy[:, layout], circuit) for b in bridges], -1)
                c = np.stack([b.voltage(extended_changed[:, layout], circuit) for b in bridges], -1)
                vh.append(h)
                vc.append(c)
                omission.append({"rank": rank, "channels": channels, "layout_seed": seed,
                    **summarize_voltage(h, c, arrays[f"r{rank}_c{channels}_optimized_noise_variance"][layout])})
            arrays[f"r{rank}_c{channels}_extended_healthy_v"] = np.stack(vh, 1)
            arrays[f"r{rank}_c{channels}_extended_changed_v"] = np.stack(vc, 1)
    summary = {"configuration": config, "model": "Linear elastic plane Euler--Bernoulli frame; fixed bases; joint forces; "
        "member-axis finite-gauge strain; modulus steps split exactly. Uniform modulus change remains in the healthy space.",
        "uniform_change_projected_norm_fraction_max": max(null_projection),
        "voltage_array_axes": ["test_case", "layout", "channel"], "studies": rows,
        "omitted_load_study": omission,
        "medians_of_layout_statistics": [{"rank": r, "channels": c,
            **{mode: {key: float(np.median([row[mode][key] for row in rows if row["rank"] == r and row["channels"] == c]))
                      for key in rows[0][mode]} for mode in modes}}
            for r in config["healthy_ranks"] for c in config["channels"]],
        "confirmation_medians": [{"rank": r, "channels": c,
            **{mode: {key: float(np.median([row[f"{mode}_confirmation"][key] for row in rows if row["rank"] == r and row["channels"] == c]))
                      for key in rows[0][mode]} for mode in modes}}
            for r in config["healthy_ranks"] for c in config["channels"]]}
    output = Path("results/frame-bridges")
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "synthesis.npz", **arrays)
    (output / "synthesis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "netlists.json").write_text(json.dumps(netlists, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["medians_of_layout_statistics"], indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
