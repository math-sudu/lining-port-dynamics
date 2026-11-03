"""Resolve mechanical capture and circuit gain in the retained passive networks."""
import json
from pathlib import Path

import numpy as np

from .bridge_network import Circuit
from .passive_design_study import restore_network
from .passive_network import channel_capture
from .passive_resources import minimum_arm_units, routed_hardware
from .passive_robustness_study import source_data


MODES = {"original": "acquisition", "joint_regrouped": "acquisition",
         "scaled": "acquisition", "arms": "constrained", "constrained": "constrained"}


def circuit_allocation(hardware, routes, gauge_count):
    """Nominal arm occupancy and actual component power in each routed bridge."""
    rows = []
    for device, route in zip(hardware, routes):
        bridge, circuit = device.bridge, device.circuit
        cells = np.sqrt(bridge.resistance_ratio * bridge.attenuation)
        occupancy = np.array([cells[bridge.arm == j].sum() for j in range(4)])
        sign_totals = [cells[bridge.weight > 0].sum(), cells[bridge.weight < 0].sum()]
        op = device.operating_point(np.zeros(gauge_count))
        power = {key: float(np.sum(value)) for key, value in op.items() if key.endswith("power_w")}
        rows.append({
            "arm_units": bridge.arm_units,
            "minimum_arm_units": minimum_arm_units(bridge, circuit, route),
            "partition_lower_bound_without_wire": float(max(sign_totals[0] / 2, sign_totals[1] / 2, cells.max())),
            "cell_occupancy": occupancy.tolist(),
            "arm_wire_occupancy": (device.arm_lead / circuit.resistance).tolist(),
            "bridge_rail_voltage_v": float(2 * circuit.voltage * bridge.arm_units * circuit.resistance
                / (bridge.arm_units * circuit.resistance + 2 * device.supply_lead)),
            "unit_direction_gain_v": float(device.nominal_linear_gain() * np.linalg.norm(bridge.weight)),
            "nominal_power_w": power,
            "nominal_peak_gauge_power_w": float(op["gauge_power_w"].max()),
        })
    return rows


def run():
    folder = Path("results/passive-network")
    config = json.loads(Path("inputs/passive_readout_task.json").read_text())
    saved = np.load("results/passive-robustness/hardware.npz")
    studies = {name: np.load(folder / f"{name}.npz") for name in ("acquisition", "constrained")}
    exports = {name: {(r["structure"], r["layout"], r["mode"]): r
                     for r in json.loads((folder / f"{name}_netlists.json").read_text())}
               for name in studies}
    output, example, minima = {}, [], []
    for data in source_data():
        name = data["name"]
        fractions = {mode: [] for mode in MODES}
        for layout, seed in enumerate(data["seeds"]):
            response = saved[f"{name}_l{layout}_c4_{config['source_condition']}_response"]
            increments = data["design"][:, :, layout] - data["calibration"][:, layout]
            ratio = saved[name + "_actual_nominal_ratio"][layout]
            for mode, study in MODES.items():
                prefix = f"{name}_l{layout}_{mode}"
                arrays = studies[study]
                network = restore_network(arrays, prefix, ratio)
                record = exports[study][name, layout, mode]
                circuit = Circuit(gauge_factor=record["gauge_factor_reference"],
                                  resistance=record["resistance_reference_ohm"],
                                  voltage=record["source_voltage_v"] / 2)
                routes = [b["route"] for b in record["bridges"]]
                hardware = routed_hardware(network, circuit, routes,
                                           record["bridges"][0]["receiver_resistance_ohm"])
                values = channel_capture(response, increments, network)
                gains = np.array([h.nominal_linear_gain() * np.linalg.norm(h.bridge.weight) for h in hardware])
                values["gain_v"] = gains
                values["linear_channel_energy_v2"] = values["channel_energy_strain2"] * gains**2
                output.update({prefix + "_" + key: value for key, value in values.items()})
                full = values["global_energy_strain2"]
                group = values["group_energy_strain2"].sum(axis=-1)
                captured = values["channel_energy_strain2"].sum(axis=-1)
                fractions[mode].append(group / full)
                if name == "frame" and seed == 8206:
                    region = 13
                    example.append({
                        "mode": mode, "source_study": study, "array_prefix": prefix,
                        "global_energy_microstrain2": float(full[region] * 1e12),
                        "group_energy_microstrain2": float(group[region] * 1e12),
                        "captured_energy_microstrain2": float(captured[region] * 1e12),
                        "group_fraction": float(group[region] / full[region]),
                        "channel_fraction": float(captured[region] / full[region]),
                        "capture_within_group": float(captured[region] / group[region]),
                        "linear_channel_rms_uv": (np.sqrt(values["linear_channel_energy_v2"][region]) * 1e6).tolist(),
                        "exact_channel_rms_uv": (np.sqrt(arrays[prefix + "_design_channel_energy_v2"][region]) * 1e6).tolist(),
                        "linear_network_rms_uv": float(np.sqrt(values["linear_channel_energy_v2"][region].sum()) * 1e6),
                        "exact_network_rms_uv": float(np.sqrt(arrays[prefix + "_design_channel_energy_v2"][region].sum()) * 1e6),
                        "conductor_length_m": float(sum(r["total_conductor_m"] for r in routes)),
                        "design_peak_gauge_power_w": float(arrays[prefix + "_design_gauge_peak_power_w"].max()),
                        "channels": circuit_allocation(hardware, routes, len(response)),
                    })
        for mode in ("original", "joint_regrouped"):
            values = np.array(fractions[mode])
            layout, region = np.unravel_index(values.argmin(), values.shape)
            minima.append({"structure": name, "mode": mode, "group_fraction": float(values[layout, region]),
                           "layout_seed": data["seeds"][layout], "region_index": int(region)})
    metadata = {
        "coordinates": "The commissioned effective healthy response H and nominal forward increments D used by acquisition and constrained design. No gauge-factor ratio is applied again to D.",
        "averaging": "Uniform mean over the original design loads, equivalently columns D/sqrt(number of loads). Energies have strain-squared units; channel rows are Euclidean unit vectors in the full gauge order.",
        "array_axes": {"unit_directions": ["channel", "gauge"],
                       "group_energy_strain2": ["region", "channel"],
                       "channel_energy_strain2": ["region", "channel"],
                       "linear_channel_energy_v2": ["region", "channel"]},
        "mode_sources": MODES,
        "sources": ["results/passive-robustness/hardware.npz", "results/random-bridges/healthy_exchange.npz",
                    "results/passive-bridges/synthesis.npz", "results/frame-bridges/synthesis.npz",
                    *[f"results/passive-network/{s}{suffix}" for s in studies
                      for suffix in (".npz", "_netlists.json")]],
        "group_fraction_minima": minima,
        "example": {"structure": "frame", "layout_seed": 8206, "region_index": 13,
                    "region": "Middle right-beam patch (member 4, center 0.5, length fraction 0.18, modulus reduction 0.35)",
                    "records": example},
    }
    np.savez_compressed(folder / "observables.npz", **output)
    (folder / "observables.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for r in example:
        print(r["mode"], f"group={r['group_fraction']:.6f}", f"channel={r['channel_fraction']:.6f}",
              f"linear/exact RMS={r['linear_network_rms_uv']:.6f}/{r['exact_network_rms_uv']:.6f} uV")


if __name__ == "__main__":
    run()
