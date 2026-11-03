"""Export buildable acquisition networks, or refresh exports from retained arrays."""
import argparse
import json
from pathlib import Path

import numpy as np

from .bridge_network import Circuit
from .frame_response import asymmetric_frame
from .passive_bridge import TrimmedBridge
from .passive_resources import minimum_arm_units, routed_hardware, source_power


def gauge_geometry(data, layout):
    """Global coordinates and longitudinal unit directions in gauge-index order."""
    depth = data["depth"][layout]
    if data["name"] == "ring":
        config = json.loads(Path("inputs/installation_task.json").read_text())
        radius = config["nominal_mechanics"]["radius"]
        theta = data["theta"][layout]
        radial = np.column_stack((np.cos(theta), np.sin(theta)))
        center = radius * radial
        xy = (radius + depth[:, None]) * radial
        direction = np.column_stack((-np.sin(theta), np.cos(theta)))
        geometry = {"centerline_radius_m": radius, "instrument_xy_m": [radius, 0.],
                    "routing": "Shortest centerline arcs with radial depth leads; instrument at angle zero."}
        locations = [{"theta_rad": float(t)} for t in theta]
    else:
        frame = asymmetric_frame()
        member, fraction = data["member"][layout], data["fraction"][layout]
        endpoints = frame.joints[frame.members[member]]
        vector = endpoints[:, 1] - endpoints[:, 0]
        direction = vector / np.linalg.norm(vector, axis=-1, keepdims=True)
        center = endpoints[:, 0] + fraction[:, None] * vector
        xy = center + depth[:, None] * np.column_stack((-direction[:, 1], direction[:, 0]))
        geometry = {"joint_xy_m": frame.joints.tolist(), "member_nodes": frame.members.tolist(),
                    "instrument_xy_m": frame.joints[0].tolist(),
                    "routing": "Shortest connected-member centerline paths with normal depth leads; instrument at joint zero."}
        locations = [{"member_index": int(m), "longitudinal_fraction": float(f)}
                     for m, f in zip(member, fraction)]
    gauges = [location | {"gauge_index": i, "xy_m": point.tolist(),
              "centerline_xy_m": station.tolist(), "direction_xy": tangent.tolist(),
              "depth_m": float(z)} for i, (location, point, station, tangent, z)
              in enumerate(zip(locations, xy, center, direction, depth))]
    return geometry, gauges


def network_export(candidate, data, layout, factor_ratio, design_peaks, service_peaks):
    """Nominal settings; power arrays follow the concatenated bridge gauge order."""
    circuit = candidate["circuit"]
    device_config = json.loads(Path("inputs/device_node.json").read_text())
    geometry, locations = gauge_geometry(data, layout)
    zeros = np.zeros(len(locations))
    bridges, offset = [], 0
    for q, (device, route) in enumerate(zip(candidate["hardware"], candidate["routes"])):
        bridge = device.bridge
        m = bridge.arm_units
        rail = 2 * circuit.voltage * m * circuit.resistance / (
            m * circuit.resistance + 2 * device.supply_lead)
        gain = device.nominal_linear_gain()
        gauges = []
        for i, g in enumerate(bridge.gauge_indices):
            gauges.append(locations[g] | {
                "arm_index": int(bridge.arm[i]), "bypassed": bool(bridge.attenuation[i] == 0),
                "attenuation": float(bridge.attenuation[i]),
                "weight_effective_strain": float(bridge.weight[i]),
                "weight_physical_strain": float(bridge.weight[i] * factor_ratio[g]),
                "linear_voltage_per_strain_v": float(gain * bridge.weight[i] * factor_ratio[g]),
                "nominal_resistance_ohm": float(device.gauge_nominal[i]),
                "gauge_factor": float(circuit.gauge_factor * factor_ratio[g]),
                "shunt_ohm": float(device.shunt[i]) if np.isfinite(device.shunt[i]) else None,
                "design_peak_power_w": float(design_peaks[offset + i]),
                "service_peak_power_w": float(service_peaks[offset + i]),
            })
        offset += len(bridge.arm)
        bridges.append({
            "channel_index": q, "gauges": gauges, "arm_units": float(m),
            "minimum_arm_units": minimum_arm_units(bridge, circuit, route),
            "nominal_arm_resistance_ohm": device.arm_resistance(zeros).tolist(),
            "arm_ballast_ohm": device.ballast.tolist(),
            "nominal_bridge_rail_voltage_v": float(rail),
            "receiver_resistance_ohm": float(device.input_resistance),
            "head_xy_m": locations[route["head_gauge_centerline"]]["centerline_xy_m"],
            "route": route, "zero_strain_netlist": device.netlist(zeros),
        })
    name = data["name"]
    cases = ({"strain_file": "results/random-bridges/healthy_exchange.npz",
              "evaluation_keys": ["test_healthy_strain", "test_changed_strain"],
              "load_case_file": "results/fixed-installation/cases.json", "evaluation_split": "test",
              "design_file": "results/passive-bridges/synthesis.npz",
              "region_key": "design_sector_center_rad", "config": "inputs/passive_bridge_task.json"}
             if name == "ring" else
             {"strain_file": "results/frame-bridges/synthesis.npz",
              "evaluation_keys": ["r6_confirmation_healthy_strain", "r6_confirmation_changed_strain"],
              "evaluation_region_key": "confirmation_patch", "load_key": "r6_confirmation_coefficients",
              "design_file": "results/frame-bridges/synthesis.npz", "region_key": "design_patch",
              "config": "inputs/frame_bridge_task.json"})
    return {
        "structure": name, "layout": layout, "layout_seed": data["seeds"][layout],
        "index_base": 0, "arm_signs": [-1, 1, 1, -1],
        "weight_definition": "weight_effective_strain = arm_sign * attenuation; "
            "weight_physical_strain includes the gauge-factor ratio. Linear voltage coefficients "
            "refer to zero strain and include supply drop and receiver loading.",
        "component_convention": "Bypassed gauges have no cell or shunt; null shunt on an active gauge means bare gauge. "
            "Ballast includes all series compensation; nominal arm resistance includes arm wire. "
            "Netlist node names are local to each bridge except source rails H and L; output is A minus B.",
        "resistance_reference_ohm": circuit.resistance, "gauge_factor_reference": circuit.gauge_factor,
        "gauge_length_m": device_config["gauge_length_m"],
        "source_voltage_v": 2 * circuit.voltage,
        "rails_v": {"H": circuit.voltage, "L": -circuit.voltage},
        "nominal_source_power_w": float(source_power(candidate["hardware"], zeros)),
        "geometry": geometry, "case_sources": cases,
        "acquisition_config": "inputs/passive_readout_task.json", "bridges": bridges,
    }


def refresh(study):
    """Rebuild component delivery fields without repeating design or evaluation."""
    from .passive_robustness_study import source_data

    folder = Path("results/passive-network")
    path = folder / f"{study}_netlists.json"
    exports = json.loads(path.read_text())
    structures = {d["name"]: d for d in source_data()}
    with np.load(folder / f"{study}.npz") as arrays, np.load("results/passive-robustness/hardware.npz") as saved:
        for index, record in enumerate(exports):
            name, layout = record["structure"], record["layout"]
            prefix = f"{name}_l{layout}_{record['mode']}"
            n = len(arrays[prefix + "_arm_units"])
            ratio = saved[name + "_actual_nominal_ratio"][layout]
            network = [TrimmedBridge(g, a, c, float(m), ratio[g]) for g, a, c, m in zip(
                np.array_split(arrays[prefix + "_gauge_order"], n),
                np.array_split(arrays[prefix + "_arm"], n),
                np.array_split(arrays[prefix + "_attenuation"], n), arrays[prefix + "_arm_units"])]
            circuit = Circuit(gauge_factor=50., voltage=float(arrays[prefix + "_supply_voltage_v"]) / 2)
            routes = [b["route"] for b in record["bridges"]]
            config = json.loads(Path("inputs/passive_robustness_task.json").read_text())
            hardware = routed_hardware(network, circuit, routes, config["differential_input_resistance_ohm"])
            candidate = {"network": network, "hardware": hardware, "circuit": circuit, "routes": routes}
            exports[index] = record | network_export(candidate, structures[name], layout,
                saved[name + "_actual_gauge_factor_ratio"][layout],
                arrays[prefix + "_design_gauge_peak_power_w"], arrays[prefix + "_service_gauge_peak_power_w"])
    path.write_text(json.dumps(exports, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Exported {len(exports)} {study} networks to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("acquisition", "constrained"), required=True)
    refresh(parser.parse_args().study)
