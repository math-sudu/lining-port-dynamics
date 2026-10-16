"""Physical conductor routes from the 64 gauges to four module-head interfaces."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .acquisition_comparison import bridge_netlists, bridge_voltage
from .bridge_network import Circuit
from .installation import installed_network
from .mechanism_study import dump


SCOPE = (
    "Unshielded conductor length from the installed gauges to the four module-head "
    "supply/receiver interfaces at the inner surface, including all inter-module "
    "connections, series wires, shared buses, radial feeds and two leads per "
    "differential input. External power/data trunks beyond those interfaces and "
    "receiver/controller power are outside this boundary."
)


def angle_delta(start, stop):
    return float((stop - start + np.pi) % (2 * np.pi) - np.pi)


def conductor(start, stop, radius, role, **identity):
    """Route along one radius, with radial legs at its ends; angles are unwrapped."""
    theta, r = start
    end_theta = theta + angle_delta(theta, stop[0])
    points = [[theta, r], [theta, radius], [end_theta, radius], [end_theta, stop[1]]]
    points = [p for i, p in enumerate(points) if i == 0 or not np.allclose(p, points[i - 1], atol=1e-14, rtol=0)]
    length = abs(r - radius) + radius * abs(end_theta - theta) + abs(stop[1] - radius)
    return {"role": role, "path_theta_rad_radius_m": points, "length_m": float(length), **identity}


def geometry_points(geom):
    radius = geom["radius"]
    gauges = list(zip(geom["theta"], radius + geom["z"]))
    heads = [(m * np.pi / 2, radius - geom["thickness"] / 2) for m in range(4)]
    return gauges, heads


def arm_order(ids, geom):
    """Within each half-module, traverse from the outer angular end to its head."""
    ordered = []
    # Preserve the netlist's module order for a two-module series arm.
    for module in dict.fromkeys(int(geom["module"][i]) for i in ids):
        group = [int(i) for i in ids if geom["module"][i] == module]
        group.sort(key=lambda i: abs(angle_delta(module * np.pi / 2, geom["theta"][i])), reverse=True)
        ordered.extend(group)
    return ordered


def bridge_layout(name, nodes, geom):
    """Two series arms meet at a mid-depth tap; each supply has a shared radial feed.

    R1/R3 run from the positive bus to a midpoint; R2/R4 run from the negative
    bus to that same midpoint. These rail-to-midpoint orders specify both gauge
    terminals and every conductor independently of the algebraic netlist order.
    """
    radius = geom["radius"]
    gauges, heads = geometry_points(geom)
    receiver_heads = {"mechanical_bridge": [0, 1, 2, 3],
                      "composite4": [0, 2, 1, 3], "composite2": [2, 3]}[name]
    routes, arms, midpoints, used_heads = [], [], {}, set()
    for q, bridge in enumerate(nodes):
        for a, ids in enumerate(bridge):
            order = arm_order(ids, geom)
            source = int(geom["module"][order[0]])
            used_heads.add(source)
            rail = "+" if a % 2 == 0 else "-"
            midpoint = (gauges[order[-1]][0], radius)
            midpoints[q, a // 2] = midpoint
            identity = {"bridge": q, "arm": a}
            wires = [conductor((heads[source][0], radius), gauges[order[0]], radius,
                               "excitation", source_head=source, rail=rail, **identity)]
            for i, j in zip(order[:-1], order[1:]):
                cross = bool(geom["module"][i] != geom["module"][j])
                route_radius = radius if cross else gauges[i][1]
                wires.append(conductor(gauges[i], gauges[j], route_radius, "series",
                                       from_gauge=i, to_gauge=j, inter_module=cross, **identity))
            wires.append(conductor(gauges[order[-1]], midpoint, radius, "series",
                                   from_gauge=order[-1], midpoint=a // 2, **identity))
            routes.extend(wires)
            arms.append({**identity, "source_head": source, "rail": rail,
                         "gauges_rail_to_midpoint": order,
                         "wire_length_m": sum(w["length_m"] for w in wires)})
        for side in range(2):
            midpoint = midpoints[q, side]
            receiver = receiver_heads[q]
            routes.append(conductor(midpoint, heads[receiver], radius, "readout", bridge=q,
                                    midpoint=side, receiver_head=receiver,
                                    inter_module=bool(abs(angle_delta(midpoint[0], heads[receiver][0])) > np.pi / 4)))
    for module in sorted(used_heads):
        for rail in ("+", "-"):
            routes.append(conductor(heads[module], (heads[module][0], radius), radius,
                                    "excitation", source_head=module, rail=rail, shared_feed=True))
    return {"routes": routes, "arms": arms, "receiver_heads": receiver_heads,
            "source_heads": sorted(used_heads), "shared_feed_length_m": geom["thickness"] / 2}


def distributed_bus(geom, rail):
    """One branched bus per module, including its shared inner-surface feed."""
    radius = geom["radius"]
    gauges, heads = geometry_points(geom)
    routes = []
    for module, head in enumerate(heads):
        theta = head[0]
        routes.append(conductor(head, (theta, radius), radius, "excitation",
                                source_head=module, rail=rail, shared_feed=True))
        for sign in (-1, 1):
            start = (theta, radius)
            for offset in np.arange(.5, 4.) * np.pi / 16:
                stop = (theta + sign * offset, radius)
                routes.append(conductor(start, stop, radius, "excitation", source_head=module, rail=rail))
                start = stop
        for i in np.flatnonzero(geom["module"] == module):
            routes.append(conductor((gauges[i][0], radius), gauges[i], radius, "excitation",
                                    to_gauge=int(i), rail=rail))
    return routes


def routed_layouts(geom):
    gauges, heads = geometry_points(geom)
    radius = geom["radius"]
    layouts = {name: bridge_layout(name, nodes, geom) for name, nodes in bridge_netlists().items()}
    rc = [wire for rail in ("+", "-", "ground") for wire in distributed_bus(geom, rail)]
    for i, j in zip(*np.nonzero(np.triu(geom["W"], 1))):
        rc.append(conductor(gauges[i], gauges[j], gauges[i][1], "series", from_gauge=int(i), to_gauge=int(j)))
    for module, i in enumerate(geom["port_nodes"]):
        rc.append(conductor(gauges[i], heads[module], gauges[i][1], "readout", receiver_head=module))
    scan = distributed_bus(geom, "+")
    for i, point in enumerate(gauges):
        module = int(geom["module"][i])
        scan.append(conductor(point, heads[module], point[1], "readout", from_gauge=i, receiver_head=module))
    layouts.update(rc={"routes": rc}, scan={"routes": scan})
    return layouts


def route_totals(layout):
    parts = {role + "_route_m": float(sum(w["length_m"] for w in layout["routes"] if w["role"] == role))
             for role in ("series", "excitation", "readout")}
    return {**parts, "signal_route_m": parts["series_route_m"] + parts["readout_route_m"],
            "local_conductor_m": sum(parts.values()),
            "inter_module_conductor_m": float(sum(w["length_m"] for w in layout["routes"] if w.get("inter_module")))}


def routed_bridge_response(strain, layout, circuit, resistance_per_m):
    """Exact DC solution, including shared supply feeds and series wire resistance.

    Midpoint leads terminate in ideal high-impedance differential receivers.
    The two source feeds have equal resistance. Each source head supplies its
    two divider branches in parallel, so their common bus drop is solved first.
    """
    if resistance_per_m < 0:
        raise ValueError("Conductor resistance must be nonnegative")
    r = circuit.resistance * (1 + circuit.gauge_factor * np.asarray(strain))
    if np.any(r <= 0):
        raise ValueError("Gauge resistance must be positive")
    channels = len(layout["receiver_heads"])
    arms = np.empty(r.shape[:-1] + (channels, 4))
    source = np.empty((channels, 2), dtype=int)
    for arm in layout["arms"]:
        q, a = arm["bridge"], arm["arm"]
        arms[..., q, a] = r[..., arm["gauges_rail_to_midpoint"]].sum(axis=-1) + resistance_per_m * arm["wire_length_m"]
        source[q, a // 2] = arm["source_head"]
    resistance = arms.reshape(r.shape[:-1] + (channels, 2, 2))
    branch_r = resistance.sum(axis=-1)
    midpoint = np.empty_like(branch_r)
    power = np.zeros(r.shape[:-1])
    feed_r = resistance_per_m * layout["shared_feed_length_m"]
    for module in layout["source_heads"]:
        selected = source == module
        conductance = (1 / branch_r[..., selected]).sum(axis=-1)
        span = 2 * circuit.voltage / (1 + 2 * feed_r * conductance)
        midpoint[..., selected] = span[..., None] * (
            (resistance[..., 1] - resistance[..., 0]) / (2 * branch_r))[..., selected]
        power += 2 * circuit.voltage * span * conductance
    return midpoint[..., 0] - midpoint[..., 1], power


def main():
    from .acquisition_comparison import resources

    config = json.loads(Path("inputs/installation_task.json").read_text())
    device = json.loads(Path("inputs/device_node.json").read_text())
    circuit = Circuit(**device["circuit"])
    _, geom = installed_network(config["layout_seeds"][0], config["nominal_mechanics"]["radius"],
                                config["nominal_mechanics"]["thickness"], config["depth_m"])
    layouts = routed_layouts(geom)
    result = {"scope": SCOPE, "coordinates": "polar [theta_rad, radius_m]; all IDs zero-based",
              "layouts": layouts, "resources": resources(config, geom, circuit)}
    strains = np.load("results/fixed-installation/strains.npz")
    healthy = np.concatenate((strains["calibration_healthy"], strains["test_healthy"]))
    coefficient = .0787
    result["lead_resistance"] = {
        "ohm_per_m": coefficient, "source": "Belden 9501NH: nominal conductor DCR 78.7 ohm/km",
        "source_url": "https://www.belden.com/products/cpr-euroclass/cpr-copper/9501nh",
        "model": "Uniform conductor resistance, shared supply feeds, ideal high-impedance midpoint receivers",
        "candidates": {},
    }
    for name, nodes in bridge_netlists().items():
        layout = layouts[name]
        h, _ = routed_bridge_response(healthy, layout, circuit, coefficient)
        s, _ = routed_bridge_response(strains["test_soft"], layout, circuit, coefficient)
        ideal_change = (bridge_voltage(strains["test_soft"], nodes, circuit)
                        - bridge_voltage(strains["test_healthy"], nodes, circuit)[:, None, :])
        change = s - h[config["calibration_cases"]:, None, :]
        _, nominal_power = routed_bridge_response(np.zeros(64), layout, circuit, coefficient)
        result["lead_resistance"]["candidates"][name] = {
            "healthy_peak_v": float(abs(h).max()),
            "target_change_max_absolute_difference_v": float(abs(change - ideal_change).max()),
            "target_change_max_relative_vector_error": float(np.max(
                np.linalg.norm(change - ideal_change, axis=-1) / np.linalg.norm(ideal_change, axis=-1))),
            "zero_strain_excitation_energy_j": float(nominal_power * 4),
        }
    output = Path("results/composite-bridges/routing.json")
    dump(output, result)
    acquisition_path = Path("results/fixed-installation/acquisition.json")
    acquisition = json.loads(acquisition_path.read_text())
    acquisition["resources"] = result["resources"]
    dump(acquisition_path, acquisition)
    print(json.dumps({"resources": result["resources"], "lead_resistance": result["lead_resistance"]}, indent=2))


if __name__ == "__main__":
    main()
