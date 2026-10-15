"""Fixed two-depth ring installation and its routed four-port ladder network."""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components

from .network import make_layout


def installed_network(seed, radius=3., thickness=.3, depth=.1):
    """Eight radial gauge pairs per quadrant; one central inner node per port.

    Adjacent stations connect along each embedded layer; radial rungs join pairs.
    All 88 links use the same resistor. Wire routes follow arcs and radial lines.
    The seed selects capacitors only, retaining the original inventory draws.
    """
    if not 0 < depth < thickness / 2 < radius:
        raise ValueError("Embedded depths must lie inside the ring thickness")
    module = np.repeat(np.arange(4), 16)
    station = np.repeat(np.arange(8), 2)
    offsets = (station - 3.5) * np.pi / 16
    theta = (module * np.pi / 2 + np.tile(offsets, 4)) % (2 * np.pi)
    z = np.tile([-depth, depth], 32)
    radial = radius + z
    xy = radial[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
    weights = np.zeros((64, 64))
    lengths = np.zeros_like(weights)
    for start in range(0, 64, 16):
        for pair in range(8):
            i = start + 2 * pair
            weights[i, i + 1] = weights[i + 1, i] = 1
            lengths[i, i + 1] = lengths[i + 1, i] = 2 * depth
            if pair < 7:
                for j in (i, i + 1):
                    weights[j, j + 2] = weights[j + 2, j] = 1
                    lengths[j, j + 2] = lengths[j + 2, j] = radial[j] * np.pi / 16
    ports = np.arange(4) * 16 + 6
    port_theta = np.arange(4) * np.pi / 2
    port_xy = (radius - thickness / 2) * np.column_stack((np.cos(port_theta), np.sin(port_theta)))
    h = np.eye(64)[ports]
    count, labels = connected_components(weights, directed=False)
    reachable = np.isin(labels, labels[ports])
    degree = np.count_nonzero(weights, axis=1)
    layout = make_layout(seed)
    layout["theta"] = theta.copy()
    layout["area_quantile"] = (radial**2 - (radius - thickness / 2)**2) / (2 * radius * thickness)
    return layout, {
        "theta": theta, "z": z, "xy": xy, "W": weights, "H": h,
        "mask": np.ones(4, dtype=bool), "port_nodes": ports, "port_xy": port_xy,
        "module": module, "component_labels": labels, "reachable": reachable,
        "route_length": lengths,
        "port_lead_length": np.full(4, (radius - depth) * np.pi / 32 + thickness / 2 - depth),
        "radius": radius, "thickness": thickness,
        "diagnostics": {"nodes": 64, "edges": int(np.count_nonzero(weights) // 2),
                        "components": int(count), "component_sizes": np.bincount(labels).tolist(),
                        "valid_ports": 4, "reachable_node_fraction": float(reachable.mean()),
                        "degree_min": int(degree.min()), "degree_max": int(degree.max()),
                        "link_route_length_m": float(lengths.sum() / 2)},
    }
