from dataclasses import replace
import json

import numpy as np

from lining_port.bridge_network import Circuit
from lining_port.frame_response import random_installation
from lining_port.passive_acquisition_study import routed_candidate
from lining_port.passive_bridge import TrimmedBridge
from lining_port.passive_export import gauge_geometry, network_export
from lining_port.passive_robustness_study import distances
from test_passive_bridge import solve_netlist


def test_export_reconstructs_loaded_voltage_and_physical_strain_gain():
    rng = np.random.default_rng(41)
    positions = random_installation(51)
    data = {"name": "frame", "seeds": [51], "depth": positions["depth_m"][None],
            "member": positions["member"][None], "fraction": positions["fraction"][None]}
    weights = rng.normal(size=60)
    weights[0] = 0.
    ratios, factors = rng.uniform(.99, 1.01, (2, 60))
    network = [replace(TrimmedBridge.from_weight(g, weights[g]),
                       attenuation=.8 * abs(weights[g]) / max(abs(weights[g])),
                       nominal_ratio=ratios[g]) for g in np.array_split(np.arange(60), 4)]
    circuit = Circuit(gauge_factor=50.)
    candidate = routed_candidate(network, circuit, distances(data, 0), data["depth"][0],
        {"wire_resistance_ohm_per_m": .0842, "differential_input_resistance_ohm": 1e9})
    strain = rng.normal(0., 2e-4, 60)
    peaks = np.concatenate([h.operating_point(strain * factors)["gauge_power_w"] for h in candidate["hardware"]])
    export = network_export(candidate, data, 0, factors, peaks, peaks)
    export = json.loads(json.dumps(export, allow_nan=False))
    for q, bridge in enumerate(export["bridges"]):
        gauges = {g["gauge_index"]: g for g in bridge["gauges"]}
        rows = []
        for element in bridge["zero_strain_netlist"]:
            row = element.copy()
            if row["kind"] == "gauge":
                g = gauges[row["gauge_index"]]
                row["resistance_ohm"] *= 1 + g["gauge_factor"] * strain[g["gauge_index"]]
            rows.append(row)
        actual = solve_netlist(rows, export["rails_v"]["H"])
        expected = candidate["hardware"][q].voltage(strain * factors)
        np.testing.assert_allclose(actual, expected, atol=3e-11)
        linear = sum(g["linear_voltage_per_strain_v"] * strain[g["gauge_index"]] for g in gauges.values())
        h = candidate["hardware"][q]
        derivative = (h.voltage(1e-3 * strain * factors) - h.voltage(-1e-3 * strain * factors)) / .002
        np.testing.assert_allclose(linear, derivative, atol=5e-13)
        for g in gauges.values():
            np.testing.assert_allclose(np.linalg.norm(g["direction_xy"]), 1.)
            np.testing.assert_allclose(np.linalg.norm(np.array(g["xy_m"]) - g["centerline_xy_m"]), abs(g["depth_m"]))
        assert set(bridge["route"]["arm_gauge_order"][0]) == {
            i for i, g in gauges.items() if g["arm_index"] == 0 and not g["bypassed"]}
    assert export["bridges"][0]["gauges"][0]["bypassed"]
    assert export["bridges"][0]["gauges"][0]["shunt_ohm"] is None


def test_ring_export_uses_radial_depth_and_circumferential_direction():
    theta = np.array([[0., np.pi / 2, np.pi]])
    depth = np.array([[.1, -.1, .05]])
    geometry, gauges = gauge_geometry({"name": "ring", "theta": theta, "depth": depth}, 0)
    radius = geometry["centerline_radius_m"]
    for g in gauges:
        xy, direction = np.array(g["xy_m"]), np.array(g["direction_xy"])
        np.testing.assert_allclose(np.linalg.norm(xy), radius + g["depth_m"])
        np.testing.assert_allclose(xy @ direction, 0., atol=1e-15)
