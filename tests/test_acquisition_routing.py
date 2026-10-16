import numpy as np

from lining_port.acquisition_comparison import bridge_netlists, bridge_voltage
from lining_port.acquisition_routing import routed_bridge_response, routed_layouts, route_totals
from lining_port.bridge_network import Circuit
from lining_port.installation import installed_network


def test_routes_connect_every_gauge_and_both_differential_inputs():
    _, geom = installed_network(7100)
    for name, layout in routed_layouts(geom).items():
        for wire in layout["routes"]:
            points = np.asarray(wire["path_theta_rad_radius_m"])
            dtheta, dr = np.diff(points, axis=0).T
            assert np.all((abs(dtheta) < 1e-14) | (abs(dr) < 1e-14))
            distance = np.sum(abs(dr) + abs(dtheta) * points[:-1, 1])
            np.testing.assert_allclose(distance, wire["length_m"], atol=1e-13)
        if name in ("rc", "scan"):
            continue
        used = [g for arm in layout["arms"] for g in arm["gauges_rail_to_midpoint"]]
        np.testing.assert_array_equal(sorted(used), np.arange(64))
        for q in range(len(layout["receiver_heads"])):
            for side in range(2):
                ends = [w["path_theta_rad_radius_m"][-1] for w in layout["routes"]
                        if w.get("bridge") == q and w.get("midpoint") == side and w["role"] == "series"]
                lead = next(w for w in layout["routes"] if w["role"] == "readout"
                            and w["bridge"] == q and w["midpoint"] == side)
                assert len(ends) == 2
                for endpoint in ends:
                    np.testing.assert_allclose(np.cos(endpoint[0]), np.cos(lead["path_theta_rad_radius_m"][0][0]))
                    np.testing.assert_allclose(np.sin(endpoint[0]), np.sin(lead["path_theta_rad_radius_m"][0][0]))
                    assert endpoint[1] == lead["path_theta_rad_radius_m"][0][1]


def test_shared_feeds_and_cross_module_connections_are_counted_once():
    _, geom = installed_network(7100)
    layouts = routed_layouts(geom)
    expected_feeds = {"rc": 12, "scan": 4, "mechanical_bridge": 8, "composite4": 8, "composite2": 4}
    for name, layout in layouts.items():
        feeds = [wire for wire in layout["routes"] if wire.get("shared_feed")]
        assert len(feeds) == expected_feeds[name]
        np.testing.assert_allclose(sum(w["length_m"] for w in feeds), .15 * expected_feeds[name])
    # Four antipodal midpoint returns versus eight cross-module arm connections.
    assert sum(bool(w.get("inter_module")) for w in layouts["composite4"]["routes"]) == 4
    assert sum(bool(w.get("inter_module")) for w in layouts["composite2"]["routes"]) == 8
    # The old signal-only lengths are unaffected by adding the supply feeds.
    np.testing.assert_allclose(route_totals(layouts["rc"])["signal_route_m"],
                               geom["route_length"].sum() / 2 + geom["port_lead_length"].sum())
    np.testing.assert_allclose(route_totals(layouts["scan"])["readout_route_m"], 24 * np.pi + 9.6)


def test_dc_solution_matches_ideal_bridges_and_preserves_healthy_balance_with_wires():
    _, geom = installed_network(7100)
    circuit = Circuit(gauge_factor=50)
    layouts = routed_layouts(geom)
    rng = np.random.default_rng(20261016)
    strain = rng.normal(0, .001, (2, 3, 64))
    half = rng.normal(0, .001, (5, 32))
    healthy = np.concatenate((half, half), axis=-1)
    for name, nodes in bridge_netlists().items():
        voltage, power = routed_bridge_response(strain, layouts[name], circuit, 0)
        np.testing.assert_allclose(voltage, bridge_voltage(strain, nodes, circuit), atol=5e-16)
        arms = circuit.resistance * (1 + circuit.gauge_factor * strain[..., nodes]).sum(axis=-1)
        expected_power = 4 * circuit.voltage**2 * np.sum(
            1 / (arms[..., 0] + arms[..., 1]) + 1 / (arms[..., 2] + arms[..., 3]), axis=-1)
        np.testing.assert_allclose(power, expected_power)
        if name != "mechanical_bridge":
            h, _ = routed_bridge_response(healthy, layouts[name], circuit, .0787)
            np.testing.assert_allclose(h, 0, atol=2e-16)
            changed = healthy.copy()
            changed[:, 0] += .0002
            v, _ = routed_bridge_response(changed, layouts[name], circuit, .0787)
            assert np.all(v[:, 0] < 0)


def test_shared_supply_drop_uses_both_divider_currents():
    _, geom = installed_network(7100)
    layout = routed_layouts(geom)["mechanical_bridge"]
    circuit = Circuit(gauge_factor=50)
    resistance_per_m = 1e4  # Large enough to resolve the shared-feed effect.
    strain = np.linspace(-.001, .002, 64)
    voltage, power = routed_bridge_response(strain, layout, circuit, resistance_per_m)
    expected, expected_power = [], 0.
    for module in range(4):
        arms = [arm for arm in layout["arms"] if arm["bridge"] == module]
        r = np.array([sum(circuit.resistance * (1 + circuit.gauge_factor * strain[i])
                          for i in arm["gauges_rail_to_midpoint"])
                      + resistance_per_m * arm["wire_length_m"] for arm in arms])
        # Independent four-node Kirchhoff system: two buses and two midpoints.
        g, gf = 1 / r, 1 / (resistance_per_m * .15)
        matrix = np.array([[gf + g[0] + g[2], 0, -g[0], -g[2]],
                           [0, gf + g[1] + g[3], -g[1], -g[3]],
                           [-g[0], -g[1], g[0] + g[1], 0],
                           [-g[2], -g[3], 0, g[2] + g[3]]])
        bus_p, bus_n, left, right = np.linalg.solve(matrix, [gf * circuit.voltage, -gf * circuit.voltage, 0, 0])
        expected.append(left - right)
        expected_power += circuit.voltage * gf * (2 * circuit.voltage - bus_p + bus_n)
    np.testing.assert_allclose(voltage, expected, atol=1e-16)
    np.testing.assert_allclose(power, expected_power)
