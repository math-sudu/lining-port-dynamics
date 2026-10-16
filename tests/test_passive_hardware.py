from dataclasses import replace

import numpy as np

from lining_port.bridge_network import Circuit
from lining_port.passive_bridge import TrimmedBridge
from lining_port.passive_hardware import BridgeHardware, route_bridge
from test_passive_bridge import solve_netlist


def test_component_errors_leads_and_receiver_match_independent_kcl():
    rng = np.random.default_rng(72)
    circuit = Circuit(gauge_factor=50.)
    bridge = TrimmedBridge.from_weight(np.arange(16), rng.normal(size=16))
    bridge = replace(bridge, attenuation=.95 * bridge.attenuation, nominal_ratio=rng.uniform(.99, 1.01, 16))
    for leads, receiver in ((0., np.inf), (3., 8e6)):
        hardware = BridgeHardware.from_design(bridge, circuit, np.full(4, leads), leads, leads, receiver)
        shunt_error, ballast_error = rng.uniform(-.001, .001, 16), rng.uniform(-.001, .001, 4)
        for strain in rng.normal(0., 2e-4, (3, 16)):
            expected = solve_netlist(hardware.netlist(strain, shunt_error, ballast_error), circuit.voltage)
            actual = hardware.voltage(strain, shunt_error=shunt_error, ballast_error=ballast_error)
            np.testing.assert_allclose(actual, expected, atol=3e-11)
        if not leads:
            np.testing.assert_allclose(hardware.voltage(strain), bridge.voltage(strain, circuit), atol=4e-16)


def test_drift_is_not_removed_by_a_fixed_zero_reference():
    bridge = TrimmedBridge.from_weight(np.arange(8), np.array([1., -.7, .6, -.5, .8, -.3, .4, -.9]))
    hardware = BridgeHardware.from_design(bridge, Circuit(gauge_factor=50.))
    zero = np.zeros(8)
    drift = np.zeros(8)
    drift[0] = 1e-5
    assert abs(hardware.voltage(zero, gauge_drift=drift) - hardware.voltage(zero)) > 1e-7


def test_routing_accounts_for_each_arm_and_four_instrument_leads():
    bridge = TrimmedBridge.from_weight(np.arange(8), np.array([1., -.7, .6, -.5, .8, -.3, .4, -.9]))
    x = np.arange(8.)
    distance = abs(x[:, None] - x)
    route = route_bridge(bridge, distance, distance, x + 2, np.zeros(8), .1)
    np.testing.assert_allclose(route["arm_length_m"], [10., 8., 12., 14.])
    assert route["total_conductor_m"] == 52.
