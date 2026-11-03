from dataclasses import replace

import numpy as np
import pytest

from lining_port.bridge_network import Circuit
from lining_port.passive_bridge import TrimmedBridge
from lining_port.passive_hardware import BridgeHardware
from lining_port.passive_resources import (acquisition_variances, design_ratio,
                                          minimum_arm_units, matched_power_network, source_power)
from lining_port.passive_network import network_objective


def nodal_solution(rows, rails, injection=None):
    nodes = sorted({row[end] for row in rows for end in ("from", "to")} - rails.keys())
    index = {node: i for i, node in enumerate(nodes)}
    matrix, rhs = np.zeros((len(nodes), len(nodes))), np.zeros(len(nodes))
    for row in rows:
        g = 1 / row["resistance_ohm"]
        for a, b in ((row["from"], row["to"]), (row["to"], row["from"])):
            if a in rails:
                continue
            matrix[index[a], index[a]] += g
            if b in rails:
                rhs[index[a]] += g * rails[b]
            else:
                matrix[index[a], index[b]] -= g
    for node, current in (injection or {}).items():
        rhs[index[node]] += current
    return rails | dict(zip(nodes, np.linalg.solve(matrix, rhs)))


def example():
    bridge = TrimmedBridge.from_weight(np.arange(8), np.array([1., -.7, .6, -.5, .8, -.3, .4, -.9]))
    bridge = replace(bridge, attenuation=.6 * bridge.attenuation,
                     nominal_ratio=np.linspace(.99, 1.01, 8))
    route = {"arm_resistance_ohm": [3., 5., 4., 2.], "one_instrument_lead_ohm": 2.}
    circuit = Circuit(gauge_factor=50.)
    return bridge, route, circuit


def test_continuous_feasibility_at_zero_ballast_boundary():
    bridge, route, circuit = example()
    lower = minimum_arm_units(bridge, circuit, route)
    assert lower < np.bincount(bridge.arm).max()
    scaled = replace(bridge, arm_units=lower)
    hardware = BridgeHardware.from_design(scaled, circuit, route["arm_resistance_ohm"])
    assert np.min(hardware.ballast) == 0.
    assert all(row["resistance_ohm"] > 0 for row in hardware.netlist(np.zeros(8)))
    np.testing.assert_allclose(hardware.arm_resistance(np.zeros(8)), lower * circuit.resistance)
    with pytest.raises(ValueError, match="physical lead"):
        BridgeHardware.from_design(replace(bridge, arm_units=lower - 1e-6), circuit,
                                   route["arm_resistance_ohm"])
    with pytest.raises(ValueError, match="ballast"):
        replace(bridge, arm_units=lower / 2)


def test_power_and_receiver_potentials_match_component_nodal_solution():
    bridge, route, circuit = example()
    hardware = BridgeHardware.from_design(bridge, circuit, route["arm_resistance_ohm"], 2., 2., 3e6)
    strain = np.linspace(-.001, .002, 8)
    errors = {"shunt_error": np.linspace(-1e-4, 1e-4, 8),
              "ballast_error": np.array([.001, -.002, -.003, .004])}
    rows = hardware.netlist(strain, **errors)
    nodes = nodal_solution(rows, {"H": circuit.voltage, "L": -circuit.voltage})
    point = hardware.operating_point(strain, **errors)
    powers = np.array([(nodes[r["from"]] - nodes[r["to"]])**2 / r["resistance_ohm"] for r in rows])
    np.testing.assert_allclose(point["receiver_nodes_v"], [nodes["A"], nodes["B"]], atol=2e-12)
    np.testing.assert_allclose(point["voltage_v"], hardware.voltage(strain, **errors), atol=1e-16)
    np.testing.assert_allclose(point["source_power_w"], powers.sum(), rtol=2e-12)
    component_sum = sum(np.sum(value) for key, value in point.items()
                        if key.endswith("_power_w") and key != "source_power_w")
    np.testing.assert_allclose(component_sum, point["source_power_w"], rtol=3e-15)
    gauge_rows = [power for row, power in zip(rows, powers) if row["kind"] == "gauge"]
    ordered = np.concatenate([point["gauge_power_w"][bridge.arm == j] for j in range(4)])
    np.testing.assert_allclose(ordered, gauge_rows, rtol=2e-12)


@pytest.mark.parametrize("lead", [0., 2.])
def test_receiver_johnson_resistance_matches_current_injection(lead):
    bridge, route, circuit = example()
    hardware = BridgeHardware.from_design(bridge, circuit, route["arm_resistance_ohm"], lead, lead, 3e6)
    strain = np.linspace(-.001, .002, 8)
    nodes = nodal_solution(hardware.netlist(strain), {"H": 0., "L": 0.}, {"A": 1., "B": -1.})
    np.testing.assert_allclose(hardware.output_resistance(strain), nodes["A"] - nodes["B"], rtol=2e-12)


def test_common_supply_matches_total_power_and_balanced_closed_form():
    bridge, route, circuit = example()
    network = [bridge, replace(bridge, gauge_indices=bridge.gauge_indices + 8, arm_units=3.)]
    route = {"arm_resistance_ohm": [0.] * 4, "one_instrument_lead_ohm": 0.}
    zeros = np.zeros(16)
    power = sum((2 * circuit.voltage)**2 / (b.arm_units * circuit.resistance) for b in network)
    units = [1.5, 1.7]
    _, supply, hardware = matched_power_network(network, units, circuit, [route, route], np.inf, power)
    np.testing.assert_allclose(source_power(hardware, zeros), power, rtol=3e-16)
    np.testing.assert_allclose(supply.voltage / circuit.voltage,
                              np.sqrt(sum(1 / b.arm_units for b in network) / sum(1 / m for m in units)))


def test_routed_coefficient_gradient_matches_reset_physical_components():
    bridge, route, circuit = example()
    hardware = BridgeHardware.from_design(bridge, circuit, route["arm_resistance_ohm"], 900., 700., 3e5)
    strain = np.array([np.linspace(-.001, .002, 8), np.linspace(.002, -.0005, 8)])
    voltage, gradient = hardware.voltage_and_gradient(strain)
    np.testing.assert_allclose(voltage, hardware.voltage(strain), atol=4e-16)
    for i in range(len(bridge.attenuation)):
        step = np.eye(len(bridge.attenuation))[i] * 1e-5
        values = []
        for direction in (1, -1):
            trial = replace(bridge, attenuation=bridge.attenuation + direction * step)
            physical = BridgeHardware.from_design(trial, circuit, route["arm_resistance_ohm"], 900., 700., 3e5)
            values.append(physical.voltage(strain))
        np.testing.assert_allclose(gradient[:, i], (values[0] - values[1]) / 2e-5,
                                   rtol=2e-7, atol=2e-11)


def test_loaded_reference_objective_and_power_use_the_same_hardware():
    bridge, route, circuit = example()
    network = [bridge, replace(bridge, gauge_indices=bridge.gauge_indices + 8)]
    routes = [route, route | {"one_instrument_lead_ohm": 800.}]
    power = 3e-5
    _, _, hardware = matched_power_network(network, [1.8, 1.9], circuit, routes, 3e6, power)
    np.testing.assert_allclose(source_power(hardware, np.zeros(16)), power, rtol=5e-16)
    rng = np.random.default_rng(12)
    healthy = rng.normal(size=(7, 16)) * 2e-4
    changed = healthy + rng.normal(size=(3, 7, 16)) * 1e-5
    config = {"average_time_s": 4.}
    acquisition = {"reference_calibration_index": 3, "reference_average_time_s": 2.,
                   "temperature_k": 300., "independent_sample_sigma_v": 1e-6, "sample_period_s": .005}
    ratio, _, y = design_ratio(hardware, healthy, changed, config, acquisition)
    objective = network_objective(hardware, healthy, changed,
        sum(acquisition_variances(hardware, config, acquisition)), reference_index=3)
    coefficients = np.concatenate([h.bridge.attenuation for h in hardware])
    np.testing.assert_allclose(objective(coefficients)[0], -ratio, atol=2e-11)
    np.testing.assert_allclose(y[3], 0., atol=0.)
    for i, device in enumerate(hardware):
        np.testing.assert_allclose(y[:, i], device.voltage(healthy) - device.voltage(healthy[3]), atol=3e-16)


def test_constructive_gain_includes_supply_and_receiver_loading():
    bridge, route, circuit = example()
    supply_lead, sense_lead, receiver = 900., 700., 3e5
    hardware = BridgeHardware.from_design(bridge, circuit, route["arm_resistance_ohm"],
                                          supply_lead, sense_lead, receiver)
    direction = np.linspace(-1., 1., 8)
    nominal_arm = bridge.arm_units * circuit.resistance
    rail = 2 * circuit.voltage * nominal_arm / (nominal_arm + 2 * supply_lead)
    gain = rail * circuit.gauge_factor / (4 * bridge.arm_units)
    gain *= receiver / (receiver + nominal_arm + 2 * sense_lead)
    expected = gain * (bridge.weight @ direction)
    step = 1e-7
    numeric = (hardware.voltage(step * direction) - hardware.voltage(-step * direction)) / (2 * step)
    np.testing.assert_allclose(numeric, expected, rtol=1e-7)
