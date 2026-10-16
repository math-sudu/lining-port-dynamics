import numpy as np
import pytest
from dataclasses import replace

from lining_port.acquisition_comparison import bridge_voltage
from lining_port.bridge_network import Circuit
from lining_port.passive_bridge import (TrimmedBridge, optimize_exact_bridge,
                                        response_spaces, target_null_weight)


def solve_netlist(rows, voltage):
    """Independent nodal KCL solve using individual physical resistor values."""
    nodes = sorted({row[end] for row in rows for end in ("from", "to")} - {"H", "L"})
    index = {name: i for i, name in enumerate(nodes)}
    matrix, drive = np.zeros((len(nodes), len(nodes))), np.zeros(len(nodes))
    rails = {"H": voltage, "L": -voltage}
    for row in rows:
        conductance = 1 / row["resistance_ohm"]
        for first, second in ((row["from"], row["to"]), (row["to"], row["from"])):
            if first in rails:
                continue
            i = index[first]
            matrix[i, i] += conductance
            if second in rails:
                drive[i] += conductance * rails[second]
            else:
                matrix[i, index[second]] -= conductance
    potential = np.linalg.solve(matrix, drive)
    return potential[index["A"]] - potential[index["B"]]


def test_exact_bridge_matches_component_kcl_with_shunts_and_unused_slots():
    rng = np.random.default_rng(5)
    # Unequal signed counts require arm ballast; zero c bypasses one gauge.
    weight = np.array([1., -.4, .7, 0., .2, .31, .5, -.83, .45])
    bridge = TrimmedBridge.from_weight(np.arange(len(weight)), weight)
    circuit = Circuit(gauge_factor=50.)
    for strain in rng.uniform(-.005, .005, (6, len(weight))):
        rows = bridge.resistor_netlist(strain, circuit)
        assert all(row["resistance_ohm"] > 0 for row in rows)
        active = [row["gauge_index"] for row in rows if row["kind"] == "gauge"]
        assert len(active) == len(set(active)) == len(weight) - 1
        np.testing.assert_allclose(bridge.voltage(strain, circuit), solve_netlist(rows, circuit.voltage), atol=2e-15)
    np.testing.assert_allclose(bridge.arm_resistance(np.zeros(len(weight)), circuit),
                               bridge.arm_units * circuit.resistance)


def test_bare_cells_recover_existing_full_bridge():
    rng = np.random.default_rng(6)
    nodes = np.arange(20).reshape(1, 4, 5)
    bridge = TrimmedBridge(np.arange(20), np.repeat(np.arange(4), 5), np.ones(20), 5)
    strain = rng.normal(0., 2e-4, (12, 20))
    circuit = Circuit(gauge_factor=50.)
    np.testing.assert_allclose(bridge.voltage(strain, circuit), bridge_voltage(strain, nodes, circuit)[:, 0], atol=3e-16)


def test_measured_unequal_gauge_resistance_and_finite_shunt_cap():
    rng = np.random.default_rng(61)
    circuit = Circuit(gauge_factor=50.)
    bridge = TrimmedBridge.from_weight(np.arange(16), rng.normal(size=16))
    ratio = rng.uniform(.99, 1.01, 16)
    bridge = replace(bridge, attenuation=bridge.attenuation * .95, nominal_ratio=ratio)
    np.testing.assert_allclose(bridge.arm_resistance(np.zeros(16), circuit), bridge.arm_units * circuit.resistance)
    strain = rng.normal(0., 2e-4, 16)
    np.testing.assert_allclose(bridge.voltage(strain, circuit), solve_netlist(bridge.resistor_netlist(strain, circuit), circuit.voltage), atol=2e-15)
    _, gradient = bridge.voltage_and_gradient(strain, circuit)
    for i in range(16):
        change = np.eye(16)[i] * 1e-5
        numeric = (bridge.voltage(strain, circuit, bridge.attenuation + change) - bridge.voltage(strain, circuit, bridge.attenuation - change)) / 2e-5
        np.testing.assert_allclose(gradient[i], numeric, rtol=2e-6, atol=2e-11)
    assert abs(bridge.voltage(strain, circuit) - bridge.linear_voltage(strain, circuit)) <= bridge.nonlinear_bound(strain, circuit)


def test_attenuation_gradient_and_finite_strain_bound():
    rng = np.random.default_rng(11)
    bridge = TrimmedBridge.from_weight(np.arange(13), rng.uniform(-1., 1., 13))
    circuit = Circuit(gauge_factor=50.)
    strain = rng.normal(0., .002, (17, 13))
    voltage, derivative = bridge.voltage_and_gradient(strain, circuit)
    for i, c in enumerate(bridge.attenuation):
        # Keep finite differences strictly inside the physically feasible box.
        if c > .999:
            continue
        perturbation = np.zeros(13)
        perturbation[i] = 1e-5
        numerical = (bridge.voltage(strain, circuit, bridge.attenuation + perturbation)
                     - bridge.voltage(strain, circuit, bridge.attenuation - perturbation)) / 2e-5
        np.testing.assert_allclose(derivative[:, i], numerical, rtol=2e-6, atol=3e-11)
    error = np.abs(voltage - bridge.linear_voltage(strain, circuit))
    assert np.all(error <= bridge.nonlinear_bound(strain, circuit) + 2e-15)


def test_general_response_separability_and_load_basis_invariance():
    rng = np.random.default_rng(17)
    basis, target = rng.normal(size=(17, 5)), rng.normal(size=(17, 8))
    weight, details = target_null_weight(basis, target)
    np.testing.assert_allclose(basis.T @ weight, 0., atol=3e-15)
    assert details["healthy_rank"] == 5
    assert details["null_dimension"] == 12
    rotation = rng.normal(size=(5, 5))
    alternative, _ = target_null_weight(basis @ rotation @ np.diag([1e-12, .1, 1., 1e4, 1e12]), target)
    np.testing.assert_allclose(alternative, weight, atol=3e-13)
    with pytest.raises(ValueError, match="inseparable"):
        target_null_weight(basis, basis @ rng.normal(size=(5, 4)))
    with pytest.raises(ValueError, match="spans all gauges"):
        target_null_weight(np.eye(7), np.ones((7, 2)))
    healthy, null = response_spaces(np.zeros((9, 2)))
    assert healthy.shape == (9, 0)
    assert null.shape == (9, 9)


def test_exact_optimization_retains_null_constraint_and_target():
    rng = np.random.default_rng(31)
    basis = rng.normal(size=(12, 3)) * 1e-4
    coefficients = rng.uniform(.1, 1., (25, 3))
    healthy = coefficients @ basis.T
    change = rng.normal(size=(12, 3)) * 2e-6
    changed = healthy + coefficients @ change.T
    weight, _ = target_null_weight(basis, (changed - healthy).T)
    bridge = TrimmedBridge.from_weight(np.arange(12), weight)
    circuit = Circuit(gauge_factor=50.)
    fitted, info = optimize_exact_bridge(bridge, basis, healthy, changed, circuit, 2e-15)
    assert info["fitted_design_ratio"] > info["initial_design_ratio"]
    np.testing.assert_allclose(basis.T @ fitted.weight, 0., atol=1e-17)
    # Independent loads check the exact physics, not just the optimizer objective.
    test = rng.uniform(.1, 1., (30, 3)) @ basis.T
    assert np.linalg.norm(fitted.voltage(test, circuit)) < np.linalg.norm(bridge.voltage(test, circuit))
    assert np.linalg.norm(fitted.voltage(changed, circuit) - fitted.voltage(healthy, circuit)) > 1e-7
