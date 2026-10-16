import numpy as np
from scipy.integrate import solve_ivp

from lining_port.acquisition_comparison import (
    bridge_netlists, bridge_voltage, scan_operator, divider_voltage, mechanical_bridge_voltage,
)
from lining_port.bridge_network import Circuit
from lining_port.network import receiver_matrices


def test_scan_retains_filter_memory_and_all_64_inputs():
    operator, times = scan_operator()
    assert operator.shape == (2944, 64)
    assert len(times) == 736
    assert np.linalg.matrix_rank(operator) == 64
    values = np.linspace(-.002, .002, 64)
    actual = (operator @ values).reshape(-1, 4)
    a, b, c = receiver_matrices()
    state = np.zeros(4)
    for _ in range(3):
        expected = []
        for slot in range(16):
            reset = solve_ivp(lambda t, x: a @ x, (0, .02), state, rtol=1e-10, atol=1e-13)
            sol = solve_ivp(lambda t, x: a @ x + b * values[slot], (0, .23), reset.y[:, -1],
                            rtol=1e-10, atol=1e-13, dense_output=True)
            expected.extend(c @ sol.sol(np.arange(46) * .005))
            state = sol.y[:, -1]
    np.testing.assert_allclose(actual[:, 0], expected, atol=2e-12)
    assert times[-1] == 3.995


def test_mechanical_bridge_uses_all_gauges_and_cancels_common_extension():
    circuit = Circuit(gauge_factor=50)
    np.testing.assert_allclose(mechanical_bridge_voltage(np.full(64, 1e-4), circuit), 0)
    pure_bending = np.tile([1e-4, -1e-4], 32)
    np.testing.assert_allclose(mechanical_bridge_voltage(pure_bending, circuit), -.01)
    for i in range(64):
        strain = np.zeros(64)
        strain[i] = 1e-4
        values = mechanical_bridge_voltage(strain, circuit)
        assert values[i // 16] != 0
        assert np.count_nonzero(values) == 1
    np.testing.assert_allclose(divider_voltage(np.array([0., 1e-4]), circuit),
                               [0., -0.005 / 2.005])


def test_composite_bridges_cancel_antipodal_loading_and_keep_signed_changes():
    circuit = Circuit(gauge_factor=50)
    rng = np.random.default_rng(20261016)
    half = rng.uniform(-.003, .003, (3, 32))
    healthy = np.concatenate((half, half), axis=-1)
    for name, nodes in bridge_netlists().items():
        np.testing.assert_array_equal(np.sort(nodes.ravel()), np.arange(64))
        if name == "mechanical_bridge":
            continue
        np.testing.assert_allclose(bridge_voltage(healthy, nodes, circuit), 0, atol=3e-16)

        changed = healthy.copy()
        changed[:, 0] += .0002
        output = bridge_voltage(changed, nodes, circuit)
        assert np.all(output[:, 0] < 0)
        assert np.all(np.count_nonzero(abs(output) > 1e-14, axis=1) == 1)
        rotated = np.roll(changed, 32, axis=-1)
        np.testing.assert_allclose(bridge_voltage(rotated, nodes, circuit), -output,
                                   atol=8 * np.finfo(float).eps * circuit.voltage)


def test_composite_four_measures_normalized_curvature_difference_exactly():
    circuit = Circuit(gauge_factor=50)
    depth = .1
    membrane = np.linspace(-.002, .001, 32)
    curvature = np.sin(np.arange(32)) * .003
    strain = (membrane[:, None] - np.array([-depth, depth]) * curvature[:, None]).ravel()
    expected = []
    for start in range(0, 16, 4):
        a, b = slice(start, start + 4), slice(start + 16, start + 20)
        expected.append(circuit.voltage * circuit.gauge_factor * depth * (
            curvature[b].mean() / (1 + circuit.gauge_factor * membrane[b].mean())
            - curvature[a].mean() / (1 + circuit.gauge_factor * membrane[a].mean())))
    np.testing.assert_allclose(bridge_voltage(strain, bridge_netlists()["composite4"], circuit),
                               expected, rtol=1e-12, atol=3e-16)
