import numpy as np
from scipy.integrate import solve_ivp

from lining_port.acquisition_comparison import scan_operator, divider_voltage, mechanical_bridge_voltage
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
