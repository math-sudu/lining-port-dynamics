import numpy as np

from lining_port.acquisition_comparison import bridge_voltage
from lining_port.bridge_network import Circuit
from lining_port.network import geometry, make_layout
from lining_port.random_bridge_study import exchange_arm_membership, random_gauge_geometry


def test_random_geometry_restores_original_positions():
    seeds = [7100, 7101]
    actual = random_gauge_geometry(seeds, 3., .3)
    for index, seed in enumerate(seeds):
        original = geometry(make_layout(seed), 3., .3)
        np.testing.assert_array_equal(actual["theta"][index], original["theta"])
        np.testing.assert_allclose(actual["z"][index], original["z"], atol=1e-15)
    assert not np.array_equal(actual["theta"][0], actual["theta"][1])


def test_exchanges_preserve_gauges_and_reach_exact_voltage_local_minimum():
    rng = np.random.default_rng(20261016)
    strain = rng.normal(0, 1e-4, (8, 16))
    circuit = Circuit(gauge_factor=50)
    initial = np.arange(16).reshape(1, 4, 4)
    nodes, steps = exchange_arm_membership(strain, initial, circuit)
    np.testing.assert_array_equal(np.sort(nodes.ravel()), np.arange(16))
    np.testing.assert_array_equal(initial.ravel(), np.arange(16))
    score = np.sum(bridge_voltage(strain, nodes, circuit)**2)
    assert steps > 0
    assert score < np.sum(bridge_voltage(strain, initial, circuit)**2)
    for first in range(16):
        for second in range(first + 1, 16):
            candidate = nodes.ravel().copy()
            candidate[first], candidate[second] = candidate[second], candidate[first]
            alternative = np.sum(bridge_voltage(strain, candidate.reshape(nodes.shape), circuit)**2)
            assert alternative >= score - 1e-16


def test_common_extension_stays_balanced_without_exchanges():
    nodes = np.arange(64).reshape(4, 4, 4)
    strain = np.broadcast_to(np.array([0., 1e-4, -.0002])[:, None], (3, 64))
    designed, steps = exchange_arm_membership(strain, nodes, Circuit(gauge_factor=50))
    np.testing.assert_array_equal(designed, nodes)
    assert steps == 0
