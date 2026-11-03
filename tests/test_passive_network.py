from dataclasses import replace

import numpy as np
import pytest

from lining_port.bridge_network import Circuit
from lining_port.passive_bridge import TrimmedBridge, target_null_weight
from lining_port.passive_hardware import BridgeHardware
from lining_port.passive_network import (channel_capture, network_energies, network_objective,
                                         optimize_exact_network, regional_projection)


def example():
    rng = np.random.default_rng(261016)
    h = rng.normal(size=(20, 3)) * 2e-4
    healthy = rng.uniform(-1, 1, (24, 3)) @ h.T
    increments = rng.normal(size=(2, 24, 20)) * 1e-5
    # Each region can be observed by only one bridge. Both must remain in the
    # network objective even though per-bridge geometric energies contain zeros.
    increments[0, :, 10:] = 0.
    increments[1, :, :10] = 0.
    groups = np.array_split(np.arange(20), 2)
    network = []
    for group in groups:
        w, _ = target_null_weight(h[group], increments[..., group].reshape(-1, 10).T)
        b = TrimmedBridge.from_weight(group, w)
        network.append(replace(b, attenuation=.9 * b.attenuation))
    return network, h, healthy, healthy + increments, Circuit(gauge_factor=50.)


def test_joint_objective_gradient_includes_loaded_reference_and_all_regions():
    network, h, healthy, changed, circuit = example()
    variance = np.array([2e-15, 3e-15])
    hardware = [BridgeHardware.from_design(b, circuit, [2., 3., 4., 5.], 11., 7., 3e6) for b in network]
    objective = network_objective(hardware, healthy, changed, variance)
    c = np.concatenate([b.attenuation for b in network])
    value, gradient = objective(c)
    energy, y = network_energies(hardware, healthy, changed)
    np.testing.assert_allclose(value, np.log(np.mean(np.sum(y*y, axis=-1)) + variance.sum())
                               - np.log(energy.sum(axis=-1)).mean())
    for i in range(len(c)):
        step = np.eye(len(c))[i] * 1e-5
        numeric = (objective(c + step)[0] - objective(c - step)[0]) / 2e-5
        np.testing.assert_allclose(gradient[i], numeric, rtol=4e-5, atol=2e-6)


def test_joint_fit_keeps_physical_membership_and_healthy_nulls():
    network, h, healthy, changed, circuit = example()
    hardware = [BridgeHardware.from_design(b, circuit) for b in network]
    fitted, info = optimize_exact_network(hardware, h, healthy, changed,
                                         [2e-15, 3e-15], [.9, .9])
    assert info['fitted_ratio'] > info['initial_ratio']
    for old, new in zip(network, fitted):
        np.testing.assert_array_equal(new.gauge_indices, old.gauge_indices)
        np.testing.assert_array_equal(new.arm, old.arm)
        np.testing.assert_allclose(h[new.gauge_indices].T @ new.weight, 0., atol=1e-17)
        assert all(row['resistance_ohm'] > 0 for row in new.resistor_netlist(healthy[0], circuit))
    full, grouped = regional_projection(h, changed - healthy, [b.gauge_indices for b in fitted])
    assert np.all(grouped.sum(axis=-1) <= full * (1 + 1e-12))
    assert grouped[0, 1] == grouped[1, 0] == 0.
    with pytest.raises(ValueError, match='only one bridge'):
        optimize_exact_network([hardware[0], hardware[0]], h, healthy, changed,
                               [2e-15, 3e-15], [.9, .9])


def test_channel_capture_resolves_both_losses_and_preserves_same_sign_moves():
    network, h, healthy, changed, _ = example()
    increments = changed - healthy
    capture = channel_capture(h, increments, network)
    u = capture['unit_directions']
    np.testing.assert_allclose(u @ u.T, np.eye(2), atol=1e-15)
    np.testing.assert_allclose(u @ h, 0., atol=1e-18)
    grouped = capture['group_energy_strain2'].sum(-1)
    actual = capture['channel_energy_strain2'].sum(-1)
    np.testing.assert_allclose(capture['global_energy_strain2'],
        actual + capture['grouping_loss_strain2'] + capture['direction_loss_strain2'])
    np.testing.assert_allclose(grouped, actual + capture['direction_loss_strain2'])
    assert np.all(capture['grouping_loss_strain2'] > 0)
    assert np.all(capture['direction_loss_strain2'] > 0)
    # Gauge-order permutations, repeated loads and same-sign arm changes
    # preserve the same physical channel directions and load-mean energies.
    moved = [replace(b, gauge_indices=b.gauge_indices[::-1],
                     attenuation=b.attenuation[::-1], arm=3 - b.arm[::-1]) for b in network]
    repeated = channel_capture(h, np.repeat(increments, 2, axis=1), moved)
    for key in capture:
        np.testing.assert_allclose(repeated[key], capture[key], rtol=1e-13, atol=1e-25)
