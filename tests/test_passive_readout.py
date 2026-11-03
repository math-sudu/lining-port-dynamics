from dataclasses import replace

import numpy as np

from lining_port.bridge_network import Circuit
from lining_port.passive_bridge import TrimmedBridge
from lining_port.passive_hardware import BridgeHardware
from lining_port.passive_readout import sample_fixed_reference
from test_passive_bridge import solve_netlist


def test_loaded_reference_matches_component_netlist_and_recentered_coordinates():
    rng = np.random.default_rng(205)
    circuit = Circuit(gauge_factor=50.)
    bridge = TrimmedBridge.from_weight(np.arange(16), rng.normal(size=16))
    bridge = replace(bridge, attenuation=.9 * bridge.attenuation,
                     nominal_ratio=rng.uniform(.99, 1.01, 16))
    device = BridgeHardware.from_design(bridge, circuit, np.arange(1., 5.), 2., 3., 1e9)
    gain = rng.uniform(.99, 1.01, 16)
    reference = rng.normal(2e-4, 1e-4, 16) * gain
    strain = rng.normal(2e-4, 1e-4, (3, 16)) * gain
    errors = dict(shunt_error=rng.uniform(-1e-4, 1e-4, 16),
                  ballast_error=rng.uniform(-1e-4, 1e-4, 4))
    expected_ref = solve_netlist(device.netlist(reference, **errors), circuit.voltage)
    expected = np.array([solve_netlist(device.netlist(e, **errors), circuit.voltage) for e in strain])
    actual = device.voltage(strain, **errors) - device.voltage(reference, **errors)
    np.testing.assert_allclose(actual, expected - expected_ref, atol=3e-11)
    # R_ref and F_ref describe the same components at a loaded installation.
    scale = 1 + circuit.gauge_factor * reference
    recentered = replace(device, gauge_nominal=device.gauge_nominal * scale)
    relative_strain = (strain - reference) / scale
    np.testing.assert_allclose(recentered.voltage(relative_strain, **errors),
                               device.voltage(strain, **errors), atol=5e-16)


def test_shared_recorded_reference_has_the_required_cross_reading_covariance():
    draws = 120000
    current = np.zeros((2, draws, 3, 2))
    reference = np.zeros((draws, 2))
    measured, recorded, output = sample_fixed_reference(
        current, reference, np.array([4., 1.]), np.array([9., 1.]), np.random.default_rng(51))
    np.testing.assert_array_equal(output, measured - recorded[:, None, :])
    # Different readings and alternative physical states share one reference;
    # separate channels retain independent electronics and thermal noise.
    covariance = np.cov(np.array([output[0, :, 0, 0], output[0, :, 1, 0],
                                  output[1, :, 2, 0], output[0, :, 0, 1]]))
    expected = np.array([[13., 9., 9., 0.], [9., 13., 9., 0.],
                         [9., 9., 13., 0.], [0., 0., 0., 2.]])
    np.testing.assert_allclose(covariance, expected, atol=.15)


def test_noiseless_readout_retains_signed_state_identity_and_raw_offset():
    reference = np.array([[1e-3, -2e-3], [2e-3, 3e-3]])
    healthy = reference[:, None, :] + np.array([[[1e-6, -1e-6], [2e-6, 3e-6]]])
    increment = np.array([[[-1e-6, 2e-6], [4e-6, -3e-6]]])
    current = np.stack([healthy, healthy + increment])
    measured, recorded, output = sample_fixed_reference(current, reference, 0., 0., np.random.default_rng(0))
    np.testing.assert_array_equal(measured, current)
    np.testing.assert_array_equal(recorded, reference)
    np.testing.assert_allclose(output[1], output[0] + increment, atol=1e-18)
