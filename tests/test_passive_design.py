from dataclasses import replace

import numpy as np

from lining_port.bridge_network import Circuit
from lining_port.passive_bridge import TrimmedBridge
from lining_port.passive_design import arm_neighbors, constrained_scales, design_gauge_peaks, projection_swaps
from lining_port.passive_resources import design_ratio, routed_hardware, source_power


def test_same_sign_moves_preserve_the_mechanical_operator():
    bridge = TrimmedBridge.from_weight(np.arange(8), np.array([1., -.7, .6, -.5, .8, -.3, .4, -.9]))
    bridge = replace(bridge, attenuation=.7 * bridge.attenuation)
    for trial in arm_neighbors(bridge):
        np.testing.assert_array_equal(trial.weight, bridge.weight)
        np.testing.assert_array_equal(trial.gauge_indices, bridge.gauge_indices)
        assert 1 <= np.count_nonzero(trial.arm != bridge.arm) <= 2


def test_constrained_scales_keep_power_equality_and_all_gauge_limits():
    circuit = Circuit(gauge_factor=50.)
    bridge = TrimmedBridge.from_weight(np.arange(8), np.array([1., -.7, .6, -.5, .8, -.3, .4, -.9]))
    bridge = replace(bridge, attenuation=.6 * bridge.attenuation)
    network = [bridge, replace(bridge, gauge_indices=bridge.gauge_indices + 8)]
    routes = [{"arm_resistance_ohm": [1., 2., 3., 4.], "one_instrument_lead_ohm": 7.,
               "total_conductor_m": 30.}] * 2
    config = {"differential_input_resistance_ohm": 1e6, "average_time_s": 4.}
    acquisition = {"reference_calibration_index": 0, "reference_average_time_s": 4.,
                   "temperature_k": 300., "independent_sample_sigma_v": 1e-6, "sample_period_s": .005}
    rng = np.random.default_rng(91)
    healthy = rng.normal(size=(5, 16)) * 1e-5
    changed = healthy + rng.normal(size=(3, 5, 16)) * 1e-5
    hardware = routed_hardware(network, circuit, routes, 1e6)
    budget = {"conductor_length_m": 60., "source_power_w": float(source_power(hardware, np.zeros(16))),
              "gauge_peak_power_w": float(design_gauge_peaks(hardware, healthy, changed).max())}
    result = constrained_scales(network, circuit, routes, healthy, changed, config, acquisition, budget)
    np.testing.assert_allclose(source_power(result["hardware"], np.zeros(16)), budget["source_power_w"], rtol=1e-13)
    assert design_gauge_peaks(result["hardware"], healthy, changed).max() <= budget["gauge_peak_power_w"] * (1 + 1e-10)
    assert design_ratio(result["hardware"], healthy, changed, config, acquisition)[0] >= design_ratio(
        hardware, healthy, changed, config, acquisition)[0] - 1e-9
    assert constrained_scales(network, circuit, routes, healthy, changed, config, acquisition,
                               budget | {"conductor_length_m": 59.}) is None


def test_projection_exchange_retains_exclusive_groups_and_improves_available_energy():
    rng = np.random.default_rng(18)
    basis = np.column_stack((np.ones(12), np.arange(12)))
    increments = rng.normal(size=(3, 5, 12))
    groups = np.array_split(np.arange(12), 3)
    network = [TrimmedBridge.from_weight(g, np.array([1., -1., -1., 1.])) for g in groups]
    proposals = projection_swaps(network, basis, increments, np.arange(12))
    assert len(proposals) == 3
    for proposal in proposals:
        np.testing.assert_array_equal(np.sort(np.concatenate(proposal["groups"])), np.arange(12))
        assert proposal["projection_log_gain"] > 0
        for q, group in enumerate(proposal["groups"]):
            assert len(set(group) - set(groups[q])) == int(q in proposal["channels"])
