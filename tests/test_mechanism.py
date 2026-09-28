import numpy as np
import pytest
from scipy.linalg import expm

from lining_port.network import (
    make_layout, simulate, strain_directional_response, receiver_matrices,
)


def small_network():
    layout = make_layout(17, 3)
    geom = {"W": np.array([[0., .7, 0.], [.7, 0., .2], [0., .2, 0.]]),
            "H": np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 0., 0.]]),
            "mask": np.array([True, True, True, False])}
    return layout, geom, np.array([-7e-5, 2e-5, 8e-5])


def test_p1_static_mixing_with_independent_receiver_propagation():
    layout, geom, strain = small_network()
    gain = layout["a0"] * (1 + .5 * np.tanh(strain / 1e-4))
    w = geom["W"]
    mixed = geom["H"] @ np.linalg.solve(np.eye(3) + .3 * (np.diag(w.sum(1)) - w), gain)
    a, b, c = receiver_matrices()
    aug = np.zeros((5, 5)); aug[:4, :4], aug[:4, 4] = a, b
    step = expm(aug * .005)
    x = np.zeros(5); scalar = []
    for q in np.repeat(np.random.default_rng(4100).choice([-1., 1.], 100), 4):
        scalar.append(c @ x[:4]); x[4] = q; x = step @ x
    actual = simulate(strain, geom, layout, family="P1", method="expm", noise_sigma=0)
    np.testing.assert_allclose(actual["z"], np.outer(scalar, mixed), atol=1e-13)
    assert actual["diagnostics"]["node_states"] == 0
    p0 = simulate(strain, geom, layout, family="P0", method="expm", noise_sigma=0)
    p1_zero = simulate(strain, geom, layout, family="P1", kappa=0., method="expm", noise_sigma=0)
    np.testing.assert_array_equal(p0["z"], p1_zero["z"])


def test_p2_matches_uniform_fixed_relaxation_and_ignores_random_tau():
    layout, geom, strain = small_network()
    p2 = simulate(strain, geom, layout, family="P2", method="expm", noise_sigma=0)
    p3 = simulate(strain, geom, dict(layout, tau0=np.full(3, .05)), family="P3", method="expm", noise_sigma=0)
    np.testing.assert_array_equal(p2["z"], p3["z"])
    assert p2["diagnostics"]["tau_min_s"] == p2["diagnostics"]["tau_max_s"] == .05


@pytest.mark.parametrize("family", ["P0", "P1", "P2", "P3", "P4"])
@pytest.mark.parametrize("law", ["tanh", "algebraic"])
def test_directional_sensitivity_matches_actual_perturbed_response(family, law):
    layout, geom, strain = small_network()
    # A long relaxation deliberately makes reset sensitivity visible in this test.
    layout["tau0"] = np.array([.2, .4, .8])
    direction = np.array([2e-5, -3e-5, 5e-5])
    options = dict(family=family, strain_law=law, strain_transfer=.7)
    derivative = strain_directional_response(strain, direction, geom, layout, **options)
    h = 1e-3
    plus = simulate(strain + h * direction, geom, layout, **options, method="expm", noise_sigma=0)["z"]
    minus = simulate(strain - h * direction, geom, layout, **options, method="expm", noise_sigma=0)["z"]
    np.testing.assert_allclose(derivative["total"], (plus - minus) / (2*h), rtol=2e-6, atol=2e-8)
    if family != "P4":
        assert np.count_nonzero(derivative["time"]) == 0


def test_separate_gain_and_time_pathways_and_unreachable_component():
    layout, geom, strain = small_network()
    direction = np.array([2e-5, -3e-5, 5e-5])
    derivative = strain_directional_response(strain, direction, geom, layout)
    h = 1e-3
    gain_only = []
    for sign in (1, -1):
        shifted = strain + sign*h*direction
        # Compensate the changed exponential factor to keep actual tau fixed.
        compensated = dict(layout, tau0=layout["tau0"] * np.exp(.5 * (np.tanh(strain/1e-4) - np.tanh(shifted/1e-4))))
        gain_only.append(simulate(shifted, geom, compensated, family="P4", method="expm", noise_sigma=0)["z"])
    np.testing.assert_allclose(derivative["gain"], (gain_only[0]-gain_only[1])/(2*h), rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(derivative["total"], derivative["gain"]+derivative["time"], atol=1e-15)
    geom["W"][1, 2] = geom["W"][2, 1] = 0.
    geom["H"][1, :] = 0.; geom["mask"][1] = False
    invisible = strain_directional_response(strain, np.array([0., 0., 1e-4]), geom, layout)
    assert np.max(np.abs(invisible["total"])) == 0.
