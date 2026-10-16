import numpy as np
from dataclasses import replace

from lining_port.frame_response import BeamSection, PlaneFrame, asymmetric_frame, joint_load_basis, random_installation
from lining_port.bridge_network import Circuit
from lining_port.frame_bridge_study import load_samples
from lining_port.passive_bridge import TrimmedBridge, distributed_target_weight, optimize_exact_bridge, target_null_weight


def test_cantilever_force_solution_and_gauge_average():
    length, force = 3., 1200.
    section = BeamSection()
    beam = PlaneFrame([[0., 0.], [length, 0.]], [[0, 1]], [0], section)
    load = np.zeros((2, 3, 1))
    load[1, 1, 0] = force
    displacement = beam.displacement(load)
    np.testing.assert_allclose(displacement[4, 0], force * length**3 / (3 * section.youngs * section.inertia), rtol=1e-13)
    x, z = np.array([.13, .5, .88]), .12
    strain = beam.strain(displacement, np.zeros(3, int), x, np.full(3, z), .1)[:, 0]
    np.testing.assert_allclose(strain, -z * force * length * (1 - x) / (section.youngs * section.inertia), rtol=2e-13)


def test_piecewise_stiffness_step_preserves_force_and_compatible_deflection():
    section = BeamSection()
    length, force, reduction = 3., 1200., .3
    beam = PlaneFrame([[0., 0.], [length, 0.]], [[0, 1]], [0], section, [(0, .3, .6, reduction)])
    load = np.zeros((2, 3, 1))
    load[1, 1, 0] = force
    displacement = beam.displacement(load)
    integral = length**3 / 3 + (1 / (1 - reduction) - 1) * ((.7 * length)**3 - (.4 * length)**3) / 3
    np.testing.assert_allclose(displacement[4, 0], force * integral / (section.youngs * section.inertia), rtol=1e-12)
    internal = beam.stiffness @ displacement
    np.testing.assert_allclose(internal[1:3, 0], [-force, -force * length], rtol=1e-12)
    # A gauge crossing the modulus step averages its two constant-modulus pieces.
    strain = beam.strain(displacement, np.array([0]), np.array([.3]), np.array([.1]), .06)[0, 0]
    expected = -.1 * force / (section.youngs * section.inertia) * .5 * ((length - .885) + (length - .915) / .7)
    np.testing.assert_allclose(strain, expected, rtol=1e-12)


def test_frame_rank_and_uniform_softening_are_physical():
    frame, installation = asymmetric_frame(), random_installation(8200)
    h = frame.response(joint_load_basis(6), installation)
    changed = asymmetric_frame(modulus_scale=.9).response(joint_load_basis(6), installation)
    assert np.linalg.matrix_rank(h) == 6
    np.testing.assert_allclose(changed, h / .9, rtol=2e-11, atol=1e-17)
    displacement = frame.displacement(joint_load_basis(6))
    applied = joint_load_basis(6).reshape(-1, 6)
    np.testing.assert_allclose((frame.stiffness @ displacement)[frame.free], applied[frame.free], atol=3e-9)
    distance, instrument, center = frame.path_distances(**{
        "gauge_member": installation["member"], "fraction": installation["fraction"], "depth": installation["depth_m"]})
    np.testing.assert_allclose(distance, distance.T)
    assert instrument.min() > 0 and np.isfinite(center).all()


def test_frame_active_bound_restoration_and_spatial_target_aggregation():
    installation = random_installation(8200)
    loads = joint_load_basis(6)
    basis = asymmetric_frame().response(loads, installation)
    coefficients = load_samples(np.random.default_rng(2026101607), 32, 6)
    healthy = coefficients @ basis.T
    changed = np.stack([coefficients @ asymmetric_frame([(member, center - .09, center + .09, .35)]).response(loads, installation).T
                        for member in range(5) for center in (.2, .5, .8)])
    circuit = Circuit(gauge_factor=50.)
    cap = (10e6 / 10.1e6)**2
    order = np.lexsort((installation["fraction"], installation["member"]))
    for channel, group in enumerate(np.array_split(order, 4)):
        targets = (changed - healthy)[..., group].reshape(-1, len(group)).T
        for aggregation in ("mean", "geometric"):
            weight = target_null_weight(basis[group], targets)[0] if aggregation == "mean" else distributed_target_weight(basis[group], targets, 8200 + channel)
            initial = TrimmedBridge.from_weight(group, weight)
            initial = replace(initial, attenuation=cap * initial.attenuation)
            variance = initial.mean_noise_variance(circuit, 4., 1e-6, .005)
            fitted, result = optimize_exact_bridge(initial, basis, healthy, changed, circuit, variance, cap, aggregation)
            assert np.all((fitted.attenuation >= 0) & (fitted.attenuation <= cap))
            np.testing.assert_allclose(basis[group].T @ fitted.weight, 0., atol=5e-16)
            assert result["fitted_design_ratio"] >= result["initial_design_ratio"]
