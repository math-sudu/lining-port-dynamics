import numpy as np
import pytest
from scipy.optimize import least_squares

from lining_port.conditional_matching import ConditionalCoordinates, initial_points
from lining_port.fe import RingParameters
from lining_port.matching import MatchingObjective, ParameterCoordinates, forward
from lining_port.network import make_layout


BOUNDS = {'youngs': [27e9, 33e9], 'thickness': [.27, .33], 'radial_modulus': [40e6, 60e6],
          'pressure': [1e5, 3e5], 'harmonic': [0, .3], 'phase': [0, 2*np.pi]}
GROUPS = [('pressure', 'harmonic', 'phase'), ('youngs', 'thickness', 'radial_modulus')]


@pytest.mark.parametrize('free', GROUPS)
def test_fixed_group_is_preserved_even_for_phase_above_pi(free):
    base = RingParameters(phase=5.7, youngs=29.87654321e9, thickness=.30123456789)
    coords = ConditionalCoordinates(base, BOUNDS, free)
    changed = coords.decode(coords.encode(base) + .025)
    for field in coords.fixed_parameters:
        assert getattr(changed, field) == getattr(base, field)
    assert changed.reduction == 0.


@pytest.mark.parametrize('free', GROUPS)
def test_conditional_jacobian_and_actual_known_sequence_recovery(free):
    base = RingParameters(phase=5.7)
    layout = make_layout(5100)
    target = forward(base, layout, 'P3_coupled', 64)
    coords = ConditionalCoordinates(base, BOUNDS, free)
    obj = MatchingObjective(coords, layout, 'P3_coupled', target['z'], 64, .01, None)
    x = coords.encode(base)
    full_coords = ParameterCoordinates(base, BOUNDS)
    full_obj = MatchingObjective(full_coords, layout, 'P3_coupled', target['z'], 64, .01, None)
    np.testing.assert_allclose(obj.jac(x), full_obj.jac(full_coords.encode(base))[:, coords.indices],
                               rtol=1e-5, atol=2e-6)
    initial = x + np.array([.015, -.015, .015])
    before = np.linalg.norm(obj.fun(initial))
    fit = least_squares(obj.fun, initial, jac=obj.jac, bounds=coords.bounds, max_nfev=30)
    assert fit.success
    assert np.linalg.norm(fit.fun) < before * 1e-3
    for field in coords.fixed_parameters:
        assert getattr(coords.decode(fit.x), field) == getattr(base, field)


def test_linear_initialization_solves_reduced_problem_and_repeats_seed():
    base = RingParameters()
    coords = ConditionalCoordinates(base, BOUNDS, GROUPS[0])
    scales = np.r_[coords.full.span, np.pi]
    normalized_jacobian = np.eye(6)
    normalized_jacobian[3, 0] = 2.
    # The omitted structural column changes the full solution but must not
    # contribute to the load-only initialization.
    delta = np.array([.01, 0., 0., .07, -.02, .01])
    tangent = {'nuisance_jacobian': normalized_jacobian / scales,
               'healthy': np.zeros(6), 'softened': delta}
    points, _ = initial_points(coords, tangent, 31)
    np.testing.assert_allclose(points['clipped_linear'] - points['healthy'], [.07, -.02, .01])
    repeated, _ = initial_points(coords, tangent, 31)
    for name in points:
        np.testing.assert_array_equal(points[name], repeated[name])
