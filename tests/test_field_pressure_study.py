import numpy as np

from lining_port.fe import RingParameters, recover_gauge_strain, solve_ring
from lining_port.field_pressure_study import pressure_response


def test_pressure_input_matches_existing_uniform_load_solution():
    params = RingParameters(harmonic=0.)
    theta = np.array([.13, .69, 1.28, 2.13, 3.2, 4.77, 5.67])
    depth = np.linspace(-.1, .1, len(theta))
    station_angles = (np.pi / 2 - np.arange(8) * np.pi / 4) % (2 * np.pi)
    result = pressure_response(params, station_angles, np.full(8, params.pressure), theta, depth)
    expected = recover_gauge_strain(solve_ring(params), theta, depth, .0002)
    np.testing.assert_allclose(result, expected, rtol=2e-7, atol=2e-12)
