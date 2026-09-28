import numpy as np
import pytest

from lining_port.fe import (
    RingParameters, _beam_check, _support_direction_check, recover_gauge_strain,
    run_fe_validation, solve_ring,
)


def test_beam_analytic_and_fiber_sign():
    assert _beam_check()['pass_']


def test_support_directions():
    assert _support_direction_check()['pass_']


def test_gauge_integrates_across_periodic_boundary():
    result = solve_ring(RingParameters(reduction=.6, sector_center=.02), 64)
    angles = np.array([0, 2*np.pi, -2*np.pi, .123])
    recovered = recover_gauge_strain(result, angles, .13, gauge_length=.1)
    assert np.allclose(recovered[:3], recovered[0], atol=1e-15)
    sample_angle = np.linspace(-.05, .05, 20001)/result.params.radius
    dense = recover_gauge_strain(result, sample_angle, .13, gauge_length=0)
    assert recovered[0] == pytest.approx(np.trapezoid(dense, sample_angle)/np.ptp(sample_angle), abs=1e-10)


def test_same_seed_same_structure_result():
    p = RingParameters(reduction=.4, sector_center=.37)
    first, second = solve_ring(p, 64), solve_ring(p, 64)
    assert np.array_equal(first.displacements, second.displacements)
    assert np.array_equal(first.local_forces, second.local_forces)


def test_invalid_support_and_gauge_fail_explicitly():
    with pytest.raises(ValueError, match='foundation'):
        solve_ring(RingParameters(tangential_ratio=0), 64)
    result = solve_ring(RingParameters(), 64)
    with pytest.raises(ValueError, match='inside the section'):
        recover_gauge_strain(result, 0, 1)


def test_fe_validation_matrix(tmp_path):
    report = run_fe_validation(tmp_path)
    assert report['pass'], report
    assert report['selected_elements'] in [128, 256]
    assert (tmp_path/'structure_example.npz').exists()
