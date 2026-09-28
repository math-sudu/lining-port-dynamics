import numpy as np
import pytest
from lining_port.network import make_layout, node_parameters, simulate
from lining_port.pathway_attribution import decompose_pair


@pytest.mark.parametrize('alpha', [.5, -.5])
def test_frozen_time_control_has_identical_healthy_response(alpha):
    layout = make_layout(77, 3)
    geom = {'W': np.array([[0., .2, .3], [.2, 0., .4], [.3, .4, 0.]]),
            'H': np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]),
            'mask': np.array([True, True, True, False])}
    healthy = np.array([-1e-4, 3e-5, -2e-5]); softened = healthy+np.array([2e-5, -1e-5, 3e-5])
    result = decompose_pair(healthy, softened, geom, layout, alpha=alpha)
    matched = dict(layout, tau0=node_parameters(healthy, layout, 'P4', alpha)['tau'])
    control = simulate(healthy, geom, matched, family='P3', method='expm', noise_sigma=0)['z_clean']
    np.testing.assert_array_equal(control, result['healthy'])
    np.testing.assert_allclose(result['gain']+result['time'], result['total'], atol=1e-16)
    null = decompose_pair(healthy, softened, geom, layout, alpha=0.)
    np.testing.assert_array_equal(null['time'], np.zeros_like(null['time']))
