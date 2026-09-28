import numpy as np
from dataclasses import replace

from lining_port.fe import RingParameters
from lining_port.continuum import (
    plane_stress_matrix, shape, strain_matrix, solve_continuum,
    point_fields, recover_continuum_gauge,
)


def test_q4_constant_strain_patch_on_distorted_element():
    coords = np.array([[0., 0.], [2., .1], [1.8, 1.], [-.1, 1.1]])
    gradient = np.array([[.03, .04], [-.02, .01]])
    displacements = coords@gradient.T+np.array([.3, -.4])
    expected = np.array([.03, .01, .02])
    for xi, eta in [(-.7, .4), (.2, -.3), (0., 0.)]:
        _, dn = shape(xi, eta)
        b, _ = strain_matrix(coords, dn)
        np.testing.assert_allclose(b@displacements.ravel(), expected, atol=1e-15)
    stress = plane_stress_matrix(30e9, .2)@np.array([1e-4, -2e-5, 0.])
    np.testing.assert_allclose(stress, [3e6, 0., 0.], atol=1e-8)


def test_uniform_supported_annulus_against_lame_solution():
    p = RingParameters(harmonic=0., reduction=0.)
    result = solve_continuum(p, 256, 4)
    inner, outer = p.radius-p.thickness/2, p.radius+p.thickness/2
    nu = result.poisson
    compliance = outer/p.youngs*((1-nu)*outer**2+(1+nu)*inner**2)/(outer**2-inner**2)
    pressure = p.pressure*p.radius/outer/(1+compliance*p.radial_modulus*p.radius/outer)
    aa = -pressure*outer**2/(outer**2-inner**2)
    bb = aa*inner**2
    theta = np.linspace(0, 2*np.pi, 31, endpoint=False)+.013
    depth = np.linspace(-.13, .13, len(theta))
    radius = p.radius+depth
    expected_u = ((1-nu)*aa*radius+(1+nu)*bb/radius)/p.youngs
    fields = point_fields(result, theta, depth)
    radial = np.column_stack((np.cos(theta), np.sin(theta)))
    actual_u = np.sum(fields['displacement']*radial, axis=1)
    np.testing.assert_allclose(actual_u, expected_u, rtol=1e-3)
    np.testing.assert_allclose(fields['hoop_strain'], expected_u/radius, rtol=2e-3)
    assert result.diagnostics['relative_equilibrium_residual'] < 1e-9


def test_softened_field_rotation_covariance_and_periodic_gauge():
    p = RingParameters(reduction=.5, sector_center=.4, phase=.17)
    shift = 2*np.pi/128*9
    base = solve_continuum(p, 128, 3)
    rotated = solve_continuum(replace(p, sector_center=p.sector_center+shift, phase=p.phase+shift), 128, 3)
    theta = np.array([-.001, .3, 1.4, 5.8]); depth = np.array([.02, -.04, .1, -.11])
    eps = recover_continuum_gauge(base, theta, depth)
    np.testing.assert_allclose(eps, recover_continuum_gauge(rotated, theta+shift, depth), rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(eps, recover_continuum_gauge(base, theta+2*np.pi, depth), rtol=1e-10, atol=1e-14)
