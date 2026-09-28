from dataclasses import asdict, replace
import numpy as np

from lining_port.fe import RingParameters, solve_ring, recover_gauge_strain
from lining_port.mechanism_study import generate, nuisance_projection, signature_metrics


def test_joint_mechanical_scaling_preserves_the_strain_state():
    p = RingParameters(reduction=.4, sector_center=.7)
    scaled = replace(p, youngs=p.youngs*1.1, pressure=p.pressure*1.1,
                     radial_modulus=p.radial_modulus*1.1)
    theta = np.linspace(0, 2*np.pi, 37, endpoint=False)
    original = recover_gauge_strain(solve_ring(p), theta, .07)
    repeated = recover_gauge_strain(solve_ring(scaled), theta, .07)
    np.testing.assert_allclose(original, repeated, rtol=1e-8, atol=1e-13)


def test_amplitude_matching_and_nuisance_projection_have_distinct_meanings():
    healthy = np.column_stack((np.arange(1., 11.), np.sin(np.arange(10.))))
    softened = healthy*np.array([1.2, .8])
    summary = signature_metrics(healthy, softened, .01)
    assert summary['distance'] > 100
    assert summary['shape_distance'] < 1e-10
    delta = np.array([2., 3., 4.])
    jac = np.array([[1., 2.], [0., 0.], [0., 0.]])
    projection = nuisance_projection(delta, jac, 1.)
    assert projection['rank'] == 1
    np.testing.assert_allclose(projection['residual_distance'], 5.)


def test_generation_keeps_acquisition_noise_independent_and_models_paired(tmp_path):
    p = asdict(RingParameters()); p.pop('reduction')
    manifest = [{'scenario_family_id': f'N{i:03d}', 'split': split, 'params': p,
                 'positive_reduction': .45, 'acquisition_noise_seeds': [[100+4*i+2*j, 101+4*i+2*j] for j in range(2)]}
                for i, split in enumerate(('train', 'test'))]
    config = {'layout_seeds': [7100, 7101], 'nodes': 64, 'beam_elements': 128,
              'gauge_length_m': .01, 'receiver_noise_sigma': .01,
              'physical_models': {'P0': {'family': 'P0', 'kappa': 0., 'alpha': 0.},
                                  'P1_zero': {'family': 'P1', 'kappa': 0., 'alpha': 0.}}}
    _, metadata, observations, clean = generate(config, manifest, tmp_path)
    np.testing.assert_array_equal(observations['P0'], observations['P1_zero'])
    assert len(np.unique(metadata['noise_seed'])) == 8
    assert len(np.unique(metadata['sample_id'])) == 8
    for label in (0, 1):
        index = 4+label
        reference_noise = np.random.default_rng(metadata['noise_seed'][index]).normal(0, .01, (400, 4))*metadata['mask'][index]
        recovered = observations['P0'][index].astype(float)-clean[(1, 0, 'P0')][label]
        np.testing.assert_allclose(recovered, reference_noise, atol=1e-7)
