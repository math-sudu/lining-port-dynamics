import itertools

import numpy as np

from lining_port.acquisition_architecture import allocate_samples


def test_sample_allocation_matches_integer_optimum():
    weights = np.array([.3, 1.7, 4.])
    allocation = allocate_samples(weights, 12)
    candidates = [n for n in itertools.product(range(1, 11), repeat=3) if sum(n) == 12]
    optimum = min(np.sum(weights / n) for n in candidates)
    assert np.isclose(np.sum(weights / allocation), optimum)


def test_bypassed_input_gets_no_samples():
    allocation = allocate_samples(np.array([1., 0., 1.]), 20)
    np.testing.assert_array_equal(allocation, [10, 0, 10])
