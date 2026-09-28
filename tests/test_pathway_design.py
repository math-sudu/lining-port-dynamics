import numpy as np

from lining_port.pathway_design import projected_pathways


def test_projection_can_reverse_raw_pathway_interaction():
    gain = np.array([2., 1.])
    time = np.array([1., -1.])
    actual_nuisance = np.array([[1.], [0.]])
    frozen_nuisance = np.array([[0.], [1.]])
    result = projected_pathways(gain, time, actual_nuisance, frozen_nuisance, 1.)
    assert gain @ time > 0
    assert result["projected_cosine"] == -1.
    assert result["distance"] == 0.
    assert result["frozen_distance"] == 2.
    assert result["time_effect_squared"] == -1.
    assert result["nuisance_space_effect_squared"] == -3.
    assert result["total_effect_squared"] == -4.


def test_unchanged_nuisance_space_has_no_space_change_term():
    gain = np.array([1., 2., 3.])
    time = np.array([3., -1., 2.])
    nuisance = np.array([[1.], [0.], [0.]])
    result = projected_pathways(gain, time, nuisance, nuisance, .5)
    assert result["nuisance_space_effect_squared"] == 0.
    np.testing.assert_allclose(result["total_effect_squared"], result["time_effect_squared"])
