"""Finite-time bridge measurements using one recorded installation reference."""
import numpy as np


def sample_fixed_reference(current_v, reference_v, current_variance_v2,
                           reference_variance_v2, rng):
    """Sample currents and subtract the same noisy reference from every case.

    Current arrays end in (hardware_draw, case, channel); arbitrary leading
    state/scenario axes share the reference of shape (hardware_draw, channel).
    Variances use the existing white-noise model for each averaging interval.
    Returns current averages, recorded reference and their signed difference.
    """
    current_v, reference_v = np.asarray(current_v), np.asarray(reference_v)
    reference = reference_v + rng.normal(size=reference_v.shape) * np.sqrt(reference_variance_v2)
    current = current_v + rng.normal(size=current_v.shape) * np.sqrt(current_variance_v2)
    return current, reference, current - reference[..., None, :]
