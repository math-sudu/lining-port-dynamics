"""Joint regional design of bridges with exclusive gauge membership."""
from dataclasses import replace

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize

from .passive_bridge import ARM_SIGN, response_spaces


def regional_projection(healthy_basis, increments, groups):
    """Full-array and grouped extractable energies for each prescribed region."""
    _, null = response_spaces(healthy_basis)
    full = np.sum((increments @ null)**2, axis=(1, 2))
    grouped = np.stack([
        np.sum((increments[..., group] @ response_spaces(healthy_basis[group])[1])**2,
               axis=(1, 2)) for group in groups], axis=-1)
    return full, grouped


def channel_capture(healthy_basis, increments, network):
    """Mean-square strain available globally, within groups and in channels.

    Increments have region/load/gauge axes in the same coordinates as the
    healthy basis. Each nonzero bridge weight lies in its group's healthy
    null space; gauge membership is exclusive. All design loads have equal
    weight, matching ``network_energies``.
    """
    _, null = response_spaces(healthy_basis)
    projector = null @ null.T
    grouped = np.zeros_like(projector)
    directions = np.zeros((len(network), len(healthy_basis)))
    group_energy = []
    for q, bridge in enumerate(network):
        group = bridge.gauge_indices
        norm = np.linalg.norm(bridge.weight)
        if norm == 0:
            raise ValueError("A channel direction requires a nonzero bridge weight")
        directions[q, group] = bridge.weight / norm
        local_null = response_spaces(healthy_basis[group])[1]
        grouped[np.ix_(group, group)] = local_null @ local_null.T
        group_energy.append(np.mean(np.sum((increments[..., group] @ local_null)**2,
                                           axis=-1), axis=1))
    channel_projector = directions.T @ directions

    def energy(operator):
        return np.mean(np.sum((increments @ operator)**2, axis=-1), axis=1)

    return {
        "unit_directions": directions,
        "global_energy_strain2": energy(projector),
        "group_energy_strain2": np.stack(group_energy, axis=-1),
        "channel_energy_strain2": np.mean((increments @ directions.T)**2, axis=1),
        "grouping_loss_strain2": energy(projector - grouped),
        "direction_loss_strain2": energy(grouped - channel_projector),
    }


def network_energies(hardware, healthy, changed, reference_index=0):
    """Regional channel energies and the signed, loaded-reference healthy output."""
    vh = np.stack([h.voltage(healthy) for h in hardware], axis=-1)
    vc = np.stack([h.voltage(changed) for h in hardware], axis=-1)
    return np.mean((vc - vh)**2, axis=1), vh - vh[reference_index]


def regional_log_ratio(energies, healthy_output, noise_variance):
    """Geometric regional energy over the common recorded-reference background."""
    background = np.mean(np.sum(healthy_output**2, axis=-1)) + np.sum(noise_variance)
    return float(np.mean(np.log(energies.sum(axis=-1))) - np.log(background))


def network_objective(hardware, healthy, changed, noise_variance,
                      reference_index=0, target_aggregation="geometric"):
    """Return the negative log regional signal/background ratio and its gradient.

    Each region is retained in the sum over channels, including regions with
    zero projection in an individual group. Noise variance includes both the
    reference and service averages; coefficients do not change arm resistance.
    """
    if changed.ndim != 3 or changed.shape[1:] != healthy.shape:
        raise ValueError("Changed strains must have region-by-load-by-gauge axes")
    if target_aggregation not in ("mean", "geometric"):
        raise ValueError("Target aggregation must be mean or geometric")
    network = [h.bridge for h in hardware]
    offsets = np.r_[0, np.cumsum([len(b.attenuation) for b in network])]

    def objective(c):
        energies = np.zeros(len(changed))
        background = float(np.sum(noise_variance))
        d_energy = np.empty((len(changed), len(c)))
        d_background = np.empty(len(c))
        for q, device in enumerate(hardware):
            sl = slice(offsets[q], offsets[q + 1])
            vh, dh = device.voltage_and_gradient(healthy, c[sl])
            vc, dc = device.voltage_and_gradient(changed, c[sl])
            y, dy = vh - vh[reference_index], dh - dh[reference_index]
            delta, dd = vc - vh, dc - dh
            energies += np.mean(delta**2, axis=1)
            background += np.mean(y**2)
            d_energy[:, sl] = 2 * np.mean(delta[..., None] * dd, axis=1)
            d_background[sl] = 2 * np.mean(y[:, None] * dy, axis=0)
        if target_aggregation == "mean":
            energies = np.mean(energies, keepdims=True)
            d_energy = np.mean(d_energy, axis=0, keepdims=True)
        if np.any(energies <= 0):
            return np.inf, np.zeros_like(c)
        value = np.log(background) - np.mean(np.log(energies))
        gradient = d_background / background - np.mean(d_energy / energies[:, None], axis=0)
        return value, gradient

    return objective


def optimize_exact_network(hardware, healthy_basis, healthy, changed,
                           noise_variance, max_attenuation, reference_index=0,
                           target_aggregation="geometric"):
    """Jointly refine all coefficients at fixed membership, signs and arm scales."""
    network = [h.bridge for h in hardware]
    indices = np.concatenate([b.gauge_indices for b in network])
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Each gauge may belong to only one bridge")
    constraints = [response_spaces(healthy_basis[b.gauge_indices])[0].T
                   * ARM_SIGN[b.arm] for b in network]
    constraint = block_diag(*constraints)
    caps = np.concatenate([np.full(len(b.attenuation), cap)
                           for b, cap in zip(network, max_attenuation)])
    initial = np.concatenate([b.attenuation for b in network])
    if len(caps) != len(initial) or np.any(initial > caps) or np.any(caps <= 0):
        raise ValueError("Attenuation caps must contain the initial network")
    objective = network_objective(hardware, healthy, changed, noise_variance,
                                  reference_index, target_aggregation)
    initial_value = objective(initial)[0]
    if not np.isfinite(initial_value):
        raise ValueError("A prescribed region has zero initial network voltage energy")
    fit = minimize(objective, initial, jac=True, method="SLSQP",
                   bounds=list(zip(np.zeros(len(caps)), caps)),
                   constraints={"type": "eq", "fun": lambda c: constraint @ c,
                                "jac": lambda c: constraint},
                   options={"ftol": 1e-9, "maxiter": 400})
    if not fit.success:
        raise RuntimeError(f"Network optimization failed: {fit.message}")
    c = fit.x.copy()
    tolerance = np.sqrt(np.finfo(float).eps)
    c[c < tolerance] = 0.
    c[caps - c < tolerance] = caps[caps - c < tolerance]
    free = (c > 0) & (c < caps)
    c[free] -= np.linalg.lstsq(constraint[:, free], constraint @ c, rcond=tolerance)[0]
    if np.any((c < 0) | (c > caps)):
        raise RuntimeError("Restoring active bounds violated physical coefficient limits")
    parts = np.split(c, np.cumsum([len(b.attenuation) for b in network])[:-1])
    fitted = [replace(b, attenuation=part) for b, part in zip(network, parts)]
    return fitted, {"iterations": int(fit.nit),
                    "initial_ratio": float(np.exp(-initial_value)),
                    "fitted_ratio": float(np.exp(-objective(c)[0])),
                    "null_constraint_max": float(np.max(np.abs(constraint @ c), initial=0.))}
