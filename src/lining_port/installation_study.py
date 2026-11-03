"""Prepare fixed-installation loads, strains and response projections."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2

from .fe import RingParameters, recover_gauge_strain, solve_ring




def orth(columns, rtol=None):
    norms = np.linalg.norm(columns, axis=0)
    scaled = np.divide(columns, norms, out=np.zeros_like(columns), where=norms > 0)
    u, s, _ = np.linalg.svd(scaled, full_matrices=False)
    tolerance = np.finfo(float).eps * max(scaled.shape) if rtol is None else rtol
    return u[:, s > tolerance * s[0]]


def detector(reference, nuisance, targets):
    """Fixed nuisance removal followed by the design-target response subspace."""
    # Finite-difference sensitivities lose digits through subtraction. Do not
    # amplify the roundoff direction of a mechanically dependent load mode.
    background = orth(nuisance, rtol=np.sqrt(np.finfo(float).eps))
    retained = targets - background @ (background.T @ targets)
    retained -= background @ (background.T @ retained)
    u, s, _ = np.linalg.svd(retained, full_matrices=False)
    tolerance = np.sqrt(np.finfo(float).eps) * np.linalg.norm(targets, ord=2)
    signal = u[:, s > tolerance]
    signal = np.linalg.qr(residual(signal, background), mode="reduced")[0]
    return reference, background, signal


def residual(values, background):
    return values - background @ (background.T @ values)


def threshold_and_rates(calibration, healthy, softened, rank, sigma, quantile):
    """Expected rates for independent Gaussian ADC errors after filtering."""
    noncentrality = (np.asarray(calibration) / sigma)**2
    upper = float(np.max(rank + noncentrality + 20 * np.sqrt(2 * (rank + 2 * noncentrality))))
    q = brentq(lambda t: np.mean(ncx2.cdf(t, rank, noncentrality)) - quantile, 0, upper)
    return {"threshold_v": float(sigma * np.sqrt(q)), "score_rank": rank,
            "false_positive_rate": float(np.mean(ncx2.sf(q, rank, (np.asarray(healthy) / sigma)**2))),
            "true_positive_rate": float(np.mean(ncx2.sf(q, rank, (np.asarray(softened) / sigma)**2)))}


def make_cases(config):
    rows = []
    for split in ("calibration", "test"):
        rng = np.random.default_rng(config[f"{split}_seed"])
        for i in range(config[f"{split}_cases"]):
            load = {key: float(rng.uniform(*limits)) for key, limits in config["operating_ranges"].items()}
            rows.append({"scenario_family_id": f"{split}-{i:02d}", "split": split,
                         "params": load, "sector_center": float(rng.uniform(0, 2 * np.pi))})
    return rows


def load_coefficients(parameters):
    """Pressure coefficients of 1, cos(2 theta), sin(2 theta), in Pa."""
    p, h, phi = parameters.pressure, parameters.harmonic, parameters.phase
    return p * np.array([1., h * np.cos(2 * phi), h * np.sin(2 * phi)])


def strain_load_basis(parameters, geom, n_elements, gauge_length):
    """Gauge-strain response per Pa for the three load coefficients.

    Three admissible forward loads isolate the basis at the supplied support
    state. Linearity of the elastic model makes this valid across load amplitude
    and phase, for both healthy and softened support.
    """
    def sample(harmonic, phase):
        field = solve_ring(replace(parameters, harmonic=harmonic, phase=phase), n_elements)
        return recover_gauge_strain(field, geom["theta"], geom["z"], gauge_length)

    constant = sample(0., 0.)
    cosine = 2 * (sample(.5, 0.) - constant)
    sine = 2 * (sample(.5, np.pi / 4) - constant)
    return np.column_stack((constant, cosine, sine)) / parameters.pressure


def strain_inputs(config, geom, gauge_length, cases):
    nominal = RingParameters(**config["nominal_mechanics"])

    def sample(p):
        field = solve_ring(p, config["beam_elements"])
        return recover_gauge_strain(field, geom["theta"], geom["z"], gauge_length)

    arrays = {"nominal": sample(nominal)}
    for key, step in config["finite_difference_steps"].items():
        for sign in (-1, 1):
            arrays[f"{key}_{sign}"] = sample(replace(nominal, **{key: getattr(nominal, key) + sign * step}))
    centers = np.arange(config["design_sector_count"]) * 2 * np.pi / config["design_sector_count"]
    arrays["design_soft"] = np.stack([sample(replace(nominal, sector_center=center,
                                        reduction=config["design_reduction"])) for center in centers])
    for split in ("calibration", "test"):
        selected = [row for row in cases if row["split"] == split]
        arrays[f"{split}_healthy"] = np.stack([sample(replace(nominal, **row["params"])) for row in selected])
        if split == "test":
            arrays["test_soft"] = np.stack([[sample(replace(nominal, **row["params"],
                sector_center=row["sector_center"], reduction=delta)) for delta in config["test_reductions"]]
                for row in selected])
    return arrays
