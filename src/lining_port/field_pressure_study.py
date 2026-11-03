"""Measured Shiziyang pressure profiles as inputs to reference-space design.

The two measured sections supply spatial loads, not strain-validation targets.
The existing representative supported ring, fixed gauge layout and design
increments are retained. No field stiffness or damage parameter is fitted.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

from .fe import (FEResult, RingParameters, _element_matrices, _foundation,
                 _geometry, assemble_frame, recover_gauge_strain)
from .passive_bridge import response_spaces, target_null_weight
from .passive_robustness_study import source_data


def measured_profiles(path):
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    stations = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    angles = (np.pi / 2 - np.arange(8) * np.pi / 4) % (2 * np.pi)
    profiles = []
    for ring in (75, 170):
        rows = table[table["ring"] == ring]
        pressure = np.array([rows[rows["station"] == station]["pressure_MPa"].item()
                             for station in stations]) * 1e6
        profiles.append(pressure)
    return angles, np.asarray(profiles)


def pressure_response(params, angles, pressure, gauge_theta, gauge_depth, elements=128):
    """Periodic linear interpolation of measured normal pressure onto ring nodes."""
    theta, xy, radial, _ = _geometry(params, elements)
    ring, support, _ = assemble_frame(params, elements)
    loads = np.zeros((elements, 3))
    nodal_pressure = np.interp(theta, angles, pressure, period=2 * np.pi)
    tributary = 2 * np.pi * params.radius / elements
    loads[:, :2] = -nodal_pressure[:, None] * params.width * tributary * radial
    displacement = spsolve(ring + support, loads.ravel()).reshape(elements, 3)
    forces = []
    for i in range(elements):
        j = (i + 1) % elements
        _, transform, local = _element_matrices(params, xy[i], xy[j])
        forces.append(local @ transform @ np.r_[displacement[i], displacement[j]])
    kr, kt = _foundation(params, theta)
    result = FEResult(params, elements, theta, xy, displacement,
                      -(support @ displacement.ravel()).reshape(elements, 3),
                      np.asarray(forces), loads, kr, kt, {})
    return recover_gauge_strain(result, gauge_theta, gauge_depth, .0002)


def selected_rows(reference, design, groups):
    rows = np.zeros((len(groups), reference.shape[0]))
    for q, group in enumerate(groups):
        weights, _ = target_null_weight(reference[group], design[:, group].T)
        rows[q, group] = weights / np.linalg.norm(weights)
    return rows


def run():
    config = json.loads(Path("inputs/installation_task.json").read_text())
    params = RingParameters(**config["nominal_mechanics"])
    data = next(d for d in source_data() if d["name"] == "ring")
    theta, depth = data["theta"][0], data["depth"][0]
    angles, pressure = measured_profiles("data/field-monitoring/shiziyang_sections.csv")
    healthy = np.array([pressure_response(params, angles, p, theta, depth) for p in pressure])
    original = data["basis"][0]
    enriched = np.column_stack((original, healthy.T))
    groups = np.array_split(np.argsort(theta), 4)
    design = (data["design"][:, :, 0] - data["calibration"][:, 0]).reshape(-1, len(theta))
    target = []
    for sector in range(12):
        changed = replace(params, reduction=.1, sector_center=sector * np.pi / 6)
        target.append([pressure_response(changed, angles, p, theta, depth) - h
                       for p, h in zip(pressure, healthy)])
    target = np.asarray(target)
    rows, results = {}, []
    target_energy = float(np.mean(np.sum(target**2, axis=-1)))
    for label, reference in (("analytic_load_family", original), ("measured_pressure_enrichment", enriched)):
        Q, null = response_spaces(reference)
        U = selected_rows(reference, design, groups)
        selected = target @ U.T
        projected_energy = float(np.mean(np.sum((target @ null)**2, axis=-1)))
        channel_energy = float(np.mean(np.sum(selected**2, axis=-1)))
        rows[label] = U
        result = {"reference": label, "rank": Q.shape[1],
                        "global_available_fraction": projected_energy / target_energy,
                        "channel_capture_fraction_of_total": channel_energy / target_energy,
                        "channel_capture_fraction_of_available": channel_energy / projected_energy,
                        "target_norm_rms_microstrain": float(1e6 * np.sqrt(channel_energy)),
                        "regional_target_norm_rms_microstrain": (1e6 * np.sqrt(np.mean(np.sum(selected**2, axis=-1), axis=1))).tolist()}
        if label == "analytic_load_family":
            result["healthy_profile_readout_norm_microstrain"] = (1e6 * np.linalg.norm(healthy @ U.T, axis=-1)).tolist()
        results.append(result)
    summary = {
        "pressure_sections": [75, 170], "station_order": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
        "pressure_MPa": (pressure / 1e6).tolist(), "gauge_layout_seed": 7100,
        "model_source": "inputs/installation_task.json nominal_mechanics; no field-parameter fit",
        "reference": "Uniform, cosine-two and sine-two pressure responses, augmented with responses "
                     "to the two measured spatial profiles. Periodic linear interpolation, all eight readings retained.",
        "direction_design": "One unit row per contiguous group; leading projected target singular vector, "
                            "using the existing 12-region, 32-load, 45-percent support-loss design increments.",
        "evaluation": "Calculated increments under each measured profile and 10-percent local support loss "
                      "in each of the 12 existing sectors; field pressures are load inputs, not independent stress validation.",
        "results": results,
    }
    output = Path("results/field-bridges")
    (output / "pressure_reference.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output / "pressure_reference.npz", pressure_station_angles=angles, pressure_Pa=pressure,
                        gauge_theta=theta, gauge_depth=depth, healthy_strain=healthy, target_increment=target,
                        original_reference=original, enriched_reference=enriched, **rows)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
