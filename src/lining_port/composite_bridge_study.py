"""Mechanics-matched bridge construction and forward response evaluation."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, ncx2
from threadpoolctl import threadpool_limits

from .acquisition_comparison import bridge_netlists, bridge_voltage, divider_voltage, scan_operator
from .bridge_network import Circuit
from .fe import RingParameters, recover_gauge_strain, solve_ring
from .installation import installed_network
from .installation_study import load_coefficients, strain_load_basis
from .mechanism_study import dump


def low_response_load(basis, nodes, circuit, pressure, harmonic_max):
    """Find a weak forward load at a fixed sector and pressure.

    Cartesian harmonic coefficients occupy a disk. The exact resistor fractions
    and their derivatives are used in this two-variable minimization.
    """
    arm_basis = circuit.resistance * circuit.gauge_factor * basis[nodes].sum(axis=2)
    arm_nominal = nodes.shape[-1] * circuit.resistance

    def objective(x):
        coefficients = pressure * np.r_[1., x]
        arms = arm_nominal + arm_basis @ coefficients
        left, right = arms[:, 0] + arms[:, 1], arms[:, 2] + arms[:, 3]
        voltage = 2 * circuit.voltage * (arms[:, 1] / left - arms[:, 3] / right)
        derivative = 2 * circuit.voltage * np.column_stack((
            -arms[:, 1] / left**2, arms[:, 0] / left**2,
            arms[:, 3] / right**2, -arms[:, 2] / right**2))
        jacobian = np.einsum("ca,caj->cj", derivative, arm_basis[:, :, 1:]) * pressure
        return float(voltage @ voltage * 1e12), 2e12 * voltage @ jacobian

    result = minimize(objective, np.zeros(2), jac=True, method="SLSQP",
                      bounds=[(-harmonic_max, harmonic_max)] * 2,
                      constraints=[{"type": "ineq", "fun": lambda x: harmonic_max**2 - x @ x,
                                    "jac": lambda x: -2 * x}],
                      options={"ftol": 1e-12, "maxiter": 100})
    if not result.success:
        raise RuntimeError(f"Weak-load search failed: {result.message}")
    coefficients = pressure * np.r_[1., result.x]
    parameters = {"pressure": pressure, "harmonic": float(np.linalg.norm(result.x)),
                  "phase": float((np.arctan2(result.x[1], result.x[0]) / 2) % np.pi)}
    return coefficients, parameters


def detection(voltage, sigma, false_positive_rate):
    rank = voltage.shape[-1]
    threshold = chi2.isf(false_positive_rate, rank)
    probability = ncx2.sf(threshold, rank, np.sum((voltage / sigma)**2, axis=-1))
    return {"threshold_v": float(sigma * np.sqrt(threshold)),
            "mean_probability": float(np.mean(probability)),
            "minimum_probability": float(np.min(probability))}


def main():
    source, output = Path("results/fixed-installation"), Path("results/composite-bridges")
    config = json.loads(Path("inputs/installation_task.json").read_text())
    device = json.loads(Path("inputs/device_node.json").read_text())
    circuit = Circuit(**device["circuit"])
    nominal = RingParameters(**config["nominal_mechanics"])
    _, geom = installed_network(config["layout_seeds"][0], nominal.radius,
                                nominal.thickness, config["depth_m"])
    netlists = bridge_netlists()
    cases = json.loads((source / "cases.json").read_text())
    strains = np.load(source / "strains.npz")
    healthy = np.concatenate((strains["calibration_healthy"], strains["test_healthy"]))
    test_cases = [row for row in cases if row["split"] == "test"]
    basis = strain_load_basis(nominal, geom, config["beam_elements"], device["gauge_length_m"])
    coefficients = np.array([load_coefficients(replace(nominal, **row["params"])) for row in cases])
    reconstructed = coefficients @ basis.T
    cycle = config["reset_duration_s"] + config["probe_duration_s"]
    samples = round(cycle / config["sample_period_s"])
    sigma = config["sample_noise_sigma_v"] / np.sqrt(samples)
    rc = json.loads((source / "detection.json").read_text())
    false_positive_rate = float(np.mean([row["false_positive_rate"] for row in rc
                                        if row["mode"] == "residual_designed" and row["reduction"] == .1]))
    result = {
        "source_cases": str(source / "cases.json"), "source_strains": str(source / "strains.npz"),
        "netlists_1_based_R1_R2_R3_R4": {name: nodes + 1 for name, nodes in netlists.items()},
        "coefficient_order": ["p0", "p0*h*cos(2*phi)", "p0*h*sin(2*phi)"],
        "healthy_basis_relative_reconstruction_error": float(
            np.linalg.norm(reconstructed - healthy) / np.linalg.norm(healthy)),
        "readout": {"cycle_s": cycle, "samples_per_channel": samples,
                    "channel_mean_noise_sigma_v": sigma,
                    "noise": "Independent Gaussian sampled-voltage errors after receiver filtering",
                    "statistic": "Norm of the signed channel-mean vector",
                    "matched_false_positive_rate": false_positive_rate},
        "candidates": {},
    }
    arrays = {"healthy_basis_per_pa": basis}
    original_healthy = bridge_voltage(healthy, netlists["mechanical_bridge"], circuit)
    original_soft = bridge_voltage(strains["test_soft"], netlists["mechanical_bridge"], circuit)
    arrays.update(original_healthy_v=original_healthy, original_soft_v=original_soft)
    result["original_bridge_healthy_peak_v"] = float(abs(original_healthy).max())
    for name in ("composite4", "composite2"):
        nodes = netlists[name]
        healthy_v = bridge_voltage(healthy, nodes, circuit)
        soft_v = bridge_voltage(strains["test_soft"], nodes, circuit)
        design_v = bridge_voltage(strains["design_soft"], nodes, circuit)
        arrays.update({name + "_healthy_v": healthy_v, name + "_soft_v": soft_v,
                       name + "_design_v": design_v})
        channels, _, per_arm = nodes.shape
        weights = np.zeros((channels, 64))
        for channel, arms in enumerate(nodes):
            weights[channel, arms] = np.array([-1., 1., 1., -1.])[:, None] / per_arm
        arrays[name + "_strain_weights"] = weights
        result["candidates"][name] = candidate = {
            "physical_channels": channels, "gauges_per_arm": per_arm,
            "healthy_peak_v": float(abs(healthy_v).max()),
            "healthy_false_positive_rate": detection(healthy_v, sigma, false_positive_rate)["mean_probability"],
            "linear_load_leakage_relative_norm": float(np.linalg.norm(weights @ basis) / np.linalg.norm(basis)),
            "design_voltage_singular_values_v": np.linalg.svd(design_v, compute_uv=False),
            "antipodal_design_pair_sum_max_v": float(abs(design_v[:6] + design_v[6:]).max()),
            "nominal_excitation_energy_j": channels * (2 * circuit.voltage)**2 / (per_arm * circuit.resistance) * cycle,
            "evaluation": [],
        }
        for i, delta in enumerate(config["test_reductions"]):
            norms = np.linalg.norm(soft_v[:, i], axis=1)
            weakest = int(np.argmin(norms))
            original_change = original_soft[:, i] - original_healthy[config["calibration_cases"]:]
            candidate["evaluation"].append({
                "reduction": delta, "voltage_norm_v_min_median_max": np.percentile(norms, [0, 50, 100]),
                "original_bridge_change_norm_v_min_median_max": np.percentile(
                    np.linalg.norm(original_change, axis=1), [0, 50, 100]),
                "detection": detection(soft_v[:, i], sigma, false_positive_rate),
                "weakest_case": test_cases[weakest],
            })

    # Half-station spacing resolves the distinct bridge groups within a quadrant.
    centers = np.arange(17) * np.pi / 32
    pressure = config["operating_ranges"]["pressure"][0]
    harmonic_max = config["operating_ranges"]["harmonic"][1]
    scan, _ = scan_operator(switching=config["scan_switch_s"])
    arrays["weak_load_sector_centers_rad"] = centers
    probe_bases = []
    result["weak_loads"] = {"purpose": "Resolve weak forward responses between the random evaluation cases",
                            "reduction": .1, "sector_step_rad": float(np.pi / 32),
                            "pressure_pa": pressure, "cases": []}
    for center in centers:
        softened_basis = strain_load_basis(replace(nominal, sector_center=center, reduction=.1),
                                           geom, config["beam_elements"], device["gauge_length_m"])
        probe_bases.append(softened_basis)
        for name in ("composite4", "composite2"):
            a, load = low_response_load(softened_basis, netlists[name], circuit, pressure, harmonic_max)
            healthy_e, soft_e = basis @ a, softened_basis @ a
            voltage = bridge_voltage(soft_e, netlists[name], circuit)
            original_change = (bridge_voltage(soft_e, netlists["mechanical_bridge"], circuit)
                               - bridge_voltage(healthy_e, netlists["mechanical_bridge"], circuit))
            scan_change = scan @ (divider_voltage(soft_e, circuit) - divider_voltage(healthy_e, circuit))
            result["weak_loads"]["cases"].append({
                "mode": name, "sector_center_rad": float(center), "params": load,
                "signed_voltages_v": voltage, "voltage_norm_v": float(np.linalg.norm(voltage)),
                "detection_probability": detection(voltage, sigma, false_positive_rate)["mean_probability"],
                "gauge_strain_change_max": float(abs(soft_e - healthy_e).max()),
                "gauge_strain_change_norm": float(np.linalg.norm(soft_e - healthy_e)),
                "original_bridge_change_norm_v": float(np.linalg.norm(original_change)),
                "scan_change_record_norm_v": float(np.linalg.norm(scan_change)),
            })
    arrays["weak_load_softened_basis_per_pa"] = np.array(probe_bases)

    # Examine full-field mechanics at the smallest four-bridge output, then
    # compare the same support change under uniform pressure at that location.
    weak = min((row for row in result["weak_loads"]["cases"] if row["mode"] == "composite4"),
               key=lambda row: row["voltage_norm_v"])
    result["weak_loads"]["mechanical_cause"] = cause = []
    field_theta = np.repeat((np.arange(config["beam_elements"]) + .5) * 2 * np.pi / config["beam_elements"], 2)
    field_z = np.tile([-config["depth_m"], config["depth_m"]], config["beam_elements"])
    for load_name, load in (("weak_load", weak["params"]),
                            ("uniform_pressure", {"pressure": pressure, "harmonic": 0., "phase": 0.})):
        parameters = replace(nominal, **load, sector_center=weak["sector_center_rad"])
        fields = [solve_ring(replace(parameters, reduction=delta), config["beam_elements"]) for delta in (0., .1)]
        gauge_e = [recover_gauge_strain(field, geom["theta"], geom["z"], device["gauge_length_m"]) for field in fields]
        field_e = [recover_gauge_strain(field, field_theta, field_z, device["gauge_length_m"]) for field in fields]
        radial = np.column_stack((np.cos(fields[0].theta), np.sin(fields[0].theta)))
        radial_u = np.sum(fields[0].displacements[:, :2] * radial, axis=1)
        weight = (fields[0].radial_stiffness - fields[1].radial_stiffness) / fields[0].radial_stiffness / .1
        cause.append({"load": load_name, "params": load, "sector_center_rad": weak["sector_center_rad"],
                      "weighted_healthy_radial_displacement_rms_m": float(np.sqrt(np.sum(weight * radial_u**2) / weight.sum())),
                      "full_field_strain_change_max": float(abs(field_e[1] - field_e[0]).max()),
                      "gauge_strain_change_max": float(abs(gauge_e[1] - gauge_e[0]).max()),
                      "displacement_change_max_m": float(np.linalg.norm(
                          fields[1].displacements[:, :2] - fields[0].displacements[:, :2], axis=1).max()),
                      "signed_voltages_v": bridge_voltage(gauge_e[1], netlists["composite4"], circuit)})
        arrays[load_name + "_gauge_strains"] = np.array(gauge_e)
        arrays[load_name + "_field_strains"] = np.array(field_e)
    arrays["field_theta_rad"], arrays["field_z_m"] = field_theta, field_z
    result["weak_loads"]["direct_solve_voltage_difference_v"] = float(np.linalg.norm(
        cause[0]["signed_voltages_v"] - weak["signed_voltages_v"]))

    output.mkdir(parents=True, exist_ok=True)
    dump(output / "summary.json", result)
    np.savez_compressed(output / "responses.npz", **arrays)
    print(json.dumps({"healthy_basis_relative_error": result["healthy_basis_relative_reconstruction_error"],
                      "four_bridge_test_at_10_percent": result["candidates"]["composite4"]["evaluation"][0]["detection"],
                      "weak_load": {key: weak[key] for key in ("sector_center_rad", "params", "voltage_norm_v")}}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
