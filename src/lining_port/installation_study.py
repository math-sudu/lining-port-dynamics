"""Design and evaluate fixed installations under operating loads and local support loss."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2
from threadpoolctl import threadpool_limits

from .bridge_design import record_scores, synthesize
from .bridge_network import Circuit, components, response_basis
from .fe import RingParameters, recover_gauge_strain, solve_ring
from .installation import installed_network
from .mechanism_study import dump


MODES = ("uniform", "random", "raw_designed", "residual_designed")


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


def run(config_path, output_path, mechanical=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    device = json.loads(Path("inputs/device_node.json").read_text(encoding="utf-8"))
    circuit = Circuit(**device["circuit"])
    nominal = RingParameters(**config["nominal_mechanics"])
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    cases = make_cases(config)
    dump(output / "cases.json", cases)
    _, geom = installed_network(config["layout_seeds"][0], nominal.radius, nominal.thickness, config["depth_m"])
    strains = strain_inputs(config, geom, device["gauge_length_m"], cases)
    np.savez_compressed(output / "strains.npz", **strains)
    modes = MODES + (("layer_difference", "strain_matched") if mechanical else ())
    saved, wirings, records, scores = {}, [], [], []
    for li, seed in enumerate(config["layout_seeds"]):
        layout, geom = installed_network(seed, nominal.radius, nominal.thickness, config["depth_m"])

        def response(e, code=None):
            return response_basis(e, layout, geom, circuit, code, reset_duration=config["reset_duration_s"])

        baseline = response(strains["nominal"])
        nuisance = np.stack([(response(strains[f"{key}_1"]) - response(strains[f"{key}_-1"])) / (2 * step)
                            for key, step in config["finite_difference_steps"].items()])
        contrasts = np.stack([response(e) - baseline for e in strains["design_soft"]])
        uniform = np.ones(64)
        random = np.random.default_rng(20260928 + li).choice([-1., 1.], 64)
        raw, raw_info = synthesize(contrasts, nuisance, [uniform, random], "raw")
        designed, info = synthesize(contrasts, nuisance, [uniform, random], "residual")
        codes = np.column_stack((uniform, random, raw, designed))
        mechanical_choice = None
        if mechanical:
            layer = np.tile([1., -1.], 32)
            candidates = np.where(strains["design_soft"] - strains["nominal"] >= 0, 1., -1.)
            candidate_scores = [record_scores(np.einsum("mti,i->tm", contrasts, b),
                                               np.einsum("pti,i->tp", nuisance, b), "residual") for b in candidates]
            mechanical_choice = int(np.argmax(candidate_scores))
            codes = np.column_stack((codes, layer, candidates[mechanical_choice]))
        jac = np.einsum("pti,ic->ptc", nuisance, codes)
        targets = np.einsum("mti,ic->mtc", contrasts, codes)
        references = baseline @ codes
        detectors = [detector(references[:, ci], jac[:, :, ci].T, targets[:, :, ci].T)
                     for ci in range(len(modes))]
        wirings.append({"layout": li, "codes": dict(zip(modes, codes.T)), "raw_search": raw_info,
                        "residual_search": info, "strain_matched_sector": mechanical_choice,
                        "geometry": geom["diagnostics"]})
        parts = components(np.zeros(64), layout, geom, circuit)
        suffix = f"_L{li:02d}"
        saved.update({key + suffix: value for key, value in {
            "codes": codes, "capacitance": parts["capacitance"], "edge_conductance": parts["edges"],
            "port_nodes": geom["port_nodes"], "coordinates": geom["xy"],
            "design_contrasts": targets, "design_nuisance": jac,
            "nominal_reference": references, "route_length": geom["route_length"]}.items()})
        for split in ("calibration", "test"):
            selected = [row for row in cases if row["split"] == split]
            for ci_case, (case, eps) in enumerate(zip(selected, strains[f"{split}_healthy"])):
                healthy = response(eps, codes)
                for mi, mode in enumerate(modes):
                    ref, background, signal = detectors[mi]
                    scores.append({"family": case["scenario_family_id"], "split": split, "layout": li,
                                   "mode": mode, "reduction": 0., "rank": signal.shape[1],
                                   "score_v": float(np.linalg.norm(signal.T @ (healthy[:, mi] - ref)))})
                if split == "calibration":
                    continue
                for di, delta in enumerate(config["test_reductions"]):
                    soft = response(strains["test_soft"][ci_case, di], codes)
                    for mi, mode in enumerate(modes):
                        ref, background, signal = detectors[mi]
                        change = soft[:, mi] - healthy[:, mi]
                        scores.append({"family": case["scenario_family_id"], "split": split, "layout": li,
                                       "mode": mode, "reduction": delta, "rank": signal.shape[1],
                                       "score_v": float(np.linalg.norm(signal.T @ (soft[:, mi] - ref)))})
                        if delta == config["design_reduction"]:
                            raw_norm = float(np.linalg.norm(change))
                            residual_norm = float(np.linalg.norm(residual(change, background)))
                            records.append({"family": case["scenario_family_id"], "layout": li, "mode": mode,
                                "contrast_l2_v": raw_norm, "residual_l2_v": residual_norm,
                                "distance": raw_norm / config["sample_noise_sigma_v"],
                                "residual_distance": residual_norm / config["sample_noise_sigma_v"]})
                            if ci_case == 0:
                                saved["example_healthy" + suffix] = healthy
                                saved["example_soft" + suffix] = soft
                                saved["example_nuisance" + suffix] = jac
                                saved["example_strain" + suffix] = np.stack((eps, strains["test_soft"][ci_case, di]))
        print(f"Installation {li + 1}/{len(config['layout_seeds'])}: fixed wiring and calibration evaluated", flush=True)
    design = {"config": str(config_path), "manifest": str(output / "cases.json"), "circuit": asdict(circuit),
              "nominal_mechanics": asdict(nominal), "gauge_length_m": device["gauge_length_m"], "device": device,
              "design_sector_centers": np.arange(12) * np.pi / 6, "design_reduction": config["design_reduction"],
              "layouts": list(range(len(config["layout_seeds"]))), "modes": modes,
              "receiver_noise_sigma_v": config["sample_noise_sigma_v"], "normal_parameters": config["operating_parameters"],
              "calibration": "Nominal reference, nuisance projection and target subspace fixed before independent healthy calibration",
              "cycle_s": config["reset_duration_s"] + config["probe_duration_s"], "geometry": geom["diagnostics"]}
    dump(output / "design.json", design)
    dump(output / "wirings.json", wirings)
    dump(output / "evaluation.json", records)
    dump(output / "scores.json", scores)
    np.savez_compressed(output / "responses.npz", **saved)
    summarize(output, config, design, records, scores)


def summarize(output, config, design, records, scores):
    indexed = {(r["family"], r["layout"], r["mode"]): r for r in records}
    pairs = sorted({(r["family"], r["layout"]) for r in records})
    comparisons, medians, rates = {}, {}, []
    for mode in design["modes"]:
        medians[mode] = {key: float(np.median([r[key] for r in records if r["mode"] == mode]))
                         for key in ("contrast_l2_v", "residual_l2_v")}
        if mode != "residual_designed":
            ratios = np.array([indexed[f, l, "residual_designed"]["residual_l2_v"] /
                               indexed[f, l, mode]["residual_l2_v"] for f, l in pairs])
            comparisons[mode] = {"median_ratio": float(np.median(ratios)), "improved_pairs": int((ratios > 1).sum())}
        for li in design["layouts"]:
            selected = [r for r in scores if r["layout"] == li and r["mode"] == mode]
            calibration = [r["score_v"] for r in selected if r["split"] == "calibration"]
            healthy = [r["score_v"] for r in selected if r["split"] == "test" and r["reduction"] == 0]
            for delta in config["test_reductions"]:
                soft = [r["score_v"] for r in selected if r["reduction"] == delta]
                result = threshold_and_rates(calibration, healthy, soft, selected[0]["rank"],
                                             config["sample_noise_sigma_v"], config["healthy_quantile"])
                rates.append({"mode": mode, "layout": li, "reduction": delta, **result})
    dump(output / "detection.json", rates)
    dump(output / "summary.json", {"pairs": len(pairs), "medians": medians, "comparisons": comparisons})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="inputs/installation_task.json")
    parser.add_argument("--output", default="results/fixed-installation")
    parser.add_argument("--mechanical", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.config, args.output, args.mechanical)


if __name__ == "__main__":
    main()
