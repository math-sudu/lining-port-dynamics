"""Evaluate existing wirings with their nominal nuisance projector held fixed."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import ncx2
from threadpoolctl import threadpool_limits
import yaml

from lining_port.bridge_network import Circuit, response_basis
from lining_port.fe import RingParameters, solve_ring, recover_gauge_strain
from lining_port.network import make_layout, geometry


ROOT = Path(__file__).resolve().parents[1]


def basis(columns):
    norms = np.linalg.norm(columns, axis=0)
    scaled = np.divide(columns, norms, out=np.zeros_like(columns), where=norms > 0)
    left, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    keep = singular > np.finfo(float).eps * max(scaled.shape) * singular[0]
    return left[:, keep]


def main():
    config = yaml.safe_load((ROOT / "inputs/mechanism_config.yaml").read_text())
    design = json.loads((ROOT / "results/device-network-01/design.json").read_text())
    manifest = json.loads((ROOT / "inputs/mechanism_manifest.json").read_text())
    rows = json.loads((ROOT / "results/device-network-01/evaluation.json").read_text())
    original = {(row["family"], row["layout"], row["mode"]): row for row in rows}
    saved = np.load(ROOT / "results/device-network-01/responses.npz")
    circuit = Circuit(**design["circuit"])
    nominal = RingParameters()
    nominal_field = solve_ring(nominal, config["beam_elements"])
    setups = {}
    for li in design["layouts"]:
        layout = make_layout(config["layout_seeds"][li], config["nodes"])
        geom = geometry(layout, nominal.radius, nominal.thickness)
        code = saved[f"codes_L{li:02d}"]
        nuisance = saved[f"design_nuisance_L{li:02d}"]
        projectors = [basis(nuisance[:, :, ci].T) for ci in range(len(design["modes"]))]
        strain = recover_gauge_strain(nominal_field, geom["theta"], geom["z"], design["gauge_length_m"])
        baseline = response_basis(strain, layout, geom, circuit, code)
        setups[li] = (layout, geom, code, projectors, baseline)
    output = []
    test_rows = [row for row in manifest if row["split"] in ("validation", "test")]
    for ri, case in enumerate(test_rows):
        p = RingParameters(**case["params"], reduction=0.)
        fields = [solve_ring(p, config["beam_elements"]),
                  solve_ring(replace(p, reduction=case["positive_reduction"]), config["beam_elements"])]
        for li, (layout, geom, code, projectors, baseline) in setups.items():
            sampling = geometry(layout, p.radius, p.thickness)
            records = [response_basis(recover_gauge_strain(
                field, sampling["theta"], sampling["z"], design["gauge_length_m"]),
                layout, geom, circuit, code) for field in fields]
            for ci, (mode, u) in enumerate(zip(design["modes"], projectors)):
                delta = records[1][:, ci] - records[0][:, ci]
                healthy_drift = records[0][:, ci] - baseline[:, ci]
                target = delta - u @ (u.T @ delta)
                drift = healthy_drift - u @ (u.T @ healthy_drift)
                out = {"family": case["scenario_family_id"], "split": case["split"], "layout": li, "mode": mode,
                       "frozen_residual_l2_v": float(np.linalg.norm(target)),
                       "healthy_drift_l2_v": float(np.linalg.norm(drift)),
                       "softened_deviation_l2_v": float(np.linalg.norm(drift + target)),
                       "projector_rank": int(u.shape[1])}
                if case["split"] == "test":
                    ref = original[out["family"], li, mode]
                    out["local_residual_l2_v"] = ref["residual_distance"] * design["receiver_noise_sigma_v"]
                    if not np.isclose(np.linalg.norm(delta), ref["contrast_l2_v"], rtol=1e-7, atol=1e-12):
                        raise ValueError("Recomputed records do not reproduce the original raw contrasts")
                output.append(out)
        print(f"Fixed calibration: {ri + 1}/{len(test_rows)}", flush=True)
    indexed = {(r["family"], r["layout"], r["mode"]): r for r in output if r["split"] == "test"}
    pairs = sorted({(r["family"], r["layout"]) for r in output if r["split"] == "test"})
    ratios = [indexed[f, l, "residual_designed"]["frozen_residual_l2_v"] /
              indexed[f, l, "uniform"]["frozen_residual_l2_v"] for f, l in pairs]
    drift_ratios = [indexed[f, l, "residual_designed"]["healthy_drift_l2_v"] /
                    indexed[f, l, "residual_designed"]["frozen_residual_l2_v"] for f, l in pairs]
    summary = {"pairs": len(pairs), "median_frozen_ratio": float(np.median(ratios)),
               "improved_pairs": int(np.count_nonzero(np.array(ratios) > 1)),
               "median_drift_to_target": float(np.median(drift_ratios)),
               "drift_exceeds_target_pairs": int(np.count_nonzero(np.array(drift_ratios) > 1))}
    destination = ROOT / "results/fixed-calibration"
    destination.mkdir(exist_ok=True)
    (destination / "evaluation.json").write_text(json.dumps(output, indent=2) + "\n")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    noise_results = []
    for sigma in (0., .1e-6, 1e-6, 10e-6):
        for mode in design["modes"]:
            for li in design["layouts"]:
                validation = [r for r in output if r["split"] == "validation" and r["layout"] == li and r["mode"] == mode]
                tests = [r for r in output if r["split"] == "test" and r["layout"] == li and r["mode"] == mode]
                v = np.array([r["healthy_drift_l2_v"] for r in validation])
                h = np.array([r["healthy_drift_l2_v"] for r in tests])
                s = np.array([r["softened_deviation_l2_v"] for r in tests])
                if sigma == 0:
                    threshold = float(np.quantile(v, .95, method="higher"))
                    false_positive, true_positive = np.mean(h > threshold), np.mean(s > threshold)
                else:
                    dof = 1600 - validation[0]["projector_rank"]
                    noncentrality = (v / sigma) ** 2
                    upper = np.max(dof + noncentrality + 20 * np.sqrt(2 * (dof + 2 * noncentrality)))
                    quantile = brentq(lambda t: np.mean(ncx2.cdf(t, dof, noncentrality)) - .95, 0., upper)
                    threshold = float(sigma * np.sqrt(quantile))
                    false_positive = float(np.mean(ncx2.sf(quantile, dof, (h / sigma) ** 2)))
                    true_positive = float(np.mean(ncx2.sf(quantile, dof, (s / sigma) ** 2)))
                noise_results.append({"sample_noise_sigma_v": sigma, "mode": mode, "layout": li,
                                      "threshold_l2_v": threshold, "test_false_positive_rate": float(false_positive),
                                      "test_true_positive_rate": float(true_positive)})
    (destination / "noise.json").write_text(json.dumps(noise_results, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    for sigma in (0., .1e-6, 1e-6, 10e-6):
        for mode in ("uniform", "residual_designed"):
            selected = [r for r in noise_results if r["sample_noise_sigma_v"] == sigma and r["mode"] == mode]
            print(sigma, mode, "FPR", np.mean([r["test_false_positive_rate"] for r in selected]),
                  "TPR", np.mean([r["test_true_positive_rate"] for r in selected]))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
