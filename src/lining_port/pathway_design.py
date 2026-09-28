"""Compare strain-time polarities with matched healthy dynamics and nuisance spaces."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from .fe import RingParameters, solve_ring, recover_gauge_strain
from .network import make_layout, geometry, node_parameters, simulate
from .pathway_attribution import decompose_pair
from .mechanism_study import dump, signature_metrics


POLARITIES = {"same_sign": 0.5, "opposite_sign": -0.5}
PARAMETERS = ("pressure", "radial_modulus", "thickness", "harmonic", "phase")


def residual(vectors, jacobian):
    """Project every column using one nuisance space, including its numerical rank."""
    norms = np.linalg.norm(jacobian, axis=0)
    scaled = np.divide(jacobian, norms, out=np.zeros_like(jacobian), where=norms > 0)
    coef, _, rank, _ = np.linalg.lstsq(scaled, vectors, rcond=None)
    return vectors - scaled @ coef, int(rank)


def cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator else None


def projected_pathways(gain, time, jacobian, frozen_jacobian, sigma):
    """Separate a time intervention from the accompanying nuisance-space change.

    The time and gain vectors share the actual model's projector. The frozen
    comparator has its own nuisance projector and constant healthy node times.
    """
    projected, rank = residual(np.column_stack((gain, time)), jacobian)
    gain_q, time_q = projected.T / sigma
    frozen_gain, frozen_rank = residual(gain, frozen_jacobian)
    frozen_gain = frozen_gain / sigma
    gain_sq = float(gain_q @ gain_q)
    time_sq = float(time_q @ time_q)
    cross = float(2 * gain_q @ time_q)
    frozen_sq = float(frozen_gain @ frozen_gain)
    total_sq = float((gain_q + time_q) @ (gain_q + time_q))
    return {
        "distance": float(np.sqrt(total_sq)),
        "frozen_distance": float(np.sqrt(frozen_sq)),
        "gain_in_actual_space": float(np.sqrt(gain_sq)),
        "time_in_actual_space": float(np.sqrt(time_sq)),
        "projected_cosine": cosine(gain_q, time_q),
        "time_effect_squared": time_sq + cross,
        "nuisance_space_effect_squared": gain_sq - frozen_sq,
        "total_effect_squared": total_sq - frozen_sq,
        "rank": rank,
        "frozen_rank": frozen_rank,
    }


def knowledge_spaces(jacobian, p):
    pressure, support, thickness, harmonic, phase = jacobian.T
    youngs = -(p.pressure * pressure + p.radial_modulus * support) / p.youngs
    return {
        "all_unknown": jacobian,
        "structure_known": np.column_stack((pressure, harmonic, phase)),
        "load_known": np.column_stack((youngs, support, thickness)),
    }


def run(config_path, manifest_path, output_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    layouts = [make_layout(seed, config["nodes"]) for seed in config["layout_seeds"]]
    target = set(config["layout_domains"]["target_test"])
    sigma = config["receiver_noise_sigma"]
    gauge_length = config["gauge_length_m"]
    elements = config["beam_elements"]
    kappa = 0.3
    records, projections, examples = [], [], {}
    test_rows = [row for row in manifest if row["split"] == "test"]
    for index, row in enumerate(test_rows):
        p = RingParameters(**row["params"], reduction=0.0)
        pair = [solve_ring(replace(p, reduction=y * row["positive_reduction"]), elements)
                for y in (0, 1)]
        perturbed = {}
        for key in PARAMETERS:
            settings = config["mechanism"]["finite_difference_steps"]
            step = 0.5 * (settings[key] if key in ("harmonic", "phase")
                          else settings["relative"] * getattr(p, key))
            perturbed[key] = (step, [
                solve_ring(replace(p, **{key: getattr(p, key) + sign * step}), elements)
                for sign in (-1, 1)
            ])
        for li, layout in enumerate(layouts):
            geom = geometry(layout, p.radius, p.thickness)
            strains = [recover_gauge_strain(fe, geom["theta"], geom["z"], gauge_length)
                       for fe in pair]
            displaced = {}
            topology_changed = []
            if li in target:
                for key, (step, fields) in perturbed.items():
                    values = []
                    for fe in fields:
                        other_geom = geometry(layout, fe.params.radius, fe.params.thickness)
                        if (not np.array_equal(other_geom["W"] > 0, geom["W"] > 0)
                                or not np.array_equal(other_geom["port_nodes"], geom["port_nodes"])):
                            topology_changed.append(key)
                        strain = recover_gauge_strain(fe, other_geom["theta"], other_geom["z"], gauge_length)
                        values.append((strain, other_geom))
                    displaced[key] = (step, values)
            for polarity, alpha in POLARITIES.items():
                values = decompose_pair(*strains, geom, layout, kappa=kappa, alpha=alpha)
                gain, time = values["gain"].ravel(), values["time"].ravel()
                records.append({
                    "family": row["scenario_family_id"], "layout": li,
                    "polarity": polarity, "alpha": alpha,
                    "gain_distance": float(np.linalg.norm(gain) / sigma),
                    "time_distance": float(np.linalg.norm(time) / sigma),
                    "gain_time_cosine": cosine(gain, time),
                    "full": signature_metrics(values["healthy"], values["soft"], sigma),
                    "frozen": signature_metrics(values["healthy"], values["frozen_time_soft"], sigma),
                })
                if index == 0 and li == min(target):
                    for key, value in values.items():
                        examples[f"{polarity}_{key}"] = value
                if li not in target:
                    continue
                frozen_layout = dict(layout, tau0=node_parameters(strains[0], layout, "P4", alpha)["tau"])
                columns, frozen_columns = [], []
                for key in PARAMETERS:
                    step, perturbations = displaced[key]
                    dynamic, frozen = [], []
                    for strain, other_geom in perturbations:
                        dynamic.append(simulate(strain, other_geom, layout, family="P4", alpha=alpha,
                                                kappa=kappa, method="expm", noise_sigma=0)["z_clean"].ravel())
                        frozen.append(simulate(strain, other_geom, frozen_layout, family="P3",
                                               kappa=kappa, method="expm", noise_sigma=0)["z_clean"].ravel())
                    columns.append((dynamic[1] - dynamic[0]) / (2 * step))
                    frozen_columns.append((frozen[1] - frozen[0]) / (2 * step))
                spaces = knowledge_spaces(np.column_stack(columns), p)
                frozen_spaces = knowledge_spaces(np.column_stack(frozen_columns), p)
                for condition, jacobian in spaces.items():
                    projections.append({
                        "family": row["scenario_family_id"], "layout": li,
                        "polarity": polarity, "condition": condition,
                        "topology_changed_parameters": sorted(set(topology_changed)),
                        **projected_pathways(gain, time, jacobian, frozen_spaces[condition], sigma),
                    })
        print(f"Completed {row['scenario_family_id']} ({index + 1}/{len(test_rows)})", flush=True)
    dump(output / "pathways.json", records)
    dump(output / "nuisance_pathways.json", projections)
    np.savez_compressed(output / "response_examples.npz", **examples)
    dump(output / "design.json", {
        "config": str(config_path), "manifest": str(manifest_path),
        "polarities": POLARITIES, "kappa": kappa,
        "selection": "Opposite strain-time polarity follows the signed transfer-function interaction; no coefficient search.",
        "population": "Original held-out mechanical families on all original layouts; nuisance attribution uses original target layouts.",
        "frozen_control": "Times equal the actual healthy times and remain fixed during support and nuisance perturbations.",
        "finite_difference_step_factor": 0.5,
        "observation_permission": "Conditional model-based encoding design; no paired template is supplied to an external classifier.",
        "pairs_per_polarity": len(records) // len(POLARITIES),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.config, args.manifest, args.output)


if __name__ == "__main__":
    main()
