"""Synthesize a physical wiring at nominal mechanics, then evaluate fixed wirings."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from .bridge_design import synthesize
from .bridge_network import Circuit, components, response_basis
from .fe import RingParameters, solve_ring, recover_gauge_strain
from .mechanism_study import dump, nuisance_projection
from .network import make_layout, geometry


PARAMETERS = ("pressure", "radial_modulus", "thickness", "harmonic", "phase")
MODES = ("uniform", "random", "raw_designed", "residual_designed")


def mechanical_fields(p, config):
    fields = {"healthy": solve_ring(p, config["beam_elements"])}
    steps = {}
    settings = config["mechanism"]["finite_difference_steps"]
    for key in PARAMETERS:
        step = .5 * (settings[key] if key in ("harmonic", "phase")
                     else settings["relative"] * getattr(p, key))
        steps[key] = step
        for sign in (-1, 1):
            fields[f"{key}_{sign}"] = solve_ring(replace(p, **{key: getattr(p, key) + sign * step}),
                                               config["beam_elements"])
    return fields, steps


def sample_strains(fields, layout, config):
    strains = {}
    for key, field in fields.items():
        # Gauge area coordinates follow the host thickness. Installed resistors,
        # capacitors, connections and receiver leads remain fixed throughout.
        p = field.params
        geo = geometry(layout, p.radius, p.thickness)
        strains[key] = recover_gauge_strain(field, geo["theta"], geo["z"], config["gauge_length_m"])
    return strains


def responses(strains, layout, geom, steps, circuit, codes=None):
    values = {key: response_basis(strain, layout, geom, circuit, codes) for key, strain in strains.items()}
    nuisance = np.stack([(values[f"{key}_1"] - values[f"{key}_-1"]) / (2 * steps[key])
                         for key in PARAMETERS])
    return values, nuisance


def run(config_path, manifest_path, output_path, layout_indices=None, device_path=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    device = None if device_path is None else json.loads(Path(device_path).read_text(encoding="utf-8"))
    circuit = Circuit() if device is None else Circuit(**device["circuit"])
    if device is not None:
        config["gauge_length_m"] = device["gauge_length_m"]
    # The original fixed noise reference is retained. Also save distances in V.
    sigma = config["receiver_noise_sigma"]
    nominal = RingParameters()
    design_fields, design_steps = mechanical_fields(nominal, config)
    sectors = np.arange(12) * 2 * np.pi / 12
    for i, center in enumerate(sectors):
        design_fields[f"sector_{i}"] = solve_ring(replace(nominal, sector_center=center, reduction=.45),
                                                 config["beam_elements"])
    indices = list(range(len(config["layout_seeds"]))) if layout_indices is None else layout_indices
    layouts, geometries, codes_by_layout, design_rows = {}, {}, {}, []
    saved = {}
    for li in indices:
        layout = make_layout(config["layout_seeds"][li], config["nodes"])
        geom = geometry(layout, nominal.radius, nominal.thickness)
        layouts[li], geometries[li] = layout, geom
        strains = sample_strains(design_fields, layout, config)
        values, nuisance = responses(strains, layout, geom, design_steps, circuit)
        contrasts = np.stack([values[f"sector_{i}"] - values["healthy"] for i in range(len(sectors))])
        uniform = np.ones(config["nodes"])
        random = np.random.default_rng(20260928 + li).choice([-1., 1.], config["nodes"])
        raw, raw_info = synthesize(contrasts, nuisance, [uniform, random], "raw")
        designed, designed_info = synthesize(contrasts, nuisance, [uniform, random], "residual")
        codes = np.column_stack((uniform, random, raw, designed))
        codes_by_layout[li] = codes
        design_rows.append({"layout": li, "codes": dict(zip(MODES, codes.T)),
                            "raw_search": raw_info, "residual_search": designed_info,
                            "geometry": geom["diagnostics"]})
        saved[f"codes_L{li:02d}"] = codes
        zero_parts = components(np.zeros(config["nodes"]), layout, geom, circuit)
        saved[f"capacitance_L{li:02d}"] = zero_parts["capacitance"]
        saved[f"edge_conductance_L{li:02d}"] = zero_parts["edges"]
        saved[f"port_nodes_L{li:02d}"] = geom["port_nodes"]
        saved[f"coordinates_L{li:02d}"] = geom["xy"]
        saved[f"design_contrasts_L{li:02d}"] = np.einsum("mti,ic->mtc", contrasts, codes)
        saved[f"design_nuisance_L{li:02d}"] = np.einsum("pti,ic->ptc", nuisance, codes)
        print(f"Designed L{li:02d}; accepted exchanges: "
              f"{[len(s['history']) - 1 for s in designed_info['restarts']]}", flush=True)
    dump(output / "wirings.json", design_rows)
    # Save the manufactured codes before reading any evaluation responses.
    dump(output / "design.json", {
        "circuit": asdict(circuit), "nominal_mechanics": asdict(nominal),
        "gauge_length_m": config["gauge_length_m"], "device": device,
        "design_sector_centers": sectors, "design_reduction": .45,
        "config": str(config_path), "manifest": str(manifest_path), "layouts": indices,
        "modes": MODES, "receiver_noise_sigma_v": sigma,
        "design_objective": "Mean log squared residual contrast across twelve nominal sectors",
        "search": "Best improving single rail exchange, from uniform and one seeded random wiring",
        "electrical_geometry": "Fixed at nominal installed geometry, including all nuisance perturbations",
        "initialization": "Discharged capacitors and filters; zero equilibrium preserved by the two-second reset",
        "evaluation": "All original test mechanical families; codes fixed for each known installed layout",
    })
    records = []
    test_rows = [row for row in manifest if row["split"] == "test"]
    for index, row in enumerate(test_rows):
        p = RingParameters(**row["params"], reduction=0.)
        fields, steps = mechanical_fields(p, config)
        fields["soft"] = solve_ring(replace(p, reduction=row["positive_reduction"]), config["beam_elements"])
        for li in indices:
            layout, geom, codes = layouts[li], geometries[li], codes_by_layout[li]
            strains = sample_strains(fields, layout, config)
            values, nuisance = responses(strains, layout, geom, steps, circuit, codes)
            delta = values["soft"] - values["healthy"]
            parts = components(strains["healthy"], layout, geom, circuit)
            for ci, mode in enumerate(MODES):
                projected = nuisance_projection(delta[:, ci], nuisance[:, :, ci].T, sigma)
                records.append({"family": row["scenario_family_id"], "layout": li, "mode": mode,
                                **projected,
                                "contrast_l2_v": float(np.linalg.norm(delta[:, ci])),
                                "contrast_peak_v": float(np.max(np.abs(delta[:, ci]))),
                                "tau_min_s": float(parts["tau"].min()), "tau_max_s": float(parts["tau"].max()),
                                "max_gauge_fractional_change": float(circuit.gauge_factor * max(
                                    np.max(np.abs(strains["healthy"])), np.max(np.abs(strains["soft"]))))})
            if index == 0:
                saved[f"example_healthy_L{li:02d}"] = values["healthy"]
                saved[f"example_soft_L{li:02d}"] = values["soft"]
                saved[f"example_nuisance_L{li:02d}"] = nuisance
                saved[f"example_strain_L{li:02d}"] = np.stack((strains["healthy"], strains["soft"]))
        print(f"Evaluated {row['scenario_family_id']} ({index + 1}/{len(test_rows)})", flush=True)
    dump(output / "evaluation.json", records)
    np.savez_compressed(output / "responses.npz", **saved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="inputs/mechanism_config.yaml")
    parser.add_argument("--manifest", default="inputs/mechanism_manifest.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layouts", nargs="+", type=int)
    parser.add_argument("--device", help="Device dimensions and circuit parameter specification")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.config, args.manifest, args.output, args.layouts, args.device)


if __name__ == "__main__":
    main()
