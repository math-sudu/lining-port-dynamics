"""Generate the nominal strain-field illustration for a saved wiring study."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import yaml

from .continuum import recover_continuum_gauge, solve_continuum
from .fe import RingParameters, recover_gauge_strain, solve_ring


def generate(study: Path):
    design = json.loads((study / "design.json").read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(design["config"]).read_text(encoding="utf-8"))
    nominal = RingParameters(**design["nominal_mechanics"])
    softened = replace(nominal, reduction=design["design_reduction"],
                       sector_center=design["design_sector_centers"][0])
    theta = (np.arange(512) + .5) * 2 * np.pi / 512
    depth = np.linspace(-.45, .45, 9) * nominal.thickness
    positions = np.meshgrid(theta, depth)
    gauge_length = design["gauge_length_m"]
    beam_elements = config["beam_elements"]
    beam_name = f"beam_{beam_elements}"
    arrays = {
        "theta": theta, "depth": depth,
        "models": np.array([beam_name, "continuum_512x8"]),
        "gauge_length_m": gauge_length, "beam_elements": beam_elements,
    }
    with threadpool_limits(limits=1):
        for label, params in enumerate((nominal, softened)):
            beam = solve_ring(params, beam_elements)
            continuum = solve_continuum(params, 512, 8, .2)
            arrays[f"{beam_name}_strain_y{label}"] = recover_gauge_strain(
                beam, *positions, gauge_length)
            arrays[f"continuum_512x8_strain_y{label}"] = recover_continuum_gauge(
                continuum, *positions, gauge_length)
    target = study / "strain_fields.npz"
    np.savez_compressed(target, **arrays)
    print(f"Wrote {target}: {beam_name}, gauge length {gauge_length:g} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=Path("results/device-network-01"))
    generate(parser.parse_args().study)
