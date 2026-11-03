"""Extract the manuscript's frame circuit and original design inputs."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "representative-frame"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    folder = ROOT / "results/passive-network"
    example = json.loads((folder / "observables.json").read_text())["example"]
    records = json.loads((folder / "constrained_netlists.json").read_text())
    networks = {}
    arrays = {}
    with np.load(folder / "constrained.npz") as stored:
        for label, mode in (("scaled", "baseline"), ("constrained", "constrained")):
            record = next(r for r in records if r["structure"] == "frame"
                          and r["layout_seed"] == example["layout_seed"] and r["mode"] == mode)
            networks[label] = {k: v for k, v in record.items()
                               if k not in ("case_sources", "acquisition_config")}
            prefix = f"frame_l{record['layout']}_{mode}"
            arrays[label + "_expected_increment_v"] = stored[prefix + "_design_increment_v"]
    layout = networks["constrained"]["layout"]
    acquisition = json.loads((ROOT / "inputs/passive_readout_task.json").read_text())
    with np.load(ROOT / "results/passive-robustness/hardware.npz") as hardware, \
            np.load(ROOT / "results/frame-bridges/synthesis.npz") as mechanics:
        arrays["healthy_basis"] = hardware[
            f"frame_l{layout}_c4_{acquisition['source_condition']}_response"]
        arrays["load_coefficients"] = mechanics["r6_calibration_coefficients"]
        arrays["regional_increments"] = (
            mechanics["r6_design_changed_strain"][:, :, layout]
            - mechanics["r6_calibration_strain"][:, layout])
        arrays["regions"] = mechanics["design_patch"]
    data = {
        "description": "Representative four-channel frame: joint voltage and resource-constrained circuits with the same mechanical weights.",
        "example_region_index": example["region_index"],
        "example_region": example["region"],
        "coordinates": "healthy_basis is the commissioned effective basis; regional_increments are "
                       "the nominal forward increments used in the design. They are added in "
                       "effective design coordinates without another gauge-factor multiplication.",
        "array_axes": {
            "healthy_basis": ["gauge", "load_mode"],
            "load_coefficients": ["design_load", "load_mode"],
            "regional_increments": ["region", "design_load", "gauge"],
            "*_expected_increment_v": ["region", "design_load", "channel"],
            "regions": ["region", "member_index/start_fraction/end_fraction/modulus_reduction"],
        },
        "networks": networks,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    write_json(DESTINATION / "Frame_Network.json", data)
    np.savez_compressed(DESTINATION / "Frame_Design.npz", **arrays)
    print(f"Exported two circuits, {len(arrays['regions'])} regions and "
          f"{len(arrays['load_coefficients'])} loads to {DESTINATION}")


if __name__ == "__main__":
    main()
