"""Unrestricted arm-scale control for the existing joint voltage designs.

Connections and already optimized coefficients are fixed. The control uses the
same design objective, source power, conductor allowance and gauge-power ceiling
as the connection search; only the former arm-scale upper bound is removed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .bridge_network import Circuit
from .passive_acquisition_study import evaluate_candidate, routed_candidate
from .passive_design import constrained_scales, design_gauge_peaks
from .passive_design_study import restore_network
from .passive_resources import source_power
from .passive_robustness_study import distances, source_data


def work_installation(task):
    name, layout = task
    config = json.loads(Path("inputs/passive_robustness_task.json").read_text())
    acquisition = json.loads(Path("inputs/passive_readout_task.json").read_text())
    data = next(d for d in source_data() if d["name"] == name)
    prefix = f"{name}_l{layout}"
    with threadpool_limits(limits=1), np.load("results/passive-robustness/hardware.npz") as saved, \
            np.load("results/passive-network/acquisition.npz") as previous:
        ratio = saved[name + "_actual_nominal_ratio"][layout]
        response = saved[f"{prefix}_c4_{acquisition['source_condition']}_response"]
        healthy = data["coefficients"] @ response.T
        changed = healthy + data["design"][:, :, layout] - data["calibration"][:, layout]
        network = restore_network(previous, prefix + "_scaled", ratio)
        circuit = Circuit(gauge_factor=50., voltage=float(previous[prefix + "_scaled_supply_voltage_v"]) / 2)
        initial = routed_candidate(network, circuit, distances(data, layout), data["depth"][layout], config)
        budget = {
            "source_power_w": float(source_power(initial["hardware"], np.zeros(healthy.shape[-1]))),
            "conductor_length_m": sum(r["total_conductor_m"] for r in initial["routes"]),
            "gauge_peak_power_w": float(design_gauge_peaks(initial["hardware"], healthy, changed).max()),
        }
        candidate = constrained_scales(network, circuit, initial["routes"], healthy, changed,
                                       config, acquisition, budget)
        _, row, _ = evaluate_candidate(candidate, data, layout, saved, healthy, changed,
                                        config, acquisition)
        row.update(budget=budget, scale_fit=candidate["scale_fit"])
        return row


def run(structure=None, layout=None, workers=4, resume=False):
    output = Path("results/passive-network")
    records = []
    if resume and (output / "fixed_wiring.json").exists():
        records = json.loads((output / "fixed_wiring.json").read_text())["records"]
    done = {(r["structure"], r["layout"]) for r in records}
    tasks = [(name, i) for name, count in (("ring", 16), ("frame", 8))
             for i in range(count) if (structure is None or name == structure)
             and (layout is None or i == layout) and (name, i) not in done]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(work_installation, tasks):
            records.append(row)
            metadata = {
                "method": "Joint design coefficients, signs, groups and routes retained; "
                          "arm scales optimized without an upper bound using the same "
                          "regional objective and all three resource allowances.",
                "evaluation": "Existing independent conditions, component draws and acquisition noise.",
                "records": records,
            }
            (output / "fixed_wiring.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"{row['structure']} {row['layout']}: J={np.log(row['design_ratio']):.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", choices=("ring", "frame"))
    parser.add_argument("--layout", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Reuse completed installations")
    args = parser.parse_args()
    run(args.structure, args.layout, args.workers, args.resume)
