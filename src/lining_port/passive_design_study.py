"""Resource-constrained arm and regional-group design on retained installations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .bridge_network import Circuit
from .passive_acquisition_study import evaluate_candidate, routed_candidate
from .passive_bridge import TrimmedBridge
from .passive_design import (design_gauge_peaks, physical_candidates,
                             projection_swaps, refine_arms, regroup_candidate)
from .passive_resources import design_ratio, minimum_arm_units, source_power
from .passive_robustness_study import distances, source_data


def restore_network(arrays, prefix, ratio):
    channels = len(arrays[prefix + "_arm_units"])
    return [TrimmedBridge(group, arm, c, float(m), ratio[group])
            for group, arm, c, m in zip(
                np.array_split(arrays[prefix + "_gauge_order"], channels),
                np.array_split(arrays[prefix + "_arm"], channels),
                np.array_split(arrays[prefix + "_attenuation"], channels),
                arrays[prefix + "_arm_units"])]


def design_installation(data, layout, saved, previous, config, acquisition):
    name, seed = data["name"], data["seeds"][layout]
    response = saved[f"{name}_l{layout}_c4_{acquisition['source_condition']}_response"]
    ratio = saved[name + "_actual_nominal_ratio"][layout]
    healthy = data["coefficients"] @ response.T
    increments = data["design"][:, :, layout] - data["calibration"][:, layout]
    changed = healthy + increments
    geometry, depth = distances(data, layout), data["depth"][layout]
    prefix = f"{name}_l{layout}"
    baseline_network = restore_network(previous, prefix + "_scaled", ratio)
    circuit = Circuit(gauge_factor=50., voltage=float(previous[prefix + "_scaled_supply_voltage_v"]) / 2)
    baseline = routed_candidate(baseline_network, circuit, geometry, depth, config)
    budget = {
        "source_power_w": float(source_power(baseline["hardware"], np.zeros(healthy.shape[-1]))),
        "conductor_length_m": sum(r["total_conductor_m"] for r in baseline["routes"]),
        "gauge_peak_power_w": float(design_gauge_peaks(baseline["hardware"], healthy, changed).max()),
    }
    score = lambda c: design_ratio(c["hardware"], healthy, changed, config, acquisition)[0]
    trace = []

    def physical(network, origin, extra=None):
        candidates = physical_candidates(network, circuit, geometry, depth, healthy, changed,
                                          config, acquisition, budget)
        for candidate in candidates:
            candidate["origin"] = origin
            candidate["proposal"] = extra or {}
        trace.append({"origin": origin, **(extra or {}), "feasible_arm_candidates": len(candidates),
                      "design_ratio": [float(np.exp(score(c))) for c in candidates]})
        return candidates

    baseline["origin"] = "scaled"
    arms = max([baseline, *physical(baseline_network, "same_sign")], key=score)
    arms = refine_arms(arms, geometry, depth, healthy, changed, config, acquisition, budget)
    best = arms
    order = np.argsort(data["theta"][layout]) if name == "ring" else np.lexsort(
        (data["fraction"][layout], data["member"][layout]))
    print(name, seed, "same_sign", round(np.exp(score(baseline)), 2), "->", round(np.exp(score(arms)), 2), flush=True)
    for mode in ("joint_fixed", "joint_regrouped"):
        network = restore_network(previous, prefix + "_" + mode, ratio)
        candidates = physical(network, mode)
        if candidates:
            best = max([best, *candidates], key=score)
        proposals = projection_swaps(network, response, increments, order)
        for proposal in proposals:
            fitted, fit = regroup_candidate(network, proposal, response, healthy, changed, ratio,
                circuit, geometry, depth, config, acquisition, budget)
            extra = {key: value for key, value in proposal.items() if key != "groups"}
            extra["coefficient_fit"] = fit
            candidates = physical(fitted, mode + "_exchange", extra)
            if candidates:
                best = max([best, *candidates], key=score)
        print(name, seed, mode, "proposals", len(proposals), "best", round(np.exp(score(best)), 2), flush=True)
    if best is not arms:
        best = refine_arms(best, geometry, depth, healthy, changed, config, acquisition, budget)
    arrays, records, exports = {}, [], []
    # Held-out strains and component errors are first used after all selection.
    for mode, current in (("baseline", baseline), ("arms", arms), ("constrained", best)):
        values, row, export = evaluate_candidate(current, data, layout, saved, healthy, changed, config, acquisition)
        arrays.update({f"{prefix}_{mode}_{key}": value for key, value in values.items()})
        row.update({"mode": mode, "origin": current["origin"], "budget": budget,
                    "arm_method": current.get("arm_method", "retained_arms"),
                    "arm_moves": current.get("arm_moves", []),
                    "scale_fit": current.get("scale_fit"), "proposal": current.get("proposal", {}),
                    "minimum_arm_units": [minimum_arm_units(b, current["circuit"], r)
                                          for b, r in zip(current["network"], current["routes"])]})
        records.append(row)
        exports.append(export | {"mode": mode, "origin": current["origin"], "budget": budget})
    return arrays, records, exports, {"structure": name, "layout": layout, "layout_seed": seed,
                                     "trials": trace}


def summarize(records, arrays):
    summaries = []
    for name in ("ring", "frame"):
        for mode in ("baseline", "arms", "constrained"):
            rows = [r for r in records if r["structure"] == name and r["mode"] == mode]
            if not rows:
                continue
            result = {"structure": name, "mode": mode, "installations": len(rows)}
            for key in ("nominal_source_power_mw", "conductor_length_m", "operational_noise_rms_uv", "design_ratio"):
                result[key] = float(np.median([r[key] for r in rows]))
            for key in ("design_gauge_peak_power_uw", "service_gauge_peak_power_uw"):
                result[key] = max(r[key] for r in rows)
            for key in ("healthy_sampled_norm_uv", "changed_sampled_norm_uv", "paired_increment_norm_uv"):
                result[key] = {stat: float(np.median([r[key][stat] for r in rows])) for stat in ("median", "p10", "p95")}
            increments = np.concatenate([np.linalg.norm(arrays[
                f"{name}_l{r['layout']}_{mode}_paired_increment_v"], axis=-1).ravel() for r in rows])
            result["pooled_increment_p10_uv"] = float(np.quantile(increments, .1) * 1e6)
            summaries.append(result)
    return summaries


def work_installation(task):
    name, layout = task
    config = json.loads(Path("inputs/passive_robustness_task.json").read_text())
    acquisition = json.loads(Path("inputs/passive_readout_task.json").read_text())
    data = next(d for d in source_data() if d["name"] == name)
    with threadpool_limits(limits=1), np.load("results/passive-robustness/hardware.npz") as saved, \
            np.load("results/passive-network/acquisition.npz") as previous:
        return design_installation(data, layout, saved, previous, config, acquisition)


def run(resume=False, structure=None, layout_index=None, workers=4):
    output = Path("results/passive-network")
    acquisition = json.loads(Path("inputs/passive_readout_task.json").read_text())
    arrays, records, exports, searches = {}, [], [], []
    if resume:
        arrays = dict(np.load(output / "constrained.npz"))
        retained = json.loads((output / "constrained.json").read_text())
        records, searches = retained["records"], retained["searches"]
        exports = json.loads((output / "constrained_netlists.json").read_text())
    completed = {(r["structure"], r["layout"]) for r in records}
    tasks = []
    for data in source_data():
        if structure is not None and data["name"] != structure:
            continue
        for layout in range(len(data["seeds"])):
            if (layout_index is not None and layout != layout_index) or (data["name"], layout) in completed:
                continue
            tasks.append((data["name"], layout))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for values, rows, netlists, search in pool.map(work_installation, tasks):
            arrays.update(values)
            records.extend(rows)
            exports.extend(netlists)
            searches.append(search)
            np.savez_compressed(output / "constrained.npz", **arrays)
            metadata = {
                "channels": 4, "acquisition": acquisition, "records": records, "searches": searches,
                "summaries": summarize(records, arrays),
                "budget_source": "Each installation's acquisition_scaled design: total conductor length, "
                                 "nominal source power including supply leads, and the largest single-gauge power "
                                 "over healthy and changed design loads. These are comparison budgets.",
                "method": "Retain Scaled; compare fixed-weight same-sign arm moves and swaps using length and "
                          "arm-occupancy proposals. Start from both fixed and cyclic joint templates. For each "
                          "channel pair propose its best projection-energy gauge exchange, reproject weights and "
                          "refine the full-circuit regional objective. Compare original, shorter and balanced arms; "
                          "optimize scales with gauge-power inequalities and exact nominal source-power equality. "
                          "Only wire-feasible candidates enter selection by the full-circuit design objective. "
                          "The retained fixed-group design and final chosen grouping each receive one channel "
                          "sweep of same-sign moves/swaps accepted by that exact objective within all budgets.",
                "evaluation": "Same retained ring test, six-mode frame confirmation and component-setting/noise draws "
                              "as acquisition; none enters budgets, proposal ranking or selection.",
                "array_axes": {"current": ["state", "hardware_draw", "case", "channel"],
                               "reference": ["hardware_draw", "channel"],
                               "design_increment": ["region", "design_load", "channel"]},
            }
            (output / "constrained.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            (output / "constrained_netlists.json").write_text(json.dumps(exports, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Reuse completed installations after an interruption")
    parser.add_argument("--structure", choices=("ring", "frame"))
    parser.add_argument("--layout", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.resume, args.structure, args.layout, args.workers)
