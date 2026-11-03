"""Summarize saved representative and unseen-case acquisition results."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def representative_circuit_quantities():
    """Resolve all four gains and active resource limits from the supplied circuits."""
    sys.path.insert(0, str(ROOT / "src/lining_port"))
    from reconstruct_frame import restore_circuit

    circuits = read_json("data/representative-frame/Frame_Network.json")
    result = {}
    with np.load(ROOT / "data/representative-frame/Frame_Design.npz") as arrays:
        healthy = arrays["load_coefficients"] @ arrays["healthy_basis"].T
        increments = arrays["regional_increments"]
        changed = healthy + increments
        for name, record in circuits["networks"].items():
            devices = restore_circuit(record)
            rows, directions = [], []
            zeros = np.zeros(healthy.shape[-1])
            for device, bridge in zip(devices, record["bridges"]):
                indices = device.bridge.gauge_indices
                coefficients = device.bridge.attenuation
                signs = np.array(record["arm_signs"])[device.bridge.arm]
                weight = signs * coefficients
                direction = np.zeros(healthy.shape[-1])
                direction[indices] = weight / np.linalg.norm(weight)
                directions.append(direction)
                rho = np.array([g["nominal_resistance_ohm"] for g in bridge["gauges"]]) / device.circuit.resistance
                occupancy = np.sqrt(rho * coefficients)
                minimum = max(occupancy[device.bridge.arm == j].sum()
                              + bridge["route"]["arm_resistance_ohm"][j]
                              / device.circuit.resistance for j in range(4))
                peak = max(device.operating_point(healthy)["gauge_power_w"].max(),
                           device.operating_point(changed)["gauge_power_w"].max())
                rows.append({
                    "arm_units": bridge["arm_units"],
                    "minimum_arm_units": float(minimum),
                    "gain_v": float(device.nominal_linear_gain() * np.linalg.norm(weight)),
                    "design_peak_gauge_uw": float(peak * 1e6),
                })
            linear = increments @ np.array(directions).T * np.array([r["gain_v"] for r in rows])
            exact = np.stack([d.voltage(changed) - d.voltage(healthy) for d in devices], axis=-1)
            region = circuits["example_region_index"]
            result[name] = {
                "channels": rows,
                "source_voltage_v": record["source_voltage_v"],
                "source_power_uw": float(sum(d.operating_point(zeros)["source_power_w"]
                                             for d in devices) * 1e6),
                "conductor_length_m": sum(b["route"]["total_conductor_m"] for b in record["bridges"]),
                "regional_linear_rms_uv": (np.sqrt(np.mean(np.sum(linear**2, axis=-1), axis=1)) * 1e6).tolist(),
                "regional_exact_rms_uv": (np.sqrt(np.mean(np.sum(exact**2, axis=-1), axis=1)) * 1e6).tolist(),
                "example_region_index": region,
            }
    return result


def representative_objective():
    result, energies = {}, {}
    with np.load(ROOT / "results/passive-network/constrained.npz") as arrays:
        for mode in ("baseline", "constrained"):
            prefix = "frame_l6_" + mode
            delta = arrays[prefix + "_design_increment_v"]
            energy = np.mean(np.sum(delta**2, axis=-1), axis=1)
            np.testing.assert_allclose(
                energy, arrays[prefix + "_design_channel_energy_v2"].sum(-1),
                rtol=1e-12, atol=0.)
            healthy = arrays[prefix + "_design_healthy_operational_v"]
            background = np.mean(np.sum(healthy**2, axis=-1)) + (
                arrays[prefix + "_current_noise_variance_v2"]
                + arrays[prefix + "_reference_noise_variance_v2"]).sum()
            result[mode] = {
                "geometric_region_energy_uv2": float(np.exp(np.log(energy).mean()) * 1e12),
                "background_uv2": float(background * 1e12),
                "J_net": float(np.log(energy).mean() - np.log(background)),
            }
            energies[mode] = energy
    ratios = np.sqrt(energies["constrained"] / energies["baseline"])
    result["regional_rms_ratios"] = {
        "increased": int(np.count_nonzero(ratios > 1)),
        "decreased": int(np.count_nonzero(ratios < 1)),
        "minimum": float(ratios.min()), "maximum": float(ratios.max()),
        "geometric_mean": float(np.exp(np.log(ratios).mean())),
    }
    return result


def paired_evaluation():
    """Pair signal and expected background energies on each saved evaluation set."""
    records = read_json("results/passive-network/constrained.json")["records"]
    result = {}
    with np.load(ROOT / "results/passive-network/constrained.npz") as arrays:
        for structure in ("ring", "frame"):
            installations = []
            for row in records:
                if row["structure"] != structure or row["mode"] != "baseline":
                    continue
                energies = {}
                for mode in ("baseline", "constrained"):
                    prefix = f"{structure}_l{row['layout']}_{mode}"
                    delta = arrays[prefix + "_paired_increment_v"]
                    healthy = arrays[prefix + "_operational_v"][0]
                    signal = float(np.mean(np.sum(delta**2, axis=-1)))
                    background = float(np.mean(np.sum(healthy**2, axis=-1)) + (
                        arrays[prefix + "_current_noise_variance_v2"]
                        + arrays[prefix + "_reference_noise_variance_v2"]).sum())
                    energies[mode] = {
                        "signal_uv2": signal * 1e12,
                        "background_uv2": background * 1e12,
                    }
                joint, constrained = energies["baseline"], energies["constrained"]
                signal_ratio = constrained["signal_uv2"] / joint["signal_uv2"]
                background_ratio = constrained["background_uv2"] / joint["background_uv2"]
                installations.append({
                    "layout_seed": row["layout_seed"], **energies,
                    "signal_ratio": signal_ratio,
                    "background_ratio": background_ratio,
                    "signal_background_ratio": signal_ratio / background_ratio,
                })
            summary = {}
            for key in ("signal_ratio", "background_ratio", "signal_background_ratio"):
                values = np.array([item[key] for item in installations])
                summary[key] = {
                    "increased": int(np.count_nonzero(values > 1)),
                    "decreased": int(np.count_nonzero(values < 1)),
                    "unchanged": int(np.count_nonzero(values == 1)),
                    "minimum": float(values.min()), "maximum": float(values.max()),
                }
            result[structure] = {"installations": installations, "summary": summary}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-evaluation", action="store_true",
                        help="Read constrained results and report paired evaluation energies only")
    args = parser.parse_args()
    if args.paired_evaluation:
        result = paired_evaluation()
    else:
        result = {
            "representative_circuit": representative_circuit_quantities(),
            "representative_frame": representative_objective(),
            "paired_independent_evaluation": paired_evaluation(),
        }
    print(json.dumps(result, indent=2))
