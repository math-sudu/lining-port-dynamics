"""Reconstruct the supplied frame circuits from their stored components.

Run this file directly. The delivered copy uses the accompanying
passive_hardware.py and requires only NumPy.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from passive_hardware import BridgeHardware


def restore_circuit(record):
    """Keep the exported resistors and leads; do not synthesize new values."""
    circuit = SimpleNamespace(
        resistance=record["resistance_reference_ohm"],
        gauge_factor=record["gauge_factor_reference"],
        voltage=record["source_voltage_v"] / 2,
    )
    hardware = []
    for item in record["bridges"]:
        gauges, route = item["gauges"], item["route"]
        bridge = SimpleNamespace(
            gauge_indices=np.array([g["gauge_index"] for g in gauges]),
            arm=np.array([g["arm_index"] for g in gauges]),
            attenuation=np.array([g["attenuation"] for g in gauges]),
            arm_units=item["arm_units"],
        )
        hardware.append(BridgeHardware(
            bridge, circuit,
            np.array([g["nominal_resistance_ohm"] for g in gauges]),
            np.array([np.inf if g["shunt_ohm"] is None else g["shunt_ohm"]
                      for g in gauges]),
            np.array(item["arm_ballast_ohm"]),
            np.array(route["arm_resistance_ohm"]),
            route["one_instrument_lead_ohm"],
            route["one_instrument_lead_ohm"],
            item["receiver_resistance_ohm"],
        ))
    return hardware


def voltage(record, strain, *, coordinates):
    """Read effective design strain or physical strain, applying sensitivity once."""
    effective = np.asarray(strain)
    if coordinates == "physical":
        ratios = np.empty(effective.shape[-1])
        for bridge in record["bridges"]:
            for gauge in bridge["gauges"]:
                ratios[gauge["gauge_index"]] = (
                    gauge["gauge_factor"] / record["gauge_factor_reference"])
        effective = effective * ratios
    elif coordinates != "effective":
        raise ValueError("coordinates must be 'physical' or 'effective'")
    return np.stack([h.voltage(effective) for h in restore_circuit(record)], axis=-1)


def reconstruct(folder):
    data = json.loads((folder / "Frame_Network.json").read_text(encoding="utf-8"))
    outputs = {}
    with np.load(folder / "Frame_Design.npz") as inputs:
        healthy = inputs["load_coefficients"] @ inputs["healthy_basis"].T
        changed = healthy + inputs["regional_increments"]
        for name, record in data["networks"].items():
            vh = voltage(record, healthy, coordinates="effective")
            vc = voltage(record, changed, coordinates="effective")
            delta = vc - vh
            outputs[name + "_increment_v"] = delta
            outputs[name + "_healthy_v"] = vh
            rms = np.sqrt(np.mean(np.sum(delta**2, axis=-1), axis=1))
            region = data["example_region_index"]
            print(f"{name}: regional RMS = {rms[region] * 1e6:.6f} microvolt")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, help="Optional NPZ of reconstructed voltages")
    args = parser.parse_args()
    outputs = reconstruct(args.data)
    if args.output:
        np.savez_compressed(args.output, **outputs)


if __name__ == "__main__":
    main()
