"""Passive-circuit replay of paired Taishan steel-stress measurements.

Field inputs are stress records. Voltages and injected strain
increments are calculated quantities. The empirical reference is construction
loading, with no structural-damage labels assigned to the observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bridge_network import Circuit
from .passive_bridge import TrimmedBridge, response_spaces
from .passive_hardware import BridgeHardware


ROOT = Path(__file__).resolve().parents[2]
TAISHAN = ("B2", "K", "B1", "A1", "A2", "A3", "A4")
STEEL_MODULUS_PA = 200e9
CALIBRATION_LAST_DAY = 48
ATTENUATION_SCALE = .81


def read_taishan(folder):
    table = np.genfromtxt(Path(folder) / "taishan_1690_steel.csv", delimiter=",", names=True)
    stress = np.column_stack([table[f"{s}_{side}_MPa"] for s in TAISHAN for side in ("outer", "inner")])
    return table["day"], stress * 1e6 / STEEL_MODULUS_PA


def paired_contrast(strain):
    """Inner-minus-outer strain, without assuming an unrecorded gauge depth."""
    values = np.asarray(strain)
    return values[..., 1::2] - values[..., ::2]


def calibrated_direction(calibration_strain, target_station, include_initial_pattern=True):
    """Minimum-norm contrast row with unit local response and selected null modes.

    The first observed contrast and the leading change from that observation
    define the illustrated reference. The initial pattern is an optional extra
    constraint; change-only rejection uses just the leading singular direction.
    Only the supplied calibration observations are used.
    """
    contrast = paired_contrast(calibration_strain)
    _, singular, right = np.linalg.svd(contrast - contrast[0], full_matrices=False)
    basis = np.column_stack((contrast[0], right[0])) if include_initial_pattern else right[:1].T
    _, null = response_spaces(basis)
    projection = null @ null.T
    if projection[target_station, target_station] <= np.finfo(float).eps:
        raise ValueError("The target contrast is contained in the reference space")
    q = projection[:, target_station] / projection[target_station, target_station]
    return q, basis, float(singular[0]**2 / (singular @ singular))


def paired_hardware(q, indices=None):
    """A balanced four-arm realization, using every selected gauge once.

    Same-sign cells are assigned in the established alternating order. The arm
    scale is rounded upward to a quarter R0 after including its conductor.
    All lead and receiver values here are prescribed circuit-design inputs.
    """
    weight = np.column_stack((-q, q)).ravel()
    indices = np.arange(len(weight)) if indices is None else np.asarray(indices)
    initial = TrimmedBridge.from_weight(indices, weight)
    c = ATTENUATION_SCALE * initial.attenuation
    circuit = Circuit(resistance=350., gauge_factor=2., voltage=.5)
    arm_lead = np.ones(4)
    occupancy = np.array([np.sqrt(c[initial.arm == j]).sum() for j in range(4)])
    m = float(np.ceil(4 * np.max(occupancy + arm_lead / circuit.resistance)) / 4)
    bridge = TrimmedBridge(indices, initial.arm, c, m)
    hardware = BridgeHardware.from_design(bridge, circuit, arm_lead=arm_lead,
                                         supply_lead=2., sense_lead=1., input_resistance=1e6)
    gain = hardware.nominal_linear_gain() * ATTENUATION_SCALE / np.max(np.abs(weight))
    return hardware, gain


def hardware_record(hardware, gain, names, strain_count):
    zero = np.zeros(strain_count)
    return {
        "gauge_names": names,
        "gauge_indices": hardware.bridge.gauge_indices.tolist(),
        "arm_zero_based": hardware.bridge.arm.tolist(),
        "attenuation": hardware.bridge.attenuation.tolist(),
        "arm_scale": hardware.bridge.arm_units,
        "reference_resistance_ohm": hardware.circuit.resistance,
        "gauge_factor": hardware.circuit.gauge_factor,
        "source_rail_to_rail_V": 2 * hardware.circuit.voltage,
        "shunt_ohm": hardware.shunt.tolist(),
        "ballast_ohm": hardware.ballast.tolist(),
        "arm_lead_ohm": hardware.arm_lead.tolist(),
        "each_supply_lead_ohm": hardware.supply_lead,
        "each_sense_lead_ohm": hardware.sense_lead,
        "receiver_ohm": hardware.input_resistance,
        "gain_V_per_unit_contrast": gain,
        "zero_strain_netlist": hardware.netlist(zero),
    }


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def taishan_study(folder):
    days, strain = read_taishan(folder)
    calibration = days <= CALIBRATION_LAST_DAY
    evaluation = ~calibration
    target = TAISHAN.index("A1")
    q, basis, fraction = calibrated_direction(strain[calibration], target)
    hardware, gain = paired_hardware(q)
    contrast = paired_contrast(strain)
    relative = contrast - contrast[0]
    filtered = relative @ q
    local = relative[:, target]
    voltage = hardware.voltage(strain)
    readout = voltage - voltage[0]

    # A prescribed local bending-strain perturbation, evaluated on held-out
    # absolute field strains. It is not a measured event or an inferred defect.
    injection = np.zeros(strain.shape[1])
    injection[2 * target:2 * target + 2] = [-5e-6, 5e-6]
    injected = hardware.voltage(strain[evaluation] + injection) - voltage[evaluation]
    membrane = np.repeat(np.eye(len(q)), 2, axis=1) * 50e-6
    offsets = np.concatenate((membrane, -membrane))
    leakage = hardware.voltage(strain[evaluation, None, :] + offsets) - voltage[evaluation, None]
    operating = hardware.operating_point(strain)
    summary = {
        "station_order": list(TAISHAN), "reference_day": float(days[0]),
        "calibration_days": days[calibration].tolist(), "evaluation_days": days[evaluation].tolist(),
        "reference_change_mode_energy_fraction": fraction,
        "contrast_weights": q.tolist(), "reference_contrast_basis": basis.tolist(),
        "paired_membrane_dimension": len(q), "total_reference_dimension": len(q) + 2,
        "target_available_fraction": float(1 / (q @ q)),
        "contrast_weight_norm": float(np.linalg.norm(q)),
        "unit_direction_gain_V_per_strain": float(gain * np.sqrt(2) * np.linalg.norm(q)),
        "calibration_local_rms_microstrain": rms(local[calibration]) * 1e6,
        "calibration_filtered_rms_microstrain": rms(filtered[calibration]) * 1e6,
        "evaluation_local_rms_microstrain": rms(local[evaluation]) * 1e6,
        "evaluation_filtered_rms_microstrain": rms(filtered[evaluation]) * 1e6,
        "evaluation_local_mean_microstrain": float(np.mean(local[evaluation]) * 1e6),
        "evaluation_filtered_mean_microstrain": float(np.mean(filtered[evaluation]) * 1e6),
        "evaluation_rms_reduction_fraction": 1 - rms(filtered[evaluation]) / rms(local[evaluation]),
        "evaluation_recorded_replay_rms_microvolt": rms(readout[evaluation]) * 1e6,
        "evaluation_exact_minus_linear_max_microvolt": float(np.max(abs(readout[evaluation] - gain * filtered[evaluation]))) * 1e6,
        "synthetic_target_contrast_microstrain": 10.,
        "synthetic_target_linear_microvolt": gain * 10.,
        "synthetic_target_exact_range_microvolt": [float(injected.min() * 1e6), float(injected.max() * 1e6)],
        "synthetic_membrane_offset_microstrain": 50.,
        "synthetic_membrane_max_leakage_microvolt": float(np.max(abs(leakage)) * 1e6),
        "peak_gauge_power_mW": float(operating["gauge_power_w"].max() * 1e3),
        "peak_source_power_mW": float(operating["source_power_w"].max() * 1e3),
    }
    alternatives = []
    for include_initial in (False, True):
        row, reference, _ = calibrated_direction(strain[calibration], target, include_initial)
        value = relative @ row
        alternatives.append({
            "include_initial_pattern": include_initial,
            "reference_contrast_rank": reference.shape[1],
            "contrast_weights": row.tolist(),
            "target_available_fraction": float(1 / (row @ row)),
            "contrast_weight_norm": float(np.linalg.norm(row)),
            "calibration_filtered_rms_microstrain": rms(value[calibration]) * 1e6,
            "evaluation_filtered_rms_microstrain": rms(value[evaluation]) * 1e6,
            "evaluation_filtered_mean_microstrain": float(np.mean(value[evaluation]) * 1e6),
        })
    summary["reference_choice_comparison"] = alternatives
    record = hardware_record(hardware, gain, [f"{s}_{side}" for s in TAISHAN for side in ("outer", "inner")], strain.shape[1])
    arrays = dict(day=days, calibration_mask=calibration, nominal_elastic_strain=strain,
                  paired_contrast=contrast, selected_contrast=filtered, local_A1_contrast=local,
                  exact_voltage_V=voltage, reference_subtracted_voltage_V=readout,
                  injected_target_increment_V=injected, membrane_leakage_V=leakage)
    return summary, record, arrays


def write_results(folder, output):
    taishan, ts_circuit, ts_arrays = taishan_study(folder)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"steel_modulus_Pa": STEEL_MODULUS_PA,
               "interpretation": "Offline circuit replay of nominal elastic strain equivalents; no observed resistor-bridge voltage or damage label",
               "taishan_1690": taishan}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "circuits.json").write_text(json.dumps({"taishan_1690": ts_circuit}, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "replay.npz", **{f"taishan_{k}": v for k, v in ts_arrays.items()})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/field-monitoring")
    parser.add_argument("--output", type=Path, default=ROOT / "results/field-bridges")
    args = parser.parse_args()
    print(json.dumps(write_results(args.data, args.output), indent=2))
