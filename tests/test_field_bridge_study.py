"""Checks of field-data alignment, frozen calibration and physical circuit replay."""
import numpy as np
import pytest

from lining_port.field_bridge_study import (
    ROOT, TAISHAN, ATTENUATION_SCALE, read_taishan,
    calibrated_direction, paired_hardware, paired_contrast,
    taishan_study,
)
from test_passive_bridge import solve_netlist


DATA = ROOT / "data/field-monitoring"


def test_measurement_units_and_pairing():
    days, strain = read_taishan(DATA)
    assert strain.shape == (29, 14)
    assert days.tolist()[-3:] == [133., 147., 163.]
    np.testing.assert_allclose(strain[0, :2] * 200000, [-23.570769653616537, -39.41928703315692])
    moments = np.genfromtxt(DATA / "taishan_1690_derived_moments.csv", delimiter=",", names=True)
    for station in ("A1", "A3"):
        measured_difference = abs(paired_contrast(strain)[:, TAISHAN.index(station)]) * 200000
        np.testing.assert_allclose(moments[f"{station}_magnitude_kNm"], 3.2 * measured_difference, atol=1e-10)


def test_calibration_has_unit_target_gain_and_rejects_reference_space():
    days, strain = read_taishan(DATA)
    q, basis, fraction = calibrated_direction(strain[days <= 48], 3)
    assert q[3] == pytest.approx(1.)
    np.testing.assert_allclose(q @ basis, 0, atol=1e-14)
    assert fraction == pytest.approx(.9523333459525799)
    device, gain = paired_hardware(q)
    np.testing.assert_allclose(device.bridge.weight, ATTENUATION_SCALE * np.column_stack((-q, q)).ravel(), atol=1e-14)
    assert np.min(device.shunt) > 0 and np.min(device.ballast) > 0
    np.testing.assert_allclose(device.arm_resistance(np.zeros(14)), 700., atol=1e-12)
    for direction in np.repeat(np.eye(7), 2, axis=1):
        derivative = (device.voltage(direction * 1e-7) - device.voltage(-direction * 1e-7)) / 2e-7
        assert abs(derivative) < gain * 1e-7


def test_later_observations_do_not_change_the_calibrated_circuit(tmp_path):
    source = np.genfromtxt(DATA / "taishan_1690_steel.csv", delimiter=",", skip_header=1)
    source[source[:, 0] > 48, 1:] *= np.linspace(.3, 2.7, 14)
    header = (DATA / "taishan_1690_steel.csv").read_text().splitlines()[0]
    np.savetxt(tmp_path / "taishan_1690_steel.csv", source, delimiter=",", header=header, comments="")
    original, original_circuit, _ = taishan_study(DATA)
    changed, changed_circuit, _ = taishan_study(tmp_path)
    assert original["contrast_weights"] == changed["contrast_weights"]
    assert original_circuit == changed_circuit
    assert len(original["evaluation_days"]) == 9
    assert original["evaluation_filtered_rms_microstrain"] != changed["evaluation_filtered_rms_microstrain"]


def test_paired_direction_captures_the_available_full_gauge_target():
    days, strain = read_taishan(DATA)
    q, basis, _ = calibrated_direction(strain[days <= 48], 3)
    contrast_map = np.kron(np.eye(7), [[-1., 1.]])
    common_map = np.kron(np.eye(7), [[1.], [1.]])
    reference = np.column_stack((common_map, .5 * contrast_map.T @ basis))
    reference /= np.linalg.norm(reference, axis=0)
    left, singular, _ = np.linalg.svd(reference, full_matrices=False)
    assert np.count_nonzero(singular > 1e-10) == 9
    projection = np.eye(14) - left @ left.T
    target = .5 * contrast_map.T[:, 3]
    weight = contrast_map.T @ q
    unit_weight = weight / np.linalg.norm(weight)
    np.testing.assert_allclose(reference.T @ weight, 0., atol=1e-14)
    available = np.linalg.norm(projection @ target)**2
    assert (unit_weight @ target)**2 == pytest.approx(available)
    result, circuit, arrays = taishan_study(DATA)
    assert result["target_available_fraction"] == pytest.approx(available / (target @ target))
    assert result["contrast_weight_norm"] == pytest.approx(np.linalg.norm(q))
    assert result["unit_direction_gain_V_per_strain"] == pytest.approx(
        circuit["gain_V_per_unit_contrast"] * np.linalg.norm(weight))
    evaluation = arrays["day"] > 48
    assert result["evaluation_filtered_mean_microstrain"] == pytest.approx(
        np.mean(arrays["selected_contrast"][evaluation]) * 1e6)


def test_complete_netlists_match_replayed_field_states_and_injection():
    days, strain = read_taishan(DATA)
    q, _, _ = calibrated_direction(strain[days <= 48], 3)
    device, _ = paired_hardware(q)
    injected = strain[-1].copy()
    injected[6:8] += [-5e-6, 5e-6]
    for row in np.vstack((strain, injected)):
        kcl = solve_netlist(device.netlist(row), device.circuit.voltage)
        assert device.voltage(row) == pytest.approx(kcl, abs=2e-13)


def test_injected_contrast_is_retained_over_all_later_operating_points():
    result, _, _ = taishan_study(DATA)
    expected = result["synthetic_target_linear_microvolt"]
    exact = np.asarray(result["synthetic_target_exact_range_microvolt"])
    assert np.max(abs(exact / expected - 1)) < .001
    assert result["synthetic_membrane_max_leakage_microvolt"] < .005
    assert result["evaluation_rms_reduction_fraction"] == pytest.approx(.8488536641025728)
def test_change_only_reference_is_invariant_to_instrument_offsets():
    days, strain = read_taishan(DATA)
    first = strain[days <= 48]
    offset = np.linspace(-30e-6, 40e-6, first.shape[1])
    q, basis, _ = calibrated_direction(first, 3, include_initial_pattern=False)
    shifted, _, _ = calibrated_direction(first + offset, 3, include_initial_pattern=False)
    np.testing.assert_allclose(shifted, q, atol=1e-12)
    np.testing.assert_allclose(basis.T @ q, 0., atol=1e-12)
