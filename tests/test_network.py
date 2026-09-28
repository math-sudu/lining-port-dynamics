import numpy as np
import pytest

from lining_port.network import (
    _segment_inside, geometry, make_layout, receiver_matrices, simulate,
)


def test_area_quantiles_and_parameters_are_preserved():
    layout = make_layout(15)
    g1, g2 = geometry(layout, 3., .27), geometry(layout, 3., .33)
    for g, thickness in [(g1, .27), (g2, .33)]:
        r = np.linalg.norm(g["xy"], axis=1)
        lo, hi = 3. - thickness / 2, 3. + thickness / 2
        np.testing.assert_allclose((r**2 - lo**2) / (hi**2 - lo**2), layout["area_quantile"], atol=1e-14)
    np.testing.assert_array_equal(g1["theta"], g2["theta"])


def test_edges_and_ports_obey_annulus_and_distance():
    g = geometry(make_layout(3100), 3., .3)
    np.testing.assert_array_equal(g["W"], g["W"].T)
    assert np.count_nonzero(np.diag(g["W"])) == 0
    for i, j in zip(*np.where(g["W"] > 0)):
        assert np.linalg.norm(g["xy"][i] - g["xy"][j]) <= .6
        assert _segment_inside(g["xy"][i], g["xy"][j], 2.85)
    selected = g["port_nodes"][g["mask"]]
    assert len(selected) == len(set(selected.tolist()))
    for port in np.flatnonzero(g["mask"]):
        xy = g["xy"][g["port_nodes"][port]]
        assert np.linalg.norm(g["port_xy"][port] - xy) <= .5
        assert _segment_inside(g["port_xy"][port], xy, 2.85)
    assert not _segment_inside(np.array([3., 0.]), np.array([-3., 0.]), 2.85)


def test_analog_receiver_gain_cutoff_and_high_frequency():
    a, b, c = receiver_matrices()
    transfer = lambda hz: c @ np.linalg.solve(1j * 2 * np.pi * hz * np.eye(4) - a, b)
    assert abs(transfer(0) - 1) < 1e-14
    assert abs(abs(transfer(40)) - 1 / np.sqrt(2)) < 1e-14
    assert abs(transfer(400)) < 1.01e-4


def test_observation_contract_and_missing_ports():
    layout = make_layout(3100)
    g = geometry(layout, 3., .3)
    result = simulate(np.zeros(64), g, layout, method="expm", noise_seed=17)
    assert result["z"].shape == (400, 4)
    assert result["q"].shape == (400,)
    assert result["time"].shape == (400,)
    assert np.all(np.diff(result["time"]) > 0)
    g["H"][:] = 0
    g["mask"][:] = False
    missing = simulate(np.zeros(64), g, layout, method="expm")
    assert missing["diagnostics"]["status"] == "unobservable_no_ports"
    assert np.count_nonzero(missing["z"]) == 0
    with pytest.raises(ValueError, match="finite circumferential"):
        simulate(np.full(64, np.nan), g, layout)


def test_p0_has_only_receiver_memory_and_expected_shape():
    layout = make_layout(3100)
    g = geometry(layout, 3., .3)
    result = simulate(np.zeros(64), g, layout, family="P0", method="expm", noise_sigma=0.)
    assert result["diagnostics"]["node_states"] == 0
    assert result["diagnostics"]["state_dimension"] == 16
    # A real PRBS is passed through the same analog chain, creating transients.
    channel = np.flatnonzero(g["mask"])[0]
    assert np.std(result["z"][:, channel]) > 0
    assert result["z"][0, channel] == 0
