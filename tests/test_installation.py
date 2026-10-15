from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse.csgraph import connected_components

from lining_port.bridge_network import Circuit, components, response_basis, system
from lining_port.fe import RingParameters, solve_ring, recover_gauge_strain
from lining_port.installation import installed_network
from lining_port.installation_study import detector, make_cases


def test_all_installed_cells_reach_their_port_and_change_its_voltage():
    layout, geom = installed_network(7108)
    parts = components(np.zeros(64), layout, geom)
    count, labels = connected_components(parts["edges"], directed=False)
    assert count == 4
    np.testing.assert_array_equal(np.bincount(labels), [16] * 4)
    for module, port in enumerate(geom["port_nodes"]):
        assert np.all(labels[16 * module:16 * module + 16] == labels[port])
    # Independent DC current injection tests the actual conductance/receiver pair.
    transfer = geom["H"] @ np.linalg.solve(parts["conductance"], np.eye(64))
    assert np.all(transfer[geom["module"], np.arange(64)] > 0)
    basis = response_basis(np.full(64, -1e-4), layout, geom)
    assert np.all(np.linalg.norm(basis, axis=0) > 0)
    assert geom["diagnostics"]["edges"] == 88
    np.testing.assert_allclose(geom["z"].reshape(32, 2), np.tile([-.1, .1], (32, 1)))


def test_periodic_reset_matches_repeated_full_cycles():
    layout, geom = installed_network(7108)
    eps = np.full(64, -1e-4)
    code = np.ones((64, 1))
    matrix, drive, output = system(eps, layout, geom)
    transition = expm(matrix * .005)
    reset = expm(matrix * .03)
    equilibrium = np.linalg.solve(matrix, -drive @ code)
    forcing = equilibrium - transition @ equilibrium
    state = np.zeros_like(equilibrium)
    probe = np.random.default_rng(4100).choice([-1., 1.], 100)
    for _ in range(3):
        state = reset @ state
        values = []
        for step in range(400):
            values.append(output @ state)
            state = transition @ state + forcing * probe[step // 4]
    actual = response_basis(eps, layout, geom, codes=code, reset_duration=.03)
    np.testing.assert_allclose(actual, np.array(values).reshape(1600, 1), atol=2e-14)


def test_task_keeps_installation_fixed_and_uses_independent_load_cases():
    config = json.loads(Path("inputs/installation_task.json").read_text())
    cases = make_cases(config)
    assert {r["split"] for r in cases} == {"calibration", "test"}
    assert len({tuple(r["params"].values()) for r in cases}) == len(cases)
    assert all(set(r["params"]) == {"pressure", "harmonic", "phase"} for r in cases)
    p = RingParameters(**config["nominal_mechanics"])
    _, geom = installed_network(7108, p.radius, p.thickness, config["depth_m"])
    strains = []
    for pressure in (1e5, 2e5):
        field = solve_ring(replace(p, pressure=pressure), 128)
        strains.append(recover_gauge_strain(field, geom["theta"], geom["z"], .0002))
    np.testing.assert_allclose(strains[1], 2 * strains[0], atol=1e-13)


def test_detector_removes_load_modes_and_keeps_target_coordinates():
    rng = np.random.default_rng(19)
    nuisance = rng.normal(size=(100, 3))
    targets = rng.normal(size=(100, 12))
    reference, q, u = detector(np.zeros(100), nuisance, targets)
    np.testing.assert_allclose(u.T @ nuisance, 0, atol=1e-13)
    np.testing.assert_allclose(u.T @ u, np.eye(12), atol=1e-13)
    delta = targets[:, 0] + nuisance @ [1, 2, 3]
    np.testing.assert_allclose(np.linalg.norm(u.T @ (delta - reference)),
                               np.linalg.norm(delta - q @ (q.T @ delta)), atol=1e-13)


def test_dependent_load_sensitivities_do_not_promote_roundoff_to_a_mode():
    rng = np.random.default_rng(42)
    physical, _ = np.linalg.qr(rng.normal(size=(200, 4)))
    nuisance = np.column_stack((physical[:, 0], physical[:, 1],
                               physical[:, 0] + 2 * physical[:, 1] + 1e-12 * rng.normal(size=200)))
    targets = physical @ rng.normal(size=(4, 12))
    _, background, signal = detector(np.zeros(200), nuisance, targets)
    assert background.shape[1] == 2
    assert signal.shape[1] == 2
    # The third derivative contains finite-difference roundoff. Target response
    # coordinates must remain orthogonal to the retained healthy subspace.
    np.testing.assert_allclose(signal.T @ background, 0, atol=1e-12)
