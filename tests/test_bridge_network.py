import numpy as np
from scipy.integrate import solve_ivp

from lining_port.bridge_network import Circuit, components, response_basis
from lining_port.bridge_design import record_scores, synthesize
from lining_port.network import receiver_matrices


def fixture():
    layout = {"tau0": np.array([.023, .051, .073])}
    geom = {"W": np.array([[0., .7, 0.], [.7, 0., .4], [0., .4, 0.]]),
            "H": np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 0., 0.]])}
    return layout, geom


def test_physical_kcl_against_independent_ode_and_superposition():
    layout, geom = fixture()
    strain = np.array([-1e-4, 4e-5, -2e-5])
    circuit = Circuit()
    code = np.array([1., -1., 1.])
    parts = components(strain, layout, geom, circuit)
    rx_a, rx_b, rx_c = receiver_matrices()
    values, current = [], np.zeros(19)

    def ode(_, state, q):
        voltage = state[:3]
        # Direct currents through each of the four physical branch types.
        current_in = (parts["active"] * (code * q - voltage)
                      + parts["reference"] * (-code * q - voltage)
                      - parts["leakage"] * voltage
                      + np.sum(parts["edges"] * (voltage[None, :] - voltage[:, None]), axis=1))
        filters = state[3:].reshape(4, 4)
        derivative = filters @ rx_a.T + (geom["H"] @ voltage)[:, None] * rx_b
        return np.r_[current_in / parts["capacitance"], derivative.ravel()]

    for q in np.random.default_rng(4100).choice([-1., 1.], 100):
        solution = solve_ivp(lambda t, x: ode(t, x, q), (0, .02), current,
                             rtol=1e-9, atol=1e-12, dense_output=True)
        assert solution.success
        states = solution.sol(np.arange(4) * .005).T[:, 3:].reshape(4, 4, 4)
        values.extend(states @ rx_c)
        current = solution.y[:, -1]
    exact = response_basis(strain, layout, geom) @ code
    np.testing.assert_allclose(exact.reshape(400, 4), values, atol=2e-11)
    np.testing.assert_allclose(response_basis(strain, layout, geom, codes=code[:, None])[:, 0], exact,
                               atol=1e-15)


def test_joint_constraints_and_uncoupled_wiring_invariance():
    layout, geom = fixture()
    circuit = Circuit(coupling=0)
    strain = np.array([-1e-4, 2e-4, 3e-5])
    parts = components(strain, layout, geom, circuit)
    np.testing.assert_allclose(parts["gain_unsigned"],
        1 - (2 * parts["reference"] + parts["leakage"]) * parts["tau"] / parts["capacitance"],
        atol=2e-16)
    baseline = response_basis(strain, layout, geom, circuit)
    changed = response_basis(strain * 1.2, layout, geom, circuit)
    ones, signed = np.ones(3), np.array([1, -1, 1])
    assert np.isclose(np.linalg.norm((changed - baseline) @ ones),
                      np.linalg.norm((changed - baseline) @ signed))


def test_impedance_scaling_preserves_all_voltage_records():
    layout, geom = fixture()
    strain = np.array([-1e-4, 2e-4, 3e-5])
    standard = Circuit()
    scaled = Circuit(resistance=5000., leakage_resistance=500000.)
    a = response_basis(strain, layout, geom, standard)
    b = response_basis(strain, layout, geom, scaled)
    np.testing.assert_allclose(a, b, atol=2e-16)


def test_projection_and_bit_search_against_direct_least_squares():
    rng = np.random.default_rng(18)
    contrasts = rng.normal(size=(4, 37, 7))
    nuisance = rng.normal(size=(3, 37, 7))
    starts = [np.ones(7), rng.choice([-1, 1], 7)]
    for code in starts:
        jacobian = np.einsum("pti,i->tp", nuisance, code)
        expected = []
        for contrast in contrasts:
            delta = contrast @ code
            coef = np.linalg.lstsq(jacobian, delta, rcond=None)[0]
            expected.append(np.linalg.norm(delta - jacobian @ coef)**2)
        actual = record_scores(np.einsum("mti,i->tm", contrasts, code), jacobian, "residual")
        np.testing.assert_allclose(actual, np.mean(np.log(expected)))
    chosen, info = synthesize(contrasts, nuisance, starts)
    final = info["score"]
    for i in range(7):
        alternative = chosen.copy()
        alternative[i] *= -1
        alternative_score = record_scores(np.einsum("mti,i->tm", contrasts, alternative),
                                           np.einsum("pti,i->tp", nuisance, alternative), "residual")
        assert alternative_score <= final + 1e-12
