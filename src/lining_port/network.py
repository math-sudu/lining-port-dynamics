"""Assumed strain-modulated linear nodes with four external receiver channels.

This is a lumped functional model, not an implemented physical device. All
geometry and dynamics are private simulator inputs; only z, time, q and mask
belong to the deployed observation record. SI seconds/metres and dimensionless
voltage-type variables are used throughout.
"""

from __future__ import annotations

import json
import time as walltime
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm, expm_frechet
from scipy.sparse.csgraph import connected_components


def make_layout(seed: int, n: int = 64) -> dict:
    """Freeze area quantiles, tangential axes, gains and intrinsic time scales."""
    if n < 1:
        raise ValueError("A layout requires at least one node")
    rng = np.random.default_rng(seed)
    return {
        "seed": int(seed),
        "theta": rng.uniform(0.0, 2.0 * np.pi, n),
        "area_quantile": rng.uniform(0.0, 1.0, n),
        "a0": rng.uniform(0.9, 1.1, n),
        "tau0": np.exp(rng.uniform(np.log(0.02), np.log(0.08), n)),
    }


def _segment_inside(a: np.ndarray, b: np.ndarray, inner: float) -> bool:
    """Outer disk is convex; only the minimum segment radius needs testing."""
    delta = b - a
    denom = float(delta @ delta)
    fraction = 0.0 if denom == 0.0 else np.clip(-float(a @ delta) / denom, 0, 1)
    closest = a + fraction * delta
    return bool(closest @ closest >= inner * inner - 1e-12)


def geometry(layout: dict, radius: float, thickness: float) -> dict:
    """Map the same area quantiles to an annulus without rejection or rewiring."""
    if not (0 < thickness < 2 * radius):
        raise ValueError("Geometry requires 0 < thickness < 2 * radius")
    theta = np.asarray(layout["theta"], dtype=float)
    inner, outer = radius - thickness / 2, radius + thickness / 2
    radial = np.sqrt(inner**2 + layout["area_quantile"] * (outer**2 - inner**2))
    xy = radial[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
    n = len(theta)
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            distance = float(np.linalg.norm(xy[i] - xy[j]))
            if distance <= 0.6 and _segment_inside(xy[i], xy[j], inner):
                weights[i, j] = weights[j, i] = np.exp(-distance / 0.3)

    port_theta = np.arange(4) * np.pi / 2
    port_xy = inner * np.column_stack((np.cos(port_theta), np.sin(port_theta)))
    selected = np.full(4, -1, dtype=int)
    h = np.zeros((4, n))
    used = set()
    for port in range(4):
        distance = np.linalg.norm(xy - port_xy[port], axis=1)
        for idx in np.argsort(distance, kind="stable"):
            if distance[idx] > 0.5:
                break
            if int(idx) not in used and _segment_inside(port_xy[port], xy[idx], inner):
                selected[port] = int(idx)
                used.add(int(idx))
                h[port, idx] = 1.0
                break
    mask = selected >= 0
    count, labels = connected_components(weights, directed=False)
    reachable_components = set(labels[selected[mask]].tolist())
    reachable = np.array([int(value) in reachable_components for value in labels])
    degree = np.count_nonzero(weights, axis=1)
    diagnostics = {
        "nodes": n,
        "edges": int(np.count_nonzero(weights) // 2),
        "components": int(count),
        "degree_min": int(degree.min()),
        "degree_max": int(degree.max()),
        "degree_mean": float(degree.mean()),
        "degree": degree.tolist(),
        "component_sizes": np.bincount(labels).tolist(),
        "valid_ports": int(mask.sum()),
        "reachable_node_fraction": float(reachable.mean()),
        "status": "observable" if mask.any() else "unobservable_no_ports",
    }
    return {
        "theta": theta.copy(), "z": radial - radius, "xy": xy,
        "W": weights, "H": h, "mask": mask, "port_nodes": selected,
        "port_xy": port_xy, "component_labels": labels, "reachable": reachable,
        "radius": float(radius), "thickness": float(thickness),
        "diagnostics": diagnostics,
    }


def coverage(geom: dict, theta_d: float, sector_width: float = np.pi / 6) -> dict:
    """Geometric support-sector reachability; not a proof of FE invisibility."""
    angle = (geom["theta"] - theta_d + np.pi) % (2 * np.pi) - np.pi
    inside = np.abs(angle) <= sector_width / 2
    count = int(inside.sum())
    return {
        "sector_nodes": count,
        "reachable_sector_nodes": int(np.count_nonzero(inside & geom["reachable"])),
        "reachable_sector_fraction": float(geom["reachable"][inside].mean()) if count else None,
    }


def receiver_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous 4th-order 40 Hz Butterworth as two scaled oscillator stages.

    Each stage has transfer omega**2/(s**2+d*omega*s+omega**2).
    Scaling each velocity state by omega keeps integration states well scaled.
    """
    omega = 2 * np.pi * 40
    d1, d2 = np.sqrt(2 - np.sqrt(2)), np.sqrt(2 + np.sqrt(2))
    a = omega * np.array([[0., 1., 0., 0.], [-1., -d1, 0., 0.],
                          [0., 0., 0., 1.], [1., 0., -1., -d2]])
    b = omega * np.array([0., 1., 0., 0.])
    c = np.array([0., 0., 1., 0.])
    return a, b, c


def node_parameters(strain: np.ndarray, layout: dict, family: str, alpha: float,
                    strain_law: str = "tanh", strain_transfer: float = 1.0,
                    gain_sensitivity: float = .5) -> dict:
    """Node laws and their derivatives with respect to host strain.

    P2 uses the prescribed homogeneous 50-ms time constant. The alternative
    saturation law has the same range and slope at zero as tanh; it changes
    curvature without learning a function from support-state labels.
    """
    strain = np.asarray(strain, dtype=float)
    n = len(layout["a0"])
    if strain.shape != (n,) or not np.isfinite(strain).all():
        raise ValueError("strain must contain one finite circumferential strain per node")
    if family not in {"P0", "P1", "P2", "P3", "P4"}:
        raise ValueError("family must be P0, P1, P2, P3 or P4")
    if not 0 < strain_transfer <= 1 or not 0 <= gain_sensitivity < 1 or abs(alpha) > 1:
        raise ValueError("Expected 0 < transfer <= 1, 0 <= gain sensitivity < 1, |alpha| <= 1")
    u = strain_transfer * strain / 1e-4
    if strain_law == "tanh":
        s = np.tanh(u)
        derivative = (1 - s**2) * strain_transfer / 1e-4
    elif strain_law == "algebraic":
        s = u / np.sqrt(1 + u**2)
        derivative = (1 + u**2)**-1.5 * strain_transfer / 1e-4
    else:
        raise ValueError("strain_law must be tanh or algebraic")
    gain = np.asarray(layout["a0"]) * (1 + gain_sensitivity * s)
    tau0 = np.full(n, .05) if family == "P2" else np.asarray(layout["tau0"])
    tau = tau0 * (np.exp(alpha * s) if family == "P4" else 1.0)
    if np.any(tau <= 0) or not np.isfinite(tau).all():
        raise ValueError("All intrinsic relaxation times must be positive and finite")
    return {"gain": gain, "tau": tau,
            "gain_prime": np.asarray(layout["a0"]) * gain_sensitivity * derivative,
            "tau_prime": tau * alpha * derivative if family == "P4" else np.zeros(n)}


def _system(strain: np.ndarray, geom: dict, layout: dict, family: str,
            kappa: float, alpha: float, strain_law: str = "tanh",
            strain_transfer: float = 1.0, gain_sensitivity: float = .5
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("Expected finite kappa >= 0")
    node = node_parameters(strain, layout, family, alpha, strain_law,
                           strain_transfer, gain_sensitivity)
    gain, tau = node["gain"], node["tau"]
    n = len(gain)
    rx_a, rx_b, rx_c = receiver_matrices()
    h = np.asarray(geom["H"])
    dynamic_n = 0 if family in {"P0", "P1"} else n
    size = dynamic_n + 16
    matrix = np.zeros((size, size))
    drive = np.zeros(size)
    output = np.zeros((4, size))
    w = np.asarray(geom["W"])
    network = np.eye(n) + kappa * (np.diag(w.sum(axis=1)) - w)
    if dynamic_n:
        matrix[:n, :n] = -network / tau[:, None]
        drive[:n] = gain / tau
    static_gain = np.linalg.solve(network, gain) if family == "P1" else gain
    for r in range(4):
        index = slice(dynamic_n + 4 * r, dynamic_n + 4 * r + 4)
        matrix[index, index] = rx_a
        output[r, index] = rx_c
        if dynamic_n:
            matrix[index, :n] = rx_b[:, None] * h[r]
        else:
            drive[index] = rx_b * float(h[r] @ static_gain)
    parameters = {
        "node_states": dynamic_n, "tau_min_s": float(tau.min()) if dynamic_n else None,
        "tau_max_s": float(tau.max()) if dynamic_n else None, "gain_min": float(gain.min()),
        "gain_max": float(gain.max()),
    }
    return matrix, drive, output, parameters


@lru_cache(maxsize=64)
def _transitions(matrix_bytes: bytes, size: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.frombuffer(matrix_bytes, dtype=np.float64).reshape(size, size)
    return expm(matrix * 0.005), expm(matrix * 2.0)


def _propagate(matrix: np.ndarray, drive: np.ndarray, output: np.ndarray,
               initial: np.ndarray, method: str, rtol: float, atol: float,
               max_step: float) -> tuple[np.ndarray, dict]:
    """Reset for 2 s then 100 PRBS chips; samples are t_probe = 0:0.005:1.995."""
    probe = np.random.default_rng(4100).choice([-1.0, 1.0], size=100)
    values = np.empty((400, 4))
    if method == "expm":
        transition, reset_transition = _transitions(np.ascontiguousarray(matrix).tobytes(), len(matrix))
        equilibrium = np.linalg.solve(matrix, -drive)
        forcing = equilibrium - transition @ equilibrium
        current = reset_transition @ initial
        reset_residual = float(np.max(np.abs(current)))
        for k in range(400):
            values[k] = output @ current
            current = transition @ current + forcing * probe[k // 4]
        return values, {"method": "matrix_exponential_piecewise_exact", "nfev": 0,
                        "integration_steps": 400, "reset_state_max": reset_residual}
    if method != "adaptive":
        raise ValueError("method must be adaptive or expm")
    nfev = 0
    steps = 0
    reset = solve_ivp(lambda t, y: matrix @ y, (0., 2.), initial,
                      method="DOP853", rtol=rtol, atol=atol, max_step=max_step)
    if not reset.success:
        raise RuntimeError(f"Network reset did not converge: {reset.message}")
    current = reset.y[:, -1]
    nfev += reset.nfev
    steps += len(reset.t) - 1
    reset_residual = float(np.max(np.abs(current)))
    for chip in range(100):
        # Local time prevents floating-point ambiguity at PRBS jumps. The left
        # endpoint output is continuous, whereas q is right-continuous.
        sol = solve_ivp(lambda t, y: matrix @ y + drive * probe[chip],
                        (0., .02), current, method="DOP853", rtol=rtol,
                        atol=atol, max_step=max_step, dense_output=True)
        if not sol.success:
            raise RuntimeError(f"Network chip {chip} did not converge: {sol.message}")
        values[4 * chip:4 * chip + 4] = (output @ sol.sol(np.arange(4) * .005)).T
        current = sol.y[:, -1]
        nfev += sol.nfev
        steps += len(sol.t) - 1
    return values, {"method": "DOP853_piecewise_at_input_jumps", "nfev": nfev,
                    "integration_steps": steps, "reset_state_max": reset_residual}


def simulate(strain: np.ndarray, geometry: dict, layout: dict, family: str = "P3",
             kappa: float = .3, alpha: float = .5, noise_seed: int = 0,
             noise_sigma: float = .01, rtol: float = 1e-7, atol: float = 1e-9,
             max_step: float = .005, method: str = "adaptive",
             initial_x: np.ndarray | None = None, strain_law: str = "tanh",
             strain_transfer: float = 1.0, gain_sensitivity: float = .5) -> dict:
    """Return finite external measurements plus private numerical diagnostics.

    Default initial node states are +1. The explicit two-second zero-input reset
    therefore exercises bounded initial-state decay instead of a free reset.
    Missing ports have zero-valued measurements and a separate false mask; an
    all-missing record is explicitly unobservable, never silently classified.
    z_clean is a simulation diagnostic, not a deployed classifier input.
    """
    start = walltime.perf_counter()
    if noise_sigma < 0 or rtol <= 0 or atol <= 0 or max_step <= 0:
        raise ValueError("Noise must be nonnegative and solver tolerances/step positive")
    matrix, drive, output, parameters = _system(strain, geometry, layout, family, kappa, alpha,
                                               strain_law, strain_transfer, gain_sensitivity)
    initial = np.zeros(len(matrix))
    n = parameters["node_states"]
    if n:
        initial[:n] = 1.0 if initial_x is None else np.asarray(initial_x, dtype=float)
    if not np.isfinite(initial).all():
        raise ValueError("Initial states must be finite")
    clean, solver = _propagate(matrix, drive, output, initial, method, rtol, atol, max_step)
    mask = np.asarray(geometry["mask"], dtype=bool)
    clean *= mask[None, :]
    rng = np.random.default_rng(noise_seed)
    z = (clean + rng.normal(0, noise_sigma, clean.shape)) * mask[None, :]
    if not np.isfinite(z).all():
        raise RuntimeError("Nonfinite external observation")
    diagnostics = {
        **parameters, **solver, "rtol": rtol, "atol": atol,
        "max_step_s": max_step, "wall_time_s": walltime.perf_counter() - start,
        "family": family, "kappa": 0.0 if family == "P0" else kappa,
        "alpha": alpha if family == "P4" else 0.0, "noise_seed": int(noise_seed),
        "strain_law": strain_law, "strain_transfer": strain_transfer,
        "gain_sensitivity": gain_sensitivity,
        "noise_sigma_fixed_reference": noise_sigma,
        "status": "ok" if mask.any() else "unobservable_no_ports",
        "state_dimension": len(matrix), "valid_ports": int(mask.sum()),
        "reset_duration_s": 2.0, "probe_duration_s": 2.0,
    }
    return {
        "z": z, "z_clean": clean, "time": np.arange(400) / 200,
        "q": np.repeat(np.random.default_rng(4100).choice([-1., 1.], 100), 4),
        "mask": mask.copy(), "diagnostics": diagnostics,
    }


def _tangent_response(matrix, drive, output, initial, dmatrix, ddrive):
    """Differentiate exact propagation, including the finite zero-input reset."""
    if np.any(dmatrix):
        transition, dtransition = expm_frechet(matrix * .005, dmatrix * .005)
        reset, dreset = expm_frechet(matrix * 2., dmatrix * 2.)
    else:
        transition, reset = _transitions(np.ascontiguousarray(matrix).tobytes(), len(matrix))
        dtransition, dreset = np.zeros_like(matrix), np.zeros_like(matrix)
    equilibrium = np.linalg.solve(matrix, -drive)
    dequilibrium = np.linalg.solve(matrix, -ddrive - dmatrix @ equilibrium)
    forcing = equilibrium - transition @ equilibrium
    dforcing = dequilibrium - dtransition @ equilibrium - transition @ dequilibrium
    current, tangent = reset @ initial, dreset @ initial
    values = np.empty((400, 4))
    probe = np.random.default_rng(4100).choice([-1., 1.], 100)
    for k in range(400):
        values[k] = output @ tangent
        tangent = transition @ tangent + dtransition @ current + dforcing * probe[k // 4]
        current = transition @ current + forcing * probe[k // 4]
    return values


def strain_directional_response(strain: np.ndarray, direction: np.ndarray,
                                geometry: dict, layout: dict, family: str = "P4",
                                kappa: float = .3, alpha: float = .5,
                                strain_law: str = "tanh", strain_transfer: float = 1.,
                                gain_sensitivity: float = .5) -> dict:
    """Return exact directional derivatives split into gain and time pathways.

    Geometry, node positions, edges, receivers and the initial +1 node states
    are held fixed. A mechanical geometry derivative must therefore be evaluated
    separately, not silently represented by this strain-only derivative.
    """
    node = node_parameters(strain, layout, family, alpha, strain_law,
                           strain_transfer, gain_sensitivity)
    direction = np.asarray(direction, dtype=float)
    if direction.shape != node["gain"].shape or not np.isfinite(direction).all():
        raise ValueError("direction must contain one finite strain change per node")
    a, b, c, parameters = _system(strain, geometry, layout, family, kappa, alpha,
                                 strain_law, strain_transfer, gain_sensitivity)
    initial = np.zeros(len(a))
    n = parameters["node_states"]
    initial[:n] = 1.
    dgain, dtau = node["gain_prime"] * direction, node["tau_prime"] * direction
    da_time, db_gain, db_time = np.zeros_like(a), np.zeros_like(b), np.zeros_like(b)
    if n:
        ratio = dtau / node["tau"]
        da_time[:n, :n] = -ratio[:, None] * a[:n, :n]
        db_time[:n] = -ratio * b[:n]
        db_gain[:n] = dgain / node["tau"]
    else:
        if family == "P1":
            w = np.asarray(geometry["W"])
            dgain = np.linalg.solve(np.eye(len(dgain)) + kappa * (np.diag(w.sum(1)) - w), dgain)
        _, rx_b, _ = receiver_matrices()
        for r in range(4):
            db_gain[4*r:4*r+4] = rx_b * float(geometry["H"][r] @ dgain)
    gain = _tangent_response(a, b, c, initial, np.zeros_like(a), db_gain)
    time = _tangent_response(a, b, c, initial, da_time, db_time) if np.any(dtau) else np.zeros_like(gain)
    mask = np.asarray(geometry["mask"])[None, :]
    return {"gain": gain * mask, "time": time * mask, "total": (gain + time) * mask}


def _difference(a: np.ndarray, b: np.ndarray) -> dict:
    return {"relative_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
            "max_absolute": float(np.max(np.abs(a - b)))}


def run_network_validation(output_dir: str | Path) -> dict:
    """Execute numerical/analytic controls and save actual validation results."""
    checks = {}
    layout = make_layout(3100)
    geom = geometry(layout, 3., .30)
    strain = 1e-4 * np.cos(2 * layout["theta"]) - 5e-5

    # All five implemented families share the full continuous receiver chain.
    for family, kappa in [("P0", 0.), ("P3", 0.), ("P3", .3), ("P4", 0.), ("P4", .3)]:
        adaptive = simulate(strain, geom, layout, family, kappa, noise_sigma=0.)
        exact = simulate(strain, geom, layout, family, kappa, noise_sigma=0., method="expm")
        difference = _difference(adaptive["z"], exact["z"])
        checks[f"full64_{family}_k{kappa}_adaptive_vs_expm"] = {
            **difference, "adaptive_s": adaptive["diagnostics"]["wall_time_s"],
            "expm_s": exact["diagnostics"]["wall_time_s"],
            "nfev": adaptive["diagnostics"]["nfev"],
            "pass": difference["relative_l2"] < 1e-3 and difference["max_absolute"] < 1e-4,
        }

    base = simulate(strain, geom, layout, noise_sigma=0.)
    refined = simulate(strain, geom, layout, noise_sigma=0., rtol=1e-9, atol=1e-11, max_step=.0025)
    difference = _difference(base["z"], refined["z"])
    checks["half_max_step_stricter_tolerance"] = {**difference, "pass": difference["relative_l2"] < 1e-3 and difference["max_absolute"] < 1e-4}

    p4zero = simulate(strain, geom, layout, family="P4", alpha=0., noise_seed=100)
    p3 = simulate(strain, geom, layout, family="P3", noise_seed=100)
    checks["alpha_zero_equals_fixed_tau"] = {**_difference(p4zero["z"], p3["z"]), "pass": bool(np.array_equal(p4zero["z"], p3["z"]))}
    repeat = simulate(strain, geom, layout, family="P3", noise_seed=100)
    checks["fixed_seed_bitwise_repeat"] = {"pass": bool(np.array_equal(repeat["z"], p3["z"]))}
    zero0 = simulate(np.zeros(64), geom, layout, family="P4", noise_seed=105)
    zero1 = simulate(np.zeros(64), geom, layout, family="P4", noise_seed=105)
    checks["zero_strain_label_independence"] = {"pass": bool(np.array_equal(zero0["z"], zero1["z"])), "max_absolute": float(np.max(np.abs(zero0["z"] - zero1["z"]))) }

    # Independent small-N augmented-state exponential includes q as a constant
    # state; it does not use the equilibrium-forcing implementation above.
    tiny = make_layout(11, 3)
    tiny_geom = {"W": np.array([[0., .7, 0.], [.7, 0., .2], [0., .2, 0.]]),
                 "H": np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 0., 0.]]),
                 "mask": np.array([True, True, True, False])}
    eps = np.array([-1e-4, 2e-5, 3e-5])
    a, b, c, params = _system(eps, tiny_geom, tiny, "P4", .3, .5)
    augmented = np.zeros((len(a) + 1, len(a) + 1))
    augmented[:-1, :-1], augmented[:-1, -1] = a, b
    propagator = expm(augmented * .005)
    initial = np.r_[np.ones(3), np.zeros(16)]
    current = np.r_[expm(a * 2.) @ initial, 0.]
    exact_small = []
    q = np.repeat(np.random.default_rng(4100).choice([-1., 1.], 100), 4)
    for value in q:
        current[-1] = value
        exact_small.append(c @ current[:-1])
        current = propagator @ current
    adaptive_small = simulate(eps, tiny_geom, tiny, family="P4", noise_sigma=0.)
    difference = _difference(adaptive_small["z"], np.asarray(exact_small))
    checks["small_augmented_exponential_including_receiver"] = {**difference, "pass": difference["relative_l2"] < 1e-3 and difference["max_absolute"] < 1e-4}

    # Homogeneous tau + all-node mean removes symmetric diffusive coupling.
    mean_layout = {**tiny, "tau0": np.full(3, .05)}
    mean_geom = {**tiny_geom, "H": np.full((4, 3), 1. / 3), "mask": np.ones(4, dtype=bool)}
    uncoupled = simulate(eps, mean_geom, mean_layout, kappa=0., noise_sigma=0., method="expm")
    coupled = simulate(eps, mean_geom, mean_layout, kappa=3., noise_sigma=0., method="expm")
    difference = _difference(uncoupled["z"], coupled["z"])
    checks["homogeneous_mean_coupling_counterexample"] = {**difference, "pass": difference["max_absolute"] < 1e-10}

    isolated_geom = {**tiny_geom, "W": np.array([[0., .7, 0.], [.7, 0., 0.], [0., 0., 0.]]),
                     "H": np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
                     "mask": np.array([True, True, False, False])}
    changed_eps = eps.copy()
    changed_eps[2] += 5e-4
    first = simulate(eps, isolated_geom, tiny, family="P4", noise_sigma=0., method="expm")
    changed = simulate(changed_eps, isolated_geom, tiny, family="P4", noise_sigma=0., method="expm")
    difference = _difference(first["z"], changed["z"])
    checks["unreachable_component_change_is_invisible"] = {**difference, "pass": difference["max_absolute"] < 1e-10}

    # Tau is bounded by 80 ms * exp(1), including the largest permitted alpha.
    # The maximum principle gives ||delta x(2)||_inf <= exp(-2/tau_max)
    # for initial bound one. The receiver is also integrated through reset.
    worst_layout = {**layout, "tau0": np.full(64, .08)}
    worst_eps = np.full(64, 1e-2)
    plus = simulate(worst_eps, geom, worst_layout, family="P4", alpha=1.,
                    initial_x=np.ones(64), noise_sigma=0., method="expm")
    zero = simulate(worst_eps, geom, worst_layout, family="P4", alpha=1.,
                    initial_x=np.zeros(64), noise_sigma=0., method="expm")
    residual = float(np.max(np.abs(plus["z"] - zero["z"])))
    bound = float(np.exp(-2 / (.08 * np.e)))
    checks["two_second_initial_state_decay"] = {"max_port_residual": residual,
        "node_sup_norm_bound": bound, "pass": residual < 1e-3,
        "interpretation": "Worst-case supported tau; finite reset residual is retained, not set to zero."}

    result = {"pass": bool(all(item["pass"] for item in checks.values())),
              "checks": checks, "layout_diagnostics": geom["diagnostics"],
              "sample_time": "relative to probe onset; 0 to 1.995 s, 200 Hz",
              "solver": "DOP853 adaptive primary; equivalent matrix-exponential acceleration",
              "physical_status": "Assumed functional model, unvalidated physical device"}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "network_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
