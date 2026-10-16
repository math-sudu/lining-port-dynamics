"""Response-constrained synthesis of passive bridges at fixed gauge positions.

Each gauge has a series ballast and parallel shunt. Their values preserve the
nominal cell resistance while attenuating its first derivative. Gauge signs are
physical bridge-arm assignments; no combination of digitized gauge signals is
used by this circuit model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import svd
from scipy.optimize import minimize

from .bridge_network import Circuit


ARM_SIGN = np.array([-1., 1., 1., -1.])
BOLTZMANN = 1.380649e-23


def response_spaces(healthy_basis):
    """Orthonormal healthy and null bases, independent of load-column scaling."""
    basis = np.asarray(healthy_basis, dtype=float)
    if basis.ndim != 2 or not np.isfinite(basis).all():
        raise ValueError("A finite gauge-by-load response matrix is required")
    lengths = np.linalg.norm(basis, axis=0)
    normalized = basis[:, lengths > 0] / lengths[lengths > 0]
    u, values, _ = svd(normalized, full_matrices=True)
    tolerance = max(normalized.shape) * np.finfo(float).eps * values[0] if len(values) else 0.
    rank = np.count_nonzero(values > tolerance)
    return u[:, :rank], u[:, rank:]


def target_null_weight(healthy_basis, target_strain):
    """Maximize target energy for one unit-L2 linear null row, then scale to L-inf 1.

    This is a linear-reference optimum, not an optimum over physical circuits.
    Columns of target_strain are equally weighted design target increments.
    """
    healthy, null = response_spaces(healthy_basis)
    targets = np.asarray(target_strain, dtype=float)
    if targets.ndim != 2 or targets.shape[0] != healthy.shape[0]:
        raise ValueError("Targets must have one row per gauge")
    if null.shape[1] == 0:
        raise ValueError("The healthy response spans all gauges in this group")
    projected = null.T @ targets
    directions, values, _ = svd(projected, full_matrices=False)
    if not len(values) or values[0] <= 100 * np.finfo(float).eps * np.linalg.norm(targets):
        raise ValueError("The supplied targets are inseparable from healthy response")
    weight = null @ directions[:, 0]
    pivot = np.argmax(np.abs(weight))
    weight /= weight[pivot]
    return weight, {
        "healthy_rank": healthy.shape[1], "null_dimension": null.shape[1],
        "target_projected_energy_fraction": float(np.linalg.norm(projected)**2 / np.linalg.norm(targets)**2),
        "single_row_energy_fraction": float(values[0]**2 / np.linalg.norm(targets)**2),
    }


@dataclass(frozen=True)
class TrimmedBridge:
    gauge_indices: np.ndarray
    arm: np.ndarray
    attenuation: np.ndarray
    arm_units: int
    nominal_ratio: np.ndarray | None = None

    def __post_init__(self):
        indices, arms, attenuation = self.gauge_indices, self.arm, self.attenuation
        if (len(indices) != len(arms) or len(indices) != len(attenuation)
                or len(np.unique(indices)) != len(indices)
                or not np.isin(arms, np.arange(4)).all()
                or not np.isfinite(attenuation).all()
                or np.any((attenuation < 0) | (attenuation > 1))
                or np.any(self.resistance_ratio <= 0)
                or not np.isfinite(self.resistance_ratio).all()
                or self.resistance_ratio.shape != attenuation.shape
                or np.any(attenuation > self.resistance_ratio)
                or self.arm_units < np.bincount(arms, minlength=4).max()):
            raise ValueError("Invalid gauge assignment, attenuation or arm capacity")
        if any(np.sum(np.sqrt(self.resistance_ratio[arms == j] * attenuation[arms == j]))
               > self.arm_units + 1e-13 for j in range(4)):
            raise ValueError("Nominal gauge cells exceed the available arm ballast")

    @property
    def resistance_ratio(self):
        """Measured zero-strain gauge resistance divided by the reference R0."""
        return np.ones(len(self.attenuation)) if self.nominal_ratio is None else np.asarray(self.nominal_ratio)

    @classmethod
    def from_weight(cls, gauge_indices, weight):
        """Realize signed weights with unique gauge use and four balanced arms."""
        weight = np.asarray(weight, dtype=float)
        if not np.isfinite(weight).all() or np.max(np.abs(weight)) == 0:
            raise ValueError("A finite, nonzero weight vector is required")
        arms = np.empty(len(weight), dtype=int)
        for positive, pair in ((True, (1, 2)), (False, (0, 3))):
            selected = np.flatnonzero((weight >= 0) == positive)
            arms[selected] = np.resize(pair, len(selected))
        return cls(np.asarray(gauge_indices, dtype=int), arms, np.abs(weight) / np.max(np.abs(weight)),
                   int(np.bincount(arms, minlength=4).max()))

    @property
    def weight(self):
        return ARM_SIGN[self.arm] * self.attenuation

    def arm_resistance(self, strain, circuit=Circuit(), attenuation=None):
        """Exact arm resistance, including unused-slot fixed ballast."""
        strain = np.asarray(strain, dtype=float)[..., self.gauge_indices]
        x = circuit.gauge_factor * strain
        if not np.isfinite(x).all() or np.any(x <= -1):
            raise ValueError("The gauge resistance must be finite and positive")
        c = self.attenuation if attenuation is None else np.asarray(attenuation)
        deviation = c * x / (1 + (1 - np.sqrt(c / self.resistance_ratio)) * x)
        return circuit.resistance * np.stack([
            self.arm_units + deviation[..., self.arm == j].sum(axis=-1) for j in range(4)], axis=-1)

    def voltage(self, strain, circuit=Circuit(), attenuation=None):
        r = self.arm_resistance(strain, circuit, attenuation)
        return 2 * circuit.voltage * (r[..., 1] / (r[..., 0] + r[..., 1])
                                      - r[..., 3] / (r[..., 2] + r[..., 3]))

    def voltage_and_gradient(self, strain, circuit=Circuit(), attenuation=None):
        """Exact voltage and its derivative with respect to attenuation c."""
        c = self.attenuation if attenuation is None else np.asarray(attenuation)
        x = circuit.gauge_factor * np.asarray(strain)[..., self.gauge_indices]
        root = np.sqrt(c / self.resistance_ratio)
        denominator = 1 + (1 - root) * x
        cell = c * x / denominator
        cell_gradient = x / denominator + .5 * root * x**2 / denominator**2
        arms = np.stack([self.arm_units + cell[..., self.arm == j].sum(axis=-1)
                         for j in range(4)], axis=-1)
        left, right = arms[..., 0] + arms[..., 1], arms[..., 2] + arms[..., 3]
        arm_gradient = 2 * circuit.voltage * np.stack([
            -arms[..., 1] / left**2, arms[..., 0] / left**2,
            arms[..., 3] / right**2, -arms[..., 2] / right**2], axis=-1)
        voltage = 2 * circuit.voltage * (arms[..., 1] / left - arms[..., 3] / right)
        return voltage, arm_gradient[..., self.arm] * cell_gradient

    def linear_voltage(self, strain, circuit=Circuit()):
        return (2 * circuit.voltage * circuit.gauge_factor / (4 * self.arm_units)
                * (np.asarray(strain)[..., self.gauge_indices] @ self.weight))

    def nonlinear_bound(self, strain, circuit=Circuit()):
        """Pointwise bound on |exact voltage - zero-strain linear voltage|."""
        x = circuit.gauge_factor * np.asarray(strain)[..., self.gauge_indices]
        c, q = self.attenuation, 1 - np.sqrt(self.attenuation / self.resistance_ratio)
        if np.any(q * np.abs(x) >= 1):
            raise ValueError("The stated bound requires |(1-sqrt(c)) F strain| < 1")
        cell_error = c * q * x**2 / (1 - q * np.abs(x))
        u = self.arm_resistance(strain, circuit) / (self.arm_units * circuit.resistance) - 1
        bound = cell_error.sum(axis=-1) / (4 * self.arm_units)
        for a, b in ((0, 1), (2, 3)):
            pair = np.abs(u[..., a] + u[..., b])
            if np.any(pair >= 2):
                raise ValueError("The stated divider bound requires pair deviation < 2")
            bound += np.abs(u[..., b] - u[..., a]) * pair / (4 * (2 - pair))
        return 2 * circuit.voltage * bound

    def mean_noise_variance(self, circuit, average_time_s, independent_sample_sigma_v,
                            sample_period_s, temperature_k=300.):
        """White Johnson noise plus an independent post-filter sample-noise model.

        The rectangular average has equivalent noise bandwidth 1/(2T).
        The electronics term follows the existing acquisition-study assumption.
        """
        thermal = 4 * BOLTZMANN * temperature_k * self.arm_units * circuit.resistance / (2 * average_time_s)
        electronics = independent_sample_sigma_v**2 * sample_period_s / average_time_s
        return thermal + electronics

    def resistor_netlist(self, strain, circuit=Circuit()):
        """Component-level DC netlist, rails H/L and measured nodes A/B.

        Fixed series elements are combined into one ballast per arm. At c=0
        the gauge is bypassed; at c=1 its shunt is absent. Both endpoints are
        represented topologically, without zero or infinite component values.
        """
        strain = np.asarray(strain)
        rows = []

        def add(a, b, resistance, kind, gauge=None):
            rows.append({"from": a, "to": b, "resistance_ohm": float(resistance),
                         "kind": kind, "gauge_index": gauge})

        for j, (start, end) in enumerate((("H", "A"), ("A", "L"), ("H", "B"), ("B", "L"))):
            members = np.flatnonzero((self.arm == j) & (self.attenuation > 0))
            ballast = circuit.resistance * (self.arm_units - np.sqrt(
                self.resistance_ratio[members] * self.attenuation[members]).sum())
            current = start
            for position, member in enumerate(members):
                index = int(self.gauge_indices[member])
                c = self.attenuation[member]
                node = end if position == len(members) - 1 and ballast == 0 else f"arm{j}_cell{position}"
                nominal = circuit.resistance * self.resistance_ratio[member]
                r_gauge = nominal * (1 + circuit.gauge_factor * strain[index])
                if r_gauge <= 0:
                    raise ValueError("The gauge resistance must be positive")
                add(current, node, r_gauge, "gauge", index)
                if c < self.resistance_ratio[member]:
                    root = np.sqrt(c / self.resistance_ratio[member])
                    add(current, node, nominal * root / (1 - root), "shunt", index)
                current = node
            if ballast:
                add(current, end, ballast, "arm_ballast")
        return rows


def optimize_exact_bridge(bridge, healthy_basis, calibration_strain, changed_design_strain,
                          circuit, noise_variance, max_attenuation=1., target_aggregation="mean"):
    """Locally maximize target energy / (healthy energy + noise), retaining W H=0.

    The installed gauges, signs and bridge arms stay fixed. Only physical series
    and shunt resistor values change. Target design scenarios must be independent
    of held-out evaluation cases. No global optimality is asserted.
    """
    healthy, null = response_spaces(np.asarray(healthy_basis)[bridge.gauge_indices])
    if (max_attenuation <= 0 or max_attenuation > min(1., bridge.resistance_ratio.min())
            or np.any(bridge.attenuation > max_attenuation)):
        raise ValueError("The attenuation bound must contain the physical initial design")
    constraint = healthy.T * ARM_SIGN[bridge.arm]
    reference = np.asarray(calibration_strain)
    changed = np.asarray(changed_design_strain)
    if target_aggregation not in ("mean", "geometric"):
        raise ValueError("Target aggregation must be mean or geometric")
    if target_aggregation == "geometric" and changed.ndim != 3:
        raise ValueError("Geometric aggregation requires scenario-by-load-by-gauge targets")
    if target_aggregation == "geometric":
        increments = (changed - reference)[..., bridge.gauge_indices]
        projected = increments @ null
        resolved = np.linalg.norm(projected, axis=(1, 2)) > (100 * np.finfo(float).eps
                    * np.linalg.norm(increments, axis=(1, 2)))
        if not np.any(resolved):
            raise ValueError("No target scenario is separable in this bridge group")
        changed = changed[resolved]

    def objective(c):
        v_h, d_h = bridge.voltage_and_gradient(reference, circuit, c)
        v_c, d_c = bridge.voltage_and_gradient(changed, circuit, c)
        delta, d_delta = v_c - v_h, d_c - d_h
        residual = np.mean(v_h**2) + noise_variance
        d_residual = 2 * np.mean(v_h[..., None] * d_h, axis=tuple(range(v_h.ndim)))
        if target_aggregation == "mean":
            target = np.mean(delta**2)
            d_target = 2 * np.mean(delta[..., None] * d_delta, axis=tuple(range(delta.ndim)))
        else:
            target = np.mean(delta**2, axis=1)
            d_target = 2 * np.mean(delta[..., None] * d_delta, axis=1)
        if np.any(target == 0):
            return np.inf, np.zeros_like(c)
        log_target = np.mean(np.log(target))
        d_log_target = d_target / target if np.ndim(target) == 0 else np.mean(d_target / target[:, None], axis=0)
        return np.log(residual) - log_target, d_residual / residual - d_log_target

    initial_value = objective(bridge.attenuation)[0]
    fit = minimize(objective, bridge.attenuation, jac=True, method="SLSQP",
                   bounds=[(0., max_attenuation)] * len(bridge.attenuation),
                   constraints={"type": "eq", "fun": lambda c: constraint @ c,
                                "jac": lambda c: constraint},
                   options={"ftol": 1e-10, "maxiter": 400})
    if not fit.success:
        raise RuntimeError(f"Passive bridge optimization failed: {fit.message}")
    # SLSQP represents active bounds with small numerical errors. Restore the
    # exact bypass/bare-gauge topology, then enforce the linear equality using
    # only genuinely free coefficients. This avoids fictitious near-zero series
    # resistors and near-infinite shunts in the exported component netlist.
    c = fit.x.copy()
    endpoint_tolerance = np.sqrt(np.finfo(float).eps)
    c[c < endpoint_tolerance] = 0.
    c[max_attenuation - c < endpoint_tolerance] = max_attenuation
    free = (c > 0) & (c < max_attenuation)
    # The remaining columns can be rank deficient after several bounds activate.
    # Resolving singular values below the endpoint restoration scale would amplify
    # roundoff into order-one coefficient changes in otherwise feasible circuits.
    c[free] -= np.linalg.lstsq(constraint[:, free], constraint @ c, rcond=endpoint_tolerance)[0]
    if np.any((c < 0) | (c > max_attenuation)):
        raise RuntimeError(f"Active-bound restoration violates a physical attenuation bound: "
                           f"range={c.min():.16g},{c.max():.16g}, cap={max_attenuation:.16g}, "
                           f"null={np.max(np.abs(constraint @ c), initial=0.):.3g}")
    fitted = replace(bridge, attenuation=c)
    return fitted, {"iterations": int(fit.nit), "initial_design_ratio": float(np.exp(-initial_value)),
                    "fitted_design_ratio": float(np.exp(-objective(c)[0])),
                    "null_constraint_max": float(np.max(np.abs(constraint @ c), initial=0.))}


def distributed_target_weight(healthy_basis, target_strain, seed):
    """Seed a physical row across the resolved target directions.

    A leading singular vector can vanish on entire independent target regions.
    A fixed random combination of all projected target columns avoids that
    degeneracy before geometric-mean exact-voltage optimization. The seed is a
    design setting and never selected using held-out voltages.
    """
    _, null = response_spaces(healthy_basis)
    targets = np.asarray(target_strain)
    projected = null.T @ targets
    norms = np.linalg.norm(projected, axis=0)
    resolved = norms > 100 * np.finfo(float).eps * np.linalg.norm(targets)
    if not np.any(resolved):
        raise ValueError("The supplied targets are inseparable from healthy response")
    coefficients = np.random.default_rng(seed).normal(size=int(resolved.sum()))
    weight = null @ (projected[:, resolved] / norms[resolved]) @ coefficients
    return weight / weight[np.argmax(abs(weight))]
