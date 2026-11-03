"""Component deviations, routed leads and a finite differential receiver.

The installed bridge remains passive. Individual resistances are available for
commissioning; the in-service observations are the bridge voltages only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BridgeHardware:
    bridge: object
    circuit: object
    gauge_nominal: np.ndarray
    shunt: np.ndarray
    ballast: np.ndarray
    arm_lead: np.ndarray
    supply_lead: float = 0.
    sense_lead: float = 0.
    input_resistance: float = np.inf

    @classmethod
    def from_design(cls, bridge, circuit, arm_lead=None, supply_lead=0., sense_lead=0.,
                    input_resistance=np.inf, actual_nominal_ratio=None):
        ratio = bridge.resistance_ratio
        root = np.sqrt(bridge.attenuation / ratio)
        active = bridge.attenuation > 0
        shunt = np.full(len(root), np.inf)
        finite = active & (root < 1)
        shunt[finite] = circuit.resistance * ratio[finite] * root[finite] / (1 - root[finite])
        arm_lead = np.zeros(4) if arm_lead is None else np.asarray(arm_lead)
        ballast = circuit.resistance * np.array([
            bridge.arm_units - np.sum(ratio[(bridge.arm == j) & active] * root[(bridge.arm == j) & active])
            for j in range(4)]) - arm_lead
        rounding = 8 * np.finfo(float).eps * circuit.resistance * bridge.arm_units
        if np.any(ballast < -rounding):
            raise ValueError("The prescribed arm resistance cannot accommodate its physical lead")
        ballast[abs(ballast) <= rounding] = 0.
        gauge = circuit.resistance * (ratio if actual_nominal_ratio is None else np.asarray(actual_nominal_ratio))
        return cls(bridge, circuit, gauge, shunt, ballast, arm_lead, supply_lead, sense_lead, input_resistance)

    def _components(self, effective_strain, shunt_error=0., ballast_error=0.,
                    gauge_drift=0., shunt_drift=0., ballast_drift=0.):
        x = self.circuit.gauge_factor * np.asarray(effective_strain)[..., self.bridge.gauge_indices]
        gauge = self.gauge_nominal * (1 + x) * (1 + np.asarray(gauge_drift))
        shunt = self.shunt * (1 + np.asarray(shunt_error)) * (1 + np.asarray(shunt_drift))
        if np.any(gauge <= 0) or np.any(shunt <= 0):
            raise ValueError("Physical component resistance must remain positive")
        cell = gauge / (1 + gauge / shunt)
        cell = np.where(self.bridge.attenuation > 0, cell, 0.)
        ballast = self.ballast * (1 + np.asarray(ballast_error)) * (1 + np.asarray(ballast_drift))
        if np.any(ballast < 0):
            raise ValueError("Physical ballast must remain nonnegative")
        return gauge, shunt, cell, ballast

    def arm_resistance(self, effective_strain, **errors):
        _, _, cell, ballast = self._components(effective_strain, **errors)
        return np.stack([cell[..., self.bridge.arm == j].sum(axis=-1) for j in range(4)], axis=-1) + ballast + self.arm_lead

    def _unit_arms(self, r):
        g = 1 / r
        receiver = self.input_resistance + 2 * self.sense_lead
        load_g = 1 / receiver
        matrix = np.empty(r.shape[:-1] + (2, 2))
        matrix[..., 0, 0] = g[..., 0] + g[..., 1] + load_g
        matrix[..., 1, 1] = g[..., 2] + g[..., 3] + load_g
        matrix[..., 0, 1] = matrix[..., 1, 0] = -load_g
        # Unit-voltage rails. The network is linear in the excitation voltage.
        mids = np.linalg.solve(matrix, np.stack((g[..., 0], g[..., 2]), axis=-1)[..., None])[..., 0]
        input_g = (1 - mids[..., 0]) * g[..., 0] + (1 - mids[..., 1]) * g[..., 2]
        return mids, input_g, matrix

    def _solve_arms(self, r):
        mids, input_g, _ = self._unit_arms(r)
        excitation = 2 * self.circuit.voltage / (1 + 2 * self.supply_lead * input_g)
        return excitation[..., None] * mids, excitation, excitation * input_g

    def voltage(self, effective_strain, **errors):
        mids, _, _ = self._solve_arms(self.arm_resistance(effective_strain, **errors))
        receiver = self.input_resistance + 2 * self.sense_lead
        sense_ratio = 1. if np.isinf(self.input_resistance) else self.input_resistance / receiver
        return (mids[..., 0] - mids[..., 1]) * sense_ratio

    def nominal_linear_gain(self):
        """Gain multiplying the signed attenuation at nominal arm balance."""
        m, r0 = self.bridge.arm_units, self.circuit.resistance
        rail = 2 * self.circuit.voltage * m * r0 / (m * r0 + 2 * self.supply_lead)
        loading = 1 / (1 + (m * r0 + 2 * self.sense_lead) / self.input_resistance)
        return rail * self.circuit.gauge_factor / (4 * m) * loading

    def voltage_and_gradient(self, effective_strain, attenuation=None):
        """Design voltage and coefficient gradient with compensated nominal arms.

        Ballast is reset as coefficients change, retaining the arm scale and
        known leads. Differentiate the same loaded nodal system as voltage();
        fabrication errors and post-setting drift belong to evaluation.
        """
        bridge = self.bridge
        c = bridge.attenuation if attenuation is None else np.asarray(attenuation)
        r = bridge.arm_resistance(effective_strain, self.circuit, c)
        mids, input_g, matrix = self._unit_arms(r)
        g = 1 / r
        rhs = np.zeros(r.shape[:-1] + (2, 4))
        rhs[..., 0, 0] = -(1 - mids[..., 0]) * g[..., 0]**2
        rhs[..., 0, 1] = mids[..., 0] * g[..., 1]**2
        rhs[..., 1, 2] = -(1 - mids[..., 1]) * g[..., 2]**2
        rhs[..., 1, 3] = mids[..., 1] * g[..., 3]**2
        d_mid = np.linalg.solve(matrix, rhs)
        d_input = -g[..., 0, None] * d_mid[..., 0, :] - g[..., 2, None] * d_mid[..., 1, :]
        d_input[..., 0] += rhs[..., 0, 0]
        d_input[..., 2] += rhs[..., 1, 2]
        denominator = 1 + 2 * self.supply_lead * input_g
        excitation = 2 * self.circuit.voltage / denominator
        differential = mids[..., 0] - mids[..., 1]
        d_voltage = excitation[..., None] * (d_mid[..., 0, :] - d_mid[..., 1, :]
            - differential[..., None] * (2 * self.supply_lead / denominator)[..., None] * d_input)
        sense_ratio = (1. if np.isinf(self.input_resistance) else
                       self.input_resistance / (self.input_resistance + 2 * self.sense_lead))
        x = self.circuit.gauge_factor * np.asarray(effective_strain)[..., bridge.gauge_indices]
        root = np.sqrt(c / bridge.resistance_ratio)
        cell_denominator = 1 + (1 - root) * x
        d_arm = self.circuit.resistance * (x / cell_denominator
            + .5 * root * x**2 / cell_denominator**2)
        return (excitation * differential * sense_ratio,
                d_voltage[..., bridge.arm] * d_arm * sense_ratio)

    def operating_point(self, effective_strain, **errors):
        """Receiver potentials and component dissipation at the actual source rails."""
        gauge, shunt, cell, ballast = self._components(effective_strain, **errors)
        arms = np.stack([cell[..., self.bridge.arm == j].sum(axis=-1)
                         for j in range(4)], axis=-1) + ballast + self.arm_lead
        mids, excitation, supply_current = self._solve_arms(arms)
        drops = np.stack((excitation - mids[..., 0], mids[..., 0],
                          excitation - mids[..., 1], mids[..., 1]), axis=-1)
        currents = drops / arms
        cell_voltage = currents[..., self.bridge.arm] * cell
        sense_current = (mids[..., 0] - mids[..., 1]) / (self.input_resistance + 2 * self.sense_lead)
        receiver_nodes = mids + np.stack((-sense_current, sense_current), axis=-1) * self.sense_lead
        receiver_nodes += (supply_current * self.supply_lead - self.circuit.voltage)[..., None]
        voltage = receiver_nodes[..., 0] - receiver_nodes[..., 1]
        return {
            "voltage_v": voltage,
            "receiver_nodes_v": receiver_nodes,
            "source_power_w": 2 * self.circuit.voltage * supply_current,
            "gauge_power_w": cell_voltage**2 / gauge,
            "shunt_power_w": cell_voltage**2 / shunt,
            "ballast_power_w": currents**2 * ballast,
            "arm_wire_power_w": currents**2 * self.arm_lead,
            "supply_wire_power_w": 2 * supply_current**2 * self.supply_lead,
            "sense_wire_power_w": 2 * sense_current**2 * self.sense_lead,
            "receiver_power_w": voltage**2 / self.input_resistance,
        }

    def output_resistance(self, effective_strain, **errors):
        """Driving-point resistance at the receiver, with the source shorted.

        All physical resistors, including the receiver input, share the stated
        temperature when this resistance is used for Johnson noise.
        """
        r = self.arm_resistance(effective_strain, **errors)
        if self.supply_lead == 0:
            resistance = r[..., 0] * r[..., 1] / (r[..., 0] + r[..., 1])
            resistance += r[..., 2] * r[..., 3] / (r[..., 2] + r[..., 3])
        else:
            g = 1 / r
            matrix = np.zeros(r.shape[:-1] + (3, 3))
            matrix[..., 0, 0] = g[..., 0] + g[..., 2] + 1 / (2 * self.supply_lead)
            matrix[..., 1, 1] = g[..., 0] + g[..., 1]
            matrix[..., 2, 2] = g[..., 2] + g[..., 3]
            matrix[..., 0, 1] = matrix[..., 1, 0] = -g[..., 0]
            matrix[..., 0, 2] = matrix[..., 2, 0] = -g[..., 2]
            drive = np.broadcast_to([0., 1., -1.], r.shape[:-1] + (3,))
            potential = np.linalg.solve(matrix, drive[..., None])[..., 0]
            resistance = potential[..., 1] - potential[..., 2]
        resistance += 2 * self.sense_lead
        return resistance / (1 + resistance / self.input_resistance)

    def netlist(self, effective_strain, shunt_error=None, ballast_error=None):
        """Export explicit gauge, shunt, ballast, arm, power and sense leads."""
        bridge = self.bridge
        shunt_error = np.zeros(len(bridge.arm)) if shunt_error is None else shunt_error
        ballast_error = np.zeros(4) if ballast_error is None else ballast_error
        rows = []

        def add(start, end, r, kind, gauge=None):
            if r <= 0 or not np.isfinite(r):
                raise ValueError("Netlist elements must have positive finite resistance")
            rows.append({"from": start, "to": end, "resistance_ohm": float(r), "kind": kind, "gauge_index": gauge})

        high, low = ("RH", "RL") if self.supply_lead else ("H", "L")
        left, right = ("MA", "MB") if self.sense_lead else ("A", "B")
        if self.supply_lead:
            add("H", high, self.supply_lead, "supply_lead")
            add(low, "L", self.supply_lead, "supply_lead")
        for j, (start, end) in enumerate(((high, left), (left, low), (high, right), (right, low))):
            members = np.flatnonzero((bridge.arm == j) & (bridge.attenuation > 0))
            current = start
            if self.arm_lead[j]:
                current = f"W{j}"
                add(start, current, self.arm_lead[j], "arm_lead")
            for pos, member in enumerate(members):
                node = end if pos == len(members) - 1 and self.ballast[j] == 0 else f"arm{j}_cell{pos}"
                gauge = int(bridge.gauge_indices[member])
                add(current, node, self.gauge_nominal[member] * (1 + self.circuit.gauge_factor * effective_strain[gauge]), "gauge", gauge)
                if np.isfinite(self.shunt[member]):
                    add(current, node, self.shunt[member] * (1 + shunt_error[member]), "shunt", gauge)
                current = node
            if self.ballast[j]:
                add(current, end, self.ballast[j] * (1 + ballast_error[j]), "arm_ballast")
        if self.sense_lead:
            add(left, "A", self.sense_lead, "sense_lead")
            add(right, "B", self.sense_lead, "sense_lead")
        if np.isfinite(self.input_resistance):
            add("A", "B", self.input_resistance, "receiver_input")
        return rows


def route_bridge(bridge, cell_distance, center_distance, instrument_distance, depth, ohm_per_m):
    """Four series arm loops from a module head and four instrument conductors.

    The head is at the centerline station of the first gauge in the fixed group.
    Each arm follows the supplied member/arc shortest path between successive
    active cells in group order. Shunts are local and ballast is at the head.
    """
    head = int(bridge.gauge_indices[0])
    arms = []
    paths = []
    for j in range(4):
        nodes = bridge.gauge_indices[(bridge.arm == j) & (bridge.attenuation > 0)]
        length = 0.
        if len(nodes):
            length = (center_distance[head, nodes[0]] + abs(depth[nodes[0]])
                      + center_distance[nodes[-1], head] + abs(depth[nodes[-1]]))
            length += sum(cell_distance[a, b] for a, b in zip(nodes[:-1], nodes[1:]))
        arms.append(length)
        paths.append(nodes.tolist())
    lead = float(instrument_distance[head])
    return {"head_gauge_centerline": head, "arm_gauge_order": paths,
            "arm_length_m": arms, "one_instrument_lead_m": lead,
            "total_conductor_m": float(sum(arms) + 4 * lead),
            "arm_resistance_ohm": (np.array(arms) * ohm_per_m).tolist(),
            "one_instrument_lead_ohm": lead * ohm_per_m}
