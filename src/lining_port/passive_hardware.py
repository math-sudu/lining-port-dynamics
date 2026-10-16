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
        if np.any(ballast < 0):
            raise ValueError("The prescribed arm resistance cannot accommodate its physical lead")
        gauge = circuit.resistance * (ratio if actual_nominal_ratio is None else np.asarray(actual_nominal_ratio))
        return cls(bridge, circuit, gauge, shunt, ballast, arm_lead, supply_lead, sense_lead, input_resistance)

    def arm_resistance(self, effective_strain, shunt_error=0., ballast_error=0.,
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
        return np.stack([cell[..., self.bridge.arm == j].sum(axis=-1) for j in range(4)], axis=-1) + ballast + self.arm_lead

    def voltage(self, effective_strain, **errors):
        r = self.arm_resistance(effective_strain, **errors)
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
        excitation = 2 * self.circuit.voltage / (1 + 2 * self.supply_lead * input_g)
        sense_ratio = 1. if np.isinf(self.input_resistance) else self.input_resistance / receiver
        return excitation * (mids[..., 0] - mids[..., 1]) * sense_ratio

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
