"""Passive, bipolar resistor--capacitor cells with selectable rail connections.

Each cell has one strained resistor and one unstrained reference resistor.
Exchanging their +q/-q rail connections changes the drive sign, while the
conductance matrix, capacitances and all relaxation poles remain unchanged.
Voltages, conductances, capacitances and times are in V, S, F and s.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .network import receiver_matrices


@dataclass(frozen=True)
class Circuit:
    resistance: float = 100_000.0
    gauge_factor: float = 100.0
    leakage_resistance: float = 10_000_000.0
    coupling: float = 0.3
    voltage: float = 1.0


def components(strain, layout, geom, circuit=Circuit()):
    """Map physical resistors and capacitors to the joint node coefficients."""
    strain = np.asarray(strain, dtype=float)
    n = len(layout["tau0"])
    if strain.shape != (n,) or not np.isfinite(strain).all():
        raise ValueError("Expected one finite strain per cell")
    if (circuit.resistance <= 0 or circuit.leakage_resistance <= 0
            or circuit.coupling < 0 or circuit.voltage <= 0):
        raise ValueError("Resistances and voltage must be positive; coupling nonnegative")
    resistance = circuit.resistance * (1 + circuit.gauge_factor * strain)
    if np.any(resistance <= 0):
        raise ValueError("The linear piezoresistance law requires positive resistance")
    active = 1 / resistance
    reference = 1 / circuit.resistance
    leakage = 1 / circuit.leakage_resistance
    total0 = 2 * reference + leakage
    total = active + reference + leakage
    capacitance = total0 * np.asarray(layout["tau0"])
    if np.any(capacitance <= 0):
        raise ValueError("Capacitances must be positive")
    edges = circuit.coupling * total0 * np.asarray(geom["W"])
    conductance = np.diag(total + edges.sum(axis=1)) - edges
    return {
        "resistance": resistance, "active": active, "reference": reference,
        "leakage": leakage, "total": total, "capacitance": capacitance,
        "edges": edges, "conductance": conductance,
        "gain_unsigned": (active - reference) / total,
        "tau": capacitance / total, "normalized_edges": edges / total[:, None],
    }


def system(strain, layout, geom, circuit=Circuit()):
    """Return an augmented KCL system; each drive column is one wiring bit."""
    parts = components(strain, layout, geom, circuit)
    n = len(parts["active"])
    matrix = np.zeros((n + 16, n + 16))
    drive = np.zeros((n + 16, n))
    output = np.zeros((4, n + 16))
    matrix[:n, :n] = -parts["conductance"] / parts["capacitance"][:, None]
    drive[:n] = np.diag(circuit.voltage * (parts["active"] - parts["reference"])
                        / parts["capacitance"])
    rx_a, rx_b, rx_c = receiver_matrices()
    for port in range(4):
        block = slice(n + 4 * port, n + 4 * port + 4)
        matrix[block, block] = rx_a
        matrix[block, :n] = rx_b[:, None] * geom["H"][port]
        output[port, block] = rx_c
    return matrix, drive, output


def response_basis(strain, layout, geom, circuit=Circuit(), codes=None, reset_duration=None):
    """Propagate the exact response to wiring columns, from discharged cells.

Passing a code matrix propagates only those physical wirings. With a reset
duration, the initial state is the periodic state after that zero-input reset;
otherwise the cells and receiver filters start discharged.
"""
    matrix, drive, output = system(strain, layout, geom, circuit)
    if codes is not None:
        codes = np.asarray(codes, dtype=float)
        if codes.ndim != 2 or codes.shape[0] != drive.shape[1]:
            raise ValueError("codes must have one row per cell")
        if not np.isin(codes, [-1, 1]).all():
            raise ValueError("Physical wiring codes must contain only -1 and +1")
        drive = drive @ codes
    transition = expm(matrix * .005)
    equilibrium = np.linalg.solve(matrix, -drive)
    forcing = equilibrium - transition @ equilibrium
    state = np.zeros_like(drive)
    values = np.empty((400, 4, drive.shape[1]))
    probe = np.random.default_rng(4100).choice([-1., 1.], 100)
    if reset_duration is not None:
        if reset_duration < 0:
            raise ValueError("Reset duration must be nonnegative")
        for step in range(400):
            state = transition @ state + forcing * probe[step // 4]
        # End-of-probe = exp(A * cycle) @ previous_end + zero-start response.
        previous_end = np.linalg.solve(np.eye(len(matrix)) - expm(matrix * (2 + reset_duration)), state)
        state = expm(matrix * reset_duration) @ previous_end
    for step in range(400):
        values[step] = output @ state
        state = transition @ state + forcing * probe[step // 4]
    return values.reshape(1600, drive.shape[1])
