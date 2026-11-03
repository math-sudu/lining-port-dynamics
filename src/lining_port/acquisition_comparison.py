"""Unloaded series-arm bridge voltage used by the array initialization."""
import numpy as np


def bridge_voltage(strain, nodes, circuit):
    """Exact differential midpoint voltages for series-connected active arms.

    Leading strain dimensions are retained; the last dimension holds the gauges.
    Netlists index channel, arm and series positions.
    """
    r = circuit.resistance * (1 + circuit.gauge_factor * np.asarray(strain))
    if np.any(r <= 0):
        raise ValueError("Gauge resistance must be positive")
    arms = r[..., nodes].sum(axis=-1)
    return 2 * circuit.voltage * (
        arms[..., 1] / (arms[..., 0] + arms[..., 1])
        - arms[..., 3] / (arms[..., 2] + arms[..., 3]))
