"""Exercise the actual reader delivery outside the development checkout."""
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_standalone_component_reconstruction(tmp_path):
    for name in ("Frame_Network.json", "Frame_Design.npz"):
        shutil.copyfile(ROOT / "data/representative-frame" / name, tmp_path / name)
    for name in ("reconstruct_frame.py", "passive_hardware.py"):
        shutil.copyfile(ROOT / "src/lining_port" / name, tmp_path / name)
    subprocess.run([sys.executable, str(tmp_path / "reconstruct_frame.py"),
                    "--output", str(tmp_path / "voltages.npz")],
                   cwd=tmp_path, check=True)
    with np.load(tmp_path / "Frame_Design.npz") as original, \
            np.load(tmp_path / "voltages.npz") as rebuilt:
        for mode in ("scaled", "constrained"):
            np.testing.assert_allclose(
                rebuilt[mode + "_increment_v"], original[mode + "_expected_increment_v"],
                rtol=1e-7, atol=5e-13,
            )
    # Physical and effective coordinates must give the same voltage when the
    # per-gauge sensitivities are applied exactly once.
    code = """
import json
import numpy as np
from reconstruct_frame import voltage
record = json.load(open('Frame_Network.json'))['networks']['constrained']
ratio = np.empty(60)
for b in record['bridges']:
    for g in b['gauges']:
        ratio[g['gauge_index']] = g['gauge_factor'] / record['gauge_factor_reference']
strain = np.linspace(-2e-4, 3e-4, 60)
np.testing.assert_allclose(
    voltage(record, strain, coordinates='physical'),
    voltage(record, strain * ratio, coordinates='effective'), atol=1e-15)
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
