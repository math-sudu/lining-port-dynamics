# Mechanics-matched bridge networks

Code and data for **Mechanics-matched bridge networks for load-rejecting strain acquisition in
tunnel linings**, by Pengcheng Zhu and Tielin Chen,
Beijing Jiaotong University.

The study constructs four composite bridges from 64 paired-depth gauges in a
fixed ring installation. Equal healthy arm ratios reject the prescribed
ordinary load space, while the signed outputs retain normalized curvature
differences between opposite station groups. A two-bridge connection, local
bridges, scanning and six RC wirings provide comparisons over the same loads
and acquisition cycle.

## Reproduction

The scientific runtime is Windows x64, Python 3.11 and the packages pinned in
`requirements-lock.txt`. Install with uv:

```powershell
uv venv --python 3.11 .venv
uv pip sync --python .venv/Scripts/python.exe requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
.venv/Scripts/python.exe -m lining_port.installation_study --mechanical
.venv/Scripts/python.exe -m lining_port.acquisition_comparison
.venv/Scripts/python.exe -m lining_port.composite_bridge_study
.venv/Scripts/python.exe -m lining_port.acquisition_routing
.venv/Scripts/python.exe -m pytest -q tests/test_installation.py tests/test_acquisition_comparison.py tests/test_acquisition_routing.py tests/test_bridge_network.py
```

The first command regenerates `results/fixed-installation/` from
`inputs/installation_task.json` and `inputs/device_node.json`. The second
uses those strain inputs and RC operating rates to generate the acquisition
comparison. The manuscript repository also supplies
`scripts/summarize_installation.py` to generate its tables and numerical facts.
The composite-bridge command constructs the healthy load basis, evaluates the
exact arm fractions on the existing cases, and examines low-response loads
between the random cases using new forward mechanics calculations.
The routing command enumerates all conductors to the four module-head
interfaces, updates acquisition resources and evaluates bridge lead resistance.

## Current files

| Path | Role |
| --- | --- |
| `src/lining_port/installation.py` | Four embedded ladders, routed lengths and port assignments |
| `installation_study.py` | Fixed mechanics, wiring, calibration and test evaluation |
| `acquisition_comparison.py` | Four-bank scanning, active mechanical bridges and resource calculations |
| `composite_bridge_study.py` | Composite-bridge outputs and weak-load mechanics |
| `acquisition_routing.py` | Complete conductor routes and bridge DC solution with resistive leads |
| `bridge_network.py`, `bridge_design.py` | KCL response matrices and fixed-wiring search |
| `fe.py`, `continuum.py` | Ring mechanics and strain recovery |
| `results/fixed-installation/` | Current study data, described below |
| `results/composite-bridges/` | Physical netlists, load basis and signed composite-bridge responses |
| `figures/` | Numerical source data for the manuscript figures |

The source modules above are in `src/lining_port/`.

In `results/composite-bridges/`, `summary.json` records the 1-based arm netlists,
sampling/noise model, evaluation rates and weak-load cases. `responses.npz`
stores the 64×3 healthy load basis (strain per Pa), signed outputs for all
healthy and target cases, softened load bases at 17 sector centers, and the
gauge/full-field strains used to explain a weak response. The case ordering is
in `results/fixed-installation/cases.json`; soft outputs have shape (32,5,4)
or (32,5,2). The last axis always denotes physical channels.

`routing.json` gives each route as a sequence of `[theta_rad, radius_m]`
coordinates, its conductor length and electrical role. IDs are zero-based.
For bridge arms, `gauges_rail_to_midpoint` fixes the terminal-to-terminal series
order. The scope includes shared supply feeds and all inter-module and receiver
connections, ending at the four module-head supply/receiver interfaces.
The four-bridge selection uses 104.56 m of conductor and 0.160 mJ nominal
resistive excitation per cycle; the two-channel candidate uses 111.61 m and
0.040 mJ. `lead_resistance` records the shared-feed calculation at 0.0787 ohm/m,
from the linked Belden 9501NH conductor specification.

| File in current results | Contents |
| --- | --- |
| `design.json`, `cases.json` | Fixed design, six wiring modes, independent calibration/test loads |
| `strains.npz` | Nominal, derivative, design-target and test-target strains at all 64 gauges |
| `wirings.json` | Six fixed codes per capacitor inventory and search histories |
| `responses.npz` | Components, routes, references, design templates and example records |
| `evaluation.json` | 3,072 norm records at reduction 0.45 |
| `scores.json`, `detection.json` | Fixed scores and expected rates at five support reductions |
| `summary.json` | Paired norm results |
| `acquisition.json`, `acquisition_responses.npz` | Scanning/bridge responses, operating rates, matched-FPR ROC and resources |

The internal `layout` index 0–15 identifies capacitor inventories at identical
positions. Join norm rows on `(family, layout, mode)`. Norm fields ending
`_v` are in volts. `distance` and `residual_distance` divide them by the
stored 1 µV noise scale. NPZ suffixes `L00`–`L15` identify inventories.
`codes_Lxx` is 64×6, `design_contrasts_Lxx` is 12×1600×6 and
`design_nuisance_Lxx` is 3×1600×6. Voltage records use time-major port order
and reshape to (400, 4) for one wiring.

The six modes are `uniform`, `random`, `raw_designed`,
`residual_designed`, `layer_difference` and `strain_matched`.
The median residual-design/control ratio is 1.361 against uniform and
0.988 against strain matching. At a common modeled false-positive rate
0.0641%, detection of a 10% support reduction is 86.7% for RC residual design,
100% for scanning and 99.999% for mechanical bridges.

The performance figure reads `results/composite-bridges/responses.npz`
directly: `composite4_design_v` supplies the twelve signed sector responses;
`composite4_soft_v` and `composite2_soft_v` supply the complete empirical
norm distributions. The two-depth field arrays compare a weak-response load
with uniform pressure at the same support-change location. The figure uses
separate microstrain and nanostrain scales for these fields. The same signed
bridge data supply the readout in the acquisition overview.

## Source fields and original study

`results/device-network-01/` and `results/fixed-calibration/` retain the
original random-layout study. Their geometry, five nuisance directions and
old rates belong to that earlier task. The unchanged nominal beam/continuum
field in `device-network-01/strain_fields.npz` supplies the current field plot.

```powershell
.venv/Scripts/python.exe scripts/summarize_device_study.py
```

The device transduction follows So et al., *npj Flexible Electronics* 6, 70
(2022), DOI [10.1038/s41528-022-00200-9](https://doi.org/10.1038/s41528-022-00200-9).
Circuit components and the common 1 µV sampled-voltage error are study inputs.
