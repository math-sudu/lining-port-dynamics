# Fixed wiring and four-channel strain acquisition

Code and data for **Mechanically informed fixed wiring for selective strain
acquisition in piezoresistive networks**, by Pengcheng Zhu and Tielin Chen,
Beijing Jiaotong University.

The current numerical task uses a fixed embedded ring, four 16-node modules,
64 reachable gauges and four receivers. Six RC wirings, four-channel scanning
and mechanical bridges share the same positions, loads and complete 4 s cycle.

## Reproduction

The scientific runtime is Windows x64, Python 3.11 and the packages pinned in
`requirements-lock.txt`. Install with uv:

```powershell
uv venv --python 3.11 .venv
uv pip sync --python .venv/Scripts/python.exe requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
.venv/Scripts/python.exe -m lining_port.installation_study --mechanical
.venv/Scripts/python.exe -m lining_port.acquisition_comparison
.venv/Scripts/python.exe -m pytest -q tests/test_installation.py tests/test_acquisition_comparison.py tests/test_bridge_network.py
```

The first command regenerates `results/fixed-installation/` from
`inputs/installation_task.json` and `inputs/device_node.json`. The second
uses those strain inputs and RC operating rates to generate the acquisition
comparison. The manuscript repository also supplies
`scripts/summarize_installation.py` to generate its tables and numerical facts.

## Current files

| Path | Role |
| --- | --- |
| `src/lining_port/installation.py` | Four embedded ladders, routed lengths and port assignments |
| `installation_study.py` | Fixed mechanics, wiring, calibration and test evaluation |
| `acquisition_comparison.py` | Four-bank scanning, active mechanical bridges and resource calculations |
| `bridge_network.py`, `bridge_design.py` | KCL response matrices and fixed-wiring search |
| `fe.py`, `continuum.py` | Ring mechanics and strain recovery |
| `results/fixed-installation/` | Current study data, described below |
| `figures/` | Numerical source data for the manuscript figures |

The source modules above are in `src/lining_port/`.

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
