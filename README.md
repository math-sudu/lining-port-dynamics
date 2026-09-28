# Fixed wiring for selective strain encoding

Code and simulation data for **Mechanically informed fixed wiring of distributed
piezoresistive networks for selective strain encoding of tunnel support
degradation**, by Pengcheng Zhu and Tielin Chen, Beijing Jiaotong University.

The numerical study selects a fixed binary wiring for each of sixteen installed
64-node piezoresistive networks. It evaluates the frozen wirings over 32 mechanical
cases per layout using four voltage channels. The saved study contains all 512
case-layout pairs and four wiring strategies (2,048 evaluation records).

## Contents

| Path | Contents |
| --- | --- |
| `src/lining_port/bridge_network.py` | Physical circuit components, Kirchhoff equations and exact voltage propagation |
| `src/lining_port/bridge_design.py` | Joint update of support-change and nuisance responses during wiring selection |
| `src/lining_port/bridge_study.py` | Nominal wiring synthesis and evaluation of fixed wirings |
| `src/lining_port/bridge_fields.py` | Generation of the saved strain fields |
| `src/lining_port/fe.py`, `continuum.py`, `network.py` | Mechanics, strain recovery, installed layouts and receiver models |
| `inputs/` | Device specification, mechanical configuration and complete family manifest |
| `results/device-network-01/` | Formal study data described below |
| `figures/` | Numerical source data for the manuscript figures |
| `scripts/summarize_device_study.py` | Recompute the reported paired gains from saved data |
| `tests/` | Numerical and software tests |

## Installation

The saved computation used Windows x64 and Python 3.11.13. The full environment is
pinned in `requirements-lock.txt`, including OpenSeesPy 3.5.1.12 and NumPy 2.4.6.
Run these PowerShell commands from the repository root with
[uv](https://docs.astral.sh/uv/) installed:

```powershell
uv venv --python 3.11 .venv
uv pip sync --python .venv/Scripts/python.exe requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
.venv/Scripts/python.exe -m pytest -q tests/test_bridge_network.py tests/test_fe.py tests/test_continuum.py tests/test_network.py tests/test_mechanism_study.py
```

The lock file includes the Windows OpenSees binary package. The commands above
reproduce the original platform and dependency versions.

## Read the saved results

```powershell
.venv/Scripts/python.exe scripts/summarize_device_study.py
```

Expected paired residual-norm results for `residual_designed`:

| Control | Improved pairs | Median paired increase |
| --- | ---: | ---: |
| Uniform wiring | 444 / 512 | 67.5% |
| Random wiring | 445 / 512 | 77.0% |
| Wiring optimized for the unprojected norm | 433 / 512 | 43.8% |

The median paired unprojected-norm ratio against uniform wiring is 0.994.

## Reproduce the numerical study

Use a new output directory for each run. The explicit device input supplies the
manuscript's gauge factor of 50 and strain-averaging length of 200 micrometres.

```powershell
.venv/Scripts/python.exe -m lining_port.bridge_study --output results/device-reproduction --device inputs/device_node.json
.venv/Scripts/python.exe -m lining_port.bridge_fields --study results/device-reproduction
.venv/Scripts/python.exe scripts/summarize_device_study.py --study results/device-reproduction
```

The study designs each layout at nominal mechanical parameters using twelve
softened sectors, then evaluates its fixed wiring on the 32 `test` families in
`inputs/mechanism_manifest.json`. The four strategies, in saved column order, are
`uniform`, `random`, `raw_designed` and `residual_designed`.

The exact 100-chip bipolar probe is generated in `bridge_network.response_basis`
by `np.random.default_rng(4100).choice([-1., 1.], 100)`. Each chip lasts 0.02 s;
the four-port response is sampled every 0.005 s, giving 400 samples per port.
Layout seeds are 7100 through 7115. The random wiring and second search start use
seed `20260928 + layout_index`. Actual wiring vectors are also saved.

## Data format and units

All five files in `results/device-network-01/` belong to the same formal study:

| File | Contents |
| --- | --- |
| `design.json` | Physical inputs, nominal mechanics, twelve design sectors, layouts, mode order and normalization |
| `wirings.json` | The four 64-bit wiring vectors per layout and the search histories |
| `evaluation.json` | One record per mechanical family, layout and wiring mode |
| `responses.npz` | Components, coordinates, design records and representative evaluation records for all sixteen layouts |
| `strain_fields.npz` | Healthy and softened strain fields on a common angular and depth grid |

Join evaluation rows by `(family, layout, mode)`. Both `distance` and
`residual_distance` are divided by the stored reference
`receiver_noise_sigma_v = 0.01 V`; multiply by 0.01 to obtain full-record voltage
norms in volts. `contrast_l2_v` and `contrast_peak_v` are already in volts.

NPZ files can be read with `numpy.load(path, allow_pickle=False)`. In
`responses.npz`, suffix `L00` through `L15` identifies the layout. Arrays contain:

- `codes_Lxx`: 64 nodes by four wiring modes.
- `capacitance_Lxx`, `edge_conductance_Lxx`, `coordinates_Lxx`, `port_nodes_Lxx`:
  installed components in F and S, coordinates in m, and receiver-node indices.
- `design_contrasts_Lxx`: twelve sectors by 1,600 observations by four modes.
- `design_nuisance_Lxx`: five mechanical directions by 1,600 observations by four modes.
- `example_healthy_Lxx`, `example_soft_Lxx`, `example_nuisance_Lxx`,
  `example_strain_Lxx`: records for the first `test` family in the manifest.

Flattened voltage records follow `(time, port)` order and reshape to `(400, 4)`
for one wiring. Nuisance directions are pressure, radial support modulus,
thickness, load harmonic and load phase. The strain-field arrays use depth rows
and angular columns; strain is dimensionless, depth is in m and angle is in rad.

The numerical figure data retain their manuscript source directories:
`F2-fields` (Fig. 3), `F11-wiring-design` (Fig. 4), `F6-bridge-circuit`
(Fig. 5), `F7-bridge-performance` (Fig. 6), `F8-bridge-records` (Fig. 7), and
`F9-device-node` (Fig. 2). CSV headers identify quantities and units.

## Device reference

The gauge factor and sensing length follow So et al., *npj Flexible Electronics*
6, 70 (2022), DOI [10.1038/s41528-022-00200-9](https://doi.org/10.1038/s41528-022-00200-9).
The nominal resistance, reference and leakage resistors, capacitances, links and
probe are this study's circuit design inputs. Their values and the mapping from
the published sensing element are recorded in `inputs/device_node.json`.
