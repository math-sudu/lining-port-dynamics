# Passive bridge synthesis

Code and numerical study for **Passive bridge synthesis for load-rejecting strain
acquisition at fixed sensor locations**, by Pengcheng Zhu and Tielin Chen,
Beijing Jiaotong University.

The synthesis converts healthy-response null weights into passive bridges with
gauge shunts and arm ballast. Each active gauge occurs once. Exact circuit
refinement retains change increments while suppressing the finite-strain healthy
residual. The ring uses mean target energy; the asymmetric frame also compares
a spatial target design. Hardware calculations include commissioning, finite
shunts, conductor resistance, receiver loading, setting errors and later drift.

## Scientific reproduction

The scientific runtime is Windows x64, Python 3.11 and the versions in
`requirements-lock.txt`. Set up an editable installation:

```powershell
uv venv --python 3.11 .venv
uv pip sync --python .venv/Scripts/python.exe requirements-lock.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
```

Run all studies from the repository root:

```powershell
.venv/Scripts/python.exe scripts/reproduce.py
.venv/Scripts/python.exe -m pytest -q
```

The generator first creates the 64 calibration/test load cases from their fixed
seeds. To run individual stages, use this equivalent sequence:

```powershell
.venv/Scripts/python.exe scripts/reproduce.py --prepare-only
.venv/Scripts/python.exe -m lining_port.random_bridge_study
.venv/Scripts/python.exe -m lining_port.passive_bridge_study
.venv/Scripts/python.exe -m lining_port.frame_bridge_study
.venv/Scripts/python.exe -m lining_port.passive_robustness_study
```

The ring calculation reads the generated load cases in
`results/fixed-installation/cases.json`, with mechanics and layout seeds from
`inputs/installation_task.json` and gauge parameters from
`inputs/device_node.json`. These use the same `make_cases` function as the
original fixed-installation study. Its configuration retains the historical
two-depth description; the random bridge study uses its mechanical parameters,
load ranges and seeds, and restores independent angular and annular-area
draws and computes the gauge-level healthy and changed strains; it also retains
the healthy-only arm-exchange baseline. Passive synthesis uses those strains
and its separate target-design sectors. The frame and hardware studies read
`inputs/frame_bridge_task.json` and `inputs/passive_robustness_task.json`.

## Source and result roles

| Source | Role |
| --- | --- |
| `src/lining_port/passive_bridge.py` | Null directions, positive-resistor realization, exact gradients and constrained refinement |
| `src/lining_port/frame_response.py` | Compatible frame mechanics and finite-gauge response |
| `src/lining_port/passive_hardware.py` | Actual components, routed leads, receiver loading and drift |
| `src/lining_port/passive_bridge_study.py` | Fixed random ring experiment |
| `src/lining_port/frame_bridge_study.py` | Load-rank, spatial-target and independent confirmation experiment |
| `src/lining_port/passive_robustness_study.py` | Commissioning and physical-error scenarios |
| `scripts/reproduce.py` | Regenerate the ring load cases and run the four studies in dependency order |

The repository contains source, configurations and tests. Generated numerical
arrays, statistics and component netlists are written under `results/` by the
commands above. Supporting modules are included where imported by these entry
points. No external measurements are required.

`results/passive-bridges/synthesis.npz` stores ring design coefficients and exact
outputs; its JSON companion describes the statistics and each layout.
`results/frame-bridges/synthesis.npz` separates design, development (`test_*`)
and independent confirmation (`confirmation_*`) cases. The latter supplies
reported frame distributions and hardware evaluation. The mean-energy and
spatial designs differ in initialization and target aggregation together.
`results/passive-robustness/hardware.npz` preserves effective calibration bases,
fixed gauge deviations, component draws, zero references and operating voltages.
Each study exports component-level JSON netlists.

Reported summary voltages are medians across within-installation medians.
Hardware medians pool component draws and load cases within each installation.
Empirical distributions retain all stated cases. A voltage below the modeled
noise-vector RMS is an amplitude comparison, not a measured detection rate.

## Figures and tables

Figure and table sources are maintained alongside the manuscript and read these
formal result arrays directly. The scientific calculations above generate their
numerical inputs.

## Retained symmetric construction

The exact paired-depth construction and original comparison results are retained
in the public repository history at `3373ac3`. The general manuscript includes
its exact arm-balance and curvature formulas as
a special case. Historical RC, scanning and symmetric detection statistics are
not results of the random passive networks.
