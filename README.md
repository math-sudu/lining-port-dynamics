# Passive bridge synthesis: code and data

This source set constructs positive-resistor bridges for prescribed reference-rejecting
strain directions at fixed gauge locations. It evaluates their grouping, channel
selection and voltage conversion under common source power and conductor-length
and gauge-dissipation allowances. Taishan monitoring records provide a
chronological replay with a reference defined from observations. Shiziyang
pressure profiles supply reference-load inputs to the representative ring.

## Environment and reproduction

The original numerical study used Windows 10 x64, CPython 3.11.13,
NumPy 2.4.6, SciPy 1.17.1 and OpenSeesPy 3.5.1.12 (OpenSees 3.5.1).
`requirements-lock.txt` pins this environment, including
`openseespywin==3.5.1.12`. The standalone frame reconstruction requires only
Python and NumPy.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe scripts/reproduce.py --prepare-only
```

The last command regenerates the ring load cases. Running
`scripts/reproduce.py` without that option regenerates the retained studies
in dependency order and overwrites their results. Reading the saved results
does not require this computation.

1. `installation_study.make_cases` creates
   `results/fixed-installation/cases.json` from the installation settings.
2. `random_bridge_study` generates irregular ring geometry, healthy bases and
   changed strains in `results/random-bridges/healthy_exchange.npz`.
3. `passive_bridge_study` and `frame_bridge_study` generate the ring/frame
   initialization circuits, calibration loads and design/evaluation strains.
4. `passive_robustness_study` creates the installed gauge properties,
   commissioned bases and component-setting draws in
   `results/passive-robustness/hardware.npz`.
5. `passive_acquisition_study` builds and refines contiguous/cyclic candidates,
   selects the joint voltage design and its arm scales in
   `results/passive-network/acquisition.*`.
6. `passive_design_study` selects resource-constrained connections and scales
   in `results/passive-network/constrained.*`.
7. `passive_observables_study` resolves group availability, selected-direction
   capture and loaded gains in `results/passive-network/observables.*`.
8. `fixed_wiring_study` removes the joint design's arm-scale upper bounds while
   retaining its optimized coefficients, connections and resource ceilings.
9. `acquisition_architecture` compares input-referred white noise of the same
   four frame directions using passive bridges or four multiplexed receivers.
10. `field_pressure_study` incorporates the two measured Shiziyang pressure
    profiles into the ring reference and evaluates the remaining target response.

Configuration files, input generators and their local dependencies are included.
The large generated arrays remain in the working project and can be regenerated
from these inputs. The initialization studies supply the starting circuits and
arrays needed by the network calculations.

## Network selection and stored quantities

The network comparison evaluates the joint voltage design (`baseline` in the
constrained study, `scaled` in the acquisition study) with
`constrained`. Both maximize the geometric regional mean of summed
channel voltage increments relative to the stored-reference healthy background
and both averaging variances.

Each installation's source-power budget is the zero-strain power of its
initial contiguous circuit, using the original integer arm capacities, actual
supply conductors and 2 V source. Joint arm scales lie between their routed
physical lower bounds and initial integer capacities. The joint design's
wire length and largest single-gauge power over design states set the resource
limits for the constrained design. Constrained scales have physical lower
bounds and may increase to satisfy gauge dissipation.

The fixed-wiring control applies the same unrestricted scale optimization and
gauge-power constraints without changing the joint coefficients or routes.
Its recorded readout statistics coincide with the reported joint values;
the connection-search gains do not result from the former scale upper bound.
`results/passive-network/fixed_wiring.json` preserves the control scales,
source settings and evaluation statistics; coefficients and routes remain
those of the original joint designs.
The command recomputes selected installations by default; `--resume` reuses
completed ones when recovering an interrupted run.

The starting contiguous coefficients are independently refined before joint
optimization. Their initial unloaded objectives use absolute healthy output
and one-average noise, with arithmetic regional aggregation for the ring and
geometric aggregation over locally resolvable frame regions. Loaded
reference-subtracted refinement follows before applying the network objective.
The cyclic template uses randomized projected target columns. These steps
define the initialization used by the saved networks.

The constrained result summaries supply both retained comparison rows and the
healthy/changed readout distributions. The representative frame has
`layout_seed=8206` and is the fixed-direction comparison: groups and signed
coefficients coincide, while arm assignments, scales and source voltage differ.
Across other installations, strategy comparison can also involve regrouping
and changed strain directions.

Current voltages have axes `(state, hardware_draw, case, channel)`;
the stored reference uses `(hardware_draw, channel)`.
Design increments use `(region, design_load, channel)`.
Same-load increments subtract healthy and changed voltages at identical load
and component properties. Operational readings subtract one stored reference
and include load residuals, setting errors and both averaging noises. All
current cases in a component setting share its reference noise.

The design coordinates combine the commissioned effective healthy basis with
nominal forward increments evaluated at the reference gauge factor. The
physical-to-effective sensitivity ratios are applied once in independent
evaluation; they are not applied again to saved nominal design increments.
The available and captured strain fractions use the design coordinates.

Summarize the saved results with:

```powershell
.venv/Scripts/python.exe scripts/export_measurement_tables.py
.venv/Scripts/python.exe scripts/export_measurement_tables.py --paired-evaluation
```

The default command reconstructs the representative four-channel operating
quantities from the supplied circuits. The paired-evaluation option reads
the saved evaluation arrays. Neither command reruns network optimization.

## Standalone representative circuits

`data/representative-frame/` contains both complete four-channel circuits
and the original 15-region, 32-load inputs. Their keys are `scaled` and
`constrained`.

```powershell
.venv/Scripts/python.exe src/lining_port/reconstruct_frame.py --data data/representative-frame
.venv/Scripts/python.exe -m pytest tests/test_representative_frame.py
```

The middle right-beam design RMS values are 5.527581 and 6.844849
microvolts. The data README defines component, node and coordinate conventions.
`scripts/export_representative_frame.py` extracts the circuits and inputs
from the full saved arrays.

## Taishan chronological replay

The authors' construction-monitoring observations retain their stress units,
instrument pairs and original dates in `data/field-monitoring/`.

```powershell
.venv/Scripts/python.exe -m lining_port.field_bridge_study
.venv/Scripts/python.exe -m pytest tests/test_field_bridge_study.py -q
```

Days 1–48 determine the fixed weights; all nine later observations evaluate
the combination. The calculation replays strain equivalents from
vibrating-wire steel-stress records through a specified resistance bridge.
`results/field-bridges/` holds the readouts, component networks, chronological
split, target projection and gain. RMS and mean changes are relative to day 1
and describe the accumulated construction response. The imposed local contrast
checks the specified sensitivity at the recorded operating points.

The leading pattern of contrast changes is the default reference when no
additional spatial rejection is required. The initial contrast profile
can supply an additional null direction for such a requirement.
`reference_choice_comparison` in the summary supplies both the change-only
and initial-plus-change rows, selected from the same early observations.
The illustrated circuit and its figure retain the two-pattern choice.
Its 84.9% RMS reduction is relative to the measured local A1 contrast;
the records do not independently separate local structural change from
other contributions to the combined output.

## Individual acquisition and measured pressure inputs

```powershell
.venv/Scripts/python.exe -m lining_port.fixed_wiring_study
.venv/Scripts/python.exe -m lining_port.acquisition_architecture
.venv/Scripts/python.exe -m lining_port.field_pressure_study
.venv/Scripts/python.exe -m pytest tests/test_acquisition_architecture.py tests/test_field_pressure_study.py -q
```

The acquisition comparison uses the representative frame's four unit rows,
formed by normalizing each bridge's `weight_effective_strain` coefficients.
The separate `weight_physical_strain` coefficients include the installed
gauge-factor ratio and apply to physical strains.
The digital alternative has four receivers, each reading fifteen individually
completed quarter bridges. Only the selected bridges are excited. The same
single-gauge power limit bounds their excitation; the two 4-second windows
each contain 800 independent samples per receiver. Uniform scanning retains
all gauge readings. Target-weighted integer allocation minimizes noise on
the prescribed row and skips zero coefficients. Ideal switching, exact
completion and calibration, and no settling or additional wire losses define
the comparison. Receiver, switch and processing powers are not included.
`acquisition_architecture.json` records settings, per-channel noise and sample
schedules. Its output compares the input-referred white-noise RMS of the
difference between current and stored-reference averages at the same
instantaneous single-gauge power limit, separately from the existing
component-error evaluation.

`data/field-monitoring/shiziyang_sections.csv` preserves the eight pressure
readings on each of rings 75 and 170. Monitoring ran from installation through
stabilization during shield advance; a common acquisition time for each
retained profile is not specified. Periodic linear interpolation maps these
construction-stage spatial load templates to external forces on the existing
supported-ring model. Winkler springs supply its separate displacement-dependent
reactions. The two computed strain profiles
augment its original three-mode reference. Model parameters are retained,
and no field stiffness or steel-force fit is performed. The same original
design increments select unit directions before the two pressure inputs are
combined with 10-percent support loss at twelve sector centers. The comparison
uses the RMS norms of the four captured strain outputs over these 24 cases.
The measurements serve as spatial load inputs, not independent validation
of predicted field stress. `pressure_reference.json` and `pressure_reference.npz` contain settings,
reference bases, selected directions and evaluation increments.

## Structural inputs

The ring is a linear elastic beam with radial/tangential foundation springs;
the frame uses piecewise elastic beam members. Published device dimensions and
gauge factor inform the transducer model. The study reports computational
circuit synthesis, offline replay of steel measurements and reference design
using measured pressure profiles. The Taishan
instruments measured stress; the prescribed resistance bridge evaluates the
corresponding strain equivalents.
