# Construction-monitoring inputs and reference design

The observations are the authors' construction-monitoring data for ring 1690
of the Taishan water-intake shield tunnel, at DK1+848. Seven pairs of
vibrating-wire rebar strain meters were installed on the reinforcement before
segment casting. Frequencies were read with a VW-1 receiver and converted
to stress relative to each instrument's initial-frequency datum.

## Measurements and coordinates

`taishan_1690_steel.csv` supplies all 29 observation dates from day 1 to day
163 after ring installation. Each station B2, K, B1, A1, A2, A3 and A4 has
an outer (ground-side) and an inner (tunnel-side) steel stress in MPa, with
tension positive. B1 is at the crown and A3 at the invert. The fourteen series
share the same dates. Values were extracted from the numerical data stored
with the monitoring curves, preserving dates and values without interpolation.
They are processed stress records; the measurement receiver supplied frequencies.
The analysis retains the recorded instrument datums and applies no additional
temperature correction.

The calculation uses the prescribed elastic modulus `Es = 200 GPa` and
`epsilon = sigma / Es`. It takes the instruments' measurement datum as zero
strain and the day-1 circuit voltage as the stored reference. Pair contrast
is inner minus outer. No assumed instrument separation or equal angular
spacing is required.

## Reference and target

The first 20 observations (days 1–48) determine the leading right singular vector
of the matrix of contrast changes from day 1, with observations in rows and stations
in columns. This change-only reference is the default
when no additional spatial rejection is required. The minimum-norm row
rejects that direction and has unit A1 contrast coefficient. Opposite
coefficients within each pair also reject any pair-common strain increment
to first order.

The illustrated circuit additionally rejects responses proportional to the
first contrast profile. This is a separate spatial constraint: subtracting
the stored day-1 voltage already removes the fixed initial reading, while
the additional constraint suppresses subsequent changes along that profile.
The circuit demonstrates the target cost and realization of this extra
requirement.

A1 is selected because its inner instrument has the largest compressive stress
within the fitting interval. The nine later observations (days 54–163) evaluate
the fixed row. The split is retrospective, around the transition to
approximately stable recorded stresses near day 50. Each observation has
equal weight in the reported RMS values.

The selected row captures all of the prescribed local target remaining outside
the reference space. `target_available_fraction` is the squared projected
target divided by its original squared norm, before voltage gain.
`unit_direction_gain_V_per_strain` uses a unit Euclidean norm across all
14 gauge weights. The reported 84.9% reduction compares the two-pattern
combination's RMS change from day 1 with that of the local A1 contrast.
The means change from −100.02 microstrain at A1 to −13.07 microstrain for
the combination, so the difference mainly reflects the accumulated offset.
The observations do not independently separate local structural change
from the other contributions to the combination.

The first-profile constraint is optional. The same minimum-norm construction
using only the leading change mode gives available target fraction 0.8963,
coefficient norm 1.0563 and later RMS 19.4281 microstrain. The illustrated
two-mode circuit gives 0.6292, 1.2607 and 15.1468 microstrain, respectively.
Both rows use only days 1–48. The single-mode row is invariant to fixed
instrument-datum shifts; the extra initial profile depends on those datums.
The two choices and their full coefficients are in
`summary.json["taishan_1690"]["reference_choice_comparison"]`.

## Circuit and supplied files

The prescribed circuit has 350-ohm gauges, gauge factor 2, attenuation 0.81
and a 1 V rail-to-rail source. Each arm contains 1 ohm of conductor resistance,
each supply conductor 2 ohms and each sense conductor 1 ohm; the differential
receiver resistance is 1 megohm. All four nominal arms equal 700 ohms.
These values specify the resistive circuit used to replay the strain
equivalents.

Using `o` and `i` for outer and inner instruments, the four arms are:

1. B2_i, B1_o, A2_i, A4_o.
2. B2_o, B1_i, A2_o, A4_i.
3. K_i, A1_i, A3_i.
4. K_o, A1_o, A3_o.

The accompanying `Taishan_Circuit.json` provides full-precision contrast
coefficients, gauge order, arm assignments, shunts, ballasts, conductor and
receiver resistances, and the zero-strain component netlist. Its
`arm_zero_based` entries 0–3 correspond to arms 1–4 above.
The arm signs follow the manuscript's top/bottom and left/right convention.

All voltages are calculated from the complete loaded circuit at absolute
strain equivalents. Day-1 subtraction gives the chronological readouts.
An imposed 10-microstrain A1 contrast changes the outer strain by −5 microstrain
and the inner strain by +5 microstrain. Its voltage increment illustrates
the retained local sensitivity at the nine later operating points.

## Reproduction

In the accompanying source archive:

```powershell
.venv/Scripts/python.exe -m lining_port.field_bridge_study
.venv/Scripts/python.exe -m pytest tests/test_field_bridge_study.py -q
```

The archive includes the stress observations and the previously calculated
A1/A3 moment magnitudes used to check extraction consistency.
`results/field-bridges/summary.json` gives the settings and results;
`circuits.json` contains the complete circuit.
`replay.npz` preserves absolute strain equivalents, observation split, contrasts,
voltages and imposed perturbations with named arrays. Gauge axes follow the
station order above, with outer then inner at each station. The leakage array's
second axis lists +50 microstrain at each of the seven pairs, followed by
−50 microstrain at the same pairs.

## Shiziyang pressure inputs

`shiziyang_sections.csv` preserves the authors' pressure and steel-force
observations at rings 75 and 170, transcribed from the numerical labels of
the measured spatial profiles. Each section had eight earth pressure cells,
monitored from installation through stabilization during shield advance.
A common acquisition time for the eight values in each retained profile
is not specified, so each profile serves as a construction-stage spatial
load template. N denotes crown and S invert, with the eight stations mapped
to 45-degree azimuthal intervals. The two profiles represent separate
sections. The current calculation uses the `pressure_MPa` column.

`lining_port.field_pressure_study` applies both measured pressure profiles to
the existing representative ring through periodic linear interpolation and
tributary-area nodal forces. These forces enter the external applied-load
term; the Winkler springs supply separate displacement-dependent reactions.
The measured pressure shapes therefore specify the applied loads rather
than the resulting total contact traction. Geometry and mechanical
parameters remain those in `inputs/installation_task.json`.
The resulting two strain profiles augment the original three load modes.
Four contiguous groups use the same original design increments to select
their respective unit strain rows before evaluating 10-percent support loss
at twelve sector centers under both measured pressure profiles. The reported
available fraction divides the total projected target energy over these
24 cases by their total target energy. The captured fraction divides the
four rows' captured energy by the available energy over the same cases.
Both are ratios of summed energies, rather than averages of individual
case fractions.

The pressure records are design inputs. The output examines how this load
information changes reference rejection and target retention in the
representative model. It does not fit measured steel forces or use the
reported derived moments as independent observations. Numerical inputs,
selected rows and target increments are in
`results/field-bridges/pressure_reference.npz`; the settings and reported
quantities are in `pressure_reference.json`.

```powershell
.venv/Scripts/python.exe -m lining_port.field_pressure_study
.venv/Scripts/python.exe -m pytest tests/test_field_pressure_study.py -q
```
