# Representative frame circuit

These files reconstruct the four-channel example in the passive bridge
synthesis study. The joint voltage and resource-constrained designs have
the same mechanical weights and different arm assignments, ballast scales
and excitation. Their stored keys remain `scaled` and `constrained`.

Place the five delivered files in one directory:

- `Frame_Network.json`: two complete circuits, including geometry, all 60
  gauges, shunts, ballasts, routes, leads, source rails and receivers.
- `Frame_Design.npz`: commissioned healthy basis, 32 load coefficients,
  nominal gauge-factor increments for 15 regions and the original signed
  voltage increments.
- `reconstruct_frame.py`: reconstruction entry.
- `passive_hardware.py`: the study's component circuit solver.
- `Frame_README.md`: these instructions.

The supplied reconstruction was verified on Windows 10 x64 with
CPython 3.11.13 and NumPy 2.4.6. It requires only Python and NumPy.
From that directory run:

```powershell
python reconstruct_frame.py
```

Expected middle right-beam design-region RMS outputs are 5.527581 and
6.844849 microvolts for the joint voltage and resource-constrained designs.
To save all reconstructed healthy voltages and signed increments, use
`--output frame_voltages.npz`.

In the source checkout, the equivalent command is:

```powershell
.venv/Scripts/python.exe src/lining_port/reconstruct_frame.py --data data/representative-frame
```

`Frame_Design.npz` defines healthy strains as
`load_coefficients @ healthy_basis.T` and changed strains by adding
`regional_increments`. The healthy basis uses commissioned effective strains, while the forward
increments assume the reference gauge factor. These design inputs enter the
circuit directly; do not multiply by gauge-factor ratios again. The `voltage` function also accepts
`coordinates="physical"` and applies the stored gauge factors once.
For the unit rows used in the acquisition comparison, normalize each bridge's
`weight_effective_strain = arm_sign * attenuation` vector by its Euclidean
norm. The separate `weight_physical_strain` field includes the installed
gauge-factor ratio and acts on physical strains.
`*_expected_increment_v` has axes `(region, design_load, channel)`.
`regions` gives `(member_index, start_fraction, end_fraction, modulus_reduction)`.
The center is the mean of the endpoint fractions and the length fraction is
their difference. Region 13 (zero-based) is `[4, 0.41, 0.59, 0.35]`, the
middle right-beam patch centered at 0.50 with length fraction 0.18.

All JSON indices are zero-based. Arm signs are
`(-1, +1, +1, -1)`. Source voltage is the rail-to-rail span; H and L are its
positive and negative halves. Receiver output is A minus B. Node names other
than H/L are local to each bridge.

A bypassed gauge has no sensitive cell. A null shunt on an active gauge means
a bare gauge. Ballast already includes all series compensation; add the
stored arm-wire resistance once. Preserve the small positive ballast values.
The route lists give physical series order. The zero-strain netlists use
lumped arm wire and an electrically equivalent ordering.

The machine data retain the original floating-point component values.
The two circuits and original
design inputs are extracted by `scripts/export_representative_frame.py` from
`results/passive-network/constrained_netlists.json`, `constrained.npz`,
`results/passive-robustness/hardware.npz`, and
`results/frame-bridges/synthesis.npz`. In that source export, `baseline` denotes
the joint voltage design. The selected installation has
`layout_seed=8206`. No component-setting errors or sampled noise are added in
this design reconstruction. Independent-evaluation statistics use the evaluation
load sets and component draws stored in the corresponding result arrays.
