# L1 pass 1f: the air passes of 1a with a passive feather. Frozen 2026-11-15

Pass 1a met the take-off and cruise gates with the feather **prescribed**,
which the one-degree-of-freedom drive (K1) cannot give. Here the stroke alone
is prescribed; each wing's feather is a spring hinge at its leading edge
(`rig.run(..., passive_feather=(k, zeta, com))`). Written before the grid
below was run.

## Disclosed before freezing

- The rig had to be fixed first: inside a step MuJoCo let the body and the
  stroke accelerate under the wing's pitching moment, so the hinge barely
  moved (+-3 deg). The held dofs now carry a large armature and the stroke's
  inertial push on the feather is added by hand. The prescribed-feather path
  is unchanged (1a's 4 Hz point still reads 1.70 weights, 72 W, 10.1 N m).
- `probe_passive.py` and three more points were seen (4 Hz, stroke 85 deg,
  body vertical): with the wing's trailing edge pointing up the hinge gives
  +1.56 weights for 119 W and 16.5 N m at k = 0.3 N m/rad, feather +-38 deg;
  with it pointing down the same wing gives -1.57 weights. Damping 0.1 to 0.5
  of critical and moving the wing's centre of mass from 0.25 to 0.05 of the
  chord changed nothing to two decimals: the hinge is ruled by aerodynamics.
  At k = 5 the feather was +-16 deg and at 20 +-5 deg.

## 0 Question and decision

Do the take-off and cruise rows of L0 survive when the wing feathers
passively? Decision: whether K1 (one drive, passive feather) stays the
reference, or the feather needs a driven degree of freedom (a second motor,
or a stroke-linked cam).

## 1 Rivals

(a) A passive hinge tuned in stiffness gives about what a prescribed feather
gave. (b) It costs torque and power: a weathervaning wing sits at a high
incidence (probe: about 50 deg) where drag is large. (c) Any shortfall is the
model's: its pitch damping and the strip load at the quarter chord set the
hinge's whole behaviour, and neither was checked against a real flapping
wing with a hinge.

## 2-3 Gates and predictions (gates are 1a's)

(a) From rest, stroke 85 deg, body pitched -45, -60, -75, -90 deg (trailing
edges up), 3.5-5.5 Hz, k 0.3, 0.6, 1.2, 2.4, 5 N m/rad, zeta 0.3, com 0.25.
Gate: vertical force >= 1.6 weights, peak drive torque <= 12 N m, aero power
<= 150 W.
- Prediction: **not met**; the cheapest point at 1.6 weights needs 14 N m or
  more. Falsified if any point meets it.

(b) Level at 6 and 8 m/s, stroke 40, 60, 85 deg, 2.5-4.5 Hz, k 0.6, 1.2,
2.4, 5, 10 N m/rad, the spring's rest angle trimmed so lift equals the weight.
Gate: sustains, <= 41 W, <= 12 N m.
- Prediction: met; lowest sustaining power 20-35 W (1a: 19.5 W).
  Falsified if no point meets it.

## 4-5 Measurement and design

`l1f_passive.py`. Drive torque = fluid stroke torque combined with the
wing's analytic stroke inertia, as 1a. One pass, no tuning after reading.
Predicted run time 10 minutes on 6 processes. Not covered: free body; a
hinge axis other than the leading edge; the wing's own bending.
