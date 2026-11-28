# L1 pass 1l result: the cam's feather amplitudes in water (2026-11-15)

**Erratum (2026-11-15, pass S1): the stroke torque, crank metric and power
series in this pass carried the entrained weight's cancellation**, a force
the solver adds to cancel the weight MuJoCo gives the added mass; only the
net force had it removed. On pass 1l's grid the corrected crank metric is
0.36-0.96 of the one read (median 0.66). The pass also ran with the solver's
default entrainment reaction, a one-sided drag on a flapping wing (median 23%
of thrust on that grid). Torque caps here were applied to an overstated
torque and forces are understated. Mean force and mean power are otherwise
unaffected. See `RESULT_S1_entrainment.md`; not re-run unless said there.

**Re-read in pass S1: the 0.6 wing does meet the gate under both centre-of-pressure laws; that finding below is withdrawn.**

Pre-registration: `PREREG_L1l_camwater.md` (commit 8b83c8e; it also records
the two faults found on the way). Data: `results/L1l_camwater.jsonl`, 194
rows, all finite. Reader: `l1l_read.py`. Held body, 0.3 m/s, on `rig.run`.
Gate as pass 1c, with the crank paying for the feather.

## Best net force under the caps, and points meeting the gate (of 8)

Wing folded to 0.4 of its length:

| feather amplitude | no law, hinge at the edge | plate, hinge 0.1 | insect, hinge 0.1 |
|---|---|---|---|
| 20 deg | +3.21 N (2) | +1.44 N (2) | +1.44 N (2) |
| **30 deg** | **+8.58 N (3)**, crank 8.6 N m, 7.6 W | **+4.93 N (3)**, crank 7.8, 9.7 W | **+4.93 N (3)**, crank 7.5, 9.5 W |
| 50 deg | +5.07 N (1), crank 10.1 | +1.61 N (1), crank 11.9 | +1.61 N (1), crank 11.8 |
| 60 deg | -2.01 N (0) | -3.34 N (0) | -3.34 N (0) |

Wing folded to 0.6: the gate is met at 2 points with no law (20 and 50 deg,
+0.44 and +1.77 N) and at none under either law.

## Against the frozen text

- **P1 refuted for the 0.6 wing**: under "plate" and "insect" it meets the
  gate nowhere. The 0.4 wing meets it under every model.
- **P2 refuted.** I predicted 20 deg would give too little thrust. The 0.4
  wing meets the gate at 20 deg under every model. And 60 deg, which I
  expected to work, meets it nowhere.
- **P3 confirmed where it matters**: at the 0.4 wing's best gate points the
  feather adds 63-70% to the crank metric (4.6-5.2 to 7.5-8.6 N m). The one
  0.6 gate point adds 4%.
- **P4 confirmed**: no point has negative mean power.

## Reading

- Rival (c) holds, the other way round from my guess: the amplitude decides,
  and the **cruise amplitude (30 deg) is the water amplitude**. At 30 deg the
  0.4 wing gives 4.9 N (with a centre-of-pressure law) to 8.6 N (without)
  at 0.75 Hz and stroke 40 deg, with the crank at 7.5-8.6 of 12 N m.
- The take-off amplitude is poor in water: 50 deg is marginal (1.6 N at
  11.9 N m under a law) and 60 deg fails.
- So the cam can stay engaged in water, set as for cruise. Water does not
  ask for a third amplitude.
- The centre-of-pressure law and the hinge at 0.1 chord cost 43-55% of the
  force at the same amplitude. Passes 1b-1e, run without the law, are
  optimistic by about that much if either law is right.
- Decision: the cam stays engaged in water at the cruise amplitude, with the
  wing folded to 0.4.

## What this does not establish

- **The torque leaves out the water's inertia.** The held rig reads fluid
  loads; the wing's added mass sits in the mass matrix. A rough figure is
  7 N m in quadrature with 11 N m at 0.5 Hz and 85 deg. The best points here
  use stroke 40 deg, where it is smaller, but it is not measured.
- **The rig that would measure it does not run in water** above about 30 deg
  of feather: the solver's entrainment reaction is unstable under a
  torque-driven wing (`PREREG_L1l_camwater.md`). That is a fault in the
  shared fluid solver. It does not touch the search, whose actuators are weak
  and whose step is coarse, as far as measured; it blocks every free-body
  measurement in water.
- The flap's power check (pass 1j, R5) is still open for the same reason.
- Free body; the fold's mechanism; a variable-amplitude cam.
