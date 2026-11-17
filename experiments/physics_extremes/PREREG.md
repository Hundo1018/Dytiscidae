# Physics at the extremes: pre-registration

2026-11-17, branch `triphibian-diagnosis`, on `2aedf5a`. Frozen before
`run.py` is run. The user, 2026-11-17: "are you sure the interaction of our
physics and the mechanisms is right? ... test it, with mechanisms at extreme
boundary conditions; do not hard-code."

## 0. Question and the decision that hangs on it

Every result of the flapper passes is the fluid solver acting on a MuJoCo
mechanism. The solver has been compared with wing measurements
(RESULT_W5); the *interaction* has only been checked where a pass happened to
need it. Does it obey the laws that hold whatever the coefficients are?
The decision: which figures of passes W4-W7 and C1 can be kept, and what in
the solver or the rigs has to be repaired before anything more is built on
them.

No expected value in this battery is a typed number chosen to pass. Each is
(a) an invariance: two runs that physics says are the same run; (b) a
conservation law or a bound from momentum theory, computed in the check from
the run's own density, span, speed and force; (c) a closed-form result of
potential-flow or lifting-line theory, written as its formula; or (d) one
labelled measurement with its source. A failure is a finding to be traced to
its cause; no tolerance is loosened after the run and no special case goes
into the solver to pass one.

## 1. The checks

Rigs: "bare" is one rectangular plate of strips on a joint, driven as pass
W5's rigs are (`FluidSolver.apply` on a MuJoCo body); "machine" is the
flapper built by the project's own pipeline on `freerig.run_free` or
`rig.run` (wings, hinges, servos, ballast). Uniform fluid means the plate is
far from the surface in one medium.

| | check | rig | law | tolerance |
|---|---|---|---|---|
| E1 | Galilean invariance | bare; plate moving at U in still fluid against plate still in a wind of -U; U 0.05, 8, 80 m/s; 5 and 45 deg; air and water | the same force | 1e-6 of the force |
| E2 | similarity in size | bare; chord x0.02, x1, x50 at the same Reynolds number and reduced frequency; translating (5, 20, 60 deg), revolving (40 deg), heaving | the same coefficients | 1e-3 |
| E3 | similarity in fluid | bare; air against fresh water at the same Reynolds number; the same three motions | the same coefficients | 1e-3 |
| E4 | mirror symmetry | machine (six-panel wing, tube), held, cruise and take-off | no side force, no rolling or yawing moment | 1e-4 of the vertical force (times the half span for moments) |
| E5 | momentum in a vacuum | machine, released in x, z and pitch, fluid density 1e-9; control: the same in air | beat means of the hull's forward speed, of its vertical speed plus g t, and of its pitch rate do not drift | 1e-3 of the hull's own speed swing within a beat; the control must drift by more than 10x that |
| E6 | still fluid gives no energy | machine held at rest, 40 kinematics in air and 20 in water drawn over the extremes (0.3-30 Hz, stroke 5-160 deg, feather 0-85 deg, offset +-60 deg, body pitch 0-90; flat, cambered, sectioned and six-panel wings; all-up mass 5 g to 50 kg) | mean drive power >= 0 | -1% of the mean absolute power |
| E7 | hover bound | the E6 runs, and every stored row at rest of W6 and W7 | drive power >= F^1.5 / sqrt(2 rho A), A the disc of the full span, F the mean fluid force | none: a ratio under 1 is a violation |
| E8 | cruise bound | every stored cruise row of W6 and W7, and new held runs at 2, 3, 20 and 40 m/s | drive power >= thrust x V + 2 L^2 / (rho V pi b^2) | none |
| E9 | finite wing | bare, translating, AR 0.5-40, 2-8 deg | drag >= CL^2/(pi AR); lift slope against Helmbold's 2 pi AR/(2 + sqrt(AR^2 + 4)) | bound: none; slope: 15% |
| E10 | added mass | bare plate (AR 10) on a spring in water, released 0.002 chord off, normal to itself and edge-on | period gives pi rho c^2 b / 4 normal, nothing edge-on | 15% normal; edge-on under 5% of the normal value |
| E11 | Archimedes | bare sealed body on a vertical slide: mass = rho V deep under; mass = rho V / 2 at the surface | stays; floats half under | speed under 1e-4 g x 1 s after 1 s; centre within 2% of its height of the waterline |
| E12 | impulsive start | bare, 2D (no inflow), 5 deg | Wagner: lift / steady = 1 - 0.165 e^(-0.0455 s) - 0.335 e^(-0.3 s), s in semi-chords, at s = 2, 4, 8, 16, 40 | 0.10 |
| E13 | time step | machine: W7's S6 cruise and K1 take-off at half the step; `rig.run` at 200 steps a beat against the 50 that W4-W6 used | the same lift and power | 2% (freerig), 3% cruise and 5% take-off (rig) |
| E14 | broadside plate (measured) | bare, 90 deg, AR 1 and 2D | CD 1.17 (disc and square plate; TYPED_CONSTANTS_SOURCES.md, C) and 1.98 | 10% |

Controls built in: E5's air arm shows the check can fail; E1-E3 compare a
run with itself at scale 1 first (difference exactly 0). The load limiter
(`60 x weight`) is read in every machine run and reported; a run in which it
bound is outside the model's stated envelope and is counted apart.

## 2. What was seen before freezing

Nothing of this battery has been run. Known from earlier passes and used in
the predictions: the torque-driven rig at 1,600 steps a beat reproduced the
held rig at 50 (7.239 against 7.237 W, W6's cruise point); an impulsively
started strip is put on the vortex law for its first chords (found while
writing the camber-moment test); under separation the model has no aspect
ratio in the normal force.

## 3. Predictions

- E1 holds in air and water.
- E2 and E3 hold for the translating and revolving plate. Heaving: holds.
- E4 holds. E5 holds, and the air control drifts.
- E6: every run's mean power is non-negative. If one is negative, the model
  hands a mechanism free energy and every power figure near that regime is
  void.
- E7: no violation; the stored take-off rows sit at 2-8 times the bound.
- E8: no violation in the stored rows. The new slow runs (2-3 m/s) are the
  likeliest place for one; I predict none, with less confidence.
- E9: the drag bound holds at every aspect ratio. The lift slope is within
  15% of Helmbold for AR >= 2 and off by more than 15% at AR 0.5 and 1.
- E10: normal within 15%; edge-on under 5%.
- E11: both hold.
- E12: **fails** at s = 2-8 by more than 0.10 (the start-up vortex rule),
  within 0.10 by s = 40.
- E13: holds.
- E14: **fails** at AR 1 (the model reads the 2D value, about 1.98 against
  1.17); the 2D plate holds.

## 4. Read, stopping

One run of `run.py`, one table (`results.json`, `RESULT.md`). Each failure
gets a cause before any repair, and each repair its own pre-registered pass.

## 5. The drawing

The machine checks use the bodies of passes W6 and W7, whose drawings
(`figures/W6_*.png`, `W7_*.png`) were opened and checked in those passes;
E6's extreme bodies (aspect ratio 0.4 and 30) are drawn by `draw.py PX` and
opened before the result is written. The bare plate is a rectangle on a
joint; it has no drawing.
