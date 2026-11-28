# Repair R1, pre-registration: the default law's stall and camber

2026-11-17, on `d9c8d2f`. Frozen before any solver code is changed. The
user, 2026-11-17: "verify, fix" (the stall law), "handle" (the typed
constants). This file is the stall law; the constants, the guards with units
and the typed gravity are R2 (`PREREG_R2_constants.md`).

## 0. What is wrong, with its evidence

`fluid._stall`: a strip without a measured section stalls at 11 deg times
`clip(0.55 + 0.45 log10(Re)/5, 0.5, 1)`, with a 6 deg handover to the
separated plate. Neither number has a source. Against data
(`flapper/sections/TYPED_CONSTANTS_SOURCES.md` A; RESULT_W5 P4):

- a thin flat plate bends over at 7-10 deg at CL 0.60-0.76 and stays on a
  plateau of 0.6-0.8 to 20-30 deg, nearly independent of Re from 1e4 to
  4e5; the law reads 1.0-1.16 before dropping to 0.5-0.6;
- a 4-6% cambered plate reaches 1.0-1.1 at 8-12 deg (two-dimensional, Re
  2e4-1.7e5), drops 0.15-0.25 and holds 0.85-0.90; about 1.3 by Re 4e5-1e6;
- the camber's lift at zero incidence is not thin-airfoil theory's 4 pi f at
  these Reynolds numbers: a 5% arc carries about 0.2 at Re 1e4-2e4 (theory
  0.63), 0.25-0.3 (4%) at 6e4, 0.45-0.5 (5.8%) at 4e4-1.7e5, and theory's
  value only near 4e5.

## 1. The replacement (for strips with no measured section; `SectionPolar` is untouched)

All from the table in the sources file; the fit is ours and is labelled so
there. With f the camber as a fraction of chord and a the strip's lift slope
(2 pi, or the Helmholtz form where the inflow is off):

1. Camber lift: the zero-lift shift is `2 f k(Re)`, k = 0.3 at Re <= 2e4
   rising linearly in log10(Re) to 0.9 at Re >= 3e5.
2. Limit: `CLmax = min(0.70 + 8 |f|, cap)`, cap 1.10 up to Re 2e5 rising
   linearly in log10(Re) to 1.30 at 1e6; a flat strip's 0.70 does not rise.
   On the side a cambered strip lifts away from its convex face (negative
   lift angle) the flat value is used: no data, stated.
   The stall angle is where the attached lift reaches it: `CLmax / a`.
3. Past it, over 4 deg (was 6): the lift goes to a plateau
   `CLmax - (0.04 + 3.5 |f|)` (flat 0.66; 4%: 0.84; 5%: 0.885) and stays
   there until the separated plate's `CN sin a cos a` exceeds it (21 deg
   flat, 29 deg at 4%), then follows the plate. Under a leading-edge vortex
   the plateau is weighted out with the vortex's strength, as the attached
   branch already is.
4. Drag past stall: the force is normal to the chord, so the pressure drag
   is the lift times tan(alpha) (which is `CN sin^2 a` on the plate branch,
   as now).
5. The Reynolds factor on the stall angle goes. Below Re 1e4 there is no
   data in the table; the law is held at its Re 1e4 values there, stated.

Not done here: the attached slope stays 2 pi (the table reads 0.78-0.82 of
it at Re 2e4-8e4; two facilities disagree on whether lifting line holds);
round-nosed sections 6% thick and more are a different regime and need a
measured section; the rotor model keeps the law it was fitted with (its
blades are such sections), by an explicit argument.

## 2. Where it has to change

`physics/fluid.py` (`_stall`, `lift_coefficient`, `drag_coefficient`, the
camber shift in `apply`, `strip_damping`, `chordwise_cop`), the Mojo kernel
(`mojo/src/fluid_gpu.mojo` and the two pipelines: the coefficient kernel
does not receive the camber today) and the device path
(`envs/devroll/fluid_wp.py`). Numpy first, validated; the mirrors after it,
held to it by `tests/test_gpu_mirror.py`, `tests/test_devroll.py` and
`test_search.py`'s agreement test. It is a comparability boundary for every
score of the main search: lift of unsectioned strips changes at every
incidence for a cambered strip and past 6-10 deg for all.

## 3. Checks and predictions

Rigs: `experiments/physics_extremes/run.py`'s translating plate.

- V1. Ananda's flat-plate wings (AR 2-5, 5-30 deg; the 22 points of
  `sections/separated/ananda2012_flatplate_rect_wings.csv`; rms error now
  0.165, range -0.17..+0.48): rms under 0.08, every point within 0.15.
- V2. Two-dimensional flat plate, Re 2e4 and 8e4: CL at 10, 15, 19 deg
  within 0.10 of 0.60-0.76 (the table's band); no drop of more than 0.06
  after the maximum.
- V3. Two-dimensional 5% arc at Re 2e4: CL(0) 0.15-0.25 (table 0.19-0.2);
  maximum 0.95-1.10 at 8-12 deg (table 1.09 at 9); 0.82-0.92 at 14-18 deg.
- V4. AR 6, 5% arc, Re 2e4: maximum 0.90-1.05 (table 0.99 at 10-11 deg).
- V5. Goe 417a's own measured table against the law run as a plate of the
  same camber (5.8%) at Re 1e5, 0-20 deg: within 0.15 everywhere; this
  compares the fit with a section it was partly drawn from, and is a
  consistency check only.
- V6. The battery (`run.py`) after the change: every check that held before
  still holds; E9's lift-slope rows are unchanged (the slope is not
  touched).
- V7. The machine: none of W4-W7 or C1 moves (they fly a measured section
  on the wings); C1's tail is a flat plate on this law, so its maximum
  lift falls. Not re-run here; flagged.

Falsified by the stated numbers. If V1 fails the law is not shipped as the
default; the failure is reported with the curves.

## 4. Tests and mutations

`tests/test_physics.py`: the law's numbers at the table's anchor points, the
plateau's hold, the cambered drop, camber lift at Re 2e4 against 3e5, the
rotor's unchanged thrust. Mutations: the plateau removed, the camber factor
fixed at 1, the limit made independent of camber.
