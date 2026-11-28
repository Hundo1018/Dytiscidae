# Repair R3, pre-registration: the default lift and drag from two measured thin sections

2026-11-17, on `505d48c`. Frozen before any solver code is changed. The user,
2026-11-17: the main search's default goes to the measured tables of thin
cambered plates ("改用薄彎板"); the wing will be built as a thin cambered
plate. R1 (`RESULT_R1_stall.md`) showed that no fitted formula stands; this
replaces the formula by the measurements themselves.

## 0. Seen before freezing

- The camber of the tunnel models, from their digitised coordinates
  (`flapper/sections.py coordinates`): Goe 417a 5.78% at 0.38 chord, 3.4%
  thick; BW-3 5.65%, 5.0% thick; CR-001 4.0%, 7.4% thick.
- The lift runs' summary: flat plate (3.1% thick) CL 0.60-0.65 at 5 deg,
  0.78-0.81 at 8, 0.81-0.92 at 15, 1.18-1.24 at 24, and -0.014 to -0.037 at
  0 deg; Goe 417a peak 1.36-1.40 at 12.4-13.6 deg, CL at 0 deg 0.02 (Re
  60k), 0.20 (100k), 0.41 (200k); BW-3 peak 1.32-1.41, CL at 0 deg 0.21,
  0.23, 0.45.
- Four of Ananda's flat-plate wing points (aspect ratio 2, 15-30 deg): the
  read CD is CL tan(alpha) within 0.01 at all four.
- Everything in `RESULT_R1_stall.md` and `RESULT_W8_section_to_stall.md`.

## 1. The law (`FluidSolver.stall_law = "table"`)

For a lifting strip with no `SectionPolar` of its own, of signed camber f (as
the flow meets it; reversed flow flips the sign, as now):

1. Two anchors, both UIUC two-dimensional measurements held in the repo:
   the flat plate (f = 0; lift runs at Re 60k, 80k, 100k, 120k, to 24 deg)
   and Goe 417a (f = 5.78%; lift runs to 19.5 deg and drag runs to 11.7 deg
   at Re 60k, 100k, 200k).
2. The Goe 417a polar is `SectionPolar("goe417a+lift+extend")` read in
   steady flow: the drag table, continued at its own end slope to the angle
   the lift peaks at, the force normal to the chord from there (pass W8).
3. The flat plate has no measured drag. Its pressure force is taken normal
   to the chord at every angle: `cd = cf(Re) + |cl| tan|alpha|`, cf the
   two-sided flat-plate friction the solver already has. Its lift run is
   made odd about zero by shifting the angle to the measured zero-lift
   angle (0.1-0.3 deg, a model alignment offset) and mirroring.
4. Between the anchors both coefficients are linear in |f| at the same
   angle towards the convex side; past 5.78% the Goe 417a values are held
   (the gene reaches 9%; there is no thin-plate data there in the repo).
5. Reynolds number: linear in log Re between the measured runs, held at the
   nearest run outside 60k-200k (flat plate: outside 60k-120k), and outside
   that range the drag moves by the change in flat-plate friction,
   `cf(Re) - cf(Re clipped)`.
6. Outside the measured angles each anchor goes to the separated plate
   (`CN sin a cos a`, `CN sin^2 a`), by `SectionPolar.PLATE_BY` (30 deg)
   above the lift run's end and across `SEPARATION_COMPLETE` below the drag
   run's negative end; a leading-edge vortex takes it there at any angle;
   the table is read at the Wagner-lagged angle. All as `SectionPolar` does.
7. No camber shift of the zero-lift angle: the tables carry their own.
8. The tables are baked into the package on a uniform grid (0.25 deg, five
   Reynolds numbers) by a tool from the csv files, so the three backends
   read the same numbers; a test rebuilds them from the csv.

Kept: the lift scaling by the Helmholtz factor and the induced drag when
the inflow model is off; the rotor model's own law ("legacy", by argument);
`strip_damping`'s 2 pi (a stabiliser's linearisation, not a load).

Known limits, stated and not repaired here: no data below Re 60k (held; the
Goe 417a run at 60k carries almost no camber lift at zero incidence, where
BW-3 of the same camber carries 0.21); nothing between 0 and 5.78% camber
was measured (the interpolation is ours); reversed flow reads the same
table (an arc is fore-aft symmetric, Goe 417a nearly); static tables, no
dynamic stall; the flat plate's 1.2 at 24 deg falls to the plate law's 0.86
at 30.

## 2. Order of work and the gate on the default

Numpy first, opt-in; the checks below; then the Mojo kernel and the device
path behind the same switch; then the default flips in all three in one
commit, recorded as a comparability boundary for every score of the main
search. **If T1, T2 or T7 fails the law is wrong as built and is repaired;
if T3 fails the default does not flip and the failure is reported.**

## 3. Checks and predictions

Rigs: `physics_extremes/run.py`'s translating plate; coefficient calls.

- T1. The anchors reproduce their runs: at f = 0 and 5.78%, steady, no
  vortex, two-dimensional, at each measured Reynolds number and every
  measured angle inside the table, |cl - measured| < 0.01 (the grid's
  resampling); for Goe 417a's drag runs |cd - measured| < 0.002.
- T2. At f = 5.78% the law is `SectionPolar("goe417a+lift+extend")`: over
  -10..40 deg at Re 60k, 100k, 200k, |dcl| < 0.02 and |dcd| < 0.005; at Re
  80k and 150k < 0.03 and 0.008.
- T3. Ananda's flat-plate wings (22 points, aspect ratio 2-5; the legacy
  law's rms error 0.165): rms under 0.14, and **lower than the legacy law's
  on the same rig** (the gate). Predicted worst points: aspect ratio 2,
  where the rig's lift slope is 20% high (`RESULT.md` E9), off by up to
  0.25. CD where Ananda gives it: rms under 0.05.
- T4. BW-3, which the law does not use (camber 5.65%, thicker): the law at
  that camber against BW-3's lift runs, 0-19 deg. Predicted: within 0.10 at
  Re 100k and 200k; at 60k up to 0.2 apart below 5 deg and within 0.10 from
  8 deg. Against its drag runs, 0-10 deg, within 0.008 at 100k and 200k.
  This reads how much two builds of one camber differ; it gates nothing.
- T5. A 4% cambered plate: the law's maximum at Re 60k-100k. Predicted
  1.17-1.22, which is 0.1-0.2 above the second-hand tables' 1.0-1.1
  (`TYPED_CONSTANTS_SOURCES.md` A). A stated disagreement, not a check.
- T6. The battery with the table law as the solver's default: E1, E3-E8,
  E10, E11, E13 and E14 keep their verdicts; E2 keeps its failures and
  gains none; E9's bound holds and its slope rows get worse (the flat
  plate's measured slope is 7.2 per radian against 2 pi), failing at
  aspect ratio 4 as well. E12 is not predicted.
- T7. Sanity over f in {0, 2, 4, 5.78, 9}% and Re in {1e4, 1e5, 1e6},
  -90..90 deg in 0.1 deg steps: no step of cl over 0.02; f = 0 odd to
  1e-12; cd >= cf everywhere; cd at 90 deg is 1.98 + cf; at f = 5.78%, Re
  1e5, the best cl/cd equals the Goe 417a drag run's own within 2%.
- T8. The mirrors: `tests/test_gpu_mirror.py`, `tests/test_devroll.py` and
  `test_search.py`'s agreement test hold at their present tolerances with
  the table law on.
- T9. The flapper line does not move: one W6 row re-run reads 7.237 W
  (it flies `solver.section`). Pass C1's tail is a flat plate on the
  default law and will move; not re-run here, flagged.

Not read here: the size of the boundary on a stored population (arch51's
elites re-scored under the new law).

## 4. Tests and mutations

`tests/test_physics.py`: the anchors at their measured points, the linear
blend at half camber, the held value past 5.78%, the oddness of the flat
plate, the friction's Reynolds correction, the baked grid against the csv.
Mutations: the blend ignoring camber, the flat plate's drag losing its
normal-force term, the Reynolds interpolation taking the wrong neighbour,
the camber's side ignored (the arc lifting towards its concave side).
