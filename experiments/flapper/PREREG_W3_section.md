# Pass W3 pre-registration: measured wing sections in the fluid model (2026-11-16)

Frozen before `w3_section.py check` and `sweep` are run. Code at this commit:
`fluid.SectionPolar`, `FluidSolver.section` (opt-in, numpy path only, default
unchanged), `sections.py`, `body.built(wing_shape=, wing_section=)`,
`w3_section.py`, `w3_read.py`.

## Why

The user, 2026-11-16: every pass used flat rectangular wings, with no
camber, thickness or incidence designed for the air. Measured the same day
(`probe_shape.py`): the fluid model cannot see camber (0 to 8% moves cruise
power 26.9 to 27.0 W) and has no section polar. Its attached-flow law is a
lift slope of 2 pi up to a stall at 11 deg and a laminar flat plate's skin
friction as the whole profile drag (0.0084 at Re 1e5).

## Seen before freezing

- The data: `sections/` (77 wind-tunnel polars from the UIUC Low-Speed
  Airfoil Tests, volumes 1-5, copied by a subagent with provenance in each
  file; one file checked against the raw archive by me). Drag runs: lift and
  wake-rake drag at 10-19 angles, about -5 deg to just short of stall, at Re
  60k, 100k, 200k. Lift runs: separate, to 18-24 deg, no drag.
- A summary of the drag runs (printed, not simulated). At Re 1e5: Goe 417a (a
  thin cambered plate) cd min 0.0166, best L/D 65; BW-3 55; CR-001 59;
  SD7003 cd min 0.0088, L/D 43; SD7037 48; E387 52. At small or negative
  lift the thin cambered sections have cd 0.045-0.07.
- **Not found as tabulated measurements:** a flat plate's drag, a 4%
  circular-arc plate, any finite wing at aspect ratio 3-6. So the flat
  rectangle stays on the default law and has no measured counterpart, and
  the section model has no independent finite-wing check.
- No simulation with a section has been run. The new unit test passes.

## The model (frozen)

Inside a drag run's range the strip's lift and drag are the table's, linear
in angle and in log Re between tables, the nearest table outside them. Past
either end it goes to the solver's separated plate across 6 deg, and a
leading-edge vortex takes it there at any angle, both as the default law
does. Flow over the trailing edge and strips without camber keep the default.

## Predictions

Check (no simulation), each section's model against its own separate lift
runs:
- **P1** Inside the drag run's range the rms difference in cl is at most
  0.06 for each of the six sections at each Re.
- **P2** In the 6 deg past the top of the table the model reads below the
  measured lift by more than 0.2 on average for at least four of the six
  sections: a wing just past stall keeps more lift than a flat plate's
  separated branch. If so it is recorded as a low reading near stall and not
  repaired in this pass.
- **P3** The three E387 models differ in best L/D by 10% or more at Re 1e5:
  the floor under any ranking of sections.

Sweep (pass 1i's air grid, seated fixture, still a rectangle):
- **P4** Goe 417a: the cheapest sustaining cruise costs 32-48 W (default law
  26.9 W); 2-18 points meet the cruise gate (default 28).
- **P5** SD7003 and E387 each cost more than Goe 417a at their cheapest
  sustaining point.
- **P6** Take-off with Goe 417a: lift at the default's 22 gate points within
  8% of the default's, and 18-26 gate points.

## Decision rule

The section with the cheapest sustaining cruise that keeps take-off goes to
the wing design pass (planform, twist, incidence). If no section sustains
within 41 W, pass 1i's cruise gate is not met by this wing under measured
drag, and the design pass has to recover it or the gate stands failed.

Not covered: sections at Re below 60k (the inner wing at take-off; held at
the 60k table), section data under flapping (the tables are steady), the
pitching moment (the lift runs have it; not used), water.
