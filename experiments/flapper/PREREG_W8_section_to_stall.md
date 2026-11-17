# Pass W8, pre-registration: a measured section's drag up to its measured stall

2026-11-17, on `bbf7629`. Frozen before any code is changed or run.

## 0. Question and the decision that hangs on it

Cross-check X1 found the strip solver 1.0-1.3 N short of an inviscid
vortex-lattice solver in thrust at 6 m/s, at the same lift and nearly the
same power, and traced it to pass W5's extension of the section: the UIUC
drag runs end near 11.7 deg (Goe 417a: Cd 0.044 at CL 1.37), the lift runs
peak at 12.4-13.6 deg, and between the two the model charges the drag of a
strip loaded normal to its chord, `cl tan a` (0.28 at 12 deg), blended in
over 6 deg from the drag run's end. So the model stalls the section's drag
1-2 deg before its lift. Does "nothing sustains at 6 m/s" (W4-W6) survive
when the drag follows the section's own trend up to the angle its lift
peaks at? The decision: whether 6 m/s is closed, open, or decided by a band
nobody measured.

## 1. What is measured and what is not

Measured (UIUC, in `sections/`): drag to 10.1-11.8 deg; lift to 17.5-20 deg
with its peak. Not measured by anyone here: drag between the drag run's end
and the lift's peak, and beyond. Dynamic stall (the lift and drag of a
section pitching through its static stall, which a flapping wing does every
beat) is not in the model at all. So this pass cannot give one number for
that band; it gives three stated treatments and reads whether the answer
depends on which:

- "normal" (W5, as now): `cl tan a` from the drag run's end.
- "extend": the drag run continued in a straight line at its own end slope
  (least squares through its last three points) up to the lift run's peak;
  from the peak on, `cl tan a`, entered across `SEPARATION_COMPLETE`.
- "hold": the last measured drag held up to the peak, then the same. The
  optimistic end: drag near stall never falls as incidence rises.

Loaded as `goe417a+lift` (unchanged), `goe417a+lift+extend`,
`goe417a+lift+hold`. Lift is the measured lift run in all three.

Rivals for the 6 m/s shortfall other than this band: the lift past the
peak (the same in all three); the tip's feather taking strips past stall
where no treatment helps; the hull's drag; the root gap (the outside solver
runs it open, the strip solver sealed; a hull-in-the-gap case is being run).

## 2. A check of the straight line that does not need new data

For every drag run in `sections/` of the thin and conventional sections
(Goe 417a, BW-3, CR-001, E387 a/c/e, SD7003, SD7037b, S1223): drop its last
point, continue a straight line through the three before it, and compare
with the dropped point. This tests the extrapolation over 1.4-1.8 deg on the
last measured step, which is the step nearest the band in question.

## 3. Re-measurement

Held rig, `BODY_S`, camber 4%, the tip alone feathering (W6's share 1, no
held twist), W4's cruise grid (f 2-3.5 Hz, stroke 30-60, feather 0-20) at 6
and 8 m/s, under each of the three treatments: 480 trims.

## 4. Predictions

- L1. Leave-last-out: the line is low (the measured drag curves upward):
  median error -0.002 to -0.010 in Cd, and within 0.012 in magnitude for at
  least three quarters of the runs.
- L2. 8 m/s: the cheapest sustaining point moves by under 5% between the
  three treatments (7.2 W at "normal"); the number of sustaining points
  rises from 29 by no more than 12.
- L3. 6 m/s: "normal" none (W6). "extend": at least 3 sustaining points,
  the cheapest 8-14 W. "hold": more points than "extend" and a cheapest
  within 1.5 W of it.
- L4. So the 6 m/s verdict depends on the unmeasured band: it differs
  between "normal" and "extend".

Falsified by the stated numbers. If "extend" also finds none at 6 m/s, the
shortfall is not this band and X1's reading of it is wrong.

## 5. Tests

`tests/test_physics.py`: the three treatments agree inside the drag run and
past the plate handover, "extend" equals the line at a named angle, the
handover to `cl tan a` starts at the lift's peak. Mutations for each.

## 6. The drawing

The body is pass W6's (`figures/W6_takeoff.png`, the cruise wing with no
held twist), checked there; the section's table changes, not the body.

## Addendum W8b, frozen after section 3 was read and before this is run

Read so far (RESULT_W8 has it against section 4): L1 and L2 hold; **L3 and
L4 are falsified**. Under "extend" and "hold" nothing sustains at 6 m/s
either; the band gives back 0.00-0.07 N of a 0.15-0.7 N shortfall. So X1's
reading (that the shortfall was this band) was wrong.

A second reading, to be tested: at 6 m/s the strip solver trims at a root
incidence of 11-12 deg where the outside solver needs 5-8. The grid's tip
feather stops at 20 deg; at 2.5 Hz and a 50 deg stroke the tip's plunging
speed is 4.8 m/s against 6 m/s of flight, a swing of +-38 deg in the flow's
direction there, so with 15-20 deg of feather the outer wing still swings
+-18 to 23 deg about its mean and spends part of every beat past the
section's static stall (13 deg). A linear solver does not see that; the
strip solver clips the lift and charges separated drag. If that is the
cause, a larger tip feather removes it.

Run: 6 m/s, the same rig and wing, tip feather 25, 30, 35, 40 deg, f 2-3.5
Hz, stroke 30-60 deg, treatments "normal" and "extend": 128 trims.

Prediction L5: with a tip feather of 25-40 deg at least three points
sustain at 6 m/s under "normal", the cheapest 8-16 W, at a root incidence
(cam offset) under 9 deg in magnitude. Falsified if none sustains; then the
shortfall is neither the band nor the feather range, and stays unexplained.

Caveat frozen with it: a strip swinging through its static stall at a
reduced frequency of 0.25-0.4 is dynamic stall, which the solver does not
model in either direction (RESULT_W5: no data obtained). L5 can show the
grid's edge was the limit; it cannot show the stalled part of the beat is
modelled right.
