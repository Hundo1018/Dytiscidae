# Physics at the extremes: result

2026-11-17. Frozen text: `PREREG.md` (`3a19bbe`). Run: `run.py` ->
`results.json` (107 rows), `machine_rows.jsonl` (`run.log` is ignored by git). Three runs were
needed; what changed between them is in "Repairs to the checks" below, and
no solver code and no tolerance was touched.

## Outcome in one paragraph

Where the laws are exact the solver and the mechanisms obey them: a plate
moving through still fluid and a still plate in a wind read the same force;
air and water at the same Reynolds number read the same coefficients to
twelve figures; a flapping machine in a vacuum keeps its momentum and
angular momentum to 1e-4; its two sides mirror each other; still fluid never
gave a mechanism energy in 60 extreme motions; no stored or new point is
below the actuator-disc power bound at rest or in cruise; the added mass of
a plate on a spring in water is the potential-flow value within 0.2%; a body
of the water's density stays and one of half floats half under; an impulsive
start follows Wagner within 0.09; and a five- to tenfold finer time step
moves lift and power by under 0.5%. It fails in three places, all in the
solver's handling of wings that are not slender: **the lift slope is 20-59%
high at aspect ratio 2 and below, a square plate broadside reads 36% low,
and two numerical guards carry absolute units**, which breaks similarity for
a 2 mm chord (12-22%) and for a slow 5 m one (1.7%). One more was found on
the way: the solver cancels the folded-in fluid's weight with Earth's g
whatever the model's gravity is.

## The table

| | check | holds | read |
|---|---|---|---|
| E1 | Galilean invariance | 12 of 12 | largest difference 4e-8 of the force |
| E2 | similarity in size | 13 of 22 | chord x0.02: translating at 5 deg 0.24-0.29% off, revolving 21%, heaving 4-22%; chord x50: revolving 1.7%, the rest exact |
| E3 | similarity in fluid | 11 of 11 | 1e-12 |
| E4 | mirror symmetry | 6 of 6 | side force 1e-10, rolling 1.5e-5, yawing 8e-6 of the vertical force |
| E5 | momentum in a vacuum | holds on the centre of mass | linear 1.1e-4 and 1.0e-4, angular 2.6e-4 of the scale; the air control changes by 1.1, 1.2 and 7.8 |
| E6 | still fluid gives no energy | 60 of 60 non-negative | all finite; the load limiter bound in 18 runs; the drives' worst tracking error was 28.5 deg |
| E7 | hover bound | no violation | 33 E6 air runs: least power/bound 2.30; 1,524 stored rows: least 5.45, median 7.67 |
| E8 | cruise bound | no violation | 1,645 stored rows: least margin 1.91 W; 128 new runs at 2, 3, 20, 40 m/s: none below |
| E9 | finite wing | bound 7 of 7; slope 4 of 7 | slope / Helmbold: 1.59, 1.39, 1.20 at AR 0.5, 1, 2; 1.07, 1.02, 1.00, 1.01 at 4, 8, 16, 40 |
| E10 | added mass | 2 of 2 | 7.825 kg against 7.838 (ratio 0.998); edge-on 0.079 kg, 1.0% of it |
| E11 | Archimedes | 2 of 2 | neutral body: 9.4e-4 m/s after 1 s (tolerance 9.8e-4); half density: 0.6 mm from the waterline |
| E12 | impulsive start | 5 of 5 | within 0.01 at s = 2, 8, 16, 40; 0.09 low at s = 4 |
| E13 | time step | 12 of 12 (4 of them void) | freerig at half the step: -0.2% and -0.1% in power; rig at a fifth and a ninth: -0.4% and -0.5% |
| E14 | broadside plate | 1 of 2 | AR 1: 0.746 against the measured 1.17; 2D: 1.989 against 1.98 |

## Against the predictions

Right: E1, E3, E4, E6, E7, E8, E10, E11, E13 hold; E9's bound holds; E14's
2D plate holds. Wrong:

- **E2**: I predicted it holds. It fails, from two guards with units (below).
- **E5**: holds, but not as I wrote the check (below).
- **E9**: I put the slope's failure at aspect ratio 1 and below; it is 20%
  high at 2 as well.
- **E12**: I predicted a failure at s = 2-8 from the start-up vortex rule.
  It holds: at 5 deg the vortex law's lift happens to sit near Wagner's
  half. This is agreement at one incidence, not a derivation; at other
  incidences it was not read.
- **E14**: I predicted AR 1 would read the 2D value, about 1.98, and so be
  high. It reads 0.746 and is low: the inflow model takes the plate's own
  drag as a disc loading and slows the flow through it.

## The failures, traced

1. **Two guards with units** (`fluid.py`): `dt = max(t - prev_t, 1e-6)` in
   `apply` (a floor of one microsecond on the step the unsteady state
   advances by) and `tau = 0.5 span / max(through, 0.05)` in
   `InducedFlow.update` (a floor of 0.05 m/s on the speed the wake convects
   at; also `T < 1e-9` N and `area <= 1e-9` m2). The first is why the 2 mm
   chord at 740 m/s fails (its step is 27 ns); the second is why the slow 5
   m chord's revolving case is 1.7% off. For the machines of this project
   the first never binds; the second binds when the flow through the disc
   is under 5 cm/s, which a slow swimmer's start reaches. Repair: guards
   without units (a fraction of the strip's own time and speed scales).
2. **Lift slope at low aspect ratio** (E9): the strips use the 2D slope and
   the inflow model supplies the lifting-line downwash, `CL/(pi AR)`. That is
   Prandtl's result, right for slender wings, and it overstates the slope
   below aspect ratio about 3 where Helmbold's form holds. The tail of pass
   C1 (AR 4.2) is 7% high by this; each wing alone is not the issue (the
   pair is one wing of AR 5.8). Stubby bodies in the main search are scored
   with it.
3. **Broadside plate at aspect ratio 1** (E14): 36% low. The same inflow
   model, at the other end: a bluff plate's drag is fed to the actuator
   disc as if it were a lifting system's thrust. A flapping wing at take-off
   is on the vortex law, checked against revolving wings of its own aspect
   ratio (RESULT_W5), so the take-off figures do not lean on this; a
   parachuting or pancaking body does.
4. **Earth's gravity is typed into the solver**: `finish_bodies` adds
   `m_body * GRAVITY` to cancel the weight MuJoCo gives the fluid mass folded
   into a body, with the module's 9.80665 and not the model's gravity. With
   gravity off a plate in water was pushed 0.097 m up its spring (77 N).
   Every run of this project has Earth's gravity, so no stored number is
   affected; it is a typed constant where the model's own value belongs.

Also read, not failures: the neutral body's residual acceleration is 1e-4 g,
at the edge of its tolerance and not traced; the load limiter (60 x weight
+ 60 N) bound in 18 of the 60 extreme runs, all light bodies flapped hard
(5 g to 0.26 kg), and E1's water cases at 8 and 80 m/s are both capped at
178 N, so they compare two capped loads; E12's "steady" lift is 0.438 where
2 pi alpha is 0.548 because the rig's "2D" is the machine aspect ratio 8
with the inflow off (the same holds for W5's 2D case past stall, where it
does not matter); the hover ratio of 5.5-7.7 says the take-off law spends
five to eight times the ideal power, which is the pessimistic side
RESULT_W5 already named.

## Repairs to the checks (after the first run; no tolerance changed)

- `freerig.run_free` used `ch` both for the chain's settings and as a loop
  variable over the released axes, so every released run crashed. Renamed.
  Passes W7 (held) and C1 (started before the chain code existed) did not
  run that path.
- E10 as frozen had gravity off and read no oscillation (finding 4 above).
  Gravity is on and the period is read about the settled position.
- E5 as frozen read the hull's beat-mean velocity. That moves with the
  body's attitude even when momentum is conserved (0.035 and 0.014 of the
  swing). It now reads MuJoCo's own centre-of-mass velocity and angular
  momentum of the whole tree; the frozen form is still printed.
- E13's held-rig pair, 50 against 200 steps a beat, were the same run:
  `rig.run` caps its step at 1 ms, which is 400 steps a beat at 2.5 Hz. So
  passes W4-W6 ran at 1 ms, not at 50 steps a beat as I had written. A pair
  at 1 ms against 0.2 and 0.11 ms was added and holds.
- E7 leaves the water runs of E6 out: their vertical force carries the
  buoyancy.

## What this does and does not establish

It establishes that the coupling is sound where it can be checked without a
reference: forces are frame-independent, scale with density and size as
they must outside two guards, mirror, conserve momentum when the fluid is
gone, never feed energy to a mechanism, respect the momentum-theory bounds,
and are converged in the time step. It does not establish that the
coefficients are right. Those rest on RESULT_W5's comparisons and on what is
still open there: no measurement of a wing that flaps and advances, none of
the vortex after a reversal. An outside solver for the attached cruise
(`sections/TYPED_CONSTANTS_SOURCES.md` F: an unsteady vortex-lattice code)
would test exactly that case and has not been run.

## The drawing against the design

`flapper/figures/PX_stubby.png`, `PX_slender.png`, from the built model,
opened: spans 0.380 and 1.880 m, areas 0.1200 and 0.0540 m2, chords 0.400
and 0.030 m, all as asked; roots at the hull's sides. Seen: the stubby
wings' chord is most of the hull's length and at the stroke's ends their
outlines cross in the front view; nothing in the model stops two wings
passing through each other or through the hull. They are test bodies.
