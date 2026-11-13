# Plan for a flapping triphibian that will be built — draft 2 (2026-11-13)

Status: draft 3 (draft 2 plus the user's five decisions of 2026-11-13, in
their own section), not approved, nothing implemented. Draft 1 (2026-11-12, in
git history) was a line of phases; the user's answer was "a design that has
not been iterated will not be a good design". It was then attacked by a
fresh-context reviewer (`PLAN_review_1.md`). This draft is rebuilt around
what that review found; the section "What changed" lists each point and
whether it was accepted.

## Purpose (agreed with the user)

Design in simulation a flapping-wing machine that can be built from parts a
hobbyist can buy and print, and that then works in reality: 15 minutes in
each of air, water and land, and the crossings between them; automatic
on-board control, flown by a person over a remote link; powered by 18650 or
21700 cells. Its form comes from physically correct simulation and evolution,
not shapes assembled by hand, and each evolved wing gets the motion that
suits it. A result that holds in simulation and would not hold on hardware
does not count. Sensors (what they give, where they sit, their range) are
part of the design.

## Principles

1. Every result states the assumptions it rests on and whether hardware can
   honour them.
2. A model is trusted for a decision only where something that does not
   share its blind spots has checked it; for the wing that something is a
   bench, not a second simulation.
3. Real part limits are inputs from the first step.
4. Every component has a control at matched budget that removes or
   simplifies it, and stays only if it wins.
5. Bars and predictions are written before the measurement.
6. **The plan is a loop, not a line.** A whole, crude design goes round the
   loop early and often; each pass ends with one page: what changed and why.

## What the numbers already say (reports A11, A13; arithmetic in the review §3)

- Flight for 15 minutes closes, at about 0.6-0.7 kg with one wing pair and
  0.8-0.85 kg with two (draft 1's 0.4-0.6 kg was a battery-fraction bound,
  not a mass closure): three high-drain cells, a 71 g geared drive, a wing
  set at 187 g/m2, electronics, and what is left for structure and sealing.
- **Leaving the ground or the water under its own power is not shown
  possible at that mass.** No flapper of 0.3 kg or more in the surveys has
  static thrust above its weight (Robo Raven II: 1.7-2.3 N of thrust on
  3.0-3.7 N); hover-capable machines stop at 127 g; the one flapping water
  exit on record is under 300 g. By the sizing identity of A11 a 0.5 kg
  machine needs about 5.6-6.4 Hz to hover against a drive ceiling of 5.5 Hz.
  Fifteen minutes on cylindrical cells pushes the mass up; taking off pushes
  it under 0.3 kg.
- **One drive for three media is the requirement most likely to be
  impossible with the others.** The drive that has the power to fly is a
  crank: one degree of freedom, fixed amplitude. The same wing at the same
  amplitude in water at 1 Hz asks about 33 times the air torque
  (833 x (1/5)^2); real machines and birds shed that by a wing that bends
  away or folds. A crank cannot hold a planted wing or steer on land, and a
  servo that can has 11-83 times less power per kilogram. Immersion of the
  drive, and about 50 W of heat in a sealed hull, are unsized.
- So the purpose as stated may ask for things that cannot all be had on one
  machine from these parts. The plan's first job is to find that out in the
  first week, per requirement, not after months.

## The spine: three nested loops

- **L0, hours, paper.** A closure table, one row per requirement: fly 15
  min; swim with the same wings; walk on them; take off; leave the water;
  enter the water and land; the drive survives immersion; one drive type
  serves all. Each row has its margin, the row it conflicts with, and its own
  stop condition. Re-run whenever a number changes. Its first version, with
  the failing rows named, goes to the user: it is what he confirms, not a
  specification.
- **L1, about a day a pass, simulation.** One whole crude reference machine
  (a fixture, not the design) at the L0 mass, with a torque-speed-limited
  drive in place of ideal servos, a free body, and a stabiliser that reads
  only modelled sensors, through the measuring scripts that already exist
  (flight, swim, walk, take-off, with their do-nothing arms). It will fail
  somewhere; where it fails is the next pass.
- **L2, about a week a pass, bench.** One motor and gearbox, a printed crank,
  one wing pair, a scale or load cell, a wattmeter, an IMU logging inside the
  beat; then the same wing in a tub. It gives what no model here has: the
  drive's torque-speed curve, a real wing's thrust per watt, what a
  compliant wing does in water, and what flapping does to the IMU.

Order: L0 until it closes or names what cannot close; L1 on the L0 design;
L2 on the L1 design; corrections back into L0 and L1; repeat. The
evolutionary search enters inside L1 as the thing that proposes the next
design, after L2 has corrected L1 at least once. The whole machine is built
early and badly (pass 2 or 3), tethered and hand-launched; crossings are
added as they are earned.

## First week

1. L0 closure table.
2. The user's decisions below.
3. **Reproduce two real machines end to end in the present pipeline before
   designing one**, with predictions written first: E-Flap (0.51 kg, 1.5 m,
   3.5-4 Hz cruise, about 89 W, 6.5 N m at the hinge) and Robo Raven II
   (lift 2.6-3.5 N, thrust 1.7-2.3 N, 13.6-17.6 W at 4 Hz). If the pipeline
   says E-Flap cruises on 20 W, the model is fixed before anything else.
4. Order one drive, spar stock and skin, measured on arrival.

## Models, and what each is allowed to decide

| model | allowed to decide | checked by | not allowed to decide |
|---|---|---|---|
| Strip model with MuJoCo (the project's) | body dynamics, contact, first sizing | the two reproduced machines; the bench | anything about planform at the tips; high-incidence lift until the bench has corrected it (its deep-stall branch is "usable for drag, not for lift" by its own validity note) |
| Unsteady vortex lattice (Ptera Software) | planform and tip loading in attached flow | elliptic loading, Wagner; timed on this machine before its place is fixed | thrust and power rankings on its own (it is inviscid: an efficiency peak against Strouhal number appears only with a drag typed in); anything past stall |
| Coupling of the two | induced velocity per strip per step from a vortex wake, with a guard below stall whose trip rate is reported per elite | the same checks | a correction factor fitted in one and multiplied onto the other: not built |
| Wing compliance (to be built first; the present hinge is over-damped) | twist and bend under load, in air and water | the bench wing | — |
| Drive and cells | torque against speed, sag, current limit, heat | the bench drive; cell datasheets | — |
| Sensors | noise, delay, saturation, position, flap-rate corruption of the IMU | datasheets; the bench IMU log | — |

High-fidelity cloud CFD is dropped: it cannot be validated at this Reynolds
number from open data, cannot do the free surface, and a few points a month
bound nothing. The bench is the higher model. The NT$1000 a month goes to
learning runs on finalists, or to parts.

## The search

- Variables, about 15 numbers: wing planform, **stiffness distribution**,
  linkage geometry, tail, mass and sensor placement. Wing count and drive
  type are two or three discrete options compared by L0 and L1 passes, not
  dimensions for evolution to wander in.
- **Motion with the wing, on a free body**: the stroke's parameters, a few
  feedback gains and a tail law, read through the sensor model, fitted in
  the same optimiser from the parent's values. Whether that is enough, or
  the motion must be learned, is the user's decision 3.
- Algorithm: the simplest that wins. CMA-ES with restarts, a matched random
  arm and three seeds; quality-diversity only if it beats them at the same
  evaluations.
- Objective: the worst gate margin under the uncertainty bands (spar
  stiffness and strength 3-5x apart between suppliers, the 122-175 W/kg
  spread, the 30% of the torque rule), not energy after gates met exactly.
- **Passive stability while flapping is a design gate**, measured hands-off:
  the tail on servos and a crank average to about 1 Hz of authority, and the
  free plate machine of 2026-11-12 diverged in a second.

## What shows the search goes the right way

Kept from draft 1, as checks of the optimiser: repeatability and
heritability before any long pass; random search and hill climbing at
matched evaluations. Added, as checks of the model's direction:

1. The two real machines reproduced (above).
2. **A rank test on hardware each pass**: the top wing and one the model
   predicts is poor are printed and measured, with the model's order written
   first.
3. **A design-space null**: a flat rectangular plate. If elites do not beat
   it on the bench by more than repeat noise, the shape search is decoration.
4. Sensitivity of the optimum to each uncalibrated coefficient; an elite
   that changes identity inside a coefficient's band is the coefficient's.
5. A validity guard: per elite, the share of the stroke outside each model's
   envelope. Those elites are flagged, not ranked.

## Control and sensors

Every quantity a law reads maps to a sensor with a mounting position and a
range, from the first L1 pass on; the present laws' "distance and bearing to
the waterline", "clearance" and per-wing wetness have no hobby sensor and go
or are replaced. Hardest problems, named now: attitude estimation when the
body heaves and pitches at the flapping frequency inside the band the loop
needs; state through a crossing (which medium, which way is up when the
accelerometer saturates). Radio at 2.4 GHz does not usefully reach under
water, so under water the machine is on its own: that is one of three media
not flown by a person, and is said so.

## Also in the plan now (missing from draft 1)

Landing and water entry of an airframe that may not glide; buoyancy, trim
and ballast in three media (0.6 kg needs about 600 cm3 displaced); heat in a
sealed hull; printed-part fatigue (about 3,600 beats a flight) and PLA
softening beside a 50 W source; cell safety in a sealed hull; fresh or salt
water; crash repair time as a design target; trim adjustments on the build
and which test fits which parameter; firmware and logging fast enough to see
inside a beat; where it may legally be flown (registration from 250 g in
Taiwan, to be checked at the source).

## What the work of 2026-11-10 to 12 is, against this

Its methods stand (do-nothing arms, paired commands, the mission verdict,
predictions first). Its verdicts (walks, swims, flies, takes off) were
earned at 108 g with ideal servos on four independently feathered plates and
are void at the new scale and drive. The choice of two wing pairs was made on
those results and is reopened (decision 4).

## The user's decisions (2026-11-13) and what follows from each

1. **Fifteen minutes per charge, per scenario.** Six scenarios, each on its
   own charge: water; land; air; land<->air; water<->air; land<->air<->water.
   A mixed scenario is 15 minutes in total. So the cells are sized by the
   worst single scenario (15 minutes of flight), not by 45 minutes, and the
   mass closure of "What the numbers already say" stands. A direct
   water<->land crossing is not on the list and is not a gate (the lead's
   reading; to be confirmed).
2. **Launch is assisted: by hand, or by a mother craft that throws, launches
   or otherwise helps the child off.** Static thrust above weight is no
   longer required. L0's "take off" and "leave the water" rows become
   "survives and flies away from an assisted launch at speed v" and "can be
   relaunched from the water state" (picked up and thrown, or from a
   carrier). The mother craft is a second machine: it is not designed until
   the child flies from the hand, and L0 says whether the water->air leg
   needs it. Landing and water entry remain the child's own.
3. **Motion is optimised together with the wing** (closed-loop stroke
   parameters and gains in the same optimiser). No separate learning runs;
   the monthly budget goes to parts.
4. **Wing count by merit**: one pair with a tail against two pairs, compared
   in L0 and L1 at the same mass and drive, by the worst gate margin.
5. **Fresh water.**

## What changed from draft 1, by the review's points

Accepted: no loop (now the spine); bench first; crossings as gates from the
first pass; control before scoring free bodies; the per-requirement closure
table and stop conditions; the mass correction; one drive for three regimes
and self take-off named as the likely contradictions; compliance as a model
and stiffness as a variable; the vortex lattice demoted from judge to one
witness, the correction-factor import refused; real machines reproduced
first; hardware rank test, plate null, sensitivity, guard, margins as the
objective; cloud CFD and topology-as-dimensions cut; passive stability as a
gate; the missing items. Put to the user and not decided here: the five
decisions. Not checked by anyone yet: every run time and cost; the scaling
of take-off torque and hover power to 0.5 kg (the first L1 pass replaces
it); FAAV's numbers (press only).
