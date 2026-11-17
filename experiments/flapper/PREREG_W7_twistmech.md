# Pass W7, pre-registration: the tip's twist as a mechanism

2026-11-17, branch `triphibian-diagnosis`, after the C1 pre-registration
commit. Frozen before `w7_twistmech.py` is run.

## 0. Question and the decision that hangs on it

Pass W6 turned the fluid strips by hand: the twist had no structure, no
inertia, no rate in the air's velocity and no drive, so its 7.2 W was a
figure for a shape nobody had built. The user's rule (2026-11-17): a force
may only come from the simulated shape and its motion, never from a number
added to it. So: when the twist is made by joints of the model, (a) does the
W6 cruise survive, (b) what does driving the twist cost and what torque does
it need, (c) can the air and the wing's inertia make the twist without a
drive, (d) what holds the wing flat at take-off, and what does that take?
The decision: driven tip, passive wing, or neither; and the torsional
stiffness the wing must have.

## 1. What is modelled now, and what still is not

- `body.py wing_panels=N` (N <= 6, the builder's depth limit): the wing is N
  equal rigid panels, each outer one on a hinge along the span at the tip of
  the one before. Mass, inertia, area, chord, camber and section as the
  one-panel wing. The hinge line is moved to the feather's (0.1 chord).
  Each panel's incidence, its twisting rate (in the strips' velocity, the
  rotational force and the pitch damping) and its inertia are the model's.
- `FluidSolver.camber_moment` (new, opt-in): a cambered strip's own pitching
  moment about its quarter chord, thin-airfoil theory's `Cm = -pi f` for the
  arc (f = 0.04: -0.126), in attached flow only. Without it the air's
  twisting moment on a cambered wing lacks its largest attached-flow part.
  Tested (`test_a_cambered_strip_carries_its_own_pitching_moment`, two
  mutations caught). The measured section's own moment is not in the UIUC
  table used; the arc's is the stand-in, stated.
- `freerig.run_free(chain=)`: the hinges as servos (a drive at every rib),
  as springs with one torque between root and tip (a torque tube or a line
  to the tip rib in a torsionally soft wing: by virtual work one torque on
  every hinge of the wing), or as springs alone.

Not modelled, and the user's point about cupping lands here: a strip is a
rigid chord line with a camber number. The twist changes each strip's
incidence along the span (the helical "wrap" of the wing around the air it
moves) and that is in the forces through each strip's own flow. What the
model cannot show is the chordwise cup: in separated flow the normal-force
coefficient is the flat plate's whatever the camber (RESULT_W5 P2: Goe 417a
and the flat wing identical to three decimals under the vortex), and a strip
cannot change its camber in the beat. A membrane that bellies on the power
stroke and flattens on recovery is outside this pass; it needs a separated
law that takes the section's shape (pass W8, proposed in the result).
Also not modelled: the panels' own bending, the hinges' friction, gaps
between panels, the drive's mass and gear.

## 2. What was seen before freezing (machinery checks, one point)

`scratchpad/w7p.py`, cruise 8 m/s, 2.5 Hz, stroke 40, tip 15 deg, cam offset
fixed at W6's -2.955 (not trimmed), 4 beats:

- strips turned by hand: 7.24 W, lift 1.003; with the camber moment 7.25 W,
  lift 0.993.
- 4 panels, every hinge driven: 6.93 W of which twist 0.17 W net (0.27 W
  counting only positive work), peak hinge torque 0.21 N m, lift 0.989; with
  the camber moment 6.99 W, 0.50 W positive, 0.38 N m, lift 0.952.
- 4 panels, tube with springs of 3 N m/rad: 6.94 W, tip 14.9 deg, lift 0.846.
- 4 panels, springs alone: k 3: tip mean +5.4 deg, swing 7.5 deg, 4 deg late,
  lift 0.49; k 0.5: mean +12, swing 15.5, lift -0.14.
- 8 panels read 1.7 W: the builder had cut the chain to 6 panels and the wing
  was short (found, now asserted). No 6-panel run, no take-off run, no trim.

## 3. Design

Body held, no tail, 0.60 kg; camber moment on except where said. Cruise at
8 m/s on three of W6's points (f, stroke, tip feather): 2.5/40/15, 2.0/50/15,
3.0/40/20; the cam offset trimmed to lift = weight within 1% (3% accepted,
as pass 1i); the root hinge does not feather. Take-off at rest, body pitch
90, stroke 85, the root feathering 50 deg, at 4.5 and 4.0 Hz.

| arm | wing | hinges |
|---|---|---|
| R, Rm | one panel, strips turned (W6) | without / with the camber moment |
| S3, S6 | 3 and 6 panels | each driven to its share of the tip's angle |
| S6 no Cm | 6 panels | as S6, camber moment off |
| T6 k | 6 panels | springs k = 0.3, 1, 3, 10, 30 N m/rad per hinge, one torque root to tip tracking the tip's angle |
| P6 k | 6 panels | springs k = 1, 3, 10, 30, 100 alone |
| K1 | one panel | take-off reference |
| L6 | 6 panels | take-off: each hinge driven to zero (a lock at every rib; its torque is read) |
| T6 k, P6 k | 6 panels | take-off: tube holding the tip at zero / springs alone |

k per hinge is a wing torsional stiffness GJ = k x 0.11 m (0.033 to 11 N m2).

Read: total drive power (stroke, feather and twist; a spring is not a
drive); the twist drive's net and positive-only power and peak torque; the
tip's twist (mean, swing, lag against the feather's phase); each hinge's
swing; lift, net thrust, crank metric. Gates as before: cruise sustains
(trimmed, thrust >= 0), 41 W, 12 N m; take-off 1.6 weights, 12 N m, 150 W.

## 4. Predictions

Q1. S6 trimmed: total power 6.2-8.3 W at 2.5/40/15 (W6: 7.24) and within 15%
    of R at the other two; it sustains at all three. Twist drive: positive
    power 0.2-0.8 W, peak hinge torque 0.2-0.6 N m.
Q2. S3 against S6: total power within 10% at each point.
Q3. S6 without the camber moment: total power within 3% of S6, twist peak
    torque lower by 0.1-0.25 N m.
Q4. Tube: for k <= 3 total power within 10% of S6 and it sustains; the tube's
    peak torque is under 0.6 N m for k <= 1 and over 1.2 N m at k = 30; the
    positive-only twist power rises with k and passes 1 W by k = 30.
Q5. Springs alone, cruise: at least one k trims and sustains with total power
    within 25% of S6's at the 2.5/40/15 point (the probe's k = 3 twisted in
    phase with half the swing). No k below 3 sustains (the wing washes out).
Q6. Take-off, K1: 1.75-2.0 weights at 4.5 Hz (the held rig read 1.88). L6:
    lift within 5% of K1; peak lock torque at a hinge 0.3-1.5 N m.
Q7. Take-off with springs alone: k = 100 within 5% of K1's lift, k <= 3 loses
    more than 10% or misses the gate. With the tube holding the tip: k >= 10
    within 5%, k <= 1 loses more than 10%.
Q8. A stiffness exists in the sweep for which the tube arm both cruises
    within 15% of S6's power and takes off within 5% of K1's lift (k = 10 or
    30); its tube torque at take-off is under 2 N m.

Falsified by the stated numbers. Rivals for a disagreement between S6 and R:
the staircase of 6 panels against a linear twist (S3 against S6 reads it);
the twisting rate and inertia now in the loads (the point of the pass); the
servo's compliance under the camber moment (the trim absorbs the mean).

## 5. Controls, read points, stopping

R is W6's stored row (7.237 W at 2.5/40/15 before trim to 1%); a P6 arm with
k = 100 is the stiff control for the spring machinery (must sit near L6 at
take-off). One run, read once by `w7_read.py` against section 4 verbatim.

## 6. The drawing

`draw.py W7`: 6 panels twisted +-15 deg, 6 panels locked, 3 panels. Opened
before the run: 5 hinges on each wing on the feather's line, 6 panels each,
the twist stepping up from root to tip about a line near the leading edge
and opposite at the two ends; span 1.400 m, area 0.3400 m2, chord 0.258 m,
mass 0.600 kg, all OK against the design.
