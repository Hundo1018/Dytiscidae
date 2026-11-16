# Pass W6, pre-registration: a spanwise twist that is a control

2026-11-17, branch `triphibian-diagnosis`. Frozen before `w6_twist.py` is run.

## 0. Question and the decision that hangs on it

Pass W4: the Goe 417a rectangle with 8 deg of built washout cruises at 8 m/s
for 9.7 W and loses the take-off (1.45 weights against 1.88 untwisted, gate
1.6). The wing wants a different twist in each. Can one wing whose tip turns
against its root meet both gates, how much turn does it need, and does
twisting within the beat (the tip alone feathering, the root keeping its
incidence, as a torsionally soft wing with a driven tip does) do better than
turning the whole wing as a rigid feather?

Decision: whether the wing's mechanism is specified as a rigid feather, a
held twist setting, a driven tip, or some of each, and the range the twist
actuator must cover.

## 1. What is measured

Body: seated fixture (`l1a.BODY_S`), Goe 417a rectangle, section
`goe417a+lift` (pass W5), held rig, W4's cruise grid (V 6 and 8 m/s, f 2-3.5
Hz, stroke 30-60 deg, feather 0-20 deg, trimmed to the weight by the wing's
incidence) and take-off grid (body pitch 45-90 deg, f 3-5 Hz, stroke 85 deg,
feather 30-70 deg), the cam's crank metric and the gates unchanged (take-off:
1.6 weights, 12 N m, 150 W; cruise: sustains, 41 W, 12 N m).

Two controls, `rig.run(strip_twist=dict(tip_deg, share))`:

- `tip_deg`: the tip's held twist against the root, linear along the span,
  nose up positive. Cruise: 0, -4, -8, -12, -16. Take-off: +8, +4, 0, -4, -8.
- `share`: the fraction of the feather's oscillation made as twist instead of
  by the hinge. 0 is the rigid feather of every pass so far; 1 leaves the
  root at its incidence and feathers the tip alone by the grid's angle.
  Cruise: 0, 1. Take-off: 0, 0.5, 1.

3,100 rows.

## 2. What the model does not have (so what these numbers cannot say)

- The twist moves the fluid strips only. The strips' own twisting rate (so
  the rotational force and pitch damping of the twisting), the wing's
  torsional inertia and the torque and power to hold or drive the twist are
  not in the loads. With `share` above 0 the crank metric therefore omits
  part of the feather's cost; it is the stroke's cost and the hinge's
  remaining part only.
- Take-off is on the vortex law throughout (W5): lift checked within 0.2
  against flat revolving wings, drag pessimistic for a cambered wing by a
  factor between 0.66-0.78 and under 0.5. The travel-since-reversal weight
  and a wing that both flaps and advances are unchecked (RESULT_W5).
- No tip loss beyond the one disc's inflow; strips do not see each other.
- A 3.4% thick plate, 0.66 m long, twisted 8-16 deg: no structure.

## 3. Seen before freezing

- W4 and W5 (twist 0 and -8 built, both grids, both section tables).
- The rig change's checks: a held `tip_deg` of -8 reproduces the built -8 to
  four decimals in force and power at one cruise point (8 m/s, 2 Hz, stroke
  50, feather 5, incidence +2 untrimmed: Fz -2.3975 N, 7.2581 W, both ways);
  `tip_deg` 0, `share` 0 reproduces no twist exactly. A first version turned
  the strips by a fiftieth of the angle (degrees converted twice); found by
  that check and fixed.
- The share's sign, same point untrimmed, feather 2 deg against none: rigid
  changes power by -0.69 W, share 1 by -0.36 W (same sense, smaller, as a
  feather that fades to nothing at the root must).
- One untrimmed point at feather 15: rigid 2.85 W and Fx -0.63 N, share 1
  6.40 W and Fx -0.12 N. Untrimmed, so it says nothing about a cruise that
  carries the weight.
- Nothing else of this grid.

## 4. Predictions

Q1. Cruise, 8 m/s, rigid feather (share 0): the cheapest sustaining power
    falls from twist 0 (21.5 W) through -4 to -8 (10.0 W, both W5), and the
    minimum over the five twists is at -8 or -12 and lies in 8-10 W. -16 is
    dearer than -12.
Q2. Cruise, share 1: at twist 0 the cheapest sustaining power is under 16 W
    (the rigid feather's 21.5 W), i.e. feathering the tip alone does part of
    the washout's work. The best share-1 setting is within 15% of the best
    rigid setting, either way.
Q3. Cruise with 0.1 N of net thrust in hand: the cheapest over all settings
    is under 18 W (W4's rectangle with -8 built: 21.5 W).
Q4. 6 m/s: at least one setting has 5 or more sustaining points and a
    cheapest under 25 W (W5: 1 of 80 at 30.4 W, twist 0; none at -8).
Q5. Take-off, rigid feather: the most lift under the caps is at twist 0 or
    +4 and lies in 1.80-2.00 weights; +8 is not more than 0.05 above twist
    0; -4 lies between 0's and -8's (1.88 and 1.45).
Q6. Take-off, share 0.5 and 1: less lift than the rigid feather at the same
    twist (the root, unfeathered, meets the flow broadside), with no gate
    point at share 1.
Q7. The combination: holding the tip at 0 for take-off (gate points 6 of 100
    as in W5) and at the cruise optimum, one wing meets both gates and
    cruises under 10.5 W; the turn needed between the two is 8 or 12 deg.

Falsified: Q1 if the minimum is at 0, -4 or -16 or outside 8-10 W; Q2 if
share 1 at twist 0 is 16 W or more, or the best share-1 setting differs from
the best rigid one by more than 15%; Q3-Q4 by their numbers; Q5 if the best
twist is negative or +8 beats 0 by more than 0.05; Q6 if share 1 has a gate
point or more lift; Q7 if no pair of settings meets both gates.

Rivals for a surprise: the trim (six secant steps) missing a point that would
sustain, which shows as few trimmed rows; a strip passing 30 deg and landing
on the plate law; the feather's cost leaving the crank metric with the share
(section 2), which would make share 1 look cheap for a reason that is not
aerodynamic. For the last, power is also read as the fluid power alone
(`aero_power_W`), which has no such omission on the stroke axis.

## 5. Controls, read points, stopping

`tip_deg` 0, `share` 0 must reproduce `results/W5_wing.jsonl`'s twist-0 rows,
and -8, 0 its -8 rows (cheapest cruise and best take-off to the printed
digits). One run, read once.

## 6. The drawing

`draw.py` from the built model: the wing at twist 0 (take-off setting), at
the cruise optimum, and at the two ends of a share-1 beat (the cruise twist
plus and minus the feather at the tip). Each opened and checked against the
design: span 1.40 m, area 0.34 m2, chord 0.258 m, roots at the hull's sides,
the tip's incidence against the root's as set, the section's upper side up
on both wings.
