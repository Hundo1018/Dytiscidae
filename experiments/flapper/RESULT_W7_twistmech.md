# Pass W7, result: the tip's twist as a mechanism

2026-11-17. Frozen text: `PREREG_W7_twistmech.md`. Run: `w7_twistmech.py` ->
`results/W7_twistmech.jsonl` (69 rows, all finite, every cruise row trimmed
within 1%), read by `w7_read.py`.

## Outcome in one paragraph

Made by joints, the W6 cruise stands: six panels with every hinge driven
cruise at 8 m/s for 7.1-7.5 W, twist drive included, against 7.2-7.5 W for
the strips turned by hand. The twist drive itself is small in power (0.1-0.4
W net, 0.4-0.9 W if it cannot take work back) and needs 0.40-0.45 N m at the
worst hinge, half of which is the cambered section's own pitching moment.
One torque from root to tip in a torsionally soft wing works only if the
wing is stiff enough: at 10 N m/rad per hinge (GJ 1.1 N m2) it cruises for
7.2-7.7 W with 0.55-0.69 N m in the tube and takes off within 4% of the
rigid wing's lift holding 1.0 N m; softer wings sag between root and tip and
lose the thrust, stiffer ones cost spring work. A wing that twists by itself
does not do both jobs: soft enough to twist in cruise (3 N m/rad) it is 0.07
N short of sustaining there and makes no lift at take-off; stiff enough to
take off (30-100) it does not twist in cruise and costs 12 W.

## Against the frozen predictions

| | frozen | read | |
|---|---|---|---|
| Q1 | S6: 6.2-8.3 W at 2.5/40/15, within 15% of R elsewhere, sustains at all three; twist positive power 0.2-0.8 W, peak hinge torque 0.2-0.6 N m | 7.10 W (R 7.24); 7.10 against 7.34; 7.49 against 7.50; sustains at all three; positive power 0.50, 0.39, **0.87**; torque 0.41, 0.40, 0.45 | holds but for the positive power at the 20 deg point (0.07 W over) |
| Q2 | S3 within 10% of S6 | -5.6%, -6.1%, -7.5% | holds; but S3 is 0.04 N short of sustaining at all three where S6 sustains: three panels are too few to read a verdict from |
| Q3 | without the camber moment: power within 3%, peak torque lower by 0.1-0.25 | power within 0.5%; torque lower by 0.19, 0.18, 0.19 | holds |
| Q4 | tube: k <= 3 within 10% of S6 and sustains; torque under 0.6 for k <= 1, over 1.2 at k = 30; positive power passes 1 W by k = 30 | k = 0.3, 1, 3 do **not** sustain (thrust -1.24, -0.27, -0.05 N at the first point); torque 0.18-0.23 for k <= 1, 1.50-1.96 at k = 30; positive power 1.4-3.9 W at k = 30 | **first part falsified**, the rest holds |
| Q5 | springs alone: some k sustains within 25% of S6 at 2.5/40/15; no k below 3 sustains | none sustains at that point (k = 3: 7.97 W, thrust -0.07 N; k = 10: 11.1 W, -0.01 N); k = 10 sustains at 2.0/50/15 for 11.1 W (+56%); k = 1 does not | **first part falsified**, second holds |
| Q6 | K1 1.75-2.0 weights; L6 within 5%; lock torque 0.3-1.5 N m | 1.856; -2.8%; 1.37 N m | holds |
| Q7 | springs alone: k = 100 within 5%, k <= 3 loses over 10%; tube: k >= 10 within 5%, k <= 1 loses over 10% | -2.5%; -103% and worse; -3.8% and -1.9%; -42% and -94% | holds |
| Q8 | a stiffness where the tube cruises within 15% of S6 and takes off within 5% of K1, with under 2 N m at take-off | k = 10: +2%, +2%, +3% in cruise, -3.8% at take-off, 0.98 N m. k = 30: +9%, +8%, +13%; -1.9%; 0.97 N m | holds |

Where I was wrong (Q4, Q5): I took a soft wing to be one that follows its
drive. A tube sets the tip's angle only; between root and tip the air's
nose-down moment sags a soft wing into washout (the hinges' swings are no
longer equal: 6.2, 4.6, 3.6, 3.6, 3.6 deg at k = 0.3), the cam offset has to
come up 4-11 deg to hold the lift, and the thrust is gone. The same moment
is why springs alone settle 4-11 deg washed out.

## Cruise at 8 m/s (cam offset trimmed; power is stroke + feather + twist drive)

Point 2.5 Hz, stroke 40, tip 15 deg (the other two read the same way;
`w7_read.py` prints all three):

| arm | power, W | thrust, N | twist drive: net / positive-only W, peak N m | tip: mean, swing, lag |
|---|---|---|---|---|
| R, strips turned by hand (W6) | 7.24 | +0.045 | not modelled | 0, 15, 0 |
| S6, every hinge driven | 7.10 | +0.015 | 0.15 / 0.50, 0.41 | +0.5, 15.2, 0 |
| S6 without the camber moment | 7.09 | +0.016 | 0.15 / 0.26, 0.22 | |
| T6, tube, k 3 | 6.58 | -0.048 | 0.16 / 0.50, 0.25 | +0.1, 14.9, 0 |
| T6, tube, k 10 | 7.24 | +0.023 | 0.17 / 0.72, 0.55 | +0.1, 14.7, 0 |
| T6, tube, k 30 | 7.71 | +0.065 | 0.17 / 1.81, 1.50 | +0.1, 14.1, 0 |
| P6, springs alone, k 3 | 7.97 | -0.073 | none | +11.2, 9.7, +6 |
| P6, springs alone, k 10 | 11.09 | -0.011 | none | +3.8, 4.3, +2 |
| P6, springs alone, k 100 | 12.16 | -0.292 | none | +0.4, 0.5, -3 |

The total counts the twist drive's net work. A drive that cannot take work
back pays the positive-only figure instead: the k = 10 tube is then 7.8 W at
this point and 8.9 W at 3.0 Hz/40/20 (7.73 + 1.57 - 0.44).

## Take-off (body pitch 90, stroke 85, root feather 50, 4.5 Hz; 4.0 Hz reads alike)

| arm | lift, weights | crank, N m | gate | twist: peak torque, tip swing |
|---|---|---|---|---|
| K1 one panel | 1.856 | 11.18 | yes | |
| L6 every hinge held at zero | 1.805 (-2.8%) | 11.00 | yes | 1.37 N m at a hinge, tip 2.7 deg (the servo's give) |
| T6 tube holding the tip, k 3 | 1.617 (-12.9%) | 10.79 | yes, by 0.017 | 0.94 N m, mid-span hinges 1.4-8.9 deg |
| T6 tube, k 10 | 1.786 (-3.8%) | 11.12 | yes | 0.98 N m |
| T6 tube, k 30 | 1.822 (-1.9%) | 11.32 | yes | 0.97 N m |
| P6 springs alone, k 10 | 1.231 (-33.7%) | 9.48 | no | tip 20 deg |
| P6 springs alone, k 30 | 1.692 (-8.8%) | 10.10 | yes | tip 8 deg |
| P6 springs alone, k 100 | 1.810 (-2.5%) | 10.97 | yes | tip 2.6 deg |

## What this specifies

- **A driven tip in a wing of torsional stiffness GJ about 1.1 N m2** (10
  N m/rad per 0.11 m; the window read is 10-30). One torque between the root
  rib and the tip rib, about the feather's line: 0.55-0.69 N m swinging at
  the beat in cruise, 1.0 N m held at take-off. Softer than about 3 N m/rad
  the wing does not sustain; at 30 the spring work costs 0.5-1 W more.
- The drive's mean power is under half a watt; its size is set by torque
  (1 N m at the tip rib) and by the 2-3.5 Hz it must follow, not by power.
- A passive wing is out for this pair of jobs as modelled: cruise wants 3
  N m/rad, take-off 30-100. A lock that stiffens the wing tenfold for
  take-off would make it a candidate; no lock is modelled.

## The user's point: twist that wraps the air (2026-11-17)

Nothing in this pass adds or multiplies a force. The twist is six rigid
panels on hinges; each strip's force comes from its own chord, incidence and
velocity, the twisting rate and the panels' inertia are the model's, and the
camber's pitching moment is derived from the arc. What the model shows of
"wrapping": the incidence varies along the span as the built shape says, and
at take-off W6's grid already put more twist on one half-beat than the other
(a held twist under the oscillation) and it lost lift either way.

What it cannot show, so the question is open rather than answered no: the
chordwise cup. Under separation the normal force is the flat plate's
whatever the section (`docs/TYPED_FORCES_AUDIT.md`), so a wing that bellies
towards the air on the power stroke and flattens on recovery reads the same
as a flat one. To get that from the shape the separated law must take the
section: proposed pass W8, a two-dimensional vortex-shedding computation of
an arc at 30-90 deg, concave and convex to the flow, checked against the
measured flat plate (1.98) and published cup and half-tube drag, and only
then tabulated for the strips.

## What these numbers rest on

- Held body, no tail (pass C1 has the free body, with the twist still
  turned by hand).
- The section's pitching moment is the arc's (-0.126), not Goe 417a's own,
  which the table used does not carry. It is half the hinge torque.
- Six rigid panels: the staircase is within 2% in power of the linear twist;
  three panels are not (Q2). No bending, no hinge friction, no gaps.
- The springs' damping is 5% of critical (typed); the drive's servo is ideal
  (stiff, no rate or torque limit, no mass). A real servo at 1 N m and 3.5 Hz
  with +-20 deg is a hardware question not looked at here.
- Take-off rests on the vortex law (RESULT_W5: lift supported, drag
  pessimistic for a cambered wing); the gate points sit at 10.8-11.3 N m of
  a 12 N m cap.

## The drawing against the design

`figures/W7_chain6_twist15.png`, `W7_chain6_locked.png`,
`W7_chain3_twist15.png`, from the built model, opened before the run and
again for this file:

- Span 1.400 m, area 0.3400 m2, chord 0.258 m, mass 0.600 kg: OK.
- Top view: six panels on each wing, five hinges between them on one line
  0.026 m behind the leading edge (0.1 chord, the feather's line); roots at
  the hull's sides.
- Side view: the panels fan about that line, each step the same, growing to
  the tip, opposite at the two ends. Locked: all six in one plane.
- Seen and recorded: the panels are drawn and modelled with no gap and no
  overlap, which a real segmented skin would not have; the drive (tube or
  line) is not a part of the drawing because it is not a body in the model,
  only a torque between root and tip.
