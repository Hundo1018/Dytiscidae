# Pass W6, result: a spanwise twist that is a control

2026-11-17. Frozen text: `PREREG_W6_twist.md` (`dd950a1`). Run: `w6_twist.py`
-> `results/W6_twist.jsonl` (3,100 rows, all finite, every cruise row
trimmed), read by `w6_read.py`.

## Outcome in one paragraph

Feathering the tip alone, with the root keeping its incidence, cruises at
8 m/s for 7.1-7.3 W where the rigid feather needs 10.0 W with 4-8 deg of held
washout and 21.5 W with none, and it does so with no held twist at all. With
0.1 N of thrust in hand it needs 7.7 W against the rigid feather's 21.5-22 W.
The take-off does not want twist and hardly minds it: at the same stroke a
held twist of 8 deg either way costs 2% of the lift and 17% more crank torque,
and that torque crossing the 12 N m cap is all of W4's "washout loses the
take-off". So one wing does both with the tip held at zero twist: a rigid
feather for the take-off and a tip that feathers 15 deg either way in cruise.
Nothing sustains at 6 m/s. The cost of driving the twist is not in any of
these numbers.

## Controls

Twist 0, share 0 reproduces W5's twist-0 rows (8 m/s: 6 of 80, 21.5 W;
take-off 6 of 100, 1.88 weights) and -8, 0 its -8 rows (15 of 80, 10.0 W;
0 of 100, 1.45 weights).

## Cruise (sustaining, crank under 12 N m; cheapest by fluid power)

| feather made as | tip held at | 8 m/s: points of 80 | cheapest | at | with +0.1 N |
|---|---|---|---|---|---|
| hinge (rigid) | 0 | 6 | 21.5 W | 3.5 Hz, stroke 50, feather 15 | 21.5 W |
| hinge | -4 | 15 | 10.2 W | 2.5 Hz, 40, 5 | 21.8 W |
| hinge | -8 | 15 | 10.0 W | 2.5 Hz, 40, 5 | 22.0 W |
| hinge | -12 | 11 | 11.6 W | 2.5 Hz, 50, 10 | 22.1 W |
| hinge | -16 | 3 | 23.5 W | 3.0 Hz, 60, 15 | 29.6 W |
| tip twist (share 1) | 0 | 29 | 7.2 W | 2.5 Hz, 40, tip 15 | 7.7 W |
| tip twist | -4 | 34 | 7.1 W | 2.5 Hz, 40, tip 15 | 7.7 W |
| tip twist | -8 | 30 | 7.3 W | 3.5 Hz, 30, tip 15 | 8.4 W |
| tip twist | -12 | 24 | 8.1 W | 2.5 Hz, 50, tip 20 | 9.9 W |
| tip twist | -16 | 16 | 9.7 W | 2.0 Hz, 60, tip 15 | 13.6 W |

Crank torque at these points 1.3-2.5 N m (cap 12). The wing's incidence at
trim is -3 to -7 deg, which with this section's zero-lift angle is a small
positive lift angle.

6 m/s: one point with the rigid feather and no twist (30.4 W, as W5), one at
-4 (30.5 W), none with any tip-twist setting (short by 0.17-0.29 N).

## Take-off (gate: 1.6 weights, crank 12 N m, 150 W)

| feather made as | tip held at | gate points of 100 | most lift under the caps |
|---|---|---|---|
| hinge | +8 / +4 / 0 / -4 / -8 | 0 / 0 / 6 / 0 / 0 | 1.46 / 1.47 / 1.88 / 1.48 / 1.45 |
| half hinge, half twist | +8 / +4 / 0 / -4 / -8 | 5 / 8 / 10 / 6 / 3 | 1.89 / 2.04 / 2.05 / 2.04 / 1.88 |
| tip twist | +8 / +4 / 0 / -4 / -8 | 0 / 3 / 2 / 2 / 0 | 1.44 / 1.79 / 1.80 / 1.79 / 1.42 |

The first row's cliff is the cap, not the lift. At one stroke (body pitch 90,
4.5 Hz, feather 50), rigid feather:

| tip held at | +8 | +4 | 0 | -4 | -8 |
|---|---|---|---|---|---|
| lift, weights | 1.84 | 1.87 | 1.88 | 1.87 | 1.84 |
| crank, N m | 13.25 | 12.26 | 11.45 | 12.39 | 13.40 |

Either sign costs the same: a flapping wing at rest meets the air from one
side on one half-beat and from the other on the next, so a held twist suits
one and spoils the other.

Making half or all of the take-off feather as twist lowers the lift at every
stroke (4.5 Hz, feather 50: 1.88, 1.66, 1.50 weights for hinge, half, tip)
and lowers the crank metric at the larger feathers (5 Hz, feather 60: 14.7,
10.5, 13.4 N m). The second is accounting: the crank metric charges the
hinge for the feather's inertia and fluid load, and the twist is charged
nothing (PREREG section 2). On the stroke's part of the crank alone the
counts are 8, 10 and 2 gate points and the same best lifts. **No take-off
advantage for a twisting feather is established.**

## Against the frozen predictions

| | frozen | read | |
|---|---|---|---|
| Q1 | rigid, 8 m/s: minimum at -8 or -12, in 8-10 W; -16 dearer than -12 | minimum 10.03 W at -8; -4 is 10.2 W; -16 23.5 W against 11.6 | the place holds; the number is 0.03 W outside the stated range |
| Q2 | share 1 at twist 0 under 16 W; best share-1 within 15% of best rigid | 7.2 W; best share-1 7.1 W against 10.0 W, 29% lower | first part holds, **second falsified** (tip twist is better than I allowed) |
| Q3 | +0.1 N for under 18 W somewhere | 7.7 W (tip twist, 0 or -4) | holds |
| Q4 | 6 m/s: a setting with 5+ points under 25 W | none | **falsified** |
| Q5 | take-off rigid: best at 0 or +4, 1.80-2.00; +8 not 0.05 above 0; -4 between | 1.88 at 0; +8 1.46; -4 1.48 | holds, but for the cap's reason, not the lift's |
| Q6 | share 0.5 and 1: less lift than rigid at the same twist, no gate point at share 1 | less lift at the same stroke, yes; under the caps share 0.5 has more gate points and share 1 has 2-3 | **falsified** as stated (the cap again, and the uncharged twist) |
| Q7 | one wing meets both gates and cruises under 10.5 W; the turn between the settings is 8 or 12 deg | meets both, cruise 7.2 W, with the tip held at 0 in both | holds on the gates; **the turn is wrong**: no held twist is needed, the tip's in-beat feathering replaces it |

Three of seven wrong in whole or part. What I had wrong in common: I treated
twist as a setting per flight mode. The data say it is a motion within the
beat, and the held part is worth nothing once the tip feathers.

## What this specifies for the wing, and what it does not

- Root: a feather hinge with at least 50 deg of travel either way, driven
  with the stroke (the cam of pass 1i), used for the take-off; in cruise it
  holds the incidence and trims it by a few degrees.
- Tip: turns against the root by 15 deg either way at 2.5 Hz in cruise,
  in the phase the hinge's feather has. At take-off it is held at zero
  (4 deg off costs 0.8 N m of crank and all six gate points). The pivot line in the model is the leading edge.
- Not specified: how the tip is turned. A driven tip (a servo and a torsion
  tube) and a passive one (a torsionally soft wing that the air twists, locked
  for take-off) are both open; the passive one would need pass 1f-1h's free
  feather repeated for torsion, and those passes found a free feather hard to
  hold in phase.

## What these numbers rest on

- **The twist's own cost is absent**: twisting rate in the strips'
  velocity, torsional inertia, and the torque and power to drive or hold it.
  A hand estimate for cruise (dynamic pressure 38 Pa, area 0.34 m2, chord
  0.26 m, a moment arm of a tenth of the chord, tip rate 4 rad/s falling to
  zero at the root) is some tenths of a watt against 7 W; it is an estimate,
  not a measurement, and the peak torque on a tip servo is not estimated.
- In cruise the strips fly attached on a measured section, used
  quasi-steadily in a beat (reduced frequency about 0.25 at 2.5 Hz, 8 m/s):
  unsteady lift lag is in the model (Wagner), dynamic stall data are not
  (RESULT_W5).
- The margin is thin in absolute terms: 7.2 W buys 0.045 N of net thrust;
  0.1 N costs 7.7 W. The hull's drag is in; the tail's trim drag, the free
  body's pitching and the height and speed loop are not (held rig).
- Take-off is the vortex law: lift within 0.2 of flat revolving wings, drag
  pessimistic for a cambered wing (RESULT_W5). The 6 gate points sit at
  11.4-11.9 N m of a 12 N m cap.
- Structure: a 3.4% thick cambered plate 0.66 m long that must be stiff in
  bending, soft or driven in torsion over 15 deg, and stiff again at take-off.

## The drawing against the design

`figures/W6_takeoff.png`, `W6_beat_down.png`, `W6_beat_up.png`, from the built
model with the twist built in at 0, -15 and +15 deg (the two ends of the
cruise beat; `rig.run(strip_twist=)` reproduces a built twist to four
decimals, PREREG section 3). Opened and checked:

- Span 1.400 m, area 0.3400 m2, chord 0.258 m, mass 0.600 kg: all within 2%
  of the design (the table on each figure). Roots at the hull's sides, a
  gap of the hull's width between them. OK.
- Side view, nose-down end: the outer strips' forward edge is lower than
  their rear, growing to the tip, the root strips level. Nose-up end: the
  reverse. Matches "tip -15 / +15, root unmoved". OK.
- Top view: the twisted tip's chord projects slightly shorter, the leading
  edge stays straight: the strips turn about the leading edge. The feather
  hinge of the rigid case is at a tenth of the chord, so the two motions do
  not share an axis in the model; small, recorded.
- The section inset has its curved side up; the same section on both wings.
- What the drawing shows that the design does not have: no part that makes
  the twist. The strips are simply turned.
