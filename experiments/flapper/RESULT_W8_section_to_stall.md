# Pass W8, result: a measured section's drag up to its measured stall

2026-11-17. Frozen text: `PREREG_W8_section_to_stall.md` (`7456954`, addendum
`ec6a0e6`). Runs: `w8_check.py`; `w8_cruise.py` -> `results/W8_cruise.jsonl`
(480 trims, all trimmed) and `results/W8b_cruise.jsonl` (128), read by
`w8_read.py`.

## Outcome in one paragraph

The band nobody measured is not what keeps the wing from flying at 6 m/s.
Carrying the section's drag along its own trend up to the angle its lift
peaks at, or holding it flat, returns 0.00-0.07 N of a 0.15-0.7 N shortfall
and changes nothing at 8 m/s; my reading in X1 was wrong. What the grid of
passes W4-W6 lacked was tip feather: with 25-40 deg (it stopped at 20) four
points sustain at 6 m/s, all at the strongest stroke of the grid (3.5 Hz, 60
deg), for 16.4-29.9 W. So **6 m/s is flyable in this model, at more than
twice the power of 8 m/s, and only at the grid's corner**; and how a wing
swinging through its static stall every beat really behaves is still not
modelled.

## Against the frozen predictions

| | frozen | read | |
|---|---|---|---|
| L1 | leave-last-out line: median -0.002 to -0.010, within 0.012 for three quarters | 29 runs: median -0.0063; 23 within 0.012 (79%); low in all 29, by up to 0.022 | holds |
| L2 | 8 m/s: cheapest moves under 5%; sustaining points rise from 29 by at most 12 | 7.24 W in all three; 29, 33, 33 | holds |
| L3 | 6 m/s: "normal" none; "extend" at least 3, cheapest 8-14 W; "hold" more | none, none, none; nearest -0.174, -0.151, -0.148 N at 34 W | **falsified** |
| L4 | the 6 m/s verdict differs between "normal" and "extend" | it does not | **falsified** |
| L5 (addendum) | tip feather 25-40 deg: at least 3 sustain under "normal", cheapest 8-16 W, cam offset under 9 deg | 4 sustain; cheapest 16.43 W; offset -11.2 to -11.5 | count holds; **power 0.4 W over and the offset is not what I said** |

## 6 m/s with the wider tip feather (all at 3.5 Hz, stroke 60 deg)

| tip feather | power | net thrust | crank |
|---|---|---|---|
| 40 deg | 16.4 W | +0.203 N | 4.2 N m |
| 35 | 20.8 | +0.309 | 4.9 |
| 30 | 25.4 | +0.282 | 5.6 |
| 25 | 29.9 | +0.120 | 6.3 |

Nothing sustains at 3.0 Hz or below, or at a stroke under 60 deg. Power is
still falling with feather at 40 deg and thrust is positive, so the cheapest
6 m/s point is outside this grid too (more feather; more stroke or
frequency). Under "extend" the same four sustain within 0.02 N and 0.01 W.

## What is and is not explained

- The cam offset at 6 m/s stays near -11 deg whatever the feather; the
  outside solver trims the same kinematics at 5-8 deg. Part of that is the
  root gap (its sealed case needs 5, its open case 8). The rest, 3 deg or
  more, is not explained: I predicted the larger feather would bring it
  under 9 and it did not.
- The model now has three treatments of the drag between the drag run's end
  and the lift's peak; the choice moves no conclusion of this pass. The
  default stays "normal" (stored rows reproduce).
- Dynamic stall is still absent. At 6 m/s the outer wing's flow direction
  swings by tens of degrees a beat; the solver reads static tables through
  it. Real sections overshoot their static lift when pitched through stall
  and pay for it in drag and moment; neither is here.

## What it means for the machine

- 8 m/s remains the cruise: 7.2 W at the cheapest held point, unchanged.
- That point's +0.045 N of net thrust is inside the root gap's uncertainty
  (cross-check X1, with the hull case added there: 0.18, 0.27-0.32 and 0.42
  N of inviscid thrust for an open gap, a flat plate in the gap, and a
  sealed root). With pass C1 (the cheapest points cannot climb), a usable
  8 m/s cruise is a point with 0.2-0.3 N in hand, 8.5-12 W on the held rig.
- 6 m/s costs 16 W or more here. Slow flight is expensive for this wing as
  modelled; whether a real wing does better depends on dynamic stall.
- Added 2026-11-17: every count and cheapest point above keeps only rows
  whose ideal-crank equivalent torque is at most 12 (`crank_eq_peak`,
  `w8_read.py`). No crank was simulated; the figure assumes one. Under
  another drive the filter, and so the cheapest point, may differ.

## Tests

`test_a_sections_drag_can_follow_its_table_to_the_lifts_peak` (6 checks) and
mutations `section-drag-extend-is-flat`,
`section-drag-stalls-before-the-lift`, caught.

## The drawing against the design

The body is pass W6's, unchanged (`figures/W6_takeoff.png`, opened and
checked there: span 1.400 m, area 0.3400 m2, chord 0.258 m, mass 0.600 kg).
What changed is the tip's feather, to 40 deg; `figures/W8_tip40_down.png`
and `W8_tip40_up.png` draw the two ends of that beat from the built model.
