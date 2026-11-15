# L1 pass 1n pre-registration: the gates on the fixture built to the design's span (2026-11-15)

Frozen before the sweep. Code at this commit: `body.built` and `wing_seat`,
`l1a.BODY_S`, `l1n_refit.py`, `l1n_read.py`. Drawings looked at before
freezing: `figures/L1n_fixture.png`, `figures/L1n_water0.4.png`,
`figures/L1n_water0.6.png`.

## What changed

Every L1 pass so far ran a fixture whose wing roots and stroke axes were on
the hull's centre line: span 1.32 m against the design's 1.40 m (found on
pass S1's drawing). The genome can attach a part only on its parent's axis.
`body.built` now moves each main wing's root out by the hull's radius
(0.04 m) before the model is compiled; `BODY_S` asks for it. Checked on the
drawings: span 1.40, 0.608 and 0.872 m as designed; area, chord, mass and
tail as before; one stroke axis on each side of the hull, the feather axis
on each wing's leading edge. No simulation of the seated fixture has been
run before this text.

## 0 Question and decision

Do passes 1i (air gates), S1 (water gate) and 1k (torque-driven rig, held
and released in x and z) read the same on the fixture the design describes?
Decides whether their conclusions stand, and gives the points pass 1o flies.

## 1 Rivals

(a) Nothing material moves: each strip is as far from its own stroke axis as
before, so the flapping loads are the same and only the span the induced
flow is spread over grows by 6%. (b) Cruise gets cheaper by the induced
power. (c) Something else moves: the two wings no longer overlap at the
root, and the hull sits between them.

## 3 Predictions

- **P1** Take-off: at pass 1i's 16 gate points, lift, crank metric and power
  each within 5% of 1i's. Gate points on the seated fixture: 12-20.
- **P2** Cruise: gate points 20-32 (1i: 26). Cheapest sustaining point
  25.0-27.5 W (1i: 27.1). At points meeting the gate both ways, power within
  -10% to +2% of 1i's.
- **P3** Water: of pass S1's 24 cells (wing x law x feather amplitude), 20 or
  more keep their verdict; where both pass, the best force is within 15%.
  The torque-driven rig with "wetting" stays finite at 192 of 192.
- **P4** Torque-driven rig: held take-off meets the gate at 9 or more of the
  new take-off gate points; released in x and z with the offset re-trimmed,
  40% or more of the new cruise gate points still meet the gate (1k: 13/16
  and 13/26).

## 4-5 Measurement

`l1n_refit.py` runs each earlier pass's own job with the body replaced: 1i's
1,176 air points (its 12 passive-hinge points are left out), S1's `held`,
`heldref` and `free` arms, then 1k's held arms and x-z arm on the gate points
found. One pass. Not re-run: passes 1a-1h, 1j and the pitch-released arms
(pass 1o takes those up with the tail).
