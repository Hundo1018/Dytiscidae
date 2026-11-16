# Pass W4 pre-registration: the wing's planform, twist and section (2026-11-17)

Frozen before the sweep. Code at this commit: `w4_wing.py`, `w4_read.py`,
`body.built(wing_shape=dict(taper, twist_tip_deg, camber))` with a measured
section (pass W3). Drawings from the built models, opened before freezing:
`figures/W4_taper1.0.png`, `W4_taper0.6.png`, `W4_taper0.4.png`.

## Seen before freezing

- Pass W3's result: sections barely change pass 1i's grid, because that grid
  sustains only by hard flapping; at its gentlest strokes each law is
  0.10-0.21 N short at 8 m/s for 5-7 W.
- `probe_shape.py` (default law, one point): taper 0.5 gives 19 W and
  -0.41 N at the rectangle's stroke; 8 deg of washout 27.6 W and +0.11 N.
- On the drawings and by a numerical check: span 1.40 m and area 0.340 m2
  for every taper; a straight leading edge; root chord 0.368 m and tip
  0.147 m at taper 0.4; washout of 8 deg at both tips, the same way on both
  wings; the section's upper side up on both wings (equal lift left and
  right).
- No point of this pass's grid has been run.

## 0 Question and decision

With a measured section, does a tapered or twisted wing fly at a fraction of
the rectangle's 27 W, and does it keep the take-off gate? Decides the wing
the next passes carry in place of the flat rectangle.

## 1 Rivals

(a) Cheap flight exists and needs only gentler strokes than pass 1i tried:
planform and twist are second order. (b) Twist is what opens it: a flapping
wing's tip moves fastest, so an untwisted wing has the wrong incidence
somewhere at every instant. (c) Taper buys cruise and costs take-off, which
lives on tip area. (d) Nothing sustains below about 25 W: the model's thrust
comes only from separated flow.

## 3 Predictions

Grid: 6 and 8 m/s, 2-3.5 Hz, stroke 30-60 deg, feather 0-20 deg, hinge at
0.1 chord, cam offset trimmed to the weight. "Sustains" as pass 1i, with the
crank metric at most 12 N m.

- **P1** Goe 417a rectangle, no twist: sustains at 8 m/s for 15 W or less.
- **P2** Washout of 8 deg lowers the cheapest sustaining power at 8 m/s by
  10% or more for at least one taper.
- **P3** Taper 0.4 meets the take-off gate at fewer than half as many points
  as the rectangle; taper 0.6 at half or more.
- **P4** Some shape sustains at 8 m/s for 12 W or less and meets the
  take-off gate at one point or more.
- **P5** At 6 m/s the best shape sustains for 20 W or less.
- **P6** E387 against Goe 417a, same shape: cheapest sustaining power within
  15% at 8 m/s (inside what three models of one section differ by).
- Control: the default law on the flat rectangle over the same grid. No
  prediction; it says what the finer grid alone buys.

## 4-5 Measurement

`w4_wing.py`: 11 combinations x 160 cruise points, 6 shapes x 100 take-off
points. One pass. Not covered: span and area (fixed at the design's), sweep,
a section that varies along the span, a flexible wing, the tail's trim drag,
water with the new wing, a released body.
