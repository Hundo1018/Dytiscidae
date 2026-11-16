# L1 pass 1n result: the gates on the fixture built to the design's span (2026-11-16)

Pre-registration: `PREREG_L1n_refit.md` (commit c394f45). Data:
`results/L1n_air.jsonl` (1,176 rows), `L1n_free.jsonl` (78), `L1n_water.jsonl`
(580); all finite; one pass. Reader: `l1n_read.py`. Each job is the earlier
pass's own code with the body replaced by `BODY_S`.

## The drawing against the design (looked at before the run)

`figures/L1n_fixture.png`, `figures/L1n_water0.4.png`, `L1n_water0.6.png`.
Span 1.40, 0.608 and 0.872 m as designed (1.32, 0.528, 0.792 before); area,
chord, mass and the tail unchanged; the wing roots at the hull's sides, one
stroke axis on each side, the feather axis on each leading edge. **Still a
flat rectangular plate**: no taper, twist, camber or section (see below).

## Against the frozen text

| | centre-line fixture | seated (the design's span) |
|---|---|---|
| take-off gate points (pass 1i's grid of 600) | 16 | 22, including all 16 |
| at those 16: lift, crank metric, power | | +1.3-1.5%, +1.2-4.7%, +1.1-1.5% |
| cruise gate points (of 576) | 26 | 28, including all 26 |
| cheapest sustaining cruise | 27.1 W | 26.9 W |
| cruise power where both meet the gate | | -1.5% to 0.0% |
| torque-driven rig, take-off held | 13 of 16 | 15 of 22 |
| torque-driven rig, cruise held / released in x and z | 19, 13 of 26 | 19, 14 of 28 |
| water cells meeting the gate under both added-mass models | 15 of 24 | the same 15 |
| water, best force where both pass | | 0.76-1.09 of before (median 1.03) |
| torque-driven rig in water ("wetting") | 192 of 192 stable | 192 of 192 stable |

- **P1: the loads confirmed, the count refuted narrowly.** Lift, crank and
  power are within 5% at all 16 points; 22 points meet the gate (12-20
  predicted), because the 1.4% of extra lift carries six marginal points over.
- **P2 confirmed.** 28 gate points, cheapest 26.9 W, power within 1.5%.
- **P3 confirmed on the verdicts** (24 of 24 cells) **and refuted narrowly on
  size**: one marginal cell falls by 24% (0.6 wing, feather 50 deg: 0.6 to
  0.4 N); every other is within 9%.
- **P4 confirmed.** 15 of 22 and 14 of 28.

## Reading

Rival (a) holds: with each strip as far from its own stroke axis as before,
moving the roots to the hull's sides changes the loads by about 1%. The
conclusions of passes 1i, 1k and S1 stand on the fixture the design
describes. From here on `BODY_S` is the fixture.

## What this does not establish

The wing is a flat rectangular plate, and the fluid model cannot tell it from
a designed wing in the ways that matter (`probe_shape.py`, one cruise point,
offset re-trimmed each time, not a measurement):

- camber 0, 4, 8%: power 26.9, 27.0, 27.0 W; net force +0.04, +0.05, +0.05 N.
  Camber only moves the trim offset (4.6 deg per 4%). The model has a
  zero-lift shift and nothing else.
- there is no section polar: thickness and profile do nothing to lift or
  drag.
- taper and twist do register: taper 0.5 gives 19.2 W and -0.41 N at the
  same stroke (less wing at the tip); 8 deg of washout 27.6 W and +0.11 N.

So every power, torque and gate figure of L1 is for a placeholder wing, and
a wing's section cannot be designed in this model until it carries measured
section data.
