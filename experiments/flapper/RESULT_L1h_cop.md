# L1 pass 1h result: the passive feather with the pitching-moment model on (2026-11-15)

Pre-registration: `PREREG_L1h_cop.md` (commit ab66da3). Data:
`results/L1h_cop.jsonl`: 1,520 rows (800 take-off, 720 cruise), all finite;
51 cruise points could not be trimmed to the weight. Reader: `l1h_read.py`.
One pass.

## (a) Take-off: 0 of 800 meet the gate (P1 confirmed)

| axis h | law | lowest drive torque anywhere | cheapest point at >= 1.6 weights | most lift within 12 N m |
|---|---|---|---|---|
| 0 | plate | 9.7 N m | 15.0 N m, 83 W | 1.41 weights |
| 0 | insect | 9.9 | 15.3 N m, 82 W | 0.69 |
| 0.1 | plate | 10.9 | **13.5 N m, 69 W** (3.5 Hz) | 0.01 |
| 0.1 | insect | 11.1 | 14.0 N m, 70 W | 0.72 |
| 0.2 | plate / insect | 13.9 / 13.2 | 14.6 / 15.7 N m | none within 12 |
| 0.3 | plate / insect | 15.6 / 14.9 | 15.6 / 16.3 N m | none |
| 0.4 | plate / insect | 15.0 / 15.8 | 20.9 / 16.8 N m | none |

- The closest any passive point comes is 13.5 N m and 69 W for 1.6 weights:
  12% over the torque gate and well under the power gate. The prescribed
  feather (1a) needed 10.7 N m and 64 W. Pass 1f needed 20.7 N m (built
  damping, no centre-of-pressure law); pass 1g needed 16.4.
- **P2 refuted.** Moving the axis aft raises the most lift a stroke can give
  (to 3.6-4.0 weights at 5 Hz, against 2.8 at the leading edge). But it also
  raises the least torque any stroke needs, to 13.2-15.8 N m at h >= 0.2. So
  no aft point lies within 12 N m at all.

## (b) Cruise: 0 of 720 meet the gate (P3 confirmed)

- 6 points sustain level flight. Every one costs 68-173 W and 13.2-30.1 N m,
  against the gate's 41 W and 12 N m:
  - plate, h = 0: one point, 153 W;
  - plate, h = 0.2: two points, the cheaper at 92 W;
  - plate, h = 0.4: one point, 68 W, 16.9 N m;
  - insect, h = 0.3: two points, the cheaper at 72 W, 19.7 N m.
- **P4 refuted.** I predicted that the laws would disagree, with "plate"
  sustaining aft and "insect" not. Both laws find sustaining points at
  h >= 0.2 (plate 3, insect 2), so on whether there is thrust at all they
  agree.
- Where the cheap cruise points are, at 2.5-3 Hz and 6-11 W, the passive
  wing still loses 0.18-0.77 N net.

## Reading

- Rival (b) wins: with the pitching moment modelled, an aft hinge is stable
  and earns lift and thrust, but only by holding the wing at an incidence the
  stroke pays for in torque.
- Rival (c), that the law decides it, is not borne out on the gates. Both
  laws fail them the same way, and the torque figures differ between laws by
  under 1.2 N m at the cheapest points.
- Rival (d), nothing changed from 1g, is refuted. The leading-edge hinge's
  take-off torque falls from 16.5 to 15.0 N m, and the thrust no longer
  depends on a divergent axis.
- Rival (a), that a passive feather does what a prescribed one did, fails by
  a narrow margin at take-off (13.5 against 12 N m, power under the gate)
  and by a wide one in cruise (68 W against 41 W).
- **Decision: a single spring hinge does not carry K1 on its own.** The cheap
  cruise of 1a (19.5 W) and its take-off within the cap both need a feather
  angle set by the stroke rather than by the air: a stroke-linked cam, which
  is the next pass.

## Not covered

Free body; spanwise-varying stiffness (a hinge stiff at the root and soft at
the tip); a hinge whose stiffness differs on the two half-strokes; Re 1e5
validation of either law (one point at Re 1e4 checks them, within about
20%).
