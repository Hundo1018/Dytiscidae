# L1 pass 1m result: a tail and pitch (2026-11-15)

Pre-registration: `PREREG_L1m_tail.md` (commit after 6501a9a; one probed
point disclosed there). Data: `results/L1m_tail.jsonl`, 253 rows, all finite,
one pass. Reader: `l1m_read.py`. Mechanism drawn from the built model:
`figures/L1m_tail0.06.png`, `figures/L1m_tail0.03.png`. Points: the 13 cruise
points that met the gate released in x and z in pass 1k, and pass 1i's 16
take-off points. Torque-driven rig, cam drive, all-up mass 0.60 kg, hull
inertia 0.012 kg m2 unless said. The tail is all-moving, on a 0.3 m boom,
0.59 m behind the centre of mass.

## Held: the pitching moment about the centre of mass

| | no tail | tail 0.06 m2 |
|---|---|---|
| restoring stiffness, N m per deg of nose-up | -0.036 to -0.007 (unstable) | +0.032 to +0.088 |
| mean at level, N m | -0.05 to +0.16 | -0.31 to +0.07 |
| swing through the beat, N m | 0.72-1.39 | 0.97-1.57 |

Tail authority: 0.078-0.140 N m per degree of incidence.

- **P7 refuted at both ends, narrowly**: I predicted 0.05-0.12 with the tail
  (4 points are below 0.05) and -0.03 to +0.01 without (lowest -0.036).

## Released in x, z and pitch, 40 beats (13 cruise points)

| arm | beat-mean pitch within 15 deg | flies (all gates) | of those whose pitch holds |
|---|---|---|---|
| no tail | 0 | 0 | all beyond 15 deg within 1-3 beats |
| tail 0.06, fixed | 7 | 5 | swing 1.1-3.5 deg; 25-33 W; crank 4.6-5.4 N m |
| tail 0.06, feedback | **13** | 6 | swing 1.1-3.3 deg; tail -9.9 to +7.6 deg; 27.5-33.1 W; crank 4.7-7.1 |
| tail 0.03, fixed | 5 | 3 | |
| tail 0.03, feedback | **13** | 5 | swing up to 9.5 deg; tail -12.5 to +16.6 deg |
| tail 0.06, fixed, inertia 0.0045 | 7 | 4 | swing 1.5-3.9 deg |
| tail 0.06, fixed, inertia 0.03 | 7 | 5 | swing 0.7-2.6 deg |
| released 10 deg nose-up, fixed | 7 | 5 | settles inside 20 beats at 7 of 13 |
| released 10 deg nose-up, feedback | **13** | 6 | within 5 deg from the first beat at 13 of 13 |

- **P1 confirmed.** No tail: 0 of 13.
- **P2 refuted.** The fixed tail holds pitch at 7 of 13 (11 predicted) and
  flies at 5 (7 predicted). The probed point was one of the easy ones.
- **P3: two parts confirmed, one refuted narrowly.** Feedback holds pitch at
  13 of 13 and flies at 6, no fewer than the fixed tail; tail travel stays
  within 10 deg. The late swing reaches 3.3 deg (3 predicted).
- **P4 refuted.** The 0.03 tail fixed holds pitch at 5 of 13 (9 predicted).
- **P5 confirmed.** Inertia from 0.0045 to 0.03 does not change the count (7,
  7, 7); it changes the swing (3.9 to 2.6 deg at most).
- **P6 refuted for the fixed tail** (7 of 13 inside 20 beats; 10 predicted),
  **confirmed with feedback** (13 of 13).
- **P8 confirmed.** No point whose pitch holds fails on power or the crank
  metric: at most 33.9 W of 41 and 7.1 of 12 N m. The tail's servo takes
  0.06 W.
- **What fails with feedback is the flight path, not pitch.** Of the 13
  points with pitch held, 6 leave the 3 m height band (as far as 16 m down),
  7 are still climbing or sinking by more than 0.3 m/s at the end (as much
  as 1.6 m/s down), 2 leave 5-11 m/s. Nothing holds height or speed: the cam
  offset is the trim of a body with no tail, and the tail's own load and
  drag change it.

## Take-off (16 points, 4 beats)

| arm | within 45 deg and climbs 1 m | largest pitch excursion |
|---|---|---|
| pass 1k: no tail, inertia 0.0045 | 6 | 25-118 deg |
| no tail, inertia 0.012 | 10 | 17-107 deg |
| tail 0.06 with feedback, inertia 0.012 | **16** | 6-15 deg |

- **P9 refuted, in the machine's favour.** I predicted at most 8 and that a
  tail has no flow at rest. All 16 hold within 15 deg and climb 1.10-2.12 m.
  The released body is climbing at over 1 m/s within a beat and the tail
  sits 0.59 m from the centre of mass, so it has flow and a long arm.
  Inertia alone gives 10.

## Reading

- Rival (a) fails: static stability from a fixed tail is not enough at 6 of
  13 points. Rival (b) holds: with pitch held by feedback, height and speed
  wander. Rival (c) fails: power has margin. Rival (e) fails.
- **Decision: K1 with a cam carries an all-moving tail under pitch feedback.**
  With it the released body holds pitch at every cruise and take-off point
  tried, inside the power and torque caps. 0.06 m2 leaves the tail half its
  travel; 0.03 m2 works but swings 9.5 deg and uses 16.6 of 25 deg.
- **Not yet a machine that holds level flight.** The next loop is height and
  speed (flapping frequency or the cam offset, and the pitch the tail holds).
  6 of 13 points pass the whole gate without it.

## What this does not establish

- The tail sits in the wings' wake and downwash; strips do not see each
  other. This could change the tail's authority at both gates, most at
  take-off.
- Roll and yaw; gusts; sensor noise and delay (the feedback sees the true
  pitch through a one-beat low-pass); the servo is a rate limit on an ideal
  actuator.
- What the beat-frequency pitching moment is made of (0.7-1.6 N m swing).
- The tail's mass is taken from the ballast; the boom's stiffness is not
  modelled. Water and land with a tail on.

## The drawing against the design (looked at, 2026-11-15)

`figures/L1m_tail0.06.png`, `figures/L1m_tail0.03.png`, drawn by `draw.py`
from the built model and checked by eye and by its table.

- Tail span, chord and area match what was asked for (0.50 x 0.12 = 0.060 m2
  and 0.36 x 0.085 = 0.031 m2). The tail is level with the hull, on a 0.3 m
  boom, leading edge forward, hinge on its leading edge; its centre is 0.59
  and 0.57 m behind the centre of mass. Wing chord 0.258 m, area 0.340 m2,
  mass 0.60 kg match the design.
- **The wing span differs from the design: 1.32 m built against 1.40 m.**
  The wings' roots and the stroke axis are on the hull's centre line, not at
  its sides, so 0.04 m of each wing is inside the hull. This is the fixture
  of every L1 pass. Not corrected here; its effect is not measured.
- A first version of the tail was built leading edge aft (seen in the model
  dump before the sweep, turned round before the pre-registration).
