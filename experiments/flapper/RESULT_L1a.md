# L1 pass 1a result: the reference machine at scale, held (2026-11-15)

Pre-registration: `PREREG_L1a.md` (commit e729104, before any rollout).
Data: `results/L1a.jsonl`, 272 points, all finite, one pass, nothing tuned.
`MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/l1a.py experiments/flapper/results/L1a.jsonl`

Fixture: 0.60 kg (5.88 N), one pair of rectangular plates, span 1.4 m,
0.34 m2, body held, stroke and feather prescribed. Every force is an upper
bound for a drive that cannot feather (K1).

## (a) Lift from rest, stroke 85 deg

| frequency | best vertical force / weight | aerodynamic power | peak drive torque (aero, inertia) |
|---|---|---|---|
| 0 | 0.00 | 0 | 0 |
| 3.0 Hz | 0.96 | 30 W | 6.2 N m (5.7, 2.4) |
| 3.5 Hz | 1.30 | 48 W | 8.4 (7.7, 3.3) |
| 4.0 Hz | 1.70 | 72 W | 11.0 (10.1, 4.3) |
| 4.5 Hz | 2.15 | 102 W | 13.9 (12.8, 5.5) |
| 5.5 Hz | 3.22 | 186 W | 20.7 (19.1, 8.2) |

Best with the body vertical and a feather amplitude of 60 deg; at 45 deg of
body pitch the 4.5 Hz figure is 1.53.

- Frozen gate (1.6 W, torque <= 12 N m, power <= 150 W): **met at 5 points**;
  the cheapest is 4.0 Hz, 64 W, 10.7 N m.
- Frozen prediction: 1.0 W at 3.5-4.5 Hz for 130-260 W, gate not met.
  **Falsified**: 1.0 W comes at 3.0-3.5 Hz for 43 W, and the gate is met.
  The lead over-predicted the power by about four.
- Against L0: L0 put the take-off at 4.2 Hz, 10.5 N m and 220 W electrical.
  The model, pushed to 1.6 W to cover its known optimism, says 4.0 Hz and
  10.7 N m, and 64 W in the air, which R0's factor of 2.6-4 makes 166-256 W
  electrical. Two routes (one tethered machine scaled; this model corrected
  by another machine) land within each other's range. They share the
  blade-element picture, so this is agreement, not independent proof.
- The torque gate is the tight one: 10.7 of 12 N m, and 4.5 Hz already
  breaks it.

## (b) Level cruise, lift trimmed to the weight

120 of 120 trimmed, 35 sustain. Lowest power per stroke and speed:

| | stroke 40 | stroke 60 | stroke 85 |
|---|---|---|---|
| 6 m/s | none | 30.8 W at 3.5 Hz, 8.9 N m | 38.1 W at 3.0 Hz, 9.7 N m |
| 8 m/s | 19.5 W at 4.5 Hz, 10.0 N m | 28.8 W at 4.0 Hz, 12.0 N m | 28.4 W at 2.5 Hz, 8.4 N m |

- Frozen gate (sustains, <= 41 W, <= 12 N m): **met at 6 points.**
- Frozen prediction (lowest 20-40 W, gate met narrowly): held (19.5 W).
- With R0's factor, 19.5-38 W is 51-152 W electrical against 164 W from the
  three cells over 15 minutes: the margin is 1.1 to 3.2, and it is 1.1 at
  the worst end of a band taken from one machine.
- Cruise asks 8-12 N m, as much torque as the take-off. The drive is sized
  by torque in both.

## (c) Water, held at 0.3 m/s

| stroke, frequency | forward force | peak shoulder torque | power |
|---|---|---|---|
| 85 deg, 0.15 Hz | -7.4 to 4.3 N | 25-51 N m | 1-4 W |
| 40 deg, 0.38 Hz | 11.8-14.8 N | 59-95 N m | 10-13 W |
| 15 deg, 1.0 Hz | 46-89 N | 138-230 N m | 37-170 W |
| not flapping | -0.47 N | 0 | 0 |

- Frozen gate (forward force > 0, torque <= 15.8 N m): **failed at all nine
  points**, on torque, by 1.6 to 15 times.
- Frozen prediction (gate met; force 2-10 N; torque within a factor of two of
  15.8): **falsified.**
- **`L0/mechanism.py` is wrong about water.** It scaled the air torque by the
  density and (stroke x frequency)^2 and concluded that small fast strokes
  ease the load. In the model the load at small strokes is 10 times that
  rule: a feather amplitude of 30-60 deg at 1 Hz turns a 0.26 m chord
  through water fast enough to dominate, and the solver's added-mass force
  on the wing grows with frequency squared whatever the stroke. The review
  of plan draft 1 named added mass as "tens of kilograms"; the paper pass
  left it out. K1's stated benefit in water is withdrawn until re-derived.
- What the row needs is small: the drag at 0.3 m/s is 0.47 N, and the slow
  full stroke gives 4.3 N for 33 N m, about 0.13 N per N m, so the thrust to
  swim may cost 4 N m at a slower or shallower stroke. Not measured: the
  grid had no such point.

## Belief before and after

Before: lift from rest would fail on power; cruise would pass narrowly; water
would pass. After: lift from rest and cruise pass the gates set for them on a
held body with ideal feathering, both tight on torque; the water points fail
on torque and the paper rule behind them is wrong. It moves back if a wing
that twists passively gives much less than the prescribed feather, or if a
free body cannot be held at the attitudes used here.

## Next pass (1b)

1. Water map at low frequency and small feather, with the torque cap, and
   `mechanism.py` rewritten to include added mass.
2. The feather as the wing would do it: flap-only, passive twist. This needs
   the compliance model first.
3. Free body and the drive's torque-speed line.
