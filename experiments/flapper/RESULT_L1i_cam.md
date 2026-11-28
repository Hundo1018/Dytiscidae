# L1 pass 1i result: the feather driven from the crank by a cam (2026-11-15)

Pre-registration: `PREREG_L1i_cam.md` (commit a0218dc). Data:
`results/L1i_cam.jsonl`: 1,188 rows (600 take-off, 576 cruise, 12 passive
comparison points), all finite. Reader: `l1i_read.py`. One pass. Gates use
the crank metric: the instantaneous power of stroke and feather, fluid and
inertia, over the crank rate, expressed as the shoulder torque of a K1
rocker at 85 deg. The shoulder metric of 1a-1h is listed beside it.

**Correction (2026-11-15, found in pass 1l): the two gates are met with
different feather amplitudes.** All 16 take-off gate points have Psi 50 or
60 deg; all 26 cruise gate points have Psi 20 or 30 deg. At 40 deg take-off
tops out at 1.47 weights under the caps and no cruise point sustains; at
30 deg take-off tops out at 0.92. So "one motor with a cam meets both gates"
holds only if the cam's feather amplitude can be changed between take-off
and cruise. That adjustment is not designed, and its mass and mechanism are
unaccounted.

## (a) Take-off: met under both laws (P1 confirmed)

16 points meet the gate: 7 under "plate", 9 under "insect". Every hinge axis
tried (0, 0.1, 0.2) has at least one.

| law | h | body | f | Psi | lift | power | crank (cam) | crank, stroke only | shoulder metric |
|---|---|---|---|---|---|---|---|---|---|
| plate | 0 | 90 | 4.0 | 50 | 1.64 | 100 W | 10.9 N m | 8.3 | 12.0 |
| plate | 0.1 | 90 | 4.5 | 60 | 1.84 | 118 W | 11.7 | 6.8 | 13.4 |
| plate | 0.2 | 90 | 4.5 | 50 | 1.65 | 123 W | 9.6 | 9.8 | 12.5 |
| insect | 0 | 90 | 4.0 | 60 | 1.70 | 92 W | 11.6 | 6.6 | 11.9 |
| insect | 0.1 | 90 | 4.5 | 50 | 1.85 | 129 W | 10.7 | 9.7 | 13.7 |

- The margins are thin: the crank metric at gate points is 9.6-11.7 N m
  against 12, and aerodynamic power is 92-130 W against 150.
- Over the beat the crank torque never reverses at these points: its minimum
  is +1.5 to +3.2 N m.
- The cam's feather follower carries at most 1.9-3.6 N m.
- **P3 refuted as worded.** At each law's cheapest gate point the cam adds
  -1% (insect) and -2% (plate), not 30-90%: there the feather's power peaks
  where the stroke's falls. At Psi 60, though, the cam adds 67-76%
  (6.6-6.8 to 11.0-11.7 N m). The cost depends on the feather amplitude,
  from nothing to three quarters.

## (b) Cruise: met under both laws (P2 confirmed)

26 points sustain level flight within 41 W and 12 N m.
- Cheapest: 8 m/s, 3.5 Hz, stroke 60, Psi 20, psi0 -7.4, h 0.2: 27.1 W at a
  crank metric of 5.5 N m, the same under both laws.
- Pass 1a's cheapest was 19.5 W, with no centre-of-pressure law and no cam.
- The crank torque briefly reverses here (minimum -0.8 to -1.2 N m): the wing
  drives the motor for part of the beat.

## (c) The passive hinge under the crank metric (P4 refuted; control passes)

- **Control.** All 8 of pass 1h's cheapest passive take-off points reproduce
  their 1h lift and shoulder metric exactly.
- **P4 refuted.** Under the crank metric those points read 14.0-16.5 N m,
  higher than their shoulder metric (13.5-15.7 N m), not lower. The passive
  hinge's failure was not the metric's.
- The 4 passive cruise points cost 68-173 W at 12.7-31.5 N m, as in 1h.
- So rival (c) is out. The cam is what changes the answer, not the way
  torque is counted.

## Reading

- **One motor, a crank, and a cam that also turns the feather meets both air
  gates in this model, under both centre-of-pressure laws.** It does so
  with:
  - take-off at 4.0-4.5 Hz, 92-130 W, crank metric 9.6-11.7 of 12 N m;
  - cruise from 27 W.
- The single spring hinge did not meet either gate (passes 1f-1h). Rival (a)
  holds: a cam gives what 1a's prescribed feather gave.
- Against the shoulder metric of 1a-1h, the cheapest gate points read
  11.9-13.7 N m. A drive sized by that older, more conservative metric would
  need a cap of about 14 N m for the same points.
- **Decision: K1 with a cam is the reference drive for L1's next passes.**
  Each of these is unmeasured and could move the margin:
  - a free body;
  - a real crank-rocker and cam profile, which are not sinusoidal;
  - cam friction;
  - the trim servo for psi0 (TYPED, about 10 g);
  - water and land with the cam engaged. A cam that fixes Psi ties the
    feather to the stroke in water too: Psi 30-60 there was what pass 1b
    measured.

## Not covered

Free body; motor torque-speed line; non-sinusoidal kinematics of a real
linkage; follower mass and friction; the R0 calibration band beyond the
1.6-weight take-off target (already inside the gate); Re 1e5 validation of
either centre-of-pressure law.
