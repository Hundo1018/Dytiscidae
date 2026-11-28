# L1 pass 1m pre-registration: a tail and pitch (2026-11-15)

Frozen before the sweep. Code at this commit: `body.flapper(stab_length=...)`
(an all-moving horizontal tail on a boom), `freerig.run_free(hull_inertia=,
cg_shift=, tail=)`, `l1m_tail.py`, `l1m_read.py`.

## Found before freezing

1. **Correction to pass 1k.** Its holder's pitching moment (peaks 2.1-4.2 N m
   in cruise, 5.3-8.2 at take-off) was taken about the hull frame's origin,
   which is the hull's tail end, 0.25 m behind the centre of mass. About the
   centre of mass (`probe_pitch.py`, six of pass 1i's points) it swings by
   1.4-1.5 N m in cruise and 5.1 at take-off, with means of -0.03 to +0.26.
2. Pass 1k read the moment as the cam's reaction. Not established: no choice
   of reference point within 0.3 m fore-aft and 0.1 m up-down takes more than
   18% off the cruise swing (1.21-1.42 N m left), so it is not a force on a
   fixed lever either. What it is made of is not measured here.
3. The fixture's pitch inertia (0.0045 kg m2) has the ballast as a point at
   the hull's centre. 0.56 kg spread evenly along the 0.5 m hull is 0.012.
4. `probe_tail.py`, pass 1k's cheapest released cruise point (8 m/s, 3.5 Hz,
   stroke 60, feather 20), tail 0.06 m2 (0.5 m span, 0.12 chord) 0.55 m behind
   the centre of mass:
   - held, no tail: the moment falls as the nose rises by 0.007 N m/deg the
     wrong way (unstable); with the tail it is restoring at 0.088 N m/deg, and
     the tail's authority is 0.138 N m per degree of incidence;
   - released for 12 beats, no tail: tumbles at every inertia tried (0.0045,
     0.012, 0.03); with the tail fixed at zero incidence the beat-mean pitch
     stays within 3 deg and the swing within 3.5 deg, at every inertia; with
     feedback (kp 1, kd 0.1) within 1 deg, tail travel -3.6 to +5.3 deg;
   - height change -0.3 to -0.8 m and speed 7.9-9.2 m/s over the 12 beats.
   So the predictions below for that one point are not blind.

## 0 Question and decision

Does a tail keep the released cam-driven body flying in pitch, on all of
pass 1k's released cruise points, for 40 beats (about three phugoid periods),
within the cruise gate? Decides whether K1 with a cam and a tail goes on to
the lateral axes and the bench, or pitch needs another mechanism.

## 1 Rivals

(a) A fixed tail is enough: static stability was all that was missing.
(b) Pitch holds but the flight path does not: a slow height and speed
oscillation grows without feedback. (c) The tail's drag and trim load take
the power margin (the cheapest points had -0.07 to -0.25 N net in pass 1k).
(d) It holds only on the probed point. (e) Nothing changes at take-off: a
tail has no flow at rest.

## 3 Predictions (13 cruise points, 16 take-off points; hull inertia 0.012 unless said)

"Flies": for 40 beats after release, beat-mean pitch within 15 deg of level,
height within 3 m of the start throughout, mean climb rate over the last 10
beats within 0.3 m/s, speed within 5-11 m/s throughout, mean power including
the tail servo at most 41 W, crank metric at most 12 N m. No height or speed
loop: the cam offset is pass 1k's trim, the tail's incidence zero.

- **P1** No tail: 0 of 13 fly; all pass 15 deg within 12 beats.
- **P2** Tail 0.06 fixed: pitch within 15 deg at 11 or more of 13. Flies at 7
  or more.
- **P3** Tail 0.06 with feedback: flies at no fewer points than P2; late
  pitch swing at most 3 deg; tail travel within 10 deg either way.
- **P4** Tail 0.03 (0.36 m span, 0.085 chord) fixed: pitch within 15 deg at 9
  or more of 13.
- **P5** Inertia 0.0045 and 0.03 with the fixed 0.06 tail: the count with
  pitch within 15 deg differs from P2's by at most 1.
- **P6** Released 10 deg nose-up: the fixed tail settles within 5 deg of the
  undisturbed run's end pitch, for good, inside 20 beats at 10 or more of
  13; with feedback inside 10 beats at 10 or more.
- **P7** Held: restoring stiffness with the tail 0.05-0.12 N m/deg at all 13;
  without it between -0.03 and +0.01.
- **P8** Of the points whose pitch holds under P2, at most 3 fail on power.
- **P9** Take-off, tail with feedback: at most 8 of 16 stay within 45 deg and
  climb 1 m in 4 beats (pass 1k: 6 with inertia 0.0045 and no tail); the same
  inertia with no tail gives a count within 2 of it.

## 4-5 Measurement

`l1m_tail.py`, arms as in the reader. One pass. Not covered, and each could
change the answer: the tail sits in the wings' wake and downwash, which
strips do not see; roll and yaw; gusts; sensor noise and delay beyond a
one-beat low-pass; the servo is a rate limit (8 rad/s) on an ideal actuator;
the tail's mass is taken out of the ballast, so the all-up mass stays 0.60 kg
and the centre of mass moves aft by what the tail weighs times its arm.
