# F1a-2 — can a passive wrist do what the prescribed one does?

Frozen 2026-11-12 before any rollout, after F1a was read (twist by a
prescribed wrist cut the least trimmed power from 5.28 to 3.43 W at 6 m/s
and from 5.05 to 2.11 W at 8 m/s). No learning.

## Question and decision

The same two-panel wing, the hand now on the package's passive pitch hinge
(`Part.pitch_compliance`: about the span through the leading edge, sprung to
zero, load at the quarter chord, so it feathers under its own load), no
wrist motor. Does some compliance reach the prescribed wrist's power?

Decision. Within 25% of the prescribed wrist's least power at both speeds:
the hand is a compliant surface and the wrist needs no actuator; "membrane
or stiff" for the outer wing is answered by that compliance. Otherwise the
wrist stays driven in stage 1 and the compliance question returns with the
real drive.

## Method

`rig.trim`, wing 0.25 m, `hand_fraction` 0.5, `hand_compliance` 0.5, 1, 2,
4 (the package's range is 0.05 to 4; 1 is 0.1 rad of deflection under the
hand's 1 g share). Airspeed 6, 8 m/s; frequency 5, 6, 8 Hz; stroke 60, 80,
100 deg; inner feather amplitude 0, 10, 20 deg. Flies level as before.

## Predictions (written before running)

1. At each speed some compliance flies level at a least power within 25% of
   the prescribed wrist's (3.43 W at 6 m/s, 2.11 W at 8 m/s). I hold this at
   0.5.
2. The best compliance is 1 or more at both speeds.
3. Every compliance has at least one point that flies level at 8 m/s.
