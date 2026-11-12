# F0c — trimmed level flight like a bird, on the same wing

Frozen 2026-11-12 before any rollout, commit after F0b's pre-registration.
No learning. F0b's frozen grid ran first and is reported as it came out
(`RESULT_F0b.md`): its feather offsets (5 and 15 deg) and amplitudes (20 and
45 deg) do not contain a bird's cruise, so its least power in forward flight
(8-11 W) is the cost of hover-like thrashing carried forward, while the same
wing with its stroke stopped glides at lift over drag 8-10, which puts the
drag power to overcome near 0.4-0.6 W. This file asks the question F0b could
not.

## 0 Question and decision

Body level, airspeed V, the wing's mean incidence trimmed so that mean lift
equals the weight: is there a stroke that also holds the speed (mean force
along the path >= 0), and what is its least aerodynamic power? F0b's
decision rule stands, read on this map: forward flight at most half the
hover power, and glide at lift over drag 3 or more, makes the machine a
bird-like flyer that keeps static thrust above weight only to leave the
ground and the water.

## 1 Method

As F0b's rig with theta = 0. For each (wing 0.20 or 0.25 m; V 5, 6, 7, 8,
9 m/s; f 3, 4, 5, 6, 8 Hz; stroke 40, 60, 80, 100 deg peak to peak; feather
amplitude 0, 10, 20, 30 deg): the feather offset is found by the secant
rule from -8 and -12 deg (at most four evaluations, kept within -18 to 0
deg) so that lift is within 3% of the weight; a point that cannot be
trimmed is reported as such. **Flies level** means trimmed and force along
the path >= 0.

## 2 Predictions (written before running)

1. Trimmed level flight exists at every speed from 5 to 9 m/s on the 0.25 m
   wing and at three or more of the five on the 0.20 m wing.
2. The least aerodynamic power over the map is at most 2.0 W on each wing
   (hover: 13.3 and 16.4 W on F0b's grid).
3. That point has frequency <= 6 Hz and stroke <= 80 deg.
4. That point has a feather amplitude of 10 deg or more (a wing that only
   heaves does not hold the speed as cheaply).
5. Its peak inertial torque is at most one fifth of the hover point's (0.30
   and 0.42 N m).
