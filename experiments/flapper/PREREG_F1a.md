# F1a — does a wing that twists along its span hold the speed for less?

Frozen 2026-11-12 before the grid, commit de42151 plus `rig.py` (which
reproduces a stored F0c row to the last digit; two smoke points on the
two-panel wing were run to check the rig responds, at one stroke, and are in
no table). No learning.

## 0 Question and decision

F0c: the rigid 0.25 m wing holds 5-9 m/s for 4.0-5.3 W of aerodynamic power
while its glide says the drag to overcome is about 0.4-0.5 W. The suspect is
that a rigid wing feathers by one angle from root to tip while the stroke's
speed grows with radius. If the outer half of the wing may feather by a
different amount (a wrist: an outer panel hinged about the span), does the
least trimmed power fall?

Decision. If twist cuts the least power by 30% or more at both speeds: the
wing is two panels, an arm and a hand, and the next question is whether a
passive hinge can do what the prescribed wrist does (membrane or feather
compliance decided by that number). If not: missing twist is not why
cruise is dear, the wing stays one panel, and the stroke itself (its plane,
a fore-aft sweep) is examined next.

## 1 Rivals

- **Twist is it**.
- **The model's drag at incidence**: the cost is profile drag at the
  incidence needed to carry the weight at these speeds, and no feathering
  schedule removes it.
- **Too slow a stroke on the grid**: thrust needs the tip to move at a speed
  comparable with the flight speed, whatever the twist.
- **Instrument**: two panels differ from one in discretisation and in mass
  (two wrist motors); checked by the T = 0 arm against F0c.

## 2 Method

`rig.trim` on the 0.25 m wing with `hand_fraction = 0.5` (same length, chord
and area as F0c's wing), body level. Airspeed 6 and 8 m/s; frequency 4, 5,
6, 8 Hz; stroke 60, 80, 100 deg; inner feather amplitude 0, 10, 20 deg; twist
amplitude T (the outer panel's extra feathering, same phase) -10, 0, 10, 20,
30 deg. Incidence trimmed so that lift is the weight within 3%. **Flies
level**: trimmed and mean force along the path >= 0. Per speed and per T:
the least aerodynamic power among points that fly level.

## 3 Predictions (written before running)

1. At both speeds the least power with T > 0 is at least 30% under the least
   with T = 0. I hold this at 0.6.
2. At both speeds the least-power point over all T has T >= 10, and T = -10
   is never the best.
3. With T > 0, level flight exists at 5 Hz or less at 8 m/s (the rigid wing
   needed 6 Hz).
4. At T = 0 the two-panel wing's least power is within 15% of the one-panel
   wing's over the grid points the two maps share (F0c), at each speed.
