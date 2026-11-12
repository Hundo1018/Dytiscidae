# F0b — hover, slow flight, forward flight and glide of the same wing

Frozen 2026-11-12 before any rollout, commit 963cf74. No learning. Asked by
the user's answer to F0: two wing pairs, membrane or feathers are all open
and are to be decided by what works best, not by taste; forward flight like a
bird looks better to them, and hovering is to be kept only if it earns its
place in handling, gliding or effort.

## 0 Question and decision

For the F0 wing on a rig held at an airspeed V and a body pitch, what is the
least aerodynamic power at which the weight is carried and the speed held, at
V = 0 (hover), 3, 6 and 9 m/s; and how does the same wing glide with the
stroke stopped?

Decision, fixed now. If level flight at 6 or 9 m/s takes at most half the
hover power and the stopped wing glides at a lift-to-drag ratio of 3 or
more: the machine is flown like a bird (forward flight and glide are its way
of flying), and static thrust above weight is kept only as the manoeuvre
that leaves the ground and the water, which F0 showed this wing has. If
forward flight is not found on the map: the design is a hoverer and the
wing is re-sized for it. Wing count and wing material are not decided here;
they are decided by controllability and by what stands on land (the next
measurement), with criteria written before it.

## 1 Rivals

- **A power curve with a minimum** at a few metres per second, as for every
  flying animal and helicopter.
- **Flat or inverted**: this wing is all drag at speed (aspect ratio 3, a
  hull broadside when pitched), so forward flight costs as much as hover.
- **Instrument**: the power is read from the fluid force on each wing about
  its two shoulder axes times the prescribed rates; if the solver puts part
  of the force elsewhere the power is wrong. Checked against F0's hover
  force (same rig, same point) and against a hand estimate of hover power.

## 2 Method

- Body and rig as F0 (`hover_map.py`), wing length 0.20 and 0.25 m, chord
  one third, avionics 5 g inside the experiment. The root is carried at
  speed V along world +x at body pitch theta (nose up), both rewritten every
  step. Joints prescribed: `stroke = (Phi/2) sin(wt)`,
  `feather = psi0 - Psi cos(wt)` (the sign F0 found for the oscillating
  part; `psi0` both signs). Five beats, the last three averaged; time step
  `min(1 ms, 1/(50 f))`.
- Grid: V 0, 3, 6, 9 m/s; theta 0, 20, 45, 70, 90 deg; f 5, 8, 12 Hz;
  Phi 80, 130 deg; Psi 20, 45 deg; psi0 -15, -5, 5, 15 deg.
- Read per point: mean vertical force over weight; mean force along the
  flight path; mean aerodynamic power of the two wings (minus the fluid
  torque about each shoulder axis times that axis's rate). **Flies** means
  vertical force at least the weight and force along the path at least 0.
  Per V: the flying point of least power.
- Glide: stroke and feather oscillation off, theta 0, wing incidence psi0 0
  to 20 deg in steps of 4 (both signs), V 4 to 12 m/s: lift over drag, and
  the sink rate at the speed where lift equals weight.

## 3 Predictions (written before running)

1. Level flight is found at 6 and at 9 m/s on both wings at 8 Hz or less.
2. The least power at 6 or 9 m/s is at most half the least hover power for
   the same wing.
3. At 0 m/s only theta = 70 or 90 deg flies; at 3 m/s the least-power point
   has theta of 45 deg or more.
4. Glide: best lift over drag between 3 and 7; least sink under 2 m/s on the
   0.25 m wing.
5. The hover point of F0 (0.20 m, 12 Hz, 130 deg, 45 deg) reads the same
   vertical force here within 2% (the rig did not change), and its power is
   within a factor 0.5-2 of `weight x sqrt(weight / (2 rho A))` over a figure
   of merit of 0.5, with A the swept disc area.

## 4 Limits fixed now

A held body again: nothing about stability, trim moments or control.
Quasi-steady model. Rigid wings. Aerodynamic power only: the inertial torque
of reversing the wing is reported from the kinematics, not charged.
