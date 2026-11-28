# L1 pass 1l: the cam's feather amplitudes in water. Frozen 2026-11-15

Pass 1i's cam ties the feather to the stroke. If the cam stays engaged in
water, the water stroke has the feather the cam gives. Passes 1b-1e
prescribed the water feather freely and charged the drive for the stroke
only. Here the water stroke uses the amplitudes the air needs, and the crank
pays for the feather too. Written before the grid below was run.

## Two things found first, both disclosed

- **Pass 1i's two gates use different cams.** Its 16 take-off gate points
  all have a feather amplitude of 50 or 60 deg; its 26 cruise gate points
  all have 20 or 30 deg. At 40 deg the take-off tops out at 1.47 weights
  under the caps and no cruise point sustains; at 30 deg the take-off tops
  out at 0.92. One fixed cam does not meet both. The drive needs a feather
  amplitude that can be changed between take-off and cruise (a second slow
  adjustment beside the offset), and `RESULT_L1i_cam.md` is corrected to say
  so.
- **The torque-driven rig is not usable in water above a feather of about
  30 deg** (`probe_water.py`).
  - At 30 deg it agrees with the held rig: +5.53 against +5.92 N, 7.30
    against 7.10 W.
  - At 45 and 60 deg it reads -177 to -607 N and torques in the hundreds of
    N m, growing as the step shrinks.
  - The cause is the fluid solver's entrainment reaction (`-dm/dt v` on the
    wing's added-mass tensor): with it off the same point reads +10.8 N and
    7.8 W against the held rig's +9.9 N and 8.2 W. A wing driven by torque
    has small velocity ripples; the term differentiates the added mass in
    time and feeds them back. The held rig overwrites velocities, which
    cuts the loop.
  - This is a fault in the solver for stiffly driven wings in water, not
    repaired here. So this pass runs on the held rig.
- A consequence for the flap's power check (pass 1j, R5): on the
  torque-driven rig one flap point with a feather of 30 deg reads +9.25 W
  where the held rig reads -3.90 W. That supports the suspected accounting
  fault, but that point's torque (14.1 N m against 8.4) is not trusted until
  the solver fault is fixed. The check stays open.
- Changing the torque-driven rig's servo to the model's own actuator left
  air forces and power unchanged and moved the crank metric by up to 10%
  (pass 1k's probe point: 10.1 to 9.0 N m held). Pass 1k's crank figures
  carry that uncertainty.

## 0 Question and decision

With the feather amplitudes the air needs (20-30 deg for cruise, 50-60 deg
for take-off) and the crank paying for the feather, does the folded wing
still meet pass 1c's water gate? Decision: whether the cam can stay engaged
in water, and at which amplitude.

## 1 Rivals

(a) The feather costs little in water: the gate is met as in pass 1c.
(b) The feather's hinge moment in water, through the same crank, takes the
torque the stroke needed, and fewer or no points meet the gate.
(c) The amplitude decides: 20-30 deg gives too little thrust, 50-60 deg
gives thrust and costs torque.

## 2-3 Gate and predictions

Wings folded to 0.4 and 0.6 of the length, 0.3 m/s, stroke 40 and 85 deg,
0.25, 0.5, 0.75, 1.0 Hz, feather 20, 30, 50, 60 deg; three models: no
centre-of-pressure law with the hinge on the leading edge (pass 1c's basis),
and "plate" and "insect" with the hinge at 0.1 chord. Gate as pass 1c: net
force at least the drag at rest, fluid stroke torque <= 15.8 N m, and the
crank metric with the feather included <= 12 N m.

- P1: the gate is met at some point for each wing under every model.
- P2 (rival c): at 20 deg no point meets the gate for either wing under any
  model; at 50 or 60 deg some point does.
- P3 (rival b): at the best gate point of each wing and law, the crank
  metric with the feather exceeds the stroke-only metric by 10-80%.
- P4: mean power is negative at no point (no free part here).

## 4-5 Measurement and design

`l1l_camwater.py` on `rig.run`, read by `l1l_read.py`. 192 points and 2 at
rest; 3 beats, the first dropped. One pass. Predicted 10 minutes.

Not covered, and it matters: the wing's added-mass inertia is in the mass
matrix, not in the fluid loads the held rig reads, so the torque here, as in
passes 1b-1e, leaves out the reactive torque of accelerating the water with
the wing. A rough figure for the 0.4 wing at 0.5 Hz and 85 deg is 7 N m in
quadrature with 11 N m of fluid torque. A free body; the fold's own
mechanism; the flap.
