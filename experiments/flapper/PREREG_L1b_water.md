# L1 pass 1b, water: where can the reference machine's wings swim within the torque it has? Frozen 2026-11-15

Follows `RESULT_L1a.md` (c): the three water points taken from the paper rule
failed on torque by 1.6-15 times, and the rule was withdrawn. Written before
any rollout of this grid. The one water datum in hand: stroke 85 deg,
0.15 Hz, feather 30 deg gave 4.3 N forward for a 33 N m peak; not flapping,
the drag at 0.3 m/s is 0.47 N.

## 0 Question and decision

Held level at 0.3 m/s under water, is there a stroke, frequency and feather
amplitude at which the L0 fixture's wings give at least twice the body's drag
(0.94 N) within the shoulder torque the air case already sizes? Decision:
whether rows 2 and 8 of L0 stay open with a fixed crank (K0), need the
variable throw (K1), or need something else (a wing that folds or bends away).

## 1 Rivals

The force and the torque both fall as frequency squared, so a slow enough
stroke passes (then the question is only how slow); or the torque per newton
is set by the wing's size and no frequency passes; or the answer hangs on the
feather amplitude, which the 1a grid set too large.

## 2-3 Gates and predictions

Shoulder torque = the rig's fluid torque about the stroke axes, both wings,
peak. Crank-equivalent torque = shoulder torque x sin(stroke/2) / sin(42.5 deg)
(the lever a variable throw gains, `L0/MECHANISM.md`).

- G-K0 (fixed crank): stroke 85 deg, forward force >= 0.94 N, shoulder torque
  <= 12 N m.
- G-K1 (variable throw): any stroke, forward force >= 0.94 N, shoulder torque
  <= 15.8 N m and crank-equivalent <= 12 N m.

Predictions:
- G-K0 is met, at 0.1 Hz or below only, with a feather amplitude of 10-20 deg.
- G-K1 is met at more points than G-K0, among strokes of 40-60 deg at
  0.1-0.15 Hz.
- The best forward force per unit shoulder torque anywhere on the grid is
  0.1-0.25 N per N m.
- Falsified by: no G-K0 point (a fixed crank cannot swim this wing at any
  tried frequency); or G-K1 adding nothing to G-K0 (the variable throw has no
  water case).

## 4 Measurement

`l1b_water.py` over `rig.run` unchanged, medium water, V = 0.3 m/s, body
level, feather offset 0, three beats with the first discarded. Do-nothing
reading: f = 0 gives -0.47 N (1a). The rig's torque is the solver's fluid
wrench on the wing, which includes its added-mass force; the share of added
mass folded into the mass matrix is not in it. Beside each point the script
prints an upper bound for that share as if the wing moved broadside
(rho pi (c/2)^2 L, about the shoulder), reported and not added.

## 5 Design

Stroke 15, 25, 40, 60, 85 deg; frequency 0.05, 0.1, 0.15, 0.25, 0.4, 0.6 Hz;
feather amplitude 0, 10, 20, 30, 45 deg: 150 points, one pass, no tuning.
Predicted run time 8 minutes on 6 processes. Not covered: free body (the body
will heave against forces of this size), buoyancy and trim, other speeds, a
compliant or folding wing.
