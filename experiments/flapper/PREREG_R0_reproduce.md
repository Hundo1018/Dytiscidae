# R0: does the present pipeline reproduce two machines that flew? Frozen 2026-11-15

Plan draft 3, first week, item 3. Written before any rollout of either body.

## 0 Question

At the frequency, speed and stroke two real ornithopters are reported to fly
at, does this project's strip model (held body, prescribed stroke and feather,
`rig.run`) say they fly, and at what aerodynamic power? Decision: whether the
model may rank designs in L1 as it is, or is corrected first.

## 1 Known, and rivals

- E-Flap (A13 B.1, E.1, verified paper): 0.510 kg, span 1.5 m, wing area
  0.438 m2, cruise 3.5-4 Hz, 2-6 m/s, amplitude "30-50 deg" (whether half or
  full stroke was not read: both readings are run), 89 W electrical as an
  upper bound, 130 W estimated mechanical peak.
- Robo Raven II (A13 B.2, verified article): 0.30-0.38 kg, span 1.33 m, 4 Hz,
  5.6-6.7 m/s, maximum plunge velocity 12.6-14.7 rad/s (read here as the peak
  stroke rate, pi f Phi, so Phi = 57-67 deg full), average power 13.6-17.6 W
  and torque 0.99-1.17 N m, with "per wing or both, electrical or mechanical"
  not established. Wing area was not found: 0.25 m2 is TYPED.
- Rivals: (a) the model is right enough; (b) the model is optimistic (uniform
  inflow, rigid wing, ideal feathering, no body heave on a held rig, no hull
  drag): flies them too easily; (c) pessimistic (no leading-edge-vortex lift
  beyond its stall model): cannot fly them; (d) the bodies built here are not
  the machines (rectangular plates at the same span and area; real planforms,
  camber and twist unknown).

## 2-3 Predictions and what falsifies them

Wings are rectangular at the reported span and area. The body is held level;
for each (V, f, Phi, feather amplitude Psi in 0, 10, 20, 30 deg) the feather
offset is trimmed so the mean lift equals the real weight. A point "sustains"
if it trims within 3% and its mean horizontal force is not negative.

- P1 E-Flap sustains at 4 m/s and 3.5-4 Hz for some Psi at Phi of 60-100 deg.
  Predicted: yes. Falsified by no sustaining point (rival c).
- P2 E-Flap, lowest aerodynamic power among sustaining points at 3.5-4 Hz and
  3-5 m/s. Predicted 15-35 W. **Bar, set before the run: the model reproduces
  the machine if this is 31-67 W (0.35-0.75 of 89 W, the range a motor, two
  gear stages, a linkage and unrecovered inertia can account for); below 31 W
  it is optimistic.** So the lead predicts the model fails this bar on the
  optimistic side.
- P3 null, E-Flap at 2 Hz (half its cruise frequency), 3-5 m/s, any Phi up to
  100 deg: predicted no sustaining point. A sustaining point means the model
  gives thrust too cheaply, whatever P2 says.
- P4 Robo Raven II (0.34 kg) sustains at 6.7 m/s, 4 Hz, Phi 57-67 deg.
  Predicted yes, at 6-14 W. No bar: the reported power is ambiguous; reported
  beside it as information only, with mean power over mean stroke rate
  against the reported torque.
- P5 null, Robo Raven II at 2 Hz: predicted no sustaining point.

## 4 Measurement

`reproduce.py`, which calls `rig.run` unchanged. Lift is compared with the
real machine's weight, not the built model's. Do-nothing reading: at f = 0 the
rig is a fixed wing at the feather offset; it cannot have non-negative
horizontal force with positive lift, so "sustains" cannot be earned by a wing
that does not flap (checked in the run: f = 0 rows are included). Missing is
not zero: an untrimmed point is reported as untrimmed and never as sustaining.

## 5 Design

One pass, no tuning after reading. Grid: E-Flap V 3, 4, 5 m/s; f 0, 2, 3.5,
4 Hz; Phi 40, 60, 80, 100 deg. Raven V 5.6, 6.7; f 0, 2, 4; Phi 57, 67.
Stopping rule: none needed. Verdict: "reproduces" needs P1, P3 and P5 as
predicted and P2 inside its bar; otherwise the direction of the miss is named
and the model is corrected before it ranks anything in L1. Not covered: free
body, torque limit, the wing's real shape and compliance.
