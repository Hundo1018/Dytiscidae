# L1, pass 1a: the reference machine at scale on the held rig. Frozen 2026-11-15

Plan draft 3; L0 pass 1 (`L0/L0.md`, `L0/MECHANISM.md`); R0. Written before
any rollout of this body.

**What this pass is not.** The plan's L1 is a free body with a torque-speed
limited drive and a stabiliser on modelled sensors. That needs a drive model,
a wing-compliance model and a stabiliser that do not exist yet. Pass 1a asks
the question that decides whether they are worth building for this fixture:
with the body held and the stroke and feather prescribed, do L0's rows 1, 2
and 4 survive the project's own fluid model at the L0 scale? Feathering is
prescribed, which the one-degree-of-freedom drive K1 cannot do (its wing must
twist passively), so every force here is an upper bound for K1.

## 0 Question and decision

Does the L0 fixture (0.60 kg, one pair, span 1.4 m, 0.34 m2, stroke up to
85 deg) (a) lift its weight from rest within the drive's limits, (b) cruise
within the cells' 15-minute power, (c) swim at the strokes and frequencies of
`L0/mechanism.py` within the shoulder cap? Decision: which row the next pass
attacks, and whether the fixture's wing loading changes.

## 1 Known, rivals

L0 took its take-off from one tethered machine (RoboFalcon 2.0: lift
coefficient 1.04, electrical peak 20 times ideal). R0: this model flies E-Flap
where E-Flap flies, carries 1.3-1.6 times its real limit, and reads a quarter
of its electrical power. Rivals for (a): L0's calibration is right and the
model agrees; the model says more power than L0 (L0's "merit" hides in
electrical losses what the model puts in the air); the model says less.

## 2-3 Predictions, bars, what falsifies

Real weight W = 0.60 x 9.80665 = 5.88 N. Inertial torque of the wings is
added analytically: J = m R^2 / 3 per wing with m = 0.187 kg/m2 x 0.17 m2;
peak drive torque = sqrt(aero peak^2 + inertial peak^2), both wings.

(a) From rest, body pitched 45, 60, 75, 90 deg, stroke 85 deg, 3-5.5 Hz,
feather amplitude 30-70 deg. Gate: some point with vertical force >= 1.6 W
(R0's capacity band, worst end), peak drive torque <= 12 N m and mean
aerodynamic power <= 150 W (three 21700 at their continuous maximum, 462 W,
over R0's worst electrical factor of 3).
- Prediction: vertical force reaches 1.0 W at 3.5-4.5 Hz for 130-260 W
  aerodynamic; the gate is **not** met: 1.6 W needs more than 150 W.
- L0's row 4 is supported only if the gate is met. Falsified prediction: the
  gate is met.

(b) Level, 6 and 8 m/s, stroke 40, 60, 85 deg, 2.5-4.5 Hz, feather amplitude
0-30 deg, lift trimmed to W. Gate: a sustaining point with aerodynamic power
<= 41 W (164 W from the three cells over 15 minutes, over R0's worst factor
of 4) and peak drive torque <= 12 N m.
- Prediction: lowest sustaining power 20-40 W; gate met, narrowly.

(c) Water, held at 0.3 m/s, strokes 15, 40, 85 deg at 1.0, 0.38, 0.15 Hz
(`mechanism.py`), feather amplitude 30-60 deg. Gate: mean forward force > 0
and peak shoulder torque <= 15.8 N m (the 1.5 cap).
- Prediction: gate met at all three; forward force 2-10 N; peak torque within
  a factor of 2 of 15.8 N m either way, which means some may break the cap.

## 4 Measurement

`l1a.py` over `rig.run`, to which one output is added (the stroke torque,
summed over both shoulders, mean and peak); nothing else in the rig changes.
Do-nothing readings: f = 0 rows in (a) and (c) must give no vertical force
from rest and a negative forward force in water. A point that does not trim
is reported untrimmed, never as failing or passing a power gate.

## 5 Design

One pass, no tuning after reading. The fixture's wings are rectangular
plates; that is the fixture, not the design. Not covered: free body, the
drive's torque-speed line, passive twist, the tail, planform.
