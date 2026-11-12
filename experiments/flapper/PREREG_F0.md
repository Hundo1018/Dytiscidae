# F0 — what can a flapping wing lift in this simulator, and at what size?

Frozen 2026-11-12 before any rollout, commit 27b51dd, MuJoCo 3.15. No
learning. First measurement toward the goal the user set the same day
(`experiments/triphibian_diagnosis/DIAGNOSIS.md` §7c): one flapping wing that
flies, swims and carries the body on land, designed directly; scale set by
the physics; actuation in two stages, this one with the wing's motion
prescribed and no torque limit.

## 0 Question and decision

On a fixed rig, body upright so that a stroke about the body's long axis
sweeps a horizontal plane ("normal hovering"), what mean vertical force does
one pair of wings make, across wing length, frequency, stroke amplitude and
feather amplitude? The decision is the design envelope: the wing length,
frequency and all-up mass the next steps (free hover, then swimming and
wing-walking with the same wings) are built at. If no point of the map lifts
a plausible all-up mass, the simulator's fluid model or the two-degree
shoulder is the first thing to change, before any design.

## 1 Rivals

- **It lifts**, about as quasi-steady theory says.
- **The solver under-reads or over-reads flapping**: its induced inflow or
  unsteady terms are tuned on wings in forward flight (the RoboFly fixture
  was matched to CL rms 0.16); checked against a hand blade-element estimate.
- **Time step**: at 15-25 Hz a step of 1 ms is 40-67 steps per beat; the
  force may depend on it. Checked by halving the step.
- **Instrument**: the summed `xfrc_applied` is not the whole fluid force (an
  implicit damping split exists), or the rig's rewriting of the joints
  injects force. Checked by the inert arms.

## 2 Method

- Body `experiments/flapper/body.py::flapper`: a hull and one pair of
  membrane wings, aspect ratio 3 per wing (chord = length / 3), on universal
  shoulders. `phenotype.AVIONICS_MASS` (0.30 kg, sized for the 15 kg
  reference) is set to 0.005 kg inside this experiment only; the force does
  not depend on it, the weight does, and the weight is reported apart.
- Rig, as `experiments/flight_audit/fixed_rig.py` with `kinematic=True` and
  no airspeed: the root is held upright and still every step; the four
  joints are written every step from
  `stroke = (Phi/2) sin(2 pi f t)`, `feather = s Psi cos(2 pi f t)`, both
  wings alike. Time step `min(1 ms, 1/(60 f))`. Eight beats, the last five
  averaged. Still air, unsteady and inflow terms on as the search runs them.
- Read: mean world-vertical force in newtons (and in grams-force), mean
  horizontal force, and a hand estimate: blade elements at the prescribed
  kinematics with Dickinson's translational coefficients
  `CL = 0.225 + 1.58 sin(2.13 a - 7.2 deg)`, angle of attack
  `a = 90 deg - |feather|`, no inflow, no rotational term.
- Map: wing length 0.10, 0.15, 0.20, 0.25, 0.30 m; frequency 5, 8, 12, 16,
  20, 25 Hz; stroke (peak to peak) 100, 130, 160 deg; feather amplitude 35,
  45, 55 deg. The feather sign `s` is found once at one point and kept.

## 3 Predictions (written before running)

1. One feather sign lifts and the other pushes down by a like amount
   (within 25%).
2. Force goes as frequency squared and wing length to the fourth at fixed
   aspect ratio: log-log slopes 1.7-2.1 and 3.5-4.2.
3. Solver over hand estimate: between 0.5 and 1.3 everywhere on the map.
4. There is a point with wing length <= 0.25 m and frequency <= 16 Hz that
   lifts 100 gf, and one with wing length <= 0.15 m and frequency <= 25 Hz
   that lifts 30 gf.
5. Of the three feather amplitudes, 45 deg lifts most at most map points.
6. Inert arms: wings held still read |force| under 1e-6 N; flapping with
   feather 0 (the plate broadside) reads a vertical force under 5% of the
   lifting case at the same point.
7. Halving the time step moves the force by under 5% at 25 Hz.

## 4 Limits fixed now

A fixed rig: no body motion, so nothing about stability or control. Rigid
wings: no passive twist yet. One planform. Prescribed motion: the torque and
power it asks for are recorded where the rig can read them and are not a
gate here (stage 2).
