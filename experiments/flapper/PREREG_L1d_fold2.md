# L1 pass 1d: a fold that narrows the chord as well as the span. Frozen 2026-11-15

Written while pass 1c (span only) is still running and before its results
were read, and before any rollout of these wings. The user proposed an
origami fold (Miura type) drawn by a line; such a sheet contracts along span
and chord together, which 1c does not represent.

## 0 Question and decision

In water, does narrowing the chord with the span give more thrust within the
air drive's torque than shortening the span alone? Decision: whether the fold
carried into the fixture needs to contract the chord (an origami sheet) or
only the span (a sweep or a telescoping spar would do).

## 1 Rivals

(a) Thrust per unit torque is set by wing length alone (force times an arm
that is the length), so the chord changes the frequency at which the cap is
met and not the force available: chord contraction buys nothing but a faster
beat. (b) The torque of pass 1b was largely the chord turning in water and
its added mass, which go as chord cubed or more, so narrowing it frees torque
for thrust.

## 2-3 Gates and predictions

Same fixture, measurement and gates as `PREREG_L1c_fold.md` (thrust at least
twice the drag at f = 0 for the same wing and speed; fixed crank: stroke 85
deg and shoulder <= 12 N m; variable throw: shoulder <= 15.8 N m and
crank-equivalent <= 12 N m). Wings: (span fraction, chord fraction) =
(0.6, 0.6), (0.4, 0.4), (0.4, 0.7).

- Prediction (rival a): at 0.3 m/s the best net force under the 15.8 N m cap
  for (0.4, 0.4) is within 30% of pass 1c's for span 0.4 with full chord, and
  it is reached at a frequency 1.3-2 times higher.
- Prediction: the gates each wing meets are the same ones the span-only wing
  of the same span fraction meets in 1c.
- Falsified by: the (0.4, 0.4) force under the cap exceeding 1c's span-0.4
  figure by more than 30% (rival b: the chord matters, an origami sheet earns
  its place on force), or falling short by more than 30%.

## 4-5 Measurement and design

`l1d_fold2.py` over `rig.run` unchanged; V 0.3, 0.6 m/s; stroke 25, 40, 60,
85 deg; frequency 0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5 Hz; feather amplitude 0,
10, 20, 30 deg; one pass. The f = 0 rows define the drag and are the
do-nothing reading. Predicted run time 12 minutes. Not covered: as 1c, plus
the corrugation and stiffness of a real folded sheet.
