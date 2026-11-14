# L1 pass 1c: a wing that sheds length in water. Frozen 2026-11-15

Follows `RESULT_L1b_water.md`: under the shoulder cap the full wing (0.66 m
by 0.257 m a side) holds 0.3 m/s with 0.3-0.5 N to spare and cannot give more.
Written before any rollout of a shortened wing.

## 0 Question and decision

If the wing folds in water so that only the inner k of its length works
(k = 0.6, 0.4, 0.25; same chord), does the machine get thrust of at least
twice its drag within the torque the air drive has, at 0.3 and at 0.6 m/s?
Decision: whether a fold is carried into the fixture as a design variable,
and at what fraction.

## 1 Rivals

(a) Force goes as length cubed and torque as length to the fourth at a given
frequency, so a shorter wing gives more force per unit torque (as 1/k) and
the gates open. (b) The shorter wing must beat faster, where the turning of
a 0.257 m chord and added mass dominate, and nothing is gained. (c) The gain
is real in the model and an artefact of it: at k = 0.25 the working wing is
0.165 m long and 0.257 m wide, an aspect the strip model has no business
with (no tip loss).

## 2-3 Gates and predictions

Fixture: the fold is represented as a shorter wing; the folded part is taken
to trail in the flow and is not modelled, neither its drag nor its mass.
Net force is the rig's, already net of the body's drag. D(k, V) is minus the
net force at f = 0 for that wing and speed, measured in the same run.

- Gate at each V: a point with net force >= D(k, V) (thrust twice the drag)
  and shoulder torque <= 12 N m at stroke 85 deg (fixed crank), or, for the
  variable throw, any stroke with shoulder <= 15.8 N m and crank-equivalent
  <= 12 N m.
- Prediction, 0.3 m/s: met at k = 0.4 and 0.25 (both drives); at k = 0.6 by
  the variable throw only.
- Prediction, 0.6 m/s: met at k = 0.25 by the variable throw only; not at
  0.6 or 0.4.
- Prediction: the best net force under the 15.8 N m cap at 0.3 m/s rises as k
  falls, to 1.0-1.5 N at k = 0.4 (about 2.5 times the full wing's 0.47 N).
- Falsified by: no k meeting the 0.3 m/s gate (rival b); or the force under
  the cap not rising as k falls.

## 4 Measurement

`l1c_fold.py` over `rig.run` unchanged, water, body level, feather offset 0,
three beats with the first discarded. Do-nothing reading: the f = 0 rows,
which also define D. Missing is not zero: a (k, V) with no point under the
cap is reported as none.

## 5 Design

k 0.6, 0.4, 0.25; V 0.3, 0.6 m/s; stroke 25, 40, 60, 85 deg; frequency 0,
0.1, 0.25, 0.5, 1.0, 1.5 Hz; feather amplitude 0, 10, 20, 30 deg. One pass,
no tuning. Predicted run time 12 minutes on 6 processes. Not covered: the
folded part's drag, the fold mechanism and its mass, free body, and rival
(c), which only a bench wing in a tub or a model with tip loss can settle;
results at k = 0.25 are reported with that caveat on them.
