# L1 pass 1e: a recovery half-stroke in a low-drag shape. Frozen 2026-11-15

The user: a swimmer does not keep the arm straight; after the power stroke
the wing should return in its least-drag shape, and perhaps in flight too.
Written before the grid below was run. **Disclosed:** six probe points of
this rig change were seen first (wing at 0.4 of its length, 0.3 m/s, stroke
85 deg, 0.5 Hz, feather 0 and 30 deg); they are quoted in section 4 and the
predictions were written knowing them.

## 0 Question and decision

On the held rig in water, does reducing the wing's chord (and area) to 0.4
during one half of each beat, and restoring it for the other, give more net
forward force within the torque caps than the same wing held at full chord,
or held narrow throughout? Decision: whether the fold of passes 1c-1d needs a
per-beat component (passive flaps or fringes that open on one half), or a
fold set once for water is enough.

## 1 Rivals

(a) The stroke here is lift-based (an up-down beat with feathering: both
halves make thrust), so idling one half costs thrust and gains nothing.
(b) The half that is idled costs most of the torque and gives little thrust,
so idling it raises force per unit torque, as for a rowing stroke. (c) A
model artefact: the rig changes the fluid strips' chord in time; the solver's
added mass follows by backward difference, which releases work (one probe
point read -1.23 W, the fluid doing work on the wing), so a gain may be
energy the model creates.

## 2-3 Gates and predictions

Wings (span fraction, full chord): 0.4 and 0.6. Arms: fixed; chord 0.4 on
the half with positive stroke rate; on the negative half; throughout. Gate as
pass 1c: net force at least the drag at f = 0 (thrust twice the drag) with
shoulder torque <= 15.8 N m and crank-equivalent <= 12 N m.

- P1: for each wing, the better per-half arm's best force under the cap
  exceeds the fixed arm's best by more than 20% (rival b over a).
- P2: for each wing, the better per-half arm beats the narrow-throughout arm
  under the cap by more than 20%.
- P3 (artefact check): mean power is negative at fewer than 10% of the
  per-half points; if more, every per-half gain is reported as unproven.
- Falsified: P1 fails for both wings (rival a).

## 4 Measurement

`l1e_recovery.py` over `rig.run` with its new `chord_law` argument: chord and
area of every wing strip scaled by 0.4 on one half, blended over the reversal
(tanh of cos wt over 0.3; a step change kicked the torque to 31-37 N m in the
first probe). It changes the fluid strips only, not the wing's mass or shape:
the narrow-throughout arm read 1.81 N and 2.9 N m where a wing built 0.4 as
wide read 1.32 N and 2.3 N m at the same point. So arms are compared with
each other, not with passes 1c-1d. With the law off the rig reproduces pass
1c exactly (5.92 N, 11.3 N m). Probe points (0.4 length, 85 deg, 0.5 Hz):
fixed 5.92 N, 11.3 N m; narrow throughout 1.81 N, 2.9; negative half narrow,
feather 30: 3.86 N, 4.9 N m, 1.95 W; same at feather 0: -2.44 N, 8.6 N m,
-1.23 W.

A fore-aft (rowing) stroke is not tested: pitching the held body 45 deg in a
0.3 m/s flow gave -12.6 N from its own drag at rest, which would swamp the
comparison.

## 5 Design

V 0.3 m/s; stroke 40, 85 deg; frequency 0, 0.25, 0.5, 1.0, 1.5 Hz; feather
0, 10, 20, 30 deg; 2 wings x 4 arms; one pass. Predicted 6 minutes.
