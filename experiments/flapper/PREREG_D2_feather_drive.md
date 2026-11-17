# Pass D2, pre-registration: what turns the root feather

2026-11-17, on `78113ee`. Frozen before any candidate is built. Follows D1
(the stroke by mechanisms); the user: the real drive is not fixed before
several simulated candidates exist. This pass chooses nothing.

## 0. Seen before freezing

`freerig` now records the feather servos' torque and power (no change to
any other output). Baseline, the prescribed stroke and feather with ideal
torque, dt 1e-4 (scratchpad `feather_load.py`):

- cruise (root feather held at -2.94 deg, tip feather by strips): the
  feather hinge holds 0.10 N m mean, 0.23 peak per wing at the hinge 0.1
  chord from the leading edge, and does no work (0.00 W). Holding it is a
  lock, not a drive. **D2 is therefore read at take-off only.** The twist
  that cruise needs is the tip's, made by strips here; a mechanism for it
  needs the hinged wing of W7 and is a later pass.
- take-off (4.5 Hz, stroke 85 deg, feather 50 deg leading the stroke by a
  quarter beat): the feather servos do 23.0 W net and 31.4 W counting only
  positive work, both wings, of the 129 W total; the torque peaks at 2.47
  N m per wing (rms 1.60). With the hinge at the quarter chord: 21.6 W,
  4.27 N m peak, and the lift falls to 1.57 weights.

## 1. Same for every candidate

Take-off as in D1 (at rest, body at 90 deg, 4.5 Hz, stroke 85 deg, feather
hinge 0.1 chord from the leading edge, table law, dt 1e-4, loops at solref
2 dt). The stroke by D1's four-bar at lever 60 mm and gear 24 (D1b: holds
its beat, 0.26 deg from its kinematics, 149 W). The witness motor of D1
for any motor added. Total electrical power = every motor's shaft power
plus its copper loss, plus any ideal servo's net power (the baseline's
feather).

## 2. The candidates

- **F0, reference**: the feather by its ideal servo (D1b's row re-run with
  the new records).
- **FM, a motor on the feather**: a second witness motor on one wing's
  feather hinge through a gear, the other wing by a joint coupling;
  position loop on the same reference as the servo. Gears {12, 16, 24, 32},
  best by D1's rule (least electrical power with the feather amplitude
  within 3 deg and its beat 0.98 or more).
- **FX, the feather from the stroke's crank (one motor for both)**: a
  second arm on each crank's shaft, ahead of it under a horn on the wing's
  root chord, 30 mm ahead of the feather hinge (on the stroke axis at rest,
  so the stroke alone barely moves it); a rod with a ball at each end from
  the arm's pin to the horn. Arm radius 30 mm x sin 50 deg; the arm's angle
  on the shaft set so the feather leads the stroke by a quarter beat on the
  rod-vertical approximation; rod length to put the mean feather at 0. The
  feather it actually makes is measured, not prescribed.
- **FP, passive**: no drive on the feather; a torsion spring at the hinge
  at rest at 0, k in {0.5, 1, 2, 4, 8} N m/rad per wing; the joint's own
  range (+-57 deg) is the stop.

Not built, stated: a cam on the crankshaft (contact), a separate crank
geared to the first with a phase offset (it is FX with a gear in place of
a common shaft, for this question the same constraint), cables.

## 3. Checks and predictions

- G1. In a vacuum, total input (every motor and servo) per beat within 1%
  of the largest motor's peak shaft power, for F0, FM and FX.
- G2. FM tracks the feather within 3 deg at its best gear; its electrical
  power is at least the servo's 23 W plus copper loss; total for the
  machine 175-200 W.
- G3. FX makes a feather of 40-60 deg amplitude leading the stroke by
  70-110 deg; the machine's lift within 10% of F0's; one motor's total
  electrical power within 15% of FM's machine total; its motor on the
  limit for under 10% of the beat at gear 24 (else gears 16 and 32 are run
  and the least power with the beat held is reported).
- G4. FP: at no spring does the passive feather reach 40 deg of amplitude
  with a lead within 30 deg of a quarter beat; its best lift is under 80%
  of F0's. Falsified by a spring that does both.
- G5. The drawing: FX's arm, rod and horn from the built model at four
  crank angles, checked against the dimensions above.

Each row also reports peak loop and rod forces, and the feather's amplitude,
mean and lead as made.
