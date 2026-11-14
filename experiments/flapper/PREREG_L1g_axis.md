# L1 pass 1g: the passive feather with its hinge axis moved aft. Frozen 2026-11-15

Pass 1f put the passive feather hinge on the leading edge and found that the
wing weathervanes: take-off costs 20.7 N m and 164 W, and no cruise point had
net thrust. Here the hinge axis moves aft to h x chord (`rig.run(...,
passive_feather=(k, zeta, com, h))`, the axis moved in the wing's frame,
parallel to the leading edge; the wing's shape and rest pose are unchanged).
Written before the grid below was run.

## Disclosed before freezing

- **Pass 1f's damping was not what it said.** The fluid solver's implicit
  damping rewrites `dof_damping` every step as its own copy, taken when the
  model is built, plus the fluid bound. That wiped the rig's hinge damping, so
  every pass-1f point ran with the feather joint's built damping,
  0.0365 N m s/rad (an effective zeta of about 1.3 at k = 0.3 and about 0.2 at
  k = 10, taking the hinge inertia as about 6e-4 kg m^2), not zeta = 0.3. The
  1f probe line "damping 0.1 to 0.5 of critical changed nothing" is void. The
  rig now writes the hinge damping into the solver's copy as well. This pass's
  h = 0 rows re-measure pass 1f with the damping as stated.
- **The model's load point is fixed.** Every strip force (lift, drag,
  rotational, added mass) acts at the quarter chord, and nothing moves the
  centre of pressure with incidence. On a real flat plate it moves aft towards
  mid-chord at high incidence. Here an axis at 0.25 sees no aerodynamic moment
  from the strip normal force, and an axis behind 0.25 is pushed away from
  zero incidence. An axis between 0.25 and about 0.4 chord could be
  aerodynamically stable on a real wing and is unstable in this model.
- **Probes seen** (`probe_axis.py`, `probe_axis2.py`; 4 Hz, stroke 85 deg):
  - The guess that an aft axis is only a stiffer spring
    (k at h equals k x 0.25/(0.25 - h) at the edge) is wrong. At rest and at
    -60 deg, k = 0.6 at h = 0.2 feathers +-9 deg; its supposed equivalent,
    k = 3 at h = 0, feathers +-28.
  - With the damping as stated, at rest and -60 deg:
    - h = 0, k = 0.6, zeta 0.3: 1.62 weights, 111 W, 18.5 N m fluid stroke
      torque peak.
    - h = 0.2, same spring: 0.77 weights, 142 W.
    - zeta 0 against 0.3 at h = 0.2, k = 0.6: 1.20 against 0.77 weights.
  - At 8 m/s, psi0 -8 deg, before the damping fix, moving the axis aft
    raised lift (0.66 to 0.92 weights at k = 2.4). Net force stayed between
    -0.9 and -1.6 N and power rose from 109 to 169 W.
  - An axis at 0.35 did not diverge at k = 2.4.

## 0 Question and decision

Does any hinge axis position between the leading edge and 0.3 chord let one
stroke drive with a passive feather meet the take-off and cruise gates of
pass 1a? Decision: whether K1 (one drive, passive feather) stays alive on a
moved axis. If not, the feather needs a stroke-linked cam or a second
actuator, or the model needs centre-of-pressure travel before the passive
hinge can be judged at all.

## 1 Rivals

(a) Moving the axis only rescales the moment and changes nothing that
matters. (b) An axis near the load point frees the feather from
weathervaning. The spring and the stroke's inertia then set its phase, and
thrust appears in cruise. (c) The model: with the load point fixed at 0.25,
any result for h near or behind 0.25 says more about the solver than about a
wing. (d) The damping fix moves pass 1f's own answer, so 1f's conclusion
changes at h = 0 without any axis change.

## 2-3 Gates and predictions (gates are 1a's)

(a) From rest, stroke 85 deg, body -45, -60, -75, -90 deg, 3.5, 4.0, 4.5,
5.0 Hz, k 0.3, 0.6, 1.2, 2.4, 5 N m/rad, h 0, 0.1, 0.2, 0.3, zeta 0.3 and
1.0, com 0.25. Gate: vertical force >= 1.6 weights, peak drive torque
<= 12 N m (fluid combined with the analytic stroke inertia, as 1a), aero
power <= 150 W.
- P1: **not met at any h.** Falsified if any point meets it.

(b) Level at 6 and 8 m/s, stroke 40, 60, 85 deg, 2.5, 3.0, 3.5, 4.0 Hz,
k 0.6, 2.4, 10, h 0, 0.1, 0.2, 0.3, zeta 0.3 and 1.0, the spring's rest
angle trimmed so lift equals the weight. Gate: sustains, <= 41 W,
<= 12 N m.
- P2: **not met at any h.** Falsified if any point meets it.
- P3 (rival a against b): the best trimmed net force at any h > 0 exceeds the
  best at h = 0 by less than 0.5 N.
- P4 (rival d): at h = 0 no cruise point sustains, as in pass 1f.

## 4-5 Measurement and design

`l1g_axis.py`: pass 1f's job with h and zeta as factors. 640 take-off
points and 576 cruise points; a cruise point uses 2 to 6 runs to trim. One
pass, no tuning after reading. Predicted run time 45 minutes on 6
processes. Not covered: free body; centre-of-pressure travel (rival c);
spanwise-varying stiffness; the wing's own bending. A point that is not
finite is reported and excluded from every gate.
