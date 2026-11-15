# L1 pass 1h: the passive feather hinge with a centre of pressure that moves. Frozen 2026-11-15

Pass 1g could not judge a hinge axis behind the quarter chord. The solver put
every strip load there, so an axis at 0.3 diverged, and its only thrust came
from that. The solver now has an opt-in pitching-moment model,
`FluidSolver.cop_law` (default off, so the search and every stored score are
unchanged):

- **Centre of pressure.** Lift and drag act at a chordwise point measured from
  the edge the flow meets. It stays at 0.25 in attached flow and moves to a
  separated law by the stall weight `w` the lift curve already uses.
- **Pitch damping.** A strip pitching about its quarter chord feels the
  broadside drag of its chordwise elements, using Wang, Goosen & van Keulen
  (JFM 2016) eqs 15-18 with the coefficient of eq 18. It is taken about the
  quarter chord, where the strip's velocity is already evaluated, so the
  pitching motion is not counted twice and the load does not depend on where
  the hinge is.
- Held by `test_physics::test_centre_of_pressure_moves_aft_when_the_flow_separates`
  (9 checks) and two mutations, both caught.

Written before the grid below was run.

## The two separated laws, and what checks them

- **"plate"**, Kirchhoff-Rayleigh free-streamline theory for a flat plate:
  1/2 - 3 cos a / (4 (4 + pi sin a)). It gives 0.415 at 45 deg and 0.5 at
  90 deg.
- **"insect"**, the Han et al. (2015) hawkmoth fit: 0.06 + 0.00501 a[deg]. It
  gives 0.29 at 45 deg.
- **One external point.** Percin & van Oudheusden (2015) swept a plate in
  water at Re 1e4, 45 deg, with the pitch axis at 0.05 chord. Read off Wang
  et al. 2016 fig. 11:
  - measured: lift and drag about 0.105 N each, torque about the axis about
    -2.7e-3 N m;
  - this solver at that point: lift 0.111 N, drag 0.141 N, total force 21%
    high;
  - torque with "plate" about 3.3e-3 N m (+21%); with "insect" about
    2.1e-3 N m (-22%).
- So the two laws bracket the one measurement. That is one angle, read from a
  figure, at Re 1e4; this wing runs at about 1e5. The laws are bounds, not a
  validated curve.
- **Not modelled:** the second coupling term of Wang et al. (a load at the
  three-quarter chord), and the centre of pressure of added mass. The Kramer
  force stays at the quarter chord.

## Disclosed before freezing (`probe_cop.py`)

24 points, k = 1.2, zeta 0.3, h 0, 0.2, 0.3, 0.4, with no centre-of-pressure
law, "plate" and "insect".

At rest, -60 deg, 4 Hz, stroke 85 deg:

| law | h | lift | power | fluid torque | feather |
|---|---|---|---|---|---|
| none | 0.3 | -0.75 weights | | | (diverged) |
| plate | 0 | 1.19 | 96 W | 15.2 N m | |
| plate | 0.2 | 1.50 | 132 W | 19.7 N m | |
| plate | 0.3 | 1.55 | 152 W | 21.5 N m | +-49 deg |
| plate | 0.4 | 1.23 | 161 W | | |
| insect | 0.3 | 1.44 | 148 W | 21.3 N m | |

At 8 m/s, psi0 -8 deg:

| law | h | net force | power |
|---|---|---|---|
| plate | 0.3 | +0.12 N | 113 W |
| plate | 0.2 | -1.36 N | |
| insect | 0.2 | -1.71 N | |
| insect | 0.3 | -4.11 N | |

## 0 Question and decision

With the pitching moment modelled, does any hinge axis from 0 to 0.4 chord
let one stroke drive with a passive feather meet the take-off and cruise
gates of pass 1a? Decision: whether K1 (one drive, passive feather) stays
alive; if not, the feather is driven next (a stroke-linked cam, the next
item the user ordered).

## 1 Rivals

(a) With a moving centre of pressure an aft axis is stable and earns the
lift and thrust a prescribed feather gave. (b) It is stable but costs
torque: the stroke must push a wing that sits at high incidence, and the
torque gate fails. (c) The law decides it: "plate" and "insect" disagree
about the answer, so the model cannot. (d) Nothing changes from pass 1g.

## 2-3 Gates and predictions (gates are 1a's)

(a) From rest, stroke 85 deg, body -45, -60, -75, -90 deg, 3.5, 4.0, 4.5,
5.0 Hz, k 0.3, 0.6, 1.2, 2.4, 5 N m/rad, h 0, 0.1, 0.2, 0.3, 0.4, law
"plate" and "insect", zeta 0.3, com 0.25. Gate: vertical force >= 1.6
weights, peak drive torque <= 12 N m (fluid combined with the analytic stroke
inertia, as 1a), aero power <= 150 W.
- P1: **not met under either law** (rival b). Falsified if any point meets
  it.
- P2 (rival a against d): under each law, the most vertical force at a drive
  torque <= 12 N m is larger at some h in 0.2-0.4 than at h = 0, by more
  than 10%.

(b) Level at 6 and 8 m/s, stroke 40, 60, 85 deg, 2.5, 3.0, 3.5, 4.0 Hz,
k 0.6, 2.4, 10, the same h and laws, the spring's rest angle trimmed so lift
equals the weight. Gate: sustains, <= 41 W, <= 12 N m.
- P3: **not met under either law.** Falsified if any point meets it.
- P4 (rival c): under "plate" at least one point at h >= 0.2 sustains; under
  "insect" none does. Falsified if both laws agree.

## 4-5 Measurement and design

`l1h_cop.py`: pass 1g's job with h and the law as factors, zeta fixed at
0.3. 800 take-off and 720 cruise points; a cruise point uses 2 to 6 runs to
trim. One pass, no tuning after reading. Predicted run time 60 minutes on 6
processes. A point that is not finite is reported and excluded from every
gate. Not covered: free body; spanwise-varying stiffness; the wing's own
bending; Re 1e5 validation of either law.
