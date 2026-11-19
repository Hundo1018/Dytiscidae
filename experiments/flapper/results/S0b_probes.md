# S0b probes B0-B4 (DESIGN_S0b_bold.md)

2026-11-19. Probes for designing a pre-registration: not scored, no gate read.
Every rollout is a row of `results/S0b_probes.jsonl` (probe, id). Files:
`s0b_machine.py` (T4), `s0b_laws.py` (the stroke family), `s0b_probe.py`
(the rollouts), `s0b_draw.py` (the figure). `mission.py`, `m3b_sweep.py`,
`s0_machine.py` are not edited or rebound.

## B0, build

T4 builds on `mission.Machine`. Read from the built model
(`figures/S0b_T4_built.json`, `results/S0b_calib.json`):

| | |
|---|---|
| hinges | 14: stroke, feather, sweep on each of 4 shoulders (12), 2 tail |
| motors | ideal on all 14 (tau_s 1e4 N m, omega_0 1e7 rad/s; rotor and gear as `mission.MOTORS`) |
| mass before ballast | 0.9176 kg (the built mass; no ballast at the hull's centre) |
| buoyancy, fully submerged | 11.366 N (solver, static, level at 5 m; `diag.buoyancy` agrees) |
| buoyancy ratio before ballast | 1.263 |
| ballast | 0.2414 kg, one point mass in the hull body (no contact, no volume) at hull x 0.317 m, z -0.032 m (0.8 hull radius below the axis; inside the hull's lower surface, checked on the figure) |
| mass after | 1.1590 kg; buoyancy ratio 1.0000 |
| CoM, hull frame | (0.2334, 0.0000, -0.0067) m; centre of buoyancy x 0.2334 m (the ballast station is solved for this) |
| inertia about the CoM, hull axes | Ixx 0.0296, Iyy 0.0235, Izz 0.0522 kg m2 (built; S0's A0 overrode the hull's transverse inertia to 0.012, this does not) |
| wings | fore pair shoulders at x 0.325 m (C1's), wing from x 0.132 to 0.389; hind pair shoulders at x 0.055 (station 0.11), wing from x -0.138 to 0.119; y 0.04 to 0.70 each side |
| QACC | 0 |

Hind station 0.11: the wing's chord runs 0.065 m ahead of the shoulder to
0.193 m behind it, the fore wing's trailing edge is at x 0.132, so a hind
shoulder aft of x 0.067 is the only place in the hull where the two wings
share no hull station (they stroke about the same axis). 12 mm gap.

Differences from the design note:
- no mass is added for the sweep drives (M3b typed 30 g per hinge). The
  all-up mass is fixed by b = 1.00, so this only moves 0.12 kg between the
  ballast point and the shoulders.
- the sweep hinge is M3b's on all four shoulders (range +-35 deg, axis the
  hull's z, placed before the stroke); it built on the hind pair without change.
- the strip-twist stand-in: `mission.enable_strip_twist` knows seg1/seg2 only;
  `s0b_machine.enable_twist4` extends the side map to four wings. Not used in
  B1-B4 (V+ does not use it).
- variable ballast is not exercised in any probe.

Figure: `figures/S0b_T4_built.png` (side and top, hull frame, from the built
model). Opened and checked against the model's numbers: four wings at the
stations above, stroke axes along hull x (left +x, right -x), feather axes
along y at x 0.299 / 0.029, sweep axes along z at the shoulders, ballast
square at (0.317, -0.032) inside the hull outline, CoM + on the dotted
centre-of-buoyancy line, 6.7 mm below the axis.

Not on the figure: with sweep, a hind wing swept aft reaches the tail's
station (tail at x -0.39..-0.27, |y| < 0.25). Contact counts are after B1.

## B1, hover (air, hull held nose up; K1 kinematics: 85 deg stroke, 50 deg feather, hind pair in opposition)

Held-body map, mean over 4 beats after 2, hull axes (x = nose = up, y left,
z dorsal), moments about the CoM. Weight 11.366 N. QACC 0 on all 21 rows, no
contact between parts on any. Seven rows were lost when another agent killed
two pool workers and were re-run (`b1rest`).

| row | Fx (thrust) N | Fy | Fz | Mx | My | Mz | shaft W (positive work) |
|---|---|---|---|---|---|---|---|
| base 4.5 Hz, amp 1 | 20.94 | 0.00 | 0.01 | 0.00 | 0.133 | 0.00 | 399 |
| hind pair in phase | 20.86 | 0.00 | -0.05 | 0.00 | 0.116 | 0.00 | 402 |
| still (B4) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 |
| 2.5 / 3.5 / 5.5 / 6.0 Hz | 6.43 / 12.63 / 31.36 / 37.36 | | | | 0.04 / 0.08 / 0.20 / 0.23 | | 76 / 195 / 710 / 914 |
| amp 1.15 | 26.36 | | | | 0.163 | | 510 |
| amp_lr 0.2 (left larger) | 21.13 | 1.20 | | -0.01 | 0.134 | **-2.07** | 405 |
| amp_fh 0.2 (fore larger) | 21.21 | 0.00 | | 0.00 | 0.135 | 0.00 | 409 |
| bias 0.15 rad | 20.93 | 0.00 | 1.44 | 0.00 | **1.744** | 0.00 | 400 |
| bias_lr 0.15 | 20.94 | 0.00 | | 0.04 | 0.133 | 0.00 | 399 |
| bias_fh 0.15 | 20.93 | 0.00 | | 0.00 | 0.004 | 0.00 | 399 |
| psi0 0.15 (feather offset, all) | 20.17 | 0.00 | -4.72 | 0.00 | -1.143 | 0.00 | 400 |
| psi0_lr 0.15 | 20.70 | 0.00 | | **-2.16** | 0.122 | 0.00 | 399 |
| psi0_fh 0.15 | 20.69 | 0.00 | | 0.00 | 0.646 | 0.00 | 400 |
| Psi_lr 0.15 | 20.80 | 1.31 | | -0.01 | 0.124 | -0.10 | 401 |
| Psi_fh 0.15 | 20.88 | 0.00 | | 0.00 | 0.122 | 0.00 | 397 |
| sweep, all four forward 0.2 rad | 21.68 | 0.00 | | 0.00 | 0.139 | 0.00 | 400 |
| sweep_lr 0.2 (left forward, right aft) | 20.83 | **-2.09** | | 0.01 | 0.133 | 0.06 | 399 |
| sweep_fh 0.2 | 20.83 | 0.00 | | 0.00 | 0.137 | 0.00 | 390 |

Read:
- thrust 1.84 weights at K1 amp 1; weight is carried at amp 0.74 (4.5 Hz) or
  at about 3.3 Hz amp 1. Peak hinge torque 11.8 N m stroke, 6.1 feather, 7.9
  sweep (holding against the stroke reaction): all under the witness stall
  (24 / 16 N m), so the ideal motor is not what makes this thrust.
- three-axis moments exist: My from the common stroke bias (11.6 N m/rad),
  Mz from the left-right amplitude difference (-10.3 per unit), Mx (about the
  thrust axis) from the left-right feather offset (-14.4 N m/rad). Against
  inertias of 0.024-0.052 kg m2 that is 40-70 rad/s2 at the inputs above.
- every fore-hind difference (amplitude, bias, feather amplitude, sweep) gives
  no moment: with the thrust along the hull axis the two pairs sit on the
  same line. Only the fore-hind feather offset gives My (0.51 N m), through
  opposite side forces at the two stations. The second pair doubles thrust;
  it adds no moment axis in this mode.
- the sweep offset turns the thrust sideways (left-right differential: 2.1 N
  of side force, about 0.1 of thrust at 0.2 rad, no moment): a translation
  control, not an attitude one. The common feather offset does the same along
  z (4.7 N) with a pitch moment.
- the base My 0.133 N m is the ballast: CoM 6.7 mm off the thrust axis x 21 N.
- phase opposition of the hind pair changes thrust by 0.4%.

One free attempt, untuned (`free_hover_ideal`, `free_hover_x4`): released at
rest nose up at 10.2 m over the sea, `s0b_laws.Fly4` = M1's Fly structure
(beat-mean state, PD on the error rotation, PI on climb rate into amplitude)
with the three outputs mixed through the inverse of the map above, kp, kd from
the built inertia at wn 6 rad/s, zeta 0.7 (first guesses, not tuned), no
velocity loop, targets blended in over 0.25 s.
- height 9.92-10.40 m for the whole 8 s on amplitude 0.72-0.86 (it flies its weight);
- nose within 25 deg of vertical to 6 s (nose_z >= 0.89), while it drifts in x
  at a growing 0.2 -> 1.6 m/s (nothing in this law opposes drift);
- from 6 s a lateral (y) divergence: nose_z 0.83 at 6.4 s, 0.48 at 7.6 s,
  0.19 at 8 s, speed 8.4 m/s. This looks like M1's lateral mode; untuned.
- peak torque 16.6 N m stroke, 6.2 feather, 7.2 sweep; no contact; net shaft
  work 1506 J (188 W mean); law outputs on their limit 7.8% of updates;
  QACC 0. The x4 bounded motor row is identical to the last digit (no hinge
  reached its curve: at_limit 0), so this row does not depend on the ideal motor.
- not done: a start from the ground (the take-off strike is where the lead
  reports the ideal motor makes 2.4 kN m; not attempted here).

Contact between parts: 0 on every held row (sweep offsets up to 0.2 rad) and
0 on every free row. The full +-35 deg sweep of the hind pair against the tail
was not exercised.

## After B1: the lead's note on the ideal motor (applied from here on)

Free rollouts record peak hinge torque, peak contact (weights), net shaft
energy and steps on the motor curve; every phase change is blended over 0.25 s
from the last target (no stepped target); each free rollout is run on the
ideal motor and on a bounded one ("x4": mission's witness curve x4 in torque
and speed: 96 / 64 / 20 N m stall at the stroke / feather / tail hinge, sweep
as stroke). `artifact` = a peak hinge torque above 10x the witness stall
(240 / 160 / 50 N m). Rows also count `clamped_steps`, the steps on which the
solver's own fluid-force limiter engaged: a held row with any is not a force
measurement.

## B2, water thrust (hull held level at 3 m; both entrainment modes)

Drag, wings and tail held at 0, hull moved along its axis (measured, not typed;
default and wetting agree to 0.03 N):

| speed | axial drag | dorsal force above buoyancy | My |
|---|---|---|---|
| 0.3 m/s | 0.94 N | +0.64 N | 0.06 N m |
| 1.0 m/s | 16.4 N | +77.1 N | 6.6 N m |
| 2.0 m/s | 66.1 N | +262 N | 21.1 N m |

The dorsal force is the four cambered wings at zero feather in an axial flow:
at 1 m/s it is 6.8 weights. A machine moving along its axis in water has to
feather to zero lift, or it is turned.

K1's kinematics (85 deg stroke) in water, 0.5-6 Hz: the fluid-force limiter is
engaged on every row in both modes (6,626-37,405 of the steps), thrust reads
162-219 N whatever the frequency, peak torques 155-4,080 N m (stroke) and
251-3,173 N m (feather) under the default. These 14 rows and the 14 moment rows
taken at K1, 1 Hz say only that the limiter engaged. Not evidence of thrust.

A 30 deg stroke (feather 50 deg, hind pair in opposition):

| f Hz | thrust N, default | wetting | clamped steps | peak torque stroke / feather N m, default (wetting) | shaft W + |
|---|---|---|---|---|---|
| 0.5 | **63.4** | **60.9** | 0 | 13.4 / 19.6 (11.2 / 13.3) | 64 (50) |
| 1.0 | 180 | 170 | 19,513 | 50 / 70 (32 / 43) | 442 (324) |
| 2.0 | 207 | 167 | 17,207 | 134 / 225 | 2,093 |
| 3.0-6.0 | 258-525 | 167-168 | 6,587-13,152 | 370-1,793 / 492-1,904 | 6.6-56 kW |

Only 0.5 Hz is free of the limiter, and there the modes agree within 4%:
thrust 5.4-5.6 weights, 65-67x the drag at 0.3 m/s, on torques at about the
witness stall (the x4 motor row is identical). Moments at that point
(difference from its base row; default / wetting):

| input | moment |
|---|---|
| amp_lr 0.2 | Mz -3.33 / -3.23 N m |
| bias 0.15 rad | My +4.27 / +4.02 N m |
| psi0_lr 0.15 rad | Mx -3.71 / -3.62 N m |

Same three effectors and the same signs as in air. So, held: axial thrust far
above drag and moments on three axes, in both modes, at 0.5 Hz and 30 deg.
"B2's best frequency" for B3 is therefore 0.5 Hz at 30 deg: the only measured
point that is a force measurement.

B4, still, held: force (0, 0, 11.366) N = buoyancy = weight, moment 0, both modes.

## B3, breach (free, b = 1.00, nose up, CoM at 3 m, at rest; 30 deg stroke at 0.5 Hz, open loop)

**The held thrust does not reach the free body.** With the hind pair in
opposition (B2's stroke), free:

| mode, motor | CoM rise in 7 s | max speed | peak torque stroke / feather N m | clamped | artifact |
|---|---|---|---|---|---|
| default, ideal | 0.41 m | 0.58 m/s | 119 / 404 | 0 | yes |
| default, x4 | 0.41 m | 0.58 | 96 (on its curve) / 65 | 0 | no |
| wetting, ideal | 0.74 m | 0.62 | 107 / 39 | 0 | no |
| wetting, x4 | 0.74 m | 0.62 | 96 / 38 | 0 | no |

The trace shows why: the hull swings (nose_z 1.00 -> 0.90 and back each beat,
1.3 rad/s, CoM +-0.1 m in x) while the wings barely move through the water.
The fore pair pushes the hull one way and the hind pair the other: a pitch
couple on a 1.16 kg, 0.024 kg m2 hull, against wings whose water added mass is
tens of kg each. Symmetric strokes recoil the hull instead of stroking the
water. (In phase, the same thing as heave.) No surface reached, either mode.

**The diagonal stroke** (`lr_phase = pi` with the hind pair in opposition: FL
and HR together, FR and HL half a beat later) cancels the wings' reactions on
the hull in force, roll and pitch. Held, it reads the same (water 63.4 N,
air 20.94 N). Free, wings stopped (blended to 0 over 0.25 s) when the CoM
reaches the surface:

| mode, motor | result | speed at the surface | peak CoM height | peak torque stroke / feather | clamped steps | artifact |
|---|---|---|---|---|---|---|
| wetting, x4 | surface at 4.0 s | 1.55 m/s (nose_z 0.986) | **+0.17 m** | 96 (curve) / 65 | 12,280 | no |
| wetting, ideal | surface at 4.0 s | 1.55 m/s | +0.17 m | 127 / 915 | 12,281 | yes |
| default, x4 | not at the surface in 7 s: CoM -1.55 m | - | - | 96 / 65 | 7,600 | no |
| default, ideal | CoM -2.16 m at 7 s | - | - | 663 / 1,974 | 3,692 | yes |

Mean climb 0.75 m/s (wetting) and 0.21 m/s (default, x4): a 3.6x split between
the modes. The speed is pulsed: 1.4-2.9 m/s at each half-stroke, back to about
0 (-0.1 m/s) between them. One wing length is 0.66 m; the CoM peaks 0.17 m over
the surface with the hind pair never out of the water. It does not clear, in
either mode. The limiter engaged on these free rows (the pulses), so their
speeds are lower bounds of what the model would give without it, not clean
numbers.

One attempt switching to the air stroke on clearing (diagonal stroke kept on
through the surface; B1's air law from the moment no main-wing strip is wet;
T 9 s): the switch never happened.
- wetting (x4; ideal the same to 4 s, artifact by feather torque): at the
  surface at 4.0 s, then bobs with the CoM between -0.11 and +0.14 m, 19-40
  of 40 main-wing strips wet, nose falling to nose_z 0.70 by 8.7 s.
- default (x4): CoM -1.28 m at 9 s, still climbing at about 0.2 m/s.

B4, still, free: 6 s, both modes: CoM moved 2 mm, speed < 0.004 m/s, nose_z
1.00 -> 0.992 (the low ballast starts to right it), no breach.

## QACC and what is not evidence

- QACC resets: 0 on all 106 rows (90 held, 16 free).
- Limiter engaged (`clamped_steps` > 0) on 56 rows: every K1 water row, the
  30 deg rows at 1 Hz and above, and all 8 diagonal free rows.
- `artifact` (peak torque > 10x witness stall) on 5 free rows, all on the
  ideal motor. Their x4 twins are bounded and say the same thing.
- B1's free hover and every B1 held row are inside the witness stall.

## What limits each

(a) Hover: nothing found. 1.84 weights at K1, three moment axes of 1.6-2.2 N m
at small inputs, torques under the witness stall. The untuned loop holds height
and loses the lateral axis after 6 s (drift unopposed, no velocity loop).

(b) Water: held, thrust and three-axis moments are there in both modes. Free,
the limit is reaction: the hull is too light to stroke four wings against
water unless the wings' reactions cancel among themselves. Only the diagonal
phasing does that, and it needs both pairs.

(c) Breach: not from this stroke, in either mode. Thrust comes in pulses and
the body stops between them; mean 0.2-0.75 m/s where a ballistic clearance of
one wing length needs about 3.6 m/s at the surface, and at 2 m/s the measured
drag of the stopped wings is already 66 N with 262 N across the axis. More
stroke runs into the solver's limiter and torques past 10x the witness stall,
so the harness cannot show a faster breach as evidence. The two modes differ
3.6x in climb rate.

Not probed: sweep used as a stroke (beating in another plane); variable
ballast; a ground start; any tuning; a staged exit with the fore pair on the
air stroke while the hind pair is still wet (the law here has one frequency
for all four wings).
