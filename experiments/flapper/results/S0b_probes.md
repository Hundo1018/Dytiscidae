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

---

# S0b probes B5-B8: the level hover on the sweep stroke (addendum of DESIGN_S0b_bold.md)

2026-11-19. Probes, not scored; no gate read. Every rollout is a row of
`results/S0b_probes2.jsonl`. Files: `s0b_machine.py` (two builder options added),
`s0b_laws.py` (`targets_level`, `Hover4` appended), `s0b_probe2.py` (rollouts),
`s0b_clear.py` (clearance, kinematic), `s0b_draw2.py` (figures). Primary motor on
every row: the witness curve x4 ("x4"); "ideal" only as a named twin. A row is
"not evidence" if it has a QACC reset, a step on the solver's fluid-force limiter,
or a peak hinge torque over 10x the witness stall (240 / 160 / 50 N m; sweep 240).

## Body changes against T4 as built (all in `s0b_machine.make`, defaults unchanged)

| change | value | why |
|---|---|---|
| sweep hinge range | +-75 deg (was +-35) | the sweep is the stroke: half-amplitude up to 55 deg + trim 6 + control 14 |
| feather hinge range | -150..+60 deg (was +-57.3) | the through-vertical flip: leading edge up 45 -> 90 -> 135 deg |
| dihedral | no build change: a held stroke-hinge angle (range +-80) | |
| stroke planes kept apart | not needed and not built: see clearance | |

Signs read on the built model: stroke + raises both wings; feather + lowers the
leading edge; sweep + moves the left tip aft and the right tip forward.

## The stroke (`s0b_laws.targets_level`)

Sweep `sw0 + A sin p` (tip toward the nose +), stroke hinge held at the dihedral,
pitch of the leading edge above the stroke plane `90 - (90 - Psi) cos p` deg
("vert": the wing passes through vertical, leading edge up, at each reversal and
the leading edge leads on both half-strokes) or `Psi cos p` ("flat": through flat,
trailing edge leading on the aft half-stroke). All four wings in phase unless said.
Held rows: hull held level at 30 m in air, the stroke ramped in over 2 beats
(smoothstep), mean of 3-4 whole beats after 3, hull axes (z up = thrust), moments
about the CoM of the ballast-out machine (0.918 kg, 9.00 N) unless b is given.

## B5, held level-hover map (air)

Thrust in weights of the ballast-out machine (0.918 kg) / ballast-in (1.159 kg).
81 rows, through-vertical flip: 0 QACC, 0 limiter steps, 0 over 10x stall, motor
never on its curve (at_limit 0), sweep tracking error <= 0.9 deg rms.

Mid-stroke pitch 45 deg:

| dihedral | A (half-amp) | 2.5 Hz | 4.5 Hz | 6.0 Hz | peak torque sweep / stroke / feather N m at 2.5, 4.5, 6 Hz | shaft W (+) |
|---|---|---|---|---|---|---|
| 20 | 40 | 0.81 / 0.64 | 2.64 / 2.09 | 4.73 / 3.74 | 3.4/1.9/1.8, 10.6/6.2/5.6, 18.5/11.2/9.8 | 69, 363, 836 |
| 20 | 55 | 1.42 / 1.12 | 4.63 / 3.66 | 8.29 / 6.57 | 5.2/3.1/2.1, 16.3/10.2/6.5, 28.7/18.5/11.4 | 126, 681, 1577 |
| 20 | 70 | 2.21 / 1.75 | 7.24 / 5.73 | 12.97 / 10.27 | 7.4/4.7/2.4, 23.1/15.4/7.7, 40.6/27.9/13.5 | 218, 1185, 2747 |
| 35 | 40 | 0.76 / 0.60 | 2.48 / 1.96 | 4.45 / 3.52 | 3.1/1.8/1.8, 9.7/5.9/5.7, 16.9/10.7/10.1 | 63, 328, 753 |
| 35 | 55 | 1.34 / 1.06 | 4.37 / 3.46 | 7.82 / 6.19 | 4.7/3.0/2.1, 14.8/9.8/6.6, 26.0/17.8/11.7 | 111, 594, 1373 |
| 35 | 70 | 2.10 / 1.66 | 6.85 / 5.42 | 12.25 / 9.70 | 6.6/4.6/2.4, 20.7/15.1/7.8, 36.3/27.3/13.8 | 187, 1012, 2343 |
| 50 | 40 | 0.57 / 0.45 | 1.85 / 1.46 | 3.31 / 2.62 | 2.8/1.5/1.8, 8.6/5.1/5.8, 15.1/9.1/10.2 | 56, 288, 659 |
| 50 | 55 | 0.99 / 0.78 | 3.23 / 2.56 | 5.78 / 4.58 | 4.1/2.6/2.1, 12.9/8.4/6.6, 22.7/15.2/11.7 | 93, 493, 1136 |
| 50 | 70 | 1.56 / 1.23 | 5.07 / 4.01 | 9.05 / 7.17 | 5.6/4.0/2.4, 17.6/13.0/7.7, 30.9/23.6/13.7 | 151, 807, 1867 |

Mid-stroke pitch swept (dihedral 35, A 55, 4.5 Hz): 30 deg 4.42, 45 deg 4.37,
60 deg 3.88 weights (out); sweep torque 12.7 / 14.8 / 17.0 N m; shaft 572 / 594 /
640 W. Thrust is flat from 30 to 45 deg and falls 11% at 60; 30 deg is cheapest.
All 27 pitch-30 and pitch-60 rows are in the jsonl (ids `g_d*_A*_f*_P*`).

Read: thrust goes roughly as (A f)^2 (at equal A f the smaller, faster stroke
gives more: 11.67 N at 26.7 deg, 4.5 Hz against 9.84 N at 40 deg, 3 Hz) and
falls with dihedral (35 deg costs 5-6% against 20 deg, 50 deg 30%). The witness stall (24 N m at the sweep hinge) is first passed
at 6 Hz with A >= 55 (26-41 N m): from there the row needs the x4 motor. Every row
at 2.5 Hz, and every row at 4.5 Hz but A 70 with pitch 60 (24.2-27.9 N m), is
inside the witness stall on all three hinges. The x4 curve
(96 N m) is reached on none. The machine's weight is carried at 2.5 Hz: A about
46 deg (ballast out) or 53 deg (ballast in) at dihedral 35.

Flip modes at the first-look point (dihedral 35, A 55, pitch 45, 4.5 Hz; ids `c_*`):
through-vertical 4.36 weights (out), My +1.59 N m, Fx 0.02 N; through-flat 3.12
weights, My +4.78 N m, Fx +1.50 N. The ideal-motor twin of the through-vertical
row is identical to the last digit. Fore-hind mean-sweep split 30 deg or dihedral
split 20 deg change thrust by 0-4%.

### Moments at a hover point (3 Hz, A 40 deg, dihedral 35, pitch 45; thrust 9.84 N = 1.09 / 0.87 weights)

Differences from the base row; hull axes, about the CoM (ids `m_*`). Peak torque on
every row <= 5.8 N m sweep, 3.4 stroke, 2.8 feather (8.5 feather with the sharpened
flip): a quarter of the witness stall. 0 QACC, 0 limiter steps on all 24.

| input | dF N (x, y, z) | dM N m (roll x, pitch y, yaw z) | per unit |
|---|---|---|---|
| base | (0.02, 0, 9.84) | My **+0.419** (ballast in: +0.622) | thrust centre 4.3 cm (6.3 cm) aft of the CoM |
| amplitude x1.2 | z +3.70 | My +0.143 | thrust ~ amp^1.75 |
| **amp_lr 0.2** (left larger) | y -0.65 | **Mx +0.976** | roll 4.88 N m per unit |
| amp_fh 0.2 (fore larger) | z +0.08 | My -0.253 | pitch -1.27 N m per unit |
| **sw0 +10 deg** (all four mean sweep forward) | x -0.31 | **My -0.716** | pitch -4.10 N m/rad |
| sw_fh 10 deg, sw_lr 10 deg | 0 | <= 0.017 | none |
| psi0 +10 deg (more incidence forward stroke) | x -1.69, z -0.36 | My -0.281 | fore-aft force -9.7 N/rad |
| **psi0_lr 10 deg** | z -0.09 | **Mz +0.662** | yaw 3.79 N m/rad |
| psi0_fh 10 deg | z -0.09 | 0.003 | none |
| tilt 10 deg (stroke plane) | x +0.06 | My -0.220 | |
| tilt_lr 10 deg | 0.01 | Mz -0.164 | yaw, a quarter of psi0_lr |
| dih_fh 10 deg | z -0.10 | My +0.082 | |
| Psi_lr 10 deg | y -0.38 | Mx -0.088 | side force |
| Psi_fh 10 deg | z -0.05 | My +0.036 | |
| feather advance +15 / -15 deg | z +2.23 / -2.46 | My +0.10 / -0.10 | thrust +23% / -25% |
| sharpened flip (tanh, k 2) | z +0.79 | 0.02 | +8% thrust, feather torque x3.3, shaft W +70% |
| right wings half a beat late (lr_phase pi) | z +0.04 | <= 0.016 | same thrust, shaft W -17% |
| hind pair a quarter beat late | z +0.04 | My +0.065 | |
| through-flat flip | z -3.17, x +0.64 | My +0.834 | -32% thrust |
| still, wings raised (B8) | (0, 0, 0) | (0, 0, 0) | thrust 0 |

Read: three axes, one clean input each. Roll from the left-right amplitude
difference (33 rad/s2 at 0.2 on Ixx 0.030), pitch from the common mean sweep
(30 rad/s2 at 10 deg on Iyy 0.024; 2.8x the fore-hind amplitude difference at
comparable inputs), yaw from the left-right feather offset (13 rad/s2 at 10 deg on
Izz 0.052). The hover is trimmed in pitch by a mean sweep of +5.9 deg (ballast
out) or +8.7 deg (ballast in). The fore-hind differences that the design note
named for pitch work but are the weaker input; the second pair's main use here is
thrust and the stroke's reaction, not a moment axis.

### X2, and what the robofly check covers

Every B5 row is in the X2 regime: airspeed 0 (< 6 m/s) with |feather| > 30 deg on
100% of the window (through-vertical flip; 54% with the through-flat flip). The
hover is the X2 regime by definition, and so is every B7 row below.

What `experiments/robofly` (and `test_the_model_reproduces_the_robofly`) checks:
one rectangular wing of R/c 2.9 revolving at a constant rate and a fixed incidence
(9-81 deg) at Re 136, third revolution, against Dickinson et al. 1999's fits: CL
within 0.159 rms, CD within 0.095 rms, after the LEV branch's constant was fitted
to that same data (MATH_AUDIT F-02: CD at 90 deg is not independent).
It covers: the mid-stroke force coefficients of a revolving wing at high incidence
with a stable leading-edge vortex, which is where the thrust above comes from
(R/c here 2.56).
It does not cover: (1) Reynolds number, 136 there against about 8e4 here (0.258 m
chord at 4.8 m/s at 0.55 R, 3 Hz); (2) the reversal: rotational force, added mass
and wake capture at the flip (the feather-advance row moves thrust +-24%, all of it
from terms the robofly run does not exercise); (3) a wing reciprocating in its own
wake and the induced inflow of hover (the check is a third revolution, not a
stroke); (4) the cone: a 35 deg dihedral stroke; (5) four wings: the solver has no
wing-wing interference, and fore and hind wings here beat 0.27 m apart with a
0.258 m chord; (6) the cambered section used inverted on the aft half-stroke;
(7) torque and power, which the check does not compare. The thrust figures above
are the model's in a regime the project lists as one it is known wrong in.

### Clearance through a full stroke (`s0b_clear.py`, `results/S0b_clearance.json`)

The machine's own geoms do not collide in MuJoCo (contype 1 / conaffinity 2) and
the solver has no interference model, so nothing in any rollout stops a wing
passing through another part. Minimum signed distance between the built geoms
(m; negative = interpenetration), targets tracked exactly, 240 phases per beat;
the same read is taken on every held row (`clearance_min`). At rest, wings flat:
wing-hull -0.011 (the wing box starts at the shoulder, inside the hull's radius),
wing-wing +0.012, wing-tail +0.031.

| stroke | wing-wing | wing-hull | wing-tail | where |
|---|---|---|---|---|
| vert, dihedral 35, A 55 | **-0.054** | **-0.034** | +0.015 | left root against right root over the hull on the aft half-stroke; root against the hull |
| the same with the inner 0.05 / 0.10 / 0.15 m of each wing absent | -0.022 / **+0.026** / +0.079 | -0.007 / **+0.027** / +0.062 | +0.034 | clears with 0.10 m gone |
| vert, A 40 (the hover amplitude) | -0.044; +0.060 at 0.10 | -0.030; +0.051 at 0.10 | +0.017 | clears at 0.10 (0.05: -0.005 / +0.010) |
| vert, A 70 | -0.062; +0.032 at 0.15 | -0.035; +0.022 at 0.15 | +0.013 | needs 0.15 m |
| vert, dihedral 20 | -0.042 | -0.033 | **-0.037 at any cut** | the hind wing meets the tail surface |
| vert, dihedral 50 | -0.064; +0.024 at 0.15 | -0.036 | +0.011 | |
| fore 50 / hind 20 dihedral | -0.064 | -0.036 | -0.037 | the tail again |
| fore-hind mean sweep split 40 deg | -0.067; +0.010 at 0.15 | -0.028 | +0.008 | worse: the hind roots meet |
| hind pair half a beat late | **-0.237 at any cut** | | | fore and hind wings cross |
| through-flat flip | -0.116; **-0.019 with 0.25 m gone** | -0.061 | -0.014 | does not clear |

Read: (1) with the pairs in phase the fore and hind wings never come nearer than
0.10 m to each other, so the different dihedral the addendum asked for is not
needed, and lowering the hind pair's puts it into the tail. (2) What interferes is
the inner 0.10 m of each wing (0.15 m at A 70): left against right over the hull,
and root against hull. T4 as built has its wing start at the shoulder. The stroke
needs the inner 0.10-0.15 m of the 0.66 m wing absent or the shoulders that much
further out. The B5 rows above have the full wing; the cut-out was then built as a
variant (T4c, below), where removing the inner 0.11 m costs 3.0% of the thrust
(38.02 against 39.24 N at dihedral 35, A 55, 4.5 Hz) and the held row's own
clearance read is positive on all three (+0.035 / +0.034 / +0.034 m). (3) The through-flat flip and any fore-hind phase lag do not
fit this body.

### Velocity derivatives at a hover point (4.5 Hz, A 26.7 deg; ids `v_*`; base 11.67 N, My +0.511)

Hull held level and moved at a constant velocity (not a typed force: the hull's
motion is prescribed, the wrench is what the solver returns).

| hull velocity | dF N | dM N m | per m/s |
|---|---|---|---|
| forward 0.5 / 1.5 / back 0.5 m/s | x -2.01 / -5.98 / +2.02 | My -0.36 / -1.11 / +0.45 | drag 4.0 N, nose-up 0.72-0.90 N m |
| left 0.5 / 1.5 m/s | y -0.71 / -2.10 | Mx -0.007 / -0.025, Mz 0 / +0.04 | drag 1.4 N; no roll moment |
| up 0.5 / down 0.5 m/s | z -0.67 / +0.81 | My -0.10 / +0.09 | heave damping 1.3-1.6 N |

## B6, the float (free on the water, actuators holding a pose; ids `rest_*`, `settle_*`)

Buoyancy by body, fully submerged (solver, static): hull 10.279 N, each main wing
0.239 N, tail and boom 0.130 N; 11.366 N in all. So "b = 1.263" is with the wings
under water. With the wings raised and dry the hull alone carries the 9.00 N
ballast-out machine at b = 1.142, and the 1.048 kg machine at 1.00.

**Wings raised (dihedral 35, sweep 0), ballast out: it does not float level.**
From level at rest the nose rises without stopping: 15 deg (default) / 20 deg
(wetting) at 6 s, **49 deg / 76 deg at 20 s** and still moving (2.4 / 5.3 deg in
the last second), 26 / 35 of 40 main-wing strips wet by then. A 20 deg push from
there: roll or nose-up, it goes on to 76-89 deg of pitch (default) or turns over
(wetting, roll -141 to -174 deg). The cause is trim, read on the model: the
ballast-out CoM is at hull x 0.211 m, the hull's own centre of buoyancy at 0.25
(the keel station was solved against the centre of buoyancy with the wings
submerged, 0.233). 9.0 N x 0.039 m = 0.35 N m nose-up on a hull that is nearly
awash. Dihedral 60 is the same (19-24 deg at 6 s).

Intermediate ballast, wings raised, 6 s from level (not settled: the pitch still
moves 1.4-3.8 deg in the last second):

| b (wings wet) | hull-only b | pitch at 6 s, default / wetting | wet main strips (of 40) | hull |
|---|---|---|---|---|
| 1.263 (ballast out) | 1.142 | +15.0 / +19.9 | 10 / 12 | tail end under |
| 1.15 | 1.040 | +5.6 / +8.4 | 6 / 6 | panels 70-100% under |
| 1.08 | 0.977 | -3.9 / -0.6 | 12 / 12 | fully under: it hangs from the wing roots |
| 1.00 | 0.904 | -25 / -25 | 26 / 26 | sinking nose first |

**Wings flat on the water, ballast out: a stable float, and every wing strip is
wet.** Settled in 8 s: hull origin (tail end of the hull axis) at z -0.023 m,
pitch +3.6 deg (default; -0.056 m and +10.9 deg under "wetting"), roll 0.0, no
motion in the last second (0.00 / 0.01 deg), 40 of 40 strips wet (the wings lie in
the surface: their lowest corner at -0.04 m). Pushed 20 deg in roll: back to 6.7
deg after 8 s (5.2 wetting) and still returning; nose down 20: back within 3 deg;
nose up 20: back to +14.5 after 8 s. The wings are the outriggers.

**Wings raised and swept forward (the wings' own mass as trim), ballast out.**
With the four raised wings held 42-45 deg forward the CoM moves over the hull's
centre of buoyancy:

| pose | settle | pitch | roll | wet strips FL FR HL HR | hull origin z |
|---|---|---|---|---|---|
| sweep 45, default, 10 s | last-second motion 0.01 / 0.11 deg | -2.2 | 0.1 | 1 1 1 1 | -0.011 |
| sweep 55, default, 10 s | | -6.0 | 0.8 | 2 2 1 1 | +0.002 |
| sweep 65, default, 10 s | | -13.3 | 2.3 | 2 3 0 0 | +0.026 |
| sweep 55, wetting, 10 s | rolling at 6.7 deg/s | -5.5 | **27.9** | 0 5 0 4 | +0.004 |
| sweep 42, default, 20 s | rolling at 6.0 deg/s | -2.2 | **14.6** | 1 2 0 2 | -0.010 |
| sweep 42, wetting, 20 s | | +0.2 | **47.2** | 0 10 0 10 | -0.017 |

Level in pitch within 2.2 deg with one root strip per wing wet (the hull axis is
0.01-0.02 m under the surface, so the wing roots are in it at any dihedral), and
pitch pushes of 20 deg return (to -3.5 / -2.2 after 8 s). **In roll it is not
stable**: top-heavy (four 41 g wings raised on a round hull), it leans from 0.1
deg at 10 s to 14.6 deg at 20 s (default) or 47 deg (wetting), and a 20 deg roll
push goes on (37.6 deg, default) or capsizes (130 deg, wetting). The two modes
split on how fast, not on the direction.

So there is no pose in which this body floats level, upright and dry-winged at
rest. It floats stably with the wings flat and wet; with the wings raised and
swept forward it is level for some seconds (roll under 1 deg for about 10 s,
default) and then leans over. `figures/S0b_T4_float.png`.

### The hover stroke over the float (hull held at its float pose; ids `stroke_on_float_*`)

4.5 Hz at the hover amplitude (A 23 deg), hull held level with its origin at
z -0.01 m. The wrench includes the hull's buoyancy, so it is not a thrust figure.

| dihedral | wet main strips, max (per wing) | lowest wing corner z | limiter steps, default / wetting | peak torque sweep N m |
|---|---|---|---|---|
| 35 | 8 (2) | -0.163 / -0.147 m | 4,144 / 6,816 | 114 / 107 |
| 50 | 4 (1) | -0.118 | 3,427 / 3,841 | 103 / 84 |
| 65 | 4 (1) | -0.081 | 5,713 / 4,177 | 96 / 71 |

All six rows are not evidence (limiter), and say why: the shoulders are on the
hull axis, the hull axis floats 0.01-0.03 m under the surface, so the root strip
of every wing is in the water at any dihedral, and at each flip the chord stands
vertical with the trailing edge 0.19 m under the spar (figure, row c: the wing
box reaches z -0.16 m). In air this stroke takes 6-8 N m at the sweep hinge; with
the roots wet it runs onto the x4 curve (96 N m).

## T4c: the wing's inner 0.11 m absent (`s0b_machine.make(root_cut=0.11)`)

The same cut the clearance read asks for, built: each wing's box starts 0.11 m
out from the shoulder (its geom mass scales with what is left: 34.2 g a wing, was
41.1) and the two inner strips of each wing are removed from the panel set (8 of
10 left). No force is typed and none is removed except with the surface that made
it. 0.8903 kg, CoM at hull x 0.214, 11.207 N fully submerged.

- air, held: -3.0% thrust at dihedral 35; at dihedral 50, 4.5 Hz, A 33 deg:
  11.36 N, roll +5.95 N m per unit amp_lr, pitch -5.09 N m/rad of sw0, yaw
  +3.17 N m/rad of psi0_lr, base My +0.549; still 0.
- float, wings flat: as T4 (pitch +3.5, roll -0.2, all 32 strips wet, stable).
- float, wings raised 50 deg and swept 42 deg forward, 10 s: pitch -1.3, roll 0.2,
  **0 wing strips wet**, lowest wing corner 0.056 m above the water; the tail is
  under (20 of 20 tail strips).
- the hover stroke over that float, held: **0 wet strips and 0 limiter steps in
  both modes**, and the two modes give the same wrench within 0.4% (17.31 /
  17.24 N including buoyancy). The lowest wing corner still reaches z -0.037 m at
  the flips: the solver places a strip at its mid-chord, so it does not see the
  trailing edge touch.

## B7, free: lift-off, 2 m, 10 s hold, descent

Law (`s0b_laws.Hover4`, hand-set): 4.5 Hz, through-vertical flip, pitch 45, pairs
in phase. Beat means of the true state; PD on the rotation vector to "level at
the starting heading" mixed through B5's map (roll -> amp_lr, pitch -> sw0, yaw ->
psi0_lr); tilt command -0.3 rad per m/s and -0.3 rad per m of horizontal error,
capped at 0.25 rad; PI on climb rate into the common amplitude. Gains from the
built inertia at wn 20 / 12 / 5 rad/s (roll, pitch, yaw), zeta 0.8. Targets: the
stroke is ramped in over 3 beats (smoothstep), law outputs low-passed at half a
beat, the wings are raised from flat over 0.5 s (beach) or 3 s (water) before it.
Tuning: 33 rollouts (ids `t01`..`t33`, probe `B7tune`, all in the jsonl), of 40.

What tuning found, in order: (1) at 3 Hz and wn 8 it is lost in 2.5 s; at 4.5 Hz,
wn 8 the roll axis diverges over 5 s although the loop's output has the right
sign, with or without the velocity loop; wn 14-20 holds. The cause was not
isolated (the held map shows no roll moment from side-slip: -0.014 N m per m/s).
(2) The hull pitches +-10 deg at the stroke frequency in free flight (four wings
in phase carry the thrust centre fore and aft); the loop works on beat means.
(3) Right wings half a beat late removes that and yaws the hull instead: lost in
4.3 s, limiter engaged. (4) Dihedral 50 flown on its own map: roll oscillates
+-40 deg, sat 54%.

| row (probe B7) | reached | hold 10 s: z, drift, tilt max | touch-down | after | peak torque sweep / stroke / feather N m | contact peak, weights (wing) | limiter | evidence |
|---|---|---|---|---|---|---|---|---|
| air_hover_T4_x4 (released at 10 m) | - | z 0.30-0.51 (target 0.5), 0.08 m, 10.6 deg | - | - | 12.8 / 4.2 / 5.4 | - | 0 | yes |
| **beach_T4_x4_ulim0.2** | 2.01 m | 1.80-2.01, 0.07 m, 11.0 deg | beat-mean vz -0.25 m/s | upright: roll -2.1, pitch +10.4 (the slope's rest pitch is 10.7), yaw +17.9 | 18.6 / 18.3 / 32.5 | up 38.2 (38.2), down 20.6, after 69.9 (20.3) | 0 | rigid-contact figure |
| beach_T4_ideal_ulim0.2 | identical to the last digit | | | | | | 0 | |
| beach_T4_x4 (pitch input limit 0.35 rad instead of 0.2) | 0.36 m, rolled 90 deg at 3.5 s | - | - | - | 89 / 53 / 33 | up 34.8 (34.8) | 0 | rigid-contact figure |
| **beach_T4c_x4** (limit 0.35) | 2.01 m | 1.80-2.01, 0.10 m, 11.0 deg | -0.25 m/s | upright: roll +0.2, pitch +10.7, yaw -1.4 | 12.9 / 15.7 / 19.1 | up 33.8 (33.8), down 17.9, after 42.0 (30.4) | 0 | rigid-contact figure |
| **float_T4, default** | 0.15 m, inverted 0.6 s after the stroke starts | - | - | - | 102 / 98 / 74 | - | 130 | no |
| **float_T4, wetting** | 0.05 m, inverted at 1.6 s | - | - | - | 112 / 111 / 82 | - | 11,560 | no |
| float_T4c (lift-off at dihedral 50, sweep 58), default | 2.32 m | 1.80-2.32, 0.33 m, 30 deg | -0.94 m/s | nose up 80 deg, wings in the water | 110 / 96 / 62 | - | 287 | no |
| float_T4c, wetting | 0.33 m, inverted at 3.4 s | - | - | - | 95 / 43 / 63 | - | 0 | yes |
| water_descent_T4 (from a hover at 2 m), default / wetting | - | - | none: inverted on the way down at 9.1 / 8.2 s | - | 97 / 99 / 65 | - | 0 / 56 | yes / no |
| water_descent_T4c, default | - | - | -0.33 m/s | nose up 28 deg after 5 s, roll -2 | 96 / 26 / 65 | - | 114 | no |
| water_descent_T4c, wetting | - | - | -0.39 m/s | nose up 79 deg, roll -29 | 37 / 10 / 28 | - | 0 | yes |

Mean positive shaft power in the hold: 164-207 W. Law outputs on a limit: 0.3-1.7%
of updates on the rows that fly. Every flying row is in the X2 regime (share
0.97-0.99). `figures/S0b_T4_liftoff.png` is beach_T4_x4_ulim0.2.

**Beach.** T4 as built lifts off, climbs to 2 m, holds 10 s within 0.2 m and
0.07 m of drift, descends at 0.25 m/s and is left upright where it started, on
the x4 motor inside the witness stall on sweep and stroke (feather 32.5 against
16) and identically on the ideal one. But not with dry-clean wings on the way
up: at each flip the trailing edge is 0.19 m under the spar and the hull's axis
is 0.04 m over the sand, so the inner half of every wing strikes the ground
until the hull is about 0.15 m up. Those strikes are 34-38 weights of rigid
contact, and they decide the row: the same law with a wider pitch-input limit
rolls over at 3.5 s. T4c (no inner wing) is the same picture with strikes of
34 weights. All four are rigid-contact figures; the contact-x2 re-run of S0 §4
was not done. The landing has 18-21 weights for the same reason.

**Water.** No row is evidence of a lift-off or of a landing.
- T4 as built: the wing roots are in the water, the stroke runs onto the limiter
  and the motor curve, and it turns over within 1.6 s in both modes.
- T4c lifts its wings dry (wet main strips while flying <= 2 of 32), and then
  the two modes split: under the default mode it reaches 2 m and holds 10 s but
  with 287 limiter steps on the way out; under "wetting" (no limiter step) it
  turns over 3.4 s after the start at 0.33 m. Six other lift-off poses and limits
  (t21-t31) gave 1 more that flew (wetting, limiter) and 5 that did not. What
  holds it is the tail and the hull's stern: at the float the tail surfaces lie
  0.01-0.03 m under the surface, the nose rises first (to 19 deg in 3.5 s under
  the default mode, pitch input at 0.26-0.32 of its 0.35 rad limit and the
  amplitude on its cap; t24), and it hangs by the tail. Sweeping the raised wings further forward (58
  deg) trims the float nose-down 6 deg and lifts the tail clear at rest, which
  is the pose of the one row that flew.
- Landing on water: T4 turns over on the way down (the wing roots meet the water
  first). T4c touches down at 0.33-0.39 m/s in both modes and is then left nose-up
  (28 / 79 deg): with the stroke stopped the raised wings no longer trim it (B6).
  A landing sequence (wings forward, then flat) was tried once per mode (t32, t33)
  and turned over before touching; it is untuned.

## B8, still machine (actuators held)

| | |
|---|---|
| held, air, wings raised (T4, two rows; T4c, one) | force (0, 0, 0.001) N, moment 0: thrust 0 |
| free, beach, 8 s (wings raised at 1.0 s) | stays: CoM +0.09 m (the raised wings), pitch 10.7, contact 1.0 weight |
| free, float, 12 s, default / wetting | stays: CoM +0.01 / +0.04 m, pitch -2.0 / -2.5, roll -0.3 / -0.6 |
| free, air, 1.5 s | falls |

## Flags, B5-B8

223 rows (141 held, 31 float, 51 free). QACC resets: 0. Peak hinge torque over
10x the witness stall: 0 rows (the x4 curve caps it at 4x). Limiter steps: 22
rows, all with a wing or the tail in the water (6 held over the float, 11 tuning,
5 B7), all "not evidence". Rigid-contact figures (over 10 weights outside the
placement): 9 rows, every beach row that flew or tried. Over the witness stall
(1x) but under 10x: 17 of the 81 grid rows (6 Hz, or A 70 at 4.5 Hz with pitch
60), at most 2.0x.

## What limits it

1. **Level hover in air: nothing found in the model.** 1.3-4.4 weights at 4.5 Hz
   inside the witness stall, three moment axes with one input each, a free 10 s
   hold within 0.08 m. All of it in the X2 regime, on a leading-edge-vortex
   branch fitted at Re 136, with no wing-wing interference between wings 0.27 m
   apart.
2. **The body does not clear its own stroke.** The inner 0.10-0.15 m of each
   wing passes through the opposite wing and the hull on the aft half-stroke;
   the through-flat flip and any fore-hind phase lag do not fit at all; a hind
   dihedral of 20 deg meets the tail.
3. **The float.** With the wings raised the bare machine is tail-heavy against
   its hull's centre of buoyancy and stands on its tail; trimmed by sweeping the
   wings forward it is level but top-heavy in roll (leans from about 10 s);
   only the wings-flat float is stable, and it is wet.
4. **Lift-off from water.** The shoulders are on the axis of a hull that floats
   awash, so the wing roots are in the water, and the flip puts the trailing edge
   0.19 m under the spar. With the roots cut away the wings are dry and the modes
   agree on the stroke; the split moves to the tail and stern leaving the water.
5. **Lift-off from land.** The same 0.19 m against a hull axis 0.04 m over the
   ground: the wings strike the ground at every flip until the hull is 0.15 m up.
   It flies through it, on rigid contact of 34-38 weights.
6. **The stroke wobbles the hull** +-10 deg in pitch at 4.5 Hz.

Not done: contact-x2 and dt/2 re-runs; a wetter/drier tail or a raised tail;
a hull standing higher (on land) or floating higher (b is fixed by the bare
mass); scored starts; any optimiser; cruise; under water with this body.

---

# S0b probes B9-B13: T5, wings above and floats below (addendum 2 of DESIGN_S0b_bold.md)

2026-11-19. Probes, not scored; no gate read. Every rollout is a row of
`results/S0b_probes3.jsonl`. Files: `s0b_t5_machine.py` (T5), `s0b_t5_probe.py`
(rollouts; B5-B8's `held`, `floatq`, `fly` reused with T5's builder, plus `held_w`
and `mission`), `s0b_t5_draw.py` (figures), `results/S0b_T5_calib.json`.
`s0b_machine._compile` gained two optional hooks (`xml_hook`, `panel_hook`, default
None); the default T4 build and the T4c build are bit-identical before and after
(sha256 over model arrays, panels and 200 steps: `3d879778...` both times).
Primary motor "x4"; "ideal" as a named twin. "Not evidence" as in B5-B8.

## T5 as built (`s0b_t5_machine.make`; `figures/S0b_T5_built.png`, `.json`)

T4c (inner 0.11 m of each main wing absent, sweep +-75 deg, feather -150..+60 deg,
no keel) plus, below the hull:

| part | geometry (hull frame: x from the tail end to the nose, z up) | mass | fluid elements |
|---|---|---|---|
| 2 pontoons | capsules, radius 0.030 m, 0.50 m long over all, axis parallel to the hull's at y +-0.20, z -0.160, centred at x 0.235; each its own body welded to the hull | 0.2127 kg each (half spread over the capsule, half at one point 0.018 m under the axis, 0.087 m ahead of the centre) | 8 bluff slices each, built as the hull's own slices (centre, extents, displaced volume, half height; cd 0.20, the hull's value). 1.357 L each |
| 4 struts | capsules radius 0.004 m: at x 0.110 and 0.360 a post from the hull's underside (z -0.03) down to z -0.160 on the centre line, and a crossbar from pontoon to pontoon at z -0.160 | the tail boom's built mass per length (17.6 g/m): 4.2 g in all | one bluff element each (cd 0.9, the solver's strut value), on the hull body |
| 2 tanks | the pontoons' own volume; a body at each pontoon's centre, 1 g empty | water: fill x 1025 kg/m3 x 1.357 L each (uniform fill of the capsule, mass and inertia; `body_mass`, the solver's dry mass, `mj_setConst`) | none: the displaced volume does not change with the fill |

No force is typed: buoyancy and drag of the pontoons and struts come from the
solver's elements above, contact from the capsule geoms (contype/conaffinity as
every machine geom).

**Masses are closed by the model, not by a material.** The pontoon mass is what
makes the flooded machine b = 1.00: weight = buoyancy of everything that is not
pontoon (11.741 N: hull 10.279, four cut wings, tail, struts) plus the unflooded
share of the pontoons. Whether a pontoon can be built at 0.213 kg is not examined.

| | ballast out (fill 0) | fill 0.95 | fill 1.00 |
|---|---|---|---|
| mass | **1.3364 kg** | 3.9795 kg | 4.1186 kg |
| buoyancy fully submerged | 39.025 N | | |
| b | 2.978 | **1.0000** | 0.966 |
| CoM (hull frame) | (0.2349, 0.0000, -0.0561) | (0.2349, 0, -0.1251) | (0.2349, 0, -0.1263) |
| centre of buoyancy, submerged | x 0.2349 (the pontoon station is solved for this) | | |
| inertia about the CoM, hull axes (Ixx, Iyy, Izz) | 0.0550, 0.0352, 0.0727 kg m2 | 0.1714, 0.0961, 0.2297 | |

**My additions beyond the specification** (each the smallest change found for a
stated fact):
1. *Neutral at fill 0.95, not 1.00.* A machine that is exactly neutral with every
   tank full cannot leave the surface without thrust. The last 5% of the pontoons'
   volume (+-0.139 kg, +-1.36 N) is the depth control; this costs 0.139 kg of dry mass.
2. *Posts and crossbars, not diagonal struts.* Struts running from the hull's
   flank to each pontoon's top were built first: the wing roots pass through them
   (-0.007 m at mean sweep -14 deg, A 44 deg). A centre-line post and a crossbar at
   the pontoons' axis height clear by >= 0.034 m on every stroke scanned.
3. *Pontoon mass split* half on the capsule, half at a low point: the point's
   station (0.087 m ahead) is what puts the dry CoM over the pontoons' centre while
   the pontoons sit under the centre of buoyancy of everything else. Without it the
   float or the flooded machine is out of trim by 0.02 m (0.23 N m).

### Sizing (kinematic scan on the built model, targets tracked exactly, 120 phases per beat)

Lowest corner of any main-wing box in the hull frame through the hover stroke
(through-vertical flip, pitch 45, mean sweep -14 / +6 / +26 deg, A 30 / 44 / 55):
dihedral 25: z -0.106 m; **dihedral 35: -0.074 m**; dihedral 45: -0.041 m (the same at
every amplitude and mean sweep: it is the trailing edge of the root at the flip).
Tail: lowest corner -0.005 m at 0 deflection, -0.055 m at its -0.6 rad stop.
With the pontoon tops at z -0.130: wing to pontoon >= 0.056 m, wing to strut >=
0.034 m (A 55, sweep -14; 0.066 at A 44), tail to pontoon 0.29 m. The held rows'
own clearance read (240 phases): wing-wing +0.094, wing-hull/strut +0.071,
wing-tail +0.102, wing-float +0.056 m at the hover amplitude; +0.048 / +0.043 /
+0.043 / +0.057 at the largest amplitude the law may use (0.9).

## B9, build and float

### Held hover map in air at T5's hover point (4.5 Hz, dihedral 35, pitch 45, through-vertical flip; ids `m_*`)

Hull held level at 30 m, 4 beats after 3. 8 rows: 0 QACC, 0 limiter steps, motor
never on its curve; the ideal-motor twin of the base row is identical to the last
digit. Every row X2 share 1.00 (hover).

| row | thrust N (weights of 1.336 kg) | moment N m (roll, pitch, yaw) | peak torque sweep / stroke / feather N m |
|---|---|---|---|
| base, amp 0.56 (A 30.8 deg) | 13.79 (1.052) | (0, +0.902, 0) | 6.9 / 4.2 / 5.0 |
| amp 0.672 | 18.67 (1.425) | (0, +1.202, 0) | 8.7 / 5.4 / 5.3 |
| amp 0.9 (the law's upper limit) | 31.31 (2.389) | (0, +1.946, 0) | 12.8 / 8.5 / 5.9 |
| amp_lr 0.2 | 13.91 | roll **+1.418** (7.09 per unit) | 7.7 / 4.5 / 5.1 |
| sw0 +10 deg | 13.81 | pitch **-0.182** (-6.21 N m/rad) | 6.8 / 4.3 / 5.0 |
| psi0_lr 10 deg | 13.67 | yaw **+0.903** (5.17 N m/rad) | 7.1 / 4.2 / 5.1 |
| still, wings raised (B12) | 0.001 | 0 | 0 |

All inside the witness stall (24 / 24 / 16). The machine's weight (13.105 N) is
carried at amp 0.544 (A 29.9 deg); pitch trim sw0 = +8.3 deg. The pontoons add no
thrust loss in this model: the solver has no wing-pontoon interference and the
pontoons sit 0.06-0.25 m under the stroke plane, in the downwash. Stated blind spot.

### The float (free, fill 0, wings raised 35 deg and flat, mean sweep 6.2 deg; ids `float_default`, `float_wetting`)

Settled 12 s from a level start, then pushed 20 deg from the settled state, 8 s each.

| | default | wetting |
|---|---|---|
| hull origin z, pitch, roll at 12 s | +0.1626 m, -0.3 deg, 0.0 | +0.1634 m, -0.3, 0.0 |
| motion in the last second (roll, pitch range) | 0.00, 0.04 deg | 0.00, 0.26 deg |
| wet main-wing strips (of 32) / tail strips (of 20) / hull slices (of 6) | 0 / 0 / 0 | 0 / 0 / 0 |
| lowest wing corner at rest | +0.215 m over the water | +0.216 m |
| what is wet | the pontoons (8 + 8 slices) and the two crossbars (0.31, 0.45 submerged) | the same (0.23, 0.38) |
| roll +20 deg push | back to 0.03 deg at 8 s (overshoot -6.2) | -0.05 deg (overshoot -6.4) |
| nose down 20 deg | back to +0.26 deg (overshoot +8.6) | +0.24 (overshoot +8.1) |
| nose up 20 deg | back to -0.21 deg (overshoot -8.4) | -0.44 (overshoot -7.8) |
| wet wing strips during any push | 0 | 0 |
| limiter steps / QACC | 0 / 0 | 0 / 0 |

**Pass**: it floats level, upright, wings, tail and hull dry, and a 20 deg push in
roll or pitch returns, in both modes, which agree. The pitch return is lightly
damped (one overshoot of 8 deg).

### The hover stroke held over the float (hull held level at its settled float height, origin z +0.1626 m; ids `stroke_on_float_*`, `still_on_float_*`)

4.5 Hz, 4 beats after 3. The wrench includes the pontoons' buoyancy (12.53 N at
this height, the still row), so it is not a thrust figure.

| row | default | wetting |
|---|---|---|
| hover amplitude (0.544): wet main-wing strips, max | **0** | **0** |
| limiter steps | **0** | **0** |
| lowest wing corner over the water through the stroke | +0.088 m | +0.088 m |
| vertical force (thrust + buoyancy) | 25.71 N | 25.69 N |
| peak torque sweep / stroke / feather | 6.7 / 4.0 / 5.0 N m | 6.6 / 4.0 / 4.8 |
| the law's largest amplitude (0.9): wet strips / limiter steps / lowest corner | 0 / 0 / +0.088 m | 0 / 0 / +0.088 m |
| vertical force | 43.87 N | 43.88 N |
| still (B12): force, wet strips | 12.53 N (buoyancy only), 0 | 12.53 N, 0 |

**Pass** (targets 0 and 0): no wing strip is wet and the limiter never steps, the
lowest wing point stays 0.088 m over the water (criterion 0.03), and the two
entrainment modes agree to 0.1%. 6 rows, 0 QACC.

## B10, lift-off and return: law and tuning (ids `t01`..., probe `B10tune`)

Law: `s0b_laws.Hover4` unchanged, mixed through T5's own map above. Tuning rollouts
are on other starts than the reported rows (beach 2 m further down the slope; the
float dropped from 2 cm higher; a release at 10 m in air).

| ids | what | result |
|---|---|---|
| t01 (3 rollouts: beach, float default, float wetting) | T4's gains (wn 20 / 12 / 5 rad/s from T5's own inertia, zeta 0.8) | all three leave the surface with no wing contact, no wet strip and no limiter step, reach 2 m, and then lose the roll axis: a 1 Hz roll oscillation that grows to +-40 deg with the roll input on its limit (sat 34-55%), inverted 0.5-3.5 s into the hold |
| t02-t05 (4, air) | roll / pitch / yaw wn 14.8/9.8/4.2 (zeta 0.6 and 0.8: T4's absolute gains), 12/9/4, 10/8/4 | 14.8: inverted at 5.9 and 11.6 s; 12: holds 10 s, tilt to 50 deg; **10/8/4: holds 10 s, drift 0.06 m, tilt 11.9 deg, 0% saturated** |

Read: the loop works on beat means (0.22 s of lag at 4.5 Hz). T5's roll inertia is
1.83x T4's (0.055 against 0.030 kg m2, the pontoons at y +-0.20), so gains scaled
with inertia double the loop gain against the same lag and the roll loop limit-cycles.
Lower gains hold. Inferred from four rollouts, not isolated.

## B11 (held part): thrust against measured drag under water (hull held level, CoM at 3 m, fill 0.95; ids `w_*`)

The stroke-hinge stroke of B2 (`s0b_laws.targets`), diagonal phasing, 0.5 Hz, 3
beats after 1. 30 rows, 0 QACC, 0 limiter steps on every one, torques inside the
witness stall except the reversed-feather row (60 N m feather, 2.5x).

| row | default: axial force N | wetting | note |
|---|---|---|---|
| still (B12) | 0.00, net vertical 0.00 (39.025 N of buoyancy = weight) | 0.00 | b = 1.000 |
| hull moved at +0.3 m/s, wings flat | **drag 2.77**, dorsal +1.63 | 2.77, +1.63 | T4's was 0.94: the pontoons and struts add 1.8 N |
| hull moved at -0.3 m/s | 2.49, dorsal +1.16, My +0.38 | 2.49, +1.16 | |
| stroke 10 / 20 / 30 deg, feather 50 deg | **36.6 / 44.1 / 57.3** | 33.6 / 41.9 / 55.4 | 13-21x the drag; nose-down moment 4.6-7.2 N m (the thrust line is 0.125 m over the CoM) |
| the same with the feather's sign reversed (20 deg) | +17.9 (still forward) | +15.4 | the feather stroke is a paddle hinged near its leading edge: thrust is toward the leading edge whatever the phase. There is no reverse. |
| throttle a (stroke 20 a, feather 50 a deg): 0.15 / 0.3 / 0.5 | 0.80 / 3.47 / 10.52 | 0.84 / 3.59 / 10.60 | thrust = drag at 0.3 m/s near a 0.27 |
| at a 0.3: Psi_lr 0.1 rad | yaw -0.570 N m | -0.576 | -5.7 N m/rad: the yaw input |
| amp_lr 0.4 | yaw -0.127 | -0.124 | weak |
| stroke bias +0.15 rad | pitch +0.184 (more nose-down), dorsal +0.12 | +0.240, +0.18 | +1.2..1.6 N m/rad |
| feather offset psi0 +0.1 rad | dorsal **+0.31** | **-0.35** | **the modes disagree in sign: not established**; not used |
| psi0_lr 0.1 / psi0_fh 0.2 | roll +0.072 / pitch -0.035 | not run | weak |

**Pass, held**: thrust exceeds the measured drag at 0.3 m/s by 13-21x in both
modes (agreeing within 8%). Two facts for the free swim: (1) thrust pitches the
machine nose-down, 0.435 N m at the throttle that balances drag, against a static
righting stiffness of only 0.52 N m/rad (the flooded CoM is 0.013 m under the centre
of buoyancy: flooded pontoons are neutral, so they right nothing); (2) the antipode
needs a 180 deg turn, since there is no reverse thrust.

### More tuning (ids t06-t16; 16 rollouts used of 80 in all, B10 and B11 together)

| ids | what | result |
|---|---|---|
| t06-t08 (beach, float default, float wetting; tuning starts) | wn 10/8/4, T4's velocity and position gains (k_v 0.3 rad per m/s, k_p 0.3 rad/m) | beach: the whole sequence, upright at the end; float default: the whole sequence, upright; **float wetting: lift-off and 10 s hold, then a lateral oscillation (period 3.3 s) grows through the hold and the descent and it turns over at 0.3 m, before any part touches the water** |
| t11-t14 (float, default; 1 m, 12 s hold; **a test disturbance at the start of the hold: +1 rad/s of roll rate and +0.3 m/s sideways, a state step, used on tuning rows only**) | k_v / k_p / roll wn: 0.3/0.1/10, 0.2/0.08/10, 0.3/0.1/8, 0.4/0.1/10 | all four hold 12 s and land upright; with wn 10 the roll swings grow to +-30 deg; with wn 8 roll stays within +-4 deg but the sideways swing grows from +-0.1 to +-0.48 m in 12 s (period 6 s) |

Read: t08 and t07 differ only in the entrainment mode, and nothing is in the water
during the hold. The default-mode row is exactly symmetric (roll 0.0 deg, y 0.00 m
throughout); the wetting row is not, and the asymmetry grows. So a clean row from a
symmetric start says the lateral mode was not excited, not that it is stable. The
lateral position loop on this body is slowly unstable at every gain set tried so
far; what a reported row shows is how far it gets in the time.
| t15-t17 (as t11-t14, roll wn 8) | k_v / k_p: 0.3/0, 0.5/0.05, **0.5/0.1** | 0.5/0.05 and 0.5/0.1: the disturbance is gone in 2 s (roll back within 1 deg, y back within 0.01 m) and nothing grows in the 12 s; a steady 0.4-0.6 m offset in x remains (trim). 0.3/0: no return, drifts 0.3 m |

**The law of the reported rows**: wn 8 / 8 / 4 rad/s, zeta 0.8, k_v 0.5 rad per m/s,
k_p 0.1 rad/m, everything else as B7.

### B10 rows (2 m, 10 s hold, descent at 0.25 m/s, 5 s after; ids `beach_*`, `float_*`)

| row | reached | hold 10 s: z over the start, drift, tilt max | touch-down (beat-mean vz) | after 5 s | peak torque sweep / stroke / feather N m | contact peak, weights: all / wing | wet wing strips flying | limiter / QACC | X2 share |
|---|---|---|---|---|---|---|---|---|---|
| **beach_x4** | 2.01 m | 1.81-2.01 m, 0.21 m, 12.2 deg | -0.25 m/s | upright: roll 0.0, pitch +6.8 (the slope), yaw -0.3 | 9.6 / 6.2 / 5.0 | up 35.4, down 37.3, after 36.0 / **0** | - | 0 / 0 | 0.988 |
| beach_ideal | identical to the last digit | | | | | | | | |
| **float_default_x4** | 2.01 m | 1.80-2.01, 0.32 m, 12.2 deg | -0.23 m/s | upright: roll 0.0, pitch -0.4, yaw -0.1 | 9.4 / 3.8 / 5.7 | none | **0** | 0 / 0 | 0.988 |
| **float_wetting_x4** | 2.01 m | 1.80-2.01, 0.15 m, 11.1 deg | -0.23 m/s (0.31 m/s sideways) | upright: roll 0.0, pitch -0.5, yaw -3.5 | 9.2 / 3.8 / 4.9 | none | **0** | 0 / 0 | 0.988 |
| float_default_ideal, float_wetting_ideal | identical to their x4 rows | | | | | | | | |

Motor never on its curve (at_limit 0), law outputs never on a limit, every torque
inside the witness stall (24 / 24 / 16), mean positive shaft power 199-213 W.

**Float: pass in both modes, which agree** (lift-off with dry wings, 10 s at 2 m,
back onto the water upright; limiter 0, wing strips wet 0). **Beach: the sequence
passes with no wing contact, but on rigid-contact figures**: the pontoons carry
35-37 weights at peaks on the way up, at touch-down and after (the hull pitches
+-10 deg at 4.5 Hz while the stroke ramps, and the pontoons hammer the sand). B13's
contact x2 re-run decides whether the beach row is evidence.
Every flying row is in the X2 regime (0.99), as every hover is.

### B10 with a disturbance (ids `float_kick_default_x4`, `float_kick_wetting_x4`; `mission` rollout, the same law)

The reported float rows start mirror-symmetric, and the default-mode row stays so
(roll 0.0 deg, y 0.00 m). A twin of each with the tuning rows' state step at the
start of the hold (+1 rad/s roll rate, +0.3 m/s sideways):

| | default | wetting |
|---|---|---|
| hold at 2 m | 10 s, 1.80-2.01 m, drift 0.32 m, tilt max 12.4 deg | **turns over 8.3 s into the hold** (drift 2.95 m, law saturated 5%) |
| descent, touch-down | -0.23 m/s, upright after 5 s (roll 0.0, pitch -0.4) | - |
| limiter / QACC / wet wing strips | 0 / 0 / 0 | 0 / 0 / 0 |

**Not established**: after a disturbance the two entrainment modes disagree on a
hover in air, where nothing is wet. The modes differ in air only through the
reaction of the air's added mass (reacted in the default mode, not under
"wetting"). So the lateral stability of this hover rests on a term of the model
that the two modes treat differently; undisturbed, both pass.

## B11, under and back (free; `mission` rollouts)

Ballast: the fill moves at no more than **0.1 of the pontoons' volume per second
(0.278 kg/s)**, the rate used; the depth loop is a PD on the CoM depth about fill
0.95 (0.4 per m, 1.5 per m/s). While sinking the wings are held at the pose of the
hover stroke's flip (raised 35 deg, chord vertical). The first two rows
(`flood_hold_*`) held the wings level with the chord vertical instead: that pose
puts the trailing edges 0.03 m inside the pontoons (the machine's parts do not
collide in the model), so those two rows are void and were re-run.

### Flood, level at 3 m, hold (ids `flood_hold2_*`; "seed" = the left tank 2% fuller than the right at every fill)

| | default | wetting | default, seed | wetting, seed |
|---|---|---|---|---|
| float to within 0.1 m of 3 m and slow | 45.6 s | 43.7 s | 47.8 s | 47.6 s |
| attitude on arrival (roll, pitch) | 0.0, -16.8 deg | 0.0, -22.9 | -12.5, -15.7 | -19.1, -17.7 |
| after 10 s more: CoM depth | 2.98 m | 2.97 m | 2.99 m | 2.99 m |
| roll, pitch | 0.0, -4.6 deg | 0.0, -2.3 | **-17.2**, -6.5 | **-14.7**, -8.0 |
| limiter / QACC / peak torque N m | 0 / 0 / 1.4 | 0 / 0 / 1.4 | 0 / 0 / 1.6 | 0 / 0 / 1.7 |

**Depth: pass in both modes** (2.97-2.99 m on the ballast alone, which agree).
**Level: only as far as the tanks match.** It arrives nose-down 16-23 deg and
comes back to 2-5 deg in 10 s; a 2% left-right mismatch leaves it listing 15-17 deg,
because the flooded machine's righting stiffness is 0.5 N m/rad (above). The sink
takes 44-48 s for 3 m: the depth control is 5% of the pontoons' volume (1.36 N).

## B12, still machine (every actuator holding its start pose)

| scenario | row ids | result |
|---|---|---|
| held in air, wings raised | `still_air_raised` | force (0, 0, 0.001) N, moment 0 |
| held over the float, both modes | `still_on_float_*` | 12.53 N = the pontoons' buoyancy at that height; 0 wet wing strips |
| held at 3 m, fill 0.95, both modes | `w_still_*` | axial 0.00 N, net vertical 0.00 N |
| free on the beach, 8 s | `still_beach` | stays: where it was put, pitch 6.8 deg (the slope's), contact 1.0 weight (pontoons), no wing contact |
| free on the float, 10 s, both modes | `still_float_*` | stays: CoM z 0.13 m, roll 0.0, pitch -0.5 deg, 0 wet strips |
| free at 3 m, fill 0.95, 15 s, both modes | `still_deep_*` | stays: CoM z -3.00 m, attitude 0.0 / 0.0, speed 0. A still machine does not swim on either heading (mean signed speed 0) |

Not run still: the surfacing (the ballast alone brings a still machine up: that is
the surfacing itself, below) and the flooding (the same).

## B13, the x2 checks (rows that passed)

| row | change | result against the base row |
|---|---|---|
| `beach_x4_tc2` | every geom's contact time constant x2 (0.004 -> 0.008 s) | the same sequence: hold 1.80-2.01 m, drift 0.22 m (0.21), tilt 12.3 deg, left upright (roll 0.0, pitch 6.8, yaw -0.2); contact peaks **18.1 / 12.9 / 18.8 weights** up / down / after (35.4 / 37.3 / 36.0): about half. The outcome does not depend on the contact stiffness; the peak loads do, so they are not load figures |
| `beach_x4_dt2` | dt 5e-5 | the same: hold 1.80-2.01, drift 0.21, upright; contact 35.0 / 25.6 / 36.4 |
| `float_default_x4_dt2` | dt 5e-5 | the same: hold 1.80-2.01, drift 0.32, tilt 12.2, upright (roll 0.0, pitch -0.4); 0 limiter, 0 wet |

The beach row ends 0.69 m down the slope from where it started (CoM x 19.54 against
20.23 for the still machine): the law holds a steady 0.4-0.6 m offset in x.

### Surfacing: tuning (ids t18-t26, start at rest at 2.5 m, default mode; "seed" as above)

| ids | what | result |
|---|---|---|
| t18 (no seed) | wings raised and flat, tanks blown from 0.95 to 0 at 0.1 /s from depth | **arrives upside down** (roll -178 deg) and stays so: inverted it floats as stably as upright |
| t19, t20 (seed) | wings raised and flat, rise at fill 0.88 (+1.9 N), then blow at 1.0 or 0.3 /s near the surface | rolled 94 deg before reaching the surface; ends inverted at both rates |
| t21, t22 (seed) | the same with the wings at the flip pose (chord vertical) | rolled 55 deg on the way up; blown at 1.0 /s it ends inverted, at 0.3 /s it happens to come up right |
| t23, t24 (seed) | the same with a roll trim by the two tanks' difference (PI on roll, at the fill rate, +-0.15) | worse: inverted before the surface (roll -154 deg), both |
| **t25** (seed) | **flip pose, rise at fill 0.94 (+0.27 N, 0.03 m/s), blow at 0.3 /s from 0.35 m under** | roll -8 deg on arrival, **afloat upright 3.6 s after the blow starts**, wings dry, roll 0.0 / pitch +0.1 after 6 s |
| t26 (seed) | the same with the wings flat | rolled 20 deg, ends inverted |

Why (geometry, read on the built model): the float's stability is its waterplane
(two pontoons 0.40 m apart). Under water there is none, and with the tanks empty
the buoyancy is the pontoons', 0.16 m under the hull, while the mass is the hull's:
the CoM is above the centre of buoyancy for every fill under 0.56. Blowing the tanks
at depth therefore turns the machine over; so does a fast rise, because the wings'
and hull's drag acts above the CoM when it moves up (it acts as a stabiliser when
it sinks, which is why the flooding rows are level). What works is to stay heavy
(fill 0.94: the CoM 0.011 m under the centre of buoyancy), rise slowly, and blow
only when the pontoons are about to break the surface, fast enough to cross the
unstable range of fills (0.56 down to about 0.5) in about a second: **0.3 of the
pontoons' volume per second (0.83 kg/s), three times the rate used for flooding.**
This is the one place a rate above 0.1 /s was needed. 26 tuning rollouts used so far.

### Free swim on a heading and its antipode (ids `swim_fwd_*`, `swim_anti_*`; tuning t09, t10)

The same start for both (at rest, CoM at 3 m, fill 0.95, wings flat). Law
(`mission`, kind "swim"): the diagonal stroke at 0.5 Hz, throttle a (stroke 20 a,
feather 50 a deg) from a PI on the speed along the commanded heading (target 0.3
m/s, a <= 0.5), yaw by Psi_lr (+-0.15 rad), pitch by the stroke bias (+-0.15 rad:
more puts the wings inside the pontoons, t09 read -0.035 m at -0.4), depth by the
ballast. The antipode is a commanded heading 180 deg away (there is no reverse
thrust). 30 s forward, 45 s antipode. One tuning pair only; the law is not tuned.

| | default | wetting |
|---|---|---|
| forward: distance along the command / mean speed over the run / over the last 8 s | 2.13 m / 0.071 / 0.056 m/s | 4.29 m / 0.143 / 0.144 m/s |
| forward: attitude at the end (roll, pitch, yaw error), depth range | 0.9, -19.1, 1.3 deg; 2.96-3.69 m | 2.5, +6.6, **41.6 deg off heading**; 2.96-3.60 m |
| antipode: turned | **22 of 180 deg** (error -158 deg at 45 s) | 132 of 180 deg (error 48 deg) |
| antipode: distance along the command / mean speed / last 8 s | **-0.26 m** / -0.006 / -0.004 m/s | +3.12 m / 0.069 / 0.185 m/s |
| **mean signed speed** (half the sum of the two, whole run / last 8 s) | **0.033 / 0.026 m/s** | **0.106 / 0.165 m/s** |
| peak torque stroke / feather N m | 71 / 64 | 63 / 12 |
| limiter / QACC / wing-to-pontoon clearance min | 0 / 0 / +0.02 m | 0 / 0 / +0.02 m |
| still machine (B12) | 0.000 m/s | 0.000 m/s |

**Fail against 0.3 m/s in both modes, and the modes disagree by 3-6x: not
established.** Under the default mode the machine makes a tenth of the held
thrust's speed, pitches nose-down and does not turn; under "wetting" it moves at
0.14-0.19 m/s, turns most of the way and wanders 40-50 deg off heading. The held
thrust (3.5 N at a 0.3, 13-21x drag at full stroke) does not reach the free body,
as T4's B3 found. Torques 2.6-4.0x the witness stall (under the 10x flag). Depth
is not held while swimming (0.6-0.7 m of excursion on a 1.36 N ballast margin).

### Surface, settle, lift off (ids `surface_liftoff_seed_*`: from rest at 3 m, 2% tank mismatch, t25's recipe; then 2 m, 5 s hold)

| | default | wetting |
|---|---|---|
| rise at fill 0.94 to 0.35 m under | 62.3 s, roll -6.8 deg on arrival | 62.6 s, roll -6.2 |
| blow at 0.3 /s: afloat | 3.1 s | 3.8 s |
| after 6 s on the float: roll, pitch; wet wing strips | 0.0, +0.1 deg; **0** | +0.1, +0.3; **0** |
| lift-off, hold at 2 m | 5 s, 1.81-2.01 m, drift 0.22 m, tilt 12.7 deg | 5 s, 1.80-2.01 m, drift 0.11 m, tilt 11.2 |
| limiter / QACC | 0 / 0 | 0 / 0 |
| peak torque sweep N m (in the lift-off) | **106** (4.4x the witness stall, on the x4 curve; not explained; the chain rows' lift-offs read 9-12) | **105** |

**Pass in both modes, which agree**, with the recipe above, seeded. Not evidence of
robustness: one start, one seed size, and the recipe was found on the sixth try.

### The whole sequence in one rollout (ids `chain_default`, `chain_wetting`; `figures/S0b_T5_mission.png` is chain_wetting)

float 3 s > lift-off, 2 m, 10 s, descent, touch-down > flood to 3 m > hold 5 s >
wings flat > swim forward 16 s > flip pose > rise at fill 0.94 > blow at 0.3 /s >
wings raised, 4 s > lift-off, 2 m, 5 s. No seed (mirror-symmetric start).

| phase | default | wetting |
|---|---|---|
| first flight | hold 1.80-2.01 m, drift 0.32, tilt 12.2; down at -0.23 m/s, upright (roll 0.0, pitch -0.3) | 1.80-2.01, 0.15, 11.1; -0.23 m/s (0.30 sideways), upright |
| flood | **stalls at 1.05 m with the tanks full (fill 1.00) for the last part of 70 s; not explained** (the same phase from the float alone reached 3 m in 45.6 s, `flood_hold2_default`) | 2.90 m in 42.4 s; 2.95 m, pitch -8.6 deg after 5 s more |
| swim 16 s | 0.32 m, 0.02 m/s, nose up 40 deg, at 0.7-0.9 m depth | 2.94 m, 0.15 m/s, 35 deg off heading, depth 2.76-3.47 m |
| rise, blow | 110 s (the phase's limit) to 0.36 m under; afloat 3.4 s after the blow | 69.7 s; afloat 3.5 s |
| on the float | roll 0.0, pitch -1.0, wings dry | roll 0.0, pitch +0.1, wings dry |
| second lift-off | hold 5 s at 1.80-2.01 m, drift 0.22, tilt 12.2 | 5 s, 1.80-2.01, drift 0.10, tilt 11.1 |
| limiter steps (every phase) / QACC | 0 / 0 | 0 / 0 |
| peak torque stroke / feather / sweep N m | 41.9 / 29.9 / 12.1 (the swim) | 26.2 / 4.9 / 9.2 |
| X2 share of the flights | 0.97-0.99 | 0.97-0.99 |

The wetting row runs the whole chain; the default row runs it too but its
underwater part happens at 1 m, not 3 m.

## Flags, B9-B13

104 rows in `results/S0b_probes3.jsonl` (14 held in air or over the float, 30 held
under water, 2 float, 22 `fly`, 36 `mission`), 28 of them tuning (t01-t26; t01 is
three rollouts): **28 of the 80 allowed**. QACC resets: 0. Limiter steps: 0 on
every row. Peak torque over 10x the witness stall: 0 rows. Over the witness stall
but under 10x: 17 rows (the swims 2.6-4.0x, the failed surfacings, the seeded
surfacings' lift-off 4.4x, one reversed-feather held row 3.8x). X2: every flying
row 0.94-0.99, as any hover. Rigid-contact figures: every beach row (35-37 weights
on the pontoons; 13-19 at twice the contact time constant; wing contact 0
everywhere). Mode agreement: float, stroke over the float, lift-off and return,
held thrust and drag, depth, surfacing agree; **the disturbed hover and the free
swim do not**. Two rows are void (`flood_hold_default`, `flood_hold_wetting`: a
wing pose inside the pontoons).

## What limits T5

1. **Float, dry wings: nothing found.** Level within 0.3 deg, wings 0.215 m over
   the water at rest and 0.086 m at the lowest point of the largest stroke, 20 deg
   pushes return, modes agree.
2. **Lift-off and return, float and beach: nothing found undisturbed**, inside the
   witness stall, wings never wet and never on the sand. Disturbed, the modes
   split on a hover in air (item below). On the beach the pontoons take the
   hull's +-10 deg pitch wobble as hammering.
3. **The lateral mode of the hover.** T5's roll inertia is 1.83x T4's; T4's gains
   limit-cycle, lower ones hold, and after a kick the default mode recovers while
   "wetting" turns over. A mirror-symmetric rollout cannot show this.
4. **Under water the pontoons right nothing.** Flooded they are neutral, so the
   righting stiffness is 0.5 N m/rad: a 2% tank mismatch lists it 15-17 deg, the
   thrust at drag-balancing throttle (0.435 N m) is most of a radian of pitch, and
   the stroke that would trim it puts the wings into the pontoons.
5. **Free swimming is not shown**: 0.03 m/s (default) against 0.11-0.17 (wetting)
   mean signed speed, target 0.3; the turn for the antipode happens in one mode only.
6. **Surfacing is unstable by geometry**: empty tanks under a heavy hull. It comes
   up right only heavy, slow (60-70 s for 3 m) and with a fast last blow (0.83 kg/s).
7. **Depth authority** is 1.36 N: 44-48 s to sink 3 m, and not enough to hold
   depth against a swim.
8. Unchanged from T4: everything flown is in the X2 regime, on a leading-edge-vortex
   branch fitted at Re 136, with no interference between the four wings or between
   the wings' downwash and the pontoons under them.

## Not done

- B13 for the water scenarios (dt/2 of the flood, the surfacing, the chain); the
  contact x2 of the float rows does not apply (no contact).
- B12 for the flooding, the surfacing and the chain as separate still rows (the
  still machine was run on the beach, on the float, held and free at 3 m).
- Why `chain_default` stalls at 1.05 m; why the seeded lift-offs read 105 N m.
- Any pontoon geometry other than the first that met the criteria (r 0.030, L 0.50,
  +-0.20, z -0.160); the smallest one that still passes was not looked for.
- A disturbed twin of the beach row; a disturbance in pitch or yaw; any second seed.
- Swim tuning beyond one pair (52 rollouts of the budget unused); a pitch input
  that clears the pontoons; a turn in place; another stroke under water.
- Whether a pontoon and its tank can be built at 0.213 kg; the pump the 0.83 kg/s
  blow implies. No material, construction or drive is chosen.

# D1-D2 and B14-B17: two diagnostics on T5, then T5b

2026-11-20. Rows: `results/S0b_probes4.jsonl` (every rollout one row; ids `d1_*`, `d2_*`, `b14_*` ...). Files:
`s0b_t5b_*.py`. Nothing in `s0b_machine.py`, `s0b_t5_machine.py`, `s0b_t5_probe.py`, `s0b_laws.py`,
`s0b_probe2.py` or `mission.py` is edited (sha256 below). Build hash before (sha256 over model arrays, panels
and 200 steps, `s0b_t5b_hash.py`; this digest is not the one B9-B13 printed): T4 `6070e5a5353cace3`,
T4c `281eed84715ae7df`, T5 `f8e72992cbafc736`, T5 at fill 0.95 `6f735d284395364f`.
Tuning budget: read as a fresh 80 for this part (28 were used in B9-B13); every tuning rollout carries id
`u01`, `u02`, ...; reported rows use starts not used by a tuning rollout.

## D1: the kicked hover, float against 10 m of air (ids `d1_*`)

Harness `s0b_t5b_d1.py` (calls `s0b_t5_probe.mission` unchanged; ``z_float`` moves the start so that nothing
touches the water). Kick as B10: +1 rad/s roll rate and +0.3 m/s sideways at the start of the 10 s hold at 2 m.
From the source (`fluid.py:1865-1885`) the modes are not identical in air: the default mode reacts
`-dm/dt v` of the air's added-mass tensor as it turns with the wing, "wetting" reacts only a density rise
(none in air). So the question is the size of that term and whether it splits the outcome.

Rows: `d1_*` (6 rollouts: float and air at 10 m, kick; air without kick; both modes) and `d1b_*` (3 rollouts with
`entrainment=False`, plus 15 held rows). Kick = +1 rad/s roll rate, +0.3 m/s sideways, at 2 m (z_start + 1.8 m
reached), hold 10 s, then descent. 0 QACC, 0 limiter steps on every rollout. The two float rows are B10's
`float_kick_*` again: **their traces equal B10's to the digit** (max abs difference 0.0), so the probe path has not moved.
The air rows end with the stroke cut at z_start + 0.03 = 10.2 m and a 10 m fall that tumbles the machine
(abort "inverted" at 25.6-27.0 s on rows whose hold was fine); only the hold and descent are read.

Roll (deg) and lateral position (m) after the kick, largest value in each 2 s window (from the traces):

| row | 0-2 s | 2-4 | 4-6 | 6-8 | 8-10 | 10-12 | outcome |
|---|---|---|---|---|---|---|---|
| float, default | 6.7 / 0.08 | 1.4 / 0.02 | 0.1 / 0.01 | 0.1 | 0.0 | 0.0 | recovers; hold tilt 12.4 deg; lands |
| float, wetting | 7.0 / 0.17 | 4.2 / 0.23 | 4.7 / 0.37 | 10.6 / 0.52 | 89 / 2.95 | | turns over (16.32 s) |
| float, off (False) | 7.0 / 0.17 | 4.2 / 0.23 | 4.7 / 0.37 | 11.1 / 0.53 | 89 / 2.8 | | turns over (16.27 s) |
| **air 10 m, default** | 6.9 / 0.08 | 1.3 / 0.02 | 0.1 / 0.01 | 0.1 | 0.0 | 0.0 | recovers; hold tilt 15.2 deg; descends |
| **air 10 m, wetting** | 8.4 / 0.08 | 2.4 / 0.06 | 2.0 / 0.12 | 1.9 / 0.17 | 3.2 / 0.26 | 6.0 / 0.29 | grows; turns over in the descent (17.41 s) |
| air 10 m, off (False) | the same to the digit (max abs trace difference 0.0) | | | | | | 17.41 s |
| air 10 m, no kick, default | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | |
| air 10 m, no kick, wetting | 0.1 / 0.01 | 0.1 | 0.1 | 0.1 | 0.2 | 0.2, then 0.5 | slow growth with no disturbance |

**Answer.** Not identical in air, so the second branch of the question: the solver differs when dry, and it is found.
`fluid.py:1865-1885` reacts `-dm/dt v` of the added-mass tensor's growth in the default mode; in air the tensor
turns with every wing, so the term is there. "wetting" reacts only a density rise, and in air (rho constant) it
does nothing: **in air "wetting" and `entrainment=False` are the same computation** (traces equal to the digit),
and the float rows' `off` agrees with `wetting` to 1% (89 / 2.8 against 89 / 2.95). No water is needed for the split:
the 10 m air row reproduces it (default recovers in 3 s, wetting/off grows and turns over). What the term does to
the hover, measured on the hull held level in air (T5 hover point, 4.5 Hz, amp 0.544, hull moved at +-0.3 m/s;
`d1b_held_*`, 4 beats after 3; the three modes agree on thrust 13.16-13.17 N at rest):

| derivative (per m/s of hull velocity) | default | wetting = off |
|---|---|---|
| lateral force, d Fy / d Vy | **-1.57 N** | -0.82 N |
| roll moment from sideways speed, d Mx / d Vy | **+0.149 N m** (restoring: left side up when moving left) | **-0.050 N m** (the other sign) |
| yaw moment from sideways speed | -0.200 N m | -0.195 N m |
| fore-aft force, d Fx / d Vx | -3.80 N | -2.91 N |
| pitch moment from forward speed | -1.15 N m | -0.90 N m |

So the default mode carries 1.9x the lateral damping and a roll-from-sway coupling of the stabilising sign;
without the term the lateral mode of this hover (sideways drift, roll, tilt command) is slowly unstable even
undisturbed (no-kick air row: roll envelope 0.1 -> 0.5 deg in 12 s) and a kick sets it going. T5's tuned loop
(wn 8/8/4, k_v 0.5, k_p 0.1; tuned on the default mode) therefore stands on a term that is a blind spot of the
air model (added-mass reaction of an unsteady wing in air has no check against a measurement here). **A hover
law is established only if it holds in the mode without the term.**

What differs at the start of the float rows (the part "say what differs"): both modes start mirror-symmetric
and kick at the same time in the hold, but their lift-offs have already parted: settled float z 0.1626 against
0.1634 m, pitch range 0.04 against 0.26 deg (B9), and at the kick instant the stroke phase differs (vz -0.23
against +1.82 m/s, pitch -4.2 against +5.5 deg, a beat-phase effect). The start state is not the cause: air
from rest shows the same split, only slower (roll envelope at 6-8 s: 1.9 deg in air against 10.6 deg on the
float); the float start doubles the residual (4.2 against 2.4 deg at 2-4 s) and
the pontoons are the only wet thing there. Margin problem of the hover law on a term the two modes treat
differently, amplified by the float start, not a water effect.

## D2 (a): the 105 N m sweep torque in the surfaced lift-off (ids `d2_kp_*`, `d2a_*`)

**Cause: stale hinge gains, a harness artifact, not the machine.** `T.regain` (mission setup) sets each hinge's PD
gains from the mass matrix in the medium the rollout starts in. The sweep hinge's kp read 1,595 N m/rad at the
float start (`d2_kp_float`) and **128,449 at the deep start (`d2_kp_deep`), 80.5x**, because under water the wing
carries its added mass. `surface_liftoff_seed_*` start deep and never re-read the gains, so the lift-off in air flew on gains
80x too stiff; B10 and the chain start on the float (gains right for air). Reproduced from a shallow start
(CoM 0.7 m, same recipe, 2% tank seed, default mode, `d2a_*`; spikes logged above 20 N m):

| arm | sweep kp in the fly phase | peak torque sweep / stroke / feather N m | sweep motor on its curve |
|---|---|---|---|
| gains as at the start (`d2a_default_noregain`) | 128,449 | **106.1** / 21.2 / 26.1 | 0.30% of steps |
| gains re-read when the wings leave the water (`d2a_default_regain`) | 1,715 | **9.4** / 3.8 / 4.9 | 0 |

When: 23.7-24.5 s, i.e. from 0.3 s into the fly phase (it starts at 23.43 s, `sub` = "up", the stroke amplitude ramping in
and the hull climbing); the largest value, -105.9 N m, at 24.328 s with the wing at q 0.301 rad, target 0.306
(error 5 mrad x 128,000 = the torque), speed 16.5 rad/s, on the motor's curve. Command at that instant: the hover stroke at
its ramp (amp 0.54 -> 0.69), nothing unusual. With the right gains the same lift-off reads 9.4 N m, the value
of every other lift-off. **Consequence: a rollout that changes medium must re-read the gains at the transition
(`regain_at`)**; the surfaced lift-off rows of B11 (peak 105-106 N m, "4.4x the witness stall") are rerun-worthy and are
not evidence of a limit of the machine. The converse also holds and matters more: **the chain rows' swim and
surfacing ran with the gains of the float start (kp 1,595) in water, where 128,000 is right** (80x too soft:
`chain_*`'s swim, 0.02-0.15 m/s, is not comparable with `swim_*`, which started deep). T5b's mission must re-read
the gains at each medium change.

Second anomaly of that part (the default-mode rise with the wings held read 36.3 N m on the stroke hinge,
wetting 0.1): a **single step** at t = 2.18 s (1.2 s after the rise starts), q = target to 1e-4 (error x kp = torque;
`d2a_default_*` both arms, the spike log has one entry in the phase), default mode only. The command is constant; the
only default-mode-only quantity is the added-mass reaction impulse (`entrainment_reaction`). **Explained to that
extent: not a command, not a gain error. The impulse itself was not traced step by step.**

## D2 (b): why the default-mode flood stalled at 1.05 m in `chain_default` (ids `d2b_*`)

The chain's first three phases (settle 3 s, lift-off and 10 s hold at 2 m, landing, flood to 3 m at fill rate 0.1/s with
the wings moving to the sink pose), default mode, with the extended trace:

| arm | flood phase result (70 s limit) |
|---|---|
| A as is (`d2b_default_A_asis`) | **stalls: CoM z -1.05 m at 70 s, fill 1.00, roll +2.3, pitch -4.8, vz -0.001**: the same numbers as `chain_default` (z -1.05, att 2.3 / -4.8 / -0.3): reproduced to the digit |
| B `s.reset()` at the start of the flood (`d2b_default_B_reset`) | reaches 3 m in **45.6 s**, pitch -16.8 deg on arrival: the same numbers as `flood_hold2_default` (45.6 s, -16.8) |

So the stall is **solver state carried from the flight**, not physics: a rollout that has flown and then floods stalls, a
reset at the flood start removes it. In arm A the net vertical force goes to zero by itself: at fill 1.0 (weight 40.39 N)
the solver's vertical fluid force reads 41.4-41.9 N at 12-20 s (buoyancy 38.7-39.0 N: 2.4 N of non-buoyant lift) and
39.05 N at 30 s, with vz decaying from -0.06 to -0.007 m/s while the sinking force stays 1.3 N, so the velocity is being
absorbed, not the force. Which carried component it is: see the next table.

## T5b: what the model says about righting, and the one change (`s0b_t5b_machine.py`, `s0b_t5b_size.py`, `results/S0b_T5b_size.json`)

**Targets, written before sizing.** The tilting moments measured on T5: thrust at the drag-balancing throttle 0.435 N m
(B11: the thrust line is 0.125 m over the flooded CoM) and a 2% tank mismatch 2 x 0.026 kg x g x 0.20 m = 0.104 N m,
together 0.54 N m. Criterion R1 (mine, stated here, not loosened afterwards): the righting moment at a 20 deg tilt, K sin 20 deg,
at least **1.0 N m (1.9x the tilting moments)**, so K >= 2.9 N m/rad in roll and in pitch, fully submerged at the neutral fill.
R2: every wing clears every part by >= 0.03 m over the whole input range of the under-water stroke, including the
largest pitch input the law may use. R3: the float criteria of B9 again (0.03 m dry clearance of every wing point over the
hover stroke, no wet wing strip, no limiter step, 20 deg pushes return, modes agree).

**The fact that blocks the brief's first idea.** Measured with the buoyancy's moment about the dry CoM at a tilt of 0.02 rad
(`righting`, wings in the sink pose, fully submerged): T5 at fill 0.95 K = **0.34 N m/rad** (B11 read 0.52 with another
wing pose), at fill 0.5 -0.41, at fill 0 **-2.86** (unstable, which is the surfacing result of B11). Moving ballast volume
down or compensating a fixed float with fill cannot change K: a flooded tank's water weighs what the volume it fills
displaces, at the same point, so tanks cancel at every fill near 1; what is left is
K = (dry structure weight) x (height of the dry structure's buoyancy centre over its mass centre) + (1 - fill) x (tank volume's buoyancy) x (its height over the dry mass centre).
The second term is 1.36 N x 0.2 m = 0.27 N m/rad at most. For a neutral machine, extra fixed buoyancy high up is paid for
only by **dense dry mass**, not water (nothing denser than the volume it displaces fits in a flooded tank). Fill compensates
at most the last 5% of the tanks (1.36 N, which is already the depth control). So the smallest change that removes the fact:

**Change (my addition, labelled): a dorsal float.** One air-filled capsule on the hull's centre line above it, between the
raised wings (the wing roots are off the centre line; clearance read kinematically, below). Its buoyancy needs equal dense dry mass
for neutrality; in T5's own rule that mass is what the pontoon shells come to (derived, not typed), and they sit at the lowest
height the beach allows (pontoon axis z -0.16, shell mass at 0.6 r under the axis). Built as T5's pontoon slices, on the hull body.
Nothing else changes. Sizes tried (K at fill 0.95 / fill 0.5 / fill 0, N m/rad; all wings sink pose; mass at fill 0):

| dorsal (radius, length, height z) | volume | dry mass | K 0.95 | K 0.5 | K 0 |
|---|---|---|---|---|---|
| none (T5) | 0 | 1.336 | 0.34 | -0.41 | -2.86 |
| 0.020, 0.40, 0.09 | 0.49 L | 1.834 | 1.60 | 0.99 | -0.67 |
| 0.020, 0.40, 0.20 | 0.49 L | 1.834 | 2.14 | 1.52 | -0.14 |
| **0.025, 0.40, 0.20** | **0.75 L** | **2.108** | **3.13** | **2.57** | **+1.18** |
| 0.030, 0.40, 0.20 | 1.07 L | 2.438 | 4.32 | 3.82 | +2.67 |
| 0.025, 0.45, 0.25 | 0.85 L | 2.208 | 3.92 | 3.38 | +2.07 |

**T5b = dorsal 0.025 / 0.40 / z +0.20** (the smallest of the rows with K >= 2.9; also the only one of the cheaper ones that
is stable at every fill, so the surfacing instability of B11 is gone by geometry). Dry mass **2.108 kg** (+0.772 kg, +58%; weight
20.67 N), pontoon shells 2 x 0.599 kg (derived by neutrality at fill 0.95; whether a shell can be built at that mass is not examined,
as in T5), 4.751 kg at fill 0.95, b = 1.000 at fill 0.95 exactly as T5. Inertia about the CoM at fill 0.95: 0.211 / 0.116 / 0.259 kg m^2
(T5 0.178 / 0.107 / 0.222). Hover needs amp 0.713 (T5 0.544) of the law's 0.9 limit: thrust at amp 0.9 is 31.3 N = 1.51 weights
(T5 2.39), so the machine's lift margin falls from 2.4 to 1.5 weights; that is the price of the passive righting.

Clearance of the dorsal float (kinematic, targets tracked exactly; `s0b_t5b_clear.py`, dorsal 0.02/0.40/0.20 built, hover stroke over
amp 0.544 / 0.9 and mean sweep sw0 0 / +0.3 / -0.3 rad): smallest distance from any wing 0.046 m (amp 0.544, sw0 -0.3; `seg1_wing`)
and 0.046-0.073 m elsewhere, **0.021 m only at amp 0.9 with sw0 -0.3 rad, where wing-wing (0.003), wing-hull (0.010) and
wing-tail (0.020) are already worse on T5**: the dorsal float is never the binding part. (The built 0.025 radius float is 5 mm wider:
re-read with the T5b build in B14/B15 rows' `clearance_min`.)

### B14, T5b: the float again (ids `b14_*`; `floatq` and `held`, both modes, 0 QACC, 0 limiter steps on every row)

Float (free, fill 0, wings raised 35 deg, settled 12 s, pushed 20 deg for 8 s):

| | default | wetting |
|---|---|---|
| hull origin z, pitch, roll at 12 s | +0.1461 m, -0.43 deg, 0.00 | +0.1461 m, -0.40, 0.00 |
| wet main-wing strips / tail strips / hull slices | 0 / 0 / 0 (the two crossbars are the only wet struts) | 0 / 0 / 0 |
| lowest wing corner at rest | +0.198 m over the water | +0.198 m |
| roll +20 push, end after 8 s (roll, pitch, yaw) | 0.06, -0.42, +2.9 deg (yaw 2.9 deg from the push) | -0.04, -0.05, +4.7 |
| nose down 20 / nose up 20, end | -0.92 / -0.22 deg | -1.37 / +0.18 |
| wet wing strips during any push | 0 | 0 |

The hover stroke held over the float (hull held at 0.1461 m, 4.5 Hz, 4 beats after 3; `b14_stroke_on_float_*`):
hover amp 0.713 and the law's largest amp 0.9, **0 wet main-wing strips, 0 limiter steps, lowest wing point 0.072 m over the water**
(criterion 0.03), in both modes, vertical force 41.2 / 51.8 N in both (buoyancy at that height 20.51 N, the still row), torques
sweep 9.6 / feather 5.3 / stroke 6.0 N m at hover (witness stall 24 / 16 / 24), wing-pontoon clearance 0.056 m (0.059 at amp 0.9),
wing-hull 0.048, wing-wing 0.069. **R3 passes in both modes, which agree (0.1%).** Held hover map in air for the law
(`b14_m_*`, amp 0.70: 20.03 N = 0.969 weights, pitch +1.257 N m; roll +10.9 N m per unit amp_lr, pitch -9.5 N m/rad of sw0, yaw +7.94 N m/rad of psi0_lr; peak sweep torque
9.3 N m; amp 0.9: 31.3 N, wing-hull clearance 0.040). Hover amp 0.713 from the map (thrust ~ amp^1.75), trim sw0 +0.132 rad.
Every flying row is in the X2 regime (share 1.00, hover).

**Which carried state (arms C and D, same rollout, one piece of the solver's state cleared at the start of the flood; `reset_parts`):**

| cleared at the flood start | flood result |
|---|---|
| nothing (A) | stalls at -1.05 m |
| everything, `s.reset()` (B) | 3 m in 45.6 s |
| the implicit aero damping only (`d2b_default_C_damping`) | stalls at -1.05 m (identical path) |
| the previous-step added mass and density, entrainment bookkeeping only (`C_prev`) | stalls at -1.05 m |
| `_unsteady` (Wagner lag, LEV history) only (`D_unsteady`) | stalls at -0.99 m |
| `_unsteady` and `_inflow` (`C_unsteady_inflow`) | 3 m in 45.6 s |
| **`_inflow` only** (`D_inflow`) | **3 m in 45.5 s, pitch -16.8 deg: the same as B** |

**D2 (b) answer: the stall is the momentum-theory downwash state `InducedFlow.w` (`fluid.py:800`, one 3-vector for the whole machine,
first-order lag with time constant 0.5 x span / |V + w| and a floor of 0.05 m/s on the through-flow), carried from the flight into the flood.** Clearing it
alone restores the 45.5 s sink; clearing any other piece does not. Why a stale `w` holds a sinking machine at 1.05 m
(a net vertical force going to zero as the machine sinks, with the velocity absorbed) was not traced: `w` itself was not logged
(the trace has the fluid force, mass, buoyancy and damping), and the lag with the through-flow floor makes a fixed point at small `w` plausible, not shown.
Consequence for the harness: a rollout that flies and then floods must clear the inflow state when the medium
changes (T5b's mission does, `reset_parts`); `chain_default`'s underwater part (swim at 0.7-0.9 m depth, 0.02 m/s) is not a measurement of the machine,
and `chain_wetting` (reached 3 m) was not affected.

## B15: held thrust under water at the swim speed (ids `b15_*`; T5b at 3 m, fill 0.95, hull held level and moved at U; 0 QACC)

Not the held-at-rest thrust B11 used: the free swim did not follow it, and the reason is in the speed. **The paddle cannot push the machine faster than
its own tip speed**: the B11 swim law's stroke was 20 deg x a, a <= 0.6 (+-6 deg) at 0.5 Hz, a tip speed of 0.13 m/s; B16's free swims on T5 and
on T5b with that law read 0.06-0.14 m/s, the same number. Net axial force on T5b at U = +0.3 m/s (thrust less hull, strut and pontoon drag; stroke-hinge stroke, feather 50 deg, diagonal
phasing; default mode):

| f, stroke (full angle) | net Fx at 0.3 m/s | peak torque stroke / feather N m | note |
|---|---|---|---|
| 0.5 Hz, 10 deg | **-6.5 N** | 40 / 29 | below the drag |
| 0.5 Hz, 20 deg | **+10.3 N** | 35 / 10 | |
| 0.5 Hz, 30 deg | **+24.7 N** | 35 / 9 | |
| 1.0 Hz, 20 / 30 deg | +111 / +152 N | 96 / 41-63 (the x4 motor's stall) | motor on its curve |
| 1.0 Hz, 10 deg; 2.0 Hz, all three | -195; -693 ... -1715 N | 97-100, limiter steps up to 706 | **not evidence**: the hinge cannot follow in water above 0.5 Hz (motor-limited, negative forces are tracking failure) |

So at 0.5 Hz the net force at 0.3 m/s changes sign at a stroke of ~13-14 deg: the law needs a throttle up to a ~ 1.3 (stroke 26 deg), not the 0.6 the B11 law allowed.
The water stroke is torque-limited above 0.5 Hz on this motor. No reverse thrust in this family (B11: the sign-reversed feather still thrusts forward; here the same by the paddle's
geometry: thrust is toward the spar's leading edge whatever the phase); the level-hover family used as a paddle (`pf_*`, `pr_*`, `hv_*`, sweep stroke at 0.2 rad, 0.5 Hz) gave Fx +24 / +8
(forward again for the reversed pitch pattern) and large upward force (+20 to +38 N), the reversed-flip variant with the sharpened flip blew up (-437 N, limiter): **no reverse thrust and no
clean heave found; the antipode is a turn** (B16 below). 7 rows of that grid completed before it was stopped (large amplitude, ~100 s each; `b15_pf_*`, `b15_pr_*` in the jsonl; default and wetting for the first three, default only for the last); they are not used. The `b15b`-`b15e` rows (the stroke-hinge family) are default only; the wetting rows are the `b15w_*` table.

**R2, clearance of the swim stroke on T5b (kinematic, 120 phases per beat; `results/S0b_T5b_clear_b2.json`, mean stroke angle = dihedral "bias" in rad, stroke 10 / 20 / 30 deg full, feather 50 deg, diagonal):**

| bias | wing-pontoon | wing-dorsal float | wing-strut |
|---|---|---|---|
| 0.00 (B11's flat wings) | **-0.006 / -0.009 / -0.015 m (the wing passes through the pontoon)** | 0.136 / 0.130 / 0.122 | 0.069 / 0.062 / 0.053 |
| 0.30 | 0.035 / 0.033 / 0.030 | 0.085 / 0.080 / 0.074 | 0.097-0.086 |
| 0.46 | 0.057 / 0.054 / 0.050 | 0.058 / 0.053 / 0.048 | 0.115-0.102 |
| 0.61 (the hover dihedral) | 0.081 / 0.077 / 0.072 | 0.033 / 0.030 / 0.026 | 0.134-0.120 |
| 0.76 | 0.107 / 0.102 / 0.096 | 0.012 / 0.011 / 0.008 | |
| 0.90 | 0.131-0.119 | -0.005 / -0.007 / -0.011 (touching) | |

So B11's swim (flat wings, bias 0 plus the pitch input) had the wings through the pontoons at every amplitude: that is the "stroke that would trim it puts the wings into the pontoons"
fact in numbers. The window with both >= 0.03 m is bias 0.30-0.58 for strokes up to 30 deg. **First choice (superseded by the pontoon change below, and by the +-0.12 / a <= 1.0 limits found at feather 65 deg): mean dihedral 0.45 rad (25.8 deg), pitch input +-0.12 rad (0.33-0.57)**, which keeps wing-pontoon >= 0.040 and wing-dorsal >= 0.035
over the whole range and amplitude (confirmed on the extremes below). The wings' own spacing (fore against hind, 0.012 m at every bias and stroke: `seg1|seg3`) is T4's hind_u = 0.11 design fact and is not changed.

### The second geometry change, forced by R2 (my addition, labelled): the pontoons 0.07 m lower (T5b = zp -0.23)

B15's clearance table above says the swim stroke's flipping trailing edge sweeps 0.15 m under the wing plane (chord 0.19 m behind
the spar x sin 50 deg = 0.146 m); with the pontoon tops at z -0.13 (axis -0.16) a flat wing passes through them (-0.006 to -0.015 m),
and raising the wings to clear them (bias 0.45) costs a large nose-down moment and upward force (B15c at 0.3 m/s, Phi 14: pitch +3.6 N m, Fz +7.5 N; flat: +1.6 N m, +2 N),
which the free swim at that setting showed as a climb with a growing nose-up pitch (`b16_v2_*`, below). The geometric fix is the pontoons' height: **T5b's pontoon axis is z -0.23 in the hull frame
(T5 and the first T5b: -0.16), top at -0.20.** Everything the first T5b rows measured with -0.16 (`b14_*`, `u01`-`u06`, `b16_v2_*`, `b15c_*`) is superseded by the rows
`b14z_*` and later; those rows stay in the jsonl, with the geometry in the row's `machine` and ids, and are used here only as the evidence for the change.
Cost and effect (model, `s0b_t5b_size.py`): posts longer, dry mass 2.115 kg (was 2.108), K roll = pitch **3.84 N m/rad** at fill 0.95
(3.13 before), 3.06 at fill 0.5, **+1.13 at fill 0** (unstable-free at every fill), CoM at the neutral fill z -0.184 (was -0.127; the thrust line is 0.18 m above it
instead of 0.13: the pitch moment per newton of thrust is 40% larger and K is 23% larger); inertia about the CoM at fill 0.95 0.234 / 0.139 / 0.259 kg m^2. Clearance of the swim stroke on the built T5b (kinematic, `results/S0b_T5b_clear_final.json`,
diagonal, mean stroke angle 0, +-0.12 rad of pitch input): stroke 20 deg, feather 50 deg (throttle a = 1): wing-pontoon **0.038** (input -0.12) / 0.061 / 0.079 m, wing-strut 0.095 / 0.114 / 0.130, wing-dorsal 0.149 / 0.130 / 0.110;
stroke 26 / feather 65 deg (a = 1.3, the B16 v2 setting): wing-pontoon **0.012 (input -0.12)** / 0.035 / 0.055. **R2 holds with the throttle limit a <= 1.0 (stroke 20 deg at 0.5 Hz) and the pitch input +-0.12 rad; at a = 1.3 it
does not (0.012 m): the law's throttle is capped at 1.0 for that reason, and it is enough (net 10 N at 0.3 m/s at stroke 20, B15).** Wing-wing (fore against hind, 0.012-0.024 m) is T4's `hind_u` design, not changed.
The hover stroke on T5b, kinematic: wing-pontoon 0.126 m, wing-dorsal 0.050 (amp 0.713) / 0.041 (0.9) at mean sweep 0, down to 0.016 at amp 0.9 with sw0 -0.3 (wing-wing 0.003, wing-hull 0.010 there are T4's, as on T5).

### B14 again on T5b (final, zp -0.23; ids `b14z_*`; both modes; 0 QACC, 0 limiter steps)

| | default | wetting |
|---|---|---|
| float: hull origin z, pitch, roll at 12 s | +0.2187 m, -0.5 deg, 0.0 | +0.2183 m, -0.4, 0.0 |
| wet main-wing / tail strips; hull slices | 0 / 0; 0 | 0 / 0; 0 |
| lowest wing corner at rest | +0.270 m | +0.270 m |
| roll +20, end after 8 s (roll, pitch, yaw) | +0.04, -0.73, +3.3 deg | -0.11, -1.26, +5.1 |
| nose down 20 / nose up 20, end | -0.53 / -0.86 | -0.45 / -1.37 |
| hover stroke held on the float (amp 0.713 / 0.9): wet strips, limiter, lowest wing point | 0 / 0, 0.144 m | 0 / 0, 0.144 m |
| vertical force (stroke + buoyancy) amp 0.713 / 0.9; still | 40.08 / 50.72 N; 19.39 N | 40.09 / 50.73; 19.39 |
| torques sweep / feather / stroke at amp 0.713 (witness 24 / 16 / 24) | 9.6 / 5.3 / 6.0 N m | 9.4 / 5.2 / 6.0 |

**R3 passes in both modes, which agree to 0.1%; R1: K = 3.84 N m/rad against the 2.9 required.** Held hover map in air (`b14z_m_*`, amp 0.70: 20.03 N = 0.966 weights of 2.115 kg; pitch +1.259 N m; roll 11.2 N m
per unit amp_lr; pitch -9.67 N m/rad of sw0; yaw +7.80 N m/rad of psi0_lr; sweep torque 9.3 N m; wing-hull 0.051): hover amp 0.713, trim sw0 +0.132 rad. Mass: dry 2.115 kg, 4.76 kg at fill 0.95.

### B15, the tail as the pitch input (ids `b15d_*`, T5b at 3 m, hull moved at 0.3 m/s, stroke 14 deg, feather 35, flat wings; default mode)

| tail deflection | Fx | Fz net | pitch moment |
|---|---|---|---|
| -0.3 rad | 6.6 N | +4.8 N | +3.05 N m |
| 0 | 7.2 | +2.6 | +1.69 |
| +0.3 | 6.7 | +0.6 | +0.21 |

**-4.9 N m/rad of tail at 0.3 m/s, 3.5x the stroke bias's 1.4 N m/rad** (it scales with the flow past the tail, so it is nothing at rest). The thrust's own nose-down moment at this throttle is
1.69 N m (thrust 7.2 N, line 0.18 m over the CoM) against K 3.84 N m/rad: 25 deg of nose-down if nothing counters it, which is what the free swims did (-27 deg at throttle 0.87).
So: the thrust-offset moment is the pitch problem under water at any throttle above the drag-balancing one, and the tail is the only pitch actuator with enough authority (stall 5 N m on the witness curve, x4 here).

## B16: free swim, heading and antipode (ids `b16_*`; T5b from rest at 3 m, fill 0.95, same start for both; `mission2` swim phase)

All rows 0 QACC, 0 limiter steps. Mean signed speed = half the sum of the speeds along the command on the heading and on its antipode (as B11). "last" = over the last whole beats of the last 8 s.
Laws (one family, `s0b_t5b_probe.SWIM` and variants; stroke-hinge stroke at 0.5 Hz, stroke 20 a deg, feather 50 a deg, diagonal; throttle a by a PI on the speed along the command, target 0.3 m/s; yaw by Psi_lr; pitch by an input; depth by the ballast PD):

| law | what changed | fwd whole run (default / wetting), m/s | where it failed |
|---|---|---|---|
| `b11law` (B11's, on the first T5b, zp -0.16) | a <= 0.6, flat wings through the pontoons, bias as pitch input | 0.074 / 0.141 (last 0.061 / 0.135) | the stroke is 12 deg at most: the tip speed (0.13 m/s) is the speed limit (B15) |
| `v2` (zp -0.16) | a <= 1.3, wings raised to dihedral 0.45 to clear the pontoons | 0.067 / 0.060 | climbs (Fz +7.5 N at 0.3 m/s) and pitches up to 54-82 deg |
| `v3` (T5b final, zp -0.23) | a <= 1.0 (R2), flat wings, bias +-0.12 as pitch input, yaw lim 0.3 | **0.197 / 0.281** (last 0.202 / 0.151) | thrust moment 1.8 N m nose-down (-27 deg, depth to -5.05 m in default); the antipode barely turns |
| v3 antipode | | -0.008 / +0.065 (last -0.006 / +0.097) | the throttle is gated by cos^2 of the heading error with a_min 0.2: at 180 deg error the stroke is 4 deg and has no yaw authority: turned 44 of 180 deg (default), 127 (wetting, overshoot to +53) |
| mean signed speed, v3 | | **0.095 / 0.173** (last 0.098 / 0.124) | target 0.3: fail, modes disagree by 1.8x |

Peak torques of v3: stroke 96 N m (the x4 stall) in default-mode rows (isolated entrainment impulses; 4x the witness stall, under the 10x flag), 73 wetting. Wing-pontoon clearance during the free swim >= 0.055 m, wing-dorsal >= 0.154 m (min over the run).

## B17: lift-off, hold, kicked hover and landing on the float (T5b final; ids `b17_float*`, `u*`)

Law: `Hover4` of B10 on T5b's own map (`law_cfg_b`: hover amp 0.713, trim sw0 +0.132) with gains **wn 6 / 6 / 3 rad/s, zeta 1.0, k_v 0.5, k_p 0.1** (T5's were 8 / 8 / 4, 0.8). Tuning (ids `u01`-`u16`, in the ledger below):

| ids | start | change | result |
|---|---|---|---|
| u01-u04 | air at 10 m, wetting, kick | T5's gains, k_v 0.3 / 0.8, k_p 0.2 | **not usable as a start**: the free fall of 2 m before the stroke is up saturates the amplitude at its 0.9 limit (roll authority 1.6x the map): roll oscillation to inversion at 3 s in all four. Later tuning starts on the float |
| u05, u06 | float +2 cm, zp -0.16, wetting / default, kick | T5's gains | both: roll 39 deg at 14 s, drift 14.7 / 1.7 m |
| u07, u08 | float +1.8 cm, T5b final, no kick, wetting / default | T5's gains | **roll grows from 0.3 deg with no kick** (e-fold ~1.2 s): 55-62 deg at 5 s in both modes (the loop is unstable on this body at wn 8, in both modes; T5's default row hid it by exact mirror symmetry) |
| u09, u10 | same, wetting, no kick | wn 6/6/3 zeta 1.0 | **holds 10 s, roll 0.0, drift 0.46 m, tilt 15 deg**; k_v 0.8 / k_p 0.2 (u10): inverted at 13.5 s |
| u11, u12 | same, kick, wetting / default | wn 6/6/3, zeta 1.0 | both hold 10 s: roll within 5.3 deg (wetting, creeping -1.9 -> -5.3 deg over 9 s) and 1.8 deg (default) |
| u13, u14 | u13 became a surfacing row (below); u14 was not run | | |
| u15, u16 | float +1.8 cm, no kick, both modes | descent 0.15 m/s (was 0.25), stroke ramped out over 1.5 s (was 0.3-0.4) | lands upright (pitch -1.0 / -1.1 deg after 5 s), 0 wet wing strips |

Reported rows, from the settled float (hull z 0.2185 m), settle 3 s, lift-off, 2 m, 10 s hold, descent at 0.15 m/s, stroke ramped out over 1.5 s, 5 s after; ids `b17_float2_*` (and `b17_float_*` for the first landing law):

| | kick, default | kick, wetting |
|---|---|---|
| hold: z over the start, drift, tilt max | 1.80-2.02 m, 0.56 m, 14.4 deg | 1.81-2.03, 0.46 m, 17.0 deg |
| roll envelope per 2 s after the kick (max roll deg) | 8.4, 4.7, 3.0, 2.2, 1.6, 1.2 | 9.6, 7.7, 6.9, 6.6, 6.4, 6.2 |
| touch-down (beat-mean vz) | -0.011 m/s | -0.010 m/s |
| after 5 s: roll, pitch | 0.0, -0.9 deg | 0.0, -1.1 deg |
| wet wing strips, limiter steps, QACC, wing contact | 0, 0, 0, none | 0, 0, 0, none |
| peak torque sweep / feather / stroke N m (witness stall 24 / 16 / 24) | 12 / 5 / 5 | 12 / 5 / 6 |
| X2 share | 0.991 | 0.991 |

**Reading, for the criterion of D1: the hover holds the 10 s window, lands upright in both modes with the kick, and the turn-over of B10 is gone; but not "the modes agree".** Default recovers to 1.2 deg; wetting
decays from 9.6 to **6.2 deg and stays there (a plateau, not a recovery; the air tuning row u11 with the same gains grew -1.9 -> -5.3 deg over 9 s)**, i.e. marginal: it meets the window, not stability. The two modes differ by 5x in
the residual roll. With the first landing law (descent 0.25 m/s, ramp-out 0.4 s: `b17_float_kick_*`): default ends pitched -71 deg nose down, wetting inverts at 29.6 s, though the float itself takes 20 deg pushes (B14) and
static drops from 0.08 and 0.18 m with 12 deg of pitch (`d3_drop_*`) recover to level; the slow landing (cause **not isolated**: the stroke ramp-out against the surface at 0.25 m/s) fixes it.

### B15 in both modes (ids `b15w_*` = wetting; the `b15b`-`b15e` rows above are default; T5b at 3 m; stroke-hinge stroke, flat wings, 0.5 Hz)

| quantity | default | wetting |
|---|---|---|
| net Fx at U = 0.3 m/s, stroke 14 deg | 7.2 N | 9.8 N (+36%) |
| pitch moment at U = 0.3, tail -0.3 / 0 / +0.3 rad | 3.05 / 1.69 / 0.21 N m (-4.9 N m/rad) | 3.52 / 2.17 / 0.65 N m (-4.8 N m/rad) |
| thrust at rest, stroke 10 deg, feather 25 deg (throttle a = 0.5) | 10.50 N, pitch +1.99 | 10.59 N, +2.00 |
| yaw moment at rest from Psi_lr 0.3 / amp_lr 0.8 | -3.07 / -1.05 N m | -2.95 / -1.07 N m |

Tail and yaw authority agree to 5%, the thrust at speed differs by 36%: **the tail and the yaw inputs are established, the net thrust at 0.3 m/s is not (modes differ)**; the B16 speeds inherit that.

### The reverse thrust, and what it costs (my addition: the feather hinge's range from -150..+60 to -250..+60 deg; ids `b15g_*`, `b15w_flip_*`; geom `feather_range`)

The thrust of this stroke is toward the leading edge whatever the phase (B11). Turning every wing over on its spar (feather offset -180 deg, which the old range cannot reach) puts the trailing edge first:

| held, stroke 14 deg, feather 35 deg | U = 0 | U = +0.3 m/s | U = -0.3 m/s | pitch moment at U = 0 |
|---|---|---|---|---|
| normal, default | +21.7 N | +7.2 N (net of drag) | | +4.10 N m |
| normal, wetting | +21.3 N | +9.8 N | | +4.02 N m |
| **flipped, default** | **-11.2 N** | -21.2 N | +40.2 N (not understood) | -2.83 N m |
| **flipped, wetting** | **-9.4 N** | -19.2 N | +12.3 N | -1.65 N m |

**Reverse thrust exists, in both modes (-9.4 / -11.2 N at the throttle where forward gives +21), at about half the forward thrust, with the pitch moment reversed** (nose-up). The held row at U = -0.3 disagrees between the modes (+40 / +12 N),
so the reverse thrust against an inflow is not established. In default the feather torque at rest reads 64 N m (the x4 stall: 4x the witness) while the wing is flipped under the stroke. Cost: the feather hinge turns 180 deg between the forward and the reverse pose (6 s in the chain). The antipode is then a reverse swim, no turn.

### Why the turn fails (ids `b16_v4_*`, `b16_v5_*`): a weathercock, not an actuator

Held, a = 0.5, Psi_lr 0.3 gives 3.0 N m of yaw at rest (B15, both modes). Free, with Psi_lr saturated (and a_turn 0.8 with amp_lr coupled in v5) the machine turns 0.7 deg/s (v4 default), or yaws +-8 deg and
returns while it swims away along its old heading at 0.25 m/s (v5 default): -11 m along the antipode in 45 s, yaw error -180 deg at the end. The thrust that makes the yaw moment (a left-right difference of a forward thrust) also
accelerates the machine to 0.25-0.5 m/s in a fraction of a second, and at that speed the long hull, pontoons and struts give a yaw restoring stiffness (about 20 N m/rad, from the 3 N m / 8 deg)
that the same differential cannot overcome. Same pattern as B11's thrust (held 13-21x drag, free a tenth): a held map at rest does not predict the free body at speed. A turn needs a thrust that is not forward; the reversed wing is that (above).

### B16, the reported rows (ids `b16_final_*`): heading and its antipode from the same fresh start (CoM at 2.5 m, hull heading 30 deg; tuning rows were at 3 m, heading 0)

Heading: the forward law of `v4` (tail as the pitch input with k -1.0 rad/rad, kd -0.3; pitch reference from the depth error, k 0.2 rad/m, lim 0.15; Psi_lr yaw, lim 0.3; throttle a 0.45-1.0; target 0.3 m/s; `mission2` swim, 30 s).
Antipode: **reverse swim**, the same law with the wings turned over (`pose` "flipped", 6 s, then swim 40 s), the loops' signs flipped (`v6`), no turn. Ids: `b16_final_fwd_{default,wetting}`, `b16_final_rev_{default,wetting}`. 0 QACC, 0 limiter steps; the motor on its curve 0-2% of steps; peak torque
at most 4.1x the witness stall (feather 65, stroke 86, sweep 96 N m in the reverse default row: the x4 stall, entrainment impulses), under the 10x flag; wing-pontoon clearance during the runs >= 0.069 m, wing-dorsal >= 0.135 m.

| | default | wetting |
|---|---|---|
| heading: speed along the command, whole run 30 s / last 8 s | **+0.250 / +0.264 m/s** | +0.112 / **-0.243** |
| heading: yaw error at the end, depth range, pitch range | +4 deg, -2.50..-2.00 m, -4..+3 deg | **-152 deg**, -2.52..-1.57 m, -3..+10 |
| antipode (reverse): whole run 40 s / last 8 s | +0.078 / +0.096 | +0.112 / -0.166 |
| antipode: yaw error at the end, depth range, pitch range | +14 deg, -2.56..-2.12 m, **-42 (right after the flip) settling to -10 deg** | **-179 deg**, -2.74..-2.56 m, -10..+13 |
| **mean signed speed, whole run / last 8 s** | **+0.164 / +0.180 m/s** | **+0.112 / -0.20 m/s** |
| still machine (B18) | see B18 | |

**Fail against 0.3 m/s in both modes, and the modes disagree (default holds its heading and makes 0.25 m/s forward; wetting makes 0.11 and loses its heading by 150-180 deg in both directions, which makes the last-window speed negative). Not established.**
The best single number is the default forward swim, 0.25 m/s whole run and 0.26 m/s in the last window (83-88% of the target) with heading held to 4 deg.

What was tuned and what limits it (ids `b16_*`; the rows `v2`-`v6` and the turn rows were run at 3 m / heading 0 and are tuning, listed in the ledger):
- B11's swim law gave 0.07 / 0.14 m/s on T5b too: a stroke of 12 deg at 0.5 Hz has a tip speed of 0.13 m/s, below the swim speed (B15: net force at 0.3 m/s goes through zero at a stroke of ~13-14 deg). Throttle to 1.0 (stroke 20 deg, the R2 limit) gives 0.20-0.28 m/s.
- The thrust's nose-down moment (10 N, 0.18 m over the CoM, 1.8 N m, against K 3.84) tilted the machine 27 deg and drove it to the sea floor in v3 (default: -5.05 m, fill 0.21); the tail as the pitch input (4.9 N m/rad at 0.3 m/s) holds pitch to -4..+3 deg in the default forward row; at rest nothing covers it.
- Depth: the depth loop is the ballast alone (1.36 N, 0.1 per s) plus a pitch reference from the depth error; it did not hold depth in v3 (-5.05 m in default, fill 0.21), v4 antipode (wetting, -0.16 m, fill 1.00) and the reverse swims (range 0.4-0.5 m); in the best row (default forward) it drifts 0.5 m up in 30 s. Not met.
- A turn does not work as the antipode (weathercock, above). The reverse swim does turn the antipode into a straight swim, but in default it starts the swim at -42 deg of pitch (the flip pose's transient: the figure shows it) and settles at -10 deg nose-down, and the wetting row spins the heading: not tuned beyond one law (v6), 2 rollouts of 80.

### B17 surfacing: not met (ids `b17_surf_*`, `u13`, `u17`-`u20`; start deep, wings in the sink pose, hinge gains re-read at the lift-off)

K is positive at every fill on T5b (B14 sizing), but the machine does not come up upright in any of five recipes:

| recipe | start | arrival at the surface | after |
|---|---|---|---|
| blow at 0.3/s from the start (`b17_surf_d3_seed_*`, the reported start 3 m, 2% seed, both modes) | 3.0 m | default: roll 179, pitch 67, yaw 158 deg at 13.2 s; wetting: roll -169, pitch 64 at 11.6 s | stays inverted; lift-off fails (inverted at 20 s) |
| blow at 0.1/s (`u13`, default, 2.0 m, 1% seed) | 2.0 m | roll -180, pitch 51 at 10.6 s | inverted |
| rise at fill 0.94 to 0.35 m under, then blow 0.3/s (T5's recipe, `u17`, default, 2.0 m, 1% seed) | 2.0 m | roll -4, pitch +32 deg at 3.1 s after the blow starts | pitch grows to 57 deg in the 6 s on the float; the lift-off inverts (121 s) |
| blow at 0.03/s and 0.1/s from 0.6 m (`u18`, `u19`, default; `u20`, wetting) | 0.6 m | rises to z -0.35 and **stays there while the fill drops from 0.80 to 0.44** (12 s with vz = 0), then arrives rolled -45 deg (u18, u19) / -46 (u20) | stays at roll -45; settles at roll -65, pitch +71 deg (default) |

Two facts in these rows. (1) The plateau at -0.35 m with the buoyancy rising is the same kind of stall as D2 (b): net vertical force going to zero at a low speed with the wings in the sink pose; whether it is `InducedFlow.w` here was not tested
(`reset_parts` was not used in these rows); it is the first reason the slow blow does not help. (2) Once at the surface with the tanks partly full (fill 0.2-0.4) the pontoons sit just under the surface and the waterplane is the hull and the dorsal float:
the machine heels to 45 deg and stays; with the tanks empty (B14) it floats level and returns from 20 deg. T5's recipe crosses that range fast; on T5b it arrives pitched +32 deg, and +32 is outside what the float
returns from (B14 pushed 20 deg; 20 deg returns, 32 deg went to 57). The pitch range of the float's positive stability, not tested between 20 and 32 deg, is the open number.
Starts and seeds were varied only by the three listed here: this part does **not** show a surfacing; T5's one working recipe did not transfer.

B17 without the kick (ids `b17_float2_nokick_*`, same law, landing at 0.15 m/s, ramp-out 1.5 s): both modes hold 10 s at 1.80-2.02 / 1.81-2.03 m (drift 0.56 / 0.46 m, tilt 13.5 / 15.1 deg), roll 0.0 throughout (mirror-symmetric), land at -0.011 / -0.011 m/s,
upright after 5 s (roll 0.0, pitch -0.9 / -1.1), 0 wet wing strips, 0 limiter steps, X2 share 0.991: **pass in both modes, which agree**.

**The depth-holding turn (`b16_turn90_*`, 2.5 m, heading 30 deg, 90 deg commanded, a_turn 0.8, yaw by Psi_lr and amp_lr): fails in both modes**: yaw error at 30 s +136 deg (default) and -168 (wetting) (it does not turn, or turns the wrong way and hunts),
depth range -2.57..-1.61 m (default) and -2.75..-0.07 m (wetting; roll to 51 deg, pitch -31..+27).

## The chain on T5b (ids `b17_chain_default`, `b17_chain_wetting`; `figures/S0b_T5b_mission.png`)

float 3 s > lift-off, 2 m, 10 s hold, descent at 0.15 m/s, landing > flood to 3 m (`pose` sink over 8 s; **`InducedFlow` state cleared at the start of the flood and of the rise**, hinge gains re-read at the swim and the second lift-off) > settle 5 s > wings flat > swim forward 20 s (the `v4` law) > sink pose >
rise at fill 0.94 to 0.35 m under (140 s limit) > blow at 0.3/s > raise the wings, settle 4 s > second lift-off. No tank seed, mirror-symmetric start. 0 QACC; the limiter steps (1,537 / 11,999) are all in the last phase, after the machine is inverted: that phase is not evidence, the others are (0 steps).

| phase | default | wetting |
|---|---|---|
| lift-off, hold, kick-free landing | upright, pitch 0 | upright, pitch -1 |
| flood to 3 m (CoM -2.9 m) | **39.3 s** (D2 b: the stall of `chain_default` is gone with the inflow reset) | **38.3 s** |
| swim forward 20 s: speed along the command, yaw error at the end, depth range | **0.256 m/s, +3 deg, -2.84..-2.35 m**; dx 4.62 m | 0.147 m/s, **-83 deg**, -2.97..-2.51; dx 5.39 m |
| peak torque in the swim stroke / sweep / feather N m | 89 / 96 / 64 (x4 stall) | 89 / 47 / 25 |
| rise at fill 0.94 | -0.41 m after the phase's 140 s limit | -0.43 m |
| blow 3.1 s: attitude on arrival | roll 0, pitch **+31 deg** | roll -4, pitch **+33 deg** |
| settle on the float 6 s | pitch **+54 deg** | pitch **+73 deg**, yaw 98 |
| second lift-off | inverted at 1.1 s (257 s) | inverted at 1.9 s (257 s) |

The chain reaches the surface, not the second lift-off: it fails exactly where B17 surfacing fails (above), in both modes. Every step before it, including the flood that stalled `chain_default`, runs in both modes.

## B13, the dt/2 checks (the water scenarios that passed; ids `b14z_float_dt2_*`, `b17_float2_kick_dt2_*`, `b17_flood_dt2_*`)

Float at dt 5e-5 (`b14z_float_dt2_*`): hull z 0.2187 / 0.2183 m (dt 1e-4: 0.2187 / 0.2183), pitch -0.5 / -0.4 deg (the same), 0 wet strips, 0 limiter steps; the 20 deg pushes end at roll +0.03 / -0.14, +20 nose down -0.49 / -0.34, nose up -0.79 / -1.24 deg
(dt 1e-4: +0.04 / -0.11, -0.53 / -0.45, -0.86 / -1.37): **the same to within 0.15 deg.**
Flood at dt 5e-5, from the float with no flight (`b17_flood_dt2_*`): 3 m (CoM -2.9 m) in **39.9 s (default), 38.2 s (wetting)**, pitch -0.7 deg on arrival, 0 limiter, 0 QACC; the chain's flood at dt 1e-4 took 39.3 / 38.3 s (after a flight, inflow state cleared): the same within 1.5%.
Kicked hover, landing, at dt 5e-5 (`b17_float2_kick_dt2_*`): the same outcome in both modes: hold 1.80-2.02 / 1.80-2.03 m, drift 0.559 / 0.461 m (dt 1e-4: 0.557 / 0.456), tilt 14.4 / 17.1 deg, roll envelope 8.4 -> 1.2 deg (default) and 9.6 -> 6.1 deg (wetting), landed upright (pitch -0.9 / -1.1), 0 limiter, 0 QACC.
The swims and the surfacing did not pass, so they have no dt/2 row.

## B18, the still machine (ids `b18_*`, `b14z_still_*`)

| scenario | result, both modes |
|---|---|
| swim at 2.5 m, 30 s, every actuator held at the start pose (`b18_swim_still_*`) | moves 0.000 m on every axis, attitude unchanged: **a still machine does not swim on either heading** (mean signed speed 0) |
| lift-off, hold, landing on the float (`b18_float_still_*`, 14 s) | stays: CoM z +0.002 m, pitch -0.45 / -0.30 deg, 0 contact, 0 limiter steps; peak torque 0.11 N m |
| held in air, wings raised; held over the float (`b14z_still_*`) | 0.001 N; 19.39 N = the pontoons' buoyancy, 0 wet strips |
| held at 3 m (`b15_still_*`) | not run (the level-family grid that held it was stopped); the free swim still row covers it |
| flood and surfacing | not run still (the ballast alone brings a still machine down and up: that is the flood and the rise themselves) |

## What limits T5b

1. **Surfacing (blocking the chain).** Five recipes arrive inverted, heeled 45 deg, or pitched +31 deg; the float returns from 20 deg of pitch but not from +31 (B14 tested 20). The numbers are above; the cause (the waterplane of a pontoon pair that is just under the surface at partial fill; the stall at -0.35 m) is partly inferred.
2. **Swimming: 0.25 m/s forward in one mode, 0.11 in the other; the antipode is a reverse swim at 0.08-0.11 m/s; mean signed speed 0.16 / 0.11 against 0.3.** The thrust at speed differs by 36% between the modes (B15), the reverse thrust is half the forward, a turn is blocked by a
   weathercock restoring moment of the long hull, and the reverse law was not tuned. Pitch under water is limited by the thrust line 0.18 m over the CoM: K 3.84 holds 9 deg at the drag-balancing throttle and nothing but the tail (4.9 N m/rad at 0.3 m/s) holds it above.
3. **Depth authority (1.36 N, 0.1 per s) is unchanged and does not hold depth under a swim** (ranges of 0.5 to 2.7 m across the rows above); the wings' heave was not found (B15: the level family gave +20-38 N of upward force with the paddle kinematics).
4. **The hover is marginal in the mode without the air added-mass reaction**: the lateral loop is unstable at T5's gains (wn 8) in both modes on this body and holds at wn 6 / zeta 1.0; the wetting row keeps a 6 deg roll offset after the kick (not converged).
5. **R1, the target from T5's thrust, is 4x below T5b's swim operating point** (10 N of thrust at a = 0.87, 1.7 N m of nose-down moment, against the 0.54 N m R1 was sized from). R1 is met as written (K 3.84 against 2.9); it is not enough at speed, and the tail is the part that covers it.
6. **The lift margin falls from 2.4 to 1.5 weights** (dry mass +58%); the hover amplitude is 0.713 of the law's 0.9 limit.
7. Unchanged and stated: every flying row is in the X2 regime (share 0.69-0.99; the failed lift-offs 0.69-0.75), on a leading-edge-vortex branch fitted at Re 136; no wing-wing or wing-pontoon aerodynamic interference; the solver's `InducedFlow` state carries across media (D2 b) and the hinge gains are medium-dependent (D2 a): both are harness rules now.

## Flags, ledger, what was not done

**Rows**: `results/S0b_probes4.jsonl`, 181 rows (81 free `mission` rollouts, 91 held rows (51 under water `held_w2`, 25 `held` in air and over the float, 15 in air for D1b), 9 float rows; 37 rows on T5, 144 on T5b). **QACC resets: 0 on every row.**
Rows with limiter steps (not evidence): the 2 Hz stroke rows of B15 (`b15_b2_f2.0_*`, 96-706 steps), and the final failing phases of the surfacing and chain rows (`b17_surf_d3_seed_wetting`, `u17`, `b17_surfR2_d25_*`, `b17_chain_*`: the machine is inverted and flying). Rows over 10x the witness stall: none.
Rows over the witness stall but under 10x: the default-mode swims and reverse swims (stroke / sweep 86-96 N m = the x4 stall, 3.6-4.0x; at limit 0-2% of steps), the chain's swim; the surfaced lift-offs read 9.4 N m with the gains re-read. X2: every flying row 0.69-0.99.
Mode agreement: float (B14), the stroke over the float, lift-off and landing (without kick: agree; with kick: both hold, residual roll 1.2 against 6.2 deg), tail and yaw authority (5%), the flood (39.3 / 38.3 s) agree; **net thrust at 0.3 m/s (36%), the free swim (0.25 against 0.11 m/s forward),
the reverse swim, the reverse thrust against an inflow (+40 / +12 N), the turn and the heading hold in wetting: do not.**

**Tuning ledger (free rollouts; 80 allowed, read as fresh for this part)**: lift-off and landing `u01`-`u12`, `u15`, `u16` (14); landing drops `d3_drop_*` (3); surfacing `u13`, `u17`-`u20` and the first reported recipe `b17_surf_d3_seed_*` (2) (7); swim law iterations on T5b: `b16_b11law_fwd_*` (2), `b16_v2_fwd_*` (2), `b16_v3_*` (4), `b16_v4_anti_*` (2),
`b16_v5_anti_*` (2), `b16_v6_rev_*` (2), `b16_turn90_*` (2) (16). **40 of 80.** Held maps (`b15*`, `b14*`) are measurements, not counted. Reported-row starts differ from the tuning starts as follows: lift-off from the settled float (tuning +1.8 / +2 cm above it, or from air); swim from 2.5 m, heading 30 deg (tuning 3 m, heading 0; the final rows
reproduce the tuning rows' numbers to the digit for the reverse swim in wetting: the start's depth and heading do not change this dynamics, so the start is fresh in name only); surfacing: `b17_surfR2_d25_*` from 2.5 m with no seed (tuning 2.0 m, 1% seed; the 3 m, 2%-seed start was used for the first, failed recipe and is not a fresh start for any later one).

**Not done**: a surfacing that works; a swim at 0.3 m/s in both modes; the depth-holding turn; tuning the reverse law (one law, 2 rollouts); a seeded (2%) surfacing with a recipe that arrives upright, and a second start for it; the beach row (lift-off from the sand: not asked for T5b and not run; the pontoons are lower now: the hull is 0.07 m higher on the ground);
the mission's pitch range of the float beyond 20 deg (needed by the surfacing); the physical cause of the stall at -0.35 m in `u18`-`u20` (tested for `InducedFlow` only in the flood, D2 b); why the landing at 0.25 m/s capsizes while 0.15 m/s with a slow ramp-out does not (cause not isolated);
`b15_still_*`; a dt/2 of the swim rows (they did not pass); the level-hover family as a thrust vector (paddle kinematics gave only forward thrust and a large upward force).

**Measured against inferred.** Measured: every number in a table above that carries an id; the stall's cause (the `InducedFlow` state) by clearing it alone; the stale gains (kp ratio 80.5x) and their 105 N m by re-reading them; K and the masses on the built model; the clearances kinematically on the built model; the thrust/drag/yaw/pitch maps held.
Inferred: why `InducedFlow.w` holds the sinking machine (not logged); the weathercock explanation of the turn (the 20 N m/rad is read off one held and one free row); that the surfacing arrival pitch comes from the blow and the water plane at partial fill; that the roll-from-air-reaction term explains the D1 split (measured as derivatives on T5, and the split reproduces in air; the mechanism of growth is not isolated further).

**Hash check (nothing in T4, T4c or T5 moved).** `s0b_t5b_hash.py` (sha256 over the model arrays, the panels and 200 steps) before and after this part, identical: T4 `6070e5a5353cace3`, T4c `281eed84715ae7df`, T5 `f8e72992cbafc736`, T5 at fill 0.95 `6f735d284395364f`
(`results/t5b_hash_before.txt`, `t5b_hash_after.txt`). sha256 of the untouched files, unchanged and `git status` clean for them: `s0b_machine.py` 49cf256b..., `s0b_t5_machine.py` b5ef075f..., `s0b_laws.py` 90b91abd..., `s0b_probe2.py` bbebfa6d..., `mission.py` bee12386..., `s0b_t5_probe.py` d5d76f6e..., `s0b_t5_draw.py` d49b586b..., `results/S0b_T5_calib.json` bae5660a....

**Figures** (opened and read against the rows): `figures/S0b_T5b_built.png` (+ `.json`; side, top, front, CoM and CoB, the waterline at the float, dry 2.115 kg, 4.758 kg at fill 0.95, K 3.82 N m/rad from the built model);
`figures/S0b_T5b_water.png` (the four `b16_final_*` swims: displacement, speed, depth, yaw error, attitude; title mean signed speed +0.164 / +0.180 default, +0.112 / -0.205 wetting, the table's numbers);
`figures/S0b_T5b_mission.png` (`b17_chain_default`: flood, swim, rise, surfacing arrival, the failed second lift-off; the drawn bodies are at a rest pose at the recorded height and attitude).

# B19-B23: surfacing, swim speed, depth-holding turn, still machine, whole chain on T5b

2026-11-21. Rows: `results/S0b_probes5.jsonl` (every free rollout one row; ids `b19_*`, `b20_*` ...; held measurements are not rollouts and are
not counted in the 80). Harness: `s0b_t5c_surf.py` (new; T5b, `s0b_t5b_machine.py`, `s0b_t5b_probe.py` and everything earlier are not edited).
Tuning ledger for B19+B20+B21 (80 allowed), rollouts: see the end of the part.

## B19, step 1: why the machine capsizes on the way up (ids `b19_diag_*`; start 3.0 m, 2% tank seed, wings in the sink pose, default and wetting; 0 QACC, 0 limiter steps)

The four diagnostic rollouts (blow at 0.3/s default, wetting, no seed default, blow at 0.1/s default) all arrive inverted, as B17's five recipes did. The traces
(0.05 s) say the capsize is **a pitch divergence while the machine is fully submerged and rising, not a surface or partial-fill effect**:

- noseed default (exact mirror symmetry, so the seed is not the cause): from rest at -3.0 m the nose-up pitch grows from 0.5 deg at 0.45 s to 5.4 deg at 1.05 s,
  20.5 deg at 1.85 s, 44.5 deg at 2.65 s, 89.8 deg at 3.65 s (z -2.0 m, still wet: pontoon, dorsal float, hull, wings and tail strips all at wet fraction 1.0;
  the dorsal float's top at -2.27 m). The fill reaches 0 at 3.2 s (z -2.13 m). The roll angle in the table is an Euler artefact once pitch passes 78 deg: the machine ends nose-over-the-top
  (roll 180, pitch 67 = pitch 113).
- The fluid moment about the dry centre of mass is nose-up and grows with the pitch: -0.33 N m at 16 deg, -1.08 N m at 37.5 deg, -1.40 N m at 61 deg (a stiffness of about -1.2 N m/rad),
  with the vertical force falling from 46.7 N (weight) as the tanks empty and the climb speed 0.23 m/s. The effective centre of vertical force is 0.04-0.05 m ahead of the CoM at 37-60 deg.
  The static righting stiffness K of the same machine is **positive at every fill** (`righting`, wings sink pose, 5 m: 3.83 N m/rad at fill 0.95, 3.06 at 0.5, 2.14 at 0.2, 1.13 at 0). So a static
  argument (the one B17 made: pontoons just under the surface at partial fill) does not explain it.

**Measured mechanism (held, steady flow; `/scratchpad` k3/k4 method: the hull pose and velocity rewritten every step, 3000 steps, the last 500 averaged; moment of every body's fluid force about the dry CoM; pitch or
roll tilt +-0.05 rad; stiffness N m/rad, positive = restoring; fill 0.5, default mode).** (A first version of this probe that called `apply` three times read the
unsteady start-up state of the tail strips (-1.2 N m each); the stepped average is the one quoted.)

| wing pose (dihedral, feather) | vz m/s | pitch K total | of which wings | roll K total | of which wings |
|---|---|---|---|---|---|
| sink (35 deg, -90 deg: raised, chord vertical) | 0 | 2.46 | -0.31 | 3.58 | 0.80 |
| sink | **0.25** | **-14.48** | **-17.34** | 2.68 | 0.20 |
| flat (0, 0) | 0 | 3.04 | -0.04 | 3.25 | 0.14 |
| flat | 0.25 | **+4.92** | +1.73 | 2.93 | 0.14 |
| 35 deg, feather 0 | 0.25 | 4.14 | 1.34 | **-26.19** | -28.61 |
| 0, -90 deg | 0.25 | -4.86 | -8.12 | 2.96 | 0.11 |
| 35 deg, -45 deg | 0.25 | -1.85 | -4.61 | -11.87 | -14.33 |

(fill 0.0 the same pattern: sink pitch -14.35, flat +3.23, roll of 35/feather 0 -26.63.) The hull's own contribution is +4.5 and the pontoons' -1.7 in every row; **the wings carry the whole change.**
Reading: with the chord vertical (the sink pose: the pose that clears the pontoons when the wing flips) a wing is edge-on to a vertical flow; a nose-up tilt gives it an angle of attack,
the lift is a horizontal force acting 0.3-0.4 m above the CoM (the wings are on top of the hull, the CoM is low in the pontoons), and the moment is nose-up and quadratic in speed. A flat wing is a plate across the flow: the tilted
normal force acts the other way (restoring). This is the pendulum-with-a-fin problem: **the sink pose is a destabilising fin in any vertical climb, and the machine's static K (3.83) is the number that does not matter.**
It also bounds the sink pose's safe speed (the next table). (B17's reading of the fast-blow rows as a partial-fill waterplane effect was wrong for them; the slow-blow rows u18-u20, which stalled at -0.35 m and then heeled to
-45 deg, are a separate case and are not explained by this.)

### Safe speed of the sink pose (held, same method, fill 0.5, default; pitch K total / of which wings, N m/rad; `results/b19/k4_f05.log`)

| vz m/s | sink pose | flat wings |
|---|---|---|
| 0 | 2.46 / -0.31 | 3.04 / -0.04 |
| 0.05 | 2.34 / -0.44 | 3.32 / +0.20 |
| 0.10 | 0.20 / -2.58 | 3.52 / +0.39 |
| 0.15 | -3.33 / -6.13 | 3.85 / +0.71 |
| 0.25 | -14.48 / -17.34 | 4.92 / +1.73 |

The fin term grows as v^2 (-0.4, -2.9, -6.8, -17 at 0.05, 0.10, 0.15, 0.25 relative to rest): the sink pose is statically safe below about 0.08 m/s of vertical speed
(K = 1 N m/rad at 0.09), which is why the flood at 0.06-0.08 m/s (B11, B17 chain) stayed upright. The rises of B17 reached 0.2 m/s.

### B19 step 2: flat wings (ids `b19_t1_*`, `b19_t2_*`): the fin is gone and the machine still pitches up, for a second reason

Flat wings (dihedral 0, feather 0, K positive at every speed in the table above) do not cure it: `b19_t1_flat_blow03_{default,wetting}` and `b19_t1_flat_blow01_default`
(same 3.0 m start) arrive at pitch 68 / 69 / 59 deg and the lift-off inverts (they are tuning rows, `not_evidence` where inverted). The traces (pitch 0.4 deg at 0.8 s, 5.7 deg at 1.7 s,
28 deg at 3.05 s at 0.03-0.08 m/s of climb) rule the fin out. What is different about a flat wing is **the water it carries**: the solver's entrained mass (body_mass less dry mass,
the added mass of a plate moving along its normal) is **125 kg for flat wings** (29.4 kg on each main wing, 2.2 kg on each tail surface) against 15.6 kg in the sink pose, on a 2.1-4.8 kg machine
(`/scratchpad` d2: dry CoM hull x 0.235 m; **centroid of the entrained mass x 0.112 m**, 0.123 m behind the buoyancy and weight lines; sweeping the four wings forward moves it, 0.216 m at +0.3 rad, 0.005 m at -0.3 rad).
Three things follow, all measured:

1. The vertical acceleration is small (net force 26 N on about 130 kg effective: 0.2 m/s^2; terminal climb speed about 0.135 m/s, where the wings' drag balances the net buoyancy), so the rise takes 8-10 s.
2. While the net buoyancy is not yet balanced by drag, the net force acts 0.12 m ahead of the inertial centre of the entrained water, a nose-up torque of about F_net x 0.12 m (26 N: 3 N m) on a very large
   effective pitch inertia: pitch builds with the CoM-line stiffness K = 3 N m/rad unable to hold it.
3. At terminal speed the same fact is a couple: the wings' drag (about 20 N) acts at their centroid, not at the buoyancy line (held, vz 0.135, fill 0, pitch 0, **moment about the dry CoM**, + = nose down;
   `results/b19/k5_*.log`):

| mean sweep of all four wings (rad, tips to the nose +) | centroid of entrained mass, hull x m | moment, flat wings | of which wings / tail / pontoons | moment at rest (vz 0) |
|---|---|---|---|---|
| 0.00 | 0.107 | **-4.09** | -2.90 / -1.03 / 0.00 | 0.00 |
| 0.35 | 0.234 | -1.55 | -0.74 / -1.05 / +0.24 | +0.30 |
| 0.50 | 0.286 | -0.45 | +0.20 / -1.05 / +0.33 | +0.44 |
| 0.00, dihedral 0.3 | 0.106 | -3.71 | -2.52 / -1.03 / 0 | |
| 0.35, dihedral 0.3 | 0.226 | -1.16 | -0.34 / -1.05 / +0.23 | |
| 0.50, dihedral 0.3 | 0.276 | -0.07 | +0.60 / -1.05 / +0.32 | |

(the pontoons' term is the dry CoM moving forward with the wings' mass, a static trim of +0.3-0.4 N m at rest; the tail's -1.05 N m is its drag at 0.135 m/s with its 2.8 kg of entrained water each, x -0.5 m.)
The sweep is therefore a pitch actuator with a large authority while climbing (7.3 N m per rad of mean sweep at 0.135 m/s; 0.85 N m/rad at rest). Static K (3.04, flat) is not the number that decides the rise either.
Sweep 0.35 alone (b19_t2_flat_sw35_*): pitch stays under 3.5 deg for the first 3 s (fill emptying), then drifts to 36-40 deg by arrival at terminal speed (default); the wetting-mode row arrived at pitch 42 and, after
settling on the float, lifted to 2 m and held 5 s (`b19_t2_flat_sw35_wetting`: hold 1.80-2.02 m, tilt 15.4 deg); the default-mode rows at sw 0.25, 0.35, 0.45 all inverted at the lift-off (arrival pitch 49, 35, 27 deg).

### B19 step 3: the sweep as a pitch actuator, and the partial-fill window at the surface (ids `b19_t3_*` ... `b19_t6_*`; tuning rows, 3.0 m and 1.5 m starts, 2% tank seed)

Wings flat (dihedral 0, feather 0), mean sweep of all four wings commanded by a pitch loop while climbing: sw = sw0 + kp x pitch(nose-up, rad) + kd x pitch rate (nose-up), clipped to 0..0.9 rad,
rate-limited to 0.5 rad/s (`pitch_loop` in `s0b_t5c_surf.py`; the hinge is the sweep hinge of the shoulder, the loop reads the hull attitude: no force is typed, the actuator is a hinge on its motor).
Blow from the start (fill 0 at 0.3/s), 3.0 m:

| id | sw0, kp, kd | rise time | pitch during the rise | arrival (first sample afloat) | settled 6 s later | lift-off to 2 m, hold 5 s |
|---|---|---|---|---|---|---|
| `b19_t3_A_default` | 0.45, 1, 2 | 53 s | within 2 deg | pitch -33, roll -5 | 0.0 / -0.1 | held 1.80-2.02 m, tilt 13.5 deg |
| `b19_t3_A_wetting` | same | 19 s (wetting climbs 2.8x faster) | | -22 roll, +11.5 pitch | 0.1 / -0.3 | held, tilt 15.3 |
| `b19_t3_B_default` | 0.55, 1, 2 | 50 s | | pitch -42 | 20 / -69 | inverted |
| `b19_t3_C_default` | 0.45, 2, 3 | 45 s | within 1.1 deg (trace: -0.2..-1.1) | pitch -17, roll -6 | -0.1 / +0.9 | held, tilt 13.6 |

With the loop the climb is straight (pitch under 1.5 deg for 40 s at 0.07 m/s) and the fin and the entrained-mass offset are both out of the picture. **What is left is the exit**: when the flat wings
cross the surface (0.4-0.5 s) their 125 kg of entrained water vanishes from the body at once and the machine, still at a net buoyancy of 26 N, goes from 0.06 to 0.48 m/s in 0.5 s (a 0.28 m jump over the float height;
the solver keeps the velocity when the mass goes), the pontoons leave the water at that speed, and pitch swings -15 deg (V4/C from 1.5 m: peak |pitch| 15.1, |roll| 4.4; the two runs with the loop frozen at -0.6 m, `b19_t4_V1`, `V3`, left the surface nose UP at +28 deg: the loop is what keeps the exit
tame, and a frozen sweep of 0.45-0.5 rad puts the dry CoM 7 mm ahead of the pontoons' buoyancy: a static +0.3 N m nose-down in the air).

**Floating stability against fill (static, `/scratchpad` kf; hull level, heave solved so that buoyancy = weight, tilt +-0.01 rad about the dry CoM, wings flat; N m/rad):**

| fill | hull origin over water m | K pitch / K roll, flat wings sw 0.45 | raised wings |
|---|---|---|---|
| 0.0 | +0.216 | +7.4 / +16.6 | +7.1 / +16.3 |
| 0.2 | +0.204 | +6.4 / +15.9 | +6.1 / +15.6 |
| 0.3 | **+0.023** | **-1.4 / -1.65** | -1.7 / -1.95 |
| 0.4 | +0.008 | +2.9 / +4.8 | +1.7 / -1.3 |
| 0.5 | -0.004 | +4.9 / +7.3 | +2.6 / -0.7 |
| 0.6 | -0.016 | +2.2 / +0.3 | +0.5 / -0.1 |
| 0.7 | -0.180 (submerged) | +3.0 / +1.0 | +2.8 / +0.85 |

Two geometric facts. (1) The floating machine has **two draft branches 0.18 m apart** (fill 0.2: hull +0.204, fill 0.3: +0.023) with no waterplane between them: between pontoon tops (hull -0.20) and the dorsal float
(hull +0.175..+0.225) there is only the thin hull tube, so the heave stiffness is nil and the buoyancy is nearly constant across the gap. A machine that passes from the lower branch to the upper one (deflating) rises through the gap
with its net buoyancy unspent and arrives at the pontoons with speed. (2) K is **negative at fill 0.3** (and roll is weak at 0.4-0.6 with the wings raised): the waterplane there is the hull tube. B17's inference (pontoons just under the surface at partial fill) is right for the *floating*
phase; the failures of the fast-blow rises had the other cause above. With flat wings in the waterplane (hull origin about 0 at fill 0.5-0.6) K is +5 / +7 (the wings are the waterplane).
Drooping the wings (dihedral -0.2 to -0.6) changes nothing in the table (wings do not reach the water at those drafts).

Deflating on the surface (V6, `b19_t6_*`, loop kept on while the tanks empty, fill 0.5 at rest first, 1.5 m start): 0.3/s arrives pitched -40 deg and inverts at the lift-off; 0.1/s arrives -43 deg, settles -70 (lift-off "held" at tilt 33 deg).
The trace (r1): pitch goes -1 deg to -54 deg in 1.3 s at fill 0.35-0.2 while the hull jumps from the lower to the upper branch and the pontoons leave the water: the K < 0 window and the jump of fact (1), not the loop.

### B19 step 4: tuning the exit (ids `b19_t4_*` ... `b19_t9_*`, tuning rows on the 1.5 m start, default mode unless stated; 20 rollouts; wetting only where named)

All with flat wings, the sweep pitch loop, 2% tank seed. "Arrival window peak" = largest |pitch| from the first sample with a wing strip dry to 2 s after "afloat" (z over -0.15 and |vz| under 0.05, tanks empty).

| ids | what changed | arrival window peak pitch / roll | afloat attitude | after |
|---|---|---|---|---|
| `t4_V4_C` | loop kp 2, kd 3, rate 0.5, fill 0 all the way | 15.1 / 4.4 | -14.2 / -3.6 | upright, lift-off held |
| `t4_V1`, `V3` | sweep frozen from -0.6 m | 27.9, 28.5 (nose up) | +27.8, +28.4 | settles upright, lift-off held |
| `t4_V2`, `t5_V5` | fill ramped to 0.5 over the last 0.4 m, rest at fill 0.5 on the surface (hull origin 0.0, wings awash, pitch +-0.1 deg), then tanks emptied at 0.3/s with the wings raised | default 8.0 / 3.9; wetting 17.9 / 8.9 | -1.3 / -0.3; wetting +4.7 / -0.3 | default and wetting settle upright and lift off; `V5b` (ramp from -0.8 m) arrives upright then inverts at the lift-off |
| `t6_V6` (r = 0.3, 0.1 per s) | rest at fill 0.5, tanks emptied with the loop on | 74.6, 77.2 | -40.5, -42.9 | inverts / settles -70 deg |
| `t7_V7` (0.3, 1.0 per s; loop off in the deflation) | | 74.3, 21.1 | -30, -20 | inverts / lift-off held; wetting rows never came to rest at fill 0.5 (a +-0.02 m/s heave limit cycle: my 0.02 dwell test never passed; run to T = 100 s, no result) |
| `t8_E1,E2,E4` | sweep fixed at 0.25 or 0.0 for the last 0.6 m (loop off) | 61.6, 80.1, 52.3 (nose up) | +60, +77, +51 | inverted lift-off, 1,000-1,700 limiter steps (not evidence) |
| `t8_E3` | dihedral ramp 0 -> 0.5 rad over the last metre (tips out first) | 23.1 / 3.9 | +23.0 | settles, lift-off held |
| `t9_G1` | loop kp 2, kd 3, rate 2, sweep limit -0.5..0.9 | 16.9 / 1.5 | +4.9 / +0.3 | held |
| `t9_G2` | kp 3, kd 4, rate 2 | 15.6 / 0.2 | +3.8 / -0.2 | held |
| **`t9_G3`** | **kp 3, kd 4, rate 1, sweep limit -0.5..0.9** | **13.3 / 3.8** | **+0.1 / -0.1** | **held 1.80-2.02 m, tilt 13.4 deg** |
| `t9_G4` | G2 + dihedral ramp | 31.1 / 6.1 | +30.9 | held |

What the table says. (a) The sweep loop is what makes the rise and the exit controllable: without it (fixed sweep) the exit is nose-up by 28-80 deg, with it the exit excursion is 13-17 deg and the machine
settles upright every time. (b) Resting at the surface at fill 0.5 is a clean arrival (peak 8 deg, wings awash and in the waterplane) but **the way from fill 0.5 to the float crosses the fill 0.3 window of K < 0
and the 0.18 m draft jump, and every recipe tried for it (six rows, two rates, loop on and off) pitched the machine 20-77 deg**; V5 passed by luck of the pose blend and is not repeatable (V5b). I dropped the rest stop.
(c) The exit excursion (13-17 deg with every loop setting that worked) is the entrained water leaving with the wings plus the pontoons coming out at 0.45 m/s: the sweep has authority over it only while the wings are
wet (7 N m/rad), then through the dry CoM shift (about 2 N m/rad); I did not find a setting under 10 deg. **The recipe adopted: G3 with fill 0 at 0.3/s from the start** (the plain blow of the first diagnostic runs),
wings level, sweep loop kp 3 / kd 4 / rate 1 rad/s, limits -0.5..0.9 rad, sw0 0.45.

### B19 step 5: the recipe G3 on three starts, both modes (ids `b19_rep_*`; 8 rollouts; they were meant as the reported rows, they are now tuning rows, see the next step)

Recipe G3: tanks emptied at 0.3/s from the start, wings level, sweep loop (sw0 0.45, kp 3, kd 4, rate 1 rad/s, -0.5..0.9), hinge gains re-read and `InducedFlow` cleared at the lift-off, then wings raised (2 s), settle 4 s, lift-off to 2 m, hold 5 s.
Starts: S1 2.5 m with 2% tank mismatch; S2 3.5 m, no mismatch, 4 deg initial pitch; S3 2.0 m, 3% mismatch, 3 deg initial roll. 0 QACC everywhere. "Peak" = largest |pitch|, |roll| from the first dry wing strip to 2 s after afloat.

| start, mode | peak pitch / roll deg | afloat attitude (roll, pitch) | 6 s later | wet wing strips on the float | lift-off 2 m, 5 s hold | flags |
|---|---|---|---|---|---|---|
| S1 default | 13.8 / 2.8 | -0.1 / +0.1 | -0.1 / -2.3 | 0 | held 1.80-2.02 m, tilt 13.5 | none; sweep 96.5 N m (the x4 stall, 4.0x witness); X2 0.97 |
| S1 default, unlimited twin | 12.7 / 3.1 | -0.1 / +0.1 | 0.1 / -1.1 | 0 | held | sweep 253 N m = **10.5x witness: not evidence** (the twin only) |
| S1 wetting | 8.1 / **49.6** | -33.8 / +0.5 | **-45.6** / -1.9 | **16 (one side)** | inverted, 2,322 limiter steps | **fail** |
| S1 wetting, unlimited twin | 10.7 / 44.3 | -31.4 / -7.0 | -45.6 | 16 | inverted | **fail** (twin torque 32x witness: not evidence either way) |
| S2 default | 13.5 / 2.2 | -0.1 / +0.1 | 0.0 / -2.2 | 0 | held, tilt 13.5 | sweep 96.5 N m |
| S2 wetting | 26.3 / 13.1 | -10.9 / +26.2 | -0.2 / -0.8 | 0 | held, tilt 15.3 | peak over 10 deg |
| S3 default | 2.0 / **38.0** | +3.2 / +0.4 | **+39.7** / -2.6 | **14** | inverted, 1,366 limiter steps | **fail** |
| S3 wetting | 9.8 / **41.3** | +30.9 / -3.5 | **+44.2** / -3.7 | 16 | inverted | **fail** |

Three of eight rows fail by the same thing: the machine **ends heeled at 40-46 deg on its side with the wings of the low side in the water** (14-16 of 32 strips wet) and stays there, 6 s later and at the lift-off. It is a
second resting state of the floating machine with the wings out; the upright float is entered only if the roll at the exit is small. The roll before the exit was small in all of them (S1 wetting -0.7 deg at 8 s,
creeping to -9 deg at 19.5 s with the yaw drifting 2 -> 33 deg; S3 default the initial +2.7 deg unchanged for 22 s: no roll restoring at the climb speed), and the exit multiplies it: a roll of 2.7 deg before the exit ends at 39.7 deg, -9 deg ends at -34..-46.
The creep in wetting is the lateral mode D1 found unstable without the added-mass reaction (default mode holds roll and yaw to 2-5 deg over the same climb). A pitch loop alone is not a surfacing law; roll needs its own loop.

### B19 step 6: a roll loop, and what the exit is (ids `b19_t10_*` ... `b19_t12_*`; 11 tuning rollouts on a 2.2 m start with 3 deg initial roll and 2% tank mismatch, or the variants named)

Roll loop (`roll_loop`): the wings' dihedral, one side at a time (raising a wing of 0.9 m span moves its entrained water inboard, so the net buoyancy then acts off the inertial centre and rolls the machine; a wing at
dihedral +-d moves the same way, so only the side to be raised is moved): u = k roll + kd roll rate, left wings raised when u < 0 and right wings when u > 0, limit 0.6 rad, 1 rad/s.

| id | what | peak pitch / roll in the exit window | afloat attitude | 6 s later | lift-off, hold 5 s |
|---|---|---|---|---|---|
| `t10_base_default` / `_wetting` | G3, 3 deg initial roll, no roll loop | 1.3 / 38.2; 8.2 / 41.5 | +3.2 / +0.7; +30.1 / -0.9 | **+39.9** / -2.6; **+44.3** / -3.6 (14-16 wet strips) | inverted both |
| `t10_roll4_default` / `_wetting` | + roll loop k 4, kd 4 | 26.2 / 16.9; 45.1 / 12.0 | +12.4 / +26.2; -4.8 / +44.3 | 0.1 / -0.8; 0.0 / +0.7 | held both |
| `t11_roll4off6_default` / `_wetting` | loop off above -0.6 m | 15.9 / 17.6; 37.5 / 28.1 | -13.9 / +15.9; +23.7 / +36.3 | 0.2 / -0.6; 0.1 / +0.7 | held both |
| `t11_roll4off3_default` / `_wetting` | loop off above -0.3 m | 30.3 / 16.6; 47.2 / 12.0 | +0.2 / +30.1; -3.3 / +46.4 | 0.0 / +0.1; 0.0 / -2.7 | held both |
| `t12_vz08_default` / `_wetting`, `vz05_wetting` | + a fill integrator holding the climb rate to 0.08 / 0.05 m/s | 15.3 / 5.4; 7.0 / **61.1**; 8.8 / **43.6** | -0.3 / +4.5; -42.9 / -2.6; +26.6 / -5.1 | 0.0 / +0.6; **-45.7**; **+45.5** | held; inverted; inverted |

(Peak pitch / roll are over the exit window; afloat attitude is (roll, pitch).) The roll loop turns the "heeled on its side" end state into an upright float in every row that has it alone (6 of 6: `t10_roll4`, `t11_*`) while the same start without it heeled in 2 of 2 and the G3 rows of the previous step heeled in 3 of 8; it does not shrink the excursion at the exit, it adds a pitch kick of its own (peak pitch 26-47 deg in
the loop-on rows against 1-8 deg without). In wetting mode the climb is **0.33 m/s against 0.07 in the default mode** (no added-mass reaction), the pitch loop pins at its 0.9 rad limit (pitch +22 deg at 6.4 s in `t11_roll4off6_wetting`), and the exit is nose-up 37-47 deg. A fill
integrator that holds the climb to 0.05-0.08 m/s did not remove the heeled end state in wetting mode (2 of 2 failed), because the exit is not set by the climb speed but by the 0.18 m draft gap and the 26 N buoyancy surplus at the pontoons' exit (below).

**The exit, in numbers.** Net buoyancy of the empty machine is 46.7 N against a dry weight of 20.7 N (b = 2.25); the flat wings carry 125 kg of entrained water until they are clear, then the 2.1 kg machine is accelerated by 26 N (12 m/s^2) while the pontoons
are still under: a breach of 0.3-0.5 m/s. Between the draft at which the hull tube leaves the surface (hull origin +0.02 m) and the one at which the pontoons' tops come out (+0.15 m) the machine has **no waterplane** (the fill table above): the heave is neutral over 0.11-0.18 m,
so slowing the tank rate does not slow the breach (the speed at the pontoons goes as the cube root of the rate, 0.32 m/s even at 0.02 /s of fill). Closing the gap needs buoyancy volume at mid-height; neutrality turns every litre into a kilogram of dry shell
(`/scratchpad` kf, geometry dorsal radius 0.04 / 0.05 at hull z 0.12: dry mass 3.27 / 4.30 kg against T5b's 2.115; K positive at every fill, hull at the surface at fill 0, no draft gap), and the lift available is 31.3 N = 3.19 kg at the law's amplitude limit: **a machine without
the gap cannot lift off** (3.27 kg dry against 3.19 kg of thrust). That is the geometric fact the surfacing criterion runs into on T5b: a pontoon pair sized by neutrality leaves a draft gap that the breach has to cross.

## B19, the reported rows (ids `b19_final_*`; recipe R; fresh starts S4-S6, none used by a tuning rollout; 8 rollouts; 0 QACC; X2 share 0.97 on every flying row)

**Recipe R** (all of it kinematics and ballast, no typed force): tanks emptied at 0.3/s from the start; the four wings level (dihedral 0, feather 0) at a mean sweep set by the pitch loop
(sw = 0.45 + 3 pitch + 4 pitch rate, limits -0.5..0.9 rad, 1 rad/s); the left or right wings raised by a roll loop (k 4, kd 4, limit 0.6 rad, 1 rad/s) below -0.6 m; when afloat with the tanks empty (z over -0.15 m, |vz| under 0.05) the InducedFlow state is cleared, the wings go to the raised
pose (2 s), 4 s of rest, hinge gains re-read, Hover4 (B17's gains) lifts off to 2 m and holds 5 s. Starts: S4 2.8 m, 2% tank mismatch; S5 1.8 m, no mismatch, 4 deg initial pitch and -2 deg roll; S6 3.3 m, 3% mismatch, 3 deg initial roll.

| start, mode | peak pitch / roll in the exit window, deg | afloat attitude (roll, pitch) | after 6 s (roll, pitch), wet wing strips | lift-off 2 m, 5 s hold | notes |
|---|---|---|---|---|---|
| S4 default | 17.0 / 16.1 | -12.2 / +17.0 | +0.1 / -1.6, 0 | held 1.80-2.02 m, drift 0.35 m, tilt 13.6 | sweep 96.0-96.6 N m (4.0x witness = the x4 stall) |
| S4 default, unlimited twin | 20.8 / 10.6 | -7.1 / +18.9 | -0.1 / -0.9, 0 | held | twin torque 10.7x witness: not evidence, agrees |
| S4 wetting | 37.8 / 32.4 | +27.7 / +34.3 | 0.0 / +0.1, 0 | held 1.81-2.02, drift 0.32, tilt 15.3 | |
| S4 wetting, unlimited twin | 50.9 / 3.0 | -2.4 / +49.6 | 0.0 / -0.4, 0 | held | twin torque 13x witness: not evidence, agrees |
| S5 default | 5.6 / **38.3** | -3.1 / +4.6 | **-40.1** / -2.6, **15** | **inverted at the lift-off**, 589 limiter steps | heeled on its side with the low wings wet |
| S5 wetting | 32.3 / 21.4 | -15.6 / +30.9 | 0.0 / +0.4, 0 | held, drift 0.31, tilt 15.3 | |
| S6 default | 17.3 / 5.3 | -0.4 / +6.4 | -0.1 / +0.5, 0 | held, drift 0.35, tilt 13.6 | |
| S6 wetting | 53.5 / 10.1 | +10.0 / +51.8 | -0.2 / +16.2, 0 | held, drift 0.30, tilt 14.6 | still 16 deg nose-up after 6 s, peak 53 deg |

**Against the target (arrives upright, |roll| and |pitch| under 10 deg, wings dry, lifts off to 2 m for 5 s, 3 of 3 starts, both modes): not met.**
Lift-off and hold: **5 of 6** starts-and-modes (the S5 default heeled), and the settled float is upright and dry in those 5 (|roll|, |pitch| under 2 deg at 6 s except S6 wetting at +16 deg). The arrival itself is not under 10 deg in any row: the exit window peaks are 5.6-53.5 deg of pitch or 5.3-38.3 deg of roll,
and the afloat attitude is outside 10 deg in 5 of 6. Both modes: default mode peaks 17-21 deg (pitch), wetting 32-54 deg, and the modes disagree on the transient by 2-3x (not established as a law; both settle upright in 5 of 6).
**What fixes the size of the transient, and why it was not brought under 10 deg by control**: (i) the empty machine floats with 26 N of buoyancy surplus on 2.1 kg, and the 0.11-0.18 m draft gap with no waterplane (fill table above) lets it breach at 0.3-0.5 m/s once the flat wings' 125 kg of entrained water is gone;
(ii) a roll at the exit grows (the side whose wings clear first loses its entrained mass first and rises, which adds heel: -4.5 deg to -32 deg in 0.8 s in `b19_final_S5_default`) and the sweep and dihedral hinges act only while the wings are in the water; (iii) the one geometry that removes (i) (a dorsal float
of 0.04-0.05 m radius at hull z 0.12, no draft gap, K > 0 at every fill) is 3.27-4.30 kg dry against 3.19 kg of thrust at the law's amplitude limit: no lift-off. The one lever left unprobed and indicated by the E3/G4 rows (a symmetric dihedral ramp before the exit: roll peaks 3.9 and 6.1 deg
with the pitch loop alone) is not part of recipe R; it was not run on the reported starts.
Ledger: the B19 tuning rollouts are 58 (diagnostics 4, t1 3, t2 4, t3 4, t4 4, t5 4, t6 4, t7 4, t8 4, t9 4, the 3-start trial 8, t10 4, t11 4, t12 3), the reported rows 8: **66 of the 80 are gone** (B20 + B21 have 14).

## B20, step 1: held thrust against speed, forward and reversed, 0.5-0.9 Hz, both modes (ids `b20_held_*`; T5b at 3 m, fill 0.95, hull held level and moved at U; stroke 20 deg, feather 50 deg, diagonal phasing, a = 1; 36 held rows, 0 QACC)

Net hull-axis force Fx in N (stroke wrench less the static buoyancy at the same pose; + = toward the nose); U in m/s along the nose (the reverse rows are the wings turned over on their spars, feather offset -180 deg, hull moved backward, U < 0). Rows with limiter steps are not evidence (2: `rev_f07_Um45_default`, `rev_f09_Um45_default`).

| forward, f Hz | default U = 0 / 0.3 / 0.45 | wetting U = 0 / 0.3 / 0.45 | speed where Fx = 0, default / wetting (linear between rows) |
|---|---|---|---|
| 0.5 | +43.9 / +10.0 / -21.1 | +41.7 / +16.6 / -8.5 | 0.34 / 0.40 m/s |
| 0.7 | +86.0 / +42.6 / +12.1 | +81.6 / +51.9 / +26.9 | above 0.45 / above 0.45 |
| 0.9 | +140.9 / +85.3 / +11.1 (stroke peak 97 N m, sweep 96) | +133.7 / +98.7 / +70.9 | above 0.45 / above 0.45 |

| reversed, f Hz | default U = 0 / -0.3 / -0.45 | wetting U = 0 / -0.3 / -0.45 | speed where Fx = 0, default / wetting |
|---|---|---|---|
| 0.5 | -17.7 / **+60.6** / +106.6 | -15.1 / +22.7 / +55.5 | -0.07 / -0.12 m/s |
| 0.7 | -34.7 / +122.0 / (not evidence) | -29.5 / +15.3 / +52.7 | -0.07 / -0.20 |
| 0.9 | -57.2 / +163.3 / (not evidence) | -48.3 / +4.9 / +72.0 | -0.08 / -0.27 |

Readings. (1) **Forward thrust is not the limit**: at 0.5 Hz the machine's drag balances the stroke at 0.34 (default) and 0.40 (wetting) m/s, so a whole-run mean of 0.3 is out of reach at 0.5 Hz (the starts-from-rest transient); at 0.7 Hz the balance
is above 0.45 m/s in both modes with the stroke hinge at 56-72 N m (witness x4 curve stall 96); the two modes differ by 20% at 0.5 Hz and by more at 0.7-0.9 (wetting +12 to +60 N at U = 0.45).
(2) **The reverse thrust cannot reach 0.3 m/s in the default mode at any frequency**: the force turns against the motion at about 0.07 m/s whatever the frequency (+60, +122, +163 N at U = -0.3 for 0.5, 0.7, 0.9 Hz): turned over, the wings are a paddle that moves the water
backward at the stroke's tip speed, not a thruster, and once the hull moves backward at 0.07 m/s the same plates are the drag. In wetting mode the same rows cross at -0.12, -0.20 and -0.27 m/s; **the modes disagree by 3x on the reverse swim**, so a reverse antipode is "not established" at best and
0.07-0.08 m/s in the default mode (as B16 measured: 0.078). The antipode therefore has to be a turn (B21) and a forward swim.
(3) The pitch moment about the CoM at net thrust is +2.3 N m at +10 N (0.5 Hz, U 0.3), +8.5 N m at +43 N: the thrust line is 0.2 m above the CoM (B15), and at 0.7 Hz it is the tail's 4.9 N m/rad at 0.3 m/s that has to cover it.

## B20 and B21: swim speed and the turn (ids `b20_*`, `b21_*`; the coordinator closed the reverse swim as "not established": the modes disagree 3x, step 1 above)

**Turn in place, held** (`b21_held_turn_*`: left wings turned over, right forward, f 0.8, a 1.0 on the flipped side, forward side at rr x a; hull held, U = 0; N m / N): rr 0.64: net surge -1.3 (default) / +1.8 (wetting), **yaw +17.3 / +17.2**, pitch -0.15 / +0.41, roll +3.1 / +3.0;
rr 1.0: surge +32.7 / +33.0, yaw +36.7 / +35.4; rr 0.3: surge -19.3 / -15.8, yaw +7.5 / +7.4. The turn couple and its surge balance **agree between the modes to 1-3%**. In the free machine they do not carry the turn:

| id (2.5 m start, heading 30 deg unless noted) | law | result at the end of the turn phase | depth swing |
|---|---|---|---|
| `b21_t1_turn_{default,wetting}` | f 0.5, a 0.7, rr 0.5 | 1.9 deg/s (default: 84 deg in 40 s), wetting 0.8 deg/s | 0.06 / 0.10 m |
| `b21_t2_turn_f08_*` | f 0.8, a 1.0, rr 0.64 | default **pitched -40..-58 deg, no turn**; wetting 105 deg in 45 s while translating backward 18 m (0.39 m/s) | 0.31 / 0.26 m |
| `b21_t3_turn_f05_*` | f 0.5, a 1.0, rr 0.6 | default **150 deg in 50 s (3 deg/s), depth 0.11 m**; wetting 53 deg in 50 s, translating backward 10 m | 0.11 / 0.19 m |
| `b20_rep_anti_*` (2.9 m, heading 75 deg; + surge loop ks 2, up to 80 s) | f 0.5, a 1.0, rr 0.6 | default: pitch -53 deg, **no turn (error -179 deg after 80 s)**; wetting: 107 of 180 deg in 80 s (error -30.7 deg), depth 0.17 m | 0.12 / 0.17 m |

The depth swing is under 1 m in every row; the 180 deg is not reached in any: **B21 not met**. What blocks it: turning one side's wings over costs a nose-down 12-18 deg in the flip itself (the left wings' mass and entrained water move chordwise), at rest the tail has no authority and the
sweep loop has 0.85 N m/rad, and the mode-dependent surge (wetting drives backward where the default pitches down) is what the surge loop would have to cover; the yaw rate that results is 1-3 deg/s against 17 N m of held couple (yaw damping of four large plates). Ledger: 6 (B21 t1-t3) + 2 (t1 forward) + 4 (reported) = 12 on top of B19's 66 = **78 of 80**; I stop here.

**Forward swim** (f 0.7 Hz, stroke 20 deg, feather 50 deg, a 0.3-1.0, speed PI to 0.55 m/s, tail pitch loop, Psi_lr yaw loop limit 0.6, ballast depth PD; start 2.9 m, heading 75 deg, whole run 60 s; 0 QACC, 0 limiter steps):

| row | whole-run speed along the heading | last 8 s | yaw error at the end | depth range | torque stroke / sweep (witness x4 stall 96) |
|---|---|---|---|---|---|
| `b20_rep_fwd_default` | **+0.299 m/s** | 0.242 | +31.8 deg | -3.31..-2.90 | 97 / 96 |
| `b20_rep_fwd_wetting` | **+0.214 m/s** | 0.151 | -50.0 deg | -2.92..-1.99 | 96 / 23 |
| `b20_rep_anti_default` (turn failed, then 45 s forward) | -0.100 (136 s whole run) | 0.229 | | -3.22..-2.77 | |
| `b20_rep_anti_wetting` | +0.196 (136 s) | 0.154 | +50 deg | -3.02..-2.15 | |

Mean signed speed (half the sum of heading and antipode): default +0.100, wetting +0.205. **B20 not met** (target 0.3, modes within 25%): the modes differ by 1.4x on the heading (0.299 against 0.214; on the tuning start with a_max 0.8: 0.302 / 0.306, which agreed, so the
wetting mode's heading loss, not its thrust, is what cut the run) and the antipode needs a turn that does not complete. The thrust is not the limit (held balance speed above 0.45 m/s at 0.7 Hz); the heading hold in wetting and the turn are.

## B22, the still machine (actuators held; ids `b22_*`; separate from the 80, controls)

| row | result, both modes |
|---|---|
| `b22_still_surf_hold_*` (2.8 m, wings level, tanks held at 0.95, 30 s) | stays at -2.80 m, attitude within 2 deg: no surfacing |
| `b22_still_surf_blow_*` (hinges held level at sweep 0.45, tanks emptied at 0.3/s, lift-off law absent) | rises on buoyancy alone (default arrives pitched +68.6 deg, wetting upright at +0.09 m); **no lift-off is possible with the hinges held** (it stays afloat at the surface): the ballast alone is the rise, not the surfacing criterion |
| `b22_held_still_U{p00,p30,m30}_*` (the b15_still rows: hull held and moved at 0, +0.3, -0.3 m/s, hinges held) | net force 0 at rest; -3.29 N (drag) at +0.3, +2.95 N at -0.3, pitch +-0.26..0.58 N m, no thrust; modes identical |
| `b22_still_fwd_*`, `b22_still_anti_*` (the B20 phases with every hinge held, ballast PD running) | displacement 0.000 m along both commands in both modes, attitude unchanged (75 deg heading), torque 0.0: speed 0, no turn |
The B19 still lift-off (hinges held on the float) is `b18_float_still_*` of the previous part.

## B23, the whole chain and dt/2 (ids `b23_chain_*`, `b19_final_S4_*_dt2`; `figures/S0b_T5c_mission.png` = `b23_chain_default`)

Chain (`s0b_t5c_chain.py`, mission3): float 3 s, lift-off to 2 m, 10 s hold, landing at 0.15 m/s, flood to 3 m (InducedFlow cleared at the flood and the rise, hinge gains re-read at the swim and the lift-offs), 5 s, wings level, forward swim 20 s (f 0.7), surfacing by recipe R, wings raised, second lift-off to 2 m, 5 s hold. 0 QACC, 0 limiter steps in every phase of both rows.

| phase | default | wetting |
|---|---|---|
| lift-off, hold, landing | held 1.80-2.03 m, upright | held 1.81-2.03 m, upright |
| flood to 3 m | 39.3 s, upright | 38.3 s, upright |
| forward swim 20 s | 0.345 m/s last window, yaw error +11.7 deg, depth -2.87 | 0.176 m/s, yaw error -47 deg, depth -2.9..-2.4 |
| surfacing | arrives pitch +22, settles 0 / -1.3, wings dry; the climb stalled near -0.3 m for about 35 s before the exit (phase 77 s) | **arrives roll -34 deg, heeled -42.6 deg with 15-16 wet strips**, lift-off inverts at 0.6 s |
| second lift-off, hold 5 s | **held 1.80-2.02 m** (tilt 11 deg) | inverted: **fail** |

**Chain: complete in the default mode (every part, ending in the second lift-off); not complete in wetting (stops at the surfacing, the same heeled end state as `b19_final_S5_default`). Modes disagree: not established.** X2 share (flying phases) 0.98 / 0.76 (the wetting figure includes the failing lift-off). Torque peaks: stroke 97, sweep 97-110, feather 66-71 N m
(witness x4 stall 96 / 96 / 64; 4.0-4.5x the witness, under the 10x flag); the default row's sweep and stroke sit on the x4 curve during the swim and the surfacing loop.

dt/2 (5e-5 s) on the two S4 surfacing rows: default `b19_final_S4_default_dt2` upright, lift-off held 1.80-2.02 m, exit-window peak 23 / 2 deg (dt 1e-4: 17 / 16), afloat attitude -2.0 / +22.8; wetting `b19_final_S4_wetting_dt2` upright, lift-off held, peak 46 / 16 deg (dt 1e-4: 38 / 32): **the pass/fail outcome is the same, the exit transient differs by 6-14 deg**, so the transients above are not converged to better than that. No swim, turn or chain row passed, so no dt/2 for them.

## Summary of B19-B23 (T5b, no geometry changed, so no `figures/S0b_T5c_built.png`)

B19 surfacing: not met as stated (arrival under 10 deg in no row); recipe R settles upright, dry, and lifts off 2 m for 5 s in 5 of 6 reported starts-and-modes plus the default chain; the failure is a heeled end state in wetting (chain) and one default start. B20: not met (0.299 / 0.214 m/s heading; mean signed 0.10 / 0.205). B21: not met (turn reaches 0-107 of 180 deg; depth swing 0.1-0.4 m). B22: all still rows zero. B23: default complete, wetting stops at the surfacing.
Rollout ledger for B19+B20+B21: 78 of 80 (B22 and B23 and the dt/2 rows are separate). Held measurements (k1-k5, kf, b20_held, b21_held, b22_held) are not rollouts.
Hash check after this part: see the report; `s0b_t5b_machine.py`, `s0b_t5b_probe.py`, `s0b_laws.py`, `mission.py` not edited; the T5b calibration store gained keys only for the geometries probed in `kf` (the default key unchanged).
