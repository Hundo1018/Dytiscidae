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
