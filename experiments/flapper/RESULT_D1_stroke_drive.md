# Pass D1, result: the stroke driven by mechanisms

2026-11-17. Frozen text: `PREREG_D1_stroke_drive.md` (`f73c0be`). Code:
`drive/drive.py` (the mechanisms), the `drive` hook in `freerig.run_free`,
`drive/d1_run.py`, `drive/d1_read.py`, `drive/draw_d1.py`. Rows:
`results/D1_sweep.jsonl` (76), `results/D1_checks.jsonl` (18). Nothing is
chosen here; the user decides when candidates exist (the user, 2026-11-17).

## Outcome in one paragraph

Four drives were built as bodies and constraints in the model and driven by
one witness motor: direct drive at the hinge (DD), a crank-rocker four-bar
(FB), a Scotch yoke with two links (SY), and a crank pulling a cable against
a spring (CT). **At the 8 m/s cruise every one makes the stroke the
prescribed sinusoid made, within 1% of its lift and 0.03 N of its thrust; the
three cranks cost 7.1-7.4 W of electrical power through the witness motor
and direct drive 14.6 W, because its rotor reverses every half beat.** At
take-off the picture splits: the links carry 450-960 N on 40 mm levers, the
yoke's loop is too compliant in this engine to be read (9.5 deg off its own
kinematics), the cable goes slack and snaps taut at every spring that keeps
the amplitude, and the witness motor is on its limit for 40-55% of the beat
in DD and FB. A crank's amplitude is fixed by its geometry, so no crank built
for cruise can do take-off's stroke: each point had its own linkage.

## What was the same for every candidate

W7's cruise point ("R strips": 8 m/s, 2.5 Hz, stroke 40 deg, tip feather
15 deg by strips, root feather held at -2.94 deg) and take-off point ("K1
one panel": at rest, body at 90 deg, 4.5 Hz, stroke 85 deg, feather 50 deg
on its servo), the table law, dt 1e-4 s, loops at solref 2 dt; the feather
and strip twist prescribed and phased to each mechanism's own stroke; one
witness motor (tau_s 1.0 N m, omega_0 1000 rad/s, rotor 2e-5 kg m^2, copper
loss tau^2 omega_0/tau_s); each candidate at its best gear from a sweep.
The baseline is the prescribed sinusoid with ideal torque at the same dt:
cruise 7.25 W, 1.000 weights, +0.046 N; take-off 129 W, 1.856 weights.

## The candidates at their best gear

Best = least electrical power with the stroke within 2 deg of its amplitude
(the frozen rule). Power columns: "mech" is shaft plus servos (comparable
with the baseline's ideal torque), "elec" adds the witness motor's copper
loss.

| point | drive | gear | mech W | elec W | lift vs base | thrust vs base | motor on limit | peak loop force | beat made |
|---|---|---|---|---|---|---|---|---|---|
| cruise | DD | 48 | 7.18 | 14.59 | +0.1% | -0.006 N | 0 | - | 1.000 |
| cruise | FB | 48 | 7.12 | 7.32 | +0.2% | -0.026 N | 0 | 86 N | 1.000 |
| cruise | SY | 48 | 7.15 | 7.35 | +0.0% | -0.024 N | 0 | 156 N | 1.000 |
| cruise | CT (k 1) | 48 | 6.86 | 7.12 | -0.1% | -0.073 N | 0 | 1683 N (cable slack 30%) | 1.000 |
| cruise | CT (k 2) | 48 | 7.06 | 7.44 | +0.2% | -0.032 N | 0 | 101 N (never slack) | 1.000 |
| take-off | DD | 32 | 128.1 | 294.9 | +0.9% | +0.024 N | 41% | - | 1.000 |
| take-off | FB | 32 | 127.1 | 121.2 | +1.9% | -0.125 N | 55% | 449 N | **0.951** |
| take-off | FB | 24 | 139.3 | 152.4 | +5.4% | +0.038 N | 0 | 575 N | 1.000 |
| take-off | SY | - | | | | | | 925-966 N | none within 2 deg |
| take-off | CT (k 16) | 32 | 93.2 | 222.2 | -34.4% | +1.297 N | 56% | 1163 N | **0.776** |

Two rows are there because of something the frozen rule did not foresee
and I added after seeing the first sweep (disclosed): a crank whose motor
saturates keeps its amplitude and loses frequency, which the 2 deg rule
cannot see. The sweep was re-run with the beat recorded. FB's best at
take-off by the frozen rule beats 5% slow; its best that holds the beat is
gear 24 at 152 W. CT's best at take-off beats 22% slow and lifts a third
less: it does not do the task. CT's best at cruise by the frozen rule is
the slack cable; the stiffer spring (k 2) is listed because it is the one
that never goes slack. Every row is in the sweep file.

## Against the frozen checks

| | frozen | read | |
|---|---|---|---|
| K1 | in a vacuum, mean shaft power within 1% of peak; loop gap under 0.5 mm | cruise: 0.00-0.32% for all four; gaps 0.04 (FB), **0.54 (SY)** mm. Take-off: the motor alone reads 3-5% because the feather servo trades about 7 W with it each beat; motor plus servo reads 0.09% (FB) and 0.14% (DD) of peak. **CT at take-off: -19 W**, cable snapping, motor on its limit 43% | holds for DD and FB; SY's gap just over; CT fails at take-off. The check as frozen (motor alone) was wrong where the feather moves |
| K2 | FB and SY within 0.5 deg of closed form at cruise | FB 0.20; **SY 1.77** (halves with the step: 0.72 at dt 5e-5, a disclosed probe) | FB holds, **SY fails**: its rows are labelled |
| K3 | FB and SY: down share 0.50 +- 0.02; second harmonic under 5% | down 0.501, 0.494; second harmonic (closed form) 0.044 and 0.001 at cruise, **0.093** for FB at take-off | holds at cruise; FB at take-off fails |
| K4 | FB, SY at cruise: lift 3%, thrust 0.05 N, power 5%; take-off lift 5% | +0.2%, -0.026 N, -1.8% (FB); +0.0%, -0.024 N, -1.4% (SY); take-off FB +1.9% (gear 32), +5.4% (gear 24) | holds (FB gear 24 at take-off just over) |
| K5 | DD over FB by over 15% at cruise; gap smaller at take-off | +99% at cruise; **+143%** at take-off (FB gear 32) | first holds, **second wrong** |
| K6 | CT's down share off 0.5 by over 0.03; slack under 10% | 0.493; **slack 30%** at the frozen rule's best | **both wrong** |
| K7 | loop force 10-80 N cruise, 50-400 N take-off | **86-156 N; 449-966 N**; the cable 1683 N when it snaps | **wrong**: I under-counted the short levers |
| K8 | on the motor's limit under 10% at the best gear; more with the half motor | cruise 0 for all; **take-off 41-56%**; the half motor: cruise still 0, take-off 86-100% (DD's stroke falls to 52 deg) | cruise holds; **take-off fails** |

Link mass (FB, cruise): 0, 15 and 40 g/m give 7.10, 7.32 and 7.32 W and the
same lift within 0.3%: not a factor at these sizes.

## What this says, as ranges and differences, not a choice

- At cruise the stroke drive is not where the power goes: a crank of any of
  the three kinds delivers the prescribed stroke's aerodynamics for its
  aerodynamic power plus a few percent. Direct drive doubles the electrical
  cost with this witness motor, and the excess is copper loss on the rotor's
  reversal; it falls with a lighter rotor, which nothing here has measured.
- Direct drive is the only one of the four that can change amplitude and
  waveform in flight (the take-off stroke is 85 deg, the cruise 40); a crank
  needs a second mechanism or a variable-geometry linkage to do both, and
  none was built.
- Take-off is where the drive is decided: hundreds of newtons in the links,
  the motor at its limit, the cable unable to return the wing. The lever
  length (30-40 mm here, an arbitrary choice) sets the link force in inverse
  proportion and was not swept.

## What these numbers rest on

A witness motor, no gearbox loss, no bearing friction, ideal regeneration;
links with no fluid load; the feather and twist still prescribed (pass D2);
the body held. Not checked by the model: whether a link passes through the
hull. **Seen on the drawings: the take-off mechanisms reach 60-107 mm below
the shoulder line, outside a hull of 40 mm radius**; nothing in the model
places them inside it or stops them colliding.

## The drawing against the design

`figures/D1_cruise.png`, `D1_takeoff.png`, drawn from the compiled models'
geoms, sites and tendons at four crank angles, opened and checked: levers
30 and 40 mm on the span line from the moved feather hinge; four-bar crank
10.3 and 27.0 mm, coupler 60 and 80 mm, crank centre below the lever's
chord; yoke crank 10.3 and 26.9 mm on the centre line, links vertical at
mid-stroke; cable from crank pin to lever. Strokes at crank 90 and 270 deg:
+-20.0 (cruise), +-42.4 (take-off), as designed; the four-bar at crank 180
reads -3.5 and -15.0 deg where the yoke reads 0: the four-bar's stroke is
not symmetric in time between its turning points, which its second harmonic
(0.044, 0.093) measures. Each loop measured on the model closes within
0.0001 mm at every angle drawn.

## Addendum D1b: the take-off lever, 40 to 120 mm

Frozen as an addendum to the pre-registration (`ad220aa`) before any run;
`results/D1b_lever.jsonl` (30 rows, none raised), read beside D1's 40 mm
rows. Crank and links scale with the lever.

| | 40 mm | 60 mm | 80 mm | 120 mm |
|---|---|---|---|---|
| FB gear 24: peak link force | 575 N | 296 N | 210 N | 130 N |
| FB gear 24: force x lever | 23.0 | 17.7 | 16.8 | 15.5 N m |
| FB gear 24: error against its kinematics | 0.69 deg | 0.26 | 0.11 | 0.03 |
| FB gear 24: electrical power | 152.4 W | 149.4 | 148.3 | 147.7 |
| SY gear 24: peak link force | 951 N | 727 N | 564 N | 361 N |
| SY gear 24: error against its kinematics | 9.35 deg | 5.09 | 3.06 | 1.33 |
| SY gear 24: stroke made (85 asked) | 89.2 deg | 88.8 | 87.9 | 86.7 |
| CT spring 8, gear 24: beat made / slack | 0.955 / 21% | 0.958 / 23% | 0.960 / 25% | 0.962 / 25% |
| CT spring 16, gear 24: beat made | 0.756 | 0.749 | 0.744 | 0.741 |
| lowest point below the shoulder | 107 mm | 161 | 214 | 321 |

| | frozen | read | |
|---|---|---|---|
| L1 | FB's force x lever constant within 20% | falls 23.0 -> 15.5 (gear 24), 29.6 -> 17.3 (16), 17.9 -> 13.6 (32): a 1.3-1.7 spread | **wrong**: the short lever pays more than 1/a, levelling off from about 80 mm |
| L2 | SY's error as 1/a^2: under 2.5 deg at 80 mm, 1.2 at 120; holds the stroke from 80 mm | 3.06 and 1.33 (gear 24; 2.74 and 1.17 at gear 32); the stroke is 2.9 deg over at 80 mm, 1.7 at 120 | **wrong** (falls about as a^-1.8); within 2 deg only at 120 mm |
| L3 | CT does not do take-off at any lever | beat 0.91-0.96 with the amplitude 2.0-2.8 deg short (spring 8), beat 0.74-0.78 (spring 16), slack 21-25% (spring 8) | holds |
| L4 | FB's power within 10% across levers; limit share within 0.05 | 152.4 -> 147.7 W (3%); 0.553 -> 0.542 at gear 32 | holds |
| L5 | lowest point 2a + r | 107-321 mm, outside the 40 mm hull at every lever | reported |

What it changes in D1: the four-bar's take-off link force is a choice of
size, 130-575 N over this range, and its loop is read well from 60 mm on;
the motor and the power do not depend on the lever. The yoke's compliance
in this engine stays the largest of the three at every lever and is
readable only at 120 mm. The cable's failure at take-off is not a matter
of the lever. A longer lever makes a bigger mechanism: the drawing's
finding (outside the hull) gets worse with it.
