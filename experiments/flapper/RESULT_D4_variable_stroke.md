# Pass D4, result: one crank mechanism for both strokes

2026-11-17. Frozen text: `PREREG_D4_variable_stroke.md` (`e793197`). Code:
`drive/va.py` (the three mechanisms on `drive.Drive`'s interface; `drive.py`
and `freerig.py` unedited), `drive/d4_run.py`, `drive/d4_read.py`,
`drive/draw_d4.py`. Rows: `results/D4_sweep.jsonl` (51, none raised),
`results/D4_checks.jsonl` (14 vacuum rows), `results/D4_geometry.jsonl`
(built dimensions, V0, V9), `results/D4_design.json` (the solved
dimensions), the reader's print in `results/d4_logs/read.log`. Nothing is
chosen here.

## Outcome in one paragraph

Three mechanisms were built as bodies and constraints, each making the
85 deg take-off stroke and the 40 deg cruise stroke from one set of links by
a locked adjuster: the coupler's end sliding along the wing lever (VL), the
crank pin sliding along the crank arm (VR), and two cranks at a phase summed
by a beam (VP). On each, moving only the adjuster on the take-off model
reproduces the cruise model's stroke to 5e-5 deg (V0). **VR and VL do both
points: within 6% of the dedicated four-bar's electrical power at each
point (7.4 W at cruise, 140-150 W at take-off), the beat held, the motor
off its limit at gear 24 and 48.** VR's stroke gives the dedicated
four-bar's lift within 0.6% (cruise) and 2.6-3.6% (take-off); VL's gives
3.5% less at cruise and 7.4-9.4% more at take-off, a different waveform.
The adjuster holds 15-37 N at cruise and 250-410 N at take-off. **VP does
cruise and not take-off**: at cruise it costs 44% more power than the
dedicated four-bar for 6% less lift (its downstroke is 46% of the beat);
at take-off, with the cranks in phase, its rods come within 17 deg of the
lever's line, the loops carry 760-1930 N, the stroke is 6 deg off its own
kinematics and no gear holds the beat: its take-off rows are not readable.
A lock in this engine is as stiff as the inertia on its coordinate, so each
candidate was run twice, as first built and with inertia on the locked
coordinate (added after vacuum smoke rows, before any air row; disclosed
below).

## The table (each at its best gear by the frozen rule; no pick)

Best = least electrical power with the stroke within 2 deg and the beat at
least 0.98. "lock a" = as first built (the adjuster joint's armature 1e-7,
as every other joint); "lock b" = 1 kg (slides) or 0.1 kg m^2 (VP's
carrier) on the adjuster's own coordinate. FB = `drive.Drive`'s dedicated
four-bar with a 60 mm lever, re-run in this session. Baseline: cruise
7.25 W, 5.883 N, +0.046 N; take-off 129.2 W, 10.923 N.

Cruise (8 m/s, 2.5 Hz, stroke 40 deg), gear 48 for all:

| | FB | VL a | VL b | VR a | VR b | VP a | VP b |
|---|---|---|---|---|---|---|---|
| stroke amplitude, deg | 40.00 | 40.01 | 40.00 | 40.02 | 40.01 | 40.48 | 40.17 |
| mean stroke (mid of extremes), deg | +0.01 | +0.01 | +0.00 | +0.05 | +0.01 | +0.07 | +0.01 |
| down share (closed form) | 0.502 (0.500) | 0.484 (0.483) | 0.485 | 0.498 (0.497) | 0.499 | 0.462 (0.457) | 0.460 |
| second harmonic, closed form | 0.044 | 0.082 | 0.082 | 0.048 | 0.048 | 0.080 | 0.080 |
| error against kinematics, deg | 0.03 | 0.03 | 0.01 | 0.09 | 0.03 | 0.40 | 0.13 |
| largest loop gap over the beats, mm | 0.055 | 0.049 | 0.028 | 0.044 | 0.042 | 0.085 | 0.085 |
| peak force per loop, N | 43 | 28 | 28 | 45 | 45 | middle 59, rod end 29 | 59, 28 |
| beat made | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| motor on limit | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mechanical power, W | 7.13 | 7.18 | 7.18 | 7.17 | 7.16 | 10.23 | 10.12 |
| electrical power, W | 7.33 | 7.41 | 7.41 | 7.37 | 7.36 | 10.64 | 10.53 |
| lift against the baseline | +0.4% | -3.2% | -3.1% | -0.2% | -0.2% | -6.1% | -5.5% |
| lift against FB | - | -3.5% | -3.5% | -0.6% | -0.5% | -6.5% | -5.9% |
| thrust against the baseline, N | -0.022 | -0.070 | -0.070 | -0.022 | -0.022 | -0.287 | -0.280 |
| adjuster holds, peak per wing | - | 15.8 N | 15.7 N | 36.6 N | 36.4 N | 0.94 N m | 0.93 N m |
| lock moved | - | 0.035 mm | 7e-5 mm | 0.068 mm | 2e-4 mm | 0.52 deg | 2e-6 deg |

Take-off (at rest, 4.5 Hz, stroke 85 deg), gear 24. VP has no gear that
passes the rule; its gear 24 rows are shown and are **not readable** (V1,
V2 fail):

| | FB | VL a | VL b | VR a | VR b | VP a (unreadable) | VP b (unreadable) |
|---|---|---|---|---|---|---|---|
| stroke amplitude, deg | 85.26 | 86.32 | 85.28 | 86.04 | 85.26 | 87.17 | 87.13 |
| mean stroke, deg | +0.07 | -0.32 | -0.06 | +0.17 | +0.04 | -0.80 | -0.75 |
| down share (closed form) | 0.495 (0.500) | 0.524 (0.517) | 0.514 | 0.501 (0.506) | 0.501 | 0.555 (0.541) | 0.544 |
| second harmonic, closed form | 0.093 | 0.035 | 0.035 | 0.084 | 0.084 | 0.057 | 0.057 |
| error against kinematics, deg | 0.26 | **1.73** | 0.22 | **0.78** | 0.23 | **6.59** | **6.03** |
| largest loop gap over the beats, mm | 0.317 | **0.537** | 0.267 | 0.278 | 0.237 | **1.01** | **1.17** |
| peak force per loop, N (right, left) | 296 | 425, 461 | 262, 295 | 284, 293 | 269, 254 | middle 1599, 1532; rod end 793, 760 | 1895, 1762; 949, 882 |
| beat made | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.977 | 0.970 |
| motor on limit | 0 | 0 | 0 | 0 | 0 | 6% | 8% |
| mechanical power, W | 136.8 | 144.1 | 140.2 | 139.9 | 138.0 | 157.0 | 157.8 |
| electrical power, W | 149.4 | 145.6 | 140.4 | 149.9 | 147.7 | 178.0 | 184.1 |
| lift against the baseline | +3.8% | +13.6% | +11.5% | +7.5% | +6.5% | (+26%) | (+24%) |
| lift against FB | - | +9.4% | +7.4% | +3.6% | +2.6% | (+21%) | (+20%) |
| adjuster holds, peak per wing | - | 407 N | 254 N | 292 N | 247 N | 6.3 N m | 14.2 N m |
| lock moved | - | 0.58 mm | 0.001 mm | 0.43 mm | 0.001 mm | 3.5 deg | 4e-5 deg |

Geometry, read from the built models (`results/D4_geometry.jsonl`):

| | VL | VR | VP |
|---|---|---|---|
| adjuster | slide on the wing lever | slide on the crank arm | carrier hinge on the hull |
| adjuster travel between the settings | 58.5 mm | 20.0 mm | 117.4 deg |
| adjuster rides on | the flapping wing | the turning crank | the hull (its crank turns on it) |
| lowest point below the shoulder | 199.6 mm | 160.5 mm (both settings) | 149.8 mm |
| furthest point outboard of the shoulder | 118.5 mm | 90.6 mm | 172.7 mm |
| least angle between a link to the wing and the lever, take-off / cruise | 30.6 / 56.0 deg | 40.1 / 65.3 deg | 16.7 / 37.3 deg |

Take-off at gear 32, for all three as for FB: the motor is on its limit for
55-71% of the beat and the beat falls to 0.93-0.96.

## Against the frozen checks

"a", "b" are the two locks. A check read on "b" is read on a variant added
after the vacuum smoke rows; both are given so nothing rests on the choice.

| | frozen | read | |
|---|---|---|---|
| V10 | dedicated FB re-run reproduces D1b's take-off rows within 0.5% (read first) | electrical 179.793, 149.417, 119.244 W and lift 11.0634, 11.3389, 11.0167 N: identical to the stored rows | holds |
| V0 | take-off model moved by its adjuster alone: loops within 0.001 mm, the cruise model's stroke within 0.01 deg at 36 angles | loops within 2e-12 mm; stroke within 3e-8 (VL), 5e-8 (VR), 5e-5 deg (VP) | holds for all three |
| V1 | vacuum: mean motor plus servo power within 1% of peak; gap under 0.5 mm | VL 0.00-0.12%, VR 0.00-0.17%, gaps 0.006-0.37 mm, both points and locks; VP cruise 0.01%, 0.026 mm; **VP take-off 0.98% (a) and 2.15% (b), gap 0.80 and 0.90 mm**, 5 deg off its kinematics with no air | holds for VL, VR, VP at cruise; **VP fails at take-off** |
| V2 | error against kinematics: VR under 0.5 deg at both; VL under 0.3 at cruise, 0.3-1.0 at take-off (may fail the 0.5 gate); VP 0.2-1.0 at both | lock a: VR 0.09 and **0.78**; VL 0.03 and **1.73**; VP 0.40 and **6.59**. Lock b: VR 0.03 and 0.23; VL 0.01 and 0.22; VP 0.13 and **6.03** | lock a: cruise holds for all; take-off **wrong for VR** (over the gate), **wrong for VL** (over my range, fails the gate as I said it might), **wrong for VP**. Lock b: VL and VR pass the gate at both points; VP's take-off does not at either lock |
| V3 | amplitude within 1.0 deg; mid within 0.3 deg; down share within 0.02 of closed form (VL 0.517, 0.483; VR 0.506, 0.497; VP 0.542, 0.458); D1's K3 (0.50 +- 0.02) holds for VR, VL and fails for VP at both points | cruise: all hold. Take-off, lock a: amplitude **86.32 (VL), 86.04 (VR)**, mid **-0.32 (VL)**; lock b: 85.28, 85.26, mid -0.06, +0.04. Down share within 0.007 of the closed form everywhere it is readable. K3: VR 0.498-0.501 holds; VL 0.484-0.485 at cruise holds, at take-off 0.514 (b) holds and **0.524 (a) does not**; VP 0.460-0.462 at cruise fails as predicted | holds on lock b; on lock a the take-off amplitude is over by 0.04-0.32 deg for VL and VR. VP at take-off not readable |
| V4 | against the baseline: cruise lift 3%, thrust 0.05 N, mechanical power 5%; take-off lift 6%. Against FB: VL, VR lift 2% (cruise), 3% (take-off); VP 5% at both | cruise: VR -0.2%, -0.022 N, -1.2%; VL **-3.2%, -0.070 N**, -0.9%; VP **-5.5 to -6.1%, -0.28 N, +40%**. Take-off against the baseline: VR **+6.5 to +7.5%**, VL **+11.5 to +13.6%**. Against FB: VR -0.5% and +2.6% (b) / **+3.6%** (a); VL **-3.5%** and **+7.4 to +9.4%**; VP **-5.9 to -6.5%** at cruise | **holds for VR at cruise; wrong elsewhere.** The dedicated FB itself is +3.8% on the baseline at take-off, so VR's excess over FB is the smaller figure |
| V5 | best gear 48 (cruise), 24 (take-off) for all; electrical within 5% of FB | gears as predicted for VL and VR, and VP at cruise; **VP has none at take-off**. VR +0.5%, -1.2% to +0.3%; VL +1.1%, -2.6% (a) / **-6.0% (b)**; VP **+44 to +45%** at cruise | holds for VR; VL holds on lock a and is 1 point over (cheaper) on b; **wrong for VP** |
| V6 | loop force: take-off VR 250-350, VL 300-600, VP 130-260 N; cruise VR 30-60, VL 20-45, VP 15-40 N; gap under 0.5 mm; FB at cruise 35-50 N | take-off: VR 254-293; VL 425-461 (a), **262-295 (b)**; VP **760-1930**. Cruise: VR 45; VL 28; VP rod end 28-29, **beam middle 59**. Gap: **VL a at take-off 0.537 mm**, VP take-off 1.0-1.2 mm, the rest 0.03-0.28. FB cruise 43 N | holds for VR; VL in range on lock a (gap just over), under my range on b; **wrong for VP** (the middle carries both rods' load; take-off see below) |
| V7 | adjuster holds: take-off VL 100-400 N, VR 100-300 N, VP 1.5-5 N m; cruise VL 5-25 N, VR 10-50 N, VP 0.4-1.5 N m; VP at cruise over 3 times a quarter of the mean crank torque | take-off: VL **407** (a), 254 (b); VR 292, 247; VP **6.3, 14.2**. Cruise: VL 15.8; VR 36.6; VP 0.94 N m = 5.8 times a quarter of the mean crank torque (0.16 N m) | cruise holds for all, the circulating load included; take-off holds for VR and VL (a: 7 N over); **wrong for VP** |
| V8 | motor on limit 0 at the best gear; at take-off gear 32 over 40% on the limit and the beat under 0.98 | 0 for VL, VR at both points and VP at cruise; gear 32: 68-71% and 0.96 (VL), 55-57% and 0.95 (VR), 66-67% and 0.93-0.94 (VP) | holds (VP's take-off has no best gear) |
| V9 | lowest point VL 200, VR 160 / 140, VP 150 mm, within 2 mm; outermost 119, 91, 173 mm | 199.6; 160.5 / **160.5**; 149.8; outermost 118.5, 90.6, 172.7 | holds but for **VR at cruise**: the pin's circle reaches 140.5 mm, the arm it slides on keeps its take-off length and reaches 160.5 |

## What this says, as ranges and differences, not a choice

- One crank mechanism can make both strokes. For VR and VL the cost of
  doing so, against a linkage built for the one point, is inside 6% of
  electrical power at either point, with the same gear, the same beat and
  the motor off its limit. The numbers differ between the two locks by up
  to 5 W of 145 at take-off (VL), which is the size of what the lock's
  compliance in this engine does.
- The stroke each makes is not the dedicated four-bar's. VR's waveform is
  close to it (second harmonic 0.084 against 0.093 at take-off, 0.048
  against 0.044 at cruise) and so is its lift. VL's is not (0.035 at
  take-off, 0.082 at cruise; down share 0.514-0.524 and 0.484) and its lift
  moves by -3.5% and +7.4 to +9.4% against FB. VP's cruise stroke spends
  46% of the beat going down and costs 40% more mechanical power for 6%
  less lift and 0.28 N less thrust. Which part of a waveform does this was
  not separated here: no run varies one harmonic alone.
- What the adjuster must hold during a beat is of the order of the link
  force: 250-410 N at take-off on a 60 mm lever for both slides (86-100% of
  the peak loop force), 16 N (VL) and 37 N (VR) at cruise. VP's carrier
  holds 0.94 N m at cruise, 5.8 times its share of the mean crank torque:
  the two cranks at 117 deg load each other. That circulating load costs
  nothing in this model because the gear meshes are lossless equalities.
- VP as frozen cannot be read at take-off, and the reason is its geometry,
  not only the engine: with the cranks in phase its rods lie within
  16.7 deg of the lever's line near the bottom of the stroke (crank 295 deg;
  measured on the built model after the fact, not a frozen quantity), so
  the wing's reversal there is carried at about 1/sin(17 deg) of the
  torque's force, 1130-1400 N in a vacuum already. My section-0 probes
  measured the rod-to-beam angle and not the rod-to-lever angle. A VP
  re-proportioned for that angle was not built: it would be a candidate
  added after data and needs its own frozen addendum.

## What these numbers rest on

As D1: a witness motor, no gearbox loss, no bearing or slide friction,
ideal regeneration, links with no fluid load, the feather and twist
prescribed, the body held. In addition here: the adjuster is an ideal lock
(no drive of its own, no backlash, no time to move); gear meshes, VP's
phase gear included, are lossless joint equalities and the idler two
same-way cranks need is not a body; slider, pin block and carrier are 2 g
point masses. Not checked by the model: whether a link passes through the
hull or through another link. All three reach 150-200 mm below the
shoulder and 91-173 mm outboard of it. VL's adjuster rides on the flapping
wing and VR's on the turning crank; nothing here models how either would be
moved or held.

## The drawings against the design

`figures/D4_VL.png`, `D4_VR.png`, `D4_VP.png`: the right wing's mechanism
from the compiled model at both adjuster settings and crank angles 0, 90,
180, 270 deg from the built pose, with the dimensions measured on the model
in each title. Opened and checked against PREREG section 2:

- VL: crank centre (77.80, -160.00) mm, radius 39.62, coupler 163.71,
  lever 60.00 and 118.52 mm. As frozen (77.8, -160.0; 39.6; 163.7; 60.0,
  118.5). Strokes at crank 90 / 270: +42.5 / -42.2 and +19.9 / -19.8 deg.
- VR: crank centre (50.17, -120.00), coupler 120.17, lever 60.00, radius
  40.48 and 20.49 mm. As frozen (50.2, -120.0; 120.2; 40.5, 20.5). The arm
  is drawn at its take-off length at both settings, the pin block part way
  along it at cruise.
- VP: centres (34.84, -111.98) and (134.84, -111.98), radii 37.83, rods
  120.00, beam halves 50.00, lever 60.00 mm, phase 0 (printed as 360) and
  117.41 deg. As frozen. At take-off the rods and beam are a parallelogram
  at every angle drawn; at crank 270 the rods are seen nearly along the
  lever.
- Every loop on every drawn pose closes within 1e-11 mm; the stroke over 72
  crank angles agrees with the closed form within 1e-6 deg (VL, VR) and
  0.001 deg (VP, a table of 720 points).

Mismatches: (1) VR's lowest point at cruise, 160.5 mm against the 140 mm
of V9, as above. (2) The dotted circle is a 40 mm circle drawn inboard of
the shoulder for scale; the hull's place relative to the shoulder was not
read from the model. (3) The first drawing was wrong and was redone: the
pose solver jumped to another branch of the loop when asked for crank
angles 90 deg apart (strokes of -161 deg); it now walks in 5 deg steps.
The V0 and V9 numbers are from the corrected run.

## Disclosed: added or changed after seeing data

1. **The second lock ("b").** One-beat vacuum smoke rows of each candidate
   at take-off, gear 24, run to see that the models step, showed the lock
   moving 0.4 mm (the slides) and 2.3 deg (VP's carrier): an equality in
   this engine is as stiff as the inertia on its coordinate, and the
   adjuster joints had 2 g or less. Inertia on the locked coordinate itself
   (joint armature: 1 kg for the slides, 0.1 kg m^2 for the carrier) was
   added as a second variant before any air row; on a coordinate that does
   not move it changes nothing but the lock's stiffness (lock moved under
   0.002 mm and 4e-5 deg). The pre-registration says "an ideal lock,
   solref 2 dt" and fixes no armature, so both are run and both are in
   every table. No tolerance was changed.
2. In the same smoke rows VP was also tried with 0.001 kg m^2 on its second
   crank (`gear_inertia` in `va.py`); it changed the error from 4.7 to
   4.4 deg and was not carried into the sweep. The option stays in the
   code, off.
3. A vacuum trace of VP at take-off (scratchpad `d4_trace.py`, the
   `trace_every` hook in `va.py`) to find where its error arises: at crank
   300-340 deg, the bottom of the stroke.
4. The least angle between a link to the wing and the lever, added to
   `draw_d4.py` and the geometry table after the sweep, to state VP's
   failure; it is not a frozen quantity and no check uses it.
5. VP's gear-mesh slip and torque are recorded (0.05-3.5 deg, 0.9-14 N m);
   not in the pre-registration.
6. The largest loop gap is taken over every recorded step of the beats and
   each loop's peak force is recorded separately (the coordinator's notes
   after phase 1), as well as the single maximum `Drive` reports.
7. For a candidate with no gear passing the rule, the reader shows gear 24
   (take-off), labelled; the pre-registration did not say which to show.

Not run, as frozen: the half-size motor, link mass sensitivity, other
levers. Not committed; no existing file edited.
