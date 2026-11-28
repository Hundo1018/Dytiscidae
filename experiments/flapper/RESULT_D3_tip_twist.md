# Pass D3, result: what makes the spanwise twist at cruise

2026-11-17. Frozen text: `PREREG_D3_tip_twist.md` (`e220330`). Code:
`drive/twist.py` (`TwistDrive`: D1's four-bar on the chain wing, plus the
twist mechanism), `drive/d3_run.py`, `drive/d3_read.py`, `drive/draw_d3.py`.
Rows: `results/D3_grid.jsonl` (25) and `results/D3_trim.jsonl` (7), at the
frozen step; `results/D3_grid_dt5e-5.jsonl` (25) and
`results/D3_trim_dt5e-5.jsonl` (8), the same rows at half the step, **added
after the frozen grid was read** (why: below). No row raised, all finite, the
load limiter never bound. Nothing is chosen here.

## Outcome in one paragraph

At the 8 m/s cruise (2.5 Hz, stroke 40 deg by D1's four-bar, root feather
held at -2.94 deg) **a tube from the tip panel with a horn at the shoulder,
turned by a rod from a second arm on the stroke's crankshaft (TX), makes the
tip twist the ideal servos make (T0)**: 14.3-14.9 deg at a lead of -86 to
-87 deg (T0: 15.3 at -87), with one motor. With hinge springs of 10 N m/rad
it reads 0.92 weights, +0.02 N and 7.3 W mechanical, 7.5 W electrical,
against T0's 0.94, +0.03 N, 7.3 and 7.5 W; the rod carries 27 N and its loop
stays closed to 0.04 mm. How stiff the wing is between root and tip sets the
rest: at 3 N m/rad the wing sags and lift is 0.74, at 30 the rod carries 68
N and the hinges 1.6 N m. **Springs alone (TP) twist the right way and at
nearly the right phase (lead -91 to -101 deg) but not enough or around the
wrong mean**: flat-built, the wing washes out and lifts 0.29-0.86; built
with 15 deg of washin at 3 N m/rad it swings 10 deg about -1.6 deg and reads
0.83 weights, +0.006 N, 8.6 W. **The cable from the hull (TC) adds lift to a
washed-out wing and delays the twist by 10-18 deg where it acts; where the
wing is already near its shape it hardly acts.** At the frozen step the rows
with a free tip (TP, TC) carry a numerical oscillation of the four-bar's
loop and are not readable for thrust or power; they are read at half the
step, a change made after seeing data.

## What was the same for every candidate

W7's wing of 6 panels, camber 0.04, camber moment on, table law, hinges 0.1
chord behind the leading edge, body held, 0.60 kg, 8 m/s level, 2.5 Hz,
stroke 40 deg, root feather on its ideal servo at -2.94 deg (not re-trimmed
in the grid). Stroke: D1's four-bar, gear 48, lever 30 mm, witness motor.
dt 1e-4 s, loops and the cable's limit at solref 2 dt; 2 beats held, 4 read.
Lead: the tip twist's fundamental against the stroke's as made; the servo's
reference reads -90 deg. Mechanical = motor shaft plus ideal servos;
electrical = motor shaft plus copper, plus the ideal servos' net power.

## The candidates at the frozen step (dt 1e-4), root feather -2.94 deg

| row | tip twist: amp / mean / lead, deg | lift, weights | thrust, N | mech / elec, W | coupler peak, N | rod or cable peak, N | largest gap, mm | hinge peak, N m | root feather hold, N m | beat | motor on limit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prescribed stroke, strips | 15 / 0 / -90 (by hand) | 0.990 | +0.048 | 7.26 / - | - | - | - | - | 0.45 | - | - |
| B, strips under the four-bar | 15 / 0 / -90 (by hand) | 0.994 | +0.022 | 7.12 / 7.32 | 86 | - | 0.144 | - | 0.46 | 1.000 | 0 |
| T0, servo on every hinge | 15.3 / +0.5 / -87 | 0.938 | +0.029 | 7.30 / 7.48 | 70 | - | 0.118 | 0.40 (servo; 0.52 W positive-only) | 0.46 | 1.000 | 0 |
| TX, springs 3 | 14.9 / 0.0 / -86 | 0.737 | -0.008 | 6.89 / 7.03 | 58 | rod 12.4 | 0.079 (rod 0.019) | 0.26 | 0.31 | 1.000 | 0 |
| TX, springs 10 | 14.7 / 0.0 / -86 | 0.915 | +0.022 | 7.31 / 7.47 | 67 | rod 27.1 | 0.091 (rod 0.043) | 0.62 | 0.67 | 1.000 | 0 |
| TX, springs 30 | 14.3 / 0.0 / -86 | 0.975 | +0.025 | 7.41 / 7.59 | 71 | rod 68.2 | 0.106 (rod) | 1.62 | 1.66 | 1.000 | 0 |
| TP k 1, rest 0 † | 16.7 / +8.9 / -100 | -0.080 | -1.488 | 2.29 / 2.38 | 307 | - | 0.417 | 0.14 | 0.23 | 1.000 | 0 |
| TP k 3, rest 0 † | 11.3 / +6.3 / -97 | 0.291 | -0.947 | 7.12 / 7.30 | 336 | - | 0.442 | 0.27 | 0.35 | 1.000 | 0 |
| TP k 10, rest 0 †† | 4.8 / +2.9 / -93 | 0.656 | -1.812 | 11.83 / 12.30 | 456 | - | **0.529** | 0.46 | 0.46 | 1.000 | 0 |
| TP k 30, rest 0 †† | 1.7 / +1.0 / -91 | 0.783 | -2.789 | 13.62 / 14.16 | 505 | - | **0.560** | 0.89 | 0.57 | 1.000 | 0 |
| TP k 1, rest -15 † | 17.0 / +4.9 / -101 | 0.116 | -0.794 | 3.16 / 3.21 | 208 | - | 0.314 | 0.19 | 0.26 | 1.000 | 0 |
| TP k 3, rest -15 † | 10.3 / -1.6 / -99 | 0.851 | -0.465 | 8.48 / 8.76 | 254 | - | 0.365 | 0.37 | 0.44 | 1.000 | 0 |
| TP k 10, rest -15 †† | 2.9 / -9.5 / -91 | 1.419 | -3.351 | 9.91 / 10.50 | 728 | - | **0.684** | 0.73 | 0.60 | 1.000 | 0 |
| TP k 30, rest -15 †† | 0.9 / -13.0 / -84 | 1.617 | -5.299 | 10.12 / 10.78 | 792 | - | **0.716** | 1.38 | 0.79 | 1.000 | 0 |
| TC k 1, rest 0 † | 14.1 / +5.0 / -118 | 0.166 | -0.907 | 4.65 / 4.76 | 269 | cable 438 (taut 37%) | 0.381 | 0.14 | 0.20 | 1.000 | 0 |
| TC k 3, rest 0 † | 10.5 / +4.3 / -112 | 0.446 | -0.351 | 7.42 / 7.57 | 223 | cable 446 (taut 26%) | 0.331 | 0.28 | 0.34 | 1.000 | 0 |
| TC k 1, rest -15 † | 15.3 / +2.6 / -111 | 0.270 | -0.563 | 4.58 / 4.66 | 174 | cable 253 (taut 27%) | 0.270 | 0.19 | 0.25 | 1.000 | 0 |
| TC k 3, rest -15 † | 10.2 / -2.0 / -98 | 0.865 | -0.188 | 8.59 / 8.85 | 198 | cable 196 (taut 10%) | 0.302 | 0.35 | 0.42 | 1.000 | 0 |

†† fails H1's gap (over 0.5 mm in air): labelled, not read as performance.
† passes H1 as frozen, but carries the same oscillation (next section): its
thrust, power and coupler force are not read as performance either. The
twist's amplitude, mean and lead, and the lift within 0.03, are the same at
both steps.

## Added after seeing the data: the free-tip rows at half the step

TP k 10 (rest 0) read -1.81 N of drag under the four-bar where W7's stored
passive wing read -0.01 N. Two diagnostic runs (scratchpad `p4_diag.py`,
`p5_diag.py`): the same wing on the prescribed stroke at dt 1e-4 reads
+0.051 N and 0.699 weights; under the four-bar at dt 1e-4 the holder's force
carries a 42-57 Hz oscillation (9.6 N in Fz at 49 Hz, 49 N in the coupler at
47 Hz) that is not in the right wing's hinge angles, the stroke's error or
the feather; at dt 5e-5 (loops at solref 2 dt = 1e-4) it is gone: +0.054 N,
0.707 weights, coupler 91 N. T0, TX and the strips do not have it at either
step. **Cause not found** (the left wing's hinges were not recorded); it
appears only with a free tip under the four-bar's loop. The whole grid and
the trimmed read were re-run at dt 5e-5, same code, same reader.

T0, TX and the baselines at half the step move by at most 0.005 N, 0.04 W
and 0.002 weights (T0 0.937, +0.034 N, 7.34 / 7.52 W; TX k 10 0.917, +0.026
N, 7.32 / 7.49 W, rod 27.2 N, rod gap 0.011 mm). The free-tip rows:

| row (dt 5e-5) | tip twist: amp / mean / lead, deg | lift, weights | thrust, N | mech / elec, W | coupler peak, N | cable peak, N (taut share) | largest gap, mm | hinge peak, N m |
|---|---|---|---|---|---|---|---|---|
| TP k 1, rest 0 | 16.8 / +9.0 / -100 | -0.091 | -0.728 | 1.74 / 1.78 | 36 | - | 0.016 | 0.14 |
| TP k 3, rest 0 | 11.3 / +6.4 / -97 | 0.291 | -0.157 | 6.45 / 6.57 | 51 | - | 0.022 | 0.27 |
| TP k 10, rest 0 | 4.8 / +2.9 / -93 | 0.707 | +0.054 | 10.82 / 11.15 | 91 | - | 0.039 | 0.41 |
| TP k 30, rest 0 | 1.7 / +1.1 / -91 | 0.859 | -0.117 | 12.00 / 12.40 | 104 | - | 0.045 | 0.45 |
| TP k 1, rest -15 | 17.0 / +5.0 / -101 | 0.092 | -0.468 | 3.20 / 3.24 | 37 | - | 0.016 | 0.19 |
| TP k 3, rest -15 | 10.3 / -1.6 / -99 | 0.826 | +0.006 | 8.57 / 8.83 | 86 | - | 0.037 | 0.37 |
| TP k 10, rest -15 | 2.9 / -9.5 / -91 | 1.405 | -1.062 | 8.43 / 8.80 | 112 | - | 0.048 | 0.48 |
| TP k 30, rest -15 | 0.9 / -13.0 / -85 | 1.585 | -1.786 | 7.06 / 7.44 | 115 | - | 0.049 | 0.48 |
| TC k 1, rest 0 | 14.2 / +5.0 / -118 | 0.178 | -0.393 | 4.33 / 4.40 | 58 | 472 (38%) | 0.025 | 0.14 |
| TC k 3, rest 0 | 10.5 / +4.3 / -112 | 0.448 | -0.097 | 7.19 / 7.33 | 64 | 389 (26%) | 0.027 | 0.28 |
| TC k 1, rest -15 | 15.3 / +2.7 / -111 | 0.246 | -0.333 | 4.66 / 4.73 | 49 | 473 (27%) | 0.021 | 0.19 |
| TC k 3, rest -15 | 10.2 / -2.0 / -98 | 0.855 | +0.003 | 8.67 / 8.93 | 87 | 245 (10%) | 0.037 | 0.35 |

The half-step rows are not shown to be converged: one cross-check only (TP k
10 rest 0 against the prescribed stroke at dt 1e-4: 0.707 against 0.699
weights, +0.054 against +0.051 N). The cable's peak force is a snap against
a limit whose stiffness is the step's (196-446 N at dt 1e-4, 245-473 N at
5e-5): it is an impact in this engine, not a load to size anything by.

## Second read: the root feather re-trimmed to carry the weight (H9)

| row | step | root feather, deg | thrust, N | mech / elec, W | tip amp / mean / lead | rod or cable peak, N |
|---|---|---|---|---|---|---|
| T0 | 1e-4 | -3.26 | +0.016 | 7.26 / 7.45 | 15.3 / +0.5 / -87 | - |
| TX k 3 | 1e-4 | -4.43 | -0.046 | 6.80 / 6.97 | 14.9 / 0.0 / -87 | rod 13.5 |
| TX k 10 | 1e-4 | -3.39 | +0.006 | 7.26 / 7.44 | 14.7 / 0.0 / -86 | rod 27.6 |
| TX k 30 | 1e-4 | -3.07 | +0.020 | 7.39 / 7.57 | 14.3 / 0.0 / -86 | rod 68.4 |
| T0 | 5e-5 | -3.27 | +0.021 | 7.30 / 7.49 | 15.3 / +0.5 / -87 | - |
| TX k 3 | 5e-5 | -4.43 | -0.041 | 6.83 / 7.00 | 14.9 / 0.0 / -87 | rod 13.5 |
| TX k 10 | 5e-5 | -3.38 | +0.010 | 7.28 / 7.46 | 14.8 / 0.0 / -87 | rod 27.7 |
| TX k 30 | 5e-5 | -3.06 | +0.019 | 7.36 / 7.54 | 14.4 / 0.0 / -86 | rod 69.3 |
| TP k 3, rest -15 | 5e-5 | -4.33 | -0.043 | 8.57 / 8.85 | 10.1 / -0.9 / -99 | - |
| TC k 3, rest -15 | 5e-5 | -4.13 | -0.038 | 8.65 / 8.92 | 10.0 / -1.3 / -97 | cable 231 (taut 8%) |
| TP k 10, rest 0 | 5e-5 | -5.01 | -0.004 | 10.95 / 11.31 | 4.7 / +3.3 / -94 | - |
| TP k 30, rest 0 | 5e-5 | -3.92 | -0.181 | 11.90 / 12.31 | 1.7 / +1.1 / -91 | - |

Every row trimmed within 0.5% in 3 runs. The frozen-step trim of the
free-tip rows (TP k 3 rest -15: -0.500 N; TP k 30 rest 0: -3.006 N; TC k 3
rest -15: -0.213 N) carries the oscillation and is in `D3_trim.jsonl` only.
The rule's eligible set differs between steps because TP k 10 rest 0 read
0.656 weights at dt 1e-4 (not eligible) and 0.707 at 5e-5.

## Against the frozen checks

"Frozen step" is the check as registered. "Half step" is the same check on
the added rows, given where the frozen-step rows are not readable.

| | frozen | read | |
|---|---|---|---|
| H1 | vacuum: total input within 1% of the motor's peak shaft power, T0, TX k 10, TP k 3, TC k 3, TC k 1; coupler and rod gaps under 0.5 mm in a vacuum and in air | vacuum: 0.00%, 0.28%, -0.11%, 0.61%, 0.69%; vacuum gaps 0.03-0.06 mm. In air: T0 0.12, TX 0.08-0.11, TP k 1 and k 3 0.31-0.44, TC 0.27-0.38, **TP k 10 and k 30 0.53-0.72 mm** | energy holds for all five; **gap fails for the four TP rows at k 10 and 30**: labelled. (Half step: every gap 0.05 mm or less.) |
| H2 | strips without the camber moment reproduce D1 (7.12 / 7.32 W) within 1%; strips with it: lift 0.97-1.01, 6.9-7.4 W, 0 to +0.05 N; T0: tip 15.0 +- 0.5, mean within 1, lead -90 +- 3, lift 0.92-0.98, 6.5-7.4 W, -0.04 to +0.04 N, hinge 0.3-0.5 N m, positive-only 0.3-0.7 W | 7.12 / 7.32; 0.994, 7.12 W, +0.022 N; T0 15.3, +0.5, -87.2, 0.938, 7.30 W, +0.029 N, 0.40 N m, 0.52 W | holds |
| H3 | TX at each k: tip 14-16, mean within 1.5, lead -90 +- 8; rod gap under 0.2 mm; stroke within 0.5 deg; beat 0.998 or more; never on the limit | 14.9, 14.7, 14.3; 0.0; -86; 0.019, 0.043, 0.106 mm; 0.14-0.16 deg; 1.000; 0 | holds |
| H4 | rod 6-20 N (k 3), 15-35 (k 10), 40-75 (k 30); hinge spring 0.45-0.70 N m at k 10; coupler within 25% of T0's | 12.4, 27.1, 68.2 N; 0.62 N m; 58, 67, 71 N against 70 (-17% at k 3) | holds |
| H5 | lift k 3 0.60-0.90, k 10 0.85-0.97, k 30 above k 10; thrust rises with k, k 10 within 0.05 N of T0's; power at k 10 within 10% of T0's; copper under 0.5 W | 0.737, 0.915, 0.975; -0.008, +0.022, +0.025 N (T0 +0.029); 7.31 against 7.30 W; copper 0.14-0.18 W | holds |
| H6 | TP rest 0, k 3, 10, 30: lead -115 to -75; amplitude 5-13, 2-6, 0.6-2.5; mean +8 to +16, +2.5 to +6, +0.8 to +2.5; lift 0.1-0.6, 0.5-0.8, 0.75-0.95; k 1: mean over +15, lift under 0.3 | lead -97, -93, -91; 11.3, 4.8, 1.7; mean **+6.3**, +2.9, +1.0; lift 0.291, 0.656, 0.783 (half step 0.291, 0.707, 0.859); k 1: mean **+8.9**, lift -0.08 | right way and right phase: holds. **Wrong on the washout at k 3 and k 1**: less than I said, because the wing unloads as it washes out (lift 0.29 and below), which I counted in the lift and not in the moment |
| H7 | rest -15, k 3: mean within 6 of 0, lift 0.8-1.15, the best thrust of the eight TP rows; no TP row has lift 0.97 or more with thrust 0 or more | mean -1.6, lift 0.851 (half step 0.826); thrust -0.465 N, the least negative of eight (frozen step, not readable); half step: +0.006 N, second to k 10 rest 0's +0.054 N; no row has both | first two hold; **"best thrust" holds on the unreadable rows and is wrong at half step**; the last holds at both steps |
| H8a | TC's lift at least its twin's less 0.02, four pairs | +0.246, +0.155, +0.154, +0.014 (half step +0.269, +0.157, +0.154, +0.029) | holds |
| H8b | taut over 40% at k 1 rest 0; under 20% at k 3 rest -15, that row within 10% of its twin in lift and power | **37%** (half step 38%); 10%, lift +1.6%, power +1.3% | **first wrong**, second holds |
| H8c | where taut is over 20%, the lead is later than the twin's by more than 15 deg | k 1 rest 0: 17.6 later; k 3 rest 0: 15.4; **k 1 rest -15: 9.9** (taut 27%) | **wrong for one of three** |
| H8d | no TC row has lift 0.97 or more with thrust 0 or more | none | holds |
| H8e | cable peak 3-150 N | **196-446 N** (half step 245-473) | **wrong**: every row snaps |
| H9 | T0: offset -3.0 to -3.6, 6.7-7.5 W, -0.04 to +0.04 N; TX k 10: offset within 0.6 deg of T0's, power within 10%, thrust within 0.05 N; TX k 3 thrust under 0; TX k 30 over k 10's | -3.26, 7.26 W, +0.016 N; -3.39, 7.26 W, +0.006 N; -0.046 N; +0.020 against +0.006 | holds (the same at half step) |
| H10 | drawings from the built model at four crank angles, checked | below | dimensions hold; two things seen |

## The drawings against the design

`figures/D3_T0_TP.png`, `D3_TX.png`, `D3_TC.png`: the built model stepped in a
vacuum, hull held, no gravity, snapped at crank 0, 90, 180, 270 deg of the
second beat; opened and checked. Outside freerig the hinges are held by the
model's own joint springs (the root feather by 200 N m/rad at -2.94 deg; T0
by stiff springs whose rest follows the servo's reference).

- Measured on the model, every drawing: hinge line 25.8 mm behind the
  leading edge, panels 110.0 mm, chord 257.6 mm, crank 10.26 mm, coupler
  60.00 mm, lever 30.00 mm. As registered.
- TX: tube 550.00 mm on the hinge line (0.00 mm off it), horn 30.00 mm, arm
  6.82 mm and 30.00 mm aft of the crank's plane, rod 65.23 mm. As registered.
  Tip twist at the four angles -14.0, +1.6, +15.0, -2.4 deg against the
  design's -14.2, +1.6, +15.2, -2.4. Largest gap over the whole beat: rod
  0.033 mm at crank 333 deg, coupler 0.034 mm at 89 deg.
- TC: post 80.00 mm, anchor 41.21 mm under the shoulder, cable limit 628.24
  mm, post 605.0 mm from the shoulder at rest. As registered. At crank 90
  (stroke +20.0) the cable is at its limit and the tip twist is 0.0 deg, the
  design's stop; slack at the other three (616, 610, 598 mm). Largest coupler
  gap over the beat 0.049 mm.
- T0/TP: tip twist -14.9, +2.1, +14.9, -2.1 deg; largest coupler gap 0.036
  mm at crank 101 deg.
- Seen, not in the registration: **the cable runs from under the shoulder to
  the top of the post in the trailing edge's own cross-plane, so it passes
  from below the wing to above it along the trailing edge line**; nothing in
  the model stops it passing through the surface. And, as in D1, the crank
  and arm reach 60-70 mm below the shoulder line, outside a hull of 40 mm
  radius; the tube lies on the same line as D1's lever point.
- In the TX drawing the root feather reads -2.79 to -3.08 deg: the drawing's
  own root spring giving under the rod's reaction, not the runs' servo.

## What happened on the way (disclosed)

- Phase 1's probes are in the registration's section 0.
- **The half-step rows and the two diagnostic runs were added after the
  frozen grid was read.** No tolerance was moved; every check is reported at
  the frozen step first.
- `d3_run.py` gained a trailing `dt=` argument for that re-run; the frozen
  rows were written before it existed and the default is unchanged.
- The trimmed read at half step has one more row than at the frozen step,
  by the frozen eligibility rule applied to that step's lifts.
- The drawing holds hinges with joint springs because it runs outside
  freerig; stated on the figure's script.

## What this says, as ranges and differences, not a choice

- A crank-geared tube and rod delivers the servo's twist with the stroke's
  one motor, inside 0.04 weights and 0.01 N of the servos at 10-30 N m/rad
  and within 0.15 W of them. Its bill is force and stiffness, not power: rod
  12-68 N, hinge springs 0.26-1.6 N m, root feather hold 0.31-1.66 N m,
  rising with the wing's torsional stiffness; lift falls from 0.98 to 0.74
  as the stiffness falls from 30 to 3.
- Re-trimmed, T0 and TX at 10 and 30 N m/rad carry the weight with thrust of
  +0.006 to +0.021 N for 7.3-7.4 W; TX at 3 is 0.04-0.05 N short.
- A passive wing's twist has the phase (-91 to -101 deg) and, at 3 N m/rad,
  two thirds of the amplitude. Its mean is the open quantity: flat-built it
  washes out; with 15 deg of washin at 3 N m/rad, re-trimmed, it is 0.04 N
  short for 8.6 W; at 10 N m/rad flat-built, re-trimmed, 0.004 N short for
  11 W. Only 0 and -15 deg of rest twist were run.
- The cable changes a washed-out wing (lift +0.15 to +0.27) and leaves a
  wing near its shape alone (+0.01 to +0.03); its twist is later by 10-18
  deg where it is taut over a quarter of the beat. Its forces in this
  engine are impacts.

## What these numbers rest on

- As D1 and W7: a witness motor, no gearbox or bearing loss, ideal
  regeneration; links, tube, horn, rod, post and cable without fluid load;
  the body held; six rigid panels, no bending, no hinge friction, no gaps;
  the springs' damping 5% of critical (typed, freerig's); the section's
  pitching moment the arc's.
- **The panel's mass (a box geom) is centred 38.6 mm aft of the hinge line,
  the fluid surface's mid-chord 103 mm aft**: the mass sits a quarter chord
  ahead of the surface the air acts on. Not changed. It bears on every
  inertial part of the twist (in a vacuum inertia alone swings a 3 N m/rad
  wing 3.0 deg in phase with the stroke's position, lead 0).
- The root feather is an ideal servo holding -2.94 deg (or the trim); in TX
  it holds the tube's reaction, 0.31-1.67 N m.
- TX's tube is rigid in torsion and has no bearing of its own: the chain's
  hinges carry the rod's side load rigidly. The cranks' 1:1 coupling and the
  arm on the crankshaft are ideal.
- The free-tip rows are read at dt 5e-5 because of an oscillation at dt 1e-4
  whose cause was not found; convergence beyond 5e-5 was not tested.
- TX was sized at each root feather it was run at (arm, angle, rod re-fitted
  by the frozen rule in the trim); TC's anchor and length were kept as built
  at -2.94.
- Cruise only. Nothing here reads take-off, where W7 found the wing needs
  30-100 N m/rad without a drive on the tip.
