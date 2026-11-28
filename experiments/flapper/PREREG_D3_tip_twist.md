# Pass D3, pre-registration: what makes the spanwise twist at cruise

2026-11-17, on `53d3c76`. Frozen before any candidate is run in air. Follows
W7 (the twist as joints, driven by ideal servos), D1 (the stroke by
mechanisms) and D2 (the root feather at take-off; its §0: the twist cruise
needs is the tip's and "a mechanism for it needs the hinged wing of W7 and is
a later pass"). The user: no real drive is fixed before several simulated
candidates exist; forces only from the simulated mechanism and fluid. This
pass chooses nothing.

## 0. Seen before freezing (every probe, disclosed)

No candidate has been run in air. Scratchpad `p1_names*.py`, `p2_design.py`,
`p3_smoke.py`:

- **`Drive.compile` cannot build on a chain wing**: it names the second wing
  `seg2_wing`, which with `wing_panels=6` is the first wing's second panel
  (the mirrored root is `seg7_wing`); it asserts. `drive.py` is not edited
  (another worker has it): `drive/twist.py` holds `TwistDrive(Drive)`, which
  builds D1's four-bar again for a chain (same geometry, same names, found by
  parentage) and adds the twist mechanism. Kinematics, motor, control,
  records and summary are `Drive`'s own.
- The built 6-panel wing (model inspected): panels 110 mm, hinges on one line
  along the span, each with its own stop at +-57.3 deg (the travel here is
  under 10 deg a hinge); the fluid's leading edge is on the built hinge line
  and freerig's `feather_axis=0.1` moves every hinge 25.8 mm aft; the fluid's
  trailing edge is 231.8 mm aft of the moved line. **The panel's mass (a box
  geom) is centred 38.6 mm aft of the moved hinge line, the fluid surface's
  mid-chord 103 mm aft**: the mass sits a quarter chord ahead of the surface
  the air acts on. Recorded, not changed; it bears on the inertial twist.
- W7's stored row re-run in air (S6, every hinge on its servo, no drive, W7's
  dt 2.5e-4 and trimmed offset -3.288; not a candidate): 7.095 W (stored
  7.102), lift 1.0004, thrust +0.014 N (+0.015), hinge torque 0.411 N m
  (0.411). The chain wing runs here as it did.
- Kinematics only (no dynamics, no fluid): TX's arm, fitted on the model's
  own kinematics, closes at every crank angle: arm 6.82 mm at -58.7 deg on
  the shaft, rod 65.23 mm, tip twist 15.03 deg amplitude, mean 0.01, 0.72 deg
  rms from the servo's reference. TC's stop (the twist at which the cable is
  taut) was computed for posts of 30, 50 and 80 mm: at 30 and 50 mm the stop
  was not a smooth function of the stroke near its top, at 80 mm it is (below).
  **80 mm was taken after seeing that.** Two errors in my solver for the
  cable (the bisection's direction and its bracket) were fixed before any
  dynamic run.
- Three smoke runs in a **vacuum** (no fluid; four-bar at gear 48, lever 30
  mm, 2 beats after 2 held): TX, springs 10 N m/rad: tip twist 14.7 deg
  (fundamental), mean 0.01, lead -86.5 deg against the stroke (the reference
  reads -90), rod 20.8 N peak, rod gap 0.033 mm, hinge spring 0.53 N m, motor
  shaft 0.018 W mean of a 6.2 W peak. TC, springs 3, rest 0: the cable taut
  12% of the beat and **241 N** when it snaps taut; shaft 0.031 W of a 5.05 W
  peak. TP, springs 3, rest -15 deg: tip mean -15.6 (the rest twist is made),
  inertia alone swings it 2.9 deg **in phase with the stroke's position**
  (lead +1 deg), not its speed.

## 1. Same for every candidate

W7's wing of 6 panels (`wing_panels=6`), camber 0.04, section `goe417a+lift`,
the camber moment on, table law, `cop_law="plate"`, hinges 0.1 chord behind
the leading edge. Cruise: 8 m/s, level, body held, 0.60 kg, 2.5 Hz, stroke
40 deg, **root feather held at -2.94 deg by its ideal servo and not
re-trimmed** (D2 §0: it does no work at cruise; lift and thrust are read at
the same root feather, as D1). The stroke by D1's four-bar at D1's cruise
lever and best gear: `("FB", 40 deg, 2.5 Hz, N = 48, a = 30 mm)`, crank 10.26
mm, coupler 60 mm, witness motor (1.0 N m, 1000 rad/s, rotor 2e-5 kg m2,
copper loss tau^2 omega_0/tau_s), links 15 g/m with no fluid load. dt 1e-4 s,
every loop and the cable's limit at solref 2 dt. 2 beats held, 4 measured.

Read for every row: tip twist as made (the sum of one wing's five hinge
angles: fundamental amplitude, mean, lead against the stroke as made; -90
deg is the reference's phase, `-A cos` against `sin`, D2's convention), each
hinge's swing, lift over weight, thrust (Fx), mechanical power (motor shaft
plus ideal servos), electrical power (motor shaft plus copper, plus the ideal
servos' net power), peak coupler force, peak rod or cable force, largest loop
gap over the measured beats, peak hinge torque (servo or spring), root
feather hold torque, beat made (`freq_ratio`), share of the beat the motor is
on its limit, stroke error against the four-bar's closed form.

## 2. The candidates

Baselines (not candidates): **B, strips**: one-panel wing, the fluid strips
turned by hand to 15 deg at the tip (W6, D1's cruise row), under the same
four-bar, with and without the camber moment; and the prescribed stroke with
ideal torque (D1's baseline) likewise.

- **T0, reference**: W7's ideal servo on every hinge, each tracking its
  share of `-15 cos(phase)` at the tip, phase from the four-bar's crank.
- **TP, passive**: a torsion spring at each of the five hinges, k in {1, 3,
  10, 30} N m/rad per hinge (GJ = k x 0.11 m), damping 5% of critical
  (freerig's, as W7), nothing driven. Two builds: springs at rest flat
  (rest 0), and at rest with the tip twisted 15 deg leading edge up against
  the root (rest -15, shared equally: a wing built with washin, which the
  air's nose-down moment then takes out). Why: W7 read a passive wing at a
  re-trimmed root and found it washed out 4-11 deg; the rest twist is the
  one real parameter that answers that, and W7 did not have it.
- **TX, a tube and a rod from the stroke's crank**: a tube fixed to the tip
  panel, running in along the hinge line 550 mm to the shoulder (15 g/m,
  rigid in torsion); on its inboard end a horn 30 mm aft, whose end lies on
  the stroke axis at rest (so the stroke alone does not move it); a second
  arm on each stroke crank's shaft, 30 mm aft of the crank's plane, radius
  6.82 mm, set -58.7 deg on the shaft; a rod with a ball at each end, 65.23
  mm, from the arm's pin to the horn's end (connect, solref 2 dt). Arm,
  angle and rod by one rule: least squares of the tip twist the model's
  kinematics gives against `-15 cos(phase)` over 48 crank angles, root
  feather at -2.94. The five hinges are springs k in {3, 10, 30} (W7: a
  tube sets the tip only; the wing between sags if soft). Why this and not a
  rod from the hull: the twist must follow the stroke's speed, a quarter
  beat from its position; a point fixed on the hull can only gear the twist
  to the position. Why it may assemble where D2's FX did not: 15 deg of
  travel, not 50.
- **TC, a one-sided cable from the hull**: a post 80 mm high on the tip
  panel's trailing edge at the panel's middle (605 mm from the shoulder),
  upper side; a cable (spatial tendon, length limit 628.24 mm, slack below
  it) from the post's top to a point on the hull's side 41.21 mm under the
  shoulder, in the same cross-plane. Taut, it stops the tip's washout at a
  twist that falls as the wing rises (kinematics): +27.1 deg at stroke -20,
  +21.3 at -10, **+15.0 at 0**, +8.1 at +10, **0.0 at +20**. Anchor drop and
  length by that rule (the two bold values) at the 80 mm post. Hinges are
  springs k in {1, 3} at rest 0 and -15, each row the twin of a TP row.
  Why: named in this pass's brief as a candidate kind (not the user's);
  it tightens through the upper half of the
  stroke, where the downstroke begins and the air starts to wash the tip out.

Not built, stated: a rod from the hull (two-sided gearing to stroke
position; TC reads that phase one-sidedly); a cable from the crank against
the springs; a cam; a motor on the tube (W7's "tube" arm is its ideal form,
D2's FM its copper bill). Not modelled: the tube's own torsion and its
bearings (the chain's hinges carry the rod's side load rigidly), fluid load
on tube, horn, rod, post and cable, the cable's mass and stretch, hinge
friction, any clash with the hull or with D1's lever on the same line.

## 3. Checks and predictions

- **H1, vacuum energy** (D2's G1). Fluid off, gravity on, for T0, TX k 10,
  TP k 3 (rest 0), TC k 3 and TC k 1 (rest 0): mean total input (motor shaft
  plus servos) within 1% of the motor's peak shaft power; coupler and rod
  gaps under 0.5 mm, in a vacuum and in air. A row that fails is labelled
  and its candidate's other figures are not read as performance. Predicted
  to hold for all five (seen: 0.3%, 0.6%, -0.3% in the smoke runs; TC k 1
  not seen and the most likely to fail, by its snapping).
- **H2, T0 and the baselines.** Strips under the four-bar without the camber
  moment reproduce D1's stored row (7.12 W mechanical, 7.32 W electrical)
  within 1%. Strips with it: lift 0.97-1.01, mechanical 6.9-7.4 W, thrust 0
  to +0.05 N. T0: tip 15.0 +- 0.5 deg, mean within 1, lead -90 +- 3; lift
  0.92-0.98 (W7's S6 needed -3.29 to carry the weight); mechanical 6.5-7.4
  W; thrust -0.04 to +0.04 N; worst hinge 0.3-0.5 N m; twist servos'
  positive-only power 0.3-0.7 W.
- **H3, TX makes the twist.** At each k: tip fundamental 14-16 deg, mean
  within 1.5, lead -90 +- 8; rod gap under 0.2 mm; stroke within 0.5 deg of
  its closed form; beat 0.998 or more; motor never on its limit.
- **H4, TX's loads.** Rod peak force 6-20 N at k 3, 15-35 N at k 10, 40-75 N
  at k 30; hinge spring peak 0.45-0.70 N m at k 10; coupler peak force within
  25% of T0's.
- **H5, TX's flight.** Lift rises with k: k 3 0.60-0.90, k 10 0.85-0.97, k 30
  above k 10. Thrust rises with k; k 10 within 0.05 N of T0's. Mechanical
  power at k 10 within 10% of T0's. Copper (electrical less mechanical)
  under 0.5 W at every k.
- **H6, TP twists the right way at the right phase, and washes out.** Rest
  0, k 3, 10, 30: the twist's lead is between -115 and -75 deg (leading edge
  down on the downstroke, a quarter beat behind the stroke's position: the
  air's doing; inertia alone reads near 0, §0). Amplitude 5-13, 2-6, 0.6-2.5
  deg; mean (washout, positive) +8 to +16, +2.5 to +6, +0.8 to +2.5 deg; lift
  0.1-0.6, 0.5-0.8, 0.75-0.95. k 1: mean over +15 deg, lift under 0.3.
  Falsified for "right way" by a lead outside -180..0, for "right phase" by
  one outside the band.
- **H7, TP built with washin.** Rest -15, k 3: mean within 6 deg of 0, lift
  0.8-1.15, the best thrust of the eight TP rows. No TP row has lift 0.97 or
  more with thrust 0 or more. Falsified by one that has.
- **H8, TC.** (a) Lift at least its TP twin's less 0.02, all four pairs (a
  stop on washout can only add incidence). (b) Taut share over 40% at k 1
  rest 0; under 20% at k 3 rest -15, and that row within 10% of its twin in
  lift and mechanical power. (c) Where the taut share is over 20%, the
  twist's lead is later than its twin's by more than 15 deg (towards -180:
  what the cable adds is geared to stroke position). (d) No TC row has lift
  0.97 or more with thrust 0 or more. (e) Cable peak force 3-150 N (wide,
  stated: static holding is a few N, a snap read 241 N in a vacuum).
- **H9, second read, re-trimmed** (the root feather moved until lift is the
  weight within 1%, 3% accepted, W7's secant, at most 6 runs; TX's arm and
  rod re-fitted by its rule at each offset; run for T0, TX at each k, and
  any TP or TC row whose lift at -2.94 was 0.7-1.3). T0: offset -3.0 to
  -3.6 deg, mechanical 6.7-7.5 W, thrust -0.04 to +0.04 N. TX k 10: offset
  within 0.6 deg of T0's, power within 10%, thrust within 0.05 N. TX k 3:
  thrust under 0; TX k 30: thrust over k 10's.
- **H10, the drawings.** From the built model at four crank angles (0, 90,
  180, 270 deg of the second beat, vacuum, held): T0/TP's wing (five hinges
  on one line 25.8 mm behind the leading edge, six panels of 110 mm, crank
  10.26 mm, coupler 60 mm, lever 30 mm), TX (tube 550 mm, horn 30 mm, arm
  6.82 mm, rod 65.23 mm, the tip twist at each angle against the design's),
  TC (post 80 mm, anchor 41.21 mm under the shoulder, cable 628.24 mm, taut
  or slack at each angle). Opened and checked; mismatches listed. The largest
  loop gap over the whole beat is recorded, not only at the drawn angles.

Falsified by the stated numbers. Tolerances are not moved after data;
anything added after seeing data is disclosed as such.

## 4. Output

`drive/twist.py`, `drive/d3_run.py`, `drive/d3_read.py`, `drive/draw_d3.py`;
`results/D3_grid.jsonl` (air and vacuum), `results/D3_trim.jsonl` (H9);
`figures/D3_*.png`; `RESULT_D3_tip_twist.md`: a table of candidates, each
check against what was read, no pick, no recommendation, no material or part
named. `freerig.py` and `drive.py` unchanged (the springs' rest twist is set
in `TwistDrive.setup` on freerig's own hinge springs).
