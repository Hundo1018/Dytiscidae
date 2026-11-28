# Pass D4, pre-registration: one crank mechanism for both strokes

2026-11-17, on `53d3c76`. Frozen before any candidate is built in the model
or measured in air. D1 found (`RESULT_D1_stroke_drive.md`): a crank's stroke
amplitude is fixed by its geometry, so no crank built for cruise (2.5 Hz,
stroke 40 deg, 8 m/s) can do take-off (4.5 Hz, stroke 85 deg, at rest, body
at 90 deg); each point had its own linkage. This pass builds three mechanisms
in which one motor-driven crank mechanism makes both strokes, the amplitude
changed by a slow adjuster that is locked during a run.

This pass chooses nothing. It builds candidates and reports a table. No
material, construction or part is named.

## 0. Seen before freezing (probes, disclosed; no fluid, no dynamics of the flapper)

Stored rows read: `results/D1_sweep.jsonl`, `results/D1b_lever.jsonl` (the
dedicated four-bar FB: cruise, lever 30 mm, gear 48: 7.32 W electrical,
86 N peak loop force, 0.20 deg off its kinematics; take-off, lever 60 mm,
gear 24: 149.4 W, 296 N, 0.26 deg, 0.32 mm gap, beat 1.000; gear 32: beat
0.951, on the limit 55%). Baseline: cruise 7.25 W, 5.883 N; take-off
129.2 W, 10.923 N.

Scratchpad scripts, all closed-form planar kinematics in a wing's stroke
plane (u outward along the span from the shoulder, v up), none in the repo:

- `probe_d4_fourbar.py`, `probe_d4_digits.py`: a four-bar with a fixed crank
  centre and coupler, solved (scipy `fsolve`) for stroke 85 and 40 deg with
  the mid of the stroke's extremes at 0 at both settings, the setting being
  the lever length (VL) or the crank radius (VR), over crank-centre depths
  80-200 mm. Seen: the dimensions of section 2; VL did not solve from my
  start shallower than 140 mm; the closed-form waveform figures quoted in
  V3; the coupler-to-lever angle departs from 90 deg by up to 59 deg (VL,
  take-off), 34 (VL, cruise), 50 (VR, take-off), 25 (VR, cruise), against
  the dedicated FB's 46 and 22.
- `probe_d4_sum.py`, `probe_d4_sum2.py`, `probe_d4_final.py`: two cranks
  with a rod each to the two ends of a beam whose middle is pinned on the
  wing lever (two unknowns per crank angle, stroke and beam angle, by Newton
  continuation). Four layouts seen. (i) Both cranks about one centre: at a
  phase of 78-94 deg the beam swings +-50 to +-100 deg and the second
  harmonic is 0.08-0.10; dropped. (ii) Separate centres, turning the same
  way, sized as D1's FB and not re-solved: the cruise stroke's middle sits
  4.4-5.9 deg low (each rod leans as its pin moves across); so the crank
  centre and radius are solved for a middle at 0 at both settings, as for
  VL and VR. (iii) Separate centres turning opposite ways (a plain gear
  pair): down share 0.500 by symmetry and the lowest second harmonic
  (0.02-0.04), but at the cruise phase a rod comes within 0-19 deg of the
  beam's line (a dead point); dropped. (iv) Layout (ii) solved, beam half
  length 40, 50, 60 mm: least rod-to-beam angle at cruise 21, 37, 45 deg;
  50 mm taken. The figures quoted in V3 for VP are from (iv).
- A toy model unrelated to the flapper (`probe_d4_digits.py`, last lines):
  a 0.5 kg mass on a vertical slide held by a one-joint `<equality joint>`
  at solref 2 dt reads `efc_force` 4.905 N = its weight, sag 4e-5 mm. So a
  lock built this way reports its holding force.

Nothing was compiled from the flapper and no force, power or lift of any
candidate has been computed.

## 1. What is the same for every candidate

As D1 section 1, unchanged: W7's cruise point (8 m/s, level, 2.5 Hz, stroke
40 deg, tip feather 15 deg by strips, root feather held at -2.94 deg) and
take-off point (at rest, body at 90 deg, 4.5 Hz, stroke 85 deg, root
feather 50 deg on its ideal servo), table law, `freerig.run_free`, body
held, dt 1e-4 s, every loop and every lock at solref 2 dt; the feather and
strip twist prescribed and phased to the mechanism's own stroke (crank
angle less the offset at which the closed-form stroke's fundamental crosses
zero rising); the witness motor (tau_s 1.0 N m, omega_0 1000 rad/s, rotor
2e-5 kg m^2 behind an ideal gear N, copper loss tau^2 omega_0/tau_s) on one
crank, the other wing's cranks geared to it (joint equalities); links as
capsules of 15 g/m carrying no fluid load; the lever a point on the wing's
span line (its feather axis).

- Gears: cruise N in {16, 24, 32, 48}, take-off N in {16, 24, 32}. Best =
  least mean electrical power among rows whose stroke is within 2 deg of
  the intended amplitude **and** whose beat is at least 0.98 of the planned
  (D1 added the second condition after its first sweep; here it is frozen
  in advance).
- References run in the same session with the same code: the baseline
  (prescribed sinusoid, ideal torque) at both points, and the dedicated FB
  of `drive.Drive` with a 60 mm lever at both points over the same gears.
- Each candidate is one model generator whose only argument that differs
  between the two points is the adjuster's coordinate. The model is
  compiled with the adjuster at its setting, where every loop closes at
  stroke 0, and the adjuster is a joint held there by a one-joint equality
  (an ideal lock, solref 2 dt). **Holding force** (N, a slide) or **torque**
  (N m, a hinge) = the peak over the recorded beats of that lock's
  constraint force, per wing, the larger of the two wings.
- The adjuster's own drive, friction, backlash and the time to move it are
  not modelled. Gear meshes are joint equalities: no loss, so power that
  circulates between two cranks costs nothing here; stated with the result.
- Slider and pin block: a 2 g point mass each (assumed; at 4.5 Hz and
  40 mm its centrifugal load is under 0.1 N).

## 2. The candidates (dimensions solved in section 0; the code re-solves the same conditions and must reproduce them within 0.1 mm and 0.1 deg)

Conditions solved for each: stroke 85.00 deg at the take-off setting and
40.00 deg at the cruise setting, the mid of the stroke's two extremes at
0.00 deg at both, by one fixed set of link lengths and centres.

- **VL, the coupler's end slides along the wing lever.** A four-bar; the
  coupler's upper end is pinned to a slider on a slide joint along the
  wing's span line. Crank centre (77.8, -160.0) mm, crank radius 39.6 mm,
  coupler 163.7 mm. Lever 60.0 mm (take-off) and 118.5 mm (cruise):
  adjuster travel 58.5 mm. The adjuster rides on the flapping wing.
- **VR, the crank pin slides along the crank arm.** A four-bar with a
  60 mm lever; the pin block is on a radial slide on the crank and the
  coupler hinges on it. Crank centre (50.2, -120.0) mm, coupler 120.2 mm.
  Crank radius 40.5 mm (take-off) and 20.5 mm (cruise): travel 20.0 mm.
  The adjuster rides on the turning crank.
- **VP, two cranks at a phase, summed by a beam.** Two cranks of radius
  37.8 mm turning the same way at the same speed, centres (34.8, -112.0)
  and (134.8, -112.0) mm, a 120.0 mm rod from each pin to one end of a
  beam of half length 50.0 mm, the beam's middle pinned on a 60 mm lever.
  The second crank leads the first by the phase: 0 deg (take-off, the rods
  and beam a parallelogram) and 117.4 deg (cruise): travel 117.4 deg. In
  the model the second crank turns on a carrier hinged on the hull about
  the same axis; the carrier is the locked adjuster, and the crank is
  geared 1:1 to the first crank relative to the carrier (the idler that
  two same-way cranks need is not a body). The motor is on the first crank.
  What the lock holds is therefore the torque the second crank passes.

`connect` loops per wing: VL 1, VR 1, VP 2, each built closed at stroke 0.

Not built, stated: a six-bar whose control link pivots on an adjustable
point of the hull (its adjuster would not ride on a moving part; left out
to keep to three); the same-centre and the opposite-turning variants of VP
(section 0); direct drive (D1's DD already changes amplitude at will).

## 3. Checks and predictions

Each is falsified by its stated numbers. Tolerances are not loosened after
data; anything added after data is disclosed as added.

- V0, one mechanism. On the take-off model of each candidate, moving only
  the adjuster's joint by the travel above and re-solving the dependent
  joints (`mj_kinematics`, no dynamics) closes every loop within 0.001 mm
  at 36 crank angles and gives the cruise model's stroke within 0.01 deg
  at each. Fails: the two models are not one mechanism and the candidate
  is not reported as one.
- V1, vacuum (as D1's K1, in the form D1 found correct). Fluid off, gravity
  on, each candidate at its best gear, both points: mean of motor shaft
  plus servo power within 1% of its peak; loop gap under 0.5 mm.
- V2, kinematics in air (as K2; the gate stays D1's 0.5 deg). Stroke
  against the candidate's closed form with the adjuster at its nominal
  setting, at the best gear: VR under 0.5 deg at both points; VL under
  0.3 deg at cruise and 0.3-1.0 deg at take-off (the lock gives along a
  lever the coupler meets at 31 deg: I expect VL may fail the gate there);
  VP 0.2-1.0 deg at both (two loops, a gear and a lock in series; the
  least certain). A candidate over 0.5 deg at a point has its rows at that
  point labelled, as D1 did for SY.
- V3, waveform at the best gear. Amplitude within 1.0 deg of 85 and 40;
  mid of extremes within 0.3 deg of 0; downstroke share within 0.02 of the
  closed form: VL 0.517 (take-off) and 0.483 (cruise), VR 0.506 and 0.497,
  VP 0.542 and 0.458. Closed-form second harmonic, reported: VL 0.036 and
  0.082, VR 0.084 and 0.048, VP 0.057 and 0.080 (dedicated FB at 60 mm:
  0.093 and 0.044). D1's K3 (0.50 +- 0.02) is therefore predicted to hold
  for VR and VL and to **fail for VP at both points**.
- V4, lift. Against the baseline: cruise lift within 3%, thrust within
  0.05 N, mechanical power within 5%; take-off lift within 6% at the best
  gear. Against the dedicated FB at the same point, each at its best gear:
  VL and VR lift within 2% (cruise) and 3% (take-off); VP within 5% at
  both, no direction predicted (its halves of the beat are unequal).
- V5, power and gear. The best gear by the frozen rule is 48 at cruise and
  24 at take-off for all three. Electrical power at the best gear within
  5% of the dedicated FB's at that point (expected near 7.3 W and 149 W).
- V6, peak force in a loop closure at the best gear. Take-off: VR
  250-350 N, VL 300-600 N, VP 130-260 N (each rod end carries about half).
  Cruise: VR 30-60 N, VL 20-45 N, VP 15-40 N. Loop gap under 0.5 mm
  everywhere. Dedicated FB at 60 mm, for scale: take-off 296 N stored;
  cruise 35-50 N predicted (86 N at 30 mm).
- V7, what the adjuster holds, peak per wing at the best gear. Take-off:
  VL 100-400 N, VR 100-300 N, VP 1.5-5 N m. Cruise: VL 5-25 N, VR
  10-50 N, VP 0.4-1.5 N m. And for VP at cruise: the peak holding torque
  is over 3 times a quarter of the mean motor torque at the crank (mean
  shaft power over crank speed, shared by four cranks), because the two
  cranks at 117 deg load each other.
- V8, the motor. On its torque-speed limit for 0% of the beat at the best
  gear at both points, for all three. At take-off gear 32: on the limit
  over 40% and the beat under 0.98, for all three (as the dedicated FB).
- V9, size, read from the built model's link centre lines over a crank
  revolution at each setting; reported, not judged. Lowest point below
  the shoulder: VL 200 mm at both; VR 160 mm (take-off) and 140 mm
  (cruise); VP 150 mm at both; each within 2 mm. Furthest point outboard
  of the shoulder: VL 119 mm, VR 91 mm, VP 173 mm.
- V10, session guard, read first. The dedicated FB re-run at take-off,
  lever 60 mm, gears 16, 24, 32 reproduces the stored D1b rows' electrical
  power and lift within 0.5%. Fails: stop and report before reading
  anything else.

A failed V0, V1 or V2 means the candidate's loop is not modelled well
enough at that point to read its other rows; it is reported and the rows
labelled. Not run in this pass: the half-size motor, link mass sensitivity,
any lever other than those above.

## 4. Output

New files only: `drive/va.py` (the three candidates on `drive.Drive`'s
interface, `drive.py` unedited), `drive/d4_run.py`, `drive/d4_read.py`,
`drive/draw_d4.py`; rows in `results/D4_*.jsonl`; `figures/D4_*.png`, each
candidate from the built model at both adjuster settings and four crank
angles, opened and checked against section 2 (every mismatch listed);
`RESULT_D4_variable_stroke.md`: a table of the candidates at both points
(stroke amplitude, mean stroke, down share, second harmonic, error against
kinematics, loop gap, peak loop force, beat made, motor on limit,
mechanical and electrical power, lift against the baseline and against the
dedicated FB, adjuster travel, peak holding force or torque, lowest
point), each check above against what was read, and no pick.
