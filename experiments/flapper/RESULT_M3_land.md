# Pass M3, result: standing, walking and turning on the wings

2026-11-18. Frozen text: `PREREG_M3_land.md` (committed in `e70721f`, not edited).
Code: `m3_land.py` (arms and runner; sha256 `33113e20…` both at scoring time and
now), `m3_specs.py` (the scored run list), `m3_read.py` (the frozen gates),
`m3_freeze.py` (the freeze rule), `m3_table.py`, `m3_draw.py`. Rows:
`results/M3_tuning.jsonl` (146 tuning rollouts, none dropped),
`results/M3_land.jsonl` (171 scored rows: 127 main and 44 J), traces in
`results/M3_traces/`, frozen parameters in `results/M3_frozen.json` (sha256
`0e6971d5…`), the gate read in `results/M3_read.txt`, the table in
`results/M3_table.md`. Harness `mission.Machine` as of `e70721f`. CoM is read
through `Machine.com()`, which computes the same dry-mass `xipos` mean as the
probe code. The probe row `m3_probe.py stand -12 0 -30` reproduces to every
printed digit after the harness change (drift -0.0137 m, wings 2.921 N each,
peak stroke torque 4.363 N m, stroke 9.63 W). Every controller reads the true
state.

## Outcome

- **(a) Stand: yes.** Arm S (a flat-foot stance on both wing tips at stroke
  -12 deg, with a roll loop) passes the stand gate on 4 of 4 orientations
  (facing up, 45 deg, across, facing down). It also passes on 4 of 4 at each
  of the two smaller rotor inertias. Over 10 s the CoM drifts 3.5-4.6 mm and
  the yaw changes by at most 0.012 deg. The hull is off the ground the whole
  time and the wing tips carry all the weight. The hold costs 10.8-11.1 W of
  electrical power, almost all of it copper loss in the stroke motors; peak
  stroke torque is 2.0-2.2 N m. The do-nothing arm N0 lies on its hull (body
  share 0.96-0.98), slides 0.7-1.2 mm in 10 s, and fails the gate on hull
  contact, as designed.
- **(b) Walk along a commanded heading, both ways: no, at any pre-registered
  arm.** 0 of 6 pairs pass for G1, G2 or the added arm H. The gate read the
  null arms correctly: N0, N1 and U fail everywhere, and the F pairing check
  gives a pair mean of exactly 0 with equal hashes.
  - G1 is the set the frozen rule picked: a slow shuffle with a 3 s stride.
    It does not move: pair means are under 0.0015 m/s.
  - G2, alternating, comes closest. On the up/down pairs both halves make
    headway: 0.14-0.50 m up the ramp and 1.09-1.12 m down it in 12 s, pair
    means 0.052-0.066 m/s, above twice the untimed arm U. It fails on
    cross-track: the uphill half ends 0.45-0.95 m sideways. On the across
    pairs it fails because one direction goes backward (-0.13 to -0.67 m).
  - **G2 does not walk.** It hops on flapping thrust and has no ground
    contact on 65-81% of the window.
  - H (added after tuning, below) hops forward 1.3-3.3 m in 12 s whatever the
    command, and the opposite half goes the other way by 1.9-5.2 m. Its pair
    means are -0.099 to +0.014 m/s.
- **(c) Turn to a commanded heading, both ways: no.** 0 of 3 pairs pass for any
  arm.
  - G2 turns in the commanded sense by 47-79 deg in 14 s, ending 28-39 deg off
    the commanded heading against the 15 deg bar. On the way it tumbles:
    pitch from -70 to +87 deg, landing on its tail at 86 N.
  - G1 turns 3-4 deg.
  - H turns 3-11 deg on two starts, and on the third (facing down the ramp)
    49-62 deg with 31-40 deg of error.
- **What limits it.** The shoulder has two hinges: stroke, which moves the
  wing in the body's roll plane, and feather, which turns it about its span.
  There is no fore-aft sweep. A wing tip planted flat is a foot one chord
  long, and with both feet flat the body can only pitch about the shoulder
  line. The only way to move the shoulder along the body is to roll the foot
  on its trailing edge, a 0.232 m spoke. That pitches the body forward,
  because the CoM is 0.10 m behind the feather axis and ahead of any trailing
  edge placed short of the 57 deg hinge limit. The stance box also keeps the
  shoulder at least 0.07 m up.
  - What moved the machine in tuning was not stepping but flapping: a fast
    stroke at the end of the swing. In which direction it moves is set by the
    feather angles; running the cycle backward in time does not reverse it.
    Plant feather at -20 or less sends it forward, 0 or more sends it back
    (tuning grid t057-t076).
  - The frozen steering law walks backward by running the cycle in reverse,
    so no gait inside the box can do the second half of a pair. The antipodal
    pair (CLAUDE.md rule 8) caught this: it is locomotion that goes one
    body-fixed way whatever the command.
  - F2's 108 g machine walked on 4.5 cm "thumbs". This machine has none; a
    thumb is the separately pre-registered geometric arm, not run here.
- **The model is blind where the hops happen.** The hops (G2, H, the best
  tuning rows) sit in the regime where the strip solver is known to be wrong:
  large feather at low speed, with no dynamic stall (X2: forces off by 1.5-3.5
  times). The air's added mass is published into the wing bodies (0.083 kg
  against 0.041 kg dry on land). The slams are a rigid box on a rigid plane at
  solref 0.004: 24-56 weights on one wing, up to 652 N on the hull.
  - So the hop distances are not to be trusted. Only the walking negative
    stands, and it rests on geometry and contact, not on the fluid.
- The rotor sensitivity (J) changes no verdict.

## 1. The table (main arms, J_r 2e-5; J rows in section 3)

| arm | drive | pairs passed | along per half in 12 s (m) | P window (W) | peak P, all motors (W) | peak wing N (weights) | stroke tau (N m) | stroke on curve | airborne | pitch p2p (deg) |
|---|---|---|---|---|---|---|---|---|---|---|
| S stand | 6 motors; one crank could hold the symmetric stance, not the roll loop | stand 4/4 | drift 3.5-4.6 mm | 10.8-11.1 | 13-16 window, 131-277 run (the drop) | 4 (0.7-0.8) | 2.0-2.2 | 0 | 0 | 0 |
| N0 flat, held | none | 0 (stand fails on hull) | drift 0.7-1.2 mm | 0.1 | 0 window | 0 | 0.1 | 0 | 0 | 0 |
| N1 propped | 6 motors | 0/6 walk, 0/3 turn | 0.004 | 9.4-10.4 | 21-23 | 5-6 (0.9) | 2.3-2.4 | 0 | 0 | 0 |
| G1 (frozen by rule) | 6 motors, no crank gait | 0/6 walk, 0/3 turn | -0.12 to +0.13 | 9.5-15.1 | 472-741 | 20-32 (3.4-5.4) | 7.7-8.9 | 0 | 0.05-0.07 | 21 |
| U of G1 | 6 motors | 0/6, 0/3 | -0.14 to +0.15 | 18.6-21.2 | 623-938 | 35-85 | 12.7-15.8 | 0 | 0.15-0.22 | 21-24 |
| G2 (frozen by rule) | 6 motors, sides half a period apart | 0/6 walk, 0/3 turn | -0.67 to +2.67 | 124-154 | 2539-3158 | 142-296 (24-50) | 28.7-31.8 | 0.045-0.054 | 0.65-0.81 | 62-121 |
| U of G2 | 6 motors | 0/6, 0/3 | -1.99 to +1.96 | 180-221 | 2843-4075 | 168-317 | 31.6-35.2 | 0.070-0.095 | 0.70-0.82 | 53-113 |
| H (added) | 6 motors | 0/6 walk, 0/3 turn | -5.18 to +3.34 | 98-144 | 4293-4350 | 189-263 (32-45) | 26.7-28.0 | 0.028-0.061 | 0.47-0.65 | 18-68 |

The walk gates, pair by pair (from `results/M3_read.txt`):

| pair | G1 pair mean (m/s) | G2 pair mean, halves along/across (m) | G2 fails on | H pair mean | H fails on |
|---|---|---|---|---|---|
| W1 up/down, yaw 0 | 0.0008 | 0.066; up 0.50/0.60, down 1.09/0.12 | up half cross-track | -0.025 | down half -1.89 |
| W2 up/down, yaw +15 | 0.0010 | 0.066; up 0.45/-0.45, down 1.12/0.06 | up half cross-track | 0.011 | down half -2.00 |
| W3 up/down, yaw -15 | 0.0001 | 0.052; up 0.14/-0.95, down 1.10/0.05 | up half cross-track | 0.014 | down half -1.92 |
| W4 across, yaw 90 | 0.0002 | 0.106; +y 2.67/1.05, -y -0.13 | -y half backward | -0.099 | -y half -5.13 |
| W5 across, yaw 105 | 0.0012 | 0.073; +y 2.24/1.86, -y -0.49 | both | -0.080 | both |
| W6 across, yaw 75 | -0.0014 | 0.079; +y 2.56/0.97, -y -0.67 | -y half backward | -0.076 | -y half -5.18 |

The frozen sets (`results/M3_frozen.json`). G1: T 3.0 s, stance stroke -10,
drop 5, lift 0, feather -20 to +20, stance share 0.6, tail -20, pitch loop
1.0. G2: T 1.5 s, stance stroke -10, drop 10, lift +15, feather -20 to +20,
stance share 0.85, tail -20, pitch loop 1.0. S: stroke -12, feather 0, tail
-20, roll loop 1.0. H: as G1's box with T 1.5, lift +10, feather -40 to 0,
stance share 0.85.

## 2. The frozen checks against what was read

- **C1, pairing. Held.** Every scored pair has equal hashes at t_c. F's pair
  mean is 0.000e+00 m/s. The N0 and N1 pair means are exactly 0.
- **C2, stand. Partly falsified.** "S stands on at least 3 of 4" (p 0.8):
  held, 4 of 4. Held: drift under 5 mm (3.5-4.6); hold power 6-15 W (10.8-11.1);
  each wing within 25% of half the weight facing up or down (0.500/0.500);
  peak stroke torque 1.5-3 N m (1.95-2.19). **Falsified:** "across the slope
  the downhill wing carries 0.55-0.70". It carries 0.513, because the roll
  loop holds the body near level with the downhill wing reaching further
  down (figure `M3_stand_across.png`).
- **C3. Held.** N0 fails stand (i) on 4 of 4 (body share 0.96-0.98) and drifts
  0.7-1.2 mm. N1 and S facing up or down are within 20% on power (10.4
  against 10.8-11.1 W).
- **C4, walk.** G1 "walks" p 0.35: no. G2 p 0.2: no. At least one p 0.45: no.
  The conditional predictions (speed, uphill slower, "on its wings") have no
  passing pair to be read on.
- **C5, turn.** G1 "turns" p 0.5: no (3-4 deg in 14 s). The 5-20 deg/s rate
  is not reached by any arm: G2 reaches 3.4-5.6 deg/s with tumbling, and does
  not pass.
- **C6, U. Held.** U of G1 is under the bar on 6 of 6 pairs (pair means under
  0.0023 m/s). U of G2 is under the bar on 5 of 6 (W5: 0.028).
- **C7, pitch coupling. Held.** Pitch peak-to-peak exceeds 10 deg in every
  gait run: G1 21-22, G2 62-130, H 18-92, U 21-113.
- **C8, loads. Falsified** for every arm that moves.
  - Peak wing force, predicted 1-4 weights: G1 reads 3.4-5.4, G2 24-50, H
    32-45.
  - Peak stroke torque, predicted 3-15 N m: G1 is inside it (7.7-8.9), but G2
    reads 28.7-31.8 and H 26.7-28.0. That is above the 24 N m stall: the
    motor's band widens when it is back-driven, and the harness allows that.
  - Steps on the motor's curve, predicted under 2%: stroke 4.5-5.4% in G2,
    2.8-6.1% in H. G1 and S are at 0.
  - Feather at its range limit, predicted under 1%: 0 while walking; 0.7-1.8%
    in G2's turns.
- **C9, energy. Falsified** for G2 and H.
  - Walking power, predicted 10-40 W: G1 reads 9.5-15.1, G2 124-154, H
    98-144.
  - Cost of transport has no passing pair to be read on. For reference, on
    |along|: G2 142-296, H 189-263.
- **C10, QACC. Held.** No reset in 171 of 171 scored runs, nor in the tuning
  rows.
- **C11, rotor. Partly falsified.**
  - Held: every verdict of G1 and S is the same at J_r 5e-6 and 1e-6. S
    passes 4 of 4 at each, and G1 fails every pair at each.
  - Falsified: "window electrical energy changes by under 10%". It falls by
    0.1-13.3% (largest on G1's walks and turns).
  - G1's peak electrical power falls from 472-741 W to 113-240 W (5e-6) and
    55-213 W (1e-6). The peaks are the rotor being stopped at each foot strike.

## 3. Rotor sensitivity (J), G1 and S

| | J_r 2e-5 | 5e-6 | 1e-6 |
|---|---|---|---|
| S passes | 4/4, 10.8-11.1 W | 4/4, 10.8-11.1 W | 4/4, 10.8-11.0 W |
| G1 walk pairs, turns | 0/6, 0/3 | 0/6, 0/3 | 0/6, 0/3 |
| G1 walk P (W) | 9.5-15.1 | 8.6-14.2 | 8.3-14.0 |
| G1 walk peak P window (W) | 472-741 | 113-240 | 55-213 |
| G1 walk peak stroke torque (N m) | 7.7-8.9 | 4.3-7.7 | 3.8-7.3 |

## 4. Figures (from the built model at recorded states, opened and checked)

Each figure is in body-aligned views, side and front. It shows the beach line,
the CoM (+), the contact points MuJoCo finds at that pose (x), and the vertical
ground force on each group at that instant.

- `figures/M3_stand_across.png`, S at yaw 90:
  - Resting on the hull, then propped at t_c, then the end of the hold.
  - At the end, two feet on the ramp, the downhill wing lowered further, the
    hull 0.08 m clear, and 3.0 / 2.8 N on the wings.
  - Checked against section 0's geometry: the foot is the tip's chord, and
    the CoM is inside it.
- `figures/M3_G1_stride.png`: one stride of the frozen G1.
  - Both tips flat during stance, then the hull set down in the swing (5.7 N
    on the body at 7.28 s).
  - The body pitches between -13 and +7 deg while the feet stay put: no step.
- `figures/M3_G2_stride.png`, G2 up the ramp:
  - The apex of a hop at pitch +30 deg, the wing tips touching (4 N), a 97 N
    touch-down, then airborne again.
  - The first draft drew the stride's nominal phases, and all five instants
    were in the air. It was redrawn at contact instants, which are rare.
- `figures/M3_H_stride.png`, H:
  - The apex, a landing on both tips at 52 N with the body 26 deg nose-down,
    stance on two feet, and the take-off stroke at 92 N on the trailing-edge
    corners.
- `figures/M3_G2_turn.png`, G2's +90 turn:
  - At t_c, then at 9 s pitched 45 deg nose-up on its tail at 86 N, then at
    14 s yawed 55 deg.

## 5. Disclosed: everything changed or added after data

1. **Tuning rows t001-t032 used a different backward law** from the
   prereg's: they flipped the feather sweep's sign against the stroke timing.
   From t033 on, backward is the cycle run in reverse, as the prereg says.
   Only t033 and later are candidates for freezing.
2. **Controller details not in the prereg:**
   - S's attitude loops fade in over 0.3 s. Added after t005/t006, where a
     step in the loop target saturated the 300 rad/s PD (a 2 kW spike).
   - The gait blends from the stance pose over 0.3 s.
   - A 10 deg hysteresis on the forward/backward/turn latch.
   - The tail's 1.07 N m peak at t_c in every stand row is the rate reference
     stepping at the end of the rise, not a load.
3. **The freeze rule was completed during tuning, before any scored row.** I
   defined its normalisation (`m3_freeze.py` docstring: each gate quantity
   over its bar, minus 1) and its eligibility (a set must have been read on
   all three tuning starts and both turns). Under it the rule picked the G1
   that does not move. That is reported as the frozen row; it was not
   overridden.
4. **Budgets.**
   - G1 used 114 of 120 rollouts, 28 of them under the old law of item 1.
   - G2 used 28 of 120: stopped short, because the finding that the feather
     angles set the direction, not the direction of time, carried over from
     G1 (G2's own rows agree).
   - S used 4 of 20: it froze on passing every tuning start, as the rule says.
5. **Arm H was added after tuning:** the forward hop of t033 (the largest
   uphill progress). It runs under the frozen controller and gates, and the
   rule did not choose it. No other steering law was tried: a "turn, then hop
   forward" law is the user's call.
6. **Other changes to the run list and rows:**
   - N1 was run for both gaits (the prereg counted one set).
   - `airborne_frac` was added to the rows from tuning batch 4 on (t041); all
     scored rows have it.
   - The best gait for J was meant to be the one with more passing pairs. No
     gait passed any pair, so by the tie rule it is G1.
   - F used G1.
7. **Peak power is reported twice.** The prereg's peak electrical power over
   the run includes the drop of about 0.1 m that `drop_to_ground` leaves
   (131-277 W, the same in S and N0). The peak over the window is reported
   beside it.
8. **Reader edits after scoring:** `m3_read.py` changed in two labels only,
   the mirrored null's heading and a window-peak column. No gate logic
   changed (sha `ddcf681b…` at scoring, `a49da15a…` now).
9. **Wall time** was about 3.3 h for the 127 main runs, against 1.8 h
   predicted: the machine's load was 15-17 from the other passes. Four
   workers were used throughout.

## 6. Not checked

- **The fluid in the hop regime.** No dynamic stall; feather of 20-40 deg at
  near-zero airspeed (X2). The hop distances may be off by factors.
- **Contact.** Rigid boxes on a rigid beach at friction 0.9 and solref 0.004,
  with no structure. Impacts of 24-56 weights on a 1.4 m wing say a real wing
  would not survive these gaits.
- **The harness motor band.** It lets the stroke hinge see 28-35 N m when
  back-driven, above the 24 N m stall at gear 24. A real gearbox may not take
  that.
- **Gaits outside the box** (stance stroke above -5 deg, so the hull rides
  low), and a thumb or skid. Both would be new pre-registered arms.
- **Sensors.** Every verdict assumes the true state (sensors come in M5).
