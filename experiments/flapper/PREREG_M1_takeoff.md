# Pass M1, pre-registration: a standing take-off from the beach, free body, every hinge on its motor

2026-11-18, on `52ea567`. Frozen before any tuning or scored rollout.
`TRIPHIBIAN_SPEC.md` pass M1 (L0 row 4, a standing take-off; row 4b, a
ground run). The machine is M0's (`mission.Machine`, `RESULT_M0_harness.md`),
unchanged. The question: can it leave the ground under its own power from
rest on its belly on the beach and climb away, with no hand launch, no
stand and no added part?

The controller reads the **true state** (attitude, rates, centre-of-mass
height and velocity, contact). Modelled sensors are M5's; every verdict here
is "with a true-state controller".

## 0. Seen before freezing (probes, disclosed)

Code `m1_probe.py`, helpers `m1_lib.py`, rows `results/M1_probe.jsonl` (29
rows, none with a QACC reset). All at `x = 17 m`, nose up the beach, except
where stated. **Two of them were take-offs or jumps, so the predictions for
N2 and VO below are not blind.**

- **P0, geometry at rest** (wings level, 1 s settle, x 17 and 30, nose up,
  down and across the beach). Dry mass 0.600 kg; model mass 0.690 kg (the
  wings' and tail's air added mass, whose weight the solver cancels). The
  dry centre of mass is 0.226 m from the hull's tail end; the shoulders are
  0.099 m ahead of it, the tail hinge 0.526 m behind. Inertia about it (body
  frame, added mass included): roll 0.030, pitch 0.022, yaw 0.048 kg m2.
  Rests on the hull and both tail halves, 3.9 deg nose-up against the ramp
  (10.7 deg in the world facing up, -3.0 facing down); across the slope on
  the hull and one tail half. Shoulders 0.055 m over the ground, centre of
  mass 0.048 m. Stroke positive is up for both wings (both axes checked).
  Wall time 11.6 s per simulated second on one process.
- **The wing's mass and contact box (D3's note, stated, not changed):** the
  wing panel's mass, its fluid strips' reference point and its collision box
  are all centred a quarter chord aft of the fluid leading edge: 38.6 mm aft
  of the feather hinge (0.1 c), where a uniform plate's would be 103 mm. The
  collision box therefore runs from 0.35 c ahead of the hinge to 0.65 c aft
  of it (0.09 m ahead, 0.17 m aft), from the hull side to the tip.
- **P1, held rig** (body held at 90 and 45 deg pitch at 30 m, M0's method:
  hull reset every step, mean hull acceleration). K1 at 90 deg: 1.840
  weights (M0's figure, reproduced); at 45 deg 1.314 up and 1.272 forward
  (1.83 along the nose). Amplitude x1.15: 2.24; 5.2 Hz: 2.35 (stroke on
  its limit 30% of the beat). Signs of the hull's angular acceleration:
  stroke bias +0.2 rad (wings toward the back) +50 rad/s2 about the
  left-wing axis (the nose turns toward the belly); left amplitude larger by
  0.2: -30 rad/s2 about the back axis; feather offset left-right +0.2 rad:
  -65 rad/s2 about the nose axis; common feather offset +0.2 rad: -23
  rad/s2 about the left-wing axis.
- **P1b, a disagreement between two readings of the held rig.** Under a
  stroke bias of +-0.2 rad M0's method reads a horizontal acceleration of
  -+1.32 g, while the summed fluid force (`xfrc_applied` less the added-mass
  weight cancellation; it omits the implicit-damping part in
  `qfrc_applied`) reads -+0.15 weights. In a vacuum the method reads -0.06
  g. Vertically the two agree (1.84 against 1.89; freerig 1.856). The
  method's mean hull acceleration is not the held force over a mass when the
  wings' configuration is asymmetric. **So no held-rig number is used here
  as a control derivative or a force: only the signs above, and those are
  re-measured on the free body in tuning (section 5).** M0's lift check is
  not affected.
- **P1g, gear by D1's rule at K1** (held, least electrical power that holds
  the stroke within 2 deg, gears 16, 24, 32; higher gears exceed the motor's
  speed): J_r 2e-5: gear 24, stroke 405 W + feather 255 W, tracking 0.51 deg;
  5e-6: gear 32, 201 + 118 W, 0.80 deg; 1e-6: gear 32, 168 + 92 W, 1.19 deg.
  Lift 1.840, 1.829, 1.818. Feather gear 16 throughout (M0's; not swept).
- **P2, push-up only, no flapping.** Stroke to -1.35 rad (-77 deg; hinge
  stop 1.40) with the feather at 0. Ramped over 0.5 s: the body rises level
  on its two wing tips to a centre of mass 0.61 m over the ground, then
  tips back over 1 s onto its tail (43 deg nose-up at 1.45 s), because the
  tips stand under the shoulders, 0.10 m ahead of the centre of mass.
  **Stepped (the motor's curve is the only limit): the wings strike the
  beach and throw the body 2.08 m up (centre of mass), turning to 87 deg
  nose-up at the top 0.55 s after the command; it falls and turns over.**
  Stroke on its limit 7% of the 1.5 s. Load at one sampled instant of the
  strike 7.6 weights (the peak was not recorded).
- **P3, two open-loop attempts (counted as data).** (a) The ramped push-up,
  then K1 from the bottom of the stroke once the pitch reached 35 deg (1.38
  s): the wings struck the beach, the body fell back, flipped and was on its
  back on the ground by 2.9 s. (b) **The stepped strike, then K1 open loop
  from the bottom of the stroke once the stroke passed -1.2 rad (0.13 s):
  off the ground from 0.2 s, 11.3 m up and 7.8 m forward at 2.9 s, pitch
  (sampled every 0.1 s, so aliased by the beat) 50-75 deg and rising.**
  Electrical 391 W mean over the 3 s (stroke 256, feather 135). This is a
  take-off of 3 s seen before freezing, at one tuning entry.

Not looked at: anything at the scoring entries; any closed-loop run; any
lighter rotor on the free body; the ground run.

## 1. What is the same for every arm

- **The machine:** `mission.Machine("land")` with its defaults (dt 1e-4 s,
  PD natural frequency 300 rad/s, joint damping as built, 0.0366-0.0406 N m
  s/rad, which costs power; hull inertia as built, the ballast a point at
  the hull's centre; the solver's table stall law, plate centre of pressure,
  entrainment and implicit damping on). BODY_S, the W6 wing, the 0.06 m2
  tail, 0.600 kg. No part added in any arm.
- **Motors:** M0's witness (tau_s 1.0 N m, omega_0 1000 rad/s) on each
  stroke and feather hinge, the tail's a tenth. Rotor J_r 2e-5 (the
  witness, primary), 5e-6 and 1e-6 (sensitivity: whether such a motor exists
  is not asked). Stroke gear by P1g's rule: 24, 32, 32. **Because the rule
  also raises the stall torque at the hinge from 24 to 32 N m with the
  lighter rotors, the vault arm V is also run at gear 24 with each lighter
  rotor (section 2), so a rotor effect can be told from a gear effect.**
  Feather gear 16.
- **Ground:** the project's beach (slope 0.12, friction 0.9, MuJoCo soft
  contact with time constant 4 ms, damping ratio 1). The wing is a rigid
  box; the sand does not give. A strike of the wing on the beach is
  therefore a rigid lever on a soft-contact plane; how far that is from a
  membrane wing on sand is not in the model (T1 section 7). One sensitivity
  row set (section 2, S) changes the contact on the compiled model, from
  M1's own file.
- **Start:** placed lying on the belly, wings level, nose at the entry's
  heading, dropped to 1 cm over the beach (`drop_to_ground`), then **1.0 s
  with every actuator held at 0** (settling, and the hold of a do-nothing
  arm). The command is given at t_c = 1.0 s; the run ends at t_c + 9 s.
- **Scoring entries (8):** x in {15, 20, 25, 30} m, nose up the beach (+x)
  and down it (-x). **Tuning entries (5):** (17, up) (the probes'), (18,
  down), (27, up), (27, down), (22, across the slope). **Stress entries (2,
  reported, not counted):** (25, across, both ways), arm V at J_r 2e-5.
- **Reads, every arm**, from t_c on: centre-of-mass rise H (dry-mass centre
  of mass over its height at t_c); centre-of-mass height over the terrain
  under it; machine-world contact load (`Machine.contact_load`) and which
  bodies touch; nose elevation, span-axis elevation and body rates, each
  also as a **beat mean** (a moving mean over 1/4.5 s = 0.222 s, the K1
  beat, for every arm); electrical power per motor per step, giving energy
  from t_c, the instantaneous peak, the largest 0.1 s mean and the largest
  beat mean (all motors summed); the share of steps each motor is on its
  curve, overall and in the strike window (t_c to the start of flapping);
  the peak normal contact force on each wing body (weights); the largest
  moment about the shoulder of the contact forces on a wing (strike), and of
  the fluid forces on a wing (`xfrc_applied` on the wing body, so without
  the implicit-damping part; in free flight from t_c + 2 s on), and their
  ratio; the peak stroke-hinge torque; `Machine.bad_qacc`.
- **Model blind spots for every verdict:** the strip solver has no dynamic
  stall (X2: at low speed and large feather OpenFOAM read 1.5-3.5 times the
  forces, in the direction of more force, unqualified), which is exactly
  the regime of K1 at rest; no ground effect; no wake between wings or onto
  the tail; rigid wings; the strike is a contact event in a soft-contact
  model. **Every outcome paragraph of the result states the first and the
  last of these.**

## 2. The arms

Procedures (all end in the same flying law, FLY, section 4):

- **V, vault.** At t_c the stroke target steps to -1.35 rad (feather 0,
  tail 0): the motors give what their curve gives. FLY starts when the mean
  stroke angle passes -1.2 rad or 0.25 s after t_c, whichever is first, at
  the bottom of the K1 stroke, blended in over 0.1 s.
- **Vs, slower vault.** As V with the stroke target ramped to -1.35 rad
  over 0.2 s. A dose of the strike: less force, less height.
- **P, push-up.** The target ramps to -1.35 rad over 0.5 s; held until the
  nose is 35 deg over the horizontal (beat mean) or 3 s after t_c; then FLY
  as in V. No strike by intent (the stand on the tips of P2).
- **VO, vault, open loop.** V's procedure, then K1 at amplitude 1, no bias,
  no offsets, no feedback. Says whether the loop is needed.
- **R0, ground run (row 4b), one frozen stroke, open loop.** On the belly at
  t_c: 4.5 Hz, stroke 75 deg peak to peak about a bias of +42.5 deg (from +5
  to +80 deg: the tip clears the beach at the bottom, the hinge stop caps
  the top), feather 15 deg a quarter beat ahead, tail 0. The feather is
  capped by the wing's own box: at the root the box reaches 0.17 m aft of
  the hinge and 0.09 m ahead of it with the shoulder 0.055 m over the beach,
  so more than about 15 deg puts the root's trailing corner in the sand on
  every beat. No tuning. Run facing down the beach as well as up; the
  slope gives 0.12 weights downhill.

Do-nothing and controls (must fail the gates they are named for; predicted
in section 6, not designed to fail):

- **N0, do nothing.** Every actuator held at 0 for the whole run.
- **N1, flap from the belly, no procedure.** At t_c, K1 open loop at
  amplitude 1, no bias, starting at phase 0 (stroke rising). Every
  downstroke reaches 42.5 deg below the shoulder, 0.45 m at the tip, with the
  shoulders 0.055 m up: each one strikes the beach. **If N1 passes, the
  take-off is made by the ground strike of an ordinary downstroke, not by
  the vault, in both arms, and the result says so.**
- **N2, vault, no flapping.** V's strike, then every target held where the
  strike left it (stroke -1.35, feather 0). Height from the strike alone.

Rows:

| arm | J_r (gear) | entries |
|---|---|---|
| V, Vs, P, VO, R0, N0, N1, N2 | 2e-5 (24) | 8 scoring |
| V, P | 5e-6 (32), 1e-6 (32) | 8 scoring |
| V | 5e-6 (24), 1e-6 (24) | 8 scoring |
| V | 2e-5 (24) | 2 stress |
| S: V with contact time constant x0.5 (2 ms), x2 (8 ms), friction 0.5 | 2e-5 (24) | scoring entry (20, up) |

## 3. Gates (all beat means where an attitude or rate is read)

- **G1, leaves the ground.** t_off = the end of the last step with contact
  load over 0.001 weights. G1 holds when t_off <= t_c + 4 s (so at least 5 s
  off the ground to the end, with no touch).
- **G2, height.** Over the whole of the last 3 s (t_c + 6 to t_c + 9 s): H
  >= 2.0 m and the centre of mass is >= 2.0 m over the terrain under it.
- **G3, not falling back.** From the first time H >= 2.0 m to the end, H
  never below 1.5 m.
- **G4, attitude held.** From t_off to the end: nose elevation >= -10 deg
  and |span-axis elevation| <= 35 deg; over the last 3 s, |body rate| <= 1.5
  rad/s.
- **Pass** = G1 and G2 and G3 and G4, and `bad_qacc == 0`. A run with a QACC
  reset is excluded from passing (it counts as not passing) and is listed.
- **An arm takes off** when it passes at >= 7 of its 8 scoring entries.

## 4. The flying law FLY (structure frozen; gains are tuned, section 5)

K1 (4.5 Hz, stroke 85 deg, feather 50 deg a quarter beat ahead), modulated
by a loop on beat means of the true state, its outputs low-passed with a
time constant of half a beat:

- thrust direction wanted: up, leaned against the horizontal velocity of the
  centre of mass (k_v rad per m/s, at most 0.3 rad);
- the error rotation from the nose to that direction, in the body frame,
  turned into wanted angular accelerations by a PD (natural frequency w_a,
  damping ratio 1 on the body rates); about the left-wing axis by the common
  stroke bias (|bias| <= 0.3 rad), about the back axis by the left-right
  amplitude difference (<= 0.3), spin about the nose damped by the left-right
  feather offset (<= 0.3 rad);
- height: the common amplitude, in [0.60, 1.15], from a PD on H against a
  target H_t = 4.0 m with the climb rate limited to v_c m/s.

Tunable: k_v, w_a, the gains from wanted angular acceleration to each
output (sign from P1, magnitude from the free-body measurement of section
5), the height PD and v_c, the base frequency within 4.0-5.0 Hz, the
vault's switch angle and blend, P's switch pitch, and whether the tail is
used (a pitch-damping term). Not tunable: gates, entries, machine, motors,
gears, run length, the procedures' targets (-1.35 rad, ramps 0 / 0.2 / 0.5
s), N0-N2, VO and R0.

## 5. Tuning and freezing

- First, on tuning entry (17, up) at J_r 2e-5: the free body released at 30
  m, nose vertical and still, K1 at the amplitude that holds it, then a step
  of each output (bias 0.1, amplitude difference 0.1, feather offset
  difference 0.1, amplitude +0.1), rates read over the first 0.2 s, as T1
  did: the signs and sizes the loop uses. Logged.
- Then V, then P, by hand, on the tuning entries only, at J_r 2e-5. One law
  for all arms and rotors; the lighter rotors are run with it unchanged.
  If V cannot be made to pass on the tuning entries at 2e-5, tuning moves to
  5e-6, then 1e-6, and the result says so.
- **Budget: 80 tuning rollouts (each at most 10 s simulated) or 4 hours of
  wall time, whichever first.** Every rollout appended to
  `results/M1_tuning.jsonl`, with its entry, arm, gains and gate reads.
- Freeze: `m1_takeoff.py` is hashed (sha256) and the hash written as the
  first row of `results/M1_takeoff.jsonl` before any scored row. Scored
  once. A bug found afterwards in the reading code (not the law) is fixed,
  every scored row re-run, and both disclosed.
- Stopping rule: every row of section 2 filled; or a procedure shown
  blocked with its trace (thrust, torque, motor curve, ground strike).

## 6. Checks, with the numbers I expect

- **C1, N0:** 0 of 8; H under 0.01 m throughout; load 0.98-0.99 weights.
- **C2, N2 (not blind, P2):** 0 of 8; peak H 1.5-2.6 m within 0.7 s of t_c;
  back on the beach within 1.6 s of t_c.
- **C3, N1:** 0 of 8, failing G1 or G4. I hold this at 0.65. Falsified at
  >= 1; at >= 7 the take-off is the downstroke's strike (section 2).
- **C4, V at 2e-5 takes off (>= 7 of 8).** I hold this at 0.65. Timing:
  t_off within 0.4 s of t_c; H >= 2 m within 1.2 s of t_c, at every passing
  entry.
- **C5, VO (not blind, P3b):** G1 and G2 hold at 8 of 8; G4 fails at >= 3 of
  8 (the open-loop attitude drifts or tumbles within 9 s). I hold this at
  0.5. If VO passes as often as V, the loop is not what makes the take-off.
- **C6, P:** passes at <= 4 of 8 at 2e-5: starting the stroke from the
  stand drops the wings onto the beach (P3a). I hold this at 0.6.
- **C7, Vs:** passes at no more entries than V; its peak wing contact force
  at least 30% under V's at every entry.
- **C8, the strike, V at 2e-5:** peak normal contact force on a wing 5-30
  weights; the contact moment about the shoulder 2-10 times the largest
  fluid moment in free flight; the stroke motors on their curve for >= 90%
  of the strike window; the stroke hinge at 22-24 N m.
- **C9, electrical power, V:** instantaneous peak 1.8-3.0 kW at every rotor
  (two stroke motors near stall cost about 1 kW each in copper); largest
  0.1 s mean 1.2-2.5 kW. Mean over the last 3 s: 300-550 W at 2e-5, 150-300
  W at 5e-6 (32), 120-250 W at 1e-6 (32). Against L0's paper take-off of
  196-240 W and three cells' 462 W continuous: the 2e-5 rows over both in
  the last 3 s; every row over both at the strike.
- **C10, rotors:** V takes off at 5e-6 and 1e-6 with gear 32 (>= 7 of 8
  each). I hold this at 0.6. At gear 24 the lighter rotors vault higher than
  at 2e-5 (peak H in the first 0.5 s), so the vault's height is not only the
  stall torque.
- **C11, R0:** 0 of 8; the run does not start: the centre of mass moves
  under 1 m along the beach in 9 s, facing either way.
- **C12, QACC:** no reset in any scored row. Any is listed and excluded.
- **C13, stress and contact sensitivity (reported):** V passes both
  across-slope entries (0.5); passes all three S rows (0.6), with the peak H
  of the first 0.5 s within 30% of the base contact's (0.5).

Each check is falsified by its stated numbers. A failed C1 or C2 means a
gate reads something other than a take-off, and every pass of the other
arms is labelled until it is understood.

## 7. Output

`m1_takeoff.py` (the arms, FLY, the reads; one process per row, at most 4
processes), `m1_read.py`; rows `results/M1_tuning.jsonl`,
`results/M1_takeoff.jsonl`; figures `figures/M1_*.png` drawn from the built
model at several instants of a scored V take-off (at rest, the strike,
leaving the ground, the top of the vault, the climb, the hover) and of N2,
opened and checked against the gates' reads; `RESULT_M1_takeoff.md`,
outcome first. If no procedure arm takes off, that is the result, with what
limits it.
