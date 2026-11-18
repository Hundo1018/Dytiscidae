# Pass M2, pre-registration: free swim with the flying wings, and water to air

2026-11-18, on `52ea567`. Frozen before any scored rollout. `TRIPHIBIAN_SPEC.md`
pass M2 (L0 rows 2, 5, 9). Harness: `mission.py` (`Machine`, M0), not edited.
New code only in `experiments/flapper/m2_*.py`; rows in
`experiments/flapper/results/M2_*.jsonl`.

Two questions. (a) Does the 0.6 kg machine, with the wings and motors it flies
with, swim where it is told (a heading and a depth, each as an antipodal pair,
`CLAUDE.md` "Designing a measurement" rule 8), at the surface and under it?
(b) From floating or swimming, does it leave the water under its own power
and get airborne (L0 row 5: no machine above 0.16 kg has done it)?

The controllers read the **true state** (position, attitude, rates, depth,
which strips are wet). Modelled sensors come in M5. Every verdict below is a
verdict on a machine with a perfect state estimate.

## 0. Seen before freezing (probes, disclosed)

Six short runs, scratchpad scripts `p1_facts.py` .. `p6_swim.py`
(`/tmp/claude-1000/.../scratchpad/`), all at the harness's dt 1e-4, gear 24,
J_r 2e-5, the solver's default entrainment. No row of them is a scored row.

1. **`Machine(domain)` does not put the machine in its domain.** `mission.py`
   line 75 calls `mj.mj_setConst(m, d)` after `env.reset`; `mj_setConst`
   leaves `d` at `qpos0` (the solver's own comment, `fluid.py` ~1257, hands it
   a scratch `MjData` for that reason). `Machine("water")` without `pose=`
   starts at (-6, 0, +1.5), in the air, and falls in. M0's checks passed
   `pose=` or `drop_to_ground`, so it did not show. **Every M2 machine is
   built with an explicit `pose=`.** The harness diff that would fix it is in
   the report, not applied. Probe p1's "held still" row is therefore a body
   dropped 1.5 m into the water, not a water fact; it came to rest floating
   at the surface, 27-58 deg nose-down, CoM 0.06-0.11 m above the surface.
2. **Buoyancy, as the model gives it (p1, p2).** Sealed volume 1083 cm3
   (hull 1023, wings 2 x 24, tail and strut 13); fully submerged buoyancy
   10.89 N against a weight of 5.88 N: **1.85 times the weight.** L0 row 9
   assumed a 10-20% excess. The centre of buoyancy is 18 mm ahead of the dry
   CoM along the hull and level with it (a nose-up couple of about 0.2 N m
   when level, no righting moment in roll when submerged). Added mass at
   rest, level, under water: 26.4 kg (the two wings carry nearly all of it).
3. **The still machine under water (p2,** built at 4 m depth, level, joints
   held at 0 for 3 s): it rises and turns nose-up, 0.12 m and +15 deg in 3 s,
   vz at most 0.053 m/s; no QACC reset.
4. **The PD gains depend on where the machine is built.** `Machine` sets kp
   and kv from the mass matrix at construction. Stroke-hinge inertia: 0.020
   to 0.024 kg m2 built in air, 1.83 built under water (90 times). A machine
   built under water and then flying has a stroke loop 9.5 times stiffer in
   natural frequency than M0's validated take-off; one built in air and
   swimming has a loop of about 31 rad/s in water. See section 1 for which
   state sets the gains in each arm.
5. **The take-off stroke from afloat (p4, p5).** Placed nose-up at 90 deg
   with the hull's foot 0.13 m under the surface: held still, it falls back
   and settles at 55 deg nose-up with all 20 main-wing strips wet (45 of 47
   strips wet). With M0's take-off stroke (4.5 Hz, 85 deg, feather 50, a
   0.25 s amplitude ramp) from the same placement: 3 s, no rise (CoM within
   0.03 m of the surface throughout), pitch falling 66 to 14 deg, 46 of 47
   strips wet at the end, **stroke and feather motors on their limit 95% and
   92% of steps, 1.81 kW electrical per role** (both stroke motors). The
   wing roots sit at the hull's centre line, and a floating hull's centre
   line is at the waterline: the wings beat the water. This is the result of
   the "take-off stroke from afloat" strategy and it is not an arm below.
6. **A level flapping stroke under water (p6,** built at 3 m, level, stroke
   40 deg at 0.3 Hz, feather 20 deg a quarter beat ahead, offset 0 and
   +8 deg LE-down, 1 s ramp, 10 s): forward progress 0.08 m in 10 s, the hull
   heaving +-0.2 m/s each half beat and pitching up to 33-47 deg, rising
   0.37 m; motors on the limit 0.4% of steps; mean electrical 45 W, single-
   step peaks 2.5-3.8 kW (the start transient). The wings' added mass is 44
   times the body's mass: a level stroke moves the body, not the water.
   Recoil, not torque, limited that stroke. This sets the arms in section 2.
7. Cost: 13-15 s of wall per simulated second, one process.

## 1. What is the same for every arm

- **Machine.** `Machine` as built: 0.6 kg, BODY_S, W6 wing (Goe 417a, camber
  0.04, feather hinge 0.1 chord behind the LE), 0.06 m2 all-moving tail,
  witness motors (stroke and feather tau_s 1.0 N m, omega_0 1000 rad/s, gear
  24 and 16; tail 0.1 N m, gear 50), dt 1e-4, wn 300. At gear 24 the stroke
  hinge has 24 N m at stall, twice L0's 12 N m crank cap; stated, not
  changed. Kinematic box, from `jnt_range`: stroke +-80 deg (1.4 rad),
  feather +-57 deg (1.0 rad), tail +-34 deg (0.6 rad). Signs (read from the
  joint axes, re-measured open-loop on a tuning seed before any law uses
  them): stroke + is up on both wings; feather + is leading edge down on
  both; the tail halves pitch about +y together.
- **Sea and world.** Calm (`SeaState` amplitude 0): the free surface is the
  plane z = 0. Every start at x = -30 m (beach from x = 0.09 m, seabed at
  -18 m). What the solver does at the surface (`medium.py`): each strip's
  density and viscosity blend linearly with its submerged fraction, a ramp
  over +-`half_height` (1-35 mm here) about its centre; buoyancy acts on the
  submerged fraction of the sealed volume. There is no wave, splash, spray,
  free-surface deformation, wave drag or planing lift, and no suction at
  exit. The strip solver has no dynamic stall (X2) and no wake between wings.
- **Entrainment mode.** Primary: `solver.entrainment = "wetting"` (S1: reacts
  only water arriving at a strip, which the term was written for; the
  default reacts every change of a strip's projected added mass and is a
  one-sided drag on a flapping wing, 23% of thrust on S1's grid). This is a
  public attribute set in `m2_*.py`, not a harness edit, and **it is the
  user's call at review**; if the user keeps the default, it becomes primary
  and "wetting" the sensitivity. Sensitivity in either case: the other mode
  on the best swim arm (one seed pair) and the best exit arm (one seed).
- **Where the gains come from.** Swim arms: built at their start pose under
  water (or afloat), so the PD sees the water inertia. Exit arms: built in
  air at x = -30, z = 5 m at the start attitude, then placed at the start
  state with `place()` and `solver.reset()`, so the stroke loop is M0's
  validated take-off loop (check C0); their water phase runs on that softer
  loop, and its tracking error is reported.
- **Seeds.** Tuning seeds 100-103; scored seeds 0-5, never seen before the
  scored run. A seed sets: attitude scatter (axis-angle, angle N(0, 0.18
  rad), as `TriphibianEnv.scatter`), position jitter N(0, 0.2 m) in x, y,
  the stroke's initial phase U(0, 2 pi), the commanded heading h U(0, 2 pi)
  in world, the depth step Delta U(0.8, 1.5) m.
- **Recording.** Per row: arm, seed, half, the whole command, per-role
  electrical and shaft energy (`Machine.E_elec`, `E_shaft`), mean electrical
  power, peak electrical power over a 50 ms window and the single-step peak,
  share of steps on the motor limit per role, `bad_qacc`, the 10 Hz trace
  (position, attitude, rates, wet strips, depth). **A row with `bad_qacc` >
  0 is excluded from every verdict, counted as not met, and listed.**
- **Tuning.** By hand, no optimiser. Every tuning rollout (signs, open loop,
  closed loop) is appended to `results/M2_trials.jsonl` with its law and
  seed. Cap: **60 rollouts and 1500 simulated seconds in total**, across all
  arms, seeds 100-103 only. The law of each arm is frozen (file sha256
  written into the row) before its scored rows. Workers: at most 4.

## 2. The arms

The law of every do-something arm is one family, by beat phase phi
(frequency f):

    stroke_i  = b + s_i A/2 sin(phi + p_i)          i = right, left
    feather_i = psi0 + d_i - Psi cos(phi + chi)
    tail      = e +- a                              (both halves; difference)

with feedback from the true state onto pitch (e, psi0), heading (amplitude
difference s_R - s_L, phase p_i, feather difference d_i), roll (d_i, a) and
the depth or height error (the pitch reference, psi0, A). Each arm fixes
what may move; the gains and set points are tuned.

### Swim at the surface (gate GS)

- **WS, paddle.** Floating level; stroke biased down (b < 0, the beat inside
  -80..0 deg so it stays under the floating hull); heading by the amplitude
  difference. The waterplane takes the heave that p6 saw.
- **S0, still.** Every joint held at 0 from the same start.
- **SF, frozen.** WS's law with the heading command frozen at the first
  half's in both halves.

### Swim under water (gate GW)

- **WV, pitched sweep** (F3e's route). The body pitched nose-up 40-85 deg, so
  the stroke sweeps the wings across rather than up and down; force along
  the body's axis from the feather sign; translation by tilting the body
  toward the heading; depth by the pitch reference and the stroke amplitude.
- **WK, scull.** Stroke held at a bias b, feather alone oscillating (F3b's
  S3), body near level, depth by psi0 as a diving plane at the speed it
  makes.
- **W0, still.** Every joint held at 0 from the same start.
- **WF, frozen.** The better of WV and WK (by the tuning seeds) with the
  heading and depth target frozen at the first half's in both halves.
- **VB, buoyancy variant (labelled; a separate arm).** Every strip's sealed
  volume times 0.584, which gives a fully submerged buoyancy of 1.08 times
  the weight with the centre of buoyancy where it was (a flooded hull
  fraction; mass, shape, drag and added mass unchanged). Arms WV-on-VB (or
  WK-on-VB, the one that was better as built) and W0-on-VB. It answers
  whether the as-built failure, if there is one, is buoyancy or thrust. It
  is not the machine and no verdict on the machine reads it.

### Water to air (gate GX)

Starts: **float** (placed level at the surface, joints held at 0 for 3 s,
then the arm starts) and **depth** (level at 3 m, joints held). Each arm
runs 20 s from a float start and 40 s from a depth start (time to come up).

- **XB, biased stroke afloat.** From float: the stroke biased up (b up to
  +37.5 deg with the full 85 deg, or a smaller A with a larger b, inside
  +-80 deg) so the wings beat in air above the floating hull; the tail and
  the feather offset pitch the body up as it gathers lift and way. Frequency
  and amplitude may change with the wet strip count and the body's pitch.
- **XD, breach from depth.** From depth: up the water column nose-up on the
  buoyancy and the best water stroke (WV's), the wings edge-on, then M0's
  take-off stroke (or XB's) starting when the shoulder has cleared the
  surface by a tuned height.
- **X0f, X0d, still.** Joints held at 0 from float and from depth.
- **X1, swim, no exit procedure.** WS from float and the better underwater
  arm from depth, heading -x, for as long as the other arms from that start.

## 3. The gates

All at 10 Hz samples. "Depth" is the CoM's depth below z = 0. Hull attitude:
nose axis x_b, dorsal axis z_b (world-z components written x_b.z, z_b.z).

**Not tumbling (part of GS and GW):** over the scored window, z_b.z >= -0.25
at every sample (the dorsal axis never more than 105 deg from up), and the
mean of |omega| <= 1.0 rad/s.

**GW, under water, per seed.** One initial state at 3 m (after scatter), two
halves of 40 s each: half A told heading h and depth 3 + Delta, half B told
h + pi and depth 3 - Delta. The scored window is t = 15-40 s of each half
(half of the headings lie behind the machine: 15 s to turn).
- progress: each half's displacement along its own heading over the window,
  divided by 25 s; the pair mean P >= 0.05 m/s **and** each half's own
  progress > 0;
- depth response: each half's signed movement toward its target (mean depth
  over t 35-40 s minus 3 m, signed by the command, over Delta); the pair mean
  R >= 0.5;
- held: |mean depth over t 35-40 s - target| <= 0.5 m in both halves, and the
  machine never at the surface (depth >= 0.3 m at every sample of the
  window);
- not tumbling, both halves.
Met on a seed when all four hold. **GW is met by an arm on 5 or more of 6
scored seeds, with its still and frozen arms met on 0 of 6.** The
two halves share one initial state, so for the still arm (and any law that
ignores the command) they are the same rollout and P and R are exactly 0:
the still machine's rise (p2) cancels by construction.

**GS, at the surface, per seed.** Placed level at the surface, joints held
for 3 s (not scored), then two halves of 40 s from that one state: A told h,
B told h + pi; scored window t = 15-40 s. Progress as GW (P >= 0.05 m/s,
each half > 0); stays at the surface: depth <= 0.5 m at every sample;
not tumbling. **GS is met on 5 or more of 6, with S0 and SF met on 0 of 6.**

**GX, water to air, per seed.** Met when, at some time t* inside the run,
for 3.0 s from t* every strip's submerged fraction is 0 (no strip wet, hull
and tail included; `medium.submerged_fraction` over all 47 panels), the
CoM's height above the surface at t* + 3 s is at least 0.5 m and at least
its height at t*, and its mean vertical speed over [t* + 2, t* + 3] s is
>= 0. Without power the machine cannot be clear for 3 s (that needs a
vertical exit at 14.7 m/s). **An arm exits if it meets GX on 4 or more of 6
scored seeds; X0f, X0d and X1 must meet it on 0 of 6.** The count is
reported for every arm whatever it is.

Every GX verdict states in its outcome paragraph that the model has no
dynamic stall (X2: forces off by 1.5-3.5 times at large feather and low
speed, which is where exit happens) and no wave, splash or free-surface
deformation (the surface is a density ramp per strip, section 1).

## 4. Checks and predictions (numeric; falsified by the stated number)

- **C0 (gains).** An exit-arm machine, built as section 1 says, held at 90
  deg in air with M0's take-off stroke for 6 beats: lift within 2% of M0's
  1.840 weights. If it fails, the exit rows are labelled as on a different
  loop.
- **C1 (the still machine).** W0: rises on 6 of 6 seeds in both halves
  (depth falls by 0.2 m or more over 40 s); GW met on 0 of 6. S0 drifts
  under 0.02 m/s; GS met on 0 of 6. W0 and S0 read P = 0 and R = 0 exactly. X0f and X0d: GX met on 0 of 6; the
  largest CoM height above the surface in X0d under 0.15 m.
- **C2 (surface swim).** WS meets GS on 5 or more of 6 (I hold 0.5), at a
  pair-mean progress of 0.05-0.25 m/s, mean electrical power 10-80 W.
- **C3 (under water, as built).** No as-built arm meets GW (I hold 0.65).
  The failing clause is the deeper half's depth (the 1.85 times buoyancy):
  half A's mean depth over t 35-40 s is short of its target by more than
  0.5 m on 4 or more of 6 seeds for the better arm. Progress P >= 0.05 m/s
  on 3 or more of 6 for the better arm.
- **C4 (variant).** On VB the better arm meets GW on more seeds than as
  built, and meets the depth-held clause on 4 or more of 6.
- **C5 (frozen).** SF and WF: |P| < 0.02 m/s and |R| < 0.2 on 6 of 6.
- **C6 (exit).** No arm meets GX on any seed (0 of 6, I hold 0.8). XB's
  largest CoM height above the surface under 0.5 m on 6 of 6, with the
  stroke motors on their limit more than 30% of steps while any main-wing
  strip is wet. XD's: the shoulder clears the surface on 3 or more of 6, and
  no strip is clear for 1 s on any seed.
- **C7 (power).** Every do-something swim arm: mean electrical power under
  100 W and peak (50 ms window) under 1 kW. Every exit arm: peak over
  400 W.
- **C8 (dt and rotor, reported, not gating).** The best swim arm, one seed
  pair at dt 5e-5: P within 10% and each half's final depth within 0.1 m of
  the dt 1e-4 row. J_r 5e-6 and 1e-6 on the best swim arm (one seed pair)
  and the best exit arm (one seed): mean electrical power falls with J_r;
  the gate clauses are reported as read.
- **C9 (entrainment mode, reported).** The other mode on the same rows as
  C8: P within 20%, each half's final depth within 0.2 m, GX count unchanged.
- **QACC.** Under 1 in 10 rows has a reset; excluded rows are listed.

## 5. What a negative means

If no arm meets GW or GX, that is the result, reported with the measurement
of what limited it (the clause that failed, the force or torque at the
limit, the wet strips, the buoyancy). No gate, band, window or seed count
here is changed after a scored row is seen; anything added after data is
disclosed in the result as such. Stopping rule: the scored rows filled, or
the tuning cap spent with an arm not frozen (then that arm is reported as
not tuned to a law, with its best tuning trace, and its scored rows are run
with the last law tried).

## 6. Output

`m2_common.py` (building machines per section 1, the law family, the
recorder), `m2_run.py` (rows), `m2_read.py` (gates and checks),
`m2_draw.py` (figures from the built model: `figures/M2_*.png`, at several
instants of a scored swim pair and of each exit arm's best seed, opened and
checked against section 1's geometry), `results/M2_trials.jsonl`,
`results/M2_rows.jsonl`, `RESULT_M2_water.md` (outcome first, a table, each
check against what was read).

Budget (from section 0, item 7): scored rows about 25 h of CPU (GW 6 seeds x
6 arms x 80 s, GS 6 x 3 x 83 s, GX 6 x 6 x 23-43 s, C8-C9 about 4 h), on 4
workers about 6-7 h; tuning up to 1500 simulated s, about 6 h of CPU.
