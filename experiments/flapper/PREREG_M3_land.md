# Pass M3, pre-registration: standing, walking and turning on the wings

2026-11-18, on `52ea567` (branch `triphibian-diagnosis`), MuJoCo 3.15.
Written before any arm below is tuned or scored. `TRIPHIBIAN_SPEC.md` names
this pass M3 (L0 row 3). The question: on land, with the wings, tail and
motors it flies with, can the 0.6 kg machine of M0 (a) stand still where it
is put on the 0.12 ramp, not sliding, (b) walk along a commanded heading both
ways (up and down the ramp, and across it), (c) turn to a commanded heading
both ways. It has no legs: wings, tail and hull are all that can touch.

This pass decides no mechanism, material or construction. A clean negative
("it cannot walk at any pre-registered arm") is a result and is reported as
one.

## 0. Seen before freezing (probes, disclosed)

All in `m3_probe.py` (kept, reproducible) and two scratch checks (seed, and
`subtree_com`); twelve short runs (geom 1, signs 1, rest 2, stand 2, crutch 2
of which the first read its displacement from `subtree_com` and is void, vault
1, seed 2, `subtree_com` 1), every one listed here. Harness `mission.Machine` at its defaults
(dt 1e-4, PD wn 300, witness motors, gears 24/16/50, J_r 2e-5).

- **Geometry** (`m3_probe.py geom`), hull frame x from the hull's tail end:
  hull four ellipsoids, x 0 to 0.51 m, radius 0.034-0.046; each wing a box
  0.66 x 0.258 x 0.021 m, LE at x 0.390, TE at 0.132, the feather axis 0.026
  behind the LE (so the TE is 0.232 m behind the axis); tail boom (capsule)
  x -0.30 to 0; the two tail halves boxes 0.25 x 0.12 m at x -0.33, each an
  all-moving elevator on a spanwise hinge at its quarter chord. Centre of mass
  (dry masses) x 0.226, 0.10 m behind the feather axis. Masses: hull 0.491
  (ballast in it), each wing 0.041 dry, boom 0.005, each tail half 0.011.
  Hinge ranges: stroke +-80.2 deg, feather +-57.3 deg, tail +-34.4 deg; all
  limited. Signs (`signs`): +stroke raises the wing, +feather raises the TE,
  +tail raises the tail half's TE; mirrored on the two sides.
- **Deterministic.** `Machine(seed=0)` and `seed=5` give bit-identical states
  after 0.2 s (max difference 0.0): the environment is reset with
  `randomise=False` and nothing in this path draws a number. Rest at x 17 and
  x 25 m: identical to four digits (the ramp is uniform). "Seeds" in this pass
  are therefore start states (yaw, gait phase, x), stated in section 3.
- **Cost.** 1.2 ms per step at dt 1e-4: 12 s of wall per simulated second.
- **Still on the ramp** (`rest`, x 17 and 25, 2 s, wings level): `drop_to_ground`
  leaves the hull about 0.1 m up (it subtracts no half-height for a box and the
  largest semi-axis for an ellipsoid), the body falls, settles, and slides
  6 mm down the ramp in the 2 s; it lies 10.7 deg nose-up on the 6.8 deg ramp,
  the hull carrying 5.61 N and each tail half 0.12 N (of 5.88 N), no QACC
  reset.
- **Propped** (`stand`, 1.5 s ramp then 1 s hold): stroke -12, feather 0, tail
  -30 deg: the weight is on the two wing tips alone (2.92 N each), the hull's
  origin 0.150 m over the ramp, body pitch 6.9 deg (parallel to the ramp), 14
  mm of drift over the whole 2.5 s including the rise, peak stroke hinge torque
  4.36 N m (stall at gear 24 is 24 N m), 9.6 W electrical on the strokes
  averaged since t = 0. The tip of a flat wing is a foot one chord long, and
  the CoM (x 0.226) lies inside it (x 0.132-0.390): the machine stands on two
  feet with no third point. Stroke -15, feather -40: the wings stay flat on
  the ground and the **body** pitches -32 deg nose-down about them, the hull
  origin rising 0.32 m.
- **Two hand gaits, neither walked.** `crutch`: both wings together, feather
  0 to -50 then swept back to -15 with the stroke lowered, lift, return,
  twice: the CoM moved -0.05 m (down the ramp) in 5 s. `vault`: the trailing
  edge placed first (feather -55 in the air, then the stroke lowered onto it),
  then swept to -5 with the stroke going -10 to -24, lift, return, three
  strides: -0.25 m in 6.5 s, the body pitching -32 to +7 deg each stride, some
  instants with no contact at all, a peak of 13.7 N on the wings (2.3 weights)
  and the hull slamming at 10-13 N. The mechanism read from both: with the
  CoM 0.10 m behind the feather axis and ahead of a trailing edge placed at
  any feather short of the -57 deg limit, feathering pitches the body about
  the feet instead of moving the feet under it. **Every arm below is watched
  for that pitch coupling.**
- **`data.subtree_com` is not the centre of mass in this harness.** The
  solver publishes added mass into `model.body_mass` (on land, the wings read
  0.083 kg against 0.041 dry: air's added mass) without re-running
  `mj_setConst`, so `body_subtreemass` stays at 0.6 and `subtree_com` is
  `sum(m_i x_i)/0.6`, scaled by the published-to-compiled ratio: 21.29 m where
  the CoM is at 20.226 (ratio 1.0527 = 0.6316/0.6), 23.33 m at 1.1537 =
  0.6922/0.6. The solver's own use (`fluid.projected`, `r = pos -
  subtree_com[root]`) is consistent with MuJoCo's `cdof`, which is expressed
  about the same `subtree_com`, so the dynamics are not affected as far as I
  can see; it is a trap for anyone reading it as a position. **This pass
  reads the CoM as the dry-mass-weighted mean of `xipos`.**

## 1. What is the same for every arm

- The machine as built by `mission.Machine("land")`: 0.6 kg, BODY_S, W6 wing
  (Goe 417a, camber 0.04, feather hinge 0.1 chord behind the LE), the 0.06 m2
  all-moving tail, six witness motors, dt 1e-4, wn 300. No geometric change;
  no arm adds a part (a thumb or a skid would be a separate pre-registration).
  Contact is MuJoCo's on the repo's beach (friction 0.9, condim 4); the strip
  solver stays on in air (forces only from shape and contact; nothing typed).
  `mission.py`, `freerig.py` and `drive/` are not edited.
- **Drive arrangement for every arm: one motor per hinge, as M0 builds it**
  (left stroke, left feather, right stroke, right feather, two tail halves).
  Whether a D-series mechanism could make each arm's motion is stated per arm
  in section 2.
- **The controller reads the true state** (position, yaw, pitch, roll and
  rates from `qpos`/`qvel`). Modelled sensors come in M5; every verdict here is
  "with a perfect state estimate".
- Start: `Machine("land", pose=dict(x=X, z=5))`, yaw set in the free joint's
  quaternion, `drop_to_ground()`, then 1.0 s with every hinge held at 0 to
  settle from the drop, then a 1.0 s rise to the arm's stance pose (both
  command-independent), then the commanded window. Command time t_c = 2.0 s.
- **Antipodal halves.** Each half of a pair is a fresh `Machine` and a fresh
  controller object (no state carried across halves: F2's fault). Before the
  command is read, the two halves have run identical code on an identical
  state; at t_c the runner hashes `qpos`, `qvel`, `act` and the controller's
  state of both halves and a pair whose hashes differ is not scored (reported
  as a harness fault).
- Readings, every run, over the commanded window unless stated: CoM
  (dry-mass `xipos` mean) displacement; yaw, pitch and roll traces; ground
  normal force per body group per step (wings = seg1/seg2, tail = seg4/seg5,
  body = hull and boom), its peak on each wing and the share each group
  carries (F2's "wing share" with tail and body beside it), and the share of
  steps with any body-group contact; per motor, electrical energy over the
  window (`E_elec` differenced, not `summary()`'s since-t=0 mean), the peak
  instantaneous electrical power (per-step difference of `E_elec`, summed over
  motors), peak |hinge torque| (`actuator_force`) per role and the share of
  steps on the motor's curve (`at_limit` differenced); steps with a hinge at its
  range limit; `bad_qacc`. Cost of transport = window electrical energy over
  (m g x distance along the heading).
- **QACC.** A run with `bad_qacc > 0` is excluded: it counts as not passing,
  is listed with its gate, and is never read as a zero.

## 2. The arms

Each gait is a periodic keyframed target per hinge (linear between keyframes,
rate feed-forward from the slope, the harness's PD follows), with the steering
and attitude laws below added to it. Its free parameters sit in a box fixed
now; the values are tuned in phase 2 on the tuning starts only (section 5).

- **N0, do nothing, flat.** Every hinge held at 0 from t = 0 (wings level, hull
  on the ramp). The null for stand, walk and turn.
- **N1, do nothing, propped.** The rise to the stance pose of the arm being
  compared, then every hinge held there (no gait, no attitude loop). The null
  for walk and turn: "a body standing on its wings".
- **S, stand.** The rise to a flat-foot stance (both wing tips planted, chord
  flat), then held with an attitude loop: roll by differential stroke, pitch by
  common feather, the tail as set. Box: stance stroke -20..-6 deg, feather
  -15..+15, tail -31..+31, roll gain 0..2 deg/deg, pitch gain 0..2 deg/deg.
  D-series: one stroke motor for both wings can hold a symmetric stance (up or
  down the slope) if its crank stops at a dwell and holds; the across-slope
  roll correction needs the two strokes apart, which one crank cannot give.
- **G1, crutch in phase (mudskipper).** Both wings together: plant (stroke to
  the stance angle with the feather at its plant angle), power (feather swept
  to its end angle while the stroke drops by a set amount to hold height),
  lift (stroke to the lift angle), return (feather back), the tail held or
  moved at the stride frequency. Walking backward is the same cycle run in
  reverse. Box: period 0.6..3.0 s; stance stroke -25..-5 deg; stroke drop in
  power 0..15; lift stroke -5..+15; feather plant and end angles each in
  -54..+54 (3 deg inside the hinge limit; either edge may be the spoke);
  stance share 0.4..0.85; tail mean -31..+31, tail amplitude 0..20, its phase
  free; pitch loop on the common feather offset 0..2 deg/deg; roll loop on the
  differential stroke 0..2 deg/deg.
- **G2, crutch alternating.** G1's cycle and box with the right wing half a
  period behind the left.
- **Steering, G1 and G2.** Heading error e (commanded minus body yaw, wrapped).
  Within e_go of the heading, walk forward; within e_go of its opposite, walk
  backward; otherwise turn on the spot (F2's switch). Walking: the two
  feather sweeps scaled by (1 + k e) and (1 - k e), clipped to the box.
  Turning on the spot: the two sweeps run in opposite senses at the turning
  amplitude. Box: e_go 20..60 deg, k 0..2 per rad, turning amplitude within
  the feather box. The turn gate (section 3) is run by the same controller
  with the commanded heading 90 deg off.
- **U, untimed** (for G1 and G2 each): the tuned gait's own keyframe shapes,
  amplitudes, tail and steering law, but the stroke's cycle runs at 1.618
  times the feather's period, so the lift and the sweep keep no fixed phase.
  It moves every hinge as the gait does, without the gait's timing.
- **F, heading frozen** (one start, best gait): the controller of both halves
  is given the first half's heading. Both halves are then the same run and
  the pair must cancel to rounding: a check of the pairing, not of walking.
- **J, rotor sensitivity**: the best gait (the one with more passing pairs,
  ties to G1) and S re-run on every scored start at J_r 5e-6 and 1e-6 on every
  motor role that has 2e-5 (the tail's 2e-6 scaled by the same factor), no
  re-tuning.

D-series, G1, G2, U: a D-series stroke crank makes a sinusoid at one amplitude
with no dwell, and D4's adjusters are locked during a run. G1 straight ahead
needs a stroke of a few degrees around a stance bias held below the wing's
flight bias, and a plant-power-lift-return shape that a sinusoid only
approximates: a crank could make an approximation of it only with the land
amplitude and bias set before the segment, which no D-series drive does.
Steering needs the two sides' sweeps to differ, and G2 needs the two sides out
of phase; neither is possible with one stroke crank for both wings. So **no
gait of this pass is a D-series gait**; each says what a land drive must be
able to do.

Not run, stated: a ground run on the belly pushed by the flight stroke's
thrust. At friction 0.9 the hull asks 0.86 weights of horizontal thrust to
slide, which is the take-off question (M1), not walking.

## 3. Gates (frozen)

Up the ramp is +x; across is +-y. Start states are disjoint from the tuning
starts of section 5.

**Stand (a).** Orientations yaw 0 (facing up), 45, 90 (across), 180 (facing
down), x 18, 22, 26, 30. Window: 10 s after t_c. A run stands if (i) the body
group carries at most 0.02 of the window's vertical ground force and touches on
at most 5% of its steps; (ii) the CoM's horizontal drift over the window is at
most 0.02 m; (iii) the yaw drift is at most 2 deg. Verdict "stands" if S passes
on at least 3 of 4. The drift bound alone is not a gate: N0 lying on its hull
holds by friction (6 mm in the probe's 2 s, mostly settling) and would pass
it; it fails (i), which is the honest reason it does not stand on its wings.
0.02 m in 10 s is a tenth of the walking bar below.

**Walk (b).** Six start states, each a pair of commanded headings h and -h:
up/down pairs (h = +x) at yaw offset 0, +15, -15 deg from +x, gait phase 0,
1/3, 2/3, x 18, 24, 30; across pairs (h = +y) at yaw 90, 105, 75 deg, the same
phases and x. Window 12 s after t_c. For each half, along = CoM displacement
on its own commanded heading, across = the perpendicular part. A pair passes if:

1. pair mean (along_h + along_-h) / (2 x 12 s) >= 0.020 m/s (0.24 m in 12 s,
   about half a hull length; a round bar from the machine's size, referenced
   to the null whose pair mean is exactly 0 because its halves are one run);
2. each half's own along >= 0.06 m (0.005 m/s): headway in both directions,
   so the uphill half must climb;
3. each half's |across| <= 0.5 x its along (within 27 deg of the line);
4. if the U arm's pair mean at that start is positive, the gait's pair mean is
   at least twice it;
5. both halves valid (no QACC reset, hashes equal at t_c).

Gravity cancels in the pair mean (a body sliding down the ramp at s adds +s to
one half and -s to the other); rule 2 stops a pair passing on the downhill
half alone. Verdict "walks" for a gait if at least 5 of the 6 pairs pass,
with at least 2 of 3 in each of up/down and across. Reported beside it, not
gated: the label "on its wings" when, over the window, wings and tail carry
at least 0.7 of the vertical ground force and the body touches on at most 20%
of steps (F2's bars); speed per half; cost of transport.

**Turn (c).** Starts yaw 0 (facing up), 90 (across), 180 (facing down), x 20,
gait phase 0; each a pair commanded to yaw + 90 and yaw - 90 at t_c. Window
14 s. A half passes if its heading error (mean over the last 2 s of the window)
is at most 15 deg and never exceeds 20 deg in those 2 s (it got there and
stays), and its net yaw change has the commanded sign. A pair passes if both
halves pass and are valid. Verdict "turns" for a gait if at least 2 of 3
pairs pass. CoM drift during the turn is reported, not gated.

**Nulls must fail.** N0 must fail stand (i); N0 and N1 must fail every walk
pair and every turn pair. If a null passes a gate, that gate is reported as
unable to tell a still machine from a working one and no verdict of that gate
is claimed.

## 4. Checks and predictions

Probabilities are mine, written now, after the two failed hand gaits of
section 0.

- C1. Pairing: every scored pair has equal hashes at t_c; F's pair mean
  |.| < 1e-9 m/s; N0 and N1 pair means exactly 0 (each a single run, since
  they never read the command).
- C2. S stands on at least 3 of 4 orientations: p = 0.8. Drift under 5 mm per
  10 s on each passing run; hold power 6-15 W electrical (the probe's 9.6 W
  included the rise); each wing's normal force within 25% of half the weight
  facing up or down; across the slope the downhill wing carries 0.55-0.70 of
  the wing load. Peak stroke torque in the hold 1.5-3 N m (static: half the
  weight at about 0.6 m is 1.8 N m).
- C3. N0 fails stand (i) on 4 of 4 (certain by construction: the hull carries
  about 0.95 of the weight) and drifts under 0.01 m in its 10 s; N1 and S read
  the same at yaw 0 and 180 to within 20% on power.
- C4. Walk: G1 "walks" p = 0.35; G2 p = 0.2; at least one p = 0.45. If a gait
  walks, its pair mean is 0.02-0.08 m/s, the uphill half is the slower on
  every passing up/down pair, and the across pairs' halves agree within 30%.
  "On its wings" for a walking G1 p = 0.4 (the probes set the hull down every
  stride).
- C5. Turn: G1 "turns" p = 0.5 (planted feet sweeping opposite ways is the
  simplest use of the two sides); turn rate 5-20 deg/s; CoM drift during a
  passing turn under 0.3 m.
- C6. U: below the 0.020 m/s bar on at least 5 of 6 pairs for each gait,
  and its pair mean under half the gait's wherever the gait passes: p = 0.7.
- C7. Pitch coupling: in every gait run the body's pitch relative to the ramp
  swings by more than 10 deg peak to peak (the probes: 39 deg).
- C8. Loads in walking: peak normal force on one wing 1-4 weights (probe 2.3);
  peak stroke hinge torque 3-15 N m, under the 24 N m stall at gear 24; at the
  motor's curve under 2% of steps on every role; feather at its range limit
  on under 1% of steps.
- C9. Energy: walking electrical power 10-40 W total, cost of transport
  20-200 on any passing pair.
- C10. QACC: no reset in at least 95% of runs.
- C11. J: every verdict of the best gait and S is unchanged at J_r 5e-6 and
  1e-6; window electrical energy changes by under 10% (at a stride of about a
  hertz the rotor's reflected inertia is small beside the load held).

Falsified by the stated numbers. A failed C1 means the pairing is not trusted
and no walk or turn verdict is read until it is fixed (disclosed).

## 5. Tuning protocol, budget, stopping, output

- Tuning starts (never scored): yaw +7 and -25 deg from +x, and 97 deg, gait
  phase 0.5, x 21; for S, yaw 30 and 135. Commands on them as in the gates
  (up, down, +y, -y; turns of +-90). Every tuning rollout (one half, at most
  14 s simulated) is appended to `results/M3_tuning.jsonl` with its full
  parameter set; none is dropped.
- Budget: at most 120 tuning rollouts each for G1 and G2, 20 for S. A
  family's parameters are frozen (written into `results/M3_frozen.json`, with
  its sha256 quoted in the result) when its budget is spent or when one set
  passes every gate on every tuning start, whichever is first; the frozen set
  is the one with the largest smallest margin over the tuning starts' gate
  quantities. No parameter outside a box; a box widened after data would be a
  new arm, disclosed as added.
- Scored runs: per gait 12 walk + 6 turn halves, the same for U (36 per gait,
  72), S 4, N0 6 + 3 + 4, N1 6 + 3, F 2, J 2 x 22: about 145 runs, about 170 s
  of wall each, about 6.9 h of CPU, **about 1.8 h of wall at 4 workers**;
  tuning about 1.6 h more. Run detached with a log (`results/M3_run.log`),
  output filtered; at most 4 worker processes.
- Stopping: when the scored table is filled, or when an arm is shown blocked
  with its trace, or after the tuning budget with no tuning start passing: then
  the gait is still scored as frozen, and a failing table is the result.
- Code: `m3_land.py` (arms, runner), `m3_read.py` (gates and tables),
  `m3_draw.py` or `draw.py` if it takes a recorded state; rows in
  `results/M3_land.jsonl`.
- Figures: for the best gait, from the built model at recorded states, at
  plant, mid-stance, lift and swing of one stride (side and front views,
  contacts and their forces marked), the stand on the slope across, and one
  instant of a turn: `figures/M3_*.png`, opened and checked against the
  geometry of section 0.
- The result file `RESULT_M3_land.md`: the outcome first (stand, walk, turn,
  each yes or no with the arm), a table, each check above against what was
  read, everything added after data disclosed.
