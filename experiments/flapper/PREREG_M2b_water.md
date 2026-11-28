# Pass M2b, pre-registration: stay up after the exit, swim by flapping, turn on the surface

2026-11-18, on `cadf9ab` (M2's result). Frozen before any scored row and
before any tuning rollout of the three arms below. The lead's brief for
M2b: (i) XV's exit, then M1's V+ flying law, with a 30 s gate; (ii) a swim
under water with the feather tracking the stroke so the wing meets the flow
at a set angle, at VB buoyancy and as built, on GW's pair gates; (iii) a
surface steering law that can reach a heading behind it, on GS's gates;
every row under both entrainment modes as co-primary; C0's kp mismatch
traced before scoring. Harness `mission.py` unedited; new code only in
`m2b_*.py`, rows in `results/M2b_*.jsonl`. The controllers read the true
state (modelled sensors are M5).

## 0. Seen before freezing (disclosed)

1. **All of M2** (`RESULT_M2_water.md`): every scored row, trial and check.
   In particular: the vault leaves the water under "wetting" (5/6) and not
   under the default (0/6); every M2 flight was lost in 4-10 s; the paddle
   moved 0.00-0.02 m/s and turned 4-17 deg per 10 s; the as-built body rises
   to the surface in 40 s whatever it does.
2. **C0's kp mismatch, traced** (one-line probes, no stepping).
   `TriphibianEnv.reset(AIR)` gives the body the launch speed (7.2 m/s along
   x), and `Machine` keeps it unless the pose sets `vx`. The PD gains are
   read from the mass matrix right after, with the wings' air added mass
   projected on that flow. Stroke-hinge J: 0.01792 built level at 7.2 m/s
   (M2's exit arms and C0: kp 1613), 0.01998 built at rest (M0's take-off
   check passed `vx=0`: kp 1798), 0.02106 at 45 deg pitch and 0.02427 at 90
   deg pitch at 7.2 m/s. Height, x and pitch at rest do not change it. M2b
   builds every air-gained machine at rest (`m2b_laws.build2`), and C0 is run
   again on it before any scored row. This is a property of the harness
   (`Machine("air")` without `vx`), reported, not edited.
3. **M1's V+ law** at commit 331e737 (`m1_takeoff.py` sha256
   ba0b7f69180a50ee, `m1_lib.py` 37f3452a286f64d5): `GAINS` and the class
   `Fly` are identical to M2's copy (`m2_fly.py`); `PLUS` differs (V+: kpP 1.5,
   kdP 0.2; M2's copy had 1.0, 0.3). Copied verbatim into `m2b_fly.py`.
4. **Probes of the two new laws** (`results/M2b_probes.jsonl`, seed 100,
   eight rollouts, 10-12 s, "wetting"):
   - Q1-Q4 ran with a fault (the feather-to-angle gain read from a body at
     rest was 0, so the tracking law left the feather at 0; Q3 and Q4 are
     the same rollout). Fixed: the gain is read from the built model's
     strip geometry for a fixed flow (-1 and +1 for the two wings).
   - Q5, Q6, the tracking law at 45 deg, both signs, as built from 3 m: sign
     +1 moved 1.26 m along -x and 0.50 m along -y in 10 s and **ended 0.18 m
     deeper** (the still body rises about 2 m in that time); sign -1 moved
     1.72 m sideways, yawed 64 deg, ended 0.07 m shallower. Both with the
     feather motors on their limit 52-59% of steps and **1.1-1.2 kW** mean
     electrical (the feather swings about 90 deg each half beat).
   - Q1, Q2, Q7, Q8, the rowing stroke at the surface (f 0.3, A 50 deg, bias
     -35 deg, downstroke 0.3 of the beat, feathers +-45 deg held): pushes
     (+1, +1) moved -0.47 m along x (and yawed -74 deg), (-1, -1) +0.20 m;
     turns (+1, -1) -6 deg and (-1, +1) +12 deg in 12 s. So (-1, -1) is
     nose-first and (-1, +1) turns +yaw, and both are weak: about 1 deg/s.
5. Cost: 15-40 s of wall per simulated second per process (M1b, M3b, M4
   run beside this pass).

## 1. What is the same for every arm

As M2 (`PREREG_M2_water.md` section 1): the machine, motors (gear 24,
J_r 2e-5), dt 1e-4, calm sea, every start at x = -30 m, the 10 Hz recorder
with energy, peak power over 50 ms (all motors together), motor-limit
shares, the force-limiter count, `bad_qacc` (a row with a reset is excluded
and counted not met), the seed draw (attitude scatter, jitter, stroke
phase, heading h, depth step Delta), the float start (placed level at the
surface, joints held 3 s), the depth start (CoM at 3 m). Changes:

- **Gains.** Swim arms built at their start in water (as M2); the exit arms
  built in air **at rest** (section 0, item 2), then placed.
- **Entrainment: co-primary.** Every scored row is run twice, with
  `solver.entrainment = "wetting"` and with the solver's default (`True`).
  **A verdict (met or not met) stands only where both modes give it; where
  they differ the result says "not established, the modes disagree"** and
  gives both counts.
- **Seeds.** Tuning 100-103; scored **10-15** (new; M2's scored 0-5 are not
  reused).
- **Tuning.** By hand, logged to `results/M2b_trials.jsonl`, tuning seeds
  only, either mode. Cap: 120 rollouts and 3000 simulated s across all
  arms. Laws frozen (sha256 of the code in the first row) before scoring.
  At most 4 worker processes.

## 2. The arms

### (i) Exit and stay up (gate GX30)

- **XS, vault then V+.** M2's XV exit unchanged (stroke stepped to -1.0
  rad from afloat; the flying law starts when no main-wing strip is wet, or
  after 2 s), then M1's V+ law **with its gains verbatim** (`m2b_fly.py`,
  H_t 4 m). No tuning.
- **XSt, the same, tuned.** The same structure with these alone tuned:
  the strike (-0.8 to -1.35 rad), the V+ gains kpy, kdy, kpz, kdz, kdx,
  kpP, kdP, k_v, lean_max, H_t (within 2-6 m). XSt = XS if no setting
  tried is better on the tuning seeds.
- **X0f, still afloat.**

### (ii) Swim under water by flapping with the flow angle held (gate GW)

- **WT, as built; WT-VB, on VB** (sealed volume x 0.584, 1.08 x weight, as
  M2). The stroke `b + s_i A/2 sin(phi)`; each wing's feather set every
  1 ms so that its reference strip (at 0.7 of its strip span) meets the
  flow it sees (its own velocity in the world, recoil included) at the set
  angle `a_set`, the sign alternating with the stroke's direction. **The
  angle range: a_set from 33 to 60 deg** (33: the least the +-57 deg feather
  can reach when the flow is normal to the wing, at zero forward speed).
  Depth: the angle differs between down- and upstroke by up to `zlim`
  (at most 15 deg) by the depth error; heading: the amplitude share (as
  WS), travel along the nose or the tail (tuned). Tuned: a_set, the sign,
  f (0.1-0.5 Hz), A (up to 80 deg), b, the loop gain, the depth and
  heading gains, the tail pitch loop. WT and WT-VB are tuned separately.
- **W0, VB-W0, still; WTF, WTF-VB, frozen** (the law with the first half's
  heading and depth in both halves).

### (iii) Turn on the surface (gate GS)

- **ST, rowing.** At the surface: the stroke biased down, a fast downstroke
  (a share `down` of the beat, 0.2-0.5) and a slow recovery; each wing's
  feather held at +-Psi_r (up to 55 deg). Opposite signs turn the body
  (probe Q7: (-1, +1) turns +yaw), the same signs push it along the hull
  either way ((-1, -1) nose first). The law turns toward the heading or its
  reverse, whichever is nearer, and once within `lock` (10-45 deg) rows
  along it, nose or tail first, steering by the amplitude share. Tuned: f
  (0.2-0.6 Hz), A (up to 80 deg, inside the joint range), b, down, Psi_r,
  lock, the steering gain.
- **S0, still; SF, frozen** (heading frozen at the first half's).

## 3. The gates

**GW and GS: M2's, unchanged** (`PREREG_M2_water.md` section 3): two
halves of 40 s from one state, told h and h + pi (GW: and 3 + Delta, 3 -
Delta), scored over t = 15-40 s; pair-mean progress >= 0.05 m/s with each
half > 0; GW's depth response >= 0.5, held within 0.5 m over t 35-40 s and
never shallower than 0.3 m; GS at the surface (depth <= 0.5 m); not tumbling
(dorsal z >= -0.25, mean |omega| <= 1.0 rad/s). An arm meets a gate on 5 or
more of 6 seeds with its still and frozen arms at 0 of 6, **in each mode**.

**GX30, out of the water and staying up, per seed.** Float start, the arm
runs 40 s. With t* the first sample at which no strip is wet, t* <= 5 s:
for 30.0 s from t*, (a) no strip wet at any sample; (b) the CoM's height
above the surface >= 1.0 m at every sample from t* + 2 s; (c) M1's G4 on
0.5 s means of the 10 Hz samples from t* + 1 s: nose elevation >= -10 deg
and |span-axis elevation| <= 35 deg; over the last 3 s of the 30, mean
|body rate| <= 1.5 rad/s. **An arm meets GX30 on 4 or more of 6, X0f at 0
of 6, in each mode.** M2's GX (3 s) is read on the same rows and reported.

Every exit verdict states in its outcome paragraph that the model has no
dynamic stall (X2) and no wave, splash or free-surface deformation, and
which entrainment mode it rests on.

## 4. Checks and predictions (numeric)

- **C0.** On `build2`, the held take-off lift within 0.5% of 1.840 (it is
  M0's build); if not, the exit rows are labelled.
- **C1 (nulls).** W0, VB-W0, S0, X0f, WTF, WTF-VB, SF: 0 of 6 on their gates
  in both modes; W0 at the surface by 40 s on 6/6; P = R = 0 exactly for
  the still and frozen arms.
- **C2 (exit, stay up).** XS meets GX30 on 0-2 of 6 under "wetting" (I hold
  0.6 that it fails the 4-of-6 bar: M2's flights lasted 4-10 s on a law
  with more pitch damping) and on 0 of 6 under the default (the vault did
  not leave the water under it in M2, 0/6). Verdict: not met in both modes
  (0.6). XSt: GX30 on 2-5 of 6 under "wetting" (I hold 0.5 that it reaches
  4), 0 under the default. M2's 3 s GX on XS under "wetting": 4 or more of 6.
- **C3 (swim, as built).** WT: GW 0 of 6 in both modes (0.7); half A (told
  deeper) ends short of its target by more than 0.5 m on 4 or more of 6.
- **C4 (swim, VB).** WT-VB: GW on 1-3 of 6 under "wetting" (it fails the
  bar, 0.6); depth response R >= 0.5 on 3 or more of 6; mean electrical
  power 300-1500 W (the feather swings).
- **C5 (surface turn).** ST: GS 0-2 of 6 in both modes (it fails the bar,
  0.6); its turn rate under 5 deg/s on every half (read: the largest 5 s
  change of heading); at least one half per seed gets within 45 deg of its
  heading or its reverse by t = 15 s on 3 or more of 6.
- **C6 (modes).** On every do-something arm the two modes' counts differ by
  1 or more seeds on at least one gate (the vault's dependence, M2).
- **C7 (power, reported).** Peak electrical (all motors, 50 ms) over 462 W
  in every do-something row of WT, WT-VB and XS/XSt; under 462 W mean for
  ST.
- **QACC.** Under 1 in 10 rows.

## 5. What a negative means

As M2: a gate failed at every arm is the result, with what limited it. No
gate, window, band or seed count is changed after a scored row; anything
added after data is disclosed. If the two modes disagree, that is reported
as the finding it is, not resolved by choosing one.

## 6. Output and budget

`m2b_laws.py` (build2, Stay, Track, Row), `m2b_fly.py`, `m2b_tune.py`,
`m2b_run.py` (rows, C0), `m2b_read.py` (gates per mode, the co-primary
verdict), `m2_draw.py` for figures (posed from the stored traces; at
several instants of a scored pair of each swim arm and of each exit arm's
best seed, both modes where they differ), `RESULT_M2b_water.md`.
Scored set: about 7400 simulated s (GX30 3 arms x 6 x 2 modes x 43 s, GW 6
arms, GS 3 arms, halves as above), about 12-16 h on 4 processes; tuning up
to 3000 simulated s.
