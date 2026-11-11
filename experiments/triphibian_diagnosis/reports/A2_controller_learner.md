# A2 — What the controller can command, and what the learner is given

Investigator A2, 2026-11-10, repo `main` @ `a7770c2` (read-only; nothing run but
streaming greps of `runs/*/events.jsonl` and small JSON reads; no MuJoCo, torch,
search or test). Paths are relative to the repo root. Line numbers are at `a7770c2`.

Tags: **[CODE]** what the code at `a7770c2` does (ground truth for "what happens");
**[MEASURED]** a stored measurement (doc, results file, or a number I computed from
stored telemetry with the scripts in §10); **[INFERRED]** my reasoning;
**[UNMEASURED]** / **[NOT FOUND]** as defined in the brief.

Vocabulary: a *decision* is one RL transition (one control step, what PPO
calls a transition); a *crossing* is a medium transition (`air_to_water`, ...).

---

## 0. The decision-relevant findings, first

1. **The learner's crossing reward is binary; the archive's is graded.**
   `batchroll.py:1259-1264` gives each crossing trajectory
   `float(t.crossed) * (0.40 + 0.60 * mean(shock, control, settle, economy, exit_state))`,
   where `t.crossed` is the boolean field (`transitions.py:132`, set at `:493`).
   The archive's `finalise_tier1` uses `component_means()["crossed"]`
   (`evaluate.py:185-194`), which is the graded `approach` (`transitions.py:214`,
   graded at `:713-726`). The comment at `batchroll.py:1255-1258` ("the same
   graded quantity") is wrong. **[CODE]** Consequence **[MEASURED]**: positive
   crossing rewards reached the learner on ≈47 of 55,480 crossing trajectories in
   arch51 (900 gens) and ≈8 of 20,080 in arch59 (§4.4); the crossing return is
   0.99-1.00 potential shaping (§5.3).
2. **The shared policy cannot reach a rotor.** It commands a 6-D body twist that
   reaches the gait only through each body's identified mobility basis; the
   identified bases move rotor speed by a median **0.0003 of top speed per unit
   coefficient** (191/191 arch49 rotor elites < 0.05; `experiments/rotor_authority/results_arch49.json`
   `aggregate.air.m_frac`, ROADMAP:3982-3987) **[MEASURED]**. The gait gain
   multiplies amplitude only (`cpg.py:427-430`) and a rotor's base amplitude is 0
   (`triphibian.py:734-739`), so the gain cannot throttle a rotor either **[CODE]**.
   The only machine that has held height through the air segment under any
   controller (the reference quad, 0.86-0.99 across the stored measurements) was
   flown by a hand law writing rotor speeds directly with privileged world state
   (`experiments/rotor/fly.py:41-96`), a channel the policy does not have.
3. **The per-design "own policy" is all zeros in practice.** arch59 air archive:
   0 of 146 elites have a nonzero own-policy weight; arch51: 0 of 69; arch56: 11 of
   111 **[MEASURED, archive JSON]**. The controller that actually acts is the shared
   PPO network (at its mean when scoring) on top of the evolved open-loop gait.
4. **What PPO is given in air and in crossings is, by magnitude, the shaping
   potential.** Share of |shaping| in the per-tag return (ROADMAP N10's quantity):
   air 0.95-1.00, every crossing 0.99-1.00, water 0.74-0.84, land 0.81-0.87
   (arch51/53/56/59) **[MEASURED]**. Potential-based shaping leaves advantages
   unchanged in expectation (`ppo.py:414-419` cites Ng et al. 1999), so where the
   terminal reward is zero it carries no expected gradient **[INFERRED, theory]**;
   and the air potential is altitude+upright with altitude saturated at the 30 m
   launch (tanh(d/5)), nothing about the turn (`ppo.py:428-432`, `triphibian.py:1666`)
   **[CODE]**.
5. **The only fixed-body learning experiment (M4) never finished, and its air arm
   is not evidence about learnability.** 7 of 9 runs recorded (ROADMAP:5682-5707);
   `results_arch48.json` **NOT FOUND** (main tree and the nine `../arch5x-tree`,
   `../arch60-tree` worktrees). Its air arm had ≤48,000 air decisions per seed
   (computed, §6.1; < the PREREG §3.6 line of 1e5), ran on the old unpaired air score,
   on a **rotor** body whose rotors the policy cannot reach (finding 2). Water was
   learned (0.0003 → 0.04-0.07, above both controls). E1 (shared policy on/off)
   **never ran**: `origin/e3-cpu-runs:e1/status.json` has 0 generations in all 10 arms.
6. **Budget per (body, task) is tiny and the body never repeats.** One evaluation
   banks ≤1,800 decisions: 400 air (one mirrored turn draw), 400 water, 400 land,
   150 per crossing kind **[CODE + MEASURED: 28,797-28,800 at gen 0/1]**. A 300-gen
   PPO-only run banks 7.0-8.3M decisions over ~3,500 new bodies **[MEASURED]**.
   Each crossing kind starts from one fixed entry draw for every machine of every
   generation (`transitions.py:223-235`, `batchroll.py:1313-1319`) **[CODE]**.
7. **Decoder and observation gaps for mode switches.** Land is driven through the
   *air* basis (`evaluate.py:54-58`); every crossing uses one basis for both media
   (water's for any crossing touching water, air's for `land_to_air`;
   `batchroll.py:1324`); during crossings and the continuous mission the six task
   channels are zero (`triphibian.py:1704-1711`), so hold→go is signalled only by
   the commanded-domain one-hot flipping at 1.5 s (`transitions.py:398-400`); there
   is no distance or direction to a boundary in the observation **[CODE]**.
8. **Exploration knocks gliders out of the air and does not shrink.** At the
   search's `log_std` −0.5, 3.5% of sampled gannet rollouts stay aloft through the
   turn command (`experiments/air_turn_bar/README.md:291-298`) **[MEASURED]**; the
   learned `log_std` ended above its start in 8 of the 9 runs read (arch53-60, final
   −0.23 to −0.54) and fell only in arch51 (−0.50 → −0.60) **[MEASURED]**.

---

## 1. Action space (Q1)

### 1.1 What the policy outputs each control step **[CODE]**

- **Shared policy** (PPO): `a = tanh(u)`, `u ~ N(mu(obs), exp(log_std))`
  (`ppo.py:185-199`, sampling `:262-319`). Dimension `TWIST_DIM + 1 = 7` with the
  gait gain, 6 without (`loop.py:1874-1880`; `cpg.py:183-186`). The first 6 are
  body-twist intents `[surge, sway, heave, roll, pitch, yaw]`, each read "as a
  fraction of what this body can do on that axis" (`cpg.py:504-527`); the 7th is
  the gain intent. Every run arch51-arch60 has `gait_gain = 1` (config line 1 of
  each `generations.jsonl`); arch47-50 had it off.
- **Own policy** (per design): `tanh(obs @ W + b)` with `policy_hidden = 0`
  (linear) in every stored run, 6 mode coefficients + 1 gain intent
  (`cpg.py:647-706`, `loop.py:240,491-504`).
- **How they combine** (batched path, `batchroll.py:880-916`): mode coefficients
  `c = own(obs)[:6] + basis.coeffs_for_twist(shared(obs)[:6])`; gain intent
  `own[6] + shared[6]` (each in [−1, 1]); `gait_gain(intent) = clip(1 + intent, 0, 2)`
  (`cpg.py:203-208`). Single-machine paths do the same via `SummedPolicy`
  (`evaluate.py:61-109`), shared half at its mean.

### 1.2 How it maps to actuators **[CODE]**

1. **Twist → mode coefficients:** damped least squares on `A = effects * authority`
   (`cpg.py:463-502`), target `intent * reach * INTENT_AUTHORITY`, with
   `INTENT_AUTHORITY = 0.5` (`cpg.py:242`) and `reach = ||A[:, j]||` per axis.
   All stored modes (`max_modes = 6`) are commanded; `control_rank` is used only in
   telemetry (`loop.py:1256-1260`, `ops/run.py:559-563`).
2. **Coefficients → CPG parameters:** `params = base + modes^T c`, then
   `amplitude *= gain` (`cpg.py:417-430`). `modes` is (r, P) with
   `P = 3n + 1`: per-joint amplitude, phase, offset, and one shared frequency
   (`cpg.py:45-72`).
3. **Clipping** (`cpg.py:74-92`, applied in `CPG.command` at `:160-161`): offset
   held 5% of half-travel inside the joint range (0% for rotor channels,
   `triphibian.py:731-733`); amplitude in `[0, span − |offset − mid|]`; frequency in
   [0.1, 20] Hz.
4. **CPG → joint targets:** `q_ref = offset + amplitude * sin(2πft + ψ + phase +
   phase_offset)`; ψ keeps the stroke continuous when *frequency* changes; a change
   of the *phase* or *offset* parameters jumps the target (`cpg.py:155-171`).
5. **Targets → torque:** MuJoCo position servo, `kp = 2·τ_stall·g`,
   `kv = 0.15·τ_stall·g` (`g` = the `drive_compliance` gene), torque limited to
   ±τ_stall, `ctrlrange ±3.2` (`core/mjcf.py:807-828`), plus velocity feed-forward
   `q_ref + (kv/kp)·dq_ref/dt` (`triphibian.py:647-660,1739-1746`).
6. **Rotors** (`*_r` channels): a velocity servo, `ctrlrange [0, ω_max]`
   (`mjcf.py:795-806`); the CPG channel is a speed held at its offset
   (`= rotor_throttle` gene), base amplitude 0 (`triphibian.py:734-739`). The policy
   moves it only through the mode coefficients' offset component — measured
   negligible (finding 2). **Feathering** (`*_f`) is a second hinge channel with a
   phase lead (`triphibian.py:740-745`). **Jets** take no separate command: thrust
   is a function of the bell joint's angle and rate (`physics/jet.py:158-200`).

So the policy is a **25 Hz modulator of an evolved open-loop sinusoidal gait**: it
moves amplitude/phase/offset/frequency of a fixed oscillator bank, never a joint
angle or a rotor speed directly. **[CODE]**

### 1.3 Control rate, physics dt, feedback inside a period **[CODE]**

- Physics `timestep = 0.004` s (`triphibian.py:613`); decisions every
  `int(1/(25·0.004)) = 10` steps = **25 Hz / 40 ms** (`batchroll.py:823`, crossings
  `:1326`, single path `triphibian.py:1797-1798`, mission `mission.py:216`).
  Confirmed by telemetry: 16 bodies × 1,800 = 28,800 decisions/gen (§4).
- Between decisions the CPG parameters are held (`cur[m]` reused,
  `batchroll.py:920`); targets are a function of time only. The only feedback inside
  a 40 ms period is each joint's PD servo (and the rotor velocity servo). No body
  state (attitude, altitude, rates) is fed back inside a period.
- Stroke frequencies: `flap_frequency` genes drawn in [1.5, 12] Hz
  (`core/genome.py:401,642`), clipped to [0.1, 20]: at 7 Hz a decision spans ~0.28
  of a stroke **[INFERRED arithmetic]**.

### 1.4 Definitions as implemented

- **Base gait** **[CODE]**: `env.cpg.base`, built per actuator from genome genes:
  amplitude `= stroke_amplitude × half-travel`, offset `= neutral` (or
  `rotor_throttle` for rotors), phase `= phase_offset × chain depth` (+`feather_lead`),
  frequency `= flap_frequency` (`triphibian.py:697-750`; genes `core/genome.py:125-191,262`).
  It is evolved by mutation, never by gradient. With a zero policy it is exactly what
  runs ("fall back to the raw pattern generator", `loop.py:494-498`).
- **Gait gain** **[CODE]**: one more action on each policy; summed intent →
  `clip(1 + intent, 0, 2)` × commanded amplitude; zero intent leaves the gait
  bit-identical (`cpg.py:188-222,417-430`; ROADMAP:6989-7002). Published per phase as
  `gain_cruise/hold/stop` (`triphibian.py:1889-1909`).
- **Stop authority** **[MEASURED, ROADMAP:6972-6987]**: the ability of one bounded
  `c` to zero every actuator's amplitude at once. On arch42's top 8 elites the best
  bounded `c` left a median 74% of base amplitude on land and 50% in water; an
  oracle zeroing amplitude on stop/hold lifted land task 0.362 → 0.501 and stop
  0.356 → 0.902; water hold 0.000 → 0.000 ("necessary for hold and not sufficient").
  The gait gain is the fix (on since arch51). **[INFERRED from CODE]** it does not
  give stop/throttle authority over rotors (base amplitude 0, speed = offset).

---

## 2. Observation (Q2)

### 2.1 The 33 channels **[CODE, `triphibian.py:1602-1678`, `OBS_DIM = 25 + 8` at `:1727-1730`]**

| idx | channel | note |
|---|---|---|
| 0-2 | body-frame linear velocity / 5, clip ±3 | `body_twist` is the root's own frame (`:1352-1358`) |
| 3-5 | body-frame angular rate / 4, clip ±3 | rate gyro |
| 6-8 | gravity in the body frame | roll/pitch attitude; no yaw |
| 9 | `tanh(depth / 5)` | depth below the local surface (negative above); **saturated above ~10 m**: −0.99999 at the 30 m launch |
| 10 | mean submerged fraction | wetness |
| 11-13 | one-hot of the *commanded* domain | segment's medium; in crossings the start medium then the target (`batchroll.py:1351-1352`) |
| 14 | `tanh((depth − commanded depth)/3)` | water target |
| 15 | terrain contact (0/1) | contact with world body, incl. the submerged ramp (`:1360-1375`) |
| 16 | battery fraction | |
| 17-18 | mean normalised stroke position and rate | |
| 19-24 | task: `[cruise, hold, speed/ref, cos(heading err), sin(heading err), tanh(height lost/3)]` | **all zero outside a segment** (`:1704-1711`); height lost only in air |
| 25-32 | morphology: log mass, density ratio, wing area, span, aspect ratio, wing loading, n_actuated, battery/mass | fixed transform, constant per body (`:486-516,1562-1600`) |

### 2.2 What it does and does not carry **[CODE]**

- **Medium:** yes, indirectly (wetness, depth, contact) and the commanded medium
  (one-hot). No explicit "current medium" flag.
- **Command:** in segments, heading error as (cos, sin), commanded speed, phase
  kind; the **air turn** appears only as a step in the heading-error channels at
  4 s (both air phases are `CRUISE`, `tasks.py:212-219`). In crossings and in the
  continuous mission (`mission.py:224-242`, no `_arm_task`), **no command channels
  at all**: hold-then-go is visible only as the one-hot flipping at
  `HOLD_SECONDS = 1.5` (`transitions.py:250-256,398-400`).
- **Phase (hold/go):** segments yes (channels 19-20); crossings only via the one-hot.
- **Altitude:** saturated `tanh(d/5)` and, in air segments only, height lost since
  the segment began. No absolute clearance, no world position by design (`:1605-1608`).
- **Attitude / angular rates:** gravity vector and gyro, yes. Yaw only through the
  compass in cruise phases.
- **Contact:** binary.
- **Distance or direction to the next medium boundary:** **no** — no shore bearing,
  no shore distance, nothing in crossings. (A water→land crossing starts 2-12 m
  seaward, `transitions.py:547-555`.)

### 2.3 One network, many bodies **[CODE]**

Fixed widths both ways: no per-joint channel in the observation (stroke is the mean
over joints), and the action is a twist decoded per body by its basis, so `P = 3n+1`
(16-49 on the seed plans, `experiments/rank_threshold/README.md:26-34`) never reaches
the network (`ppo.py:23-30`). Conditioning is the 8 morphology scalars plus the
per-body basis. Both are evaluated below:

- **Basis per medium, identified, not learned** (`batchroll.py:718-795`): 24 random
  probes `δ ~ N(0, 0.35)` on every CPG parameter (rad, rad/s or Hz alike), ±δ, 1.2 s
  each, from the unscattered spawn; least squares + SVD (`cpg.py:601-644`). Air and
  water only (`batchroll.py:1118`); land uses air's (`evaluate.py:54-58`). Every new
  child is identified at its own seed (`identify_axes_every = 1` in all configs;
  `loop.py:2140`); re-evaluated elites reuse their stored basis (`loop.py:2144-2146`).
- **[MEASURED] the basis is not reproducible on the parameter side**
  (`experiments/rank_threshold/README.md:21-78`): underdetermined on 5 of 7 seed
  plans (P > 24, residual reads exactly 0), `||J_a − J_b||₂/σ₀ = 1.132` between
  reseeds (no singular value resolvable from one identification), parameter-side
  principal angles 68-85° between seeds, twist-side 30-41°, singular values CV
  0.15-0.21. "What the machine can do is identified. How to ask for it is not."
- **[MEASURED] closed-loop command authority is near zero on arch48's 200 elites**
  (`experiments/step_response/README.md:46-71`): the commanded angular rate explains
  median dR² ≈ 0.002 of the realised rate beyond momentum, equal to a
  shuffled-command control (air 0.0022 vs 0.0017, water 0.0016 vs 0.0011, land
  0.0018 vs 0.0020); 23-32% of segments beat the shuffle at p < 0.05. Caveats in the
  README (`:87-92`): closed-loop, no excitation, flapping oscillation in the rate.
- **[MEASURED] morphology conditioning does not generalise across bodies**
  (ROADMAP:7484-7522, arch33 era): a conditioned student distilled from 550 per-body
  ES policies reaches held-out-body R² −0.047 against a constant 0.011 and an
  unconditioned 0.024 (train 0.664). Old observation and teachers; states sampled
  from the observation's envelope, not traces.

---

## 3. Policies (Q3)

### 3.1 Shared policy **[CODE]**

Actor and critic are separate MLPs `33 → 64 → 64 → out` with tanh, orthogonal init,
actor output gain 0.01 (starts near "command nothing"), critic 1.0
(`ppo.py:85-110,130-137`); state-independent `log_std` parameter (`:144`); running
observation mean/var buffers, clamp ±10 (`:153-161`). **Parameters: 13,133 without
the gain (6,726 actor + 6,401 critic + 6 log_std), 13,199 with it** (my arithmetic;
HYPERPARAMETERS.md:107 states 13,133). Stale numbers in the code's own prose:
`ppo.py:61` "about 1,200 weights", `ppo.py:149` "27 channels",
`cpg.py:663` "Linear is 60 weights", `LEARNER_AUDIT.md:372` "~1,200 weights".

### 3.2 Own policy **[CODE + MEASURED]**

Linear, 33 × 7 + 7 = **238 weights** with the gain (matches the stored vector
length in arch59's archive). Built from the parent's weights when the shape
matches, else zeros (`loop.py:491-504`); seeds start at zero. It changes only by
(1+1)-ES / (1+λ) refinement accepted on `mission_fraction` strictly improving
(`loop.py:916,951-960,3058-3065`): `controller_refine_steps = 0` in every run since
arch50, so only at promotion (6 trials, σ 0.02 from arch54; 0.1 default)
(`loop.py:195,227,235`; configs). `mission_fraction` was above zero in 7 of arch48's
4,792 evaluations, 29 acceptances in 300 gens (ROADMAP:266-269) — the ES has
essentially nothing to accept on. Result in the archives **[MEASURED]**: all-zero own
policies in 69/69 (arch51 air), 146/146 (arch59 air), 100/111 (arch56 air).

### 3.3 Evolved vs measured vs learned

| what | how it changes | where |
|---|---|---|
| body, actuators, battery | genome mutation, selection | `core/genome.py` |
| base gait (amplitude, neutral, phase advance, frequency, throttle, feather lead, servo compliance) | genome mutation | `core/genome.py:125-191,262`; `triphibian.py:697-750` |
| mobility basis (action decoder) | identified per evaluation, 24 probes | `batchroll.py:718-795` |
| shared network (13k params) | PPO, one update per generation | `loop.py:2233-2270` |
| own linear policy (238) | inherited; ES at promotion | §3.2 |

### 3.4 "latent" and "phi" **[CODE]**

`SharedPolicy.latent(obs)` is the pre-squash Gaussian `N(actor(norm(obs)),
exp(log_std))`; the action is `tanh` of a sample and log-probs carry the tanh
Jacobian (`ppo.py:185-214`). `phi` is the shaping potential `potential_of(obs,
domain)` recorded per decision (`ppo.py:403-440`, recorded at `batchroll.py:908-912`,
`:1375-1379`; §5.2).

---

## 4. PPO as run (Q4)

### 4.1 Settings **[CODE; values from every arch51-60 config line]**

| item | value | where |
|---|---|---|
| update cadence | one update per generation over everything the generation's *main sampled* evaluation banked | `loop.py:2152-2168,2233-2255` |
| lr | 1e-3, linearly annealed to 0 over `generations` (`frac = 1 − gen/G`) | `loop.py:256,292,2235-2237`; `ppo.py:775` |
| epochs / minibatch | 10 / 2,048, shuffled by a seeded, checkpointed stream | `loop.py:257-258`; `ppo.py:784-790` |
| clip (policy and value) | 0.2 | `ppo.py:614,794-804` |
| target KL (per epoch, k3 estimator) | 0.015 | `ppo.py:833-851` |
| entropy coef | 0.01 (reparameterised estimator) | `loop.py:265`; `ppo.py:216-251` |
| vf coef / grad-norm clip / Adam eps | 0.5 / 0.5 / 1e-5 | `ppo.py:614-615,772,821` |
| γ / GAE λ | 0.99 / 0.95 (RolloutBuffer defaults; no config field) | `ppo.py:457-458`; `loop.py:2155` |
| value target | GAE return `adv + V_old`, zero terminal bootstrap | `ppo.py:516-531`; LEARNER_AUDIT.md:319 |
| normalisation | obs running mean/var (updated after the update); terminal reward ÷ max(std of its tag, 0.05); advantages standardised batch-wide | `ppo.py:159-181,479-498,748-751,855` |
| shaping weight | 0.2 | `loop.py:213` |
| initial log_std | −0.5 (arch55: −2.0) | `loop.py:255`; configs |
| air-turn reward weight | 1.0 in arch53-60, 0 before | `loop.py:220`; configs |
| learner | ppo (arch57: ppo+grpo, groups first from measurable bodies) | configs |
| exploration noise | i.i.d. per decision, per-machine stream, only in the main evaluation; scoring rollouts act at the mean | `batchroll.py:857-861,1154-1160`; `loop.py:826-841` |

Segment lengths in decisions **[CODE]**: air/water/land half 8 s = 200 (turn command
at decision 100; scored over the late half of each phase, decisions 50-100 and
150-200, `triphibian.py:1961-1969`); a mirrored/antipodal pair = 2 × 200; crossing
6 s = 150 (hold 0-37, go 38-149).

### 4.2 Where the count lives

Not in `generations.jsonl` (its lines carry `cost.seconds.ppo` and
`cost.steps`); the field is `transitions` in `events.jsonl` lines with
`"kind": "ppo"` (`loop.py:2264-2270`), with `trajectories`, `grad_steps`,
`log_std`, `kl`, `reward_by_tag` (mean, std, n per tag) and `return_by_tag` (N10).

### 4.3 Measured totals **[MEASURED, `runs/archNN/events.jsonl`, §10 scripts]**

| run (gens) | updates | decisions total | mean/update | grad steps | stopped early | log_std first → last |
|---|---|---|---|---|---|---|
| arch51 (900) | 900 | 22,922,729 | 25,470 | 130 | 0 | −0.50 → −0.60 |
| arch53 (300) | 310 | 8,177,506 | 26,379 | 135 | 0 | −0.50 → −0.36 |
| arch54 (300) | 300 | 8,074,666 | 26,916 | 137 | 0 | −0.50 → −0.45 |
| arch55 (300) | 300 | 8,319,916 | 27,733 | 125 | 55 | −1.99 → −0.54 |
| arch56 (300) | 300 | 8,224,101 | 27,414 | 141 | 0 | −0.50 → −0.40 |
| arch57 (300, +GRPO) | 313 | 20,025,332 | 63,979 | 318 | 0 | −0.49 → −0.23 |
| arch58 (300) | 304 | 7,033,519 | 23,137 | 118 | 0 | −0.49 → −0.42 |
| arch59 (300) | 319 | 7,411,162 | 23,232 | 119 | 0 | −0.49 → −0.48 |
| arch60 (150, device) | 140 | 3,962,632 | 28,305 | 146 | 0 | −0.50 → −0.41 |

(Update counts above the generation count are resumed generations.) Mean KL
0.0058-0.0078. Gen 0/1: 28,797-28,800 = 16 machines × (6 halves × 200 + 4 crossings
× 150). PREREG §3.4's "total below 2e7": met by every 300-gen PPO-only run, **exceeded**
by arch51 (900 gens) and arch57 (GRPO rows).

### 4.4 Per (body, task) **[CODE + MEASURED]**

- Per evaluation: ≤400 air decisions on **one** turn draw (its mirror is the second
  half), ≤400 water, ≤400 land, ≤150 per crossing kind; one evaluation per new body
  (`draw_per_candidate`, `loop.py:717-720`).
- Per 300-gen run: air trajectories 9,444-10,040 (= 4,722-5,020 body evaluations, of
  which 4 per generation are re-evaluated elites, so ~3,500 new bodies); each crossing
  kind 4,722-5,020 attempts (reward_by_tag n summed).
- Air reward nonzero in any trajectory of the update: arch53 183/310, arch54
  116/300, arch56 152/300, arch57 136/313, arch58 94/304, arch59 41/319 updates;
  update-mean air reward > 0: 14-44 updates per run; arch51 (no air-turn reward)
  5/900.
- Crossing successes reaching the learner (estimated as `k = n·m²/(s² + m²)` from
  each generation's mean m, std s, count n; exact when the positives are equal, and
  every positive generation reads k ≈ 1.0): arch51 ≈47 of 55,480 (w→l 9, w→a 15,
  l→a 23); arch54 ≈30 of 18,980; arch57 ≈11 of 19,556 (all l→a); arch59 ≈8 of
  20,080 (all l→a). `air_to_water`: 0-1 per run. These are completed Tier-1
  crossings (`crossed` True), so "every crossing count exactly 0" (PREREG §3.1) does
  not hold literally; the "≥ 95% of generations at 0" clause does ((update, kind)
  pairs with a positive mean: arch51 46 over 900 updates, arch59 8 over 319, arch60
  0 over 140).
- Where the simulated steps go **[MEASURED, arch59 gen 1 `cost.steps`]**: of
  1,809,520 physics steps, 288,000 (15.9%) were banked for PPO; the rest were
  identification (345,600), the noise-free re-score (292,784) and `placement_draws 4`
  re-scores (878,352). Without placement draws (arch56-58) the banked share is ~31%
  **[INFERRED arithmetic]**. The device path's K draws feed nothing to PPO in the
  loop: `loop._device_eval` passes no buffer (`loop.py:1069-1075`), although
  `rollout_wp` can bank them since `f1281bd` (after arch60's pinned tree).

---

## 5. The reward the learner is given (Q5)

### 5.1 Terminal reward per tag **[CODE]**

`learner_reward(seg)` = the segment's competence, plus in air
`air_turn_reward × pair_turn_mean` when the pair was measurable (`ppo.py:349-371`).
Both halves of a pair receive the *pair's* score (`batchroll.py:1218-1229`): the
first half's trajectory is rewarded for an outcome partly decided by a later,
separate trajectory **[INFERRED from CODE]**.

- **Air** (`triphibian.py:2330-2678`): `credit × frac × turn × (0.5 + 0.5·height) ×
  chatter`. **0 if airborne < 0.4 s** (`:2365-2382`); **0 if airborne < 2.8 s**
  (`:2406-2455`); `turn = clip(pair mean of the signed lateral gain toward the
  commanded side, 0, 1) × factor` where `factor = served × airborne` of the worse
  half (`:2159-2194`, `paired_turn` `:3053-3072`); a non-reader's pair mean is
  `−0.5|a−b| ≤ 0` by construction (`experiments/air_turn_bar/README.md:84-95`);
  spin > 2π rad/s or a Tier-0 air gate cuts credit to 0.05 (`:2391-2397`, `:283,311`).
  So air pays **nothing until a body is aloft through the command in both mirrored
  halves and turns toward it**; with `air_turn_reward = 1` the learner additionally
  gets the unclipped (≤0 for non-readers) pair mean, hence the negative air means in
  arch53-60 (§10).
- **Water** (`:2680-2783`, `:2135-2158`): `served × submerged × (0.5 + 0.5·upright) ×
  (0.5·progress + 0.5·hold)`; progress on the mean signed speed of the antipodal
  pair (passive motion cancels), hold = at commanded depth × still × level. Graded.
- **Land** (`:2785-2998`, `:2110-2134`): `served × (0.5·contact + 0.5·upright) ×
  progress × (0.5 + 0.5·stop)`. Graded.
- **Crossings**: binary `crossed × (0.40 + 0.60·quality)` (finding 1). `crossed`
  needs the start medium, a hold score ≥ 0.5 over 1.5 s (air: energy height),
  arrival in the target after the go command and still there at the end, ashore for
  land, terminal clearance > 0.5 m for air (`transitions.py:325-499`). The graded
  `approach` (≤ 0.6 × progress × hold) exists but the learner never sees it; it
  enters `mission_fraction` (`evaluate.py:185-194`), whose weight in the selection
  scalar is 0 since ARCH51 (`loop.py:196-207`), and the curriculum's stage scores via
  `component_means` (`curriculum.py:194-218`). The air island's own score reads its
  two crossings as a *binary* factor floored at 0.15:
  `base × max(crossed_fraction × (0.4 + 0.6·quality), 0.15)` over `air_to_water`
  and `water_to_air` (`islands.py:71-73,272-285`). (Selection is other
  investigators' ground; recorded here because it is the same binary-vs-graded split.)

### 5.2 Shaping **[CODE]**

Per step `0.2 × (γΦ(s') − Φ(s))`, Φ(terminal) = 0 (`ppo.py:499-504`), so a
trajectory's discounted shaping sum is exactly `−0.2 Φ(s₀)`. Φ by the *commanded*
domain (`ppo.py:428-440`; crossings use the told medium, `batchroll.py:1375-1379`):

- air: `clip(−tanh(d/5), −1, 1) + 0.5·upright` — altitude saturates above ~10 m;
  nothing about heading or the turn;
- water: `−|tanh((d − target)/3)| + 0.25·wet` — the hold task;
- land: `contact + 0.5·upright` — contact includes the submerged ramp, so it does
  not point ashore;
- otherwise `0.5·upright`.

No term rewards approaching a boundary, staying aloft through the command, or
turning **[CODE]**.

### 5.3 What the learner's return is made of **[MEASURED, N10 `return_by_tag`]**

Mean over updates of |shaping| / (|shaping| + |terminal|):

| run | air | water | land | a→w | w→a | w→l | l→a |
|---|---|---|---|---|---|---|---|
| arch51 | 1.000 | 0.741 | 0.814 | 1.000 | 0.996 | 0.998 | 0.991 |
| arch53 | 0.949 | 0.811 | 0.860 | 0.999 | 1.000 | 0.999 | 0.999 |
| arch56 | 0.955 | 0.829 | 0.807 | 0.998 | 1.000 | 0.995 | 1.000 |
| arch59 | 0.992 | 0.842 | 0.871 | 1.000 | 1.000 | 1.000 | 0.991 |

Terminal returns did not rise: land first-50 → last-50 updates 0.050 → 0.031
(arch51), 0.034 → 0.019 (arch59); water 0.029 → 0.041 (arch51), 0.021 → 0.019
(arch59); air (arch53-59) stays ≤ 0 on average. This confirms ROADMAP T6
(`ROADMAP.md:3873-3879`: "PPO learns the shaping potential, not competence").

---

## 6. Every stored learner run on a fixed body (Q6)

### 6.1 M4, `experiments/fixed_body_learning` (2026-10-08/09) — the only one

- **Design** **[CODE, `run.py:78-138`; README:8-31]**: three arch48 elites (best air
  = elite 1, **a rotor body**; best water = 152; best land = 29), 3 learner seeds, a
  fresh shared policy, no own policy, stored bases, 30 PPO updates of 8 rollouts each
  through `evaluate_tier1_batch` (**the full Tier-1: all three media and four
  crossings banked, not only the body's medium**), held-out at 8 fixed draws every
  10 updates; controls still machine and base gait.
- **Decisions** **[INFERRED arithmetic from the harness; not read from a results
  file]**: per copy 200 air (air unpaired then: "its process loaded the code before
  the air change", ROADMAP:5684-5685) + 400 water + 400 land + 600 crossings = 1,600;
  per update ≤12,800; **per seed ≤384,000 in total, of which ≤48,000 air decisions
  (240 air rollouts) and ≤96,000 water**. Upper bounds: truncated rollouts bank fewer.
- **Result, 7 of 9 runs** **[MEASURED, ROADMAP:5688-5707]**, held-out own-medium
  competence at updates 0/10/20/30:
  air (rotor): 0.125/0.045/0.004/0.003, 0.126/0.057/0.088/0.143,
  0.133/0.070/0.053/0.127 — base gait 0.128, still 0.006, mean change −0.037: "no
  learning". Water: 0.0002→0.069, 0.0003→0.056, 0.0004→0.040 (seed spread 0.029;
  controls 0/0): learned. Land: 0.002→0.050, 0.006→0.001, third seed "running".
- "Learned" means `mean(gain) > max−min of the seeds' finals and > 0`
  (`run.py:38-44`).
- **Final read NOT FOUND**: no `results_arch48.json` or `.rows.jsonl` in the repo or
  any of the nine worktrees; nothing in ROADMAP after the partial table.
- **What the air arm can and cannot say** **[INFERRED]**: by PREREG §3.6's own line
  (< 1e5) it is **UNMEASURED for R2**; independently, the score was the pre-pair
  air score, and the body was a rotor whose rotors the policy cannot reach (§1.2.6,
  finding 2) — so "no learning in air" is uninformative about whether PPO can learn
  a commanded turn on a body it can steer.

### 6.2 Other runs with a fixed body, none of them a learning curve

| experiment | body | what acts | reward/read | result |
|---|---|---|---|---|
| Exploration read, `air_turn_bar` README:276-305 (2026-10-22) | 8 arch49 gannet gliders | shared network's mean zeroed, sampling noise only, 32 copies × 4 `log_std` | share aloft through the command (`pair_turn_factor` > 0.5) | −0.5: 9/256 (3.5%), 0 positive; −1.5: 29%; −2.5: 40%; −4.0: 38%. No learning |
| B', README:186-209 | same 8 gliders | 32 random linear own policies, σ 0.5, no shared | replicated pair mean | 6/8 bodies 0/32 measurable; 0 bodies > 0.1 |
| B'', README:216-230 | same | σ 0.1 and 0.02 | same | σ 0.02: body 159 replicated **0.119** over 4/4 fresh draws (18/32 measurable): a gannet *can* be turned ~12% by a linear reader, without learning |
| shared_policy_value, README:59-117 (arch46) | 218 elites, re-scored | shared network on vs off | competence delta | land +0.025 (t 3.5), water +0.013 (t 2.6), air +0.006 (t 1.7); medians 0; an upper bound (selection bias) |
| rotor_authority N6 (2026-10-10) | 24 arch49 rotor bodies + the quad | hand PD hover law (i) through the basis, (ii) direct on ctrl | air competence | (i) median 0.000, (ii) median 0.000 (max 0.446), quad 0.816; 153/191 rotor elites cannot lift themselves |
| distillation (arch33, ROADMAP:7484-7522) | 550 stored own policies | supervised student | held-out R² | −0.047 vs constant 0.011 |
| GRPO arch57 (ROADMAP:426-438) | groups of one body inside the search | ppo+grpo, measurable-first | air reads | 72-81% of air groups zero variance; every read fell |
| E1/E3, `experiments/e1_shared_policy/NOTES.md` | search, 5 seeds × on/off | — | — | **never ran**: `origin/e3-cpu-runs` `e1/status.json` 0 gens in all 10 arms (2026-10-12) |

---

## 7. Reference and scripted controllers (Q7)

- **Quadrotor control**, `experiments/rotor/fly.py:41-96` **[CODE]**: senses the full
  mass matrix, world rotation, world position (`xipos`, holds the start height `z0`),
  world linear velocity (`qvel[:3]`), body rates and rotor geometry, plus the task's
  heading and speed; computes a velocity→tilt cascade, attitude PD and an
  allocation mixer; **writes four rotor speeds every physics step (250 Hz)** by
  replacing `env.cpg.command`. Body: `REFERENCE_PLANS["quad"]` (never seeded).
  Scores **[MEASURED]**: 0.963/0.974/0.987 (2026-09-26, ROADMAP:6252-6255);
  0.864/0.858/0.922 in the stored `experiments/rotor/results/fly.json` (matching
  ROADMAP's "0.86-0.93 under AE's worst-second height term");
  0.816 at seed 0 on 2026-10-10 (`rotor_authority` `quad_check`) — all at commits
  **without** the mirrored air pair (`git merge-base`: 5fba1e5 lacks 79a2424). Under
  the current paired air score: **UNMEASURED**. Water and land for this body under
  any controller: **NOT FOUND**.
- **N6 hand laws**, `experiments/rotor_authority/run.py:528-562` **[CODE]**: (i) a PD law
  emitting the 6-D twist intent through `coeffs_for_twist` — **the policy's own action
  space**; (ii) the quad mixer direct on rotor ctrl. Both via `env.cpg.command` on the
  single-machine path. 0 of 24 arch49 rotor bodies flew under either (§6.2).
- **Flight audit** (`experiments/flight_audit/*.py`) **[CODE]**: open-loop gaits only.
  `feather_fly.py:34-44` flies quasi-static best `CPGParams` through `env.rollout`
  (sink 9-14 m/s, competence 0, ROADMAP:6275-6276); `fixed_rig.py`, `rig_level.py`,
  `level_flight.py` are rig probes with prescribed or free joints, not free flight.
- **Scripted swimmer, walker or crossing controller: NOT FOUND** (grep of
  `experiments/`, `dytiscidae/`, `tools/` for `def controller(`, `cpg.command =`,
  `scripted`, `teacher`, `hand law`; only the files above match).
- **As teachers without new plumbing?** **[INFERRED from CODE]** No. (a) The quad law's
  outputs are rotor speeds, which the policy's action space cannot express (finding 2);
  (b) every hand law runs only by overriding `env.cpg.command` on the single-machine
  path (`rotor_authority/run.py:236`), and records no (obs, action) pairs; worker
  processes build their own envs; (c) `learning/distill.py` distils *stored linear
  own policies* (mode coefficients in each body's private basis) into a student from
  sampled observation states (`distill.py:86-104,213-260`), not demonstrations; it is
  an offline study wired only to `ops.run distill` (`ops/run.py:957`). The N6 (i) law
  is the closest template — it acts in twist space — but it flies nothing on archive
  bodies.

---

## 8. Open learner defects that bear on this (Q8)

From `docs/LEARNER_AUDIT.md` (written 2026-09-19 against `fcd6645`; its four fixes
are in): **GAP** no end-to-end toy task (`:408-416`); **BY DESIGN** sparse terminal
reward per segment and zero bootstrap (`:70-83,319`); `ent_coef` re-tuning flagged
(`:431-436`). Its checklist is software correctness; the defects below are newer or
outside it, each verified in code or data:

| # | defect | evidence | tag |
|---|---|---|---|
| D1 | crossing reward binary for the learner, graded for the archive; stale comment | `batchroll.py:1255-1264` vs `evaluate.py:185-194`, `transitions.py:214,713-726` | CODE |
| D2 | twist channel cannot reach rotors; gain cannot throttle them | N6 `m_frac` 0.0003; `cpg.py:427-430`; `triphibian.py:734-739` | MEASURED+CODE |
| D3 | air shaping saturated at launch altitude, blind to the turn; land shaping satisfied by the submerged ramp | `ppo.py:428-438`; `triphibian.py:1666,1360-1375` | CODE |
| D4 | return is ~all shaping in air and crossings | §5.3 | MEASURED |
| D5 | decoder re-identified per body, parameter side not reproducible, linear fit unverifiable when P > 24 | `rank_threshold/README.md:36-78`; `loop.py:2140` | MEASURED |
| D6 | land on the air basis; one basis per crossing for both media; mission switches basis by command, not by medium | `evaluate.py:54-58`; `batchroll.py:1324`; `mission.py:224` | CODE |
| D7 | no command channels in crossings or the continuous mission; no boundary distance/direction | `triphibian.py:1704-1711`; §2.2 | CODE |
| D8 | exploration: i.i.d. 40 ms Gaussian at std ~0.6 crashes gliders; `log_std` rises | air_turn_bar README:291-298; §4.3 | MEASURED |
| D9 | learning data are sampled rollouts, mostly off the scored (mean) behaviour | `batchroll.py:857-861`; D8 | INFERRED |
| D10 | own-policy ES accepts on `mission_fraction` (≈0) → own policies stay zero | ROADMAP:266-269; §3.2 | MEASURED |
| D11 | first half of a pair rewarded with the pair's score | `batchroll.py:1218-1229` | CODE |
| D12 | one fixed entry draw per crossing kind for every machine and generation | `transitions.py:223-235`; `batchroll.py:1313-1319` | CODE |
| D13 | device K-draw worlds not banked for PPO in the loop | `loop.py:1069-1075`; `loop.py:188-189` | CODE |
| D14 | stale size claims in prose (1,200 weights, 27 channels, 60 weights) | §3.1 | CODE |

---

## 9. INFERRED — the binding limits

Everything in this section is inference from §1-8; the evidence each step rests on
is cited there.

**9.1 A turn in the air.**
- *In principle, on a glider:* the command is observable (heading error), the
  reward is null-referenced, and authority exists but is weak: one gannet was turned
  12% of a perfect turn by a σ 0.02 linear own policy (B''). Whether the *shared*
  policy's channel — a twist at 0.5 of reach through a basis whose parameter side
  does not survive a reseed — can deliver a yaw/roll response on these bodies is
  **UNMEASURED** (ROADMAP:3851-3852 T3(b) remains open; step_response's dR² ≈ 0.002
  is the nearest read and is closed-loop).
- *In principle, on a rotor body:* no. The policy has no rotor-speed authority (D2);
  the gain cannot throttle; the only flown rotorcraft used a direct 250 Hz mixer.
- *In budget, as the search runs it:* no, by measurement: seven air-island runs
  (arch53-59; 7.0-8.3M decisions each, arch57 20.0M with its GRPO rows) produced no
  reader (arch56 stage 2: per-draw turn response at chance, ROADMAP:416-424). The binding items, in order: (1) the air
  reward is nonzero only on rare measurable draws and ≤0 for non-readers (D4); (2)
  exploration noise removes most gliders from the air before the command (D8), so
  the informative draws are fewer still; (3) 400 air decisions per body, one draw,
  on a body that never recurs (§4.4); (4) credit: with γλ = 0.94 the terminal δ
  reaches 50 decisions back at weight 0.046 and the first phase at < 0.003, so early
  actions depend on a critic that must predict a pair-coupled score from 33 channels
  (D11) — arithmetic from §4.1.
- *What it would cost to measure properly:* a fixed-glider learning curve with
  ≥1e6 air decisions (the repo's own reference range for locomotion is 1e6-1e7,
  `ppo.py:12-13`). At M4's measured ~75 s per 8-copy evaluation (≤12,800 decisions,
  README:28-31), the full-Tier-1 harness costs ≈1.6 h per 1e6 *total* decisions, but
  only 200 of each copy's 1,600 are air (unpaired; 400 of 1,800 paired), so ≈13 h
  (unpaired) or ≈7 h (paired) per 1e6 *air* decisions. An air-only harness would be
  cheaper per air decision; its cost is **UNMEASURED**.

**9.2 Hold-then-go at a boundary.** As configured, there is no learning signal:
the learner's crossing reward is binary and was positive on ≤0.16% of attempts per
run (D1); the shaping cannot create an expected gradient where the true reward is
zero (D4, theory); the hold→go cue is a one-hot flip with no heading, distance or
phase channel (D7); the decoder does not change at the boundary (D6); each kind
always starts from one entry state (D12). A reference crossing controller does not
exist (§7). The archive already computes a graded `approach` that the learner never
sees — the cheapest available signal.

**9.3 Mode switches between media.** Never trained. The observation carries enough
to know the medium (wetness, depth, contact, commanded one-hot), but in Tier-1 the
action decoder is fixed across each crossing, land uses the air decoder, the
continuous mission switches decoders by commanded leg rather than by actual medium,
and the continuous mission and Tier-2 bank nothing for PPO (`evaluate.py:590-604`,
`mission.py:178-306`). A single shared network would have to learn medium-specific
behaviour through a decoder that was identified in the other medium.

**9.4 Budget across three runs.** Three full runs of the current loop bank
≈22-25M decisions (§4.3) over ~10k different bodies, ≈2,000-2,500 decisions per body
across all tasks. Concentrating the same wall time on a few fixed bodies would give
1e6-1e7 decisions per body-skill (§9.1 arithmetic). So the learner is capable *in
budget* only if learning is separated from the morphology search (fixed-body or
fixed-set pretraining, the ROADMAP's N12 direction), and *in principle* only on
bodies whose relevant actuators the action space can reach — which excludes every
rotor body until the rotor channel is commanded outside the identified basis
(ROADMAP:3978-3980's own suggested direction).

**9.5 Ranking of the binding limits for the plan** (most to least binding):
reward/credit in air and crossings (D1, D3, D4) ≥ action authority (D2, D5, D6) >
data per body and non-stationarity (§4.4, distillation) > exploration (D8) >
observation gaps for crossings (D7). The first two are code and scoring choices,
cheap to change and testable on a fixed body before any search run.

---

## 10. How the telemetry numbers were made (reproducible, read-only)

Scripts in the session scratchpad, run as
`grep -E '"kind": ?"ppo"' runs/<run>/events.jsonl | python3 -I <script> <run>`
(streaming; no events file loaded whole):

- `ppo_summary.py` — updates, `transitions` sum, grad steps, stopped-early count,
  KL, `log_std` first/last, air/water/land `reward_by_tag` means (first and last 50
  updates), per-tag trajectory counts → §4.3, §4.4.
- `ppo_positive.py` — per generation and tag, `k = n·m²/(s²+m²)` positive
  trajectories → §4.4 crossing successes.
- `n10_share.py` — `return_by_tag` `abs_shaping/(abs_shaping+abs_terminal)` and
  `terminal_return` first/last 50 → §5.3.
- Own-policy weights: `json.load(runs/<run>/archive_air.json)`, count elites whose
  `meta["policy"]` has any nonzero entry → §3.2.
- Configs: line 1 of each `runs/<run>/generations.jsonl` (`config`).
- Cost: line 3 of `runs/arch59/generations.jsonl` (`cost.steps`).
- E1: `git show c3c3e1a:e1/status.json` (branch `origin/e3-cpu-runs`).
- Commit containment: `git merge-base --is-ancestor 79a2424 5fba1e5` → false.
