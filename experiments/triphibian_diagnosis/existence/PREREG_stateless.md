# E0s — can the hand law be split into a stabilising inner loop and a stateless outer law that reads only an observation?

Frozen 2026-11-11 before any rollout, commit 3ce2542 (branch
`triphibian-diagnosis`), MuJoCo 3.15. No learning, no search, no optimiser,
no fitting of anything. Written by the lead; the agent that runs it may not
edit this file. This is P0.5 and P0.6 of `DIAGNOSIS.md` as a prototype under
`experiments/`: nothing under `dytiscidae/` changes until it has passed.

## 0 Question and decision

Run 1 is imitation: a feed-forward policy at 25 Hz learns the outer law, and
acts through a hand-written inner loop. That is only sound if the thing being
imitated is itself a function of what the policy will see, acting through
what the policy will act through. E0's `Hand` is neither: it reads the
simulator's state directly, runs every simulation step, and keeps 15 latches
and timers.

Can E0's law, on body L1, be rewritten as `u = teach(o)` with no state of its
own, acting through an inner loop that only stabilises and allocates, and
still pass the mission verdict, the segments and the crossings?

Decision, fixed now. Passes: the split is ported into `dytiscidae/` (inner
loop, observation channels behind a flag) and it is the teacher of run 1; its
measured rates are the student's ceiling. Fails: no run starts; the report
names which latch or which reading could not be removed, and that is the next
design question.

## 1 Rivals

- **It splits** (the law's state is incidental).
- **Memory is essential**: the water exit (stage, then go, never back) or the
  landing cannot be decided from one observation, with or without the memory
  channels below.
- **Rate is essential**: an outer law at 25 Hz with its output held between
  decisions loses the exit or the hover that the every-step law kept.
- **Position is essential**: without a held position reference the antipodal
  scores (signed speed along a commanded heading) or the station gate fail.
- **Instrument**: the prototype's observation leaks state the real
  observation would not carry (checked by the statelessness tests in §4).

## 2 The interface under test

**Observation `o`**, built by the prototype from the environment, every
channel listed in the result with the sensor it stands for:

- everything `TriphibianEnv.observation(commanded)` returns today (body
  twist, gravity in the body frame, depth, wetness, command one-hot, depth
  error, ground contact, battery, stroke, `task_channels`, morphology);
- added, by the user's decision 3 of 2026-11-11: heading (sin, cos); height
  of the lowest point over what is underneath (`clearance`); signed distance
  from the root to the waterline along the shore normal, and the bearing of
  that normal in the body's yaw frame (sin, cos);
- added, by DIAGNOSIS P0.6: each rotor's wetness; seconds since the commanded
  medium last changed (clipped); the height error to the air task's
  reference altitude while an air task is in force (the task's own
  reference, as the depth error is in water);
- **memory**: `K` channels that are the memory outputs of the previous
  decision, zero at the first decision of a rollout. `K` is as small as
  works and is reported. This is the only state there is, it is in `o`, and a
  feed-forward student sees it.

Not in `o`: absolute x, y or z; anything else read from the simulator.

**Action `u`**: a world-frame acceleration demand (3), a tilt limit, a flag
allowing thrust below the horizon, a rotors-on flag, limb drive, limb turn,
limb pose, and the `K` memory outputs. `u = 0` commands every rotor to 0 and
the limbs to rest.

**Inner loop**: weight and buoyancy compensation, thrust direction, the
attitude PD, the mixer, the medium-aware rotor law (E0's `_cascade` and
`_rotor_cmd`), and the limb gait generator. Runs every simulation step. It may
read attitude, body rates, velocity, each rotor's wetness, the submerged
fraction and constants of the model. It may not read position, the command,
the task or the clock (the gait's phase excepted), and it holds no latch.

**Outer law** `teach(o) -> u`: a pure function, evaluated at 25 Hz (the
environment's control period), its output held between decisions.

## 3 Predictions (written before running)

1. Mission verdict (`mission.evaluate_mission`, legs 7-13 s), seeds 0-31,
   each start: 25 or more of 32. I hold this at about 0.6. The every-step,
   stateful law scores 32, 32, 32 on L1.
2. Segments through `evaluate.run_segment`, seeds 0-7: air >= 0.5 on >= 7 of
   8; water >= 0.15 on >= 6 of 8; land >= 0.15 with support share >= 0.9 on
   >= 7 of 8. I expect water to lose the most, because its position
   reference goes.
3. Crossings from eight entry states with `DYTISCIDAE_CROSSING_GATES=1`: at
   least as many as the stateful law on L1 less one per kind (it has 8, 8, 5,
   8 for `air_to_water`, `water_to_air`, `water_to_land`, `land_to_air`).
4. `K` <= 6, and the water exit is what needs it.
5. Nulls through the identical inner loop, mission verdict, 32 seeds per
   start: `u = 0` completes 0 and makes 0 transitions; so does a constant
   `u` drawn once per seed from the range the teacher uses; so does the
   teacher with its command channels frozen at the first value seen. The
   teacher with its memory inputs forced to zero is reported, not predicted:
   it says whether the memory carries the mission.
6. The 25 Hz hold costs something measurable: the outer law run every step
   scores at least as well on every row.

## 4 Design

- Statelessness is tested, not asserted: (a) `teach` is called on a fresh
  instance at every decision in at least one full mission per start and the
  rollout is identical to the reused instance's, field for field; (b) the
  inner loop likewise; (c) a grep-level listing of every attribute either
  object assigns after construction, which must be empty.
- Scores and the verdict are the repo's, unmodified. Nothing under
  `dytiscidae/`, `tests/`, `tools/`, `docs/` is edited.
- Tuning by hand on seeds 100-131 only; scored seeds 0-31; every tuning
  rollout appended to `trials_stateless.jsonl`. No optimiser, no fitting.
- A rollout that ends in `bad_qacc`, divergence or a flat pack is failed,
  never 0, and is counted against the teacher in prediction 1.
- Controls: the stateful every-step law (`landgait` walk arm) on the same
  seeds, from `results/mission_eval.jsonl`; the nulls of prediction 5.
- Stopping rule: stop when the rows are filled, or when a row is shown
  blocked with its trace, or after about four hours of work. A clean
  negative is a result; do not rescue a row by widening `o` beyond §2, by
  putting logic into the inner loop, or by loosening a gate.
