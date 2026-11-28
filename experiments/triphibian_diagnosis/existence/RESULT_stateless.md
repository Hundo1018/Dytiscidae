# E0s result: the hand law as a stateless outer law over a stabilising inner loop

2026-11-11. Spec: `PREREG_stateless.md` (commit 0c1ebad, not edited). Body L1.
Code: `stateless.py` (sha256 `207322019876…`, frozen before the scored runs;
`triquad.py` 1cc5f4bd…, `landgait.py` 84336383…, `nulls.py` 72985d4a… unchanged).
No training, no optimiser, no fitting. Hand tuning on seeds 100-107 only
(`trials_stateless.jsonl`); scored seeds 0-31 (missions) and 0-7 (segments, entry
states), each run once, 688 rows in `results_stateless.jsonl`, 0 duplicates,
nothing dropped. MuJoCo 3.15, CPU, numpy fluid path.

Summary: the law survives the split. Missions 32, 32, 31 of 32 (stateful 32, 32,
32), all three segment bars cleared on 8 of 8, three crossing kinds 8 of 8. One
row is a clean negative: `air_to_water` under the gates crosses 6 of 8 against the
stateful law's 8 (bar 7). Predictions 1, 2 held; 3 refuted on one kind; 4 held
(K = 1); 5 refuted on one count (1 transition in 192 by the constant arm);
6 refuted (deciding every step is not better anywhere, and is worse on several rows).

## 1 Channels, K, and what became of each latch

`o` has 46 channels, `u` has 10, K = 1.

| `o` index | channel | sensor |
|---|---|---|
| 0-32 | `env.observation(commanded)` unchanged: velocity /5 (3), rates /4 (3), gravity in body (3), tanh(depth/5), mean wetness, command one-hot (3), tanh((depth - commanded depth)/3), contact, battery, stroke (2), task channels (6: cruise, hold, speed ratio, cos and sin heading error, tanh(height lost/3)), morphology (8) | as the learner gets them |
| 33-34 | heading sin, cos | compass |
| 35 | clearance, m | altimeter / sonar |
| 36 | signed distance to the waterline along the shore normal, m | position fix reduced to one number |
| 37-38 | shore-normal bearing in the yaw frame, sin, cos | map + compass |
| 39-42 | wetness of each rotor hub | wet switches |
| 43 | seconds since the commanded medium changed (clipped 10) | clock restarted by the command |
| 44 | air-task height error, m (0 off an air task) | the task's reference altitude |
| 45 | memory 0: this law's memory output at the previous decision | fed back by the harness |

| `u` index | meaning |
|---|---|
| 0-2 | commanded acceleration, world frame |
| 3 | tilt limit, degrees |
| 4 | allow downward thrust (under water) |
| 5 | rotors on |
| 6, 7 | limb drive, limb turn |
| 8 | limb pose (rest to stand) when not walking |
| 9 | memory 0 out |

The inner loop (`Inner.act`, every simulation step) is `triquad.Hand._cascade` and
`_rotor_cmd` unchanged (attitude PD, torque-first mixer, medium-aware rotor command)
plus the `landgait` gait generator driven by the simulation clock phase. It reads
only `u` and instantaneous body state, and holds no latch.

| E0 latch | fate |
|---|---|
| `yaw0` | dropped: the body has no yaw authority, the latch never acted |
| `z0` (air height reference) | channel 44 (air task) or the clearance band from channel 35 |
| `w_hold`, `nav_exit_xy`, `nav_to_xy`, `nav_w_xy`, `nav_land_xy` (held positions) | gone: velocity damping, and targets expressed as distance to the waterline (channel 36) |
| `w_p0`, `w_t0` (water cruise position ramp) | gone: a velocity command with the command over-asked 3x (`cruise_ff`, tuned on seeds 100-107: 2.0 and 3.0 both 8 of 8, 3.0 kept) |
| `landed`, `air_up`, `cut` | tests on `o` (contact, clearance, wetness) at every decision |
| `diving` | one threshold on the depth error, no hysteresis |
| `nav_v`, `nav_z` | continuous functions of speed and clearance |
| `g_p0`, gait `t0` | clock phase in the inner loop |
| `ph_idx`, `nav_M` | read from the command and task channels |
| `nav_go` (water exit: staged, then go) | **memory channel 0**, the only state the outer law keeps |

## 2 Statelessness tests

`logs/stateless_tests.log` (logs are gitignored; pasted in full):

```
TEST c  Teacher: attributes assigned after construction: [] -> PASS
TEST c  Inner: attributes assigned after construction: [] -> PASS
TEST a fresh Teacher at every decision, mission from air, seed 0: 12 fields, 0 differ []; completed True, held [True, True, True], decisions 884, energy 3.100177774418303 / 3.100177774418303, end x 9.642694916946702 / 9.642694916946702 -> PASS
TEST b fresh Inner at every step, mission from air, seed 0: 12 fields, 0 differ []; completed True, held [True, True, True], decisions 884, energy 3.100177774418303 / 3.100177774418303, end x 9.642694916946702 / 9.642694916946702 -> PASS
TEST a fresh Teacher at every decision, mission from water, seed 0: 12 fields, 0 differ []; completed True, held [True, True, True], decisions 735, energy 2.5516243388351025 / 2.5516243388351025, end x -1.9843364292263554 / -1.9843364292263554 -> PASS
TEST b fresh Inner at every step, mission from water, seed 0: 12 fields, 0 differ []; completed True, held [True, True, True], decisions 735, energy 2.5516243388351025 / 2.5516243388351025, end x -1.9843364292263554 / -1.9843364292263554 -> PASS
TEST a fresh Teacher at every decision, mission from land, seed 0: 12 fields, 0 differ []; completed True, held [True, True, True], decisions 702, energy 2.1101230067243892 / 2.1101230067243892, end x -1.9999134624362098 / -1.9999134624362098 -> PASS
TEST b fresh Inner at every step, mission from land, seed 0: 12 fields, 0 differ []; completed True, held [True, True, True], decisions 702, energy 2.1101230067243892 / 2.1101230067243892, end x -1.9999134624362098 / -1.9999134624362098 -> PASS
```

Test c is an AST listing of `self.<name> = ...` outside `__init__`. `Inner` writes
into a scratch array `self.M` in place (allocated in `__init__`, fully overwritten
each step); test b shows a fresh one gives the same rollout.

## 3 Predictions 1-3 beside the stateful law

Missions (`mission.evaluate_mission`, legs 7-13 s, seeds 0-31), completed of 32,
transitions made / commanded. Stateful = L1 `teacher` in `results/mission_eval.jsonl`.

| start | stateful, every step | stateless 25 Hz | transitions | bad_qacc rows |
|---|---|---|---|---|
| air | 32 | 32 | 64 / 64 | 0 |
| water | 32 | 32 | 64 / 64 | 0 |
| land | 32 | 31 | 63 / 64 | 0 |

The one miss: land start, seed 20, the first (land) leg not held, share 0.888,
support 0.92; no failure, no reset.

Segments, seeds 0-7 (stateful = L1 `walk` in `results_land.jsonl`, same seeds):

| medium | bar | stateless 25 Hz | pass | stateful |
|---|---|---|---|---|
| air | >= 0.5 on >= 7 | 0.912 0.902 0.912 0.909 0.922 0.924 0.913 0.913 | 8 / 8 | 0.908 0.898 0.910 0.892 0.921 0.920 0.899 0.908 |
| water | >= 0.15 on >= 6 | 0.250 0.735 0.379 0.656 0.528 0.249 0.375 0.248 | 8 / 8 | 0.209 0.747 0.440 0.436 0.394 0.196 0.337 0.189 |
| land | >= 0.15, support >= 0.9, on >= 7 | 0.943 0.955 0.897 0.888 0.897 0.954 0.900 0.929 (support 0.999-1.000) | 8 / 8 | 0.948 0.914 0.845 0.829 0.857 0.941 0.914 0.918 |

No failure and no `bad_qacc` on any segment row.

Crossings from entry states 0-7, `DYTISCIDAE_CROSSING_GATES=1` (recorded in each row):

| kind | stateful | bar | stateless 25 Hz |
|---|---|---|---|
| air_to_water | 8 | 7 | **6** |
| water_to_air | 8 | 7 | 8 |
| water_to_land | 5 | 4 | 8 |
| land_to_air | 8 | 7 | 8 |

`air_to_water` entries 3 and 4 fail with `did not hold before the command to
cross` (hold 0.406 and 0.371; the body does reach the water afterwards, exit depth
0.73-0.76 m). An air hold is scored on energy height; the stateless law holds air
with velocity damping and the clearance channel, without the stateful law's
latched `z0`. That this is the cause is a reading of the code, not a measurement.
Entry states 100-107 gave 8 of 8 during tuning, so the row was not seen failing
before scoring and was not re-tuned after.

## 4 Nulls (prediction 5) and the zero-memory arm

Same inner loop, mission verdict, 32 seeds per start.

| arm | air | water | land | transitions made / commanded | rows with failure or bad_qacc |
|---|---|---|---|---|---|
| `u = 0` | 0 | 0 | 0 | 0 / 192 | 0 |
| constant `u` per seed (acc +-8, tilt <= 45) | 0 | 0 | 0 | **1** / 192 | 19 (16 diverged, 2 battery, 1 unstable) |
| teacher, command frozen at first value | 0 | 0 | 0 | 0 / 192 | 0 |
| teacher, memory input forced to 0 (reported) | 26 | 22 | 31 | 161 / 192 | 8 (6 battery, 2 unstable) |

The constant arm's one transition: air start, seed 27, legs air, water, land; it
held air then water (shares 1.0, 1.0) and did not hold land (0.0). Its failed
rollouts are counted failed, not 0.

Zero memory: 17 missions lost. 16 of them fail on the leg entered by
`water -> land`; the other is the land-start first-leg miss of the kind the full
law also has. No `water -> air` leg is lost. So memory carries the staged water
exit toward land and nothing else, and even there 79 of 96 missions complete
without it.

## 5 Prediction 6: the outer law every step against 25 Hz

| row | 25 Hz | every step |
|---|---|---|
| mission air / water / land | 32 / 32 / 31 | 32 / 31 / 32 |
| segment air, mean of 8 | 0.913 | 0.910 (>= 25 Hz on 0 of 8 seeds) |
| segment water, mean of 8 | 0.428 | 0.401 (2 of 8) |
| segment land, mean of 8 | 0.920 | 0.902 (2 of 8) |
| air_to_water | 6 | 5 |
| water_to_air, water_to_land, land_to_air | 8, 8, 8 | 8, 8, 8 |

Median decisions per mission: 744 at 25 Hz, 7,435.5 every step. The every-step
miss: water start, seed 14, the land leg (share 0.896). The gains were tuned at
25 Hz only, which may be the whole of the difference; it was not separated.

## 6 The six predictions

1. "Mission verdict (`mission.evaluate_mission`, legs 7-13 s), seeds 0-31, each
   start: 25 or more of 32. I hold this at about 0.6. The every-step, stateful law
   scores 32, 32, 32 on L1." **Held**: 32, 32, 31.
2. "Segments through `evaluate.run_segment`, seeds 0-7: air >= 0.5 on >= 7 of 8;
   water >= 0.15 on >= 6 of 8; land >= 0.15 with support share >= 0.9 on >= 7 of
   8. I expect water to lose the most, because its position reference goes."
   **Held**: 8, 8, 8 of 8. The expectation in its second sentence did not happen:
   water is above the stateful law on 6 of 8 seeds.
3. "Crossings from eight entry states with `DYTISCIDAE_CROSSING_GATES=1`: at least
   as many as the stateful law on L1 less one per kind (it has 8, 8, 5, 8 for
   `air_to_water`, `water_to_air`, `water_to_land`, `land_to_air`)."
   **Refuted on one kind**: 6, 8, 8, 8 against bars 7, 7, 4, 7.
4. "`K` <= 6, and the water exit is what needs it." **Held**: K = 1; the only
   legs lost without it are water-to-land exits (16 of 16 attributable losses).
   Narrower than written: the exit to air does not need it.
5. "Nulls through the identical inner loop, mission verdict, 32 seeds per start:
   `u = 0` completes 0 and makes 0 transitions; so does a constant `u` drawn once
   per seed from the range the teacher uses; so does the teacher with its command
   channels frozen at the first value seen. The teacher with its memory inputs
   forced to zero is reported, not predicted: it says whether the memory carries
   the mission." **Refuted on one count**: all three complete 0 of 96; `u = 0`
   and frozen make 0 transitions; the constant arm makes 1 of 192.
6. "The 25 Hz hold costs something measurable: the outer law run every step scores
   at least as well on every row." **Refuted**: every step is lower on the water
   mission row, on all three segment means and on `air_to_water`.

## 7 Reproducers

Prefix each with `cd /home/hundo/Projects/Dytiscidae/dytiscidae && MUJOCO_GL=disable PYTHONPATH=.`
and give `--out` an absolute path (default is the tuning file).

```bash
.venv/bin/python experiments/triphibian_diagnosis/existence/stateless.py tests
.venv/bin/python experiments/triphibian_diagnosis/existence/stateless.py run --kind mission --arm teach,step,zero,const,frozen,nomem --seeds 0-31 --shard 0/4 --out $PWD/out.jsonl
.venv/bin/python experiments/triphibian_diagnosis/existence/stateless.py run --kind segment --arm teach,step --seeds 0-7 --out $PWD/out.jsonl
DYTISCIDAE_CROSSING_GATES=1 .venv/bin/python experiments/triphibian_diagnosis/existence/stateless.py run --kind crossing --arm teach,step --seeds 0-7 --out $PWD/out.jsonl
```

Missions took about 27 minutes on 4 processes (576 missions), segments and
crossings about 3 more.

## 8 Not checked, and departures to know about

- Why `air_to_water` loses two holds was not traced; the cause given in §3 is a hypothesis.
- Each scored row is one run per seed; no repeat for run-to-run variation (the
  two test arms reproduce bit for bit, so the path is deterministic in one process).
- The every-step arm uses gains tuned at 25 Hz; prediction 6 is not a fair test
  of the hold's cost in the other direction.
- The stateful segment and crossing rows are the stored ones (`results_land.jsonl`,
  `results/crossing_telemetry.jsonl`), not re-run here.
- `env.observation` velocity and rates are in the root body's inertial frame at
  its centre of mass, not the body frame. The outer law converts with two body
  constants (`R_i`, `ipos`) fixed at construction. A learner reading the same
  channels would have to absorb that rotation.
- Channels 35, 36, 43, 44 are in raw metres or seconds, not squashed.
- The inner loop chooses the water mixer from the rotor wetness it reads itself;
  this is the stateful law's own test, moved, not new logic. The gait phase comes
  from the simulation clock.
- The constant null draws from acc +-8 m/s^2 and tilt up to 45 degrees; other
  ranges were not tried.
- The statelessness tests ran on seed 0 only, one mission per start, after the
  code was frozen.
- `logs/stateless_dbg.py` is a trace helper used while tuning; `logs/*.log` are
  gitignored and stay local.
- Only body L1. Nothing here says a network can represent or learn `teach`.
