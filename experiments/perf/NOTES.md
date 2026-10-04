# Where a worker's time goes, and what moves it (2026-09-27)

Live arch43 sample (60 s, /proc jiffies): 4 workers at 83-96% of one core each,
parent 0%, GPU 5% busy. Machine: i7-1280P laptop, 6 P-cores + 8 E-cores (20
threads), RTX 3060 Laptop. torch is the CPU build; numba/jax/warp not installed.

cProfile, one worker's shard (4 designs, identify on, 8 s segments, shared
policy), 78.7 s under the profiler (`profile_shard.py`):

| | s | share |
|---|---|---|
| identify_batch | 48.8 | 62% |
| BatchedFluid.apply (cum) | 50.4 | 64% |
| FullPipeline.step (GPU round trip, 338 us/call) | 13.8 | 18% |
| np.cross (+ moveaxis), 4 calls/step, vel6 | 8.4 | 11% |
| energy.step | 5.1 | 6% |
| cpg.command / clipped (np.clip) | 5.9 | 8% |
| clearance + ground_heights | 3.2 | 4% |
| ImplicitAeroDamping.projected | 4.6 | 6% |
| mj_step | 3.4 | 4% |

Constraint: never rebuild `mojo/build/*.so` while a run has them mapped.

## Plan
A. Host-side numpy: remove dispatch-bound calls, same arithmetic.
B. The GPU round trip: 338 us per step for a 4-machine shard at 5% GPU
   utilisation -- try the same kernels on the CPU, fused.
C. Re-sweep workers x shard after A/B: the optimum moves with per-step cost.

## Measured 2026-09-27

- Benchmark (`profile_shard.py --n 4 --identify 0 --noprof`): 18.4-20.7 s, and
  deterministic -- 700 segment/transition numbers bit-identical run to run
  (`--dump` / `--compare`).
- GPU contention is not the limit (`gpu_contention.py`): FullPipeline.step
  317 us/call alone, 339-347 us/call with 4 copies running beside the live
  run's 4 workers. Wall per copy 19.1 -> 23-25 s: that is CPU contention, and
  8 processes still give ~1.6x the throughput of 4. So the worker count is
  bounded by per-step *fixed* cost at small shards, not by the GPU.
- `DeviceContext(api="cpu")` exists but `enqueue_function` raises
  `Unimplemented` on it (Mojo in pixi, 2026-09-27): the GPU kernels cannot
  simply be pointed at the CPU.
- Bit-exactness: manual cross == np.cross, min(max()) == np.clip; `reduceat`
  sums are NOT bit-exact to slice sums (pairwise order), maxima are.

## What was done, each step bit-identical on the 700-number benchmark

| step | benchmark CPU (4 designs, no identification) |
|---|---|
| HEAD (committed code, old kernel), two runs | 17.8, 18.1 s |
| np.cross -> explicit; CPG clip -> min/max + clipped params cached | ~16 |
| one packed upload + one download, descriptor built once, scalars by pointer | ~14 |
| eleven per-panel kernels fused into one (`panel_kernel`, `*_at` bodies) | ~13 |
| `BatchedPower`: the energy model for the whole batch in one pass | ~12.2 |
| pinned host blocks + `launch`/`wait`, next step launched after `mj_step` | ~12.2 (see below) |

With identification on (the generation's main evaluation), alternating runs:
HEAD 45.4 / 53.7 s, now 32.9-34.0 s.

- GPU step for a 4-machine, 190-panel shard, back to back: 382 -> 93 -> 89 us.
  In place, with the early launch: `launch` 25 us, `wait` 3 us median
  (8993 of 9000 steps found their fluid already launched).
- CPU time equals wall time in every run: the CUDA wait is a busy spin, so
  GPU latency was CPU time, and on this power-limited laptop (96 C, 3.2 of
  4.7 GHz under load) CPU time is what the pool competes for.
- Pool sweep, 16 designs with identification, beside the live run's 4
  workers: HEAD 4x4 77.4 s, 8x2 72.8 s; new 4x4 74.6 s, 8x2 64.3 s, 6x3 67.7 s.
  Under full load the gains shrink: the package power cap, not latency, is
  the limit then. Re-sweep with the machine to itself before choosing.
- `mojo build -o` overwrites in place (same inode): `pixi run build-all` now
  builds to a temporary name and renames. `DYTISCIDAE_KERNEL_DIR` points a
  process at a staged build so it can be verified while a run holds the old one.
- `mojo/tests/test_full_gpu.py` had been stale since 2026-09-23 (hand-written
  descriptor); it goes through `BatchedFluid` now: 7 plans, 554 panels,
  xfrc 4.0e-16, added mass 3.5e-16 relative to `FluidSolver`.

## 2026-09-27, afternoon: pinned memory cost 1.35 GB a process, so it is gone

The first launch of arch44 stopped itself at generation 0: "resident set 8793
MB is over the 5000 MB ceiling". Mojo's first `enqueue_create_host_buffer`
reserves a ~1.35 GB pinned pool (RssShmem 8 MB -> 1350 MB for a 4 MB buffer;
a second buffer adds nothing), in every process: five processes held 6.75 GB of
unswappable memory, and 8x2 was killed by systemd-oomd within a minute.

The blocks are ordinary numpy memory again. The overlap survives: `launch`
enqueues the upload (a small pageable upload is staged and returns) and the
kernels; the download moved into the pipeline's `wait`. Measured in place:
`launch` 18 us, `wait` 16 us median, 8993 of 9000 steps launched early --
34 us per step against 382 us at HEAD. Per process after building the
pipeline: VmRSS 175 MB (1.52 GB with the pinned pool).

`reference_df95498.json` is the 700-number benchmark from the last commit
before any of this (its own kernel build); the current code matches it to the
bit. On the idle machine: 13.27 s CPU at df95498, 8.25 s now (1.61x).

**2026-10-04: `reference_df95498.json` is stale; use `reference_be3dbe0.json`.**
Since 09-27 the physics moved on purpose (ray entry under corrected added mass,
the 10-03 commanded crossings), and `profile_shard --compare` against df95498
now reports 408 of 700 numbers above 1e-9 on any current tree. A bit-identity
claim must be made against the change's own base. `reference_be3dbe0.json` is
`profile_shard --n 4 --identify 0 --noprof --dump` at `be3dbe0` (780 numbers),
and the rotor vectorisation matches it to the bit.

Pool, idle machine, 16 designs with identification: HEAD 4x4 55.9 s, new 4x4
42.6 s (1.31x). 8x2 not measurable here: with the pinned pool it was OOM-killed.

## 2026-10-01: steps per phase, and what a generation repeats

`phase_budget.py` attributes every `mj_step` and every second of wall time to
the phase it ran in, on the single-machine numpy path (no GPU in the container
that measured it). Step counts are path-independent: `identify_batch` runs the
same probes as `TriphibianEnv.identify`. Wall time is the numpy path's.

Four designs (`profile_shard.designs(4, 3)`), 8 s segments, identification on:

| phase | steps | share | wall s | share | us/step |
|---|---|---|---|---|---|
| identify | 115,200 | 70.1% | 79.0 | 65.4% | 686 |
| segment | 24,000 | 14.6% | 21.1 | 17.5% | 879 |
| transition | 24,000 | 14.6% | 19.8 | 16.4% | 824 |
| level_margin | 1,196 | 0.7% | 0.8 | 0.6% | 636 |

Identical for every design: identification is 2 media x 24 probes x 2 signs x
1.2 s = 115.2 s simulated (28,800 steps), against 24 s of segments and 24 s of
transitions. On the GPU path it was 53% of the main evaluation's wall time.

**The main evaluation is not the generation.** With `--shared-policy` and
`--refine-steps 2` (every stored run since arch42), `_refine_controllers`
re-scores every candidate once without exploration noise and then runs two
(1+1)-ES steps, each a full evaluation without identification. Per design per
generation:

| | steps | share |
|---|---|---|
| re-score + 2 refine steps | 36,897 | 47.3% |
| main: identify | 28,800 | 36.9% |
| main: segments | 6,000 | 7.7% |
| main: transitions | 6,000 | 7.7% |

Tier-1.5 (60 s, 15,000 steps on each promotion candidate) is not in this table.

Prediction, from 42.6 s main evaluation at 4x4 and 47% of it being the
no-identification part: the re-score and refinement add about 3 x 20 s = 60 s,
so a steady-state generation's evaluation is ~100 s, of which identification is
~23 s. Checkable against arch44's `generations.jsonl` (`elapsed` differences
after gen 5); not checked here, `runs/` is not in this container.

## 2026-10-01: the prediction against arch44, and what the telemetry cannot say

`gen_cost.py` reads a run's `generations.jsonl`. arch44: 600 generations,
60.6 h, 16 designs evaluated every generation.

| generations | median s/gen | min | max |
|---|---|---|---|
| 6-49 | 120 | 83 | 372 |
| 50-99 | 277 | 160 | 516 |
| 100-299 | 303-341 | 85 | 1082 |
| 300-349 | 441 | 142 | 1335 |
| 500-549 | 458 | 182 | 1472 |
| all, 6-599 | 322 | p10 165 | p90 595 |

The ~100 s predicted above holds for generations 6-49 (120 s) and fails for the
rest of the run: the cost climbs over generations 40-70 to ~300 s, without a
step, and stays 2.5-3.8x the prediction.

Excluded by the telemetry: the count of designs (16 every generation), Tier-0
rejections (median 0), promotions (median 0), the critic (never fitted by
gen 75), and the periodic duties -- median extra over plain generations: Tier-2
+32 s (n 39), audit +61 s (n 19), migration +74 s (n 9), checkpoint none.
Plain generations alone have the same median (322 s). Correlation of seconds
per generation with the generation-best's DoF 0.03, with total archive cells
0.24, with diverged rollouts 0.15.

Not recorded, so not decidable from this file: seconds per phase (main
evaluation, re-score, refinement, Tier-1.5, PPO update, archive work), the size
of the designs evaluated (panels, DoF, bodies), the share of rollout steps
actually stepped (a segment stops when its battery is flat or it diverges), and
load on the host from anything else. Each of those is a competing explanation
for the climb.

### Correction, same day: Tier-2 and the audit count island visits

The duty rows above are wrong. Tier-2 and the audit fire on
`visits % every == 0` for the generation's *island* (`SearchState.island_visits`,
first visit included), not on `gen % every`; the checkpoint's cost lands in the
next generation's difference. Re-read with that (`gen_cost.py`, fixed):

| | n | median s | extra over plain |
|---|---|---|---|
| plain | 523 | 312.6 | |
| Tier-2 | 36 | 698.3 | +385.7 |
| audit (always also a Tier-2 generation) | 15 | 661.7 | +349.1 |
| checkpoint | 29 | 364.6 | +52.0 |
| migration | 9 | 396.5 | +83.9 |

A Tier-2 generation costs about twice a plain one. It does not explain the
climb: plain generations alone still go from ~120 s to ~310 s.

### `cost` on every generation line

`GenerationCost` (`evolution/loop.py`) now writes, per generation:

- `seconds`: wall time per phase -- `build`, `evaluate` (with `evaluate.tier0`,
  `evaluate.main`, `evaluate.rescore`, `evaluate.refine` inside it), `place`
  (which holds Tier-1.5 and promotion refinement), `ppo`, `tier2`, `audit`,
  `judge_scout_critic`, `refit`, `migrate`, `report`, and `untimed`; and
  `checkpoint_prev`, the checkpoint written after the previous line.
- `steps`: `MissionResult.steps` summed per call and part
  (`main.identify`, `rescore.segments`, `refine.transitions`, `*.rig`, ...).
  Scheduled per design: identify 28,800; segments 3 x segment_seconds / 0.004;
  transitions 6,000; rig 299 per evaluation that measures `level_margin`.
  A shortfall is steps not taken because a battery went flat or a rollout
  diverged.
- `designs`: count, mean and max bodies and DoF of what reached Tier 1.

Checked on a CPU run (2 designs, 0.5 s segments): identify 57,600, segments
750, transitions 12,000, rig 598 -- each exactly the schedule; the phases sum
to the difference of `elapsed` with `untimed` 0.0.

## 2026-10-01: the re-score and refinement step one in one batch

`_refine_controllers` scored the noise-free re-score and then each (1+1)-ES
step as separate batched calls. Step one's trials are drawn from the incoming
weights and the refinement's own generator, and are compared with the
re-score only afterwards, so nothing in step one waits on the re-score.
`MERGE_FIRST_REFINE` scores both in one call of 2k machines: per generation,
one trip through the pool fewer (4 -> 3 with `--refine-steps 2`) and shards
twice as wide (4x4 pool: 4 machines a shard -> 8).

**What makes it exact, and what had to change for that.** A machine's score
must not depend on what shares its batch. The physics side is asserted by
`test_search`. The policy side was not true: `SharedPolicy.act_many` on the
mean gave a row different bits at batch 1 and 2 than at 3 or more
(`batch_invariance.py`, torch 2.14 CPU, ~3e-9; rows match across every batch
of 3 to 64). A shard whose other machines have stopped already hit this; a
merged batch would hit it differently. `act_many` now pads a deterministic
batch below `MEAN_MIN_ROWS = 4` rows. This changes scores against the code
before it only where a shard was left with one or two live machines.

**Checked here** (no GPU): `test_merging_the_rescore_with_the_first_refinement_changes_nothing`
-- with an evaluator that is a pure function of each row, merged and split
keep the same weights and return the same results in one call fewer;
`test_a_decision_on_the_mean_does_not_depend_on_its_batch` (batches 1-16);
mutations `merged-rescore-halves-swapped` and `mean-not-padded`.

**Not checked here, and it decides whether this stays:** that the batched GPU
path gives a machine the same bits in a shard of 8 as in a shard of 4, on the
run's machine and its BLAS. `merge_check.py` does exactly that comparison and
prints both wall times:

    python experiments/perf/merge_check.py --n 16 --workers 4 --min-shard 4

Exit 0 means every number agreed. If it does not, set `MERGE_FIRST_REFINE =
False` and the search is as it was.

**With AK's refinement funnel (merged 2026-10-04).** `pass-1003` built a funnel
(`controller_refine_funnel`, `select`) that chooses which candidates to refine
*from the re-scored results*. Step one's trials therefore cannot be drawn before
the re-score when a funnel is on, and drawing every candidate's trial ahead of
the choice would spend what the funnel exists to save and move the generator's
draws. So the merge runs only without a funnel (`select is None`); with one, the
split order runs, draw for draw as before. Both are held by the same test (5
candidates, 3 steps, funnel picking 3: calls `[5, 3, 3, 3]` either way). In the
`stages` event, `merged_step_one` says which ran; when true, `rescore` holds
step one's wall and `refine_steps[0]` is 0.0.

## 2026-10-01: what the CPU<->GPU traffic costs now, and the host work beside it

Per batched step after 09-27: one packed upload (`nb*21 + nm*3` doubles) and
one download (`nb*8 + n*17` doubles; at 190 panels about 26 KB), 34 us of host
time (`launch` 18, `wait` 16). Against the benchmark's 8.25 s CPU over 12,000
batched steps (4 designs, no identification: 3 x 2,000 + 4 x 1,500), that is
34 of ~690 us, about 5%. It is latency, not bytes: at that size the copy
itself is a few microseconds, so trimming the download (`alpha` is never read
on this path, and `q/lift/drag/d_bluff/pos_w/force/buoy` only on the 1-in-4
damping and inflow steps) would save little and needs a kernel rebuild.

What does move the traffic's cost is **batch width**: one round trip per
physics step whatever the batch, so its cost per machine falls as 1/k. The
merged re-score (k 4 -> 8) does that for one call a generation; running
identification's probes side by side would do it for 36.9% of a generation's
steps, which is also where the wait is least hidden.

**Done here: the unread diagnostics.** `BatchedFluid.finish` refreshed all
eight `FluidDiagnostics` fields per machine per step from the downloaded
arrays. Two are read on the batched path (`mean_submerged`, `slam`) and
`added_mass` is cheap; the other six (`max_submerged`, `max_alpha`,
`max_dynamic_pressure`, `lift`, `drag`, `buoyancy`) are read nowhere outside
`physics/fluid.py` and tests of the single-machine `FluidSolver`.
`diag_cost.py` (this container's CPU, 4 machines x 48 panels): the six cost
40.0 us a batched step, the two that are read 30.6 us -- the unread ones
cost more than the device round trip. They are no longer computed; `reset_slam`
sets them to NaN at each rollout's start, so a later reader on this path gets a
value that cannot pass for a measurement. Nothing that is read changes, so
every score is unchanged by construction; not run on the GPU path here.

## 2026-10-03: what a rotor costs, and the rotor step vectorised (`rotor_cost.py`)

**Batch.** Four arch46 elites with 4, 22, 5 and 21 rotors (`rotor_cost.rotor_designs`,
seed 1, >= 4 rotors), batched path, shared policy attached, 6 s segments.

**Where a rotor's wall went** (`--count`, be3dbe0): with identification, 402.6 s wall,
of which `RotorSet.apply` 331.3 s = 82%. 1,547,286 rotor-steps, 93.2% of them
spinning, 2.00 `rotor_forces` calls per rotor-step, **198 us per rotor-step** (table
builds excluded); 10 table builds (`bemt` over the 510-point grid) took 25.2 s,
~2.5 s per (rotor spec, medium). `--micro` on the same machines: 140-192 us.

**Reconciling that with "+1.6 s per rotor".** The regression slope is fitted on the
`wall` of the placed result (`cost_model.py`), which with a shared policy is the
noise-free *re-score*: no identification, 3 x 1500 + 4 x 1500 = 10,500 steps at 6 s
segments. 10,500 x 198 us = 2.1 s per rotor, against a slope of 1.6-1.8 (shard-mates'
rotors are noise in that fit, and flat batteries end some rollouts early). The 305 us
micro-benchmark was the same cost on a loaded machine. Idle rotors (6.8% of
rotor-steps) are cheap, but that is not where the gap was. A generation pays main
(~33.6k steps with identification) + re-score + refine steps: ~10.8 s of worker time
per rotor at `--refine-steps 2`. Check against arch46: +3 rotors/design x 16 designs
x 10.8 s / 4 workers = +130 s/gen predicted, +116 s observed (114 -> 230 s/gen).

**What changed.**
- `RotorBatch` (`physics/rotor.py`): every rotor of a machine -- of every machine in
  a shard, on the batched path -- in one array computation: stacked spin axes, hub
  velocities as `BatchedFluid.launch` forms them, medium queries in one call (one
  call per rotor on waves, whose dot product's BLAS kernel depends on the row
  count), `rotor_forces_many` for both the start- and end-of-step speeds, the
  backward-Euler end-of-step speed per row, fancy-indexed writes. `RotorSet.apply`
  is a batch of one; `BatchedFluid.finish` runs one batch per step after its
  per-machine loop. The per-rotor loop stays as `RotorSet.apply_per_rotor`, the
  reference, called by no path.
- `bemt_many`: `rotor_table`'s 510-point grid in one pass, 2.7-3.1 s -> 0.10-0.16 s
  per (spec, medium).
- Bit-identity needed three choices, each probed first: a stacked `matmul` for the
  axis (the elementwise expansion misses BLAS's FMA in 19,986 of 20,000 rows);
  `np.power(om, array of 2.0)` for `om**2` (Python's `pow` and numpy's `x*x` differ
  in 19 of 20,000); row sums `(a*b).sum(1)` for the 3-vector dots (equal to `ddot` in
  20,000 of 20,000).

**Noise floor** (`--floor`, before any change; two arch46 elites with 3 rotors, 2 s
segments, identification off, `_nudged` 1e-15 dither, max of 3 seeds): 1.21e-9 and
1.23e-12, so the bar max(1e-5, 2 x floor) = 1e-5; batched against single 2.2e-9.
Not needed in the end: every change below is bit-identical.

**Agreement after the change** (`--dump` before, `--compare` after):

| batch | numbers | not bit-identical |
|---|---|---|
| rotor-heavy, identification on | 747 | 0 |
| rotor-heavy, identification off | 750 | 0 |
| `profile_shard --n 4 --identify 0` (random designs) | 780 | 0 |

and `test_the_rotor_batch_is_the_per_rotor_loop`: 32 cases (calm and waves, rotors
in air / water / across the surface, idle and reversed, one machine and two), 0
arrays differ; tables and 40 off-grid `bemt` points identical.

**Timing**, alternating processes (be3dbe0 / c35c402), 3 repeats, an R-sweep search
and other agents on the machine, so ratios:

| | old | new | new/old per repeat |
|---|---|---|---|
| `apply`, us per rotor-step (`--micro`, air) | 192.3 / 164.2 / 140.1 | 23.1 / 20.6 / 18.8 | 0.120 / 0.125 / 0.134 |
| rotor-heavy evaluation, no identification, s | 125.8 / 111.7 / 94.5 | 24.0 / 20.9 / 19.9 | 0.191 / 0.187 / 0.211 |
| rotor-heavy evaluation, identification, s | 366.2 / 348.9 / 271.6 | 71.5 / 62.7 / 55.4 | 0.195 / 0.180 / 0.204 |
| random designs (`profile_shard`), s | 11.71 / 16.26 / 15.12 | 12.72 / 13.83 / 15.15 | 1.09 / 0.85 / 1.00 |

Not measured here: a generation. Predicted from the per-rotor numbers, late-run
generations lose most of the ~+116 s rotors added; the arch47 `stages` events are
that read.

## 2026-10-03: AM, the host loops beside the rotors (`am_ab.py`)

Profile after the rotor change (`profile_shard --n 4 --identify 0`, cProfile,
cumulative of 23.8 s): `clearance` 4.36 s (18%), read up to four times a step per
machine on a state that has not moved (the segment record, the crossing tracker
twice, the transition peak).

**Done.** `clearance` is remembered, keyed on the clock and the geoms' and root's
pose bytes, so a hit is the computation's own bits; `clearance_many` fills it for
a batch in one array pass (machines on waves, or with no colliding geoms, alone).
`rollout_batch` reads `body_twist` once per step instead of twice. Same profile:
`clearance`+`clearance_many` 4.36 -> 1.77 s. In-process A/B (`am_ab.py`, on/off
alternating, results asserted identical): random designs on/off 0.870 / 0.900 /
0.881 / 1.079, mean 0.933; rotor-heavy 1.135 / 0.547 / 0.882 -- inside the load
noise, not a measurement. Tests: `test_the_clearance_of_a_batch_is_each_machines_own`
(90 reads, 0 differ, waves included; a memo does not survive a 0.7 m move at the
same clock).

**Not done, and why.**
- Damping projection (`ImplicitAeroDamping.projected`): 48 us (8 bodies, 12 DOF) to
  93 us (45 bodies, 49 DOF) per call, called on one step in four: 12-23 us per
  machine-step, 7% of the profile. Batching it across machines means a
  block-diagonal (bodies x DOF) product whose BLAS summation order differs from the
  per-machine one, so it would leave bit-identity for <= 7%.
- Inflow (`InducedFlow.update` + `machine_flow`, one step in four) ~7%, CPG command
  1.6%, entrainment reaction 3.4%: each interleaved with the per-machine limiter,
  each a few percent at most.
- `mj_step`: heterogeneous models, cannot be batched.

**Verification, 2026-10-04** (the tree of this commit): `test_physics.py` whole,
`[fail` lines 0, "all physics checks passed"; `test_search` single functions
`test_the_two_evaluation_paths_score_the_same_machine_the_same`,
`test_the_early_fluid_launch_changes_nothing` (378 numbers, moved []),
`test_sharding_a_generation_does_not_change_a_score`, 0 failures each; the seven
cheap suites pass. Mutations, each `caught 1/1`: `rotor-batch-one-table` (192
arrays differ), `rotor-table-one-sum`, `clearance-batch-first-slice` (36 of 90
reads differ), `clearance-memo-keyed-on-time`, and the two older rotor mutations
retargeted at the live code, `propellers-one-handed` and
`rotor-thrust-at-start-of-step`.
