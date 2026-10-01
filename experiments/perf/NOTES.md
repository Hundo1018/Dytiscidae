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
