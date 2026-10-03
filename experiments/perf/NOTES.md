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
