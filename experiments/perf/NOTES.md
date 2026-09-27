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
