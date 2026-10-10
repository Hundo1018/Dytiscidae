# Dytiscidae — project operating notes

Generative design + control search for triphibian flapping-wing machines.
MAP-Elites over eight islands, MuJoCo rigid bodies plus this project's own
quasi-steady fluid solver (`docs/model_validity.md`: not CFD), and a shared PPO
policy that is the controller inside every evaluation (it proposes no
variation; `batchroll.py:839`, ROADMAP 2026-10-10 §1).

`docs/ROADMAP.md` is the work list and carries the measurement behind every
item. Read it before proposing anything.

## Session start

1. `git status` and `git log --oneline -5`.
2. Read the dated paragraphs at the top of `docs/ROADMAP.md`. They name the
   current work list (newest first; as of 2026-10-08 it is §"2026-10-08, later —
   the work-list sweep", which ends with the open questions for the user) and
   the comparability boundaries. Before proposing any threshold, read §"What is
   set by measurement, and what is typed".
3. The latest run's configuration and pre-registered reads are in
   `runs/<run>_notes.md` (arch50 is running as of 2026-10-08, launched from the
   worktree `../arch50-tree` at 2b7bb53: `runs/arch50_notes.md`; the latest
   finished run is arch49, read in the sweep section). `runs/` is gitignored:
   in a fresh container it does not exist, see `docs/PORTING.md`.
4. Run the canary (below).

## Tests

There is no pytest. Every suite is a script with a `main()`.

**Canary**, ~15 min plus `test_search.py`:

```bash
PYTHONPATH=. .venv/bin/python tests/test_ppo.py 2>&1 | tail -2
PYTHONPATH=. .venv/bin/python tests/test_physics.py 2>&1 | grep -ciE '\[fail'
PYTHONPATH=. .venv/bin/python tests/test_search.py 2>&1 | tail -2
```

Expect `shared PPO checks passed`, `0`, `all search-machinery checks passed`.
`test_search.py` takes ~25 min: run it detached (`setsid nohup … &`), never in a
tool call that can time out and kill it.

**Cheap suites**, no third-party package, under two minutes together. Run them
on every change:

```bash
for s in index architecture domain application search_adapter adapters worker; do
    PYTHONPATH=. .venv/bin/python tests/test_$s.py | tail -1
done
```

`test_architecture.py` fails the build if anything in `domain/`, `ports/` or
`application/` imports torch, mujoco, numpy or sqlite3 (`docs/ARCHITECTURE.md`).

**Skips.** A suite that can skip prints `[skip]`, counts skips, and appends
`, N skipped` to its success line. The bare success strings above appear only
on a run with zero skips, so grepping for them is safe. `test_physics.py` and
`test_search.py` run every function through `run_all`: a function that hits the
missing GPU extension is reported `[skip]` and the checks it printed before
stopping are withdrawn.

**What each suite needs.**

| suite | needs |
|---|---|
| cheap suites above | nothing installed |
| `test_ppo.py` | numpy, torch (last section wants the evaluator) |
| `test_physics.py` | MuJoCo, no GPU; one function skips without the GPU extension |
| `test_search.py` | GPU; a few functions skip without it |

`tools/suite_probe.py --all` re-measures this. `.github/workflows/checks.yml`
runs the first three rows; `test_search.py` is not in CI.

A headless machine needs `MUJOCO_GL=disable`: MuJoCo 3.13 imports a renderer at
`import mujoco`.

### Whether a test is worth anything

Four tools, none needs a GPU:

```bash
PYTHONPATH=. .venv/bin/python tools/mutate.py            # break the code on purpose
PYTHONPATH=. .venv/bin/python tools/suite_probe.py --all # what can run at all
PYTHONPATH=. .venv/bin/python tools/assertion_audit.py   # assertions that cannot fail
PYTHONPATH=. .venv/bin/python tools/coverage_report.py   # lines nothing touches
```

`tools/mutate.py` is the important one. Removing PPO's clipping entirely left
all 54 checks in `test_ppo.py` green; a check on `torch.manual_seed` passed on a
`run_search` that no longer called it, because it grepped the source and a
commented-out call still contains the string. **Add a mutation whenever you add
a gate.** A surviving mutant is a hole in the tests. Method and findings:
`docs/TEST_AUDIT.md`.

## The Mojo kernel

`mojo/build/*.so` is a compiled mirror of `physics/fluid.py`. The search scores
through it; every verification, probe and film uses the numpy side. After
editing anything in `mojo/src`:

```bash
cd mojo && pixi run build-all      # full_pipeline.so, fluid_gpu.so, BUILD_MANIFEST.json
```

- `batchroll.usable()` refuses a kernel older than its source (via
  `BUILD_MANIFEST.json`). Before this check, the kernel ran five weeks behind
  `mojo/src` (2026-08-04 build vs a 09-15 source change), air measurements
  differed between the two paths by up to 2.0, and `tests/test_gpu_mirror.py`
  did not catch it because it compares the two sources.
- `mojo build -o` overwrites its output in place (same inode), and a live run
  has `mojo/build/*.so` mapped. `build-all` builds to a temporary name and
  renames. To test a new kernel while a run is live, build it elsewhere, write
  its manifest (`write_manifest(build=Path(dir))`), and point processes at it
  with `DYTISCIDAE_KERNEL_DIR=<dir>`. Install into `mojo/build` only when no run
  uses it.
- Do not use pinned host memory: Mojo's first pinned buffer reserves a ~1.35 GB
  pool per process, which put arch44's first launch over its memory ceiling at
  gen 0.

## Running a search

Two entry points, same search.

**As a job** (`docs/ARCHITECTURE.md`): recorded, paused and resumed by name, a
failure is stored with its traceback, SIGTERM becomes a resumable pause.

```bash
python -m dytiscidae.ops.run experiment new --name archNN --trainer search \
    --steps 900 --seed 20260901 --hypothesis "..." \
    --set batch=16 --set segment_seconds=8 --set controller_refine_steps=0 \
    --set use_shared_policy=true
python -m dytiscidae.ops.run job start  --experiment archNN --workers 4 --min-shard 2
python -m dytiscidae.ops.run job status <job-id>      # progress, eta, what is at risk
python -m dytiscidae.ops.run job pause  <job-id>      # honoured at the next generation
python -m dytiscidae.ops.run job resume <job-id>      # refuses without a checkpoint
```

**Directly**, which is what every stored run used:

```bash
systemd-run --user --unit archNN --same-dir bash -c \
  '.venv/bin/python -u -m dytiscidae.ops.run search \
     --generations 900 --batch 16 --workers 4 \
     --segment-seconds 8 --refine-steps 0 --shared-policy \
     --memory-ceiling-mb <MB> --seed 20260901 --run runs/archNN \
     > runs/archNN.log 2>&1 < /dev/null'
```

- **Launch with `systemd-run --user`, not `setsid nohup`.** A process started
  from a Claude session stays in that session's cgroup scope, and `setsid` does
  not leave it; arch43 died at gen 39 in that scope's OOM.
- **Launch from a pinned worktree to keep editing while it runs**
  (2026-10-08, arch50): `git worktree add --detach ../archNN-tree <commit>`,
  then `systemd-run` with `--setenv=PYTHONPATH=<tree>`,
  `--setenv=DYTISCIDAE_KERNEL_DIR=<main>/mojo/build` (`mojo/build` is
  gitignored), the main `.venv` python and `--run <main>/runs/archNN`. Worker
  processes import from the tree they were spawned in, so edits to the main
  tree cannot reach the run.
- `--memory-ceiling-mb` stops cleanly with a checkpoint when parent plus workers
  pass it (0 = off). arch35 lost 405 generations to an OOM kill it could have
  resumed from.
- **Pool shape: a queue by default since 2026-10-03, `--min-shard 2
  --pool-per-worker 2`, 8 shards of 2 pulled by 4 workers.** On random bodies
  4 shards of 4 were the optimum, swept on the real evaluation path (4×4 31.7 s,
  1×16 72.2 s, 2×8 47.6 s, 8×2 51.8 s, **16×1 93.3 s — slower than a single
  process**); re-swept 2026-09-27 under full load, 8×2 was faster (64.3 s vs
  74.6 s), but worker count is bounded by memory before cores, so 4 workers
  stay. On late, rotor-heavy bodies a design's cost is heavy-tailed (+1.6–1.8 s
  per rotor) and the queue takes 0.875× the 4×4 wall, faster on 3/3 batches
  (`experiments/budget_sweetspot/`). Sweep pool shape, never model it: a
  wall-time model predicted 16×1 would win and it lost by 3x.
- `--min-shard` defaults to 2 (with `--pool-per-worker 2`). The old default of
  8 made `split() = max(1, min(workers, n // min_shard))` return one shard after
  a single Tier-0 rejection at batch 16 (`15 // 8 = 1`), and the pool silently
  ran in the parent with the workers idle; two generations in three ran that way
  before it was found. Do not raise it.
- Cost per generation moves with every physics change; read it from the run's
  `generations.jsonl` (its `cost` field breaks it down by phase,
  `GenerationCost`). Measured 2026-09-27: main evaluation 42.6 s at 4×4; arch44
  gen 0 took 486 s. Measured 2026-10-03: a generation costs ~110 s early and
  ~225 s late in a run, because rotors accumulate. Generations 0–7 are a
  one-time eight-island verification burst (~300 s each) and are several times
  slower than steady state. Not a regression.

**Post-run runs by itself** when a search finishes (direct or job): chart report,
then films (`dytiscidae/viz/film.py`) into `runs/<run>/media/`, embedded in
`report.html`. By hand: `python -m dytiscidae.ops.run postrun --run runs/<run>`.

## Hard rules

- **Never disturb a live run.** Killing, reconfiguring or writing into a running
  `runs/archNN` destroys that arm. Keep spawned process counts low; the pool is
  contention-sensitive.
- **Killing the parent orphans the workers.** Workers run as
  `python -c from multiprocessing.spawn …`; `pkill -f "ops.run search"` misses
  them, and they once survived two hours holding 1 GB. Kill by PID: the parent,
  then `pgrep -P <parent>`.
- **`pkill -f` / `pgrep -f` match your own shell**, because the pattern is in
  your `bash -c` command line (`pkill -f spotify` killed the shell running it).
  Use `pgrep` → PID → `kill`, or a `[b]racket` in the pattern.
- **`ps -o pcpu` is a lifetime average.** It showed 91% parent / 10% workers
  while the instantaneous split was 100% / 0%. Sample `/proc/<pid>/stat`
  jiffies over a window.
- **MuJoCo QACC warnings are resets.** On `WARNING: Nan, Inf or huge value in
  QACC` MuJoCo resets the state to the default pose and keeps stepping. It
  prints once per `mjData` until `mj_resetData`, so the log count is a lower
  bound; the real rate is `diverged_rollouts / rollouts` in `generations.jsonl`.
  Scored segments are gated on `bad_qacc` and zeroed on divergence. Any new
  rollout loop must read that counter, because position checks cannot see a
  reset. `mj_forward` never resets. Filter the log with
  `grep -viE "WARNING: Nan|^$"`.

## Reading telemetry

- `filled`, `coverage`, `qd_score` in a generation line belong to the island that
  generation visited. Consecutive lines are different archives, not a series.
- `descriptor_refit` fires every 400 evaluations (~24 generations) and merges
  cells, so coverage sawtooths ~5 points. Judge archive growth per island,
  between refits.
- A step change in `mission_corr` usually means the auditor invalidated
  something. Check `auditor.invalidated` first.
- `on-task` in a showcase is which medium the machine is in, not whether it is
  doing anything: a machine sitting still on the beach scores 98% on land.
- **Comparability boundaries.** Air scores and `mission_fraction` are not
  comparable across arch33→34, arch34→35, arch36→37 (air ladder 7 → 11 rungs),
  2026-09-20, and arch43 (corrected physics: nothing before it is comparable on
  air or water forces). From 2026-10-10 (ARCH51_SPEC N5, arch51 onward) the
  stored `fitness`, `objectives[0]` and everything built on them (`qd_score`,
  parent weight, migrants, promotion and prune order, `best`) are not
  comparable with any earlier run: a design competent in none of its island's
  media (< 0.012) stands at 0 where it stood at its window's zero share
  (0.46-0.99), ranking is among designs above that floor, `mission_weight` is
  0.0 (was 0.30), and Tier-2 no longer caps `fitness`. With
  `--reeval-per-generation > 0` (N11) an elite's recorded scores are those of
  its median draw, not one draw. Per-medium competences, `mission_fraction`
  and coverage are unchanged by both. The ROADMAP lists each boundary. Write
  "not comparable"; do not quote a shrunken difference.
- `sink_rate` is the difference of two endpoints of the late airborne window, so
  it reads zero for a machine that drops and comes back. `station_keeping` sees
  the path between; where they disagree, trust `station_keeping` (arch37: 10x
  gap).
- **Two evaluation paths.** `batchroll.evaluate_tier1_batch` scores the search;
  `evaluate.evaluate_tier1` serves Tier-2 verification, offline probes and the
  showcase. They differed 60x on land locomotion until the scatter seed was
  shared, and still differ in water with identification on.
  `tests/test_search.py` asserts their agreement; read it before trusting an
  offline re-measurement.
- **A film is evidence only if it reproduces the score printed on it.** Judge a
  design from `media/evaluated_vs_mission.mp4` and `film_manifest.json`, never
  from a continuous-mission film alone. Each pair shows the Tier-1 evaluation
  re-run with a camera (same seed, scatter, task, control law) beside the
  continuous mission from the beach. Each evaluated clip is stamped with the
  recorded and reproduced score; a red stamp did not reproduce. Before
  2026-09-21 no film reproduced its score, and a water glider that could not
  leave the beach was read as "a machine that only trembles".

## Before adding anything: search for it

The expensive mistake here is a second implementation of something that already
exists under another name. Sizes are in the first line of
`docs/index/MODULES.md` (2026-09-30: 102 modules, 34,195 lines, 1,023 symbols).

Search in this order; an empty first result is normal, keep going:

1. `docs/index/FEATURES.yaml` — capabilities mapped to the symbols that implement
   them and the tests that hold them. Only this file records that e.g.
   `evaluate_tier1_batch` and `evaluate_tier1` are one feature.
2. `docs/index/SYMBOLS.md` — every public symbol with file and line.
3. `docs/index/MODULES.md` — purpose, size, imports, inside/outside the hexagon.
4. The source, its usages, its tests.
5. `rg`, with more than one spelling.

Then prefer **reuse → extend → refactor → create**. Before adding a class,
function or module that overlaps an existing one, name the overlap and say why
extending it does not work.

After changing any source file:

```bash
PYTHONPATH=. .venv/bin/python tools/index_gen.py write   # MODULES.md, SYMBOLS.md
PYTHONPATH=. .venv/bin/python tools/index_gen.py check   # exit 1 on drift
PYTHONPATH=. .venv/bin/python tests/test_index.py        # FEATURES.yaml resolves
```

A change that makes a sentence in `README.md`, `docs/` or this file false edits
that sentence in the same commit (2026-10-06: a review concluded the controller
was untrained and the machine CPU-only from README text that had been stale for
weeks). A quoted number carries its run and date.

`MODULES.md` and `SYMBOLS.md` are generated; never edit them by hand.
`FEATURES.yaml` is hand-maintained: update it when a feature gains, loses or
moves an entry point. `test_index.py` fails when it names a missing symbol, when
a feature has no test, or when a package is named by no feature.

## Designing a measurement

Each rule below was paid for by a score that rewarded the wrong thing.

1. **Ask what it reads for a machine that is falling, bouncing, and holding its
   actuators still.** The last is a two-line experiment. Instances so far: a
   wingless design thrown at 30 m/s scored 0.853 for flight; a tumble scored as
   a commanded turn; a rebound off the beach scored as take-off; arch37's
   `holds_height` was cleared by dropping and coming back; `land_speed` and
   `slope_climbed` moved 1% with actuators off (0.4312 vs 0.4262 m/s, 0.4139 vs
   0.4091) because `scatter` pushed every land segment at 0.54 m/s.
2. **Fix it with a gate, not a coefficient.** Water paid a still machine 0.422
   (air 0.002, land 0.006) because `0.2 * submerged + 0.2 * upright` was added.
   A passive twin subtracted out (2026-09-20) worked but cost 38% of every
   generation and refused to score a glide. The replacement (the user,
   2026-09-21): state multiplies instead of adds, motion terms are gated on
   `max(headway, depth_station_keeping)`.
3. **Tell the machine what to do.** A gate that blends two purposes still fights
   itself (`corr = -0.264`). Each segment is now two commanded phases the
   controller can see, each scored on its own purpose (`envs/tasks.py`). A still
   machine was measured at ~0.01 in water on that day's bodies; on arch48's 200
   elites it cleared water 0.15 in 35 cases against the elites' 45, by a dense
   body sinking along the drawn heading (`experiments/no_model_gate`,
   docs/PAPERS_2610.md §1) -- closed by rule 8. Re-measure the still machine on
   the current population, not on a fixture.
4. **"Could not measure" must not share a value with "measured zero".**
   `thrust_margin` returned 0.0 for both, the rung sat at `>= 0.0`, and 8 of 80
   re-scored elites cleared it while flapping ~1 rad to no effect. Publish
   nothing; a missing metric stops `rung_reached` where it is.
5. **Set thresholds from the measured distribution.** `moves` at 0.1 m/s left
   61.6% of arch34 below it with nowhere to stand.
6. **Run the still-machine check on every score, including the ones that look
   like plain physics.** 2026-10-03, the crossings: with the actuators held
   still, `air_to_water` paid 0.834 (it fell in, from a placed descent) and
   `water_to_land` 0.865, because the probe started the machine dry on the ramp
   and counted *any* change of wetness, so settling into the water was a
   crossing. These were the learner's two largest rewards. The fix had rule 3's
   shape: a commanded hold-then-go phase and a directional crossing
   (`transitions.CrossingTracker`, ARCH46_SPEC §8). Checking the still machine
   caught two more on the way, a floating body counted as "aloft" and a
   placement gap counted as height.
7. **A still machine must be still in every actuator kind, and a hold must read
   the quantity a passive body spends, not the one it keeps.** 2026-10-04, what
   rule 6's fix left: still machines still crossed 2-5% of the time, at or above
   the elites' own rate. A glider coasting at 20 m/s held its *height* through
   the hold while losing 3-6 m of *energy* height; a body floating over the
   submerged ramp counted as "on land"; and the "still" arm left rotors at
   throttle, because a rotor's channel is a speed held at its offset and zeroing
   amplitude does not stop it. Gated (`CrossingTracker`, and
   `TriphibianEnv.held_still_params`): still machines now cross 0 of 218 in
   every kind, and so do the elites, bar one. The crossings the elites had were
   the same leak.
8. **A command must be answerable both ways, and a score must survive a fresh
   seed.** 2026-10-06 (docs/PAPERS_2610.md): one heading per generation, shared
   by 16 candidates, made every water and land score a measurement of the draw
   (a fresh seed took Tier-1 passes 15/16 -> 3/16 in water, 13/16 -> 0/16 on
   land), and a dense still body cleared water 0.15 as often as the elites by
   sinking along whichever heading lined up. Water and land are now an
   antipodal pair (`tasks.antipode`, `evaluate.run_segment`, and the same
   sequence in `batchroll`): one initial state, the drawn heading and its
   opposite, progress on the mean signed speed, so motion the command did not
   choose cancels exactly (still machines and open-loop gaits: |mean| < 1e-12).
   The auditor's held-out seeds read each medium, not only the mission. Since
   2026-10-08 each candidate also faces its own draw
   (`SearchConfig.draw_per_candidate`, ROADMAP M2): the batched path takes one
   seed per machine, and a result equals what that machine scores alone at its
   `eval_seed`.

## Where things are

| | |
|---|---|
| `docs/ROADMAP.md` | work list and the measurement behind every decision |
| `docs/ARCHITECTURE.md` | job/ports layer: what a run is, how it starts, pauses, resumes, is recorded |
| `docs/MATH_AUDIT.md` | every physics/control/optimisation formula and constant: source, units, validation, risk rank |
| `docs/model_validity.md` | what the fluid solver is and is not |
| `docs/LEARNER_AUDIT.md` | the RL side against a standard checklist |
| `docs/TEST_AUDIT.md` | mutation survival, coverage, assertions that cannot fail |
| `docs/AUTONOMY.md` | direction: removing the human from the loop |
| `docs/PORTING.md` | moving runs off an ephemeral container |
| `docs/CPU_LEGACY.md` | older backlog; ROADMAP wins where they disagree |
| `docs/index/` | where everything is (see above) |
| `derivations/` | one document per quantity, with the measurement that checks it |
| `experiments/` | `python experiments/<name>/run.py` re-derives the audit's numbers; `experiments/perf/NOTES.md` is the worker profile |
| `runs/<run>/report.html` | a run's chart report and films |
| `runs/<run>/checkpoint.npz`, `.json` | network, Adam moments, rng state, best elites, the commit that wrote it |
| `runs/<run>_notes.md` | a run's configuration and findings |
| `.claude/skills/training-report/` | regenerates the report for any run |
| `.claude/agents/` | six roles: explorer, mutator, assumption-breaker, adversary, judge, historian |
