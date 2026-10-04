# Dytiscidae — project operating notes

Generative design + control search for triphibian flapping-wing machines.
MAP-Elites over eight islands, MuJoCo rigid bodies plus this project's own
quasi-steady fluid solver, a shared PPO policy the search keeps for its
variation operator. `docs/ROADMAP.md` is the work list and carries the
measurement behind every item — **read it before proposing anything**.

## Session start

1. `git status` and `git log --oneline -5`.
2. Read `docs/ROADMAP.md` §"arch45 — the work list" (ranked), and §"What is
   set by measurement, and what is typed" before proposing any threshold. The
   latest finished run is arch44: `runs/arch44_notes.md`.
3. Canary — the three suites, filtered, ~15 min total:

```bash
PYTHONPATH=. .venv/bin/python tests/test_ppo.py 2>&1 | tail -2
PYTHONPATH=. .venv/bin/python tests/test_physics.py 2>&1 | grep -ciE '\[fail'
PYTHONPATH=. .venv/bin/python tests/test_search.py 2>&1 | tail -2
```

Expect `shared PPO checks passed`, `0`, `all search-machinery checks passed`.
There is no pytest; every suite is a script with a `main()`. `test_search.py`
takes ~25 min — run it detached with `setsid nohup … &`, never in a tool call
that can time out and kill it.

The job/ports layer and the index have their own suites, and together they take
under two minutes, so they are cheap enough to run on every change:

```bash
for s in index architecture domain application search_adapter adapters worker; do
    PYTHONPATH=. .venv/bin/python tests/test_$s.py | tail -1
done
```

`test_architecture.py` is the one that matters most: it fails the build if
anything in `domain/`, `ports/` or `application/` starts importing torch,
mujoco, numpy or sqlite3. See `docs/ARCHITECTURE.md`.

**A skip is not a pass, and the summary line says so.** Every suite that can
skip now prints `[skip]`, counts skips separately, and appends `, N skipped` to
its success line. The bare strings above — `all physics checks passed`,
`all search-machinery checks passed` — appear **only** on a run where nothing
was skipped, which is what makes them safe to grep for.

**No suite stops at the first blocked function any more.** `main()` in
`test_physics.py` and `test_search.py` runs every test through `run_all`, which
reports a function that hits the missing GPU extension as `[skip]` and
*withdraws the checks it printed before it stopped* — a function that did not
finish is evidence neither way. Measured cost of the old flat call list:
`test_physics` ran 144 checks and now runs 203, `test_search` ran 133 and now
runs 317.

**What each canary actually needs.** Only `test_search.py` still needs a GPU:
it skips 3 of its 45 functions without one. `test_physics.py` needs MuJoCo and
**not** a GPU — 36 of its 37 functions pass on a bare CPU — so it is in CI now.
`test_ppo.py` needs numpy and torch and nothing else; only its last section
wants the evaluator. The suites in the loop above need no third-party package at
all. `.github/workflows/checks.yml` runs all three tiers.

**A headless machine needs `MUJOCO_GL=disable`**, because MuJoCo 3.13 imports a
renderer at `import mujoco`.

**After editing anything in `mojo/src`, rebuild:**

```bash
cd mojo && pixi run build-all      # full_pipeline.so, fluid_gpu.so, and the manifest
```

`mojo build -o` overwrites its output **in place** (same inode, checked
2026-09-27), and a live run has `mojo/build/*.so` mapped, so `build-all` now
builds to a temporary name and renames. Even so, verify a new kernel before a
run can pick it up: build it elsewhere and point processes at it with
`DYTISCIDAE_KERNEL_DIR=<dir>` (it needs its own `BUILD_MANIFEST.json`:
`write_manifest(build=Path(dir))`). Install into `mojo/build` only when no run
is using it.

`mojo/build/*.so` is a compiled mirror of `physics/fluid.py`, and the search
scores through it while every verification, probe and film uses the numpy half.
Measured 2026-09-20: the built kernel was from 2026-08-04 and `mojo/src` had
been changed on 09-15 by F-05 (the stall blend became a smoothstep), so the two
paths ran different physics for five weeks, including all of arch39 — air
measurements differed between the paths by up to 2.0, and agree to 0.000002
after a rebuild. `tests/test_gpu_mirror.py` did not catch it: it compares the
two *sources*, which agreed. A build now writes `BUILD_MANIFEST.json` and
`batchroll.usable()` refuses a kernel older than its source, so a run says so at
t=0 instead of scoring with physics nobody is verifying.

## Judging whether a test is worth anything

Four instruments, all of which run without a GPU:

```bash
PYTHONPATH=. .venv/bin/python tools/mutate.py            # break it on purpose
PYTHONPATH=. .venv/bin/python tools/suite_probe.py --all # what can run at all
PYTHONPATH=. .venv/bin/python tools/assertion_audit.py   # assertions that cannot fail
PYTHONPATH=. .venv/bin/python tools/coverage_report.py   # lines nothing touches
```

`tools/mutate.py` is the one that matters. A green suite proves nothing about
its own effectiveness: it found that **removing PPO's clipping entirely left all
54 checks in `test_ppo.py` green**, and that the check on `torch.manual_seed`
passed on a `run_search` that no longer called it, because the check grepped the
source and a commented-out call still contains the string. **Add a mutation
whenever you add a gate**, and treat a survivor as a hole in the tests rather
than a curiosity. Findings and method: `docs/TEST_AUDIT.md`.

## Running a search

Two ways, and they do the same search.

**As a job**, which is what `docs/ARCHITECTURE.md` describes: the run is
recorded, it can be paused and resumed by name instead of by pid, a failure
lands in the record with its traceback, and SIGTERM becomes a resumable pause
rather than a lost run.

```bash
python -m dytiscidae.ops.run experiment new --name arch39 --trainer search \
    --steps 900 --seed 20260901 --hypothesis "..." \
    --set batch=16 --set segment_seconds=8 --set controller_refine_steps=2 \
    --set use_shared_policy=true
python -m dytiscidae.ops.run job start --experiment arch39 --workers 4 --min-shard 2
python -m dytiscidae.ops.run job status <job-id>      # progress, eta, what is at risk
python -m dytiscidae.ops.run job pause  <job-id>      # honoured at the next generation
python -m dytiscidae.ops.run job resume <job-id>      # refuses without a checkpoint
```

**Directly**, unchanged, which is what every stored run used:

```bash
setsid nohup .venv/bin/python -u -m dytiscidae.ops.run search \
  --generations 900 --batch 16 --workers 4 \
  --segment-seconds 8 --refine-steps 2 --shared-policy \
  --seed 20260901 --run runs/archNN > runs/archNN.log 2>&1 < /dev/null &
```

**The pool is a queue by default since 2026-10-03: `--min-shard 2
--pool-per-worker 2`, 8 shards of 2 pulled by 4 workers.** On random bodies
4 shards of 4 were the optimum (31.7 s, where 1×16 takes 72.2 s, 2×8 47.6 s, 8×2
51.8 s and **16×1 93.3 s — slower than a single process**). On late,
rotor-heavy bodies a design's cost is heavy-tailed (+1.6–1.8 s per rotor) and the
queue takes 0.875× the 4×4 wall, faster on 3/3 batches
(`experiments/budget_sweetspot/`). Sweep pool shape, never model it: a wall-time
model predicted 16×1 would win and it lost by 3x. A generation costs ~110 s
early and ~225 s late in a run, because rotors accumulate.

`--min-shard 8` is a trap at batch 16: `split()` is
`max(1, min(workers, n // min_shard))`, so a **single** Tier-0 rejection makes
`15 // 8 = 1`, one shard, and the pool silently falls back to running in the
parent with the workers idle. Two generations in three ran that way before it
was found.

Generations 0–7 are a one-time eight-island verification burst at ~300 s each;
steady state is ~74 s. Not a regression.

## Hard rules

- **Never disturb a live run.** `runs/arch34` and any successor is a multi-hour
  single-arm experiment; killing, reconfiguring or writing into it destroys the
  arm. Keep spawned process counts low — the pool optimum is contention-sensitive.
- **Killing the parent orphans the workers.** `pkill -f "ops.run search"` matches
  only the parent; workers run as `python -c from multiprocessing.spawn …` and
  survived two hours holding 1 GB. Kill by PID: parent, then `pgrep -P <parent>`.
- **`pkill -f` and `pgrep -f` match your own shell.** Every pattern you type is
  in your own `bash -c` command line, so `pkill -f spotify` killed the shell
  running it. Use `pgrep`→PID→`kill`, or a `[b]racket` in the pattern.
- **`ps -o pcpu` is a lifetime average.** It showed 91% parent / 10% workers
  while the true instantaneous split was 100% / 0%. Sample `/proc/<pid>/stat`
  jiffies over a window instead.
- Pipe chatty output. MuJoCo prints a `WARNING: Nan, Inf or huge value in QACC`
  block per unstable rollout: `grep -viE "WARNING: Nan|^$"`. **What it means:**
  MuJoCo then *auto-resets the state and keeps stepping* — a teleport to the
  default pose. It prints once per `mjData` until `mj_resetData`, so the log
  count is a lower bound; the real rate is `diverged_rollouts / rollouts` in
  `generations.jsonl` (arch39: 0.15%). Scored segments are gated on `bad_qacc`
  and a divergence zeroes them; any *new* rollout loop must read the counter
  too, because position checks cannot see a reset. `mj_forward` never resets.

## Reading telemetry without being fooled

- `filled`, `coverage`, `qd_score` in a generation line belong to **whichever
  island that generation visited**. Consecutive lines are different archives;
  comparing them as a series is a category error.
- A `descriptor_refit` fires every 400 *evaluations* (~24 generations) and merges
  cells — coverage sawtooths ~5 points. Any judgement about archive growth must
  be per island and between refits.
- A step change in `mission_corr` usually means the auditor invalidated
  something. Check `auditor.invalidated` before reaching for another explanation.
- `on-task` in a showcase means *which medium the machine is in*, not whether it
  is doing anything: a machine sitting still on the beach scores 98% on land.
- Air scores, and `mission_fraction`, are **not comparable across arch33→arch34,
  arch34→arch35, or arch36→arch37** (the air ladder went from 7 rungs to 11).
  Each boundary redefined a term. Say "not comparable" rather than shrinking the
  difference.
- **`sink_rate` is the difference between two endpoints** of the late half of the
  airborne window, so it reads zero for a machine that goes down and comes back
  up. `station_keeping` is the one that sees the shape between them; where the
  two disagree, believe the second. arch37 measured the gap at 10x.
- **There are two evaluation paths and they are not interchangeable.**
  `batchroll.evaluate_tier1_batch` is what the search scores with;
  `evaluate.evaluate_tier1` is what Tier-2 verification, every offline probe and
  the showcase use. They differed by 60x on land locomotion until the scatter
  seed was shared, and they still differ in water when identification is on.
  `tests/test_search.py` now asserts the agreement — read it before trusting an
  offline re-measurement.
- **A film is evidence only if it reproduces the score printed on it.** Until
  2026-09-21 none did: `render --top` drove elites open-loop at seed 0, the
  showcase added a shared policy the scores were earned without (the actor pool
  had dropped it from every re-score), and the first air reset of every body
  lost its spawn offset. `film_manifest.json` records, per medium, the recorded
  and reproduced competence and which control law reproduced it.

## Before adding anything: look for it first

98 modules, 30,000 lines, 939 indexed symbols. The expensive mistake here is not
a missing feature, it is a **second implementation of one that already exists
under a name nobody searched for**. `docs/index/` exists so that "does this
already exist, and where?" is cheap to answer.

Go in this order, and do not stop early because the first search came back
empty — a different name is the normal case, not the exception:

1. `docs/index/FEATURES.yaml` — 47 capabilities, each mapped to the symbols that
   implement it and the tests that hold it. Start here, because it is the only
   one that records "these six functions across four packages are one feature".
2. `docs/index/SYMBOLS.md` — every public class, function, method and property
   with its file and line.
3. `docs/index/MODULES.md` — what each module is for, its size, its imports, and
   whether it is inside the hexagon.
4. The source, then its usages and its tests.
5. Only then `rg`, with more than one spelling of the name.
6. Only after all of that is it fair to conclude it does not exist.

Then prefer, in this order: **reuse → extend → refactor → create.** Do not add a
class, function or module that overlaps an existing one without first saying
which one it overlaps and why extending it does not work.

**After changing any source file, regenerate the index:**

```bash
PYTHONPATH=. .venv/bin/python tools/index_gen.py write   # MODULES.md, SYMBOLS.md
PYTHONPATH=. .venv/bin/python tools/index_gen.py check   # exit 1 on drift
PYTHONPATH=. .venv/bin/python tests/test_index.py        # + FEATURES.yaml resolves
```

`MODULES.md` and `SYMBOLS.md` are generated from the AST and **must not be
edited by hand**. `FEATURES.yaml` is maintained by hand and must be updated when
a feature gains, loses or moves an entry point — `tests/test_index.py` fails the
build when it names something that no longer exists, when a feature has no test,
or when a package of the project is named by no feature at all. `test_index.py`
needs no third-party package and runs in about a second.

## Where the ground truth lives

| | |
|---|---|
| `docs/ROADMAP.md` | work list + the measurement behind every decision |
| `docs/ARCHITECTURE.md` | the job/ports layer: what a run *is*, how it is started, paused, resumed, cancelled and recorded |
| `docs/MATH_AUDIT.md` | every physics/control/optimisation formula and constant: source, assumption, units, derivation and validation status, risk-ranked |
| `docs/LEARNER_AUDIT.md` | the RL half against a standard checklist: what holds, what is deliberate, what is a gap, and the four things it changed |
| `docs/TEST_AUDIT.md` | whether the tests are worth anything: mutation survival, coverage, assertions that cannot fail, and what cannot be run here |
| `docs/index/` | **where everything is.** `FEATURES.yaml` by hand, `MODULES.md` and `SYMBOLS.md` generated — see "Before adding anything" below |
| `derivations/` | one document per quantity, derived from its own premises, with the measurement that checks it |
| `experiments/` | reproducible harnesses: `python experiments/<name>/run.py` re-derives every number the audit quotes |
| `docs/CPU_LEGACY.md` | older backlog, superseded where they disagree |
| `runs/arch38/report.html` | the latest run's full chart report |
| `runs/<run>/checkpoint.npz`, `.json` | the portable checkpoint: network, Adam moments, rng state, best elites with their bases, and the commit that wrote it |
| `runs/archNN_notes.md` | each run's configuration and findings |
| `.claude/skills/training-report/` | regenerates the report for any run |
| `.claude/agents/` | six roles: explorer, mutator, assumption-breaker, adversary, judge, historian |

**Post-run happens by itself.** A search that finishes — direct or as a job —
runs `ops.run postrun`: the chart report, then the films (`dytiscidae/viz/film.py`)
into `runs/<run>/media/`, embedded at the top of `report.html`. By hand:
`python -m dytiscidae.ops.run postrun --run runs/<run>`.

**Judge a design only from `media/evaluated_vs_mission.mp4` and
`film_manifest.json`, never from a continuous-mission film alone.** Each pair
puts the medium *as it was evaluated* — the Tier-1 evaluation itself, re-run with
a camera, same seed, scatter, task and control law — beside the continuous
mission, which starts on the beach and shows only whether the machine can get
between media. Every evaluated clip is stamped with the recorded and the
reproduced score; a red one did not reproduce and is not evidence for that
medium. On 2026-09-21 the continuous mission was shown alone and a water glider
that could not leave the beach was read as "a machine that only trembles".

## The lesson this project keeps re-learning

Six times a score has paid for uncontrolled motion — a wingless design thrown
at 30 m/s scoring 0.853 for flight, a tumble scoring as a commanded turn, a
machine rebounding off the beach scoring as a take-off, arch37's `holds_height`
rung cleared by a trajectory that drops and comes back, and then **`land_speed`
and `slope_climbed`, which moved by one percent when the actuators were switched
off** — 0.4312 against 0.4262 m/s, and 0.4139 against 0.4091 — because `scatter`
shoved every land segment at 0.54 m/s and nothing gated the displacement on
posture. Each time the fix was a **gate**, not a coefficient. When adding any
measurement, ask what it reads for a machine that is falling, for one that is
bouncing, and **for one with its actuators held still** — that last one is a
two-line experiment and it has now caught two rungs.

And the seventh instance is the one that proves the rule twice over: water paid
a machine with its actuators held still **0.422** of a competence, against 0.002
in air and 0.006 on land, because `0.2 * submerged + 0.2 * upright` was *added*
— the machine is released four metres under and a hull is passively stable. The
first fix, 2026-09-20, was a subtraction: run every segment a second time with
the actuators held still and score the difference. It worked and it cost 38% of
every generation, and it also refused to score a glide. On 2026-09-21 the user
replaced it with a gate, which is what the paragraph above says to reach for:
state multiplies instead of adding, and the motion terms were gated on
`max(headway, depth_station_keeping)`. That still blended two purposes that
fight (`corr = -0.264`), so the same day the reward became **task-conditioned**:
each segment is two commanded phases the controller can see, each scored on its
own purpose (`envs/tasks.py`). A still machine now scores ~0.01 in water, with no
twin — a passive body does the same thing whatever it is commanded, so it cannot
score on both phases. **When a score pays for doing nothing, first ask whether
the machine was ever told what to do.**

The eighth, 2026-10-03, was in the crossings: with the actuators held still,
`air_to_water` paid 0.834 (it fell in, from a placed descent) and
`water_to_land` 0.865, because the probe started the machine dry on the ramp
and counted *any* change of wetness, so settling into the water was a
crossing. These were the learner's two largest rewards. The fix was the same
shape as water's: a commanded hold-then-go phase and a directional crossing
(`transitions.CrossingTracker`, ARCH46_SPEC §8). Checking the still machine
caught two more on the way, a floating body counted as "aloft" and a placement
gap counted as height. **Run the still-machine check on every score, including
the ones that look like plain physics.**

The ninth, 2026-10-04, was what the eighth's fix left: still machines still
crossed 2-5% of the time, at or above the elites' own rate. A glider coasting at
20 m/s held its *height* through the hold while losing 3-6 m of *energy* height;
a body floating over the submerged ramp counted as "on land"; and the "still"
arm left rotors at throttle, because a rotor's channel is a speed held at its
offset and zeroing amplitude does not stop it. Gated (`CrossingTracker`, and
`TriphibianEnv.held_still_params`): still machines now cross 0 of 218 in every
kind, and so do the elites, bar one. The crossings the elites had were the same
leak. **A still machine must be still in every actuator kind, and a hold must
read the quantity a passive body spends, not the one it keeps.**

And: a quantity that means "I could not measure this" must not share a value with
a quantity that means "I measured zero". `thrust_margin` returned 0.0 for both and
the rung above it sat at `>= 0.0`; eight of eighty re-scored elites cleared it
while flapping nearly a radian to no effect. Publish nothing instead — a missing
metric stops `rung_reached` where it stands.

And: set thresholds from the measured distribution, never from what the
capability ought to look like. `moves` at 0.1 m/s left 61.6% of arch34 below it
with nowhere to stand.
