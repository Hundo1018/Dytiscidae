# Dytiscidae — project operating notes

Generative design + control search for triphibian flapping-wing machines.
MAP-Elites over six islands, MuJoCo rigid bodies plus this project's own
quasi-steady fluid solver, a shared PPO policy the search keeps for its
variation operator. `docs/ROADMAP.md` is the work list and carries the
measurement behind every item — **read it before proposing anything**.

## Session start

1. `git status` and `git log --oneline -5`.
2. Read `docs/ROADMAP.md` §"arch39 — the work list", and §"What is set by
   measurement, and what is typed" before proposing any threshold. arch38 has
   run; its result and the two void launches before it are in
   `runs/arch38_notes.md`.
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

The job/ports layer has its own five suites, and together they take under two
minutes, so they are cheap enough to run on every change:

```bash
for s in architecture domain application search_adapter adapters worker; do
    PYTHONPATH=. .venv/bin/python tests/test_$s.py | tail -1
done
```

`test_architecture.py` is the one that matters most: it fails the build if
anything in `domain/`, `ports/` or `application/` starts importing torch,
mujoco, numpy or sqlite3. See `docs/ARCHITECTURE.md`.

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
python -m dytiscidae.ops.run job start --experiment arch39 --workers 4 --min-shard 4
python -m dytiscidae.ops.run job status <job-id>      # progress, eta, what is at risk
python -m dytiscidae.ops.run job pause  <job-id>      # honoured at the next generation
python -m dytiscidae.ops.run job resume <job-id>      # refuses without a checkpoint
```

**Directly**, unchanged, which is what every stored run used:

```bash
setsid nohup .venv/bin/python -u -m dytiscidae.ops.run search \
  --generations 900 --batch 16 --workers 4 --min-shard 4 \
  --segment-seconds 8 --refine-steps 2 --shared-policy \
  --seed 20260901 --run runs/archNN > runs/archNN.log 2>&1 < /dev/null &
```

**`--workers 4 --min-shard 4` is the measured optimum, not a guess.** Swept over
the real evaluation path: 4 shards of 4 take 31.7 s where 1×16 takes 72.2 s,
2×8 47.6 s, 8×2 51.8 s and **16×1 takes 93.3 s — slower than a single process.**
The optimum is a plateau at four shards with a largest shard of four. Sweep pool
shape, never model it: a wall-time model predicted 16×1 would win and it lost by
3x. Cost ≈ 74 s/generation, so 900 generations ≈ 21 h.

The default `--min-shard 8` is a trap at batch 16: `split()` is
`max(1, min(workers, n // min_shard))`, so a **single** Tier-0 rejection makes
`15 // 8 = 1`, one shard, and the pool silently falls back to running in the
parent with the workers idle. Two generations in three ran that way before it
was found.

Generations 0–5 are a one-time six-island verification burst at ~300 s each;
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
  block per unstable rollout: `grep -viE "WARNING: Nan|^$"`.

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
- **`on-task` in a showcase, and the film itself, are not evidence about a
  design unless the control law is the one that earned the score.** The showcase
  prints `driving as evaluated: ...` and says whether the mobility basis was
  stored or re-identified; re-identified is a different experiment.

## Where the ground truth lives

| | |
|---|---|
| `docs/ROADMAP.md` | work list + the measurement behind every decision |
| `docs/ARCHITECTURE.md` | the job/ports layer: what a run *is*, how it is started, paused, resumed, cancelled and recorded |
| `docs/MATH_AUDIT.md` | every physics/control/optimisation formula and constant: source, assumption, units, derivation and validation status, risk-ranked |
| `derivations/` | one document per quantity, derived from its own premises, with the measurement that checks it |
| `experiments/` | reproducible harnesses: `python experiments/<name>/run.py` re-derives every number the audit quotes |
| `docs/CPU_LEGACY.md` | older backlog, superseded where they disagree |
| `runs/arch38/report.html` | the latest run's full chart report |
| `runs/<run>/checkpoint.npz`, `.json` | the portable checkpoint: network, Adam moments, rng state, best elites with their bases, and the commit that wrote it |
| `runs/archNN_notes.md` | each run's configuration and findings |
| `.claude/skills/training-report/` | regenerates the report for any run |
| `.claude/agents/` | six roles: explorer, mutator, assumption-breaker, adversary, judge, historian |

Post-run: `.venv/bin/python .claude/skills/training-report/report.py runs/<run>`
then film with `showcase --design runs/<run> --by mission` — `--by fitness` is
the default and picks a different machine (corr was +0.70, close and not close
enough).

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

And: a quantity that means "I could not measure this" must not share a value with
a quantity that means "I measured zero". `thrust_margin` returned 0.0 for both and
the rung above it sat at `>= 0.0`; eight of eighty re-scored elites cleared it
while flapping nearly a radian to no effect. Publish nothing instead — a missing
metric stops `rung_reached` where it stands.

And: set thresholds from the measured distribution, never from what the
capability ought to look like. `moves` at 0.1 m/s left 61.6% of arch34 below it
with nowhere to stand.
