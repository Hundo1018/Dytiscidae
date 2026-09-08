# Dytiscidae — project operating notes

Generative design + control search for triphibian flapping-wing machines.
MAP-Elites over six islands, MuJoCo rigid bodies plus this project's own
quasi-steady fluid solver, a shared PPO policy the search keeps for its
variation operator. `docs/ROADMAP.md` is the work list and carries the
measurement behind every item — **read it before proposing anything**.

## Session start

1. `git status` and `git log --oneline -5`.
2. Read `docs/ROADMAP.md` §"arch38 — the work list". Nothing in it is
   implemented yet; item A is a free diagnostic and decides the rest.
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

## Running a search

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

## Where the ground truth lives

| | |
|---|---|
| `docs/ROADMAP.md` | work list + the measurement behind every decision |
| `docs/CPU_LEGACY.md` | older backlog, superseded where they disagree |
| `runs/arch37/report.html` | the latest run's full chart report |
| `runs/archNN_notes.md` | each run's configuration and findings |
| `.claude/skills/training-report/` | regenerates the report for any run |
| `.claude/agents/` | six roles: explorer, mutator, assumption-breaker, adversary, judge, historian |

Post-run: `.venv/bin/python .claude/skills/training-report/report.py runs/<run>`
then film with `showcase --design runs/<run> --by mission` — `--by fitness` is
the default and picks a different machine (corr was +0.70, close and not close
enough).

## The lesson this project keeps re-learning

Four times a score has paid for uncontrolled motion — a wingless design thrown
at 30 m/s scoring 0.853 for flight, a tumble scoring as a commanded turn, a
machine rebounding off the beach scoring as a take-off, and arch37's
`holds_height` rung cleared by a trajectory that drops and comes back (measured:
a tenth of the `station_keeping` a straight descent at the same sink would give).
Each time the fix was a **gate**, not a coefficient. When adding any measurement,
ask what it reads for a machine that is falling — and for one that is bouncing.

And: set thresholds from the measured distribution, never from what the
capability ought to look like. `moves` at 0.1 m/s left 61.6% of arch34 below it
with nowhere to stand.
