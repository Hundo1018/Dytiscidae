# Architecture

**Hexagonal Architecture with an Application/Use-Case Layer and Process-Isolated
Training Workers.**

以六角形架構為核心、搭配 Application / Use Case Layer 與獨立訓練 Worker 的
AI 訓練應用架構。

Written 2026-09-16. Every claim below that can be measured is measured, and the
command that measures it is given. Where something is asserted and not measured,
it says so.

---

## Why this shape, for this project

Dytiscidae is a generative design search: MAP-Elites over six islands, MuJoCo
rigid bodies with this project's own quasi-steady fluid solver, and a shared PPO
policy. A run is **900 generations, ~74 s each in steady state, about 21 hours**,
in a container that can be reclaimed without warning.

That last sentence is the whole argument. A twenty-one-hour computation in an
ephemeral container is not a function call. It is a *job*: it has a lifecycle, it
must be pausable, resumable and cancellable, it will sometimes fail, and every one
of those has to be a state someone can read afterwards — not an exception that
vanished with the process.

Before this refactor, the project had excellent machinery for the *inside* of a
run (`ops/checkpoint.py` carries the network, Adam's moments, the RNG state and
the commit that wrote it, each because its absence once cost a measurement) and
nothing at all for the outside. A run was started with `setsid nohup`, watched
with `tail -f`, and stopped by finding its pid. `CLAUDE.md` still records what
that costs:

> **Killing the parent orphans the workers.** `pkill -f "ops.run search"` matches
> only the parent; workers run as `python -c from multiprocessing.spawn …` and
> survived two hours holding 1 GB.

> **`pkill -f` and `pgrep -f` match your own shell.** Every pattern you type is
> in your own `bash -c` command line, so `pkill -f spotify` killed the shell
> running it.

Those are not operational mishaps. They are the symptoms of a *missing state
machine*: there was no way to ask a run to stop, so the only available verb was
kill, and the only available address was a command-line pattern.

---

## The dependency rule

```
                    ┌────────────────────────┐
                    │      Application       │
                    │       Use Cases        │
                    │                        │
                    │ StartTraining          │
                    │ PauseTraining          │
                    │ ResumeTraining         │
                    │ CancelTraining         │
                    │ CreateExperiment       │
                    │ LoadCheckpoint         │
                    │ ExportModel            │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │        Domain          │
                    │                        │
                    │ TrainingJob            │
                    │ TrainingPlan           │
                    │ TrainingState          │
                    │ Experiment             │
                    │ CheckpointRecord       │
                    │ Dataset                │
                    │ MetricPoint / JobEvent │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │         Ports          │
                    │                        │
                    │ Trainer                │
                    │ DatasetRepository      │
                    │ CheckpointStore        │
                    │ ExperimentStore        │
                    │ JobStore   ControlChannel
                    │ MetricSink EventLog    │
                    │ JobLauncher  Clock     │
                    └───────────┬────────────┘
                                │
     ┌──────────────┬───────────┼───────────┬──────────────┐
     ▼              ▼           ▼           ▼              ▼
  MuJoCo /       Filesystem   SQLite    Subprocess     Inline
  PyTorch        JSON/JSONL   DB        launcher       launcher
  SearchTrainer  adapters     adapter   (own session)  (tests)
```

Dependencies point inward, with exactly one exception, which is the composition
root (`adapters/composition.py`) — the only module in the package that is allowed
to name both an adapter and a use case.

### This is measured, not asserted

`tests/test_architecture.py` makes the rule a gate, two ways:

**Statically**, by walking the AST of every module in `domain/`, `ports/` and
`application/` and rejecting any import of `torch`, `mujoco`, `numpy`, `scipy`,
`sqlite3`, or of any outer package of this project.

**Dynamically**, by starting a fresh interpreter with those modules *hidden on
`sys.meta_path`* and running a whole job lifecycle in it. That is the honest form
of the claim: not "we believe nothing imports numpy" but "here is an interpreter
that cannot import numpy, and the domain runs in it."

```
$ PYTHONPATH=. python tests/test_architecture.py
  [ok  ] no forbidden import anywhere inside the hexagon  -- 28 modules clean
  [ok  ] the hexagon runs on an interpreter with no numpy, torch or mujoco
  [ok  ] importing the application pulls in no adapter and no heavy library
  [ok  ] listing trainers imports neither torch nor mujoco
  [ok  ] only the composition root wires adapters to use cases
```

The last two matter as much as the first. A trainer registry that imported every
trainer at module load would put MuJoCo and torch behind every use case,
including `ListJobs`; resolution is lazy and per name, so naming `search` in a
plan is what costs, not importing the package.

---

## The domain

Five words, and what each means *in this project* rather than in general.

### `TrainingPlan` — the configuration

Immutable, hashable, and the only thing a rerun needs to be the same experiment.
It carries the trainer's settings as an opaque mapping, because the domain must
not know that MAP-Elites exists.

`plan.digest` is a stable 16-hex-character content hash. Two plans with the same
digest describe the same run; if two runs that should compare do not hash the
same, something that affects the result is being carried somewhere else, and that
is a defect in the plan.

**What is deliberately not in the plan: worker count and shard size.** This
project measured the same evaluation work at **31.7 s on four shards of four and
93.3 s on sixteen of one** — pool shape changes wall time by 3x and the result by
nothing. Putting it in the plan would make the same experiment on a four-core and
a sixteen-core machine compare as two different configurations. It lives on the
job as `resources`.

### `TrainingState` — the runtime state

How far the job has got, right now. Rebuilt on every resume, cited by nobody.
Monotonic in `step` and `samples_seen`: a trainer reporting a step lower than the
last is either resuming into the wrong job or double-counting, and both deserve a
loud failure rather than a chart that goes backwards.

`steps_at_risk` is the gap between the current step and the last checkpoint —
which is exactly what a crash costs, and is worth reading without loading the
checkpoint.

### `Experiment` — the record

A *question*, under which one or more jobs are run. A resumed search is two jobs
and one experiment; an A/B is two experiments whose plans differ in exactly the
arm. This is what `runs/archNN_notes.md` is today, written by hand from memory
and a log.

It carries a `hypothesis`, recorded before the run, and a `conclusion`, recorded
after. The hypothesis field exists because a prediction recorded afterwards is
not a prediction.

### `TrainingJob` — one execution attempt, with a lifecycle

```
                    worker
                    starts        trainer returns
      PENDING ─────────────► RUNNING ───────────────► SUCCEEDED
         ▲   │                 │  │
         │   │ cancel          │  │ raises
         │   ▼                 │  └──────────────────► FAILED
         │ CANCELLED ◄─────────┤                         │
         │       ▲             │ pause requested         │
         │       │             ▼                         │
         │       │          PAUSING ──► PAUSED           │
         │       │             │           │             │
         │       │ cancel      ▼           │ resume      │ resume
         │       └────── CANCELLING        │             │
         │                                 │             │
         └─────────────────────────────────┴─────────────┘
                     prepare_resume, with a checkpoint
```

Four properties of this diagram are load-bearing.

**`PAUSING` and `CANCELLING` are states, not instants.** The request is made by
one process and honoured by another, at the trainer's next safe boundary — for
this project's search, up to one generation: ~74 s in steady state and ~300 s
during the opening six-island verification burst. Collapsing request and
acknowledgement into one transition means the caller cannot tell "asked to stop"
from "stopped", and the only remaining way to find out is to look for a process.
**The PID-hunting in `CLAUDE.md` is a workaround for a state that was never
modelled.**

**The application never writes `RUNNING`.** `StartTraining` returns with the job
`PENDING`, and the *worker* takes it to `RUNNING` once it is alive. Between the
launch and that moment is where `import torch` fails, where a container runs out
of memory, and where a bad `MUJOCO_GL` kills the process. A job reporting
`RUNNING` through that window would be wrong for as long as anyone believed it.
A resume goes through the same door — `prepare_resume` arms the job back to
`PENDING` — so the two paths cannot drift.

**`FAILED` is resumable.** A run that died at generation 611 has 611 generations
of paid-for simulation on disk. Refusing to continue it to keep the state machine
tidy would throw that away.

**A resume without a checkpoint is refused.** That is a restart wearing a
resume's name, and allowing it would produce a record claiming continuity across
a boundary where the learned state was reinitialised — the exact defect
`ops/checkpoint.py` was written to close, recreated one layer up. A resume may
raise the step budget; it may not change the trainer or the seed.

### `CheckpointRecord` — the artefact, described

The domain knows a checkpoint's identity and provenance. It does not know the
bytes are in an `.npz`, and it must not know some of them are torch tensors. So
the payload a store round-trips is `Mapping[str, bytes]` — the widest interface
that stays dependency-free.

`CheckpointKind` is `PERIODIC | LATEST | BEST | FINAL`, and it exists because
retention without it is either "keep everything", which fills the disk on a
21-hour run checkpointing every 10 generations, or "keep the last N", **which
deletes the best one**.

### `Dataset` — the seed corpus, named honestly

This project trains no supervised model on labelled examples. What plays the role
of a dataset is the *seed corpus*: body plans, the hand-built reference genome,
and the elites of an earlier run. That corpus has every property the port exists
for — it is named, versioned, it decides what the run can reach, and swapping it
silently makes two runs incomparable — so it is modelled as a dataset, and the
difference is written down here rather than discovered later.

`DatasetRef` requires a version, and `put` refuses to overwrite an existing ref:
a corpus that can change under a fixed `name:version` makes every plan naming it
irreproducible while still hashing to the same digest.

---

## The ports

The four the architecture is named for, plus the ones a long-running,
cancellable, restartable job turns out to need.

| port | what it abstracts | shipped adapters |
|---|---|---|
| `Trainer` | the training itself | `SearchTrainer` (MuJoCo + PPO), `SyntheticTrainer` |
| `DatasetRepository` | the seed corpus | `FileDatasetRepository` (JSONL) |
| `CheckpointStore` | a checkpoint's bytes | `FileCheckpointStore` |
| `ExperimentStore` | the record | `FileExperimentStore`, `SqliteExperimentStore` |
| `JobStore` | lifecycle state, written by two processes | `FileJobStore`, `SqliteJobStore` |
| `ControlChannel` | how a stop reaches a running worker | `FileControlChannel` |
| `JobLauncher` | process isolation | `SubprocessLauncher`, `InlineLauncher` |
| `MetricSink` | metrics as data | `FileMetricSink` (JSONL) |
| `EventLog` | job history, append-only | `FileEventLog` (JSONL) |
| `Clock` | the single source of "now" | `SystemClock` |

All are `typing.Protocol`, not ABCs — so a test fake is a short class inheriting
from nothing, and an adapter never has to import the ports package to satisfy one.

`JobStore` is a deviation from the four-port diagram, and the reason is worth
stating: an experiment is a question that may be attempted several times, and a
job is one attempt. They have different write rates — an experiment is written
twice in its life, a job's state is written every generation by a *different
process* — and folding the second into the first would put a hot, concurrently
written record inside a cold, human-read one.

### The `Trainer` port is the whole design decision

A port of the form `train(plan) -> result` is what turns training into a function
call, and it cannot express progress, checkpoints, resume, pause, cancel or a
recorded failure. So the port is:

```python
class Trainer(Protocol):
    def capabilities(self) -> TrainerCapabilities: ...
    def train(self, context: TrainingContext) -> TrainingOutcome: ...
```

`context` is what the runtime hands *inward*: `report`, `save_checkpoint`,
`resume_from`, `items`, `note`, and `stop_requested`. Control is inverted at
exactly one place — **the trainer decides *when* it is safe to stop, and the
application decides *whether* it should**. That is the correct split, because
only the trainer knows where its state is consistent and only the application
knows that someone pressed cancel.

The cooperative-stop contract, stated once:

- `context.stop_requested()` is polled by the trainer at its own safe points.
- A trainer that sees a stop **finishes the current unit of work, writes a
  checkpoint, and returns** an outcome saying which. **It does not raise** — a
  clean pause is not a failure and must not share a channel with one.
- A trainer that ignores the flag is not wrong, it is *uncancellable*, and it
  declares that in `capabilities().cooperative_stop`. The launcher then knows its
  only option is to kill the process, which costs everything since the last
  checkpoint. The capability is a promise a caller can plan against.

`TrainerCapabilities.step_unit` exists because the difference between
"generation" and "optimiser step" is three orders of magnitude and every budget
is expressed in one of them. For `SearchTrainer` a step is **one generation**.

---

## The training worker

```
Application
    │  TrainingJob
    ▼
Training Worker Process          python -m dytiscidae.worker
    │                              --job-id … --root … [--store sqlite]
    ├── Trainer (Model / Dataset / Optimizer / GPU runtime)
    ├── Checkpoints  → CheckpointStore
    ├── Metrics      → MetricSink
    ├── Events       → EventLog
    └── State        → JobStore
```

Started by `SubprocessLauncher` with `start_new_session=True`, which is what
`setsid nohup` does in this project's existing launch command and is here for the
same reason: the worker gets its own process group, so it **survives the terminal
that started it** and so a signal can be delivered to the *whole group* — the
search spawns an evaluation pool whose children outlive a signal sent to the
parent alone.

Neither launcher ever matches a process by its command line. The handle carries a
pid and the pid is what gets signalled.

### Six guarantees, each because its absence is a measured failure mode

1. **Every exit is recorded.** Normal return, cooperative stop, uncaught
   exception, SIGTERM — each lands on the job as a status with a reason. The
   worker catches `BaseException`, so even a `SystemExit` out of the trainer is
   recorded before it propagates.
2. **A failure carries its traceback into the record.** The worker log is on a
   container that gets reclaimed; the job record is on disk that is kept.
3. **Stops are cooperative and honoured at the trainer's own boundary.**
4. **SIGTERM means pause, not die.** The environments this runs in reclaim
   containers with a term signal and a short grace period. Treating it as a pause
   converts a reclamation into a resumable stop.
5. **A checkpoint is written before the worker stops for any reason it can see** —
   except a cancel, where the run is being discarded on purpose.
6. **The heartbeat is written whether or not the trainer reports.** A trainer that
   is slow and a trainer that is wedged look identical from the record otherwise,
   and this project's steps are 74–300 s apart.

The worker's exit code is the *worker's*, not the training's: `0` means the job
reached a recorded conclusion, **including `FAILED` and `CANCELLED`**. Whether the
training succeeded is the job's status, which is the point of having one.

### How a stop crosses the process boundary

Three mechanisms were available.

**Signals.** Needs the pid, does not survive the application restarting, and in
this project's own notes `pkill -f` killed the shell that ran it. Kept as a
*secondary* path — the worker treats SIGTERM as "the container is going away" —
but not as how a user's cancel is delivered.

**A queue or a broker.** Correct, and it adds a service to a project whose runs
are started with `setsid nohup`.

**A file the worker reads at its checkpoint boundary.** Chosen. It survives the
application exiting, it survives the worker restarting, it needs nothing running,
it is inspectable with `cat`, and the boundary it is read at is already the point
where the on-disk state is current.

The cost is latency — up to one generation — and the design makes that latency
*visible*, as `PAUSING` and `CANCELLING`, instead of hiding it behind a call that
appears to have taken effect. The CLI says so in words:

```
$ python -m dytiscidae.ops.run job pause job-…
job-…  pausing
  the request is posted. The trainer honours it at its next boundary -- for the
  design search that is the end of the current generation -- and the job reaches
  'paused' then, with a current checkpoint.
```

A cancel overrides a pending pause; a pause never overrides a pending cancel. A
user who asked to stop a run and then asked again, less firmly, has not
downgraded their own instruction.

---

## The search, behind the port

**`evolution/loop.py` is not rewritten, and must not be.** Its behaviour is
measured — pool shape, checkpoint cadence, island rotation, the verification
burst in generations 0–5 — and every number in `docs/ROADMAP.md` was produced by
it as it stands. Re-implementing it inside a hexagon would throw that away to
satisfy a diagram.

`adapters/trainers/search.py` translates and observes, and does no arithmetic:

1. plan hyperparameters → `SearchConfig`, from an explicit accept-list (a typo
   that silently fell back to a default is the failure `CLAUDE.md` records for
   `--min-shard`, where two generations in three ran under a trap);
2. `on_generation` → `context.report`, and `should_stop` → the job's stop flag;
3. the run directory's portable checkpoint → the `CheckpointStore`, on a cadence,
   and back again on resume;
4. how the loop ended → a `StopReason`.

**The search still owns its run directory.** `context.workspace` *is* that
directory; the loop writes `archive_*.pkl`, `search_state.pkl`, `events.jsonl`
and the portable checkpoint into it exactly as always, and the report skill, the
dashboard and the showcase keep working against it unchanged. What the adapter
adds is a digested, provenance-stamped *copy* in the store.

### The one change to the measured system

`run_search` gained an optional `should_stop` parameter, defaulting to `None`:

```python
def run_search(cfg, spec=None, on_generation=None, should_stop=None) -> SearchState:
```

It is polled once per generation, immediately after the periodic checkpoint
block, and a stop **forces a checkpoint before breaking** — so honouring one costs
at most the generation in flight and never the generations since the last
periodic write. Per generation rather than per evaluation, because a generation
boundary is where the archives, the shared policy and the RNG stream are all
consistent; stopping anywhere else would produce a checkpoint that cannot be
resumed from.

Per generation rather than on the checkpoint cadence, because at
`checkpoint_every = 10` and ~74 s a generation, the cadence alone would make a
pause take up to twelve minutes to take effect.

`should_stop=None` is exactly the behaviour every run before it existed had. The
diff is 43 lines, all additive.

---

## Reproducibility

Four things make a run reproducible, and each was absent at some point and cost a
measurement:

| carried by | why |
|---|---|
| `TrainingPlan.digest` | two runs of one configuration hash the same, and a changed arm label changes it |
| `Experiment.provenance` | git sha, **whether the tree was dirty**, python, numpy, torch, mujoco versions |
| `CheckpointRecord.provenance` | plan digest, trainer, seed, attempt number, and what it resumed from |
| the checkpoint payload | the network, Adam's moments, the RNG state, the mobility bases |

The RNG state is the one with a number attached. `ops/checkpoint.py` records it:

> arch38's archived `takeoff_height` of 2.288 m re-measured as 0.000 for exactly
> this reason.

A resume that silently restarts, or silently reinitialises its generator, looks
identical to one that works: the job says RUNNING, the steps climb, the log looks
right. `tests/test_worker.py` catches it the only way it can be caught — by
running the same seed twice, once uninterrupted and once paused and resumed, and
comparing the trajectories step by step:

```
  [ok  ] every step after the boundary matches the uninterrupted run
  [ok  ] the comparison actually spans the boundary  -- 24 steps observed after step 16
  [ok  ] the final losses are identical  -- 0.297285238461 vs 0.297285238461
```

---

## Testing, in three tiers

Measured on this machine, 2026-09-16 — checks counted as `[ok ]` lines:

| tier | file | what it uses | checks | runtime |
|---|---|---|---|---|
| architecture | `tests/test_architecture.py` | AST + an interpreter with numpy/torch/mujoco hidden | 23 | 0.7 s |
| unit | `tests/test_domain.py` | nothing — no IO, no clock, no dependency | 107 | 0.1 s |
| unit | `tests/test_application.py` | in-memory fakes (`tests/fakes.py`) | 81 | 0.1 s |
| unit | `tests/test_search_adapter.py` | a stub `run_search` with the real call ordering | 68 | 0.2 s |
| integration | `tests/test_adapters.py` | real files, real SQLite, real forked processes | 122 | 0.9 s |
| training runtime | `tests/test_worker.py` | real `python -m dytiscidae.worker` processes | 88 | 47.2 s |
| | | **total** | **489** | **48 s** |

```bash
for suite in architecture domain application search_adapter adapters worker; do
    PYTHONPATH=. python tests/test_$suite.py | tail -1
done
```

Forty-eight seconds for the lot, which is the number that decides whether they
get run. The 47 s is almost all `test_worker.py` sleeping: the pause and cancel
cases need a run that is still going when the request arrives, so the synthetic
trainer is slowed to 20 ms a step on purpose.

Two deliberate choices.

**The fakes are whole implementations, not simplified ones.** `FakeJobStore.update`
re-reads before applying the caller's change, exactly as the real one does. A fake
that skipped that would let a use case pass in tests and lose a heartbeat in
production.

**The adapter tests are contract tests.** One test body runs against both the file
stores and the SQLite stores. Two implementations of one port that are tested
separately will diverge, and the divergence is found by whoever swaps the backend.
Writing them this way found two real defects immediately: SQLite reporting a
foreign-key violation as "job already exists", and `TrainingJob.start` replacing
the worker record so that a forced cancel could not find the process it was meant
to kill.

**The concurrency test forks.** A lost update is exactly the class of bug that
survives being reasoned about, so eight processes update one job and all eight
writes must be present afterwards.

**The training-runtime tests start real subprocesses**, because pause, resume,
cancel, failure and container reclamation are all cross-process behaviours and a
test that fakes either side proves nothing about them. What makes that affordable
is `SyntheticTrainer`: it implements the whole `Trainer` contract with no
third-party dependency and its step is a microsecond, so a pause-and-resume that
would take two MuJoCo generations takes under a second.

---

## Using it

```bash
# Record the question, with the prediction, before running anything.
python -m dytiscidae.ops.run experiment new \
    --name arch39 --trainer search --steps 900 --seed 20260901 \
    --hypothesis "thrust_margin flattens the aspect-ratio drift" \
    --set batch=16 --set segment_seconds=8 --set refine_steps=2 \
    --set use_shared_policy=true

# Start it.  --workers 4 --min-shard 4 is the measured optimum; see CLAUDE.md.
python -m dytiscidae.ops.run job start --experiment arch39 \
    --workers 4 --min-shard 4

python -m dytiscidae.ops.run job status <job-id> --events
python -m dytiscidae.ops.run job pause  <job-id>
python -m dytiscidae.ops.run job resume <job-id> --extend 300
python -m dytiscidae.ops.run job cancel <job-id>          # asks; --force kills
python -m dytiscidae.ops.run checkpoints <job-id> --verify
python -m dytiscidae.ops.run job export <job-id> --to out/ --format native
```

Layout under the lab root (default `runs/lab`, or `$DYTISCIDAE_LAB`):

```
<root>/experiments/<experiment_id>.json
<root>/datasets/<name>/<version>/{dataset.json,items.jsonl}
<root>/jobs/<job_id>/job.json            lifecycle
<root>/jobs/<job_id>/state.json          runtime state
<root>/jobs/<job_id>/control.json        pending pause or cancel
<root>/jobs/<job_id>/events.jsonl        history
<root>/jobs/<job_id>/metrics.jsonl       metrics
<root>/jobs/<job_id>/worker.log          the worker's own stdout
<root>/jobs/<job_id>/checkpoints/<checkpoint_id>/…
<root>/jobs/<job_id>/…                   the trainer's own workspace
```

One directory per job, so a job is a unit that can be tarred, moved or deleted
whole — which is what `docs/PORTING.md` already assumes about a run.

### Adding a trainer

```python
class MyTrainer:
    def capabilities(self):
        return TrainerCapabilities(step_unit="epoch", cooperative_stop=True,
                                   checkpoints=True, resumes=True)

    def train(self, context):
        start = 0
        resumed = context.resume_from()
        if resumed is not None:
            _record, payload = resumed
            start = int(payload["model.bin"][:8])
        for step in range(start, context.plan.budget.max_steps):
            ...
            context.report(context.state.advanced(step=step + 1,
                                                  metrics={"loss": loss}))
            if step % 10 == 9:
                context.save_checkpoint({"model.bin": blob}, step=step + 1)
            stop = context.stop_requested()
            if stop is not None:
                if not stop.is_cancel:
                    context.save_checkpoint({"model.bin": blob}, step=step + 1,
                                            kind=CheckpointKind.FINAL)
                return TrainingOutcome(state=context.state, stop_reason=stop.reason)
        return TrainingOutcome(state=context.state)
```

Register it in-process with `adapters.trainers.register("mine", MyTrainer)`, or —
for a worker in a *different* process, which is the case that matters — by
environment:

```bash
export DYTISCIDAE_TRAINERS="mine=mypackage.trainers:MyTrainer"
```

Without the second mechanism, an out-of-tree trainer would work inline and fail
the moment the run was isolated, which is the moment it matters.
`tests/test_worker.py` exercises exactly that path.

### Choosing a store

`files` is the default, because this project's runs are started with
`setsid nohup` in containers that get reclaimed, and a directory of JSON survives
that with nothing running.

`sqlite` buys queries the directory answers by scanning — "every failed job across
every experiment" — and real transactions. It costs one writer at a time. It never
holds a checkpoint: those are megabytes of arrays, and a row holding one turns
every backup of the metadata into a copy of every network the run ever wrote.

Both produce identical results, which is checked rather than assumed:

```
  [ok  ] the trajectories are identical
  [ok  ] and so is the final checkpoint's digest  -- 69335203fb2b vs 69335203fb2b
```

---

## The ten principles, and where each lives

| | principle | where |
|---|---|---|
| 1 | Domain does not depend on PyTorch / CUDA / SQLite | `domain/`; gated by `tests/test_architecture.py` |
| 2 | The application layer holds the use cases | `application/`, one class per use case |
| 3 | Infrastructure implements the ports | `adapters/`, none imported from inside |
| 4 | Training is a job, not a function | `domain/job.py`, an eight-state lifecycle |
| 5 | Long training is isolated from the main process | `JobLauncher`; `python -m dytiscidae.worker` in its own session |
| 6 | Checkpoint / resume / cancel / failure | `worker/runtime.py`'s six guarantees; `tests/test_worker.py` |
| 7 | Configuration, runtime state and record are separate | `TrainingPlan` / `TrainingState` / `Experiment`; checked structurally |
| 8 | Metrics, logs and checkpoints are first-class data | `MetricSink`, `EventLog`, `CheckpointStore` |
| 9 | Experiments are reproducible | plan digest + provenance + the RNG state in the payload |
| 10 | Unit / integration / training-runtime test tiers | the five suites above |

---

## What this does not do

Stated plainly, because a design document that lists only what works is not one.

**It does not make the search faster, better or different.** The only change to
`evolution/loop.py` is a 43-line additive stop hook that is off by default. No
physics, no scoring, no operator, no threshold was touched.

**It does not replace `search_state.pkl` or the run directory.** `--resume` still
reads files, the dashboard still reads `generations.jsonl`, the showcase still
reads `archive_*.pkl`. The job layer wraps that; it does not supersede it.

**`python -m dytiscidae.ops.run search` still works exactly as before.** The `job`
commands are additive. Nothing that a `runs/archNN` directory depends on moved.

**There is no scheduler and no queue.** `StartTraining` launches immediately.
Multiple jobs on one machine will contend, and this project's own notes say the
pool optimum is contention-sensitive — so that is a real limitation, not an
oversight, and a `JobLauncher` that submits to a queue is the place to fix it.

**Retention is manual.** `CheckpointStore.prune` exists and nothing calls it
automatically. A 900-generation run checkpointing every 10 generations publishes
90 checkpoints of a few megabytes each; that is fine, and a policy that ran
without being asked could delete something a human wanted.

**The `SearchTrainer`'s determinism claim is narrow.** The RNG state, the network
and Adam's moments are in the checkpoint, so a resume continues the same
sequence. It does **not** claim determinism across a different worker count: the
pool shards the batch, and a different shard layout evaluates in a different
order.
