# What was decided because there was no GPU

Everything in this project was built in a cloud container with four CPU cores
and no accelerator. That constraint is written into the code in more places
than it looks, and several of those places are *documented decisions with
stated reasons* — which makes them easy to mistake for settled design when they
are really expedients with an expiry date.

The constraint has expired. The machine this now runs on has 20 cores, an RTX
3060, and a Mojo GPU port of the fluid solver underway. Every entry below is
outstanding work, not a design principle.

The test for inclusion is narrow: the source says, or the history shows, that
the choice was made **because of compute cost**. Things that are simply
unfinished belong in `AUTONOMY.md`; things that are wrong belong in the README's
corrections table. This file is only for decisions that a GPU changes.

---

## 1. ~~The controller is not trained at all — and is never executed either~~ RESOLVED 2026-08-05

`controller_refine_steps: int = 0` — [`evolution/loop.py`](../dytiscidae/evolution/loop.py)

A child's controller is its parent's weights, and no optimisation happens in the
loop. The 500-generation runs (`arch19`–`arch28`) all ran this way, so **no
controller learning occurred in them at any point**. Morphology evolved; control
did not.

This entry originally described the field as a dial set to zero. It is not a
dial. `controller_refine_steps` occurs exactly once in the codebase — the
declaration above — and nothing reads it. Setting it to 20 changes nothing at
all. The loop's own module docstring says "the child's controller is inherited
and locally refined", and the second half of that sentence has never been true.

That distinction decides how much work item 1 is. It is not a config change to
be made before the next run; it is writing the refinement loop, choosing what it
optimises against, and paying for it in evaluations.

### The policy is not merely untrained; it never runs

Following the call chain end to end, at default settings the `Policy` object is
built, inherited, stored in the archive — and never evaluated:

| step | file | what it does |
|---|---|---|
| `identify_axes_every: int = 1` | `loop.py:77` | default |
| `identify = (counter % max(cfg.identify_axes_every, 1)) == 0` | `loop.py:661` | `% 1` is always 0, so always `True` |
| `identify=any(b[2] for b in built)` | `loop.py:670` | one candidate wanting it forces the whole batch |
| `controllers=[None if identify else c for c in ctrls]` | `loop.py:311` | so every controller becomes `None` |
| `ctrl = controller or Controller(params=env.cpg.base)` | `evaluate.py:161` | a fresh `Controller`, whose `policy` field defaults to `None` |
| `policy=ctrl.policy` | `evaluate.py:177` | `None` reaches `env.rollout` |

There is one `evaluate_candidates` call per generation and no second pass, so no
evaluation anywhere in the search ever executes a policy. `_controller_for`'s
careful weight-inheritance logic — matching shapes, falling back to zeros — runs
on an object that is then discarded at the rollout boundary.

So item 1 is two defects, not one. Writing the refinement loop alone would
optimise weights that the evaluation still ignores. The `identify`/controller
coupling has to be separated first: identifying mobility axes and running a
policy are independent things that this flag currently forces to be mutually
exclusive.

This is the largest single item here, and it is not a subtle one: the search has
been selecting bodies on the strength of an untrained controller, which
systematically favours designs that work *without* control. Since the policy is
not even executed, that is not an approximation of the effect — it is exactly
the effect.

**What a GPU changes:** the reason to leave it unwritten was that each
refinement step costs a full Tier-1 evaluation. That is the cost the port
removes — a generation of candidates now evaluates in one set of kernel
launches, so refinement steps batch the same way.

### Resolution (2026-08-05, `ffb61d4`)

All three defects fixed together, because none of them is useful alone:

1. `identify` no longer gates the controller. It means only "identify axes".
2. `Controller(params=None)` is filled from the body by both evaluation paths.
   Nothing filled it before; honouring the controller turned that into an
   `AttributeError` on the first run, which is how it surfaced.
3. `controller_refine_steps` is implemented as a (1+1)-ES — perturb, evaluate,
   keep if the mission fraction improved. One perturbed vector per candidate
   makes a refinement step exactly one batched evaluation.

A refinement step is far cheaper than a full evaluation because it does not
re-identify axes, and identification is the sequential per-machine part that
dominates. How much cheaper depends on the policy width, which is easy to
measure at one width and then over-generalise:

| | s / refinement step |
|---|---|
| `policy_hidden=0`, measured in isolation | 8.4 |
| `policy_hidden=16`, measured in `arch29` | 21.7 |

against ~79.6 s for a full generation at batch 16. The second number was
recovered from the running search — generations 1 and 2 took 223 s and 200 s
against a predicted 130 s. Quoting the first number as if it were a property of
refinement, rather than of refinement at one width, put a run's projected
duration out by 60%.

Measured, three seeds at `segment_seconds=0.4`:

| | mission_fraction |
|---|---|
| `steps=0` | 0.093049, 0.101488, 0.012699 (all weights zero) |
| `steps=4` | 0.093049, 0.109723, 0.056915 |

## 2. ~~The policy is linear~~ RESOLVED 2026-08-05 (`--policy-hidden`)

`Policy(hidden=0)` — [`control/cpg.py:247`](../dytiscidae/control/cpg.py)

The docstring is explicit that this is a budget decision, not a capacity
judgement:

> `hidden = 0` makes it linear … that is the default, because the one hidden
> layer that used to be the default was not affordable. At `hidden = 16` this
> is 308 weights, and CMA-ES on 308 dimensions wants some thousands of
> evaluations; the default training budget was 180.

It also records the measurement that forced it: five iterations moved the
population best from 0.365 to 0.349, "not a controller failing to learn, it is
an optimiser that has barely been asked a question."

### Resolution (2026-08-05, `9de9a3f`)

`--policy-hidden` reaches the field. Re-opened and measured, three seeds:

| steps | hidden=0 | hidden=16 |
|---|---|---|
| 4 | sum 0.25969 | sum 0.24411 |
| 12 | sum 0.27008 | **sum 0.28145** |

The capacity is worth having, but only past a step budget: 308 weights need
more samples than 60 before they pay. `--policy-hidden 16` with too few
`--refine-steps` is worse than staying linear. This is a real coupling between
two flags, not a free upgrade.

**Still outstanding:** the second half of the original question — whether an ES
is the right optimiser at 308 weights, or whether item 3's policy-gradient
route is. The crossover measured above is evidence that it is being asked at
the width where the answer starts to matter.

## 3. CMA-ES instead of a policy-gradient method

[`control/train.py:11`](../dytiscidae/control/train.py), "Why CMA-ES rather than a
gradient method".

The conclusion is defensible at 60–130 weights. One of its stated reasons is
not:

> A gradient method would need all three to be differentiable

That is true of analytic policy gradients through a differentiable simulator. It
is **not true of PPO or any model-free policy-gradient method**, which need only
the policy to be differentiable and estimate the gradient from sampled returns.
The fluid solver, the contacts and the energy model can stay exactly as
non-smooth as they are.

**Outstanding:** the choice needs re-deciding on its real merits (sample
efficiency at the new width and budget), and the docstring's reasoning needs
correcting either way.

## 4. The critic is ridge regression, not a network

[`evolution/critic.py:55`](../dytiscidae/evolution/critic.py), "Why ridge
regression rather than a network":

> Four CPU cores, shared with the evaluations that generate the training data,
> and a few thousand labelled examples over a multi-day run.

Two of the three reasons are compute. The third is not, and is worth keeping:
a linear model **can be read**, so the run can say which measurement the critic
has learned to distrust. Any replacement should preserve that, or explain what
replaces it.

The file already anticipates this: "The interface takes any object with `fit`
and `predict`, so this is a starting point and not a commitment."

## 5. Descriptors are PCA, not an autoencoder

[`evolution/descriptors.py:19`](../dytiscidae/evolution/descriptors.py), "Why PCA
rather than an autoencoder":

> An autoencoder is the usual choice and is strictly more expressive. On four
> CPU cores it is also several minutes of training per refit, competing with
> the evaluations that produce the data.

`AUTONOMY.md` §3 already names AURORA (Cully 2019) as the target, so this is the
one item that is on two lists. Same escape hatch: "The interface below takes a
projector, so swapping in an autoencoder later changes one class and nothing
else."

## 6. Fidelity is dialled down by cost, not by physics

`SearchConfig` — [`evolution/loop.py:64`](../dytiscidae/evolution/loop.py)

> The defaults are tuned for a four-core CPU with no GPU … `segment_seconds` is
> the main cost/fidelity dial: halving it roughly halves the run time and
> roughly doubles the variance of every Tier-1 score.

Every run so far used `segment_seconds=8`. The variance that buys is real and it
is paid by every score in the archive. Tier-2 is likewise gated to promoted
elites only because it costs ~10× Tier-1.

**Outstanding:** once evaluation is cheaper, `segment_seconds` and `tier2_every`
should be re-chosen against the variance they control rather than against the
clock.

## 7. `torch` and `ray` are commented out, pointing at a file that never existed

[`requirements.txt`](../requirements.txt) carries:

```
#   torch          : only needed for the PPO learner (learning/ppo.py).
#   ray            : only needed for multi-machine parallel evaluation.
```

`dytiscidae/learning/` does not exist and never did. So the note is not
describing an optional component — it is describing an intention that the
compute budget prevented, left in the file as a comment. Item 3 above is that
intention.

## 8. The search is single-process

There is no `multiprocessing`, no worker pool, and no use of the batch structure
anywhere in the evaluation path — verified by grep. A generation's candidates
are evaluated one after another. On four cores that was a reasonable thing not
to have built; on twenty it is throughput left on the floor, and it is also the
precondition for GPU batching (see `mojo/README.md`: below ~1600 panels the GPU
loses to numpy, and one machine is ~70).

---

## What is *not* on this list

Worth stating so the list stays honest.

**The fluid solver being on the CPU** was never a compute decision — MuJoCo is a
CPU C library and the solver has to hand it forces every step. That is being
addressed by the Mojo port for a different reason: the solver is dispatch-bound,
and 233 µs of every ~340 µs step is Python and numpy per-call overhead, which no
device change removes.

**Quasi-steady aerodynamics, rigid spars, the scalar added-mass coefficient**
and the other entries under the README's "Known limits" are modelling
simplifications. Some are expensive to lift, but they were chosen for what they
approximate, not for what they cost.

---

## Why GPU utilisation stays in single digits (measured 2026-08-10)

The fluid solver is on the GPU and validates to 3e-16. Mobility identification
is now batched through it too, and the per-body velocity gather is vectorised.
After all of that, `nvidia-smi` sampled at its 1 Hz refresh over 100 s of a real
search reads: **0% of seconds fully idle** (down from 30%), mean 6.2%, max 8%.
The GPU is continuously engaged and almost entirely unloaded.

That is not a defect left to fix. It is what the measured split implies:

| part of a batched step | share |
|---|---|
| `mj_step` + energy, CPU | 60% |
| `bf.apply` | 37% (of which the GPU kernel is 11% of the whole step) |
| observation / prep | 3% |

Raising utilisation requires moving rigid-body integration to the GPU, i.e.
MJX. MJX cannot `vmap` across different kinematic trees, so the only route is to
group candidates by topology and vectorise within a group. **Measured, and it
does not work:** mutating archive parents into generations of 16 gives 13, 16
and 14 distinct topologies in three trials, largest group 3, nearly all
singletons. Grouping would produce ~1.1 machines per group.

So the honest position is that this architecture is CPU-bound in MuJoCo, and the
GPU port has taken everything that was takeable. Two things could change it, and
neither is a tuning exercise:

- MuJoCo Warp (`mjwarp`), which is designed for heterogeneous models. **Not
  evaluated** — do not assume it helps until measured.
- A custom GPU rigid-body integrator, which is a far larger project than the
  fluid port was.
