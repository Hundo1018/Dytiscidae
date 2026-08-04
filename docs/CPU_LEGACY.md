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

## 1. The controller is not trained at all

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

This is the largest single item here, and it is not a subtle one: the search has
been selecting bodies on the strength of an untrained controller, which
systematically favours designs that work *without* control.

**What a GPU changes:** the reason to leave it unwritten was that each
refinement step costs a full Tier-1 evaluation. That is the cost the port
removes — a generation of candidates now evaluates in one set of kernel
launches, so refinement steps batch the same way.

## 2. The policy is linear

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

**Outstanding:** with the evaluation budget raised, re-open the hidden layer, and
separately ask whether CMA-ES is still the right optimiser at that width.

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
