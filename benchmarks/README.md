# benchmarks/ — where the simulator stops agreeing with an answer you can write down

```
PYTHONPATH=. python benchmarks/run.py
```

Not "do I trust the simulator". **At which layer of modelling does it start
departing from a reference, and by how much.** Each layer adds exactly one
piece of physics to the one below it, and each carries a reference that does
not come from the simulator.

Exit code is always 0. A departing layer is a measurement, not a broken run —
`tests/test_math.py` is where an invariant that must hold is asserted; this is
where the fidelity boundary is located.

## The ladder, as measured

| layer | adds | verdict | worst error |
|---|---|---|---|
| 1 rigid body | Newton-Euler only | **holds** | see below |
| 2 added mass | entrained fluid inertia | **DEPARTS** | 0.729 |
| 3 drag | quadratic pressure drag | **holds** | 6.1e-5 |
| 4 buoyancy | hydrostatic lift, free-surface ramp | **holds** | 6.8e-4 |
| 5 jet propulsion | momentum flux from a cavity | **holds** | 3.5e-4 |
| 6 articulated body | internal joints | **DEPARTS** | 0.34 |
| 7 fluid surrogate | the assembled blade-element force | **holds** | 1.6e-12 |
| 8 controller | the identified mobility basis | **DEPARTS** | 0.341 |

The first departure is at **layer 2**. Everything above it inherits that error
whatever else is true of it.

## What each reference is

Three kinds, and the distinction is the point:

* **Closed form** — the answer written down from a definition. Layers 1, 2, 4,
  5, 7.
* **A tight Runge-Kutta integration of the same force law**, written out
  independently at `rtol 1e-11`. Used where no closed form exists (layer 3).
  This separates an error *in the force law* from an error *in the
  integration* — two failures that look identical in a trajectory and need
  completely different fixes.
* **A conservation law** — momentum, energy, symmetry of `M`. Layers 1, 5, 6.

Layer 8 is the exception and says so: there is no closed form for what a real
machine does, only the identified model's own claim, so that layer measures
*linearity* (which has an exact reference — doubling a command must double the
response) and reports the prediction error without grading it.

## Layer 1 — the integrator, measured rather than assumed

The exact relations hold to machine precision: `a = F/m` to 1e-12, `alpha =
I^-1 tau` to 1e-10, the box inertia formula exactly.

What is worth measuring is the **order of accuracy**, because that is what
separates "the model is wrong" (error flat in `dt`) from "the integrator has
not converged" (error proportional to `dt^p`):

| | measured p |
|---|---|
| ballistic flight | **1.000000000** |
| free-rotation angular-momentum drift | **1.0023** |

Exactly first order, which is the correct answer for MuJoCo's semi-implicit
schemes. At the project's own `dt = 0.002`, one second of free flight
accumulates **9.81 mm** of height error — exactly `(1/2) g dt T`, so it is
systematic rather than noise: it biases every height the same way and grows
linearly with the segment length. That is 3.4% of the 0.29 m cap the `clears`
rung sits above, before any physics is modelled at all.

## Layer 2 — added mass, and two errors that partly cancel

The analytic chain is established first, with no simulation in it: for a thin
plate, strip theory gives `m_normal = rho pi c^2/4 * b`,
`m_chordwise = rho pi t^2/4 * b`, `m_spanwise = 0`, so the normal-to-chordwise
ratio is `(c/t)^2` exactly — verified at c/t = 10, 100 and 1000.

Then the simulation, on a 1.0 x 0.2 m, 2 mm plate, 12 kg dry, in seawater:

| | chordwise | spanwise | normal |
|---|---|---|---|
| reference | 12.003 kg | 12.000 kg | 44.201 kg |
| **jointed** | 44.201 ✗ | 44.201 ✗ | 44.201 ✓ |
| **jointless** | 12.000 ✓ | 12.000 ✓ | 12.000 ✗ |

Two separate findings, and notice what they do together:

* **jointed** — the wing branch applies the plate's *normal* added mass in
  every direction (`MATH_AUDIT` **F-03**). Edgewise it is wrong by `(c/t)^2`.
* **jointless** — MuJoCo marks a body `simple` when no joint attaches to it and
  takes those DOFs' mass from the compile-time `dof_M0`, so the runtime
  `body_mass` edit never reaches `M` (**F-01**). No added mass at all.

The jointless row "passes" chordwise and spanwise **only because applying none
is nearly the right answer for those two directions.** Two bugs cancelling is
why neither showed up in a test that read `body_mass` back.

## Layers 3 and 4 — what a correct layer looks like

Worth stating, because an audit that only reports faults teaches nothing about
what "agrees" means here.

**Drag** reproduces a tight RK45 integration of its own force law to **6.1e-5**
over 6 s of accelerating from rest toward terminal velocity, and lands within
0.003% of the terminal velocity solved by bisection on the same law.

**Buoyancy** is exactly `rho g V` when submerged. The equilibrium draft of a
floating body matches `d* = 2h(m/(rho V) - 1/2)` — the closed-form solution of
the project's own linear submergence ramp — to 0.05%. And the heave period
matches `2 pi sqrt(M/k)` with `k = rho_water g V/(2h)` to **0.07%**, but only
once the reference accounts for something subtle and correct in the solver:
the entrained mass of a *partly* submerged element is computed at the
**blended** density the element sees, not at the density of the water below it.
Getting that wrong in the reference made this check fail by 11% until the
reference was fixed. The solver was right.

## Layer 5 — the jet

Two statements about the same jet, both analytic: the work the muscle must do,
`p Q = (1/2) rho Q^3/A^2`, and the kinetic energy the jet carries,
`(1/2) m_dot v_e^2`. They agree exactly, which pins the reference.

The graded check is then the identity the pumping load is built on — the torque
it puts on the driving joint times the joint rate is the power the jet carries
away. `JetSet.actuator_work` integrates to **14.9418 J** against **14.9469 J**
formed independently from the joint history: **3.5e-4**. And the actuator's
signed mechanical work covers it, because a propulsor cannot return energy to
its motor.

This layer read `DEPARTS 1.0` before **J-01** was fixed, when the actuator was
charged 0.0111 J of 141.61 J.

**One number here is reported and not graded**, and the distinction matters.
Comparing the actuator's bill across a run with the load and a run without it
gives `+4.96 J against 14.95 J pumped, +33%` — and that is not a failure. The
two runs follow different trajectories, because the load is what changes how
the servo tracks, so the bell's own inertial work differs too; and
`abs(tau * omega)`, the energy budget's own convention, charges braking as if
it were driving, so it does not close either. An apples-to-apples version of
that comparison does not exist. The identity does, which is why it is the one
with a tolerance on it.

## Layer 6 — a machine flapping in a vacuum

A free-floating machine driven only by its own joints cannot move its centre of
mass. This is the law every swimming and flying score depends on: a
displacement not paid for by a fluid force is not locomotion.

The model obeys it. The synthetic single-hinge case drifts at **order 1.028**
in `dt`, and RK4 at the same step reduces the drift by a factor of **402 000**,
which is what proves the drift is the scheme and not the momentum balance.

The fleet, at the project's own `dt = 0.004`, over an 8 s segment, with no
fluid, no gravity and no contact:

| plan | drift | as a fraction of a full-marks land segment | order |
|---|---|---|---|
| medusa | 0.59 mm | 0.012% | 1.01 |
| gannet | 1.93 mm | 0.040% | 1.00 |
| beetle | 16.29 mm | 0.339% | 1.01 |
| **teal** | **857.80 mm** | **17.87%** | **0.66** |

Three of four are negligible and converge at first order, so a smaller step
fixes them. **teal accumulates 0.86 m of displacement in a vacuum**, and its
order of 0.66 means the trajectory is changing under refinement rather than
converging — a smaller step does not fix it. Recorded as **N-01**.

This is the floor under the project's own recorded lesson that "switching the
actuators off changed `land_speed` by one percent": for at least one seed plan,
part of a scored displacement has no physics in it at all.

## Layer 7 — the surrogate assembles correctly

The assembled force matches `q S CL` and `q S CD` evaluated by hand from the
same coefficient functions, to **1e-12**; the glide relation `tan(gamma) = D/L`
holds to 7e-13; and doubling the speed with Reynolds number held fixed
quadruples the force exactly. Whatever is wrong with the *coefficients* — and
`MATH_AUDIT` lists several things — the machinery that turns them into forces
is right.

## Layer 8 — direction yes, gain no

| | reference | measured |
|---|---|---|
| `twist_of(c)` equals `A^T c` | 0 | 0 (exact) |
| doubling the command doubles the response | 2.0 | **1.318** |
| and does not change its direction | 1.0 | **0.9975** |
| prediction error against the basis | — | 0.263 (reported, not graded) |

The identified basis gets the **direction** of a body's response right to
within 0.25%, and the **gain** wrong by 34% once the command is doubled beyond
the probe scale. So the basis is usable for choosing which way to push and not
for predicting how hard, which is consistent with `experiments/rank_threshold`
finding the twist-side directions reproducible across probe seeds and the
parameter-side ones not.

## Adding a layer

A layer belongs here if it adds **one** piece of physics and has a reference
that does not come from the simulator. Put it in `benchmarks/layers.py` as a
function returning a `Layer`, add it to `LADDER` in `run.py`, and say in the
`reference_kind` field which of the three kinds of reference it is. If you
cannot name the reference, the check is not ready.
