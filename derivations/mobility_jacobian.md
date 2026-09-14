# The mobility Jacobian: what is derived, what is identified

## Definition

Two different objects are called "the Jacobian" in robotics, and this project
uses the second while the literature usually means the first.

**The kinematic Jacobian** relates joint rates to end-effector twist:

    x = f(q),    xdot = J(q) qdot,    J(q) = df/dq.

It is derived from the mechanism's geometry, exactly, with no experiment.

**The mobility Jacobian** used here relates *CPG parameter offsets* to the
*mean body twist over a stroke*:

    y = J^T dp,     y in R^6,   dp in R^P,

with `P = 3n+1` for `n` joints. There is no closed form for it, because `y` is
the outcome of a fluid-structure rollout, not of a kinematic chain.

This document derives the first for the part of the project that does have a
closed form (the CPG), states why the second is identified rather than derived,
and says what that costs.

## Assumptions

For the kinematic part: none beyond the CPG's own definition.

For the mobility part:

1. The mean twist is a differentiable function of the parameters near the base
   gait.
2. The rollout is deterministic given the parameters and the initial state.
3. The window over which the twist is averaged is long enough for the mean to
   be a property of the gait rather than of the transient.

## Derivation

### Part 1: the CPG kinematic Jacobian, in closed form

The CPG commands joint `i` at time `t`:

    theta_i(t) = o_i + A_i sin(2 pi f t + phi_i + phi_0)

with parameters flattened as `p = [A_1..A_n, phi_1..phi_n, o_1..o_n, f]`.
Write `psi_i(t) = 2 pi f t + phi_i + phi_0`. Then, term by term:

    d theta_i / d A_j   = delta_ij sin(psi_i)
    d theta_i / d phi_j = delta_ij A_i cos(psi_i)
    d theta_i / d o_j   = delta_ij
    d theta_i / d f     = A_i * 2 pi t * cos(psi_i)                    (1)

so `K(t) = d theta / d p` is `n x (3n+1)`, block-diagonal in its first three
blocks with a dense last column.

Three things fall straight out of (1) and none of them are written down in the
project.

**The frequency column grows without bound in `t`.** Amplitude, phase and
offset sensitivities are bounded by `1` and `A_i`. The frequency sensitivity is
`A_i 2 pi t`, which at `t = 1.2 s` is `7.54 A_i` and at `t = 4 s` is `25.1 A_i`.
A probe that runs longer finds more frequency authority, for a purely kinematic
reason. So the identified basis depends on `probe_time`, and two paths that use
different probe windows are not identifying the same object.

**The derivative is not the response.** `K` is the sensitivity of the
*instantaneous joint angle*; the mobility Jacobian is the sensitivity of the
*mean twist over the window*. They compose through the dynamics, and the mean
over a window of a quantity whose sensitivity grows linearly is itself
`O(t)`-weighted toward the end of the window.

**Phase and amplitude sensitivities are in quadrature.** `sin` and `cos` are
orthogonal over a cycle, so amplitude and phase perturbations on the same joint
excite orthogonal directions in the probe. This is a reason the probes condition
well, and it is free — it comes from the CPG's form, not from the probe design.

### Part 2: why the mobility Jacobian is not derived

To write `J` in closed form one needs `d(mean twist)/d(parameters)`, which
requires differentiating through:

* the joint command (1) — closed form, above;
* the actuator's torque response to a position target — a PD law, closed form;
* MuJoCo's constrained rigid-body dynamics — differentiable in principle,
  through `mjd_transitionFD` or an analytic `mjd_*`, but not exposed here;
* this project's own fluid loads — closed form per strip, but with a `where`
  on stall regime, a clip on the force limiter, and a submerged-fraction ramp,
  so piecewise;
* contact — non-smooth, and a machine on the beach is in contact.

The composition is differentiable almost everywhere and has measure-zero kinks
at the stall blend, the force clamp and every contact event. A finite-difference
identification is the appropriate estimator. That is what
`identify_mobility` is, and it is the right choice — this section exists to
record that it is a choice, with a cost, and not an absence of ability.

### Part 3: what the finite-difference identification is, precisely

`identify_mobility` draws `n_probes = 24` directions `d_k ~ N(0, 0.35^2 I_P)`,
runs each with both signs, and central-differences:

    y_k = 0.5 * ( g(+d_k) - g(-d_k) )                                  (2)

where `g` is the mean twist over a 1.2 s rollout. Then `J` solves
`min_J ||X J - Y||_F` with `X` the stacked `d_k`.

(2) is a **central difference along a random direction**, which is a
directional derivative estimate with error `O(||d||^2)` — the even-order terms
cancel. That is why it beats a one-sided probe, and the docstring says so in
different words ("cancels drift that is independent of the command"): a
constant offset in `g` cancels, and so does every even-order term in the Taylor
expansion, including the leading curvature.

The residual of the least squares is therefore a direct estimate of how
nonlinear the map is at this probe scale. `experiments/rank_threshold` records
it per body and medium as `residual_fraction`.

## Dimensional analysis

`K = d theta / d p`:

* `d theta / d A` — rad per rad — dimensionless.
* `d theta / d phi` — rad per rad — dimensionless.
* `d theta / d o` — rad per rad — dimensionless.
* `d theta / d f` — rad per Hz — `s`, from the `2 pi t` factor.

So `K`'s last column carries seconds and the rest are dimensionless: `K` is not
dimensionally homogeneous, and `||K||` has no units. The same is true of `J`,
whose parameter side is the same space. This is not an error — it is an
unavoidable consequence of a parameter vector that mixes a frequency with a set
of angles — but it means `modes` being a *unit vector* in that space is a
statement without physical content, and the relative weighting between "move
the frequency by 1 Hz" and "move a phase by 1 rad" is set by that arbitrary
norm. Recorded as **C-01**.

The twist side carries the `0.3 m` implicit length discussed in
`svd_control_basis.md`.

## Numerical implementation

`TriphibianEnv.identify` (`dytiscidae/envs/triphibian.py`) and
`identify_batch` (`dytiscidae/envs/batchroll.py`), which must agree. Both
default to `probe_time = 1.2`, `n_probes = 24`, `max_modes = 6`. CLAUDE.md
records that they were once 8/4 against 24/6 — the same failure this document's
Part 1 predicts would matter.

`experiments/mobility_data.collect_one` reproduces the probe loop so several
experiments can share one collection, and `verify_against_env` asserts the
refit matches `TriphibianEnv.identify` exactly. Measured: `max_abs_sigma_diff`
= 0.0 on beetle/water/seed 1, i.e. bit-identical, not approximately equal.

## Validation

* **The CPG kinematic Jacobian (1) against central differences through the real
  `CPG.command`**: `experiments/analytic_vs_numerical`, check A. Maximum
  relative element error `5.7e-10` over five sample times, against a tolerance
  of `1e-6`. The frequency column's growth is measured there too: 84.3x from
  `t = 0.05 s` to `t = 4.0 s`, against the 80x that exact linearity predicts.
* **The identification recovers a known basis**: `tests/test_search.py::
  test_mobility_recovers_known_basis`.
* **Probe-loop reproduction**: `experiments/mobility_data.verify_against_env`.
* **Linearity residual per body**: `experiments/rank_threshold`.

## Failure conditions

* **Contact.** A machine touching the beach has a non-smooth `g`, and a central
  difference across a contact transition estimates the derivative of something
  that does not have one. Land bases should be read with that in mind.
* **Probe scale.** `0.35` in CPG units is a fixed absolute step in a space that
  mixes radians with hertz, so it is a large step in frequency (0.35 Hz on a
  2 Hz base, 17%) and a modest one in phase. Nothing has swept it.
* **Underdetermination.** `3n+1 > 24` for more than 7 joints, so for most of
  the fleet `J` is the minimum-norm solution of an underdetermined system, not
  a measurement of the full map.
* **Instability.** `step_fn` returns a zero twist when the rollout goes
  non-finite, so a diverged probe contributes a *measured zero response* rather
  than a missing observation, and the fit treats it as evidence that the
  direction does nothing. This is the "a quantity that means 'I could not
  measure this' must not share a value with one that means 'I measured zero'"
  rule the project has already learned once, still present here. Recorded as
  **C-06**.
