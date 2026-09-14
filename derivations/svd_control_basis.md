# The SVD control basis: what `modes`, `effects` and `authority` are

## Definition

A machine's controller acts by perturbing `P` CPG parameters. The machine
responds with a body twist `y in R^6` — three linear rates and three angular
rates. Near a base gait the response is approximately linear:

    y = J^T dp,     J in R^{P x 6}     (the project's orientation; see below)

The **control basis** is the singular value decomposition of `J`. Its purpose
is to answer "what can this body actually do", in a form that does not assume
the body has a nose, a tail, or three decoupled moment axes.

## Assumptions

1. **Linearity.** The mean twist over a probe window is a linear function of
   the parameter offset. This is a local assumption around one base gait, and
   the probe scale (0.35 in CPG units) decides how local.
2. **Time invariance.** The same offset produces the same mean twist whenever
   it is applied. This fails whenever the machine's state at the start of the
   probe matters, which is why the probes are reset between runs.
3. **The mean over the window is the right response variable.** A machine that
   surges forward and back within the window reports zero.
4. **The 2-norm on `R^6` is meaningful** — i.e. that a linear rate and an
   angular rate can be added in quadrature. This one is not free; see
   *Dimensional analysis*.

## Derivation

### Orientation, because it is the thing most often stated backwards

Let `dp in R^P` be a parameter offset and `y in R^6` the twist it produces.
There are two conventions and they are transposes of each other:

* **Column convention.** `y = J_col dp` with `J_col` of shape `6 x P`. Then
  `J_col = U S V^T` with `U` `6 x k` spanning *twist* space and `V` `P x k`
  spanning *parameter* space. The columns of `V` are the parameter directions;
  the columns of `U` are the motions they produce.
* **Row convention.** `Y = X J` with `X` the `(n_probes, P)` matrix of offsets,
  `Y` the `(n_probes, 6)` matrix of responses, and `J` of shape `P x 6`. Then
  `J = U S V^T` with `U` `P x k` spanning *parameter* space and `V` `6 x k`
  spanning *twist* space. The roles of `U` and `V` are swapped.

Both are correct. Mixing them is not, and the module docstring of
`dytiscidae/control/cpg.py` states the column convention —

> Take the SVD, `J = U S V^T`. The leading columns of `V` are the parameter
> directions that move the machine most; the matching columns of `U` describe
> what motion each one actually produces

— while `basis_from_probes` fits and decomposes in the row convention, where
`U` is the parameter side and `V` the twist side. The code is self-consistent:

```python
J, *_ = np.linalg.lstsq(deltas, Y, rcond=None)       # (P, 6)
U, S, Vt = np.linalg.svd(J, full_matrices=False)     # U:(P,k) S:(k,) Vt:(k,6)
modes   = U[:, :r].T                                 # (r, P)  parameter side
effects = Vt[:r]                                     # (r, 6)  twist side
```

so nothing computes the wrong thing; the prose describes the transpose of what
runs. Recorded as **C-05**.

### What the three stored arrays are

With `J = U S V^T` in the row convention and `r` retained modes:

    modes[i]     = U[:, i]        a unit vector in parameter space
    effects[i]   = V[:, i]        a unit vector in twist space
    authority[i] = S[i]           twist produced per unit of parameter offset

and the forward model is, for a coefficient vector `c in R^r`,

    dp  = modes^T c                                                   (1)
    y   = J^T dp = (U S V^T)^T U^T c = V S U^T U c = V S c
        = (effects * authority[:, None])^T c = A^T c,   A = effects*authority.

which is what `MobilityBasis.twist_of` computes. Note that (1) uses `modes^T c`
and the result is exact only because `modes` has orthonormal rows — `U^T U = I`
for the retained columns. That orthonormality is checked in
`tests/test_math.py`.

### What the singular values mean, and what they do not

`S[i]` is the gain of mode `i`: the twist norm produced per unit norm of
parameter offset along `modes[i]`. Two standard facts that the project uses
without stating:

* **Maximum reachable twist on one axis.** For `||c|| = 1`,
  `max |(A^T c)_j| = ||A[:, j]||_2` by Cauchy-Schwarz, attained at
  `c = A[:, j]/||A[:, j]||`. `MobilityBasis._inverse` returns exactly
  `np.linalg.norm(A, axis=0)` as `reach`, so `reach[j]` is genuinely the most
  axis-`j` twist this body has. Correct, and undocumented.
* **Truncation is optimal.** By Eckart-Young, keeping the `r` largest singular
  values gives the best rank-`r` approximation of `J` in both the 2-norm and
  the Frobenius norm, with error `S[r]` and `sqrt(sum_{i>r} S[i]^2)`
  respectively. So `max_modes=6` is not an arbitrary truncation: it is the best
  6-dimensional summary of whatever `J` turned out to be. Since `J` has only 6
  columns, `k <= 6` always, and `max_modes=6` truncates nothing at all.

### Rank

`MobilityBasis.rank` returns `#{i : S[i] > 0.08 * S[0]}`. Numerical rank is
properly defined against a tolerance, and there are three candidate
tolerances, in increasing size:

1. `eps * max(P, 6) * S[0]` — floating point. About `1.3e-14 * S[0]`.
2. `||dJ||_2` — the perturbation the identification's own noise puts on `J`. By
   Weyl's inequality, `|S_i(J) - S_i(J + dJ)| <= ||dJ||_2`, so no singular value
   below this is resolvable from a single identification.
3. "the mode is worth commanding" — an engineering threshold, which depends on
   what commanding it costs in joint travel.

`0.08` is a candidate for (3) with no measurement behind it, presented as (1).
`experiments/rank_threshold` measures (2) directly by identifying each body
four times with different probe directions.

## Dimensional analysis

`J` maps CPG parameters to body twist, and neither space is dimensionally
homogeneous.

**Parameter space** is `[amplitude(n), phase(n), offset(n), frequency(1)]`.
Amplitude, phase and offset are radians — dimensionless. Frequency is hertz —
`1/s`. So `||dp||_2` adds a `1/s` to `n` dimensionless numbers. `modes` being a
unit vector in that space is a statement with no units.

**Twist space** is `[vx, vy, vz, wx, wy, wz]` — `m/s` and `rad/s`. Adding them
in quadrature requires a length. `basis_from_probes` supplies one:

```python
scale = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
Y = responses * scale
```

The three `0.3`s must be **metres** for the norm to be dimensionally sound.
They are an implicit reference length: the basis is computed in a space where
"1 rad/s of roll" counts the same as "0.3 m/s of surge", i.e. where the machine
is treated as having a 0.3 m characteristic radius. The comment says only

> Scale twist components so that rotation and translation are comparable

which states the purpose and not the quantity. Everything downstream inherits
it: every singular value, `reach`, `cond(A)`, the damping `lam = 0.01 tr(G)/r`,
and the rank threshold `0.08 * S[0]`. It is also a *fixed* length applied to
bodies whose actual sizes differ, so the same machine scaled by two gets a
different basis for a reason that is not physics. Recorded as **C-02**.

## Numerical implementation

`basis_from_probes`, in full:

1. `J = lstsq(deltas, responses * scale)` — an ordinary least squares over
   24 probes for 3n+1 parameters. For a machine with more than 7 joints,
   `3n+1 > 24` and the system is **underdetermined**: `lstsq` returns the
   minimum-norm solution, which is a particular choice among infinitely many
   `J` that fit the probes exactly. The retained `J` then lies entirely in the
   24-dimensional row space of `deltas`, and everything outside it reads as
   zero authority regardless of the machine.
2. `svd(J, full_matrices=False)`.
3. `modes = U[:, :r].T`, `effects = Vt[:r]` normalised to unit rows,
   `authority = S[:r]`.

Point 1 is worth checking against the fleet: `n_probes = 24` against `3n+1`
means a 24-joint machine (`MAX_PARTS = 24`) presents 73 parameters to 24
probes. `experiments/rank_threshold` records `n_params` per plan.

## Validation

* **Reconstruction is exact**: `tests/test_math.py` rebuilds `J` from
  `modes`, `effects` and `authority` and agrees to 3.2e-15.
* **Orthonormality of `modes`, unit rows of `effects`, descending
  non-negative `authority`**: same test, all exact.
* **A known basis is recovered**: `tests/test_search.py::
  test_mobility_recovers_known_basis` builds a synthetic `step_fn` with a known
  answer and checks the identification finds it. That test predates this
  document and is the right shape.
* **The rank threshold against the noise floor**:
  `experiments/rank_threshold`, whose README carries the measured numbers.

## Failure conditions

* **Beyond the linear range.** The basis is a first-order model at one base
  gait. `probe_scale = 0.35` decides how far "near" is and has not been swept.
* **Underdetermined fits.** More than 7 joints and `J` is a minimum-norm
  choice rather than a measurement — see above.
* **Probe duration enters the frequency column.** `d(theta)/d(frequency) =
  A 2 pi t cos(psi)` grows linearly in `t`, so the frequency parameter's
  authority scales with `probe_time`. Two identification paths with different
  `probe_time` do not produce comparable bases. Both paths currently default to
  1.2 s; nothing asserts that they must agree. Measured in
  `experiments/analytic_vs_numerical`, check A.
* **Mode index is not an interface.** Mode `i` means an unrelated thing on
  different bodies, and the sign of a singular vector is arbitrary. The project
  already knows this — it is why a shared policy commands in twist space — and
  `tests/test_search.py` pins it.
* **A body that cannot move.** All singular values zero, `rank = 0`, `reach`
  all zero, and every command is zero. Handled, and tested.
