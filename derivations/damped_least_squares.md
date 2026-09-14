# Damped least squares, and what `lam` buys

## Definition

Given a linear map `M : R^n -> R^m` and a target `b in R^m`, the **damped
least-squares** (Tikhonov, ridge) solution of `M x = b` is

    x*(lam) = argmin_x  ||M x - b||^2 + lam ||x||^2,     lam >= 0.

`lam = 0` recovers ordinary least squares. `lam -> infinity` drives `x -> 0`.
Everything interesting is in between, and the question this document exists to
answer is what "in between" is chosen by, because the project's comment

> Measured cond(A) is 13-50 across arch31 elites, so this is well inside the
> regime where a modest ridge is enough.

is not an answer. A condition number bounds how far an inverse *can* amplify a
perturbation. It says nothing about how much damping to apply, because that
depends on how large the perturbation actually is, which cond() does not know.

## Assumptions

1. `M` is a fixed, known matrix. (In this project it is not: it is fitted from
   a noisy identification. That is the whole point, and it is what the
   *Validation* section measures.)
2. The 2-norm is the right norm on both spaces. On the input side that means
   all coefficients cost the same; on the output side, that all six twist
   components are commensurable. Neither is free — see `svd_control_basis.md`
   for the length scale that makes the second one true.
3. `lam` is a scalar, so the damping is isotropic in coefficient space.

## Derivation

Write `f(x) = ||M x - b||^2 + lam ||x||^2 = (Mx-b)^T(Mx-b) + lam x^T x`.

Expand:

    f(x) = x^T M^T M x - 2 b^T M x + b^T b + lam x^T x.

`f` is a quadratic with Hessian `2(M^T M + lam I)`, which is positive definite
for any `lam > 0` regardless of `M`, so `f` has exactly one stationary point and
it is the minimum. Setting the gradient to zero,

    grad f = 2 M^T M x - 2 M^T b + 2 lam x = 0
    =>  (M^T M + lam I) x = M^T b
    =>  x*(lam) = (M^T M + lam I)^-1 M^T b.                          (1)

That is the primal form. There is a dual form, and the identity connecting
them is worth stating because it is where the project's implementation is
usually misread. From

    M^T (M M^T + lam I) = (M^T M + lam I) M^T

— both sides expand to `M^T M M^T + lam M^T` — left-multiply by
`(M^T M + lam I)^-1` and right-multiply by `(M M^T + lam I)^-1`:

    (M^T M + lam I)^-1 M^T = M^T (M M^T + lam I)^-1.                 (2)

So `x*` can be computed from an `n x n` system or an `m x m` one, whichever is
smaller. The two are algebraically identical, not approximations of each other.

### The SVD picture, which is where the meaning is

Let `M = P S Q^T` be a thin SVD: `P` is `m x k` with orthonormal columns, `Q` is
`n x k` with orthonormal columns, `S = diag(s_1 >= ... >= s_k > 0)`.
Substituting into (1) and using `Q^T Q = I`:

    x*(lam) = Q diag( s_i / (s_i^2 + lam) ) P^T b.                    (3)

and the *delivered* output is

    M x*(lam) = P diag( s_i^2 / (s_i^2 + lam) ) P^T b.                (4)

Define the **filter factor**

    phi_i(lam) = s_i^2 / (s_i^2 + lam)  in [0, 1).                    (5)

Equation (4) says exactly what damping does: it projects `b` onto the column
space of `M` and then *shrinks each singular direction by `phi_i`*. A direction
with `s_i^2 >> lam` passes through untouched. A direction with `s_i^2 << lam` is
deleted. `lam` is the squared singular value at which a direction is halved.

Two consequences follow immediately and both matter here.

**Bias.** The delivered output is never the requested one: direction `i` is
short by `(1 - phi_i) = lam/(s_i^2 + lam)`. At `lam = 0.01 * mean(s^2)` and a
spectrum spanning a decade, the weakest direction can be short by tens of
percent while the strongest is short by one. This is not a small correction
applied uniformly; it is a reweighting that falls entirely on the weak axes.

**Variance.** From (3), the coefficient magnitude in direction `i` is
`s_i/(s_i^2+lam)`, which is bounded by `1/(2 sqrt(lam))` — attained at
`s_i = sqrt(lam)`. So `lam` bounds the coefficient a request can produce,
whatever `M` turns out to be. That is the numerical guard the ridge is usually
introduced for, and it is a real one: without it, `s_i -> 0` sends the
coefficient to infinity.

The trade between the two is the only honest way to choose `lam`, and it cannot
be read off `cond(M) = s_1/s_k`: two matrices with the same condition number can
have completely different spectra in between, and the error depends on the
spectrum through (5), not on its endpoints.

## Dimensional analysis

Let the input carry units `[u]` and the output `[y]`. Then `M` carries `[y]/[u]`
and each `s_i` does too. In `s_i^2 + lam`, the addition forces

    [lam] = ([y]/[u])^2.

So `lam` is *not* dimensionless and cannot be compared against 1. Any
"lam = 0.01" is shorthand for "lam = 0.01 times something with the units of a
squared gain", and the something has to be named. Two named choices:

* `lam = eps^2 * s_1^2` — relative to the strongest direction. Then `eps` is
  dimensionless and `phi_i = 1/(1 + eps^2 s_1^2/s_i^2)`, so the cut is at
  `s_i = eps s_1`: a *relative* singular-value floor.
* `lam = eps * mean_i(s_i^2) = eps * trace(M^T M)/k` — relative to the mean
  squared gain. The cut is then at `s_i = sqrt(eps * mean(s^2))`, which moves
  when *any* singular value moves, including ones far from the cut.

The project uses the second. The difference is not cosmetic: adding one strong
mode to a basis raises `mean(s^2)` and therefore damps every weak mode harder,
even though nothing about those weak modes changed.

## Numerical implementation

`MobilityBasis._inverse` in `dytiscidae/control/cpg.py`:

```python
A   = self.effects * self.authority[:, None]      # (r, 6)
G   = A @ A.T                                     # (r, r)
lam = 0.01 * float(np.trace(G)) / max(r, 1) + 1e-12
inv = np.linalg.solve(G + lam * np.eye(r), A)     # (r, 6)
...
coeffs = inv @ (w * reach * INTENT_AUTHORITY)
```

Mapping this onto the derivation: the *forward* model is
`twist = A^T c`, so `M = A^T` with `m = 6`, `n = r`. Then
`M^T M = A A^T = G`, and (1) reads

    c = (G + lam I)^-1 A b,

which is exactly the line of code. **The implementation is the primal form (1),
not the dual form (2).** It looks like the dual because the variable is called
`A` and `A A^T` appears, but `A` here is the transpose of the map. There is no
push-through identity in use and none is needed; the `r x r` system is the
smaller one whenever `r < 6`, which is the usual case.

The solve is therefore correct. Three things about it are not established:

* `0.01` is a choice of `eps` in the second form above, with no measurement
  behind it. `experiments/damping_lambda` is the measurement.
* `+ 1e-12` is a bare number added to a quantity with units `([y]/[u])^2`. It
  only ever matters when `trace(G)` is exactly zero — a body that cannot move
  at all — which is the case its comment says it exists for. It is dimensionally
  unsound and numerically harmless, and it would stop being harmless if the
  twist scaling changed.
* `reach = np.linalg.norm(A, axis=0)` is the per-axis maximum: by
  Cauchy-Schwarz, `max_{||c||=1} |(A^T c)_j| = ||A[:, j]||`, so `reach[j]` is
  genuinely the most twist this body can put on axis `j` for a unit coefficient
  norm. That is a correct derivation and it is nowhere written down in the code.

## Validation

Two measurements, both reproducible.

**The closed form (5) is exact**, checked in
`tests/test_math.py::test_the_inverse_model_delivers_what_it_can_and_no_more`.
On an orthogonal basis with `sigma = [4, 3, 2, 1, 0.5, 0.25]` the delivered
fraction of a saturated single-axis intent is

    [0.997, 0.994, 0.988, 0.952, 0.832, 0.553]

against `phi_i = s_i^2/(s_i^2 + 0.01 mean(s^2))` predicted to within 1e-9. The
docstring of `coeffs_for_twist` says

> As a fraction, "+1 heave" means "climb as hard as this body climbs" on every
> one of them.

By (5) that is true only where `s_i^2 >> lam`. On the weakest axis of this
spectrum the command delivers 55% of what it claims. Recorded as **C-04**.

**The choice of `lam` is measured out of sample** by
`experiments/damping_lambda`: the same body is identified four times with
different probe directions, coefficients are solved on one identification and
delivered on another, and the error is reported against `lam/lam_0`. In-sample
error is monotone decreasing in the direction of `lam -> 0` by construction; the
out-of-sample error is the one that can have an interior minimum, and whether it
does is what separates "the ridge is a numerical guard" from "the ridge is a
bias-variance control". See that experiment's README for the answer as measured.

## Failure conditions

* **`lam` chosen relative to a mean rather than a maximum.** Adding a strong
  mode damps the weak ones harder. A basis truncated to `max_modes=6` therefore
  gives different filter factors from the same basis truncated to 4.
* **Isotropic damping on an anisotropic problem.** A single scalar `lam` assumes
  every coefficient costs the same. Coefficients here multiply orthonormal
  directions in a CPG parameter space that mixes radians with hertz, so they do
  not. See `svd_control_basis.md`.
* **`M` is fitted, not given.** Equations (1)-(5) treat `M` as exact. Here it
  comes from a least-squares fit to 24 noisy probes, so the singular values
  themselves carry error; by Weyl's inequality no `s_i` below `||dM||_2` is
  resolvable at all, and damping below that level is fitting noise.
  `experiments/rank_threshold` measures `||dM||_2`.
* **`lam = 0` with a rank-deficient `M`.** `np.linalg.solve` raises, and the
  code falls back to `np.linalg.pinv(A.T)`, which is the `lam -> 0` limit of (3)
  with the zero singular values dropped — a different estimator, silently
  substituted.
