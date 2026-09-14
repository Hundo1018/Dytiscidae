# Reduced frequency `k = omega c / (2 U)`

## Definition

For a surface of chord `c` oscillating at angular frequency `omega` while
translating through a fluid at speed `U`, the **reduced frequency** is

    k = omega b / U,    b = c/2 the semi-chord,   so   k = omega c / (2 U).

It is the ratio of two time scales, and that is the whole content of it:

* `t_convective = b / U` — how long a fluid particle takes to pass the
  semi-chord, which is how long the surface has to influence it;
* `t_oscillatory = 1 / omega` — how long the surface takes to change what it
  is doing.

`k = t_convective / t_oscillatory`. When `k << 1` the surface changes slowly
compared with the flow's memory of it, and the flow at each instant looks like
the steady flow at that instant's incidence: the **quasi-steady** assumption.
When `k` is order one, the wake shed a semi-chord ago is still nearby and still
influencing the surface, and no instantaneous relation between incidence and
lift can be right.

## Assumptions

The quantity is defined for a surface in *periodic* motion. Three things are
therefore assumed before `k` means anything:

1. There is a single frequency. A surface executing a compound stroke
   (flapping and pitching at different rates, or at a rate that changes within
   the stroke) has no single `k`.
2. `U` is the translational speed of the surface relative to the fluid, and it
   is non-zero. Hover, where `U -> 0` at every stroke reversal, is the case `k`
   is least able to describe: `k -> infinity` at reversal for any non-zero
   oscillation.
3. `c` is a representative chord. For a tapered surface `k` varies along the
   span, and the "reduced frequency of the wing" is a nominal quantity at a
   reference station.

## Derivation

### From the two time scales

Nondimensionalise the unsteady thin-airfoil problem. Take `b` as the length
scale, `U` as the velocity scale, and therefore `b/U` as the time scale. A
plunge or pitch history `h(t) = h_0 e^{i omega t}` becomes, in the
nondimensional time `tau = U t / b`,

    h(tau) = h_0 e^{i (omega b / U) tau} = h_0 e^{i k tau}.

`k` is what the frequency *becomes* when time is measured in semi-chords
travelled. Nothing else in the nondimensional problem carries the frequency, so
the unsteady solution can depend on it only through `k`. That is the whole
argument, and it is why `k` is the right group rather than, say, `omega c / U`
or `f c / U`: the semi-chord is the length scale the thin-airfoil problem
naturally uses (the circulation distribution is written over `x in [-b, b]`),
so the factor of 2 is not a convention, it is the length that appears.

### Why `k -> 0` recovers quasi-steady

Theodorsen's solution for a thin airfoil in incompressible potential flow
splits the lift into a circulatory part and an apparent-mass part, with the
circulatory part multiplied by the complex transfer function

    C(k) = F(k) + i G(k),      C(0) = 1,     C(infinity) = 1/2.

`C(k)` is the fraction of the steady circulatory lift that survives at
frequency `k`, and its phase is the lag. Quasi-steady aerodynamics is exactly
the statement `C(k) = 1`: the shed wake is ignored. So `k` is the parameter
that measures the error of the assumption the whole solver is built on.

Magnitudes, as a scale for "how small is small":

| k | \|C(k)\| | what quasi-steady is over-predicting the circulatory lift by |
|---|---|---|
| 0.00 | 1.00 | nothing |
| 0.05 | ~0.90 | ~11% |
| 0.10 | ~0.85 | ~18% |
| 0.20 | ~0.77 | ~30% |
| 0.50 | ~0.65 | ~54% |
| 1.00 | ~0.57 | ~75% |

A flapping machine operates at `k` of order 0.1 to 0.5. That range is the
source of the `+/- 30%` the project's README already claims for the
quasi-steady model, and it is worth seeing that the claim has a derivation
rather than being a hedge.

### The leading-edge vortex, which is why `k` is not simply an error term

At high `k` and high incidence the flow separates at the leading edge and
reattaches, forming a vortex that stays bound to the surface for part of the
stroke. While it is attached it adds circulation, so the surface keeps
generating lift far past the static stall angle. This is a *history* effect: the
vortex grows over a fraction of a stroke and is shed at reversal, so its
strength at time `t` depends on the motion over the preceding stroke, not on the
instantaneous rate.

That distinction is where the implementation departs from the derivation.

## Dimensional analysis

    [omega] = rad/s = 1/s   (radian is dimensionless)
    [c]     = m
    [U]     = m/s

    [k] = (1/s)(m) / (m/s) = (m/s) / (m/s) = 1.                       ok

Checked mechanically in `experiments/dimensional_check` ("reduced frequency
k = |omega_s| c / (2 U)"), which also fails the run if the source line it
restates ever changes.

## Numerical implementation

`dytiscidae/physics/fluid.py`:

```python
omega_s      = np.einsum("ni,ni->n", omega, s_hat)
reduced_freq = np.abs(omega_s) * p.chord / (2.0 * U_safe)
...
lev = np.clip(reduced_freq / 0.30, 0.0, 1.0)
```

`omega` is the owning body's angular velocity and `s_hat` is the strip's
spanwise axis, so `omega_s` is **the instantaneous body rate about the span
axis** — the pitch rate `alpha_dot`, in rad/s. The quantity computed is
therefore

    kappa = |alpha_dot| c / (2 U),

the **reduced pitch rate**. It is a real dimensionless group (it appears
throughout the dynamic-stall literature, where it is usually written
`alpha_dot c / 2U` and governs stall delay), and it has the same dimensions as
`k`. It is not the same number.

The difference is not subtle for a periodic motion. With `alpha(t) = alpha_0
sin(Omega t)`:

    k     = Omega c / (2U)                          — constant over the stroke
    kappa = alpha_0 Omega |cos(Omega t)| c / (2U)   — zero twice per stroke,
                                                      peak alpha_0 k at mid-stroke

So `kappa = alpha_0 |cos(Omega t)| * k`. Two consequences:

* **Amplitude dependence.** `kappa` carries a factor `alpha_0` in radians that
  `k` does not. A surface pitching +/- 0.3 rad at a given frequency gets 30% of
  the LEV credit that the same surface pitching +/- 1.0 rad does, at the same
  `k`.
* **Phase dependence.** `kappa = 0` at both stroke extremes, so the modelled
  LEV strength collapses to zero exactly at reversal — the point in the stroke
  where a real LEV is largest, having grown over the preceding half-stroke. The
  model has the history backwards, not merely absent.

The module docstring describes the term as

> a wing that is flapping fast relative to its own translation carries a stable
> LEV

which is a statement about `k`, and prints the formula `k = omega c / (2 U)`
under a variable that holds `alpha_dot`. The same `omega_s` is used, correctly,
for the Kramer rotational force on the following lines — there the pitch rate
*is* the right variable. One quantity, two roles, right in one of them.

The `0.30` in `lev = clip(kappa/0.30, 0, 1)` is a reduced-rate value, correctly
dimensionless, with no source.

`U_safe = max(U, 1e-6)` means that at a stroke reversal where the translational
speed also goes to zero, `kappa` is a 0/0 whose value is decided by which of
`alpha_dot` and `U` reaches its floor first. For a strip at radius `r` on a wing
flapping at `Omega`, both `U ~ Omega r` and `alpha_dot ~ Omega`, so `kappa`
stays finite through reversal at roughly `c/(2r)` — a geometric ratio with no
frequency in it at all.

## Validation

* **Dimensions**: `experiments/dimensional_check` — passes.
* **The `alpha_0 |cos|` relation**: algebra above; not yet measured against a
  rollout of a real machine. What that measurement would be: record `omega_s`,
  `U` and `chord` per strip over one stroke of a flapping elite and plot
  `kappa(t)` against the constant `k` of the same stroke. Nothing in the project
  records those three together today, which is why this section says "not
  measured" rather than quoting a number.
* **The consequence for lift**: `lift_coefficient` is a function of `lev`, so
  the sensitivity is measurable offline without any rollout —
  `experiments/analytic_vs_numerical` already measures that `lev = 1` raises
  post-stall CL at 25 degrees from 0.941 to 1.898, a factor of two. Whatever
  `kappa` is doing, it is worth a factor of two in lift at the angles a flapping
  wing works at.

## Failure conditions

* **Hover, and any near-hover stroke reversal.** Both `k` and `kappa` are
  singular as `U -> 0`, and the code's `1e-6` floor picks an answer rather than
  admitting there isn't one.
* **Non-periodic motion.** `k` is defined for an oscillation. A machine
  manoeuvring, or a CPG whose frequency the controller is commanding, does not
  have one frequency, and `kappa` — being instantaneous — is defined but is not
  a reduced frequency.
* **`k` above about 0.2.** Quasi-steady coefficients over-predict circulatory
  lift by 30% or more (table above). Nothing in the solver reports when a strip
  is in that regime, so the error is never visible in the telemetry. A cheap
  fix would be to record `max(kappa)` per rollout beside `max_alpha`, which
  `FluidDiagnostics` already does for incidence.
* **Compressibility.** `k` assumes incompressible flow. Irrelevant at this
  project's speeds; recorded so nobody re-derives it later.
