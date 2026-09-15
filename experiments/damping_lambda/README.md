# `lam = 0.01 * trace(G) / r` is about 30x too small

```
PYTHONPATH=. python experiments/damping_lambda/run.py
```

Same 84 cached identifications as `rank_threshold` — 7 plans x 3 media x 4
probe seeds — and 512 random intents per (body, medium, seed-pair).

## The question

`MobilityBasis._inverse` solves `c = (A A^T + lam I)^-1 A b`, which is the
Tikhonov solution of `min ||A^T c - b||^2 + lam ||c||^2`. The solve is correct
(checked exactly in `tests/test_math.py`). The justification is not:

> Measured cond(A) is 13-50 across arch31 elites, so this is well inside the
> regime where a modest ridge is enough.

A condition number bounds how far an inverse *can* amplify a perturbation. It
does not say how much ridge to apply, because that depends on how large the
perturbation actually is. The quantity that decides is the error of the
resulting command on data the ridge was not fitted to.

## The design

The same body is identified four times with different probe directions. For
each ordered pair `(a, b)` and each intent `w`:

* solve `c` on identification `a`;
* the controller sends a **CPG parameter offset** `dp = modes_a^T c`, which is
  the only thing that crosses between identifications — a coefficient vector
  lives in one SVD's private mode coordinates and carrying it across compares
  nothing;
* in-sample error is `||J_a^T dp - b_a|| / ||b_a||`;
* out-of-sample error is `||J_b^T dp - b_b|| / ||b_b||`.

**Sending no command at all scores exactly 1.0.** That is the reference any
measurement here has to beat.

## What was measured

| lam/lam0 | in-sample | out-of-sample | effort | max joint excursion | cond(G+lam I) |
|---|---|---|---|---|---|
| 0 | 0.0000 | 1.4294 | 1.887 | 1.719 | 375 |
| 0.001 | 0.0007 | 1.4173 | 1.869 | 1.701 | 375 |
| 0.01 | 0.0054 | 1.3420 | 1.759 | 1.585 | 372 |
| 0.1 | 0.0269 | 1.1551 | 1.452 | 1.272 | 342 |
| 0.3 | 0.0506 | 1.0691 | 1.280 | 1.108 | 291 |
| **1 (incumbent)** | **0.0968** | **0.9738** | **1.067** | **0.917** | **192** |
| 3 | 0.1644 | 0.8914 | 0.848 | 0.720 | 98.1 |
| 10 | 0.2669 | 0.8341 | 0.613 | 0.512 | 36.3 |
| **30 (best)** | **0.3830** | **0.8234** | **0.431** | **0.359** | **13.2** |
| 100 | 0.5343 | 0.8499 | 0.276 | 0.235 | 4.86 |
| 300 | 0.6921 | 0.8964 | 0.163 | 0.145 | 2.30 |
| 1000 | 0.8503 | 0.9488 | 0.073 | 0.068 | 1.39 |

### The hypotheses

* **H1, "the ridge is only a numerical guard"** — rejected. Out-of-sample error
  varies by 72% across the non-zero ratios. It is not flat.
* **H2, "the ridge is a bias-variance control"** — supported. In-sample error
  rises monotonically with `lam` (bias) while out-of-sample error falls and
  then rises, with a clear interior minimum at `lam/lam0 = 30`. That is the
  textbook shape and it is what the curve does.
* **H3, "the ridge bounds joint excursion"** — also true, and it is the same
  mechanism: excursion falls from 1.72 to 0.07 of the joint travel across the
  sweep. At the incumbent the worst-case excursion is 0.917 of the available
  travel, which is on the edge of saturating; at `lam = 0` it is 1.72, i.e. the
  command asks for 72% more travel than the joints have.

### The size of the error

    incumbent minus best, paired over 21 (body, medium) pairs:
        +0.1503   (95% CI +0.1008 to +0.2006)

an **18.3%** higher out-of-sample error than `lam/lam0 = 30`. The best ratio
per pair is 30 on 11 of 21, 10 on 4, 100 on 3, 3 on 2, 1000 on 1 — never the
incumbent.

Per medium, which the single ridge does not distinguish:

| medium | best ratio | best error | incumbent error | penalty |
|---|---|---|---|---|
| air | 10 | 0.7972 | 0.8789 | +10.2% |
| water | 30 | 0.7545 | 0.9181 | +21.7% |
| land | 30 | 0.9166 | 1.1243 | +22.7% |

**On land the incumbent scores 1.124, worse than sending no command at all.**
Five of the twelve swept ratios do — everything at `lam/lam0 <= 1`.

## Is the optimum predictable from something observable?

Hardcoding 30 is better than hardcoding 1 and worse than not hardcoding. If the
optimal ridge tracks a conditioning number a caller can compute from the basis
it already holds, `lam` stops being a constant.

Pearson correlation against `log10` of the best `lam/lam0`, bootstrapped over
the 21 (body, medium) pairs:

| predictor | r | 95% CI | |
|---|---|---|---|
| `log10 cond(A)` | **−0.511** | [−0.73, −0.19] | excludes 0 |
| `log10 sigma_min/sigma_max` | **+0.518** | [+0.20, +0.74] | excludes 0 |
| `log10 sigma_max` | −0.223 | [−0.70, +0.17] | includes 0 |
| probe noise `\|\|dJ\|\|/sigma_0` | +0.322 | [−0.05, +0.59] | includes 0 |
| residual fraction | +0.216 | [−0.16, +0.62] | includes 0 |
| `n_probes / n_params` | +0.203 | [−0.15, +0.62] | includes 0 |
| underdetermined | −0.158 | [−0.58, +0.19] | includes 0 |

The two that survive are the same quantity with opposite sign, so it is one
finding: **a better-conditioned basis wants a larger multiple of `lam_0`, and a
worse-conditioned one a smaller.**

That is what you would expect if the right *absolute* ridge tracks the weakest
singular value rather than the mean — `lam_0` already carries `mean(sigma^2)`,
so a ratio that falls with conditioning is a ratio correcting `lam_0` back
toward something smaller. Fitting the absolute optimum directly:

    log10(lam*) = a log10(sigma_min) + b log10(sigma_max) + c

    a = +0.859    b = +1.286    c = +0.075    R^2 = 0.904,  n = 21

For reference `lam ~ sigma_min^2` is `(2, 0)`, `lam ~ sigma_max^2` is `(0, 2)`,
and **`lam ~ sigma_min * sigma_max` is `(1, 1)`** — which is close to what the
data says, and is dimensionally sound since the exponents sum to 2.15 against
the 2 that `lam`'s units require.

So the candidate is `lam = kappa * sigma_min * sigma_max` — the geometric mean
of the extreme squared gains — rather than `0.01 * mean(sigma^2)`. On 21 points
that is a lead, not a law. What it does establish is that **the optimum is not
a constant multiple of `lam_0`**, which is what the incumbent assumes.

## Conclusion

Three separate statements, in increasing order of what they cost to act on.

1. **`0.01` should be about `0.3`.** `lam = 0.3 * trace(G)/r` minimises
   out-of-sample error on this fleet, and the curve is broad enough that
   anything from 10x to 100x the incumbent is better than the incumbent.
   Recorded as **C-09**. But see the section above: the better move may be to
   stop hardcoding a multiple of `lam_0` at all.
2. **The ridge should probably not be one number for three media.** Air wants
   10x and water and land want 30x, and the media differ in what they are
   identifying.
3. **Out-of-sample error near 0.82 at the optimum is the real headline.** Even
   at the best ridge, a command computed from one identification of a body
   delivers only about 18% closer to the intent than commanding nothing. That
   is not a damping problem; it is the same underdetermination
   `experiments/rank_threshold` found on 5 of 7 plans. Raising `n_probes` is
   the move that changes this number; retuning `lam` only stops it being worse
   than doing nothing.
