# stall_blend — each branch was being evaluated where it does not apply

`fluid.lift_coefficient` writes CL as a partition of unity between an attached
branch and a separated one:

    w  = 1 / (1 + exp(-(|alpha| - alpha_stall) / 6 deg))
    CL = (1 - w) * CL_alpha * alpha  +  w * CL_max * sin(2 alpha)

A partition of unity is a statement about **where each branch is valid**. A
logistic is never 0 and never 1, so neither statement was made.

```
PYTHONPATH=. python experiments/stall_blend/run.py
```

## Both leaks, measured

| | |
|---|---|
| `w(0)` at the static stall angle | **0.138** — 13.8% of the separated branch at zero incidence |
| worst lift-slope deficit, over 100 `(AR, Re, kappa)` points | **−9.6%**, at AR 16 and Re 1e4 |
| mean deficit, gliding (`kappa = 0`) | **−6.96%** |
| mean deficit, flapping (`kappa = 0.30`) | −0.03% |
| attached weight 2° past stall | **41.7%** |
| attached weight 10° past stall | **15.9%** |

The second row of that table is the half that had not been stated. An
**unbounded** attached-flow extrapolation was still weighted a sixth ten degrees
into the stall.

It is a **gliding-wing** defect. A high reduced pitch rate raises `alpha_stall`
from 11° to 37°, which pushes the tail away from zero — so the loss falls on
exactly the behaviour the air ladder exists to reward.

## Four explanations for the 6°, one that predicted the measurement

| | prediction | measured |
|---|---|---|
| **E1** a smoothing width with no thought to where its tails land | `w(0)` is a function of `alpha_stall` alone, and largest where the stall angle is lowest | 15 distinct stall angles, each with exactly one `w(0)`; 10° → 0.1586, 37° → 0.0021 |
| **E2** a deliberate soft stall for low Reynolds number | the leak tracks Re | it tracks Re only through `alpha_stall`; the model's Reynolds factor is already on the stall angle, not on the blend |
| **E3** calibrated against data | a source | none in the code, in `docs/model_validity.md`, or in `derivations/aerodynamic_coefficients.md` |

## The fix, and where its one constant comes from

`w` is a smoothstep `t²(3 − 2t)` on `[0, alpha_stall + SEPARATION_COMPLETE]` —
the lowest-order polynomial that is C¹ at both ends, so **`w` and `w′` are both
exactly zero at zero incidence**: the separated branch contributes neither lift
nor slope there, which is the property a logistic cannot have at any width.

`SEPARATION_COMPLETE = 16°` is measured, not chosen. Section F records the
arguments the solver hands `lift_coefficient` over **3.3 million strip-steps**
of the seven seed plans in all three media, by wrapping the function rather than
by changing the solver, and the sweep is scored against that sample:

| width past stall | 4° | 9° | 13° | 15° | **16°** | 17° | 18° | 20° | 24° | 30° |
|---|---|---|---|---|---|---|---|---|---|---|
| RMS ΔCL, as a fraction of the CL in use | 8.4% | 6.1% | 4.4% | 3.9% | **3.8%** | 3.8% | 4.0% | 4.5% | 6.6% | 10.6% |
| reaches the `1.2 CL_max` clip | no | no | no | no | **no** | no | no | yes | yes | yes |

An **interior** minimum, not the edge of the sweep — which is the check that
stops "smallest change" from degenerating into "no change".

**This is a conservatism criterion and not a physical one**, and the code says
so. Every compact blend removes F-05; the sweep only decides which of them
moves the rest of the model least, because the purpose was to remove a defect,
not to re-tune the lift model.

## What the machines actually visit

The most useful thing this experiment measured is not about the blend:

| medium | strip-steps | median \|α\| | 90th | below the model's own stall angle |
|---|---|---|---|---|
| air | 1,108,000 | 33.8° | 83.1° | 27.6% |
| water | 1,108,000 | 39.5° | 86.3° | 26.9% |
| land | 1,108,000 | 43.7° | 88.7° | 27.7% |
| **all** | **3,324,000** | **38.3°** | **86.1°** | **27.4%** |

The seed plans spend roughly three-quarters of their strip-steps **past** stall,
and remarkably consistently across the three media. So the half of F-05 that
matters most for these machines is the *second* half — the attached branch
leaking upward — and not the zero-incidence tail the finding was written about.

These are measured on the fixed tree, so the distribution is itself downstream
of the change: under the logistic the same rollouts gave a median of 37.9° and
27.7% below stall. The second digit moves because the machines fly slightly
differently once the lift is corrected; the conclusion does not.

## What moved

| | before | after |
|---|---|---|
| glider fixture, twist alone | L/W 1.26 | **1.39** |
| glider fixture, camber alone | L/W 0.99 | **1.09** |
| model peak CL | 2.280 (the clip, reached exactly) | **2.093** |
| max \|CL\| as a fraction of `1.2 CL_max` | 100.00% | **97.41%** |
| RMS change to the CL in use | — | 3.8% |
| mean change to the CL in use | — | +0.0006 |

A wing below stall now carries the full attached slope, which is about +10% of
lift on the fixture. The mean ΔCL is essentially zero: the fix redistributes
rather than adding lift.

## A null result, reported as one

Section G scores all seven seed plans with each blend. Nothing moves beyond the
fourth decimal, and no plan clears a single air gate either way. **That measures
the plans, not the change**: the seed plans are hand-written starting points
that score at the floor, so no lift-model change can move them. A ΔS for this
fix needs an archive of evolved elites, which this repository's `runs/` does not
carry. Section F is the measurement that does exist.

## A finding this produced and did not fix: F-08

Sweeping *where* CL peaks, in units of the model's own stall angle, gives
**4.50 for every candidate blend** — because the separated branch is
`CL_max sin(2 alpha)`, which peaks at 45°, and no blend changes that.

So the model has no post-stall lift loss at all: at `alpha_stall = 10°` and
`CL_max = 1.1` it says a wing lifts 0.73 at its stall angle and 1.10 at 45°.
Giving it a drop means a third branch, which is a larger change than correcting
a handover. `docs/MATH_AUDIT.md` **F-08**.

## Files

| | |
|---|---|
| `config.json` | the grids, the candidate families, the rollout settings |
| `run.py` | sections A–G |
| `results/result.json` | every number above, with provenance |
| `tests/test_stall_blend.py` | the properties, as a regression test |
