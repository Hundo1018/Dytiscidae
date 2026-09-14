# `0.08 * sigma_0` is an engineering guess, and the thing it is guessing about is not measurable from one identification

```
PYTHONPATH=. python experiments/rank_threshold/run.py
```

84 identifications: 7 seed plans x 3 media x 4 probe seeds, 24 probes each run
both signs for 1.2 s, about 22 minutes of wall clock. Cached in
`experiments/_cache/probes.npz` and shared with `damping_lambda`.

## The question

`MobilityBasis.rank` counts singular values above 8% of the largest and calls
the result "how many genuinely independent things can this machine do". That is
a claim about numerical rank, and numerical rank is defined against the level at
which the matrix's own uncertainty could have produced a singular value. `0.08`
is not that level unless someone measured it.

## What was measured

### The fit is underdetermined on most of the fleet

A machine with `n` joints presents `P = 3n+1` parameters to `n_probes = 24`
probes.

| plan | parameters | probes | |
|---|---|---|---|
| bat | 25 | 24 | underdetermined |
| beetle | 19 | 24 | determined |
| eel | 16 | 24 | determined |
| gannet | 25 | 24 | underdetermined |
| medusa | 49 | 24 | underdetermined |
| ray | 28 | 24 | underdetermined |
| teal | 31 | 24 | underdetermined |

**5 of 7.** Where the system is underdetermined, `lstsq` returns the
minimum-norm solution inside the 24-dimensional row space of whichever
directions happened to be drawn, and the residual reads **exactly zero** — not
because the fit is good, but because an underdetermined system always fits.
`residual_fraction` is 0.000 for every one of those five plans, and 0.14 to 0.58
for beetle and eel, the two that are actually overdetermined. A reader taking
`resid = 0` as evidence of linearity would have it backwards.

### The identification noise floor is larger than the leading singular value

Re-identify the same body in the same medium with different probe directions
and compare the Jacobians:

    ||J_a - J_b||_2 / sigma_0  =  1.132   (95% CI 1.039 - 1.219, n = 21)

For scale, two matrices drawn independently with the same spectrum would sit
near `sqrt(2) = 1.414`; two identical ones at 0. By Weyl's inequality
`|sigma_i(J) - sigma_i(J + dJ)| <= ||dJ||_2`, so **no singular value is
resolvable from a single identification** — the floor is above `sigma_0`, let
alone above `0.08 sigma_0`. `0.08` is below the floor on all 21 (body, medium)
pairs.

### But the two sides of the decomposition behave completely differently

Principal angles between two seeds' rank-`r` subspaces, averaged over all 21
pairs (90 degrees means the two identifications share nothing):

| rank r | modes (CPG parameter side) | effects (body twist side) |
|---|---|---|
| 1 | 67.8 deg | 36.0 deg |
| 2 | 75.3 | 41.2 |
| 3 | 80.5 | 33.3 |
| 4 | 83.0 | 37.6 |
| 5 | 84.5 | 29.5 |
| 6 | 85.3 | 0.0 (both span all of R^6) |

And the singular values themselves reproduce to a coefficient of variation of
0.15 - 0.21.

**What the machine can do is identified. How to ask for it is not.** The twist
directions and their gains survive a reseed; the parameter directions do not,
and for the five underdetermined plans they cannot, because each seed's `J`
lives in a different random 24-dimensional subspace of parameter space.

### The threshold sweep

Averaged over all 84 identifications:

| tau | rank | recon error | control error | cond | max joint excursion |
|---|---|---|---|---|---|
| 0.001 | 6.00 | 0.0000 | 0.0972 | 21.5 | 0.909 |
| 0.005 | 6.00 | 0.0000 | 0.0972 | 21.5 | 0.909 |
| 0.010 | 5.82 | 0.0012 | 0.0982 | 21.5 | 0.906 |
| 0.020 | 5.46 | 0.0044 | 0.1019 | 17.3 | 0.894 |
| 0.050 | 4.93 | 0.0161 | 0.1206 | 11.1 | 0.855 |
| **0.080** | **4.48** | **0.0337** | **0.1614** | **7.6** | **0.781** |
| 0.100 | 4.26 | 0.0467 | 0.1890 | 6.5 | 0.741 |
| 0.200 | 3.32 | 0.1197 | 0.3455 | 3.8 | 0.472 |

Everything is monotone. There is no knee, no plateau and nothing that picks out
`0.08` from its neighbours.

### There is no spectral gap to find

Of the 420 non-leading singular values across all identifications, 30.5% fall
below `0.08 sigma_0` and **30.2% land within a factor of two of it**. A genuine
gap would put that second number near zero. The spectrum is a continuum through
the threshold, so any fixed fraction cuts it somewhere arbitrary.

### The rank the threshold reports is not stable

How often all four probe seeds agree on the rank, over the 21 (body, medium)
pairs:

| tau | unanimous |
|---|---|
| 0.001 | 100.0% |
| 0.005 | 100.0% |
| 0.010 | 71.4% |
| 0.020 | 81.0% |
| 0.050 | 52.4% |
| **0.080** | **47.6%** |
| 0.100 | 23.8% |
| 0.200 | 23.8% |

At the incumbent threshold, **more than half of all bodies get a different
reported rank depending on which random directions the probe drew.**

## Conclusion

`0.08` is not a numerical rank. It is a usefulness threshold, and its sweep is
smooth, so nothing is lost by saying so and making it a parameter. What the
experiment found instead is larger than the threshold question:

1. The mobility identification is underdetermined on 5 of 7 seed plans, and a
   zero residual is the symptom rather than a reassurance.
2. Its parameter-side directions do not survive a reseed (68 - 85 degree
   principal angles).
3. Its twist-side directions and gains do (30 - 41 degrees, CV 0.15 - 0.21).

Recorded as **C-03**, **C-07** and **C-08** in `docs/MATH_AUDIT.md`. The cheap
first move is to raise `n_probes` above `max(3n+1)` for the fleet and re-measure
the floor; `medusa` at 49 parameters sets the bar.
