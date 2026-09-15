# hull_buckling — is the hull's collapse pressure the ring result, or 8x it?

`structure.hull_pressure_check` published

    p_cr = 0.6 * 2 E / (1 - nu^2) * (t / r)^3

as the external-pressure collapse pressure of a cylindrical pressure hull, and
the check **gates**: a design whose hull fails it is rejected before it is
scored. `experiments/analytic_vs_numerical` measured that expression at 8.0x
the classical ring result. This experiment asks what the factor is, and what
correcting it costs.

```
PYTHONPATH=. python experiments/hull_buckling/run.py
```

## The four explanations, and which one predicted the measurement

| | prediction | measured |
|---|---|---|
| **E1** the diameter coefficient on a radius ratio | a factor of exactly `2^3 = 8`, with no dependence on E, nu, t or r | **8.000000000000**, spread `7.1e-15` over 256 grid points |
| **E2** a deliberate finite-length credit | the allowable rises as L/r falls | the same `0.1960 bar` for L/r from 1 to 100 — `length` is an argument the function never reads |
| **E3** a different lobe count | some n with `(n^2 - 1) = 24`, i.e. n = 5 | n = 5.000000 exactly, **and the long-cylinder minimiser is n = 2** — the first available mode, because `(n^2-1)` is increasing in n |
| **E4** a units convention | a power of ten | 8 is not one, and `experiments/dimensional_check` reads both sides as Pa |

E1 is the only one that predicted what was measured. E3 is arithmetically
consistent and refuted by the minimiser: a coefficient fixed at the five-lobe
mode for every geometry is not a mode choice.

## The model, from its own premises

A unit-length slice of the wall is a ring of second moment `I = t^3/12` per unit
width. A ring under uniform external pressure buckling into `n` circumferential
lobes goes unstable at

    p_cr = (n^2 - 1) E' I / r^3,    E' = E / (1 - nu^2)

`n = 1` is a rigid translation of the section, not a buckle, so the first
available mode is `n = 2`:

    p_cr = 3 E' t^3 / (12 r^3) = E / (4 (1 - nu^2)) * (t/r)^3

which is the same physics as `2E/(1-nu^2) (t/D)^3` with D the **diameter**. The
two coefficients differ by `2^3` and the code carried the diameter one on a
radius ratio.

## What this experiment does not settle

The ring result is the **long**-cylinder limit. The seed hulls run L/r from 5.1
to 14.3, where a finite-length correction is real and raises `p_cr`: the ends
restrain the lobes. Quantifying it needs a shell result this repository does not
contain and this experiment does not derive, so **the corrected allowable is a
lower bound**, and it is labelled that way rather than presented as the answer.

What the measurement does settle is that 8 is not that correction: the
expression returns the same number for every length.

## The consequence

At the wall it had (`t/r = 0.038`), every seed hull passed the legacy check with
margin +0.300 and every one of them fails the ring result at −0.838. So the
correction cannot stop at the allowable — the sizing rule moves with it:

| | |
|---|---|
| `hull_wall` carried | `t/r = 0.0380` |
| covers the 12 m design depth | `t/r = 0.0696` |
| covers 1.3x it, which is what 0.038 delivered against the old allowable | **`t/r = 0.0760`** |
| factor on wall, and on hull mass, which is linear in t | **2.00x** |
| the 12 mm printable clip now binds above | r = 158 mm, against 316 mm before |

`hull_wall` is now `structure.wall_for_buckling` inverted, not a refitted
literal, so the sizing rule and the check that grades it cannot drift apart
again. That drift is how the 8x survived: 0.038 was the wall that number asks
for, so the two agreed with each other while neither agreed with the ring
result.

### What it costs the seed plans

| plan | mass before | after | change |
|---|---|---|---|
| bat | 3.614 kg | 3.960 kg | +9.6% |
| beetle | 4.457 kg | 5.129 kg | +15.1% |
| eel | 1.927 kg | 2.259 kg | +17.2% |
| gannet | 4.490 kg | 5.309 kg | +18.2% |
| medusa | 10.447 kg | 10.447 kg | **+0.0%** |
| ray | 4.793 kg | 5.445 kg | +13.6% |
| teal | 4.598 kg | 5.345 kg | +16.2% |

medusa does not move because its bell wall was moved onto its own constant,
`BELL_WALL_FRACTION`, frozen at the value it had. A bell is open, flooded and
has to flex; external-pressure collapse of a sealed cylinder is not its load
case, so there is no reason for its wall to follow the pressure-hull rule. It
did, through `0.35 * hull_wall(radius)`, and would have gained 40% of medusa's
mass on a correction that has nothing to do with it.

### One seed plan stops floating

| plan | floats up to | against the corrected minimum `t/r = 0.0696` |
|---|---|---|
| **bat** | `t/r = 0.0571` | **0.82x — it sinks** |
| eel | `t/r = 0.1040` | 1.49x |
| ray | `t/r = 0.1377` | 1.98x |
| teal | `t/r = 0.1372` | 1.97x |
| gannet | `t/r = 0.1435` | 2.06x |
| beetle, medusa | above the sweep's range | — |

bat carries the corrected wall only if the finite-length credit is at least
**1.82x**. That credit is real, it is the thing section "what this experiment
does not settle" declines to compute, and its magnitude at L/r = 7.1 is exactly
the open question. So whether bat survives is **open** — this experiment does
not settle it either way, and the plan is left failing its own buoyancy check
rather than rescued by a safety factor chosen to rescue it.

## Files

| | |
|---|---|
| `config.json` | depths, the grids, the knockdown, the sizing safety factor |
| `run.py` | sections A–G |
| `results/result.json` | every number above, with provenance |
| `tests/test_hull_buckling.py` | the properties, as a regression test |
