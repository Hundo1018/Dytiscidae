# wing_added_mass — the tensor is right, and the solver does not survive it

`FluidSolver.apply` gives a wing strip

    m_add = rho * pi * chord^2 / 4 * dr

whichever way the strip accelerates. That is the **normal** entry of the plate's
added-mass tensor. Strip theory gives all three:

    m_span = 0        m_chord = rho pi t^2/4 b        m_normal = rho pi c^2/4 b

`benchmarks/layers.py` layer 2 verifies that chain against the closed form at
c/t = 10, 100 and 1000 with no simulation in it. The **bluff** branch four lines
above already projects its own tensor onto the direction of motion. The wing
branch did not. `docs/MATH_AUDIT.md` **F-03**.

```
PYTHONPATH=. python experiments/wing_added_mass/run.py
```

This experiment is not here to argue the tensor is right — layer 2 settles that.
It answers the two questions that decide whether it can be applied.

## What is it worth on a machine that is flapping?

`(c/t)^2` is the error in the pure edgewise direction, and no wing spends its
time there. Section B records the direction the flow actually comes from at
every wing strip of every seed plan, in all three media:

| medium | strip-steps | span | chord | normal |
|---|---|---|---|---|
| air | 440,000 | 0.3065 | 0.4730 | 0.2203 |
| water | 440,000 | 0.2369 | 0.3644 | 0.3967 |
| land | 440,000 | 0.1466 | 0.4038 | 0.4477 |
| **all** | **1,320,000** | **0.2300** | **0.4137** | **0.3549** |

Mean squared direction cosines in the strip's own frame. The flow is **more
chordwise than normal**, and 23% spanwise — the direction the tensor says
entrains nothing at all.

Projected onto that distribution:

| plan | dry mass | coded | tensor | ratio |
|---|---|---|---|---|
| bat | 3.960 kg | 67.18 kg | 23.93 kg | 35.6% |
| beetle | 5.129 kg | 33.95 kg | 12.17 kg | 35.8% |
| eel | 2.259 kg | 4.84 kg | 1.73 kg | 35.7% |
| gannet | 5.309 kg | 70.24 kg | 25.07 kg | 35.7% |
| medusa | 10.447 kg | 3.57 kg | 1.28 kg | 36.0% |
| **ray** | **5.445 kg** | **209.88 kg** | **74.76 kg** | **35.6%** |
| teal | 5.345 kg | 69.87 kg | 24.94 kg | 35.7% |

So the correction is worth about **2.8x**, not the 176x–251x that `(c/t)^2` gives
on these wing sections. The ratio is 35.6–36.0% on *every* plan, because the
flow-direction distribution is nearly the same for all of them.

`ray` carries **38.5 times its own dry mass** in wing added mass.

## What happens to the solver if it is applied?

The tensor is in the solver behind `FluidSolver.wing_added_mass_tensor`. Turning
it on makes `benchmarks/` **layer 2 hold at 8.3e-06** where it departs at 0.729.
Then:

| plan | dt=0.004 | dt=0.002 | dt=0.001 | dt=0.0005 | still, 0.004 | incumbent driven | incumbent still |
|---|---|---|---|---|---|---|---|
| bat | **1374.70** | 0.89 | 0.76 | 0.99 | **3433.07** | 1.18 | 2.25 |
| beetle | **9466.69** | 1.24 | 1.15 | 1.11 | 1.67 | 1.33 | 0.85 |
| eel | 1.20 | 1.42 | 1.16 | 0.48 | 0.60 | 0.73 | 1.16 |
| gannet | 0.84 | 0.84 | 0.84 | 0.84 | 0.07 | 0.84 | 0.07 |
| medusa | 16581.07 | 1.21 | 1.15 | 1.18 | 43745.30 | **79044.67** | 1.02 |
| ray | **5363.59** | 2238.56 | 1.92 | 1.14 | **32271.25** | 1.59 | 1.11 |
| teal | 1.36 | 1.25 | 1.37 | 1.40 | 0.42 | 1.30 | 0.34 |

Largest joint angle over 5 s, in radians. The CPG commands under one.
**Bit-for-bit repeatable across runs**, so every entry is one measurement and not
a sample.

* **bat, beetle and ray are destabilised by the tensor** when driven, and **bat
  and ray also with the actuators held completely still**. None of them runs
  away under the incumbent in either mode. Two of the three do it with no
  actuation at all, so it is not the controller.
* **medusa was already unstable** — **79044.67 rad** driven with the switch
  *off*, which is the solver exactly as it ships. A pre-existing defect, counted
  separately as `N-02`. Unlike the others it is quiet with the actuators still
  (1.02 rad), so that one *is* the drive.
* Every plan is stable at **dt = 0.0005**.

## What that means

Added mass goes into the mass matrix, and `apply`'s own comment explains at
length why: an explicit `F = -d(m_a v)/dt` diverges once the added mass exceeds
the structural mass, which it does by a factor of 38 on `ray`.

**Lift and drag do not.** They go into `xfrc_applied`, explicitly, and they are
only stable at `dt = 0.004` because the wings are carrying about three times the
entrained mass they should. The isotropic value is not merely wrong — it is
load-bearing.

Closing F-03 therefore needs one of:

1. **`dt = 0.001` or finer.** Four to eight times the evaluation cost. At the
   measured 74 s/generation that takes a 900-generation run from 21 h to 84 h or
   more.
2. **An implicit treatment of lift and drag**, the way added mass already has.
   The right fix, and much larger than a coefficient.
3. **Leave it open**, which is what the switch's default records.

## Two details worth keeping

**Driven and held-still are different probes, and neither dominates — and which
way round `beetle` falls depends on the lift model.** Under the logistic stall
blend it was stable driven (0.39 rad) and catastrophic held still (75958); under
the compact blend that closed F-05 it is catastrophic driven (9466.69) and quiet
held still (1.67). The *specific* reading reverses. What survives is the rule:
"hold the actuators still", the probe CLAUDE.md's lesson section prescribes, is
not a strictly *gentler* test than driving, and a plan can be stable under one
and not the other in either direction. Both are needed, and neither alone is
evidence of stability.

**Convergence under refinement is not monotone.** Under the logistic, `bat` was
worse at `dt = 0.002` (2404) than at `0.004` (300) before becoming stable at
`0.001`; under the compact blend `ray` does the same, 5363.59 at `0.004` and
2238.56 at `0.002`. So "refine until it stops" needs the whole sweep rather than
two points.

## Provenance

Measured on `main` **with** F-05's compact stall blend, which is what the
repository carries. The table above was originally written before that merge and
has been re-measured, because the stability section depends on the lift model
through the motion: sections A to C barely move (the mean squared direction
cosines go 0.2300 / 0.4137 / 0.3549 to 0.2306 / 0.4164 / 0.3516, and the 35.6 to
36.0% ratios are unchanged), while section D names a partly different set of
plans and one of them, `beetle`, reverses which probe catches it.

For the record, under the logistic blend the same sweep gave bat 300.49 driven
and 3108.20 still, beetle 0.39 and 75958.41, medusa 3275.52 and 11222.92 with an
incumbent of 2443.84, ray 27395.72 and 66929.76. The conclusion is the same under
both: several plans run away at the coarse step, none at the finest, and the
runaway happens with no actuation.

## Files

| | |
|---|---|
| `config.json` | the plans, the media, the timestep sweep, the runaway threshold |
| `run.py` | sections A–D |
| `results/result.json` | every number above, with provenance |
