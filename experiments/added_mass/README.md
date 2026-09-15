# The added mass is bookkept correctly, applied isotropically, and sometimes not applied at all

```
PYTHONPATH=. python experiments/added_mass/run.py
```

One free body carrying one strip, fully submerged in seawater, gravity off,
12 kg dry. Pushed with a constant 40 N along each of its own three axes;
`m_eff = F / a` read out of the integrator. No model introspection — the number
comes from MuJoCo, not from the code that wrote it.

## Q1. Does the number written into `body_mass` reach the mass matrix?

`FluidSolver.apply` writes entrained fluid into `model.body_mass` and lets
MuJoCo invert it implicitly. `tests/test_physics.py::
test_added_mass_is_anisotropic` reads `model.body_mass[bid]` back and checks the
bookkeeping. Nothing asks MuJoCo what it did with it.

| model | `body_mass` says | the integrator says | `body_simple` |
|---|---|---|---|
| no joints (one free body) | +32.20 kg | **+0.00 kg** *(before the fix)* → **+32.20 kg** *(after)* | `[1, 1]` |
| one hinge added | +32.20 kg | +32.20 kg | `[1, 0, 0]` |

Before the fix the jointless model discarded all 32.20 kg: its effective mass
was 12.00 kg against 44.20 kg for the same body with a hinge in it. **F-01 is
now closed** and the run above reports both at 44.20 kg; the history is kept
here because the mechanism is the point.

**Mechanism.** MuJoCo marks a body `simple` when no joint attaches to it, and
takes those DOFs' mass from `dof_M0`, a constant computed at compile time. A
runtime `body_mass` edit updates `cinert` but never reaches `M`. Reproduced
directly:

    dry:      cinert[9]=0.16000  Mdiag=[0.16  0.16  0.16 ]
    edited:   cinert[9]=32.36000 Mdiag=[0.16  0.16  0.16 ]
    setConst: cinert[9]=32.36000 Mdiag=[32.36 32.36 32.36]

`mj_setConst` is what rebuilds it, and nothing in this project called it — the
string did not appear anywhere outside the mujoco package itself.
`FluidSolver._publish_inertia` now does, gated on whether any panel-carrying
body is `simple` (none of the seed plans' are, so a real machine pays nothing)
and handed a throwaway `MjData`, because `mj_setConst(model, data)` resets
`data` to `qpos0`.

**Why it matters here.** This search has produced jointless designs: arch33's
mission champion had zero actuated degrees of freedom. Such a machine swims
with none of its entrained water, so it accelerates 3.68x more easily than the
physics the solver computed, for free — and every diagnostic in the project
(`FluidDiagnostics.added_mass`, the structural overlay, the showcase) reports
the added mass as applied. Recorded as **F-01**.

Machines with at least one joint are unaffected, which is most of the fleet.

## Q2. Is the wing branch anisotropic?

`FluidSolver.apply` argues at length that bluff added mass must be
anisotropic — "treating them alike with a flat Ca = 0.5 told the search that a
plate and a sphere of equal volume cost the same to shake" — and projects a
directional coefficient onto the direction of motion. The wing branch on the
next line is `rho pi chord^2 / 4 * dr`, the 2D added mass of a plate
accelerating **normal to its own surface**, with no direction in it.

WING element, 1.0 m span x 0.2 m chord x 2 mm, jointed model:

| push | m_dry | m_eff | m_added | 2D strip theory |
|---|---|---|---|---|
| chord (+X) | 12.00 | 44.20 | 32.20 | 0.00322 |
| span (+Y) | 12.00 | 44.20 | 32.20 | 0.00000 |
| normal (+Z) | 12.00 | 44.20 | 32.20 | 32.20132 |

Spread across the three directions: **0.000%**.

The same geometry declared BLUFF, which the code does project:

| push | m_added |
|---|---|
| chord (+X) | 0.60 |
| span (+Y) | 0.02 |
| normal (+Z) | 4.10 |

Spread 258.7%, and 6.8x between broadside and edgewise.

**The size of the error.** Edgewise, the solver entrains 32.20 kg where 2D
strip theory for a 2 mm plate gives 0.00322 kg — a factor of **10 000**, which
is exactly `(chord/thickness)^2`, because the wing branch has no direction in
it. Recorded as **F-03**.

**What it costs the search.** A wing that folds before entering the water, or
slices edgewise through it, is charged the full broadside entrained mass. The
`hydrodynamic_sweep_check` docstring already identifies folding as "the escape"
from the constraint that closes the route to take-off, and says the structural
check cannot see it. The added-mass model cannot see it either, in the opposite
direction: folding buys nothing inertially.

## What would fix each

**F-01 — done.** `mj_setConst` after the mass edit, gated on detection and
handed a scratch `MjData`. Measured cost: 1.8 us per call with the scratch
reused, and zero for every seed plan because none of them triggers it. See
`tests/test_added_mass.py`.

**F-03.** The bluff branch already computes `Ca_eff = d_hat^T diag(Ca) d_hat`
from the element's three extents. A wing strip has three extents too
(`dr`, `chord`, thickness); routing it through the same projection with
`Ca_normal = pi c^2 / (4 V)` would make the wing branch directional without a
new model. This changes forces in water for every machine, so it is a
measurement to run, not a patch to apply.
