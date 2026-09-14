# What this solver is, and what it is not

**This is not CFD. There is no Navier-Stokes solve, no mesh, no pressure
Poisson equation, no turbulence model and no wake feeding back into the
forces.** `dytiscidae/physics/fluid.py` is a reduced-order, quasi-steady
blade-element surrogate whose coefficients are part derived from potential-flow
theory, part standard boundary-layer correlation, and part calibrated
interpolation between regimes.

That is the right choice for this project — a free-wake panel method costs
hours per evaluation and the search would be finished before the first
candidate was scored — and `physics/wake.py` already says so in its own
docstring. This page says it once, for the whole solver, at the level a reader
needs before believing any number that comes out of it.

The audit behind every claim here is `docs/MATH_AUDIT.md`. The derivations are
in `derivations/`. The measurements are in `experiments/`.

## What the model computes

| term | status | source |
|---|---|---|
| dynamic pressure `q = rho U^2/2` | **derived** | Bernoulli |
| finite-wing lift slope `2 pi/(1+2/AR)` | **derived** | thin-airfoil theory + elliptic downwash |
| zero-lift angle from camber `-2 f/c` | **derived** | thin-airfoil theory, parabolic arc |
| induced drag `CL^2/(pi e AR)` | **derived** | elliptic loading; `e = 0.75` is calibrated |
| laminar skin friction `1.328/sqrt(Re)` | **derived** | Blasius |
| turbulent skin friction `0.074/Re^0.2` | **correlation** | 1/7-power, valid 5e5 < Re < 1e7 |
| post-stall `CN_max sin(2a)/2`, `CN_max sin^2 a` | **derived** from a flat-plate normal force; `1.98` is the measured plate value |
| Kramer rotational force `pi(0.75-x0) rho U w c^2 dr` | **textbook** | Sane & Dickinson |
| strip added mass `rho pi c^2/4` | **textbook** | 2D plate, potential flow |
| bluff added mass `Ca rho V` | **calibrated** | exact for a sphere, `3 pi/8` from Lamb for a disc |
| buoyancy `rho g V f_sub` | **derived** | Archimedes |
| jet thrust `rho Q^2/A` | **derived** | momentum flux |
| bluff axial + cross-flow split | **textbook** | Munk slender-body + Allen & Perkins cross-flow |
| `CL_max = 1.10 + 0.80 lev` | **calibrated** | no source |
| `alpha_stall = 11 + 26 lev` degrees | **calibrated** | no source |
| low-Re stall knockdown | **calibrated** | anchored at `Re = 1e5` |
| the 6-degree stall blend | **heuristic** | costs 11% of the attached lift slope |
| `lev = clip(k/0.30, 0, 1)` | **heuristic** | no source, and `k` here is a pitch rate |

Roughly: the *shapes* are derived and the *magnitudes at the regime boundaries*
are fitted. That is the honest summary of any engineering surrogate and it is
not a criticism; it is the specification.

## Where it is valid, and where it is extrapolating

| regime | status | why |
|---|---|---|
| attached flow, `alpha` below ~10 deg, `k` below ~0.1 | **usable, +/-15%** | the derived parts dominate; unsteady correction is small |
| flapping, `k` 0.1 - 0.3 | **usable, +/-30%** | Theodorsen's \|C(k)\| is 0.85 - 0.77 here, so quasi-steady over-predicts circulatory lift by 18 - 30% |
| flapping, `k` above 0.3 | **extrapolating** | \|C(k)\| below 0.7; the LEV term is a fit, not a model of the vortex |
| stroke reversal at low `U` | **extrapolating** | `k` is singular as `U -> 0` and the code's `1e-6` floor picks an answer |
| deep stall, `alpha` above 45 deg | **usable for drag, not for lift** | the plate branch is the right shape for `CD` and a fit for `CL` |
| paddling in water | **usable for magnitude, not for detail** | plate drag dominates and is textbook; ventilation near the free surface is not modelled at all |
| water entry | **diagnostic only** | `slam` is a rate term, reported and not fed to the dynamics |
| very low `Re`, below ~1e4 | **extrapolating** | laminar separation bubbles are not modelled; the knockdown acknowledges them without representing them |
| wing in another wing's wake | **not modelled** | no wake feedback at all |
| ground effect | **not modelled** | |
| compressibility | **not applicable** | irrelevant at these speeds |

## What the model cannot represent, by construction

* **Wake history.** No shed vorticity acts back on any surface, so there is no
  wing-wing interference, no stroke-to-stroke vortex recapture, and no wake
  capture at reversal. `physics/wake.py` draws the wake *from* the forces; it
  never feeds it back.
* **Spanwise flow.** Strip theory projects it out for lifting surfaces. Real
  flapping wings have strong spanwise flow, and it is part of what stabilises a
  leading-edge vortex.
* **Leading-edge vortex dynamics.** The LEV enters as an instantaneous function
  of the local pitch rate. A real LEV grows over a half-stroke and is shed at
  reversal, so the model's version peaks where the real one is weakest.
  See `derivations/reduced_frequency.md`.
* **Ventilation and cavitation.** A paddle near the free surface entrains air
  and produces a fraction of the modelled force.
* **Flexibility.** Bodies are rigid in the dynamics. Structural compliance is
  checked statically and never deflects a surface.
* **Off-diagonal added mass.** The added-mass tensor is 6x6; MuJoCo takes a
  scalar mass and a diagonal inertia, so translation-rotation coupling cannot
  be represented at all.

## Two places the model is not merely approximate

These are model *errors*, measured, with reproductions, and they are listed here
rather than only in the audit because they change what the surrogate's outputs
mean:

* **A pulsed jet's pumping work is not charged** — `experiments/jet_energy`
  measures 0.008% of it reaching the actuator. Jet thrust in this model is very
  nearly free. **J-01.**
* **A machine with no joints receives none of its added mass** —
  `experiments/added_mass` measures `+0.00 kg` reaching the integrator against
  `+32.20 kg` in the bookkeeping. **F-01.**

## The naming question

`lift_coefficient` and `drag_coefficient` read like they return *the* lift and
drag coefficients. They return a surrogate's. Renaming them
(`surrogate_lift_coefficient`) would make the call sites say so, and it is a
reasonable change — but they sit on the live evaluation path and are mirrored
in `mojo/src/fluid_gpu.mojo`, so renaming them is a change to the scoring path
and belongs in its own commit with its own cross-path agreement check, not in
an audit. This page and the module docstrings are the interim answer.

## The one honest signal the model already has

`FluidDiagnostics.clamped` is set when the force limiter engages, which means
the state was already outside the range the quasi-steady coefficients are valid
over. It is the only flag in the solver that says "do not believe this run", and
it is worth more than any coefficient. Two things about it:

* it is raised per element rather than per machine (**F-04**), so a
  200-strip machine can carry 200x the stated bound under the same boolean;
* `JetSet` has no equivalent (**J-02**), so an out-of-domain jet is silent.

Nothing else in the solver reports when a strip is extrapolating.
`FluidDiagnostics` already records `max_alpha`; recording `max` reduced pitch
rate beside it would make the `k` column of the table above visible in the
telemetry rather than only in this document.
