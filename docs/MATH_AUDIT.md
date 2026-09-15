# Mathematical and model-assumption audit

Every formula and every literal in `dytiscidae/physics/`, `dytiscidae/control/`
and the optimisation code, with: what it is, where it came from, what it
assumes, its units, whether it has been derived, whether it has been validated,
and what changes if it moves.

Written 2026-09-14 against commit `65e21d5`. It is a **survey, not a
refactor**: nothing in `dytiscidae/` was changed to produce it. What was added
is `experiments/`, `derivations/` and `tests/test_math.py`, so that every claim
below has a command behind it.

## How to read this

Each item carries a **derivation status** and a **validation status**.

| derivation status | means |
|---|---|
| `DERIVED` | follows from stated premises by algebra; the derivation is in `derivations/` or inline here |
| `TEXTBOOK` | a standard result used as-is, correctly, with the source nameable |
| `CORRELATION` | a fit to data that someone else took; honest, bounded, not derivable |
| `CALIBRATED` | a number chosen so the model behaves plausibly; no source |
| `HEURISTIC` | a number chosen because it worked; no source and no plausibility argument beyond "it is about right" |
| `NUMERICAL` | exists to stop a division, a singular solve or a divergence; not physics |
| `UNKNOWN` | the provenance could not be established from the repository |

| validation status | means |
|---|---|
| `MEASURED` | an experiment in `experiments/` reproduces the number |
| `TESTED` | an invariant in `tests/` asserts it on every run |
| `ASSERTED` | a comment claims it and nothing checks it |
| `UNCHECKED` | nothing claims it and nothing checks it |

Findings are ranked by **risk = (how wrong) x (how much the search can exploit
it)**. A coefficient that is 20% off in a term nothing selects on is low risk. A
term that lets a design get something for nothing is high risk whatever its
size, because this project's recorded history is that the search finds those
within a few hundred generations. The module prefix is `F` fluid, `S` structure,
`C` control, `E` energy, `J` jet, `O` optimisation.

---

## The findings, ranked

| id | finding | risk | status | reproduce with |
|---|---|---|---|---|
| **J-01** | a pulsed jet's pumping work is **100% uncharged**: thrust appears with no reaction torque on the driving joint | **critical** | measured | `experiments/jet_energy` |
| **F-01** | added mass written into `body_mass` **never reaches the mass matrix** for a jointless machine | **critical** | measured | `experiments/added_mass`, `tests/test_math.py` |
| **S-01** | hull buckling allowable is **4.8x non-conservative**: the `(t/D)^3` coefficient applied to `(t/r)^3` | **high** | measured | `experiments/analytic_vs_numerical` |
| **C-07** | the mobility identification is **underdetermined on 5 of 7 seed plans**, and its residual reads exactly zero because of it | **high** | measured | `experiments/rank_threshold` |
| **C-08** | the identified **parameter-side directions do not survive a reseed** (68-85 degree principal angles); the twist-side ones do | **high** | measured | `experiments/rank_threshold` |
| **F-03** | wing added mass is **isotropic**, overstating edgewise entrained mass by `(chord/thickness)^2` | **high** | measured | `experiments/added_mass` |
| **J-02** | jet thrust goes as the **square of the joint rate with no limiter** and no out-of-domain flag | **high** | measured | `experiments/jet_energy` |
| **F-02** | the leading-edge-vortex term keys on the **instantaneous pitch rate**, not the reduced frequency its docstring named | **high** | **half closed** — named correctly now, measured at a 157% spread across one stroke; the model choice is open | `tests/test_reduced_frequency.py` |
| **C-09** | the damping `lam` is about **30x too small**; at the incumbent value a command on land is worse than no command | **high** | measured | `experiments/damping_lambda` |
| **C-06** | a **diverged probe returns a measured zero** twist rather than a missing observation | medium | derived | `derivations/mobility_jacobian.md` |
| **F-05** | the stall blend puts **13.8% of the post-stall branch at zero incidence**; the realised lift slope is 11% below the docstring's | medium | measured | `experiments/analytic_vs_numerical` |
| **C-02** | the twist scaling's three `0.3`s are an **undeclared reference length in metres**, fixed across bodies of different size | medium | measured | `experiments/dimensional_check` |
| **C-03** | `rank = #{sigma > 0.08 sigma_0}` is an engineering threshold presented as a numerical rank; the identification's own noise floor is `1.13 sigma_0` | medium | measured | `experiments/rank_threshold` |
| **C-04** | a saturated intent delivers `sigma^2/(sigma^2+lam)` of a body's reach, not all of it -- 55% on a weak axis | medium | tested | `tests/test_math.py` |
| **F-06** | `CL_MAX = 1.8` is declared as "the largest CL the strip model will produce"; the model peaks at **2.280** | medium | tested | `tests/test_math.py` |
| **C-10** | the frequency mode's authority scales with `probe_time`, which nothing asserts is equal across the two identification paths | medium | measured | `experiments/analytic_vs_numerical` |
| **C-01** | the CPG parameter vector mixes radians with hertz, so `\|\|dp\|\|` has no units and "a unit vector in parameter space" has no physical content | medium | measured | `experiments/dimensional_check` |
| **O-01** | CMA-ES's eigen-update staleness counter is in **generations** where the reference algorithm counts **evaluations** -- a factor of `lambda` | medium | measured | `tests/test_math.py` |
| **F-04** | the force limiter clamps **per element**, not per machine, so the docstring's "60x the vehicle's weight" is 60x per panel | low | tested | `tests/test_math.py` |
| **E-01** | `k_iron = 0.02 * p_cont / 1000.0` hides a **reference speed of 1000 rad/s** in a bare literal | low | measured | `experiments/dimensional_check` |
| **E-02** | `km` is scaled by the literal `0.88`, which is `BLDC_OUTRUNNER.efficiency_peak` -- editing that table entry rescales every motor class | low | measured | `experiments/dimensional_check` |
| **O-02** | CMA-ES clips the population to `bounds` and then updates the covariance from the clipped samples; harmless while the optimum is inside the box, premature convergence when it is not | low | measured | `tests/test_math.py` |
| **C-05** | `control/cpg.py`'s module docstring states the SVD in the **transposed convention** from the one the code fits in | low | derived | `derivations/svd_control_basis.md` |
| **S-03** | `flapping_inertial_check` uses a radius of gyration of `0.45 b` with no source, and computes an `r_cg` that enters nothing but the printed note | low | derived | below |
| **S-02** | the Wagner slam coefficient is `(pi/tan beta)^2` against the usually quoted `(pi/(2 tan beta))^2` -- 4x, conservative | low, **unresolved** | measured | `experiments/analytic_vs_numerical` |

---

## The critical two

### J-01 -- a pulsed jet produces thrust and nothing pays for the water

`JetSet.actuator_work` returns `0.0` with the comment "accounted through the
driving actuator's torque". `JetSet.apply` writes a force into
`data.xfrc_applied` and no joint torque anywhere. `TriphibianEnv.step` charges
the battery from `data.actuator_force`.

Measured over 4 s of a 4 L bell at 1.5 Hz, submerged:

| | |
|---|---|
| ideal pumping work, `(1/2) rho Q^3 / A^2` | 141.61 J |
| actuator work charged, jet **on** | 114.1222 J |
| actuator work charged, jet **off** | 114.1111 J |
| **difference the jet made to the bill** | **+0.0111 J = +0.008%** |

The 114 J is the cost of swinging the bell's own inertia and is the same
whether the jet fires or not. This is free thrust in water, in a module written
so the search could discover a medusa. Every previous instance of this pattern
in the project's history took a few hundred generations to be found and
exploited.

The fix is one term: `tau_reaction = p * dV/dtheta`, opposing the contraction,
which the existing battery accounting then picks up unchanged — and which makes
the orifice trade the `jet.py` docstring argues for actually exist.

### F-01 -- a jointless machine swims with none of its entrained water

`FluidSolver.apply` writes added mass into `model.body_mass` rather than
applying it as a force, correctly, for the reason `derivations/
rigid_body_dynamics.md` derives: an explicit `-d(m_a v)/dt` term is a feedback
loop of gain `m_a/m_body` and diverges above unity, which is the NaN the module
docstring reports.

But MuJoCo marks a body `simple` when no joint attaches to it and takes those
DOFs' mass from `dof_M0`, a constant computed at compile time. A runtime
`body_mass` edit updates `cinert` and never reaches `M`:

    dry:      cinert[9]=0.16000  Mdiag=[0.16  0.16  0.16 ]
    edited:   cinert[9]=32.36000 Mdiag=[0.16  0.16  0.16 ]
    setConst: cinert[9]=32.36000 Mdiag=[32.36 32.36 32.36]

Measured end to end: a jointless model reports `+32.20 kg` in `body_mass` and
**`+0.00 kg`** to the integrator; the same body with one hinge gets all 32.20 kg.
`mj_setConst` is what rebuilds it, and the string appears nowhere in this
repository.

Machines with at least one joint are unaffected, which is most of the fleet.
This search has produced jointless designs — arch33's mission champion had zero
actuated degrees of freedom — and for those, water costs nothing to accelerate
through while every diagnostic reports the added mass as applied.

`tests/test_physics.py::test_added_mass_is_anisotropic` reads
`model.body_mass[bid]` back and checks the bookkeeping. That is the gap: nothing
asked MuJoCo what it did with the number.

---

## The rest, in order

### S-01 -- hull buckling is 4.8x non-conservative

```python
p_cr = 0.6 * 2.0 * material.E / (1.0 - nu**2) * (wall / max(radius, 1e-6)) ** 3
```

The classical long-cylinder result: a unit-length ring buckling into `n` lobes
collapses at `(n^2-1) E' I / r^3` with `I = t^3/12` and `E' = E/(1-nu^2)`. The
first available mode is `n = 2`, giving

    p_cr = E/(4(1-nu^2)) (t/r)^3  =  2E/(1-nu^2) (t/D)^3,   D the DIAMETER.

The code uses the `(t/D)^3` prefactor with `(t/r)^3`. Measured on a PETG hull,
`t = 2 mm`, `r = 60 mm`: classical 22 046 Pa, the code before its own knockdown
176 367 Pa (**8.0x**), the code's allowable after the 0.6 knockdown 105 820 Pa
(**4.8x**). The docstring calls buckling "the real constraint" on hull design
and says `p_cr` "scales as `E (t/r)^3`", which is true; the prefactor is the one
belonging to the other form.

### C-07 / C-08 -- the mobility identification, and what it does and does not identify

A machine with `n` joints presents `P = 3n+1` parameters to `n_probes = 24`:

| plan | params | | plan | params | |
|---|---|---|---|---|---|
| bat | 25 | underdetermined | medusa | 49 | underdetermined |
| beetle | 19 | determined | ray | 28 | underdetermined |
| eel | 16 | determined | teal | 31 | underdetermined |
| gannet | 25 | underdetermined | | | |

Where it is underdetermined, `lstsq` returns the minimum-norm solution inside
the 24-dimensional row space of whichever directions were drawn, and
`residual_fraction` reads **exactly 0.000** — not because the fit is good but
because an underdetermined system always fits. The two determined plans read
0.14 to 0.58. A reader treating `resid = 0` as evidence of linearity has it
backwards.

The consequence, measured across four probe seeds per (body, medium):

    ||J_a - J_b||_2 / sigma_0  =  1.132   (95% CI 1.039 - 1.219, n = 21)

Two matrices drawn independently with the same spectrum sit near `sqrt(2)`. By
Weyl's inequality no singular value below `||dJ||_2` is resolvable, so **no
singular value is resolvable from a single identification**.

But the two sides of the decomposition behave completely differently. Mean
principal angle between two seeds' rank-`r` subspaces:

| rank | modes (CPG parameter side) | effects (body twist side) |
|---|---|---|
| 1 | 67.8 deg | 36.0 deg |
| 3 | 80.5 deg | 33.3 deg |
| 5 | 84.5 deg | 29.5 deg |

with singular values reproducing to a coefficient of variation of 0.15-0.21.

**What the machine can do is identified. How to ask for it is not.** That is the
headline of the control audit and it subsumes both C-03 and C-09: the threshold
and the ridge are tuning parameters on an estimator whose parameter side is
close to arbitrary. The move that changes it is `n_probes`, which needs to
exceed `max(3n+1)` over the fleet; `medusa` at 49 sets the bar.

### C-09 -- the damping is about 30x too small

Cross-validated over the same 84 identifications: solve the coefficients on one
identification, send the resulting **CPG parameter offset**, and measure the
delivered twist under an independent identification of the same body. Sending
no command scores exactly 1.0.

| lam/lam0 | in-sample | out-of-sample | max joint excursion |
|---|---|---|---|
| 0 | 0.0000 | 1.4294 | 1.719 |
| 0.1 | 0.0269 | 1.1551 | 1.272 |
| **1 (incumbent)** | **0.0968** | **0.9738** | **0.917** |
| 10 | 0.2669 | 0.8341 | 0.512 |
| **30 (best)** | **0.3830** | **0.8234** | **0.359** |
| 100 | 0.5343 | 0.8499 | 0.235 |
| 1000 | 0.8503 | 0.9488 | 0.068 |

An interior minimum at 30x, best on 11 of 21 (body, medium) pairs, never the
incumbent. Paired difference `+0.1503` (95% CI `+0.1008` to `+0.2006`), an
18.3% higher out-of-sample error. Per medium: air wants 10x, water and land 30x,
and **on land the incumbent scores 1.124 — worse than sending no command at
all.**

The comment justifying `0.01` reads "Measured cond(A) is 13-50 across arch31
elites, so this is well inside the regime where a modest ridge is enough." A
condition number bounds how far an inverse *can* amplify a perturbation; it does
not say how much ridge to apply, because that depends on how large the
perturbation is. `derivations/damped_least_squares.md` derives what does decide
it.

### F-02 -- `reduced_freq` is not a reduced frequency

```python
omega_s      = np.einsum("ni,ni->n", omega, s_hat)
reduced_freq = np.abs(omega_s) * p.chord / (2.0 * U_safe)
```

`omega_s` is the body's instantaneous rate **about the span axis** — the pitch
rate. The quantity computed is the reduced *pitch rate* `|alpha_dot| c / (2U)`,
a real dimensionless group from the dynamic-stall literature, with the same
dimensions as `k` and a different value. For `alpha(t) = alpha_0 sin(Omega t)`,

    kappa = alpha_0 |cos(Omega t)| * k

so the leading-edge-vortex term the model uses carries an amplitude factor in
radians that `k` does not, and **collapses to zero at both stroke extremes** —
the point in the stroke where a real leading-edge vortex, having grown over the
preceding half-stroke, is largest. The model has the history backwards rather
than merely absent. The docstring prints `k = omega * c / (2 * U)` and describes
"a wing that is flapping fast relative to its own translation", which is `k`.

The same `omega_s` is used, correctly, for the Kramer rotational force on the
following lines. One quantity, two roles, right in one of them.

This is worth a factor of two in lift: at 25 degrees, `lev = 1` raises `CL` from
0.941 to 1.898 — measured 2.02x.

**Half closed.** The variable is now `reduced_pitch_rate`, the docstring states
`kappa = |alpha_dot| c / (2U)` and says in terms what it is not, and
`tests/test_reduced_frequency.py` measures the difference: over one stroke at
`k = 0.1728`, `kappa` runs 0 to 0.1210, a **157% spread of its own mean**, zero
at both extremes, following `alpha_0 |cos(Omega t)| k` to 2.8e-17. **No number
changed.**

The model choice is deliberately left open, because keying the LEV on something
else changes `CL` on every wing by up to 2.02x. `derivations/
reduced_frequency.md` costs the three options: leave it, plumb the CPG
frequency into the solver, or give the LEV a history variable (non-dimensional
travel since reversal, the Wagner / Beddoes-Leishman treatment). The third is
right and is the largest change in that document.

### F-03 -- wing added mass has no direction in it

`FluidSolver.apply` argues at length that bluff added mass must be anisotropic —
"treating them alike with a flat `Ca = 0.5` told the search that a plate and a
sphere of equal volume cost the same to shake" — and projects
`Ca_eff = d_hat^T diag(Ca) d_hat`. The wing branch on the next line is
`rho pi chord^2 / 4 * dr`, the plate's **normal** added mass, with no direction.

Measured on a 1.0 x 0.2 m, 2 mm strip in seawater, 12 kg dry, jointed so the
mass matrix sees it:

| push | m_added | 2D strip theory |
|---|---|---|
| chord | 32.20 kg | 0.00322 kg |
| span | 32.20 kg | 0 |
| normal | 32.20 kg | 32.20 kg |

Spread 0.000%. Edgewise the overstatement is `(chord/thickness)^2 = 10 000`. The
same geometry declared `BLUFF` comes out 6.8x apart broadside to edgewise.

`hydrodynamic_sweep_check`'s docstring identifies folding as "the escape" from
the constraint that closes the route to take-off, and says the structural check
cannot see it. The added-mass model cannot see it either, in the opposite
direction: folding buys nothing inertially.

### J-02 -- jet thrust has no limiter

`thrust = rho Q^2 / A` with `Q` taken straight from the joint rate goes as the
square of that rate, unbounded. Measured: 1 649 N peak on an 8.6 kg machine,
20 g, from one bell, with a 36.6 m/s jet. `FluidSolver` clamps its own forces at
60x vehicle weight and raises `diag.clamped` so the scorer knows the run left
the valid domain; `JetSet` has no equivalent and nothing is flagged.

### C-06 -- a diverged probe is a measured zero

`TriphibianEnv.identify`'s `step_fn`:

```python
if not np.all(np.isfinite(self.root_pos())):
    return np.zeros(6)
```

A probe whose rollout went non-finite contributes a **measured zero response**,
and the least squares reads it as evidence that the direction does nothing. This
is the rule CLAUDE.md already records — "a quantity that means 'I could not
measure this' must not share a value with a quantity that means 'I measured
zero'" — in a place it has not yet been applied. `thrust_margin` was the
previous instance.

### F-05 -- the attached branch is never clean

`w = 1/(1 + exp(-(|alpha| - alpha_stall)/6deg))` is a logistic centred on
`alpha_stall`, and a logistic never reaches zero. At `Re = 2e5`, `k = 0`, where
`alpha_stall` clips to 11 degrees:

    w(0 deg) = 0.138        w(2 deg) = 0.182

**13.8% of the post-stall plate branch is mixed in at zero incidence**, and
there is no angle of attack at which the attached branch stands alone. Measured
lift slope at 2 degrees: 3.726 against the 4.189 the docstring names — 11.0%
low, everywhere, for every wing. The blend itself is smooth (largest slope jump
0.018/rad on a 0.09 degree grid), so the optimiser sees no kink; it is simply
wider than the linear region it is handing over from.

### C-02 -- the twist scaling hides a length

```python
scale = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
```

For an angular rate in rad/s to be added in quadrature with a linear rate in
m/s, the three `0.3`s must be a **length in metres**. They are an implicit
reference radius: the basis is computed in a space where 1 rad/s of roll counts
as 0.3 m/s of surge. Everything downstream inherits it — every singular value,
`reach`, `cond(A)`, `lam = 0.01 tr(G)/r`, and the `0.08 sigma_0` rank threshold.
It is also a *fixed* length applied to bodies of different size, so the same
machine scaled by two gets a different basis for a reason that is not physics.
The comment says only "so that rotation and translation are comparable", which
states the purpose and not the quantity.

### C-03 -- `0.08 sigma_0` is a usefulness threshold, not a rank

Swept over 84 identifications:

| tau | rank | recon error | control error | cond | max excursion |
|---|---|---|---|---|---|
| 0.001 | 6.00 | 0.0000 | 0.0972 | 21.5 | 0.909 |
| 0.020 | 5.46 | 0.0044 | 0.1019 | 17.3 | 0.894 |
| **0.080** | **4.48** | **0.0337** | **0.1614** | **7.6** | **0.781** |
| 0.200 | 3.32 | 0.1197 | 0.3455 | 3.8 | 0.472 |

Everything monotone, no knee, nothing picking `0.08` out from its neighbours. Of
the 420 non-leading singular values, 30.5% fall below `0.08 sigma_0` and **30.2%
land within a factor of two of it** — a genuine spectral gap would put that
second number near zero. And the rank the threshold reports is unanimous across
four probe seeds on only **47.6%** of (body, medium) pairs; at `tau = 0.005` it
is 100%.

`0.08` is defensible as an engineering threshold. It is not a numerical rank,
the docstring's "genuinely independent things this machine can do" is a claim
about numerical rank, and the sweep is smooth enough that nothing is lost by
making it a parameter.

### C-04 -- a saturated intent does not deliver a body's full reach

From the SVD of the damped solve, the delivered fraction on an orthogonal basis
has a closed form: `phi_i = sigma_i^2 / (sigma_i^2 + lam)`. With
`sigma = [4, 3, 2, 1, 0.5, 0.25]` and `lam = 0.01 mean(sigma^2)`:

    delivered fraction = [0.997, 0.994, 0.988, 0.952, 0.832, 0.553]

matching `phi_i` to 1e-9. `coeffs_for_twist` documents

> As a fraction, "+1 heave" means "climb as hard as this body climbs" on every
> one of them.

True only where `sigma_i^2 >> lam`. On the weakest axis of this spectrum the
command delivers 55% of what it claims, and the shortfall falls entirely on the
weak axes rather than uniformly.

### F-06 -- `CL_MAX = 1.8` against a model peak of 2.280

```python
#: The largest lift coefficient the strip model will produce (...)
CL_MAX = 1.8
```

`lift_coefficient` clips at `1.2 * cl_max` with `cl_max` reaching 1.9, so the
model's ceiling is 2.28. Measured peak over the whole `alpha`/`k` domain: 2.280.
`MAX_WING_LOADING` is derived from `CL_MAX`, so the Tier-0 wing-loading gate is
27% conservative. The declaration and the value disagree; the parenthetical in
the same comment describes the real ceiling and then a different number is
assigned.

### C-10 -- the frequency mode's authority is set by the probe window

`d(theta)/d(frequency) = A * 2 pi t * cos(psi)` grows linearly in `t`: measured
84.3x from `t = 0.05 s` to `t = 4.0 s`, against the 80x exact linearity
predicts. Amplitude and phase sensitivities are bounded; frequency's is not. So
two identification paths with different `probe_time` do not produce comparable
bases. Both default to 1.2 s today, and nothing asserts they must agree —
CLAUDE.md records that their `n_probes`/`max_modes` were once 8/4 against 24/6.

### C-01 -- the CPG parameter vector is dimensionally inhomogeneous

`p = [amplitude(n), phase(n), offset(n), frequency(1)]` mixes radians
(dimensionless) with hertz (`1/s`). `||dp||_2` therefore has no units,
`modes` being a unit vector in that space has no physical content, and the
relative weighting between "move the frequency by 1 Hz" and "move a phase by
1 rad" is set by an arbitrary norm. The probe scale `0.35` inherits it: it is a
17% step on a 2 Hz base frequency and a modest one in phase.

### O-01 -- CMA-ES counts generations where the reference counts evaluations

```python
if self._eig_stale < max(1, int(self.lam / (10 * self.n * (self.c1 + self.cmu)))):
```

`_eig_stale` is incremented once per `tell`, so it counts **generations**. The
reference algorithm compares `counteval - eigeneval` — **evaluations** — against
the same expression. Measured at `n = 24`: the interval is 6 generations = **78
evaluations** where the reference interval is **6.0 evaluations**, so the
eigendecomposition is refreshed about `lambda` times less often than prescribed
and `B`, `D` and `c_inv_sqrt` are stale for that long. The strategy still
converges on a 24-dimensional sphere (`||mean - optimum|| = 7e-7` after 300
generations), so this is a rate effect, not a correctness one.

### F-04 -- the force limiter is per element

```python
weight = float(self._dry_mass.sum()) * GRAVITY + 1.0
limit = 60.0 * weight
```

applied to `fmag = np.linalg.norm(F, axis=1)`, a per-element magnitude. The
docstring says the clamp is "to a multiple of the vehicle's weight"; it is that
multiple **per panel**. Measured with 12 panels: 7 788 N against a 649 N
per-element bound, exactly 12x. A 200-strip machine can therefore carry 200
times the stated bound while `diag.clamped` reports the same boolean.

### E-01 / E-02 -- two literals in the power train

`k_iron = 0.02 * p_cont / 1000.0`, used as `p_iron = k_iron * omega`. For that
to be a power, `k_iron` must be W/(rad/s), so the `1000` is a **reference speed
of 1000 rad/s** and the expression means "iron loss is 2% of continuous power at
1000 rad/s". Nothing says so.

`km = 0.05 * (mass/0.1)**0.75 * spec.efficiency_peak / 0.88`. The `0.88` is
`BLDC_OUTRUNNER.efficiency_peak` written as a literal, so editing that table
entry silently rescales `km` — and through it `stall_torque`, `max_speed` and
every copper loss — for **every** motor class, including the two it does not
describe.

### O-02 -- CMA-ES updates the covariance from clipped samples

```python
pop = self.mean[None, :] + self.sigma * y.T
if self.bounds is not None:
    pop = np.clip(pop, self.bounds[0], self.bounds[1])
```

and `tell` computes the new mean and `ys = (selected - old_mean)/sigma` from the
clipped population. Clipping is a nonlinear projection, so the samples are no
longer distributed as `N(mean, sigma^2 C)` and the rank-mu update is biased
toward the boundary. The standard treatments are a box-constraint penalty or
resampling.

`bounds=(-4.0, 4.0)` **is** set, on `control/train.py`'s controller
optimisation — the main use of this class. Measured over 200 generations on a
60-dimensional sphere, which is the shape of a linear policy:

| | coordinates clipped | final sigma | mean[0] |
|---|---|---|---|
| optimum inside the box (target 2.0) | 0.11% | 0.0104 | 2.0026 |
| optimum outside the box (target 6.0) | 17.53% | 0.0047 | 3.9948 |

So the bias is negligible in normal use and the failure mode is the familiar
one: with the optimum outside the box the mean parks just inside the boundary
and `sigma` collapses to 0.0047, i.e. the search stops exploring rather than
pressing against the wall. Policy weights feed a `tanh`, so +/-4 is generous
and the interior case is the expected one. Recorded so that narrowing the
bounds is a decision rather than a surprise.

### C-05 -- the SVD convention in the docstring is the transpose of the one in the code

`control/cpg.py`'s module docstring:

> Take the SVD, `J = U S V^T`. The leading columns of `V` are the parameter
> directions that move the machine most; the matching columns of `U` describe
> what motion each one actually produces

That is the column convention, `y = J dp` with `J` of shape `6 x P`.
`basis_from_probes` fits `Y = X J` with `J` of shape `P x 6`, where `U` is the
parameter side and `V` the twist side — and correctly takes `modes` from `U`
and `effects` from `Vt`. The code is self-consistent; the prose describes its
transpose. A documentation defect, listed because this is exactly the confusion
the convention exists to prevent.

### S-03 -- a radius of gyration with no source, and a dead variable

```python
r_cg   = 0.40 * semi_span
i_root = wing_mass * (0.45 * semi_span) ** 2
```

`0.45 b` is between the uniform-rod value `b/sqrt(3) = 0.577 b` and the
linearly-tapered value `b/sqrt(10) = 0.316 b`, so it is a plausible pick for a
tapered wing and it has no source. Against a uniform rod it under-predicts the
root moment by `(0.45/0.577)^2 = 0.61`, i.e. 39% non-conservative; against a
tapered one it over-predicts by 2.0x. `r_cg` is computed and enters nothing but
the `note` string.

### S-02 -- the Wagner slam coefficient, unresolved

`k = min((pi/tan beta)^2, 250)` against the `(pi/(2 tan beta))^2` usually
quoted for a Wagner wedge — 74.5 against 18.6 at 20 degrees of deadrise, a
factor of 4. Conservative for structure, so low risk, and **this audit does not
claim the code is wrong**: the source could not be established from the
repository and both numbers are recorded side by side in
`experiments/analytic_vs_numerical`. It needs a citation, not a patch.

---

## The constant ledger

Every literal in the physics and control path. `where` is the function; `moves`
is what changes if the number moves.

### `physics/fluid.py`

| constant | value | status | units | moves |
|---|---|---|---|---|
| finite-wing slope `2 pi/(1+2/AR)` | — | `DERIVED` thin-airfoil + elliptic downwash | 1/rad | all attached lift |
| camber shift `2 * f/c` | — | `DERIVED` thin-airfoil, parabolic arc | rad | zero-lift angle |
| induced drag `CL^2/(pi e AR)` | — | `DERIVED` elliptic loading | 1 | drag at high CL |
| Blasius `1.328/sqrt(Re)` | — | `TEXTBOOK` | 1 | friction below transition |
| 1/7-power `0.074/Re^0.2` | — | `CORRELATION`, valid 5e5-1e7 | 1 | friction above transition |
| transition centre `5.7`, width `4.0` | log10(Re) | `CALIBRATED` | log10(Re), 1/decade | where friction jumps |
| plate normal force `1.98` | — | `TEXTBOOK` flat plate | 1 | post-stall drag, paddling |
| cross-flow `CD_CROSSFLOW` | `1.1` | `TEXTBOOK` circular cylinder | 1 | hull side force, weathercock stability |
| `CL_max` | `1.10 + 0.80*lev` | `CALIBRATED` | 1 | all post-stall lift |
| `alpha_stall` | `11 + 26*lev` deg | `CALIBRATED` | deg | where stall begins |
| low-Re knockdown | `0.55 + 0.45 log10(Re)/5` | `CALIBRATED`, anchored Re=1e5 | 1 | small-wing stall |
| blend width | `6` deg | `HEURISTIC` | deg | **F-05**: 11% of the lift slope |
| `lev` scale | `k/0.30` | `HEURISTIC` | 1 | LEV onset |
| lift clip | `1.2 * cl_max` | `HEURISTIC` | 1 | the model's CL ceiling, **F-06** |
| Oswald `e` | `0.75` | `CALIBRATED` | 1 | induced drag |
| Kramer `C_rot = pi(0.75-x0)` | — | `TEXTBOOK` Sane & Dickinson | 1 | reversal force |
| pitch axis `x0` | `0.25` | `TEXTBOOK` quarter-chord | 1 | `C_rot` |
| strip added mass `rho pi c^2/4` | — | `TEXTBOOK` 2D plate | kg/m | **F-03** |
| bluff `Ca_i = 0.5(e_j+e_k)/(2 e_i)` | — | `CALIBRATED`, exact for a sphere, `3 pi/8` from Lamb for a disc | 1 | fin inertia |
| `Ca` clip | `[0.05, 10]` | `NUMERICAL` | 1 | binds from `D/h = 20` |
| force limit | `60 * weight` | `HEURISTIC` | 1 | **F-04** |
| `alpha` fold `1e-12`, `U_safe 1e-6`, `mu 1e-12` | — | `NUMERICAL` | — | nothing, until `U -> 0` |

### `physics/medium.py`

| constant | value | status | notes |
|---|---|---|---|
| `GRAVITY` | 9.80665 | `TEXTBOOK` | standard gravity |
| `P_ATM` | 101325 | `TEXTBOOK` | standard atmosphere |
| `AIR`, `FRESHWATER`, `SEAWATER` | — | `TEXTBOOK` | densities and viscosities at a reference state |
| density blend | linear in `f` | `DERIVED` | a volume average is exact for buoyancy |
| viscosity blend | geometric in `f` | `CALIBRATED` | sound because the exponents sum to 1; the *choice* of geometric is a modelling one |
| `half_height` floor | `1e-3` | `NUMERICAL` | keeps a thin panel's ramp finite |
| Airy orbitals `exp(kz)` | — | `TEXTBOOK` | deep-water limit; shallow water is a different formula |

### `physics/structure.py`

| constant | value | status | moves |
|---|---|---|---|
| elliptic centroid `4/(3 pi)` | — | `DERIVED`, quadratured | root moment |
| `load_factor` | `3.0` | `CALIBRATED` | spar sizing |
| `cycles` | `1e5`, `1e6`, `1e4`, `1e3` | `CALIBRATED` per load case | allowables |
| radius of gyration | `0.45 b` | `HEURISTIC` | **S-03** |
| mass centroid `r_cg` | `0.40 b` | `HEURISTIC` | nothing — dead |
| sweep `drag_coefficient` | `1.9` | `TEXTBOOK` flat plate | the binding constraint on wings |
| compliant knockdown | `0.35` | `HEURISTIC` | whether non-avian plans are feasible at all |
| `min_tip_speed` | `1.5` m/s | `CALIBRATED` | sweep load |
| tip-speed clip | `60` m/s | `NUMERICAL` | — |
| buckling prefactor | `2E/(1-nu^2)` on `(t/r)^3` | **wrong form** | **S-01**, 8x |
| buckling knockdown | `0.6` | `CALIBRATED` | hull wall |
| UDL deflection `qL^4/(8EI)` | — | `TEXTBOOK`, quadratured | efficiency warning only |
| Wagner `k = (pi/tan b)^2` | — | `UNKNOWN` | **S-02** |
| Wagner cap | `250` | `NUMERICAL` | small-deadrise limit |
| flat-panel bending | `0.30 p b^2/t^2` | `TEXTBOOK` | flat-bottom penalty |
| deadrise floor | `1` deg | `NUMERICAL` | — |
| pump efficiency | `0.35` | `CALIBRATED` | ballast power |

### `physics/materials.py`

Every entry is `CALIBRATED` with a stated manufacturing route, which is the
right way to carry them. Two notes: `allowable_stress` applies **no** knockdown
below 1e3 cycles, so the curve is flat there with a kink exactly at 1e3; and it
clamps rather than extrapolates beyond 1e5, which is the conservative choice and
is now `TESTED`.

### `physics/energy.py`

| constant | value | status | moves |
|---|---|---|---|
| gearbox `0.97^stages`, `log_5(ratio)` | — | `CALIBRATED` | drivetrain loss |
| `km = 0.05 (m/0.1)^0.75 * eta/0.88` | — | `CALIBRATED`, `0.88` is a cross-reference | **E-02** |
| `k_visc = 2e-5 * m/0.1` | — | `CALIBRATED` | no-load loss |
| `k_iron = 0.02 * p_cont/1000` | — | `CALIBRATED`, `1000` is rad/s | **E-01** |
| thermal headroom | `0.35 * p_cont` | `CALIBRATED` | overload gate |
| overdraw penalty | `1 + 0.25(x-1)` | `HEURISTIC` | battery sag |
| `SEALING_MASS_FRACTION` | `0.08` | `CALIBRATED` | mass budget |
| `SHAFT_SEAL_MASS` | `0.035` kg | `CALIBRATED` | mass per penetration |
| `SHAFT_SEAL_FRICTION` | `0.02` N.m | `CALIBRATED` | power per sealed joint |

### `physics/jet.py`

| constant | value | status | moves |
|---|---|---|---|
| `thrust = rho Q^2/A` | — | `DERIVED` momentum flux | all jet thrust |
| `refill_efficiency` | `0.25` | `CALIBRATED` | reverse thrust |
| `actuator_work` | returns `0.0` | **wrong** | **J-01** |
| thrust bound | none | **missing** | **J-02** |
| `_prev_v` | assigned, never read | dead | nothing |

### `control/cpg.py`

| constant | value | status | moves |
|---|---|---|---|
| base amplitude | `0.45 * span/2` | `HEURISTIC` | starting gait |
| base phase | `linspace(0, pi, n)` | `CALIBRATED` (a travelling wave) | starting gait |
| frequency clip | `[0.1, 20]` Hz | `CALIBRATED` | — |
| offset margin | `0.05 * span` | `NUMERICAL` | joint travel |
| `INTENT_AUTHORITY` | `0.5` | `CALIBRATED`, **measured on 48 bodies** | command gain |
| twist scale | `[1,1,1,0.3,0.3,0.3]` | `HEURISTIC` + hidden length | **C-02** |
| rank threshold | `0.08 * sigma_0` | `HEURISTIC` | **C-03** |
| damping | `0.01 * tr(G)/r` | `HEURISTIC` | **C-09** |
| damping floor | `+1e-12` | `NUMERICAL` | singular solve only |
| `n_probes` | `24` | `HEURISTIC` | **C-07** |
| `probe_scale` | `0.35` | `HEURISTIC` | linearity of the fit |
| `probe_time` | `1.2` s | `HEURISTIC` | **C-10** |
| `max_modes` | `6` | `DERIVED` — `J` has 6 columns, so it truncates nothing | — |
| `describe()` cutoff | `0.25` | cosmetic | telemetry text only |

### `evolution/cmaes.py`

`lam`, `mu`, `weights`, `mueff`, `cc`, `cs`, `c1`, `cmu`, `damps`, `chiN` all
match the reference `(mu/mu_w, lambda)`-CMA-ES exactly and are `DERIVED`/
`TEXTBOOK`. The two departures are **O-01** and **O-02**. `sigma` clip
`[1e-8, 1e3]`, eigenvalue floor `1e-20` and the degenerate-covariance restart
are `NUMERICAL`.

### What is *not* in this ledger

The scoring weights, ladder rungs and archive parameters. Those are already
inventoried in `docs/ROADMAP.md` under "What is set by measurement, and what is
typed", which is the right place for them: they decide what the search
optimises, not what the physics is. This document deliberately does not
duplicate that list, and the two do not overlap.

---

## What was checked and found sound

An audit that only reports faults is not an audit. The following were checked
against an independent computation and agree.

* **Every formula in the project is dimensionally correct.** 48 of 48, across
  six modules, in `experiments/dimensional_check`.
* **The damped inverse is exactly the Tikhonov solution it claims to be** —
  `(A A^T + lam I)^-1 A b` matched to machine precision, and `twist_of` is the
  transpose of the same `A`.
* **The SVD basis reconstructs its own Jacobian** to 3.2e-15, with orthonormal
  `modes`, unit `effects` and descending non-negative `authority`.
* **`reach` is genuinely the per-axis maximum.** `max_{||c||=1} |(A^T c)_j| =
  ||A[:,j]||` by Cauchy-Schwarz, which is what the code computes.
* **The CPG kinematic Jacobian** matches central differences to 5.7e-10.
* **The elliptic lift centroid, tube section properties and cantilever
  deflection** match quadrature to 1e-9 or better.
* **The disc added-mass coefficient is within 17.8% of Lamb's result**, and the
  docstring's claim of "within 18%" is exact — the ratio is `3 pi/8`,
  scale-free. The sphere case is exact.
* **Buoyancy is exactly `rho g V`**, and the diagnostic agrees with the force.
* **The augmented mass matrix stays symmetric positive definite** over a
  300-step water rollout, and `reset()` restores dry inertia exactly.
* **A passive body in still fluid never gains speed**, in air and water, wing
  and bluff — so no force term has a sign error.
* **Force coefficients depend only on dimensionless groups**: the same problem
  at two length scales with matched Reynolds number gives a force ratio equal to
  `(U1/U0)^2 (L1/L0)^2` to 1e-9.
* **The power train obeys the first law** for all three motor classes:
  electrical power is never negative and never below mechanical.
* **CMA-ES's constants match the reference algorithm**, `C` stays symmetric
  positive definite, and the mean converges to 7e-7 on a 24-dimensional sphere.
* **`CD >= 0` and `CD >= Cf` over the whole domain**; `CL` is odd in `alpha` and
  stays inside its declared envelope.
* **Skin friction is correctly non-monotone in `Re`** — it rises through
  transition at `Re ~ 5.5e5`, as it physically must.
* **The camber shift `alpha_0 = -2 f/c`** is the thin-airfoil result, correctly
  applied, and the docstring's worked example (6% camber, 7 degrees) is right.
* **The batched and unbatched probe paths are bit-identical**:
  `max_abs_sigma_diff = 0.0`.

---

## What this audit does not cover

* **`envs/`** beyond the constants above. The ladder rungs, gates and blends
  live in `docs/ROADMAP.md`.
* **`learning/ppo.py`**. GAE, the clipped surrogate, the clipped value loss and
  the KL early stop match the standard algorithm on a read; none of it has been
  checked against an independent implementation, so it is `ASSERTED`, not
  `MEASURED`.
* **`evolution/` beyond `cmaes.py`** — the archive, descriptors, curator,
  auditor and scout.
* **The Mojo GPU port** (`mojo/`), which reimplements the coefficient models.
  `mojo/tests/test_coeff_gpu.py` checks it against the numpy versions, so any
  finding above applies to both, but the port has not been separately audited.
* **Whether any of this changes a score.** Every finding here is about the
  model. What it is worth in fitness is a different experiment, and running it
  means re-scoring an archive, which needs a run this container does not have.

## How to keep this true

* `tests/test_math.py` re-measures the open findings on every run and prints
  them under `[gap ]` with their identifier. The suite still passes; a finding
  that stops being true turns into a `FAIL` telling you to close it.
* `experiments/dimensional_check` fails if a source line it restates changes,
  so a dimensional check cannot outlive the code it describes.
* Every experiment writes `results/result.json` with the commit, the seed, the
  numpy version and the elapsed time, so a number in this document can be traced
  to the run that produced it.
