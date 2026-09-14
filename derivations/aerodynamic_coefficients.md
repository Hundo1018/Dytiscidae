# Aerodynamic coefficients: what is derived and what is fitted

## Definition

For a strip of chord `c` and span `dr` at incidence `alpha` in a flow of speed
`U` and density `rho`, the section forces are written

    q  = (1/2) rho U^2                        dynamic pressure, Pa
    L  = q * (c dr) * CL(alpha, Re, AR, k)    lift,  N,  normal to the flow
    D  = q * (c dr) * CD(alpha, Re, AR, CL)   drag,  N,  along the flow

`CL` and `CD` are dimensionless functions. The entire modelling content of a
blade-element method is in what those two functions are.

**This is a surrogate model, not a solution of the Navier-Stokes equations.**
`CL` and `CD` below are partly derived from potential-flow theory, partly
standard boundary-layer correlations, and partly a fitted interpolation between
regimes with no derivation at all. Which is which is the point of this
document; `docs/model_validity.md` states the same thing at the level of the
whole solver.

## Assumptions

1. **Strip theory.** Each spanwise strip is treated as a 2D section in the
   local flow, with the spanwise velocity component projected out. Spanwise
   pressure communication is ignored.
2. **Quasi-steady.** The coefficients are functions of the instantaneous state.
   The error this carries is quantified in `reduced_frequency.md`.
3. **No wake.** No shed vorticity acts back on the surface, so no
   wing-wing interference, no ground effect, no stroke-to-stroke recapture.
4. **Thin section.** Thickness affects nothing but the bluff branch.
5. **Incompressible.** Trivially satisfied here.

## Derivation

### Dynamic pressure, from Bernoulli

Along a streamline in steady incompressible flow, `p + (1/2) rho u^2` is
constant. Bringing a stream of speed `U` to rest raises the pressure by
`(1/2) rho U^2`, so `q` is the pressure scale of the problem. Forces on a body
of area `S` therefore scale as `q S`, and the coefficients are what remains
after that scaling is removed. This is why a coefficient is a function of
dimensionless groups alone — checked directly in `tests/test_math.py::
test_force_coefficients_do_not_depend_on_the_units`, which builds the same
problem at two length scales with the viscosity chosen to hold Reynolds number
fixed and confirms the force ratio is `(U1/U0)^2 (L1/L0)^2` to 1e-9.

### The lift slope: derived

Thin-airfoil theory solves the potential-flow problem for a 2D plate with the
Kutta condition at the trailing edge and gives

    CL_2D = 2 pi (alpha - alpha_0),                                    (1)

`2 pi` per radian, with `alpha_0` the zero-lift angle. For a **finite** wing,
the trailing vortex system induces a downwash that reduces the effective
incidence. For an elliptic loading the downwash is uniform and the induced
angle is `alpha_i = CL / (pi AR)`. Substituting `alpha_eff = alpha - alpha_i`
into (1):

    CL = 2 pi (alpha - CL/(pi AR))
    CL (1 + 2/AR) = 2 pi alpha
    CL = [ 2 pi / (1 + 2/AR) ] alpha.                                  (2)

That is the lift-slope correction the code uses, derived, not fitted. `AR -> ∞`
recovers `2 pi`; `AR = 4` gives `4.19`.

### Camber: derived

Thin-airfoil theory for a circular-arc section of camber `f/c` gives the
zero-lift angle

    alpha_0 = -2 (f/c)     radians.                                    (3)

So a 6% cambered section still lifts at `2 * 0.06 = 0.12 rad = 6.9 deg` below
geometric zero. The solver applies exactly this as `alpha = alpha + 2 * camber`.
Correct, and one of the few places in the file where a literal has a derivation
behind it.

### Induced drag: derived

The same elliptic-loading argument gives the induced drag as the streamwise
component of the tilted lift vector:

    CD_i = CL alpha_i = CL^2 / (pi AR),

and for a non-elliptic loading an efficiency factor `e <= 1` is inserted:

    CD_i = CL^2 / (pi e AR).                                           (4)

`e` is not derivable — it depends on the planform. The code fixes `e = 0.75`.

### Skin friction: standard correlations, not derivations

The Blasius solution of the laminar flat-plate boundary layer gives a
one-sided, length-averaged friction coefficient

    Cf_lam = 1.328 / sqrt(Re).                                         (5)

For a turbulent boundary layer with a 1/7-power velocity profile the
corresponding result is

    Cf_turb = 0.074 / Re^0.2,     valid roughly 5e5 < Re < 1e7.        (6)

(5) is derived; (6) is a correlation fitted to data, honestly so. Transition
sits near `Re = 5e5` for a smooth plate at low turbulence and moves with
roughness and pressure gradient — it is not a property of `Re` alone.

### Post-stall: a flat plate, not an airfoil

Above stall the flow separates and the section behaves like a flat plate at
incidence, whose normal force is `CN ~ CN_max sin(alpha)` resolved into lift
and drag as

    CL ~ CN cos(alpha) = CN_max sin(alpha) cos(alpha)
                       = (CN_max/2) sin(2 alpha)                       (7)
    CD ~ CN sin(alpha) = CN_max sin^2(alpha)
                       = CN_max (1 - cos(2 alpha)) / 2.                (8)

(7) and (8) are the two branches the code uses post-stall. A flat plate normal
to the flow has `CN_max ~ 1.98`, which is the `1.98` in the drag expression —
so `CD_p` is derived from (8) with a measured plate constant, and the
corresponding lift branch uses `CL_max` instead, which is not.

### Where the derivations stop

Four things in the implementation have no derivation, in decreasing order of
how much they move the answer.

| quantity | value | what it is |
|---|---|---|
| `CL_max` | `1.10 + 0.80 * lev` | fitted interpolation |
| `alpha_stall` | `11 deg + 26 deg * lev` | fitted interpolation |
| low-Re knockdown | `0.55 + 0.45 log10(Re)/5` | fitted, anchored at Re = 1e5 |
| blend width | `6 deg` sigmoid | smoothing choice |
| `e` (Oswald) | `0.75` | planform-dependent, fixed |
| `lev` scale | `k / 0.30` | fitted |

`1.10` is a reasonable static flat-plate value and `1.9` is in the range
reported for insect wings with a stable leading-edge vortex; `11 deg` and
`37 deg` bracket static and LEV-augmented stall plausibly. None of that is a
derivation and none of it is a citation. They are **calibrated coefficients**
of a surrogate, and the model's predictions carry their uncertainty.

## Dimensional analysis

    [q]  = (kg/m^3)(m/s)^2 = kg/(m s^2) = N/m^2 = Pa                   ok
    [L]  = (N/m^2)(m^2)(1) = N                                         ok
    [Re] = (kg/m^3)(m/s)(m) / (kg/(m s)) = 1                           ok
    [k]  = (1/s)(m)/(m/s) = 1                                          ok
    [CL_max] = 1,  [alpha_stall] = rad = 1                             ok

All checked in `experiments/dimensional_check`. Two literals there carry
undeclared units: the `5.0` dividing `log10(Re)` is five decades of Reynolds
number (so the knockdown reaches 1.0 at `Re = 1e5`), and the `5.7` in the skin
friction blend is `log10(Re)` at the transition centre, i.e. `Re = 5e5`.
Neither is written down in the code.

## Numerical implementation

```python
lev         = np.clip(reduced_freq / 0.30, 0.0, 1.0)
cl_max      = 1.10 + 0.80 * lev
alpha_stall = np.radians(11.0 + 26.0 * lev)
alpha_stall *= np.clip(0.55 + 0.45*np.log10(max(re,10))/5.0, 0.5, 1.0)
cl_alpha    = 2.0*np.pi / (1.0 + 2.0/max(ar, 0.5))
cl_linear   = cl_alpha * alpha
cl_plate    = cl_max * np.sin(2.0*alpha)
w           = 1/(1 + exp(-(|alpha| - alpha_stall)/radians(6.0)))
cl          = (1-w)*cl_linear + w*cl_plate
return np.clip(cl, -1.2*cl_max, 1.2*cl_max)
```

Three departures worth recording.

**The attached branch is never clean.** The sigmoid `w` is centred on
`alpha_stall` with a 6 degree width, and a logistic is `0.5` at its centre and
never reaches `0`. At `Re = 2e5`, `k = 0` the knockdown clips to 1.0 so
`alpha_stall = 11 deg`, and

    w(0 deg) = 1/(1 + e^{11/6})  = 0.138
    w(2 deg) = 1/(1 + e^{9/6})   = 0.182

So **13.8% of the post-stall plate branch is mixed in at zero incidence**, and
there is no angle of attack at which the attached branch stands alone. The
measured lift slope at 2 degrees is `3.726` against the `4.189` that (2) gives
and the docstring names — **11.0% low**, everywhere, for every wing in the
project. Measured in `experiments/analytic_vs_numerical`, check B. Recorded as
**F-05**.

**`cl_plate` uses `CL_max` where (7) uses `CN_max/2`.** The code's post-stall
lift peaks at `cl_max`, i.e. `1.10` statically, where (7) with the plate's own
`CN_max = 1.98` would give `0.99`. Close, and the agreement is a coincidence of
the chosen `1.10` rather than a derivation — the drag branch does use `1.98`
and the lift branch does not, so the two branches are not resolved from the
same normal force.

**The clip is at `1.2 * cl_max`, not at `CL_MAX`.** The largest lift
coefficient the model can produce is `1.2 * 1.9 = 2.28`.
`dytiscidae/envs/triphibian.py` declares

```python
#: The largest lift coefficient the strip model will produce ...
CL_MAX = 1.8
```

and derives `MAX_WING_LOADING` from it. Measured peak over the whole
`alpha`/`k` domain: `2.280`, so `CL_MAX` is 27% below what it claims to be and
the Tier-0 wing-loading gate is 27% conservative. Checked in
`tests/test_math.py`. Recorded as **F-06**.

## Validation

Every claim above has a command behind it.

| claim | where | result |
|---|---|---|
| coefficients depend only on dimensionless groups | `tests/test_math.py` | force ratio matches `(U1/U0)^2(L1/L0)^2` to 1e-9 |
| `CD >= 0` over the whole domain | `tests/test_math.py` | min 0.00597 |
| `CD >= Cf` everywhere | `tests/test_math.py` | worst deficit 3.4e-8 |
| `CL` odd in `alpha` at zero camber | `tests/test_math.py` | exact, 0 |
| `CL` inside its own `1.2 CL_max` envelope | `tests/test_math.py` | ratio 1.0000 |
| LEV raises post-stall `CL` | `tests/test_math.py` | 0.941 -> 1.898 at 25 deg |
| the blend is C1 across stall | `experiments/analytic_vs_numerical` | largest slope jump 0.018/rad on a 0.09 deg grid |
| attached slope against (2) | `experiments/analytic_vs_numerical` | 3.726 vs 4.189, -11.0% |
| `CD` equals its three named terms | `experiments/analytic_vs_numerical` | exact, 0 |
| `Cf` is not monotone in `Re` | `tests/test_math.py` | rises through transition at `Re ~ 5.5e5` |
| model peak `CL` against `CL_MAX` | `tests/test_math.py` | 2.280 vs 1.8 |
| dimensions of every expression | `experiments/dimensional_check` | 48 sound, 0 failing |

## Failure conditions

* **`k` above ~0.2**, where quasi-steady over-predicts circulatory lift by 30%
  or more. Not reported by any telemetry.
* **Very low Reynolds number.** (5) and (6) are flat-plate results; below
  `Re ~ 1e4` a real section's behaviour is dominated by laminar separation
  bubbles that neither branch represents. The `0.55 + 0.45 log10(Re)/5`
  knockdown is a fitted acknowledgement of this, not a model of it.
* **Anything the strip assumption breaks**: low aspect ratio, strong spanwise
  flow, tip vortices on a paddle, a surface in another surface's wake.
* **Post-stall in water.** (7) and (8) are plate results in unbounded flow. A
  paddle near the free surface ventilates, and a ventilated paddle produces a
  fraction of the modelled force.
* **Extrapolation past the force limiter.** Once a candidate is outside the
  valid range, the solver clamps and sets `diag.clamped`. That flag is the only
  honest signal in the model that a result should not be believed, and it is
  per element rather than per machine — see **F-04**.
