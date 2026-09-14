# 20 closed forms against independent computations: 18 agree, 2 do not

```
PYTHONPATH=. python experiments/analytic_vs_numerical/run.py
```

Where a quantity has a derivative, it is differenced. Where it has an integral,
it is quadratured. Where it has a textbook coefficient, the textbook derivation
is redone from its own premises and the two numbers are printed side by side.

## The two that differ

### G. Hull buckling is 4.8x non-conservative

```python
p_cr = 0.6 * 2.0 * material.E / (1.0 - nu**2) * (wall / radius) ** 3
```

The classical long-cylinder result, derived from its own premises: a unit-length
ring of wall `t` and radius `r` buckling into `n` lobes collapses at
`p_cr = (n^2 - 1) E' I / r^3` with `I = t^3/12` per unit length and
`E' = E/(1-nu^2)`. The first available mode is `n = 2`, so

    p_cr = 3 E' t^3 / (12 r^3) = E/(4(1-nu^2)) * (t/r)^3
         = 2E/(1-nu^2) * (t/D)^3        with D the DIAMETER.

The code uses the `(t/D)^3` coefficient `2E/(1-nu^2)` applied to `(t/r)^3`,
which is the same expression with radius substituted for diameter — a factor of
`2^3 = 8`.

Measured on a PETG hull, `t = 2 mm`, `r = 60 mm`:

| | Pa |
|---|---|
| classical `n = 2` ring | 22 046 |
| the code, before its own 0.6 knockdown | 176 367 (**8.0x**) |
| the code's allowable, after the knockdown | 105 820 (**4.8x**) |

The docstring says "the critical pressure scales as `E * (t/r)^3`", which is
true; the prefactor is the one belonging to the other form. The 0.6 imperfection
knockdown is applied on top of an eight-fold overestimate, so the check that the
docstring calls "the real constraint" on hull design passes designs that the
classical result rejects. Recorded as **S-01**.

### B. The lift slope is 11% below the value its own docstring names

`lift_coefficient`'s docstring: "Below stall the strip behaves like a finite
wing with the Helmholtz lift-slope correction `2*pi / (1 + 2/AR)`."

At `AR = 4` that is 4.189 per radian. Measured by central difference through
the real function at 2 degrees, `Re = 2e5`, `k = 0`: **3.726**, which is 11.0%
low.

The cause is the stall handover. `w = 1/(1 + exp(-(|alpha| - alpha_stall)/6deg))`
is a logistic centred on `alpha_stall = 11 deg`, and a logistic never reaches
zero:

    w(0 deg) = 1/(1 + e^{11/6}) = 0.138
    w(2 deg) = 1/(1 + e^{9/6})  = 0.182

**13.8% of the post-stall plate branch is mixed in at zero incidence, and there
is no angle of attack at which the attached branch stands alone.** The blend is
smooth — largest jump in `dCL/dalpha` between adjacent samples is 0.018 per rad
on a 0.09 degree grid, so the optimiser sees no kink — it is simply wider than
the linear region it is supposed to be handing over from. Recorded as **F-05**.

## The eighteen that agree

| check | result |
|---|---|
| A. CPG Jacobian, analytic vs central difference | max relative element error 5.7e-10 |
| B. CL blend is C1 across stall, gliding and strong LEV | largest slope jump 0.018 and 0.030 per rad |
| B. CL is odd in alpha at zero camber | exact, 0 |
| C. CD is non-negative over the whole sweep | min 0.00711 |
| C. CD equals profile + induced + separated | exact, 0 |
| D. elliptic lift centroid `4/(3 pi)` vs quadrature | rel 5.7e-9 |
| E. tube area vs quadrature over the annulus | rel 4.0e-16 |
| E. tube second moment vs quadrature | rel 5.3e-13 |
| E. section modulus `Z = I/ro` | exact |
| F. cantilever UDL tip deflection vs moment-area integration | rel 4.5e-15 |
| F. `spar_deflection` returns that over the semi-span | exact |
| H. disc added-mass coefficient vs Lamb | +17.8%, and the docstring claims "within 18%" |
| H. sphere added-mass coefficient vs 0.5 | exact |
| I. Wagner slam coefficient | 4x the commonly quoted value — see below |
| J. allowable at 1e5 cycles equals the declared knockdown | exact |
| J. allowable stress is monotone in cycles | holds |
| K. `d(net buoyancy)/d(depth)` vs the Boyle analytic derivative | rel 6.2e-6 |

### Two of those deserve a note

**The CPG frequency column.** `d(theta)/df = A * 2 pi t * cos(psi)` grows
linearly in `t`, measured at 84.3x from `t = 0.05 s` to `t = 4.0 s` against the
80x exact linearity predicts. The identification integrates over a 1.2 s probe
window, so **the frequency mode's authority is set by that duration**, and two
identification paths with different `probe_time` do not produce comparable
bases. `TriphibianEnv.identify` and `identify_batch` both default to 1.2 s;
nothing asserts that they must agree, and CLAUDE.md records that their
`n_probes`/`max_modes` were once mismatched.

**The Wagner slam coefficient.** The code uses `(pi/tan beta)^2`; the
coefficient usually quoted for a Wagner wedge is `(pi/(2 tan beta))^2`, four
times lower — 74.5 against 18.6 at 20 degrees of deadrise. This is conservative
for structure, and it is an unresolved source question rather than a
demonstrated error: the check records both numbers and does not call either
wrong. Recorded as **S-02**, unresolved.

**The disc added mass.** The code's `Ca_i = 0.5 (e_j + e_k)/(2 e_i)` gives
`D/(2h)` for a disc of diameter `D` and thickness `h`; Lamb's result referred to
the displaced volume is `4D/(3 pi h)`. The ratio is `3 pi / 8 = 1.178` exactly,
scale-free, so the docstring's "within 18% of Lamb's result" is correct. The
clip at `Ca = 10` binds from `D/h = 20` upward, beyond which the code stops
tracking Lamb at all.
