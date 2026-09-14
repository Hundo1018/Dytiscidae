# 48 formulas, 48 dimensionally sound, 11 literals carrying an undeclared unit

```
PYTHONPATH=. python experiments/dimensional_check/run.py
```

Exits non-zero if any dimension fails **or if any source fragment it claims to
restate is no longer in the file**. That second condition is the point: a
dimensional restatement can drift away from the code it describes, so every
check carries the exact line it is a restatement of and cannot outlive it.

## Result

All 48 expressions across `physics/fluid.py`, `physics/medium.py`,
`physics/jet.py`, `physics/structure.py`, `physics/energy.py` and
`control/cpg.py` produce the dimension their name and docstring claim. No
formula in the project is dimensionally wrong.

The interesting output is the other column. Eleven expressions are
dimensionally sound **only because a literal written as a bare number carries a
physical unit**. Four of those are ranked findings.

| where | literal | what it must be for the dimensions to work |
|---|---|---|
| `cpg.py` `scale = [1,1,1,0.3,0.3,0.3]` | `0.3` | a **length in metres**. An angular rate in rad/s is only commensurable with a linear rate in m/s through a length. Every singular value, `reach`, `cond(A)` and the damping `lam` inherit it, and it is a *fixed* length applied to bodies of different sizes. **C-02** |
| `energy.py` `k_iron = 0.02 * p_cont / 1000.0` | `1000` | a **reference speed in rad/s**. `p_iron = k_iron * omega` is only a power if `k_iron` is W/(rad/s), so the expression means "iron loss is 2% of continuous power at 1000 rad/s". **E-01** |
| `energy.py` `km = 0.05 * (mass/0.1)**0.75 * eta / 0.88` | `0.88` | `BLDC_OUTRUNNER.efficiency_peak`, written as a literal. Editing that table entry silently rescales `km` for every motor class. **E-02** |
| `cpg.py` `lam = ... + 1e-12` | `1e-12` | nothing: it is a bare number added to a quantity with units of squared gain. Harmless only because it matters solely when `trace(G)` is exactly zero, which is the case it exists for. **C-03** |
| `fluid.py` `0.55 + 0.45*log10(Re)/5.0` | `5.0` | five decades of Reynolds number: the knockdown reaches 1.0 at `Re = 1e5`. |
| `fluid.py` `exp(-(log10(re) - 5.7)*4.0)` | `5.7`, `4.0` | `log10(Re)` at the transition centre (`Re = 5e5`) and 1/decade of blend width. |
| `fluid.py` `np.radians(11.0 + 26.0*lev)` | `11.0`, `26.0` | degrees, correctly made explicit by `np.radians`. |
| `medium.py` `mu_air**(1-f) * mu_water**f` | — | raising a dimensional quantity to a fractional power is meaningful here *only* because both factors carry the same unit and the exponents sum to 1. It is a weighted geometric mean, and it is sound for that reason alone. |
| `jet.py` `omega / span` | `span` | a joint-angle range in radians, hence dimensionless, so `omega/span` is 1/s. |
| `fluid.py` `lev = clip(k/0.30, 0, 1)` | `0.30` | correctly dimensionless — a reduced-rate value, with no source. |
| `cpg.py` CPG parameter vector | — | mixes radians (amplitude, phase, offset) with hertz (frequency) in one flat array, so `||dp||` has no units and "a unit vector in parameter space" has no physical content. **C-01** |

## Why the source-fragment tie matters

One check in this file was caught by its own mechanism while being written: the
restatement of the swept-surface root moment used `m^4` for
`integral(c(r) r^3 dr)`, which carries `m . m^3 . m = m^5`. The check failed,
the restatement was wrong, and the code was right. That is the failure mode
this file is for.
