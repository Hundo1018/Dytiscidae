# derivations/

One document per quantity the project's physics or control depends on,
derived from its own premises rather than copied back out of the
implementation.

Every file has the same seven sections, in this order:

| section | what belongs in it |
|---|---|
| **Definition** | what the symbol means, in words, before any algebra |
| **Assumptions** | every premise the derivation needs, listed so it can be attacked |
| **Derivation** | the algebra, from the definition to the result |
| **Dimensional analysis** | the result's units, worked out, not asserted |
| **Numerical implementation** | what the code actually computes, and where it departs |
| **Validation** | the measurement that checks it, and its result |
| **Failure conditions** | where the derivation stops being true |

Two rules:

1. **The derivation does not read the implementation.** It starts from a
   definition and gets to a result. Only the *Numerical implementation*
   section is allowed to look at the code, and its job is to say where the
   two differ.
2. **Validation names a command.** "Verified" with no command behind it is
   the thing this directory exists to replace. Where a quantity has not
   been measured, the section says so.

| file | quantity |
|---|---|
| `reduced_frequency.md` | `k = omega c / (2 U)`, and what the code computes instead |
| `aerodynamic_coefficients.md` | `CL(alpha, Re, AR, k)`, `CD`, and which parts are derived |
| `rigid_body_dynamics.md` | `M(q) qdd + C qd + g = tau + F`, and added mass inside `M` |
| `mobility_jacobian.md` | `J = d(twist)/d(params)`, identified rather than derived |
| `svd_control_basis.md` | `J = U S V^T`, what `modes`, `effects` and `authority` are |
| `damped_least_squares.md` | `(A A^T + lam I)^-1 A b`, and what `lam` buys |

Findings that came out of writing these are in `docs/MATH_AUDIT.md`, with the
experiment that reproduces each.
