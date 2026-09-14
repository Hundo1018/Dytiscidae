# Rigid-body dynamics, and why added mass belongs inside `M(q)`

## Definition

For a rigid multibody system with generalised coordinates `q`,

    M(q) qdd + C(q, qd) qd + g(q) = tau + J^T F_ext                    (*)

where `M` is the mass matrix, `C qd` collects Coriolis and centrifugal terms,
`g` is the gravity generalised force, `tau` is actuation, and `F_ext` is
everything else — in this project, the fluid.

The question this document answers is where the entrained fluid goes. It is not
a matter of taste: put it in `F_ext` and the integration diverges in water; put
it in `M` and it does not. The project already made the right call and wrote
down the symptom; what follows is the reason.

## Assumptions

1. Bodies are rigid. Structural compliance is checked statically
   (`physics/structure.py`) and never enters the dynamics.
2. `M(q)` is symmetric positive definite. This is not an assumption about the
   world, it is a theorem about kinetic energy, and it is what makes (*)
   solvable — see the derivation.
3. The fluid is quasi-steady and its loads are known functions of the current
   state. This is what allows the fluid to appear as `F_ext` at all.
4. Added mass is instantaneous: the fluid's momentum is a function of the
   body's current velocity, with no history. This is the same assumption as
   quasi-steady aerodynamics, made about the inertial rather than the
   circulatory part.

## Derivation

### (*) from Lagrangian mechanics

Kinetic energy of a multibody system is quadratic in the generalised
velocities:

    T(q, qd) = (1/2) qd^T M(q) qd.                                     (1)

This is the *definition* of `M(q)`, and it settles two of its properties
without further work.

* **Symmetry.** Only the symmetric part of `M` contributes to the quadratic
  form, so `M` may be taken symmetric with no loss.
* **Positive definiteness.** `T > 0` for any `qd != 0` — a moving system has
  positive kinetic energy — so `qd^T M qd > 0`, which is positive definiteness.
  `M` is therefore invertible, and (*) has a unique solution for `qdd`.

With potential `V(q)` and Lagrangian `L = T - V`, the Euler-Lagrange equations
are

    d/dt ( dL/d qd_k ) - dL/d q_k = Q_k                                (2)

for generalised force `Q`. Take the two pieces of (2) in turn. From (1),

    dL/d qd_k = sum_j M_kj qd_j,

so

    d/dt ( dL/d qd_k ) = sum_j M_kj qdd_j
                       + sum_{i,j} (d M_kj / d q_i) qd_i qd_j.         (3)

and

    dL/d q_k = (1/2) sum_{i,j} (d M_ij / d q_k) qd_i qd_j - dV/d q_k.  (4)

Substituting (3) and (4) into (2),

    sum_j M_kj qdd_j
      + sum_{i,j} [ dM_kj/dq_i - (1/2) dM_ij/dq_k ] qd_i qd_j
      + dV/dq_k = Q_k.

The bracket, symmetrised in `i` and `j`, is the Christoffel symbol of the first
kind,

    C_kij = (1/2) [ dM_kj/dq_i + dM_ki/dq_j - dM_ij/dq_k ],            (5)

giving exactly (*) with `C(q,qd)_kj = sum_i C_kij qd_i` and `g = dV/dq`.

Two structural properties follow from (5) and both are load-bearing:

* **`Mdot - 2C` is skew-symmetric** for this choice of `C`. Hence
  `qd^T (Mdot - 2C) qd = 0`, which is the statement that the Coriolis terms do
  no net work — the passivity property that every energy-based stability proof
  in robot control rests on.
* **`C` is not unique.** Only `C qd` is determined by the dynamics; the
  skew-symmetry above is a property of the Christoffel choice, not of every
  valid `C`. A `C` obtained some other way may be equally correct in (*) and
  lack it.

MuJoCo assembles `M` by the composite rigid-body algorithm and the bias term by
recursive Newton-Euler, so none of this is computed symbolically here. It is
written out because the *next* section is a statement about `M`, and a
statement about `M` needs (1) to mean anything.

### Added mass: the term, and why its placement decides stability

A body accelerating through a fluid accelerates fluid with it. For potential
flow, the fluid's kinetic energy is quadratic in the body's velocity:

    T_fluid = (1/2) v^T M_a v                                          (6)

with `M_a` the **added-mass tensor** — symmetric positive semi-definite,
6 x 6 in general, and a property of the body's shape and the fluid density
alone. Because (6) has the same form as (1), the fluid's contribution can be
carried in the Lagrangian directly, and the system's mass matrix becomes

    M_total(q) = M_body(q) + M_a(q).                                   (7)

That is the derivation of "added mass belongs in the mass matrix": it is not a
numerical convenience, it is where the term comes from. The fluid's kinetic
energy is part of `T`, so it is part of `M`.

**What goes wrong if it is moved to `F_ext` instead.** Write the same physics
as an external force, `F_a = -M_a vdot`, and integrate explicitly: the force at
step `k` uses the acceleration from step `k-1`. The scheme is then

    M_body v_{k+1} = ... - M_a vdot_k

which, linearised, has amplification factor `-M_a / M_body` per step. The
iteration diverges when

    ||M_a|| > ||M_body||.                                              (8)

This is the classical added-mass instability of partitioned fluid-structure
coupling, and (8) is its condition: it does not depend on the time step, so it
cannot be fixed by taking smaller steps. For this project's wings in water the
ratio is not marginal — a single 1 m x 0.2 m strip entrains 32.2 kg against a
12 kg vehicle — so an explicit treatment is unconditionally unstable, which is
exactly the "NaN accelerations after 1.3 s of simulated water time" the
`FluidSolver` docstring reports. Folding it into `M` and inverting implicitly
is unconditionally stable for any ratio, because `M_body + M_a` is still
symmetric positive definite by (1) and (6).

**Gravity.** Entrained fluid has inertia and no weight. Adding `M_a` to
`body_mass` makes the integrator apply `M_a g` as a weight, so it must be
cancelled. `FluidSolver` does exactly that (`F[:, 2] += m_add * GRAVITY`) and
says why. Note the corollary: with gravity switched off in the model, that
compensation becomes an unbalanced upward force — which is a real trap for
anyone writing a zero-gravity probe, and is handled explicitly in
`tests/test_math.py`.

### What is lost by making `M_a` a scalar

`M_a` in (6) is a 6 x 6 tensor. MuJoCo's `body_mass` is a scalar and
`body_inertia` is a diagonal 3-vector, so only an isotropic translational part
and a diagonal rotational part can be represented. The solver handles the
translational anisotropy by evaluating the quadratic form along the
instantaneous direction of motion,

    Ca_eff = sum_i (d_hat . e_i)^2 Ca_i,                               (9)

which is `d_hat^T diag(Ca) d_hat` — the correct effective coefficient for
motion along `d_hat`, rebuilt each step. That is a good approximation and the
code explains it. It is applied to bluff elements only; wing strips take the
plate's *normal* added mass with no projection at all, and the consequence is
measured in `experiments/added_mass`.

The off-diagonal (translation-rotation coupling) terms of `M_a` cannot be
represented at all, and the rotational part is approximated as
`m_add * mean(r^2)` applied equally to all three principal axes.

## Dimensional analysis

    [M]   = kg (translational block), kg.m^2 (rotational block)
    [C qd] = kg.m/s^2 = N   and   kg.m^2/s^2 = N.m
    [g]   = N, N.m
    [tau] = N.m

Added mass:

    m_add,wing  = rho pi c^2 / 4 * dr    [kg/m^3][m^2][m] = kg         ok
    m_add,bluff = Ca rho V               [1][kg/m^3][m^3] = kg         ok
    I_add       = m_add * lever^2        [kg][m^2] = kg.m^2            ok

All three are checked in `experiments/dimensional_check`.

## Numerical implementation

`FluidSolver.apply`, per step:

```python
m_add = where(is_wing, rho*pi*chord^2/4*dr, ca_eff*rho*volume) * scale
np.add.at(m_body, p.body_id, m_add)
model.body_mass[:]    = self._dry_mass + m_body
model.body_inertia[:] = self._dry_inertia + (m_body * self._lever2)[:, None]
F[:, 2] += m_add * GRAVITY
```

Four departures from the derivation, all measured:

1. **The wing branch is isotropic.** `rho pi c^2 / 4` is the added mass for
   acceleration *normal to the plate*. Written into a scalar `body_mass`, it
   applies to edgewise and spanwise acceleration too. Measured in
   `experiments/added_mass`: the effective mass is identical in all three
   directions to 0.000%, and edgewise it exceeds 2D strip theory for a 2 mm
   plate by `(chord/thickness)^2 = 10 000`. The bluff branch, which does
   project, comes out 6.8x apart between broadside and edgewise on the same
   geometry. Recorded as **F-03**.
2. **The added rotational inertia is isotropic.** `m_add * mean(r^2)` goes into
   all three principal axes. For a planar surface the true added inertia about
   the span axis is far smaller than about the chord axis.
3. **The edit may not reach the mass matrix.** MuJoCo marks a body `simple`
   when no joint attaches to it and takes those DOFs' mass from the
   compile-time constant `dof_M0`; a runtime `body_mass` edit then never
   reaches `M`. Measured: a jointless model reports `+32.20 kg` in `body_mass`
   and `+0.00 kg` to the integrator, while the same body with one hinge in it
   gets all 32.20 kg. `mj_setConst` after the edit is what rebuilds it, and
   nothing in the project calls it. Recorded as **F-01**.
4. **The CoM does not move.** Entrained fluid is added at the body's existing
   `body_ipos`, so a machine with added mass concentrated on one wing does not
   feel its centre of mass shift.

## Validation

* `M` stays symmetric positive definite with added mass in it, over a
  300-step water rollout: `tests/test_math.py::
  test_the_mass_matrix_stays_a_mass_matrix`. Maximum asymmetry 0, smallest
  eigenvalue `6.7e-8` (positive).
* Added mass never reduces a body's mass, and `reset()` restores dry inertia
  exactly: same test.
* A passive body in still fluid never gains speed, in air and water, wing and
  bluff: `tests/test_math.py::test_a_passive_body_cannot_gain_energy`.
* Buoyancy is exactly `rho g V`: same file.
* The four departures above: `experiments/added_mass`.

## Failure conditions

* **Jointless machines** get no added mass at all in the dynamics (point 3).
* **A wing sliced edgewise** through water is charged the broadside added mass
  (point 1), so a machine that folds its wings to swim pays as if it had not.
* **Added mass with history.** (6) assumes the entrained fluid responds
  instantly. Near a free surface, and during water entry, it does not — which
  is why `slam` is computed as a rate term and used only as a diagnostic.
* **`M_a` from potential flow.** (6) is a potential-flow result. In separated
  flow the entrained mass is not a function of shape alone. Nothing in the
  project claims otherwise, and the coefficients are within 18% of Lamb's disc
  result where they can be compared — measured in
  `experiments/analytic_vs_numerical`, check H.
