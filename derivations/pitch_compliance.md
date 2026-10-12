# Passive pitch compliance: `k = M_1g / (compliance · θ_ref)`

## Definition

A lifting surface with `Part.pitch_compliance = c > 0` is a rigid plate that
may rotate about its own span axis through the leading edge, against a torsion
spring of stiffness `k` (N·m/rad), a damper `b` (N·m·s/rad), and stops at
±1 rad. `c` is the spring's deflection, in units of `θ_ref = 0.1 rad`, when
the surface carries its share of the machine's weight at 1 g, with that share
spread uniformly over its planform and acting at the quarter chord.

## Assumptions

1. Chordwise flexibility is represented by one rotational degree of freedom per
   surface: the pseudo-rigid-body model of a flexible wing (Whitney & Wood
   2010). Camber change, spanwise bending and a twist that varies along the span
   are absent.
2. The aerodynamic load on each strip acts at its quarter chord. The hinge is at
   the leading edge, so any normal force produces a nose-down moment `0.25 c N`.
3. At 1 g the surface carries `W_i = W · S_i / Σ S_j`, its planform-area share of
   the weight `W`, uniformly per unit area.
4. The surface's mass is spread uniformly over its planform, for its inertia
   about the hinge.
5. Aerodynamic pitch damping, which the strip model under-represents (it reads
   each strip's velocity at its quarter chord, where thin-airfoil theory puts the
   pitch-rate term at three quarters), is lumped into the joint damper.

## Derivation

Lift per unit area at 1 g is `W_i / S_i`. A strip of width `du` and chord
`c(u)` carries `(W_i / S_i) c(u) du`, at `0.25 c(u)` behind the hinge. The
moment about the leading edge is

    M_1g = ∫ 0.25 c(u) · (W_i / S_i) c(u) du = 0.25 (W_i / S_i) ∫ c(u)² du.

Requiring a deflection of `c · θ_ref` under `M_1g`:

    k = M_1g / (c · θ_ref).

Aerodynamic stiffness: if the surface carries `W_i` at incidence `α_1g`, a
nose-down rotation `θ` removes lift `W_i θ / α_1g` (linear lift slope). Its
moment about the hinge is the aerodynamic stiffness

    k_a = M_1g / α_1g,  with α_1g = 0.1 rad.

Static equilibrium: a surface at incidence `α_0` that would make moment
`M_0` rigid makes `M_0 (1 − θ / α_0)` after deflecting by `θ`, so
`k θ = M_0 (1 − θ / α_0)`. With a linear lift slope, `M_0 / α_0 = k_a`:

    θ = M_0 / (k + k_a),   lift kept = 1 − θ / α_0 = k / (k + k_a).

At `c = 1`, `θ_ref = α_1g` makes `k = k_a`, so the surface keeps half its rigid
lift at a fixed attitude. That is passive load alleviation, and it is why a
flexible machine has to re-trim.

Inertia of a uniform plate of mass `m` about its leading edge:

    I = (m / S) ∫ c(u)³ / 3 du.

Damping: `b = 2 ζ √((k + k_a)(I + a))`, where `a` is the joint armature and
`ζ = PITCH_DAMPING_RATIO`.

## Dimensional analysis

`W_i / S_i` is N/m². `∫ c² du` is m³. Their product is N·m, so `M_1g` is a
moment. `θ_ref` and `c` are dimensionless (rad is dimensionless), so `k` is
N·m/rad. `I` is (kg/m²)·m⁴ = kg·m². `√(k I)` is √(N·m·kg·m²) = √(kg²·m⁴/s²) =
kg·m²/s, which is N·m·s, the unit of a rotational damper.

## Numerical implementation

`core.mjcf.passive_pitch_spring` computes `M_1g`, `k`, `I` and `b` by the
trapezoid rule over the CPPN's span stations (`SurfaceField.u`, `.chord`).
`build_model_xml` splits the weight by planform area over every surface and
emits a hinge `<seg>_jp` about local +X, mirrored like the universal joint's
feathering hinge, with `stiffness`, `springref = 0`, `damping` and `armature`.
Departures:

- MuJoCo integrates joint stiffness explicitly, so `k` is capped at
  `(0.5 / Δt)² (I + a)`. Small surfaces at low compliance reach the cap and are
  then softer than `c` says; `clipped` is returned.
- A universal joint already has a driven hinge about the same axis and gets no
  passive one.
- The aerodynamic stiffness used for damping is the 1 g air value. In water the
  aerodynamic moment is ~830× larger and the surface lies on its stop.

## Validation

Command:

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/wing_compliance/run.py \
        --out experiments/wing_compliance/results.json

Held by `tests/test_physics.py::test_a_flexible_wing_gives_way_under_load`.
On the gannet pinned at its rigid trim (15.3 m/s), joints held, 2026-10-12:

| compliance | mean deflection, rad | lift, N | drag, N |
|---|---|---|---|
| 0 | 0 | 55.96 | 4.53 |
| 0.05 | 0.011 | 53.69 | 4.13 |
| 1 | 0.086 | 39.55 | 2.68 |
| 4 | 0.159 | 17.68 | 2.19 |

At compliance 1, with the pin carrying 1.07 g (55.96 N lift for 52.0 N
weight) and the aerodynamic relief of the Derivation, the static prediction is
`θ = 1.07 M_1g / (2 k) = 1.07 θ_ref / 2 ≈ 0.054 rad`. Per hinge, the measured means are
0.057 rad on the inner wing pair (`seg1`, `seg2`) and 0.115 rad on the outer
pair (`seg3`, `seg4`). No stiffness was capped. The inner pair agrees within
10%. The outer pair deflects twice the prediction, which says it carries about
twice its area share of the lift at that pin. The model assumes the share is
by area. The outer panel's own incidence (CPPN twist and dihedral) is the
likely cause; it is not measured here.

The flapping and still-machine reads are in `docs/ROADMAP.md`
§"2026-10-12 — wing flexibility".

## Failure conditions

- At high compliance under multi-g loads the surface reaches its ±1 rad stop,
  and the response is the stop's, not the spring's.
- With damping sized on the spring alone (`b = 2 · 0.1 · √(k I)`, k = 0.29 N·m/rad,
  the first sizing), the gannet's held wing limit-cycled at ±0.3 rad and lost
  90% of its lift. Sizing on `k + k_a` removed it, with 0.1 and 1.0 of critical
  giving the same forces to 0.01 N.
- A one-degree-of-freedom hinge cannot represent a wing whose root is stiff
  and tip is soft.
