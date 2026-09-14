"""Property tests: invariants that must hold for *every* input, not one.

Why these are separate from `test_physics.py`
---------------------------------------------
`test_physics.py` pins down worked examples -- a wing at 5 degrees in a 10 m/s
wind produces this much lift, with this sign.  Those catch a formula that is
wrong in the case someone thought about.

These catch a formula that is wrong in the cases nobody thought about, by
asserting things that follow from the mathematics rather than from a
measurement: a drag coefficient is never negative, a mass matrix is symmetric
positive definite, an SVD reconstructs the matrix it came from, a passive body
in still fluid cannot gain energy, a force coefficient does not depend on the
units the problem was posed in.

Several of these were written after `experiments/` found something.  Where that
is the case the test says which experiment, so the invariant and its evidence
stay attached to each other.

Run:  PYTHONPATH=. python tests/test_math.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco  # noqa: E402

from dytiscidae.control.cpg import (  # noqa: E402
    INTENT_AUTHORITY, CPG, CPGParams, MobilityBasis, basis_from_probes,
)
from dytiscidae.evolution.cmaes import CMAES  # noqa: E402
from dytiscidae.physics.energy import Actuator  # noqa: E402
from dytiscidae.physics.fluid import (  # noqa: E402
    BLUFF, WING, FluidSolver, PanelSet, drag_coefficient, lift_coefficient,
    skin_friction_cd,
)
from dytiscidae.physics.materials import CFRP_TUBE, PETG  # noqa: E402
from dytiscidae.physics.medium import (  # noqa: E402
    AIR, GRAVITY, SEAWATER, Fluid, MediumField,
)

FAILURES: list[str] = []
#: Behaviour that has been measured and reproduced but not yet fixed.  These do
#: not fail the suite -- they are re-measured on every run and printed at the
#: end, so an open finding cannot quietly stop being true.  Each carries its
#: `docs/MATH_AUDIT.md` identifier.
GAPS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def gap(audit_id: str, name: str, holds: bool, detail: str = "") -> None:
    """Record a known finding.  `holds` is whether the *gap* is still present."""
    status = "gap " if holds else "FIXED"
    print(f"  [{status}] {audit_id} {name}{('  -- ' + detail) if detail else ''}")
    if holds:
        GAPS.append(f"{audit_id} {name}")
    else:
        FAILURES.append(f"{audit_id} {name} -- the gap is gone; update this test "
                        f"and close the finding")


def _rng(seed: int = 20260914) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# Coefficient properties
# --------------------------------------------------------------------------


def test_coefficients_are_well_behaved_everywhere() -> None:
    """Over the whole domain, not at one worked point."""
    print("\nproperty: the coefficient models over their whole domain")
    rng = _rng()
    n = 20000
    alpha = rng.uniform(-np.pi / 2, np.pi / 2, n)
    re = 10.0 ** rng.uniform(1.0, 7.0, n)
    ar = rng.uniform(0.5, 30.0, n)
    k = rng.uniform(0.0, 2.0, n)

    cl = lift_coefficient(alpha, re, ar, k)
    cd = drag_coefficient(alpha, re, ar, cl)
    cf = skin_friction_cd(re)

    check("CL is finite everywhere", bool(np.all(np.isfinite(cl))))
    check("CD is finite everywhere", bool(np.all(np.isfinite(cd))))
    check("CD is never negative", bool(np.all(cd >= 0.0)),
          f"min {cd.min():.6g}")
    check("CD is never below the skin-friction floor", bool(np.all(cd >= cf - 1e-12)),
          f"worst deficit {float(np.min(cd - cf)):.3g}")
    grid = np.logspace(1, 7, 4000)
    cf_grid = skin_friction_cd(grid)
    check("skin friction is positive everywhere", bool(np.all(cf_grid > 0.0)))
    check("skin friction falls with Re inside each branch",
          bool(np.all(np.diff(cf_grid[grid < 1e4]) < 0))
          and bool(np.all(np.diff(cf_grid[grid > 1e7 * 0.5]) < 0)))
    rise = float(np.max(np.diff(cf_grid)))
    peak = float(grid[1:][np.argmax(np.diff(cf_grid))])
    check("and rises through transition, as it physically must",
          rise > 0.0,
          f"the blend to the turbulent branch raises Cf around Re={peak:.2g}; "
          f"Cf is not monotone in Re and nothing should assume it is")

    # CL is odd in alpha at zero camber: the model has no cambered branch, so
    # this is a symmetry of the function itself.
    cl_m = lift_coefficient(-alpha, re, ar, k)
    check("CL is odd in alpha", float(np.max(np.abs(cl + cl_m))) < 1e-12,
          f"max |CL(a)+CL(-a)| = {float(np.max(np.abs(cl + cl_m))):.3g}")

    # Lift is bounded by the plate envelope the model declares.
    cl_max = 1.10 + 0.80 * np.clip(k / 0.30, 0.0, 1.0)
    check("CL stays inside its declared 1.2*CL_max envelope",
          bool(np.all(np.abs(cl) <= 1.2 * cl_max + 1e-12)),
          f"max |CL| / (1.2 CL_max) = {float(np.max(np.abs(cl) / (1.2*cl_max))):.4f}")

    # More reduced frequency never reduces the stall angle or the peak CL.
    lo = lift_coefficient(np.full(200, np.radians(25.0)), np.full(200, 2e5),
                          np.full(200, 5.0), np.zeros(200))
    hi = lift_coefficient(np.full(200, np.radians(25.0)), np.full(200, 2e5),
                          np.full(200, 5.0), np.full(200, 0.3))
    check("a leading-edge vortex raises CL at a post-stall angle",
          bool(np.all(hi > lo)), f"{lo[0]:.3f} -> {hi[0]:.3f} at 25 deg")

    # The largest CL the model can produce, against the CL_MAX Tier 0 declares.
    from dytiscidae.envs.triphibian import CL_MAX
    grid_a = np.radians(np.linspace(-90, 90, 3001))
    peak = 0.0
    for kk in (0.0, 0.15, 0.3, 1.0):
        peak = max(peak, float(np.max(np.abs(lift_coefficient(
            grid_a, np.full_like(grid_a, 2e5), np.full_like(grid_a, 6.0),
            np.full_like(grid_a, kk))))))
    check("Tier 0's CL_MAX is not above what the strip model can produce",
          CL_MAX <= peak + 1e-9,
          f"CL_MAX={CL_MAX} against a model peak of {peak:.3f} -- "
          f"the docstring calls CL_MAX 'the largest lift coefficient the strip "
          f"model will produce'; it is {100*(peak/CL_MAX - 1):+.0f}% off that")


def test_force_coefficients_do_not_depend_on_the_units(  # noqa: C901
) -> None:
    """Dynamic similarity: match Re and the reduced frequency, get the same CL/CD.

    This is the strongest statement the quasi-steady model makes -- that it is a
    function of dimensionless groups alone.  Build the same problem at two
    length scales with the viscosity chosen to hold Reynolds number fixed, and
    the force must scale exactly as rho U^2 L^2.
    """
    print("\nproperty: the fluid model is a function of dimensionless groups")

    def probe(scale: float, u: float, mu: float) -> np.ndarray:
        c, b = 0.2 * scale, 1.0 * scale
        xml = f"""
        <mujoco><option timestep="0.001" gravity="0 0 0" density="0" viscosity="0"/>
        <worldbody><body name="w" pos="0 0 100"><freejoint/>
        <geom type="box" size="{c/2} {b/2} 0.001" density="200"/>
        </body></worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "w")
        a = math.radians(8.0)
        panels = PanelSet(
            body_id=np.array([bid]), pos_local=np.zeros((1, 3)),
            span_local=np.array([[0.0, 1.0, 0.0]]),
            chord_local=np.array([[math.cos(a), 0.0, -math.sin(a)]]),
            chord=np.array([c]), dr=np.array([b]),
            volume=np.array([0.0]), volume_buoyant=np.array([0.0]),
            half_height=np.array([0.001 * scale]),
            kind=np.array([WING]), aspect_ratio=np.array([5.0]),
            cd_bluff=np.array([0.0]))
        air = Fluid("scaled", 1.225, mu)
        medium = MediumField(air=air, wind=np.array([u, 0.0, 0.0]))
        solver = FluidSolver(m, panels, medium)
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        solver.apply(d, 0.0)
        # Remove the added-mass weight compensation, which is not an
        # aerodynamic force and does not obey this scaling.
        f = d.xfrc_applied[bid, :3].copy()
        f[2] -= solver.diag.added_mass * GRAVITY
        return f

    u0, mu0, s0 = 10.0, 1.81e-5, 1.0
    s1, u1 = 3.0, 2.0
    # Re = rho U c / mu held fixed:  mu1 = mu0 * (U1 c1) / (U0 c0)
    mu1 = mu0 * (u1 * s1) / (u0 * s0)
    f0 = probe(s0, u0, mu0)
    f1 = probe(s1, u1, mu1)
    expect = (u1 / u0) ** 2 * (s1 / s0) ** 2
    ratio = np.linalg.norm(f1) / max(np.linalg.norm(f0), 1e-30)
    check("force scales as rho U^2 L^2 at matched Reynolds number",
          abs(ratio / expect - 1.0) < 1e-9,
          f"measured {ratio:.6f} against {expect:.6f}")
    check("and the force direction is unchanged",
          float(np.linalg.norm(f1 / np.linalg.norm(f1)
                               - f0 / np.linalg.norm(f0))) < 1e-9)


# --------------------------------------------------------------------------
# Solver properties
# --------------------------------------------------------------------------


def _submerged_body(dry_mass: float = 12.0, jointed: bool = True,
                    volume: float = 0.0):
    child = ("""<body name="d" pos="0 0 0.05"><joint type="hinge" axis="0 1 0"/>
                <geom type="box" size="0.01 0.01 0.01" mass="0.001"/></body>"""
             if jointed else "")
    xml = f"""
    <mujoco><option timestep="0.001" gravity="0 0 0" density="0" viscosity="0"/>
    <worldbody><body name="b" pos="0 0 -5"><freejoint/>
    <geom type="box" size="0.1 0.5 0.001" mass="{dry_mass}"/>{child}
    </body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "b")
    panels = PanelSet(
        body_id=np.array([bid]), pos_local=np.zeros((1, 3)),
        span_local=np.array([[0.0, 1.0, 0.0]]),
        chord_local=np.array([[1.0, 0.0, 0.0]]),
        chord=np.array([0.2]), dr=np.array([1.0]),
        volume=np.array([volume]), volume_buoyant=np.array([volume]),
        half_height=np.array([0.001]),
        kind=np.array([WING]), aspect_ratio=np.array([5.0]),
        cd_bluff=np.array([0.0]),
        ext_local=np.array([[1.0, 0.2, 0.002]]))
    return m, d, panels


def test_the_mass_matrix_stays_a_mass_matrix() -> None:
    """Added mass may only ever make a body harder to accelerate."""
    print("\nproperty: the augmented mass matrix is still symmetric positive definite")
    m, d, panels = _submerged_body()
    solver = FluidSolver(m, panels, MediumField(water=SEAWATER))
    dry = float(m.body_mass[panels.body_id[0]])

    worst_asym, worst_eig, ok_growth = 0.0, np.inf, True
    for _ in range(40):
        d.xfrc_applied[:] = 0.0
        solver.apply(d, d.time)
        ok_growth = ok_growth and float(m.body_mass[panels.body_id[0]]) >= dry - 1e-12
        mujoco.mj_step(m, d)
        full = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, full)
        worst_asym = max(worst_asym, float(np.max(np.abs(full - full.T))))
        worst_eig = min(worst_eig, float(np.linalg.eigvalsh(full).min()))

    check("M is symmetric", worst_asym < 1e-12, f"max asymmetry {worst_asym:.3g}")
    check("M is positive definite", worst_eig > 0.0,
          f"smallest eigenvalue {worst_eig:.6g}")
    check("added mass never makes a body lighter", ok_growth)
    check("the solver restores the dry inertia on reset",
          (solver.reset() is None)
          and abs(float(m.body_mass[panels.body_id[0]]) - dry) < 1e-12)


def test_added_mass_reaches_the_integrator() -> None:
    """Writing a number into `body_mass` is not the same as MuJoCo using it.

    Measured in `experiments/added_mass`: MuJoCo marks a body 'simple' when no
    joint attaches to it and takes those DOFs' mass from the compile-time
    constant `dof_M0`, so a runtime `body_mass` edit never reaches the mass
    matrix.  A machine with no joints -- which this search has produced --
    therefore swims with none of its entrained water.
    """
    print("\nproperty: added mass written into body_mass reaches the mass matrix")

    def effective_mass(jointed: bool) -> tuple[float, float]:
        m, d, panels = _submerged_body(jointed=jointed)
        solver = FluidSolver(m, panels, MediumField(water=SEAWATER))
        bid = int(panels.body_id[0])
        dry = float(m.body_mass[bid])
        F = np.array([0.0, 0.0, 40.0])
        vals = []
        for step in range(40):
            d.xfrc_applied[:] = 0.0
            solver.apply(d, d.time)
            d.xfrc_applied[bid, :3] += F
            total = float(d.xfrc_applied[bid, 2])
            mujoco.mj_step(m, d)
            if step >= 20 and abs(d.qacc[2]) > 1e-12:
                vals.append(total / float(d.qacc[2]))
        return dry, float(np.median(vals))

    dry_j, eff_j = effective_mass(True)
    dry_s, eff_s = effective_mass(False)
    check("a jointed machine feels its entrained water",
          eff_j > dry_j * 2.0, f"{dry_j:.2f} kg dry -> {eff_j:.2f} kg effective")
    gap("F-01", "a jointless machine does not feel its entrained water",
        eff_s < dry_s * 1.01,
        f"{dry_s:.2f} kg dry -> {eff_s:.2f} kg effective, against "
        f"{eff_j:.2f} kg for the same body with one hinge in it.  MuJoCo marks "
        f"a body `simple` when no joint attaches to it and takes those DOFs' "
        f"mass from `dof_M0`, a compile-time constant, so the runtime "
        f"`body_mass` edit never reaches the mass matrix.  `mj_setConst` after "
        f"the edit is what rebuilds it.  See experiments/added_mass.")


def test_a_passive_body_cannot_gain_energy() -> None:
    """Drag may remove kinetic energy.  It may never add any.

    The class of bug this catches is a sign error in a force term, which does
    not crash: it produces a machine that accelerates for free and an optimiser
    that finds it within a hundred generations.
    """
    print("\nproperty: a passive body in still fluid loses energy monotonically")
    for medium_name, medium in (("water", MediumField(water=SEAWATER)),
                                ("air", MediumField())):
        for kind, label in ((WING, "wing strip"), (BLUFF, "bluff volume")):
            m, d, panels = _submerged_body(dry_mass=12.0, volume=0.0)
            panels.kind = np.array([kind])
            panels.cd_bluff = np.array([0.9])
            if medium_name == "air":
                d.qpos[2] = 50.0
            solver = FluidSolver(m, panels, medium)
            d.qvel[:3] = np.array([3.0, 1.0, -2.0])
            mujoco.mj_forward(m, d)
            speeds = []
            for _ in range(300):
                d.xfrc_applied[:] = 0.0
                solver.apply(d, d.time)
                # The model has gravity switched off, so MuJoCo applies no
                # weight to the entrained fluid and the solver's compensating
                # +m_add*g is an unbalanced upward force.  Remove it: the
                # question here is whether the *hydrodynamic* terms can add
                # energy, and the compensation is not one of them.
                d.xfrc_applied[1, 2] -= solver.diag.added_mass * GRAVITY
                mujoco.mj_step(m, d)
                speeds.append(float(np.linalg.norm(d.qvel[:3])))
            speeds = np.array(speeds)
            rise = float(np.max(np.diff(speeds)))
            check(f"{label} in {medium_name}: speed never rises",
                  rise <= 1e-9,
                  f"largest single-step rise {rise:.3g} m/s, "
                  f"{speeds[0]:.3f} -> {speeds[-1]:.3f} m/s")


def test_buoyancy_is_exactly_archimedes() -> None:
    """A fully submerged volume displaces exactly its own volume of water."""
    print("\nproperty: buoyancy is rho g V, exactly")
    vol = 0.004
    m, d, panels = _submerged_body(volume=vol)
    solver = FluidSolver(m, panels, MediumField(water=SEAWATER))
    d.xfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)
    solver.apply(d, 0.0)
    fz = float(d.xfrc_applied[panels.body_id[0], 2])
    # What is left after removing the added-mass weight compensation.
    fz -= solver.diag.added_mass * GRAVITY
    expect = SEAWATER.rho * GRAVITY * vol
    check("fully submerged buoyancy equals rho g V",
          abs(fz - expect) < 1e-9 * max(expect, 1.0),
          f"{fz:.6f} N against {expect:.6f} N")
    check("the diagnostic agrees with the force",
          abs(solver.diag.buoyancy - expect) < 1e-9 * expect)

    med = MediumField(water=SEAWATER)
    depths = np.linspace(-2.0, 2.0, 41)
    pos = np.stack([np.zeros(41), np.zeros(41), -depths], axis=1)
    f = med.submerged_fraction(pos, np.full(41, 0.1))
    check("submerged fraction is monotone in depth",
          bool(np.all(np.diff(f) >= -1e-12)))
    check("submerged fraction stays in [0, 1]",
          bool(np.all((f >= 0.0) & (f <= 1.0))))
    p = med.pressure(pos)
    check("ambient pressure is monotone in depth",
          bool(np.all(np.diff(p) >= -1e-12)))


def test_the_force_limiter_bounds_each_element_not_the_machine() -> None:
    """What the limiter actually bounds, stated as a test rather than a comment.

    The docstring says it clamps "to a multiple of the vehicle's weight".  It
    clamps each element to that multiple, so a machine with N panels can carry
    up to N times the stated bound.  This test records the real behaviour so a
    later reader is not surprised by it.
    """
    print("\nproperty: the force limiter is per element")
    n = 12
    xml = """
    <mujoco><option timestep="0.001" gravity="0 0 0" density="0" viscosity="0"/>
    <worldbody><body name="b" pos="0 0 -5"><freejoint/>
    <geom type="box" size="0.1 0.5 0.01" mass="1.0"/>
    <body name="d" pos="0 0 0.05"><joint type="hinge" axis="0 1 0"/>
    <geom type="box" size="0.01 0.01 0.01" mass="0.001"/></body>
    </body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "b")
    panels = PanelSet(
        body_id=np.full(n, bid), pos_local=np.zeros((n, 3)),
        span_local=np.tile([[0.0, 1.0, 0.0]], (n, 1)),
        chord_local=np.tile([[1.0, 0.0, 0.0]], (n, 1)),
        chord=np.full(n, 0.3), dr=np.full(n, 1.0),
        volume=np.zeros(n), volume_buoyant=np.zeros(n),
        half_height=np.full(n, 0.01),
        kind=np.full(n, WING), aspect_ratio=np.full(n, 5.0),
        cd_bluff=np.zeros(n))
    solver = FluidSolver(m, panels, MediumField(water=SEAWATER,
                                                current=np.array([40.0, 0.0, 0.0])))
    d.xfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)
    solver.apply(d, 0.0)
    # The limiter's own bound, built from the dry mass it captured at
    # construction -- not from `model.body_mass`, which `apply` has by now
    # augmented with the entrained fluid.
    weight = float(solver._dry_mass.sum()) * GRAVITY + 1.0
    per_element = 60.0 * weight
    total = float(np.linalg.norm(d.xfrc_applied[bid, :3]))
    check("the limiter engaged", solver.diag.clamped)
    check("the machine total may exceed the per-element bound",
          total > per_element,
          f"{total:.0f} N against a per-element bound of {per_element:.0f} N, "
          f"over {n} elements -- the docstring says the clamp is 'to a "
          f"multiple of the vehicle's weight', and it is that multiple per "
          f"element.  See MATH_AUDIT F-04.")
    check("and stays within the per-element bound times the element count",
          total <= n * per_element * 1.001,
          f"{total:.0f} N against {n * per_element:.0f} N")


# --------------------------------------------------------------------------
# Control properties
# --------------------------------------------------------------------------


def test_the_basis_reconstructs_the_jacobian_it_came_from() -> None:
    """J = U S V^T is an identity, not an approximation, up to truncation."""
    print("\nproperty: the mobility basis reconstructs its own Jacobian")
    rng = _rng()
    n_params, n_probes = 19, 40
    deltas = rng.normal(0.0, 0.35, size=(n_probes, n_params))
    J_true = rng.normal(0.0, 1.0, size=(n_params, 6))
    scale = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])
    responses = (deltas @ J_true) / scale

    basis = basis_from_probes(deltas, responses, max_modes=6)
    J_fit, *_ = np.linalg.lstsq(deltas, responses * scale, rcond=None)
    recon = (basis.modes.T * basis.authority) @ basis.effects
    check("modes^T diag(sigma) effects reproduces the fitted J",
          float(np.max(np.abs(recon - J_fit))) < 1e-9,
          f"max element error {float(np.max(np.abs(recon - J_fit))):.3g}")
    check("and the fitted J reproduces the true one when it is noiseless",
          float(np.max(np.abs(J_fit - J_true))) < 1e-9)
    check("modes are orthonormal rows",
          float(np.max(np.abs(basis.modes @ basis.modes.T - np.eye(6)))) < 1e-9)
    check("effects are unit rows",
          float(np.max(np.abs(np.linalg.norm(basis.effects, axis=1) - 1.0))) < 1e-9)
    check("singular values are non-negative and descending",
          bool(np.all(basis.authority >= 0)
               and np.all(np.diff(basis.authority) <= 1e-12)))


def test_the_inverse_model_delivers_what_it_can_and_no_more() -> None:
    """`coeffs_for_twist` must be the damped least-squares solution, and its
    residual must lie outside what the body can reach."""
    print("\nproperty: the damped inverse is a projection onto the reachable set")
    rng = _rng(7)
    for trial in range(20):
        r, P = 4, 19
        M = np.linalg.qr(rng.normal(size=(P, P)))[0][:r]
        eff = np.linalg.qr(rng.normal(size=(6, 6)))[0][:r]
        sig = np.sort(rng.uniform(0.05, 5.0, r))[::-1]
        basis = MobilityBasis(modes=M, effects=eff, authority=sig)
        A = eff * sig[:, None]
        w = rng.uniform(-1.0, 1.0, 6)
        c = basis.coeffs_for_twist(w)

        # Against the closed form it claims to be.
        lam = 0.01 * float(np.trace(A @ A.T)) / r + 1e-12
        b = w * np.linalg.norm(A, axis=0) * INTENT_AUTHORITY
        want = np.linalg.solve(A @ A.T + lam * np.eye(r), A) @ b
        if trial == 0:
            check("coeffs_for_twist equals (A A^T + lam I)^-1 A b",
                  float(np.max(np.abs(c - want))) < 1e-12,
                  f"max error {float(np.max(np.abs(c - want))):.3g}")

        # The forward model must be the transpose of the same A.
        if trial == 0:
            check("twist_of is A^T c, the forward model of that same A",
                  float(np.max(np.abs(basis.twist_of(c) - A.T @ c))) < 1e-12)

        # Asking for an axis the body does not have must produce a small
        # coefficient, not a large one.
        weak = MobilityBasis(modes=M, effects=eff,
                             authority=np.array([5.0, 1e-6, 1e-7, 1e-8]))
        c_weak = weak.coeffs_for_twist(np.ones(6))
        if trial == 0:
            check("an unreachable request gives a bounded coefficient",
                  float(np.max(np.abs(c_weak))) < 10.0,
                  f"max |c| = {float(np.max(np.abs(c_weak))):.4g} for a body "
                  f"with one usable axis out of four")

    # Intent is read as a fraction of reach, so +1 on an axis must deliver
    # close to that axis's own maximum.
    r = 6
    eff = np.eye(6)
    sig = np.array([4.0, 3.0, 2.0, 1.0, 0.5, 0.25])
    basis = MobilityBasis(modes=np.eye(6, 19), effects=eff, authority=sig)
    got = np.array([basis.twist_of(basis.coeffs_for_twist(np.eye(6)[i]))[i]
                    for i in range(6)])
    frac = got / (sig * INTENT_AUTHORITY)
    # On an orthogonal basis the damped solve reduces to a scalar shrinkage per
    # mode, so the delivered fraction has a closed form: sigma^2/(sigma^2+lam)
    # with lam = 0.01 * mean(sigma^2).
    lam = 0.01 * float(np.sum(sig**2)) / 6
    predicted = sig**2 / (sig**2 + lam)
    check("the delivered fraction follows sigma^2/(sigma^2+lam) exactly",
          float(np.max(np.abs(frac - predicted))) < 1e-9,
          f"max error {float(np.max(np.abs(frac - predicted))):.3g}")
    gap("C-04", "a saturated intent does not deliver a body's full reach on a "
                "weak axis",
        bool(np.min(frac) < 0.95),
        f"delivered fraction {np.array2string(frac, precision=3)} across "
        f"sigma {np.array2string(sig, precision=3)}.  `coeffs_for_twist` "
        f"documents '+1 heave means climb as hard as this body climbs on "
        f"every one of them'; the ridge makes that true only for axes whose "
        f"sigma^2 is large against 0.01*mean(sigma^2).  The weakest axis here "
        f"delivers {100*float(np.min(frac)):.0f}%.")
    check("an empty basis returns an empty command rather than raising",
          MobilityBasis(modes=np.zeros((0, 7)), effects=np.zeros((0, 6)),
                        authority=np.zeros(0)).coeffs_for_twist(
                            np.ones(6)).shape == (0,))


def test_the_cpg_command_is_the_function_it_claims() -> None:
    """Amplitude, phase and offset act on the joint command as declared."""
    print("\nproperty: the CPG command")
    cpg = CPG(5, base_frequency=2.0)
    base = cpg.base
    t = np.linspace(0.0, 2.0, 401)
    cmd = np.array([cpg.command(base, tt) for tt in t])
    check("every command stays inside the joint travel",
          bool(np.all(cmd >= cpg.lo - 1e-9) and np.all(cmd <= cpg.hi + 1e-9)))
    mean = cmd.mean(axis=0)
    check("the cycle mean is the offset",
          float(np.max(np.abs(mean - base.offset))) < 5e-3,
          f"max deviation {float(np.max(np.abs(mean - base.offset))):.4g}")
    amp = 0.5 * (cmd.max(axis=0) - cmd.min(axis=0))
    check("the half peak-to-peak is the amplitude",
          float(np.max(np.abs(amp - base.amplitude))) < 5e-3)
    # Doubling the frequency must halve the period, exactly.
    fast = CPGParams(base.amplitude, base.phase, base.offset, 2 * base.frequency)
    check("doubling the frequency halves the period",
          float(np.max(np.abs(cpg.command(fast, 0.37)
                              - cpg.command(base, 0.74)))) < 1e-12)
    flat = CPGParams.from_flat(base.flat(), cpg.n)
    check("flat() and from_flat() round-trip",
          float(np.max(np.abs(flat.flat() - base.flat()))) == 0.0)


# --------------------------------------------------------------------------
# Optimiser and power-train properties
# --------------------------------------------------------------------------


def test_cmaes_keeps_its_own_invariants() -> None:
    """The recombination weights, covariance and step size stay legal."""
    print("\nproperty: CMA-ES invariants over a run")
    es = CMAES(np.zeros(24), sigma0=0.3, seed=3)
    check("recombination weights sum to one",
          abs(float(es.weights.sum()) - 1.0) < 1e-12)
    check("mueff lies between 1 and mu",
          1.0 <= es.mueff <= es.mu + 1e-12, f"mueff={es.mueff:.3f}, mu={es.mu}")
    check("c1 + cmu does not exceed one", es.c1 + es.cmu <= 1.0 + 1e-12)

    worst_asym, worst_eig, sigmas = 0.0, np.inf, []
    for _ in range(300):
        pop = es.ask()
        scores = -np.sum((pop - 0.5) ** 2, axis=1)
        es.tell(pop, scores)
        worst_asym = max(worst_asym, float(np.max(np.abs(es.C - es.C.T))))
        worst_eig = min(worst_eig, float(np.linalg.eigvalsh(es.C).min()))
        sigmas.append(es.sigma)
    check("C stays symmetric", worst_asym < 1e-9, f"max asymmetry {worst_asym:.3g}")
    check("C stays positive definite", worst_eig > 0.0,
          f"smallest eigenvalue {worst_eig:.3g}")
    check("the step size stays finite and positive",
          bool(np.all(np.isfinite(sigmas)) and np.all(np.array(sigmas) > 0)))
    dist = float(np.linalg.norm(es.mean - 0.5 * np.ones(24)))
    check("the mean converges toward the optimum of a quadratic",
          dist < 1e-3,
          f"||mean - optimum|| = {dist:.3g} after 300 generations "
          f"({300 * es.lam} evaluations) on a 24-dimensional sphere")

    # O-01: the eigen-decomposition staleness counter is incremented once per
    # `tell`, so it counts generations; the reference algorithm compares
    # `counteval - eigeneval` against the same expression, in evaluations.
    interval = max(1, int(es.lam / (10 * es.n * (es.c1 + es.cmu))))
    gap("O-01", "the eigen-update interval is counted in generations, not "
                "evaluations",
        interval * es.lam > 2 * es.lam / (10 * es.n * (es.c1 + es.cmu)),
        f"{interval} generations = {interval * es.lam} evaluations, against a "
        f"reference interval of "
        f"{es.lam / (10 * es.n * (es.c1 + es.cmu)):.1f} evaluations -- a "
        f"factor of {es.lam}, so B and D are stale for that long")

    # O-02: `ask` clips the population to `bounds` and `tell` updates the mean
    # and the rank-mu covariance from the clipped samples.  `control/train.py`
    # sets bounds=(-4, 4), so this path is live.  How much it costs depends
    # entirely on whether the optimum is inside the box.
    for label, target in (("optimum inside the box", 2.0),
                          ("optimum outside the box", 6.0)):
        b = CMAES(np.zeros(60), sigma0=0.5, seed=1, bounds=(-4.0, 4.0))
        hit, tot = 0, 0
        for _ in range(200):
            pop = b.ask()
            hit += int(np.isclose(np.abs(pop), 4.0).sum())
            tot += pop.size
            b.tell(pop, -np.sum((pop - target) ** 2, axis=1))
        print(f"         bounds, {label}: {100*hit/tot:.2f}% of sampled "
              f"coordinates clipped, final sigma {b.sigma:.4g}, "
              f"mean[0] {b.mean[0]:.4f}")


def test_the_power_train_never_gives_energy_back() -> None:
    """Electrical power in is always at least mechanical power out."""
    print("\nproperty: the actuator model obeys the first law")
    rng = _rng(11)
    for cls in ("bldc", "geared", "coreless"):
        act = Actuator(motor_class=cls, mass=0.08, gear_ratio=4.0)
        tau = rng.uniform(-3.0, 3.0, 4000)
        om = rng.uniform(-60.0, 60.0, 4000)
        p_e = act.electrical_power(tau, om)
        p_m = np.abs(tau * om)
        check(f"{cls}: electrical power is never negative",
              bool(np.all(p_e >= 0.0)), f"min {float(p_e.min()):.4g} W")
        check(f"{cls}: electrical power is never below mechanical",
              bool(np.all(p_e >= p_m - 1e-9)),
              f"worst deficit {float(np.min(p_e - p_m)):.3g} W")
        # Monotone in |torque| at fixed speed.
        t_grid = np.linspace(0.0, 5.0, 200)
        p_grid = act.electrical_power(t_grid, np.full(200, 20.0))
        check(f"{cls}: power rises with torque at fixed speed",
              bool(np.all(np.diff(p_grid) >= -1e-9)))


def test_material_allowables_are_monotone_and_bounded() -> None:
    """Fatigue may only reduce an allowable, and only within its declared range."""
    print("\nproperty: material allowables")
    for mat in (PETG, CFRP_TUBE):
        cyc = np.logspace(0, 6, 200)
        vals = np.array([mat.allowable_stress(cycles=c) for c in cyc])
        check(f"{mat.name.split('(')[0].strip()}: allowable is "
              f"non-increasing in cycles",
              bool(np.all(np.diff(vals) <= 1e-6)))
        base = mat.yield_MPa * 1e6 * mat.strength_knockdown / 2.0
        check(f"{mat.name.split('(')[0].strip()}: allowable never exceeds the "
              f"static value",
              bool(np.all(vals <= base + 1e-6)))
        floor = base * mat.fatigue_knockdown_1e5
        check(f"{mat.name.split('(')[0].strip()}: the curve is flat beyond 1e5 "
              f"cycles rather than continuing down",
              abs(mat.allowable_stress(cycles=1e9) - floor) < 1e-6,
              "extrapolation past the fitted range is clamped, not projected")


def main() -> int:
    test_coefficients_are_well_behaved_everywhere()
    test_force_coefficients_do_not_depend_on_the_units()
    test_the_mass_matrix_stays_a_mass_matrix()
    test_added_mass_reaches_the_integrator()
    test_a_passive_body_cannot_gain_energy()
    test_buoyancy_is_exactly_archimedes()
    test_the_force_limiter_bounds_each_element_not_the_machine()
    test_the_basis_reconstructs_the_jacobian_it_came_from()
    test_the_inverse_model_delivers_what_it_can_and_no_more()
    test_the_cpg_command_is_the_function_it_claims()
    test_cmaes_keeps_its_own_invariants()
    test_the_power_train_never_gives_energy_back()
    test_material_allowables_are_monotone_and_bounded()

    print()
    if GAPS:
        print(f"{len(GAPS)} open findings, re-measured and still present "
              f"(see docs/MATH_AUDIT.md):")
        for g in GAPS:
            print(f"  - {g}")
        print()
    if FAILURES:
        print(f"{len(FAILURES)} property checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all property checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
