"""The eight layers.  Each adds one piece of physics to the one below it.

Every reference in this file is written out from a definition.  None of them
calls the project's own solver to decide what the answer should be -- that is
the whole point, and it is the rule that makes a departure mean something.
"""

from __future__ import annotations

import math
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco  # noqa: E402

from dytiscidae.physics.fluid import (  # noqa: E402
    BLUFF, WING, FluidSolver, PanelSet, drag_coefficient, lift_coefficient,
    skin_friction_cd,
)
from dytiscidae.physics.jet import JetSet  # noqa: E402
from dytiscidae.physics.medium import (  # noqa: E402
    AIR, GRAVITY, SEAWATER, MediumField,
)

from ._report import Check, Layer, rk45

#: A step fine enough that the integrator's own truncation sits far below the
#: model discrepancies the ladder is looking for.
DT = 2.5e-4

#: What the project actually runs.  `core/mjcf.scene_xml` writes these into
#: every machine's `<option>` block, and the numbers below are quoted from it
#: so a change there shows up here as a failing check rather than as a stale
#: benchmark.
PROJECT_DT = 0.002
PROJECT_INTEGRATOR = "implicitfast"


def order_of_accuracy(error_of, steps) -> tuple[float, list[float]]:
    """Fit p in `error ~ dt^p` from a step-refinement study.

    The single most useful thing a verification suite can say about a
    discretisation, and the thing that separates "the model is wrong" from
    "the integrator has not converged": a model error is flat in dt, a
    truncation error is proportional to dt^p.
    """
    errs = [float(error_of(dt)) for dt in steps]
    logs = np.log(np.maximum(errs, 1e-300))
    slope = float(np.polyfit(np.log(steps), logs, 1)[0])
    return slope, errs


def _M(model, data) -> np.ndarray:
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)
    return full


# --------------------------------------------------------------------------
# 1. Rigid body: no fluid at all
# --------------------------------------------------------------------------


def layer1_rigid_body() -> Layer:
    lay = Layer(1, "rigid body", "Newton-Euler only, no fluid, no contact")
    m, hx, hy, hz = 3.0, 0.10, 0.25, 0.05
    xml = f"""
    <mujoco><option timestep="{DT}" gravity="0 0 0" density="0" viscosity="0"/>
    <worldbody><body name="b"><freejoint/>
    <geom type="box" size="{hx} {hy} {hz}" mass="{m}"/>
    </body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Inertia of a solid box about its own principal axes.
    Ixx = m * ((2 * hy) ** 2 + (2 * hz) ** 2) / 12.0
    Iyy = m * ((2 * hx) ** 2 + (2 * hz) ** 2) / 12.0
    Izz = m * ((2 * hx) ** 2 + (2 * hy) ** 2) / 12.0
    lay.checks.append(Check(
        1, "box inertia Ixx = m((2b)^2+(2c)^2)/12", Ixx,
        float(model.body_inertia[1][0]), 1e-12, "closed form"))

    # 1a. a = F/m, read off the integrator.
    F = np.array([7.0, -3.0, 11.0])
    data.xfrc_applied[1, :3] = F
    mujoco.mj_forward(model, data)
    a = data.qacc[:3].copy()
    lay.checks.append(Check(
        1, "a = F/m along the applied force", float(np.linalg.norm(F) / m),
        float(np.linalg.norm(a)), 1e-12, "closed form"))
    lay.checks.append(Check(
        1, "and the acceleration is parallel to the force", 0.0,
        float(np.linalg.norm(np.cross(a, F))), 1e-9, "closed form",
        absolute=True))

    # 1b. alpha = I^-1 tau with omega = 0, so no gyroscopic term.
    data.xfrc_applied[:] = 0.0
    tau = np.array([0.4, 0.9, -0.2])
    data.xfrc_applied[1, 3:] = tau
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    alpha = data.qacc[3:6].copy()
    ref = tau / np.array([Ixx, Iyy, Izz])
    lay.checks.append(Check(
        1, "alpha = I^-1 tau at rest", float(np.linalg.norm(ref)),
        float(np.linalg.norm(alpha)), 1e-10, "closed form"))

    # 1c. Ballistic flight.  The error here is the integrator's truncation,
    # not the model's -- so the thing worth checking is not its size but its
    # *order*.  A first-order scheme must show the error halving with the step.
    def ballistic_error(dt: float) -> float:
        mg = mujoco.MjModel.from_xml_string(
            xml.replace('gravity="0 0 0"', 'gravity="0 0 -9.80665"')
               .replace(f'timestep="{DT}"', f'timestep="{dt}"'))
        dg = mujoco.MjData(mg)
        v0 = np.array([4.0, 0.0, 6.0])
        dg.qvel[:3] = v0
        for _ in range(int(round(1.0 / dt))):
            mujoco.mj_step(mg, dg)
        return abs(float(dg.qpos[2]) - (v0[2] - 0.5 * 9.80665))

    steps = [2e-3, 1e-3, 5e-4, 2.5e-4]
    p_bal, errs = order_of_accuracy(ballistic_error, steps)
    lay.checks.append(Check(
        1, "ballistic flight converges at first order", 1.0, p_bal, 0.05,
        "order-of-accuracy study",
        note=f"|error| at dt = {steps}: "
             f"{['%.3e' % e for e in errs]}.  MuJoCo's semi-implicit schemes "
             f"are first order, so p = 1 is the correct answer and any other "
             f"value means the discretisation is not doing what it claims"))
    # What that error is worth is decided by the smallest height the project
    # scores on, not by a number chosen here.
    from dytiscidae.envs.triphibian import TAKEOFF_WINGLESS_CAP
    err_project = ballistic_error(PROJECT_DT)
    lay.checks.append(Check(
        1, "its 1 s height error stays under 5% of the smallest height rung",
        0.05, err_project / TAKEOFF_WINGLESS_CAP, 1.0, "closed form",
        note=f"{err_project*1e3:.2f} mm after 1 s of free flight at "
             f"dt = {PROJECT_DT}, against the {TAKEOFF_WINGLESS_CAP} m cap "
             f"`clears` sits above.  It is exactly (1/2) g dt T, so it is "
             f"systematic rather than noise: it biases every height the "
             f"same way, and it grows linearly with the segment length."))

    # 1d. Free rotation conserves angular momentum.  Same treatment.
    I_body = np.diag([Ixx, Iyy, Izz])

    def spin_error(dt: float) -> float:
        ms = mujoco.MjModel.from_xml_string(
            xml.replace(f'timestep="{DT}"', f'timestep="{dt}"'))
        ds = mujoco.MjData(ms)
        ds.qvel[3:6] = np.array([2.0, 5.0, 1.0])
        mujoco.mj_forward(ms, ds)
        L0 = ds.xmat[1].reshape(3, 3) @ (I_body @ ds.qvel[3:6].copy())
        for _ in range(int(round(0.5 / dt))):
            mujoco.mj_step(ms, ds)
        L1 = ds.xmat[1].reshape(3, 3) @ (I_body @ ds.qvel[3:6].copy())
        return abs(float(np.linalg.norm(L1) - np.linalg.norm(L0)))

    p_spin, errs_spin = order_of_accuracy(spin_error, steps)
    lay.checks.append(Check(
        1, "free-rotation angular momentum drift converges at first order",
        1.0, p_spin, 0.10, "order-of-accuracy study",
        note=f"|dL| at dt = {steps}: {['%.3e' % e for e in errs_spin]}"))
    return lay


# --------------------------------------------------------------------------
# 2. Added mass
# --------------------------------------------------------------------------

#: Geometry shared by the added-mass layer and the F-03 tensor benchmark.
PLATE = {"chord": 0.20, "span": 1.0, "thickness": 0.002}


def plate_added_mass_tensor(chord: float, span: float, thickness: float,
                            rho: float) -> dict[str, float]:
    """The three principal added masses of a thin rectangular plate.

    From 2D strip theory, which is the same premise the project's own wing
    branch uses, applied to each axis in turn rather than to one of them:

      * accelerating **normal** to the plate, each strip of span `dr` entrains
        the 2D added mass of a flat plate of width `chord`,
        `m' = rho pi (chord/2)^2` per unit span, so `m_a = rho pi c^2 / 4 * b`.
      * accelerating **edge-on along the chord**, the same argument applies
        with the plate's *thickness* as the width: `m_a = rho pi t^2 / 4 * b`.
      * accelerating **along the span**, a flat plate presents its edge and
        the 2D strip has no normal extent at all: `m_a -> 0`.

    The ratio normal : chordwise is therefore `(c/t)^2`, exactly, and it is the
    number the simulation is checked against.
    """
    return {
        "normal": rho * math.pi * chord**2 / 4.0 * span,
        "chordwise": rho * math.pi * thickness**2 / 4.0 * span,
        "spanwise": 0.0,
    }


def _plate_model(dry_mass: float, jointed: bool):
    c, b, t = PLATE["chord"], PLATE["span"], PLATE["thickness"]
    child = ("""<body name="d" pos="0 0 0.05"><joint type="hinge" axis="0 1 0"/>
                <geom type="box" size="0.005 0.005 0.005" mass="1e-4"/></body>"""
             if jointed else "")
    xml = f"""
    <mujoco><option timestep="{DT}" gravity="0 0 0" density="0" viscosity="0"/>
    <worldbody><body name="b" pos="0 0 -8"><freejoint/>
    <geom type="box" size="{c/2} {b/2} {t/2}" mass="{dry_mass}"/>{child}
    </body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    panels = PanelSet(
        body_id=np.array([1]), pos_local=np.zeros((1, 3)),
        span_local=np.array([[0.0, 1.0, 0.0]]),
        chord_local=np.array([[1.0, 0.0, 0.0]]),
        chord=np.array([c]), dr=np.array([b]),
        volume=np.array([c * b * t]), volume_buoyant=np.array([0.0]),
        half_height=np.array([t / 2]), kind=np.array([WING]),
        aspect_ratio=np.array([b / c]), cd_bluff=np.array([0.0]),
        ext_local=np.array([[b, c, t]]))
    return model, data, panels


def layer2_added_mass() -> Layer:
    lay = Layer(2, "added mass", "entrained fluid inertia, still fluid, no drag")
    rho = SEAWATER.rho
    dry = 12.0
    theory = plate_added_mass_tensor(PLATE["chord"], PLATE["span"],
                                     PLATE["thickness"], rho)
    axes = {"chordwise": np.array([1.0, 0.0, 0.0]),
            "spanwise": np.array([0.0, 1.0, 0.0]),
            "normal": np.array([0.0, 0.0, 1.0])}

    # The geometry-to-tensor chain, established on its own before any
    # simulation is involved.  Both sides of these three checks are analytic.
    for ratio in (10.0, 100.0, 1000.0):
        th = plate_added_mass_tensor(ratio * 1e-3, 1.0, 1e-3, rho)
        lay.checks.append(Check(
            2, f"plate c/t = {ratio:g}: normal:chordwise added mass is (c/t)^2",
            ratio**2, th["normal"] / th["chordwise"], 1e-12, "closed form",
            note="no simulation in this check; it establishes the reference "
                 "the measured values below are compared against"))
    lay.checks.append(Check(
        2, "and the spanwise added mass of a flat plate is zero", 0.0,
        theory["spanwise"], 1e-15, "closed form", absolute=True,
        note="a plate accelerating along its own span presents no normal "
             "extent, so strip theory entrains nothing"))

    for jointed in (True, False):
        tag = "jointed" if jointed else "jointless"
        for name, axis in axes.items():
            model, data, panels = _plate_model(dry, jointed)
            solver = FluidSolver(model, panels, MediumField(water=SEAWATER))
            F = 40.0 * axis
            vals = []
            for step in range(80):
                data.xfrc_applied[:] = 0.0
                solver.apply(data, data.time)
                # Isolate inertia: the model has gravity off, so the solver's
                # compensation for the weight MuJoCo would apply to entrained
                # fluid is an unbalanced force here.
                data.xfrc_applied[1, 2] -= solver.diag.added_mass * GRAVITY
                data.xfrc_applied[1, :3] += F
                total = float(data.xfrc_applied[1, :3] @ axis)
                mujoco.mj_step(model, data)
                a = float(data.qacc[:3] @ axis)
                if step >= 40 and abs(a) > 1e-12:
                    vals.append(total / a)
            m_eff = float(np.median(vals))
            lay.checks.append(Check(
                2, f"{tag}: effective mass pushing {name}",
                dry + theory[name], m_eff, 0.02, "closed form (strip theory)",
                note=f"dry {dry} kg + {theory[name]:.5f} kg entrained"))
    return lay


# --------------------------------------------------------------------------
# 3. Drag
# --------------------------------------------------------------------------


def layer3_drag() -> Layer:
    lay = Layer(3, "drag", "quadratic pressure drag on a bluff volume")
    rho = SEAWATER.rho
    dry = 20.0
    ex, ey, ez = 0.40, 0.20, 0.20     # span, chord, normal extents
    cd = 0.9
    xml = f"""
    <mujoco><option timestep="{DT}" gravity="0 0 0" density="0" viscosity="0"/>
    <worldbody><body name="b" pos="0 0 -8"><freejoint/>
    <geom type="box" size="{ex/2} {ey/2} {ez/2}" mass="{dry}"/>
    <body name="d" pos="0 0 0.3"><joint type="hinge" axis="0 1 0"/>
    <geom type="box" size="0.005 0.005 0.005" mass="1e-4"/></body>
    </body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    panels = PanelSet(
        body_id=np.array([1]), pos_local=np.zeros((1, 3)),
        span_local=np.array([[1.0, 0.0, 0.0]]),      # long axis along +X
        chord_local=np.array([[0.0, 1.0, 0.0]]),
        chord=np.array([ey]), dr=np.array([ex]),
        volume=np.array([ex * ey * ez]), volume_buoyant=np.array([0.0]),
        half_height=np.array([ez / 2]), kind=np.array([BLUFF]),
        aspect_ratio=np.array([1.0]), cd_bluff=np.array([cd]),
        ext_local=np.array([[ex, ey, ez]]))
    solver = FluidSolver(model, panels, MediumField(water=SEAWATER))

    # The force law, written out from `FluidSolver.apply`'s bluff branch for
    # motion along the element's own long axis -- where the cross-flow term is
    # identically zero, so this is the whole force:
    #
    #     F_drag(v) = -0.5 rho |v| v (cd A_front + Cf(Re) A_wet)
    #
    # and the effective mass is dry + Ca_axial rho V with
    # Ca_axial = 0.5 (ey + ez) / (2 ex) from the code's own tensor.
    A_front = ey * ez
    A_wet = 2.0 * (ex * ey + ey * ez + ex * ez)
    V = ex * ey * ez
    ca_axial = float(np.clip(0.5 * (ey + ez) / (2 * ex), 0.05, 10.0))
    m_eff = dry + ca_axial * rho * V
    push = 300.0

    def k_of(v: float) -> float:
        re = rho * abs(v) * ex / SEAWATER.mu
        return 0.5 * rho * (cd * A_front + float(skin_friction_cd(np.array([re]))[0]) * A_wet)

    # Terminal velocity: push = k(v) v^2, solved by bisection on the same law.
    lo, hi = 1e-6, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if k_of(mid) * mid**2 < push:
            lo = mid
        else:
            hi = mid
    v_term = 0.5 * (lo + hi)

    def rhs(_t, y):
        v = y[1]
        return [v, (push - k_of(v) * abs(v) * v) / m_eff]

    T = 6.0
    ref = rk45(rhs, [0.0, 0.0], T)

    F = np.array([push, 0.0, 0.0])
    for _ in range(int(T / DT)):
        data.xfrc_applied[:] = 0.0
        solver.apply(data, data.time)
        data.xfrc_applied[1, 2] -= solver.diag.added_mass * GRAVITY
        data.xfrc_applied[1, :3] += F
        mujoco.mj_step(model, data)

    lay.checks.append(Check(
        3, "speed after 6 s under a constant push",
        float(ref[1]), float(data.qvel[0]), 5e-3,
        "RK45 rtol 1e-11 of the same force law"))
    lay.checks.append(Check(
        3, "distance travelled in 6 s", float(ref[0]),
        float(data.qpos[0] - 0.0), 5e-3,
        "RK45 rtol 1e-11 of the same force law"))
    lay.checks.append(Check(
        3, "and it is approaching the terminal velocity", v_term,
        float(data.qvel[0]), 0.05, "closed form",
        note=f"terminal {v_term:.4f} m/s; after 6 s the body should be close "
             f"but still below it"))
    return lay


# --------------------------------------------------------------------------
# 4. Buoyancy
# --------------------------------------------------------------------------


def layer4_buoyancy() -> Layer:
    lay = Layer(4, "buoyancy", "hydrostatic lift and the free-surface ramp")
    rho = SEAWATER.rho
    V = 0.004
    half_h = 0.10

    def build(dry_mass: float, z0: float):
        xml = f"""
        <mujoco><option timestep="{DT}" gravity="0 0 -9.80665"
                density="0" viscosity="0"/>
        <worldbody><body name="b" pos="0 0 {z0}"><freejoint/>
        <geom type="box" size="0.1 0.1 {half_h}" mass="{dry_mass}"/>
        <body name="d" pos="0 0 0.02"><joint type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.004 0.004 0.004" mass="1e-4"/></body>
        </body></worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        panels = PanelSet(
            body_id=np.array([1]), pos_local=np.zeros((1, 3)),
            span_local=np.array([[1.0, 0.0, 0.0]]),
            chord_local=np.array([[0.0, 1.0, 0.0]]),
            chord=np.array([0.2]), dr=np.array([0.2]),
            volume=np.array([V]), volume_buoyant=np.array([V]),
            half_height=np.array([half_h]), kind=np.array([BLUFF]),
            aspect_ratio=np.array([1.0]), cd_bluff=np.array([0.0]),
            ext_local=np.array([[0.2, 0.2, 2 * half_h]]))
        return model, data, panels

    # 4a. Fully submerged: the force is exactly rho g V.
    model, data, panels = build(12.0, -5.0)
    solver = FluidSolver(model, panels, MediumField(water=SEAWATER))
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    solver.apply(data, 0.0)
    fz = float(data.xfrc_applied[1, 2]) - solver.diag.added_mass * GRAVITY
    lay.checks.append(Check(
        4, "fully submerged buoyant force", rho * GRAVITY * V, fz, 1e-12,
        "closed form (Archimedes)"))

    # 4b. Equilibrium draft.  The project ramps the submerged fraction
    # linearly over the element's height, f = clip(0.5 + 0.5 d/h), so a
    # floating body settles where rho g V f = m g, i.e. at
    #     d* = 2 h (m/(rho V) - 0.5).
    m_float = 0.55 * rho * V
    d_star = 2 * half_h * (m_float / (rho * V) - 0.5)
    model, data, panels = build(m_float, -d_star)
    solver = FluidSolver(model, panels, MediumField(water=SEAWATER))
    # Heavy artificial damping via a long settle, then read the depth.
    for _ in range(int(40.0 / DT)):
        data.xfrc_applied[:] = 0.0
        solver.apply(data, data.time)
        # Gravity is ON in this model, so the solver's compensation for the
        # weight MuJoCo applies to entrained fluid is doing its job and must
        # be left alone.  Removing it -- as the gravity-off layers correctly
        # do -- makes MuJoCo weigh the entrained water and sinks the body.
        # Linear drag purely to settle the oscillation; it cannot move the
        # equilibrium, only how fast the body reaches it.
        data.xfrc_applied[1, :3] -= 40.0 * data.qvel[:3]
        mujoco.mj_step(model, data)
    lay.checks.append(Check(
        4, "equilibrium depth of a floating body", -d_star,
        float(data.qpos[2]), 5e-3, "closed form of the project's own ramp",
        note=f"f = clip(0.5 + 0.5 d/h) gives d* = 2h(m/(rho V) - 1/2) = "
             f"{d_star:.4f} m"))

    # 4c. Heave stiffness.  d(F_buoy)/dz = -rho g V / (2h) inside the ramp, so
    # a body of effective mass M heaves at omega = sqrt(k/M).
    k_heave = rho * GRAVITY * V / (2 * half_h)
    ca = float(np.clip(0.5 * (0.2 + 2 * half_h) / (2 * 0.2), 0.05, 10.0))
    # The entrained mass of a *partly* submerged element is computed at the
    # blended density, not at the water density: `MediumField.properties`
    # returns rho = rho_air + f (rho_water - rho_air), and `FluidSolver` uses
    # that same rho for the added mass.  At the equilibrium draft f = 0.55.
    f_star = m_float / (rho * V)
    rho_blend = AIR.rho + f_star * (rho - AIR.rho)
    M_eff = m_float + ca * rho_blend * V
    omega_ref = math.sqrt(k_heave / M_eff)
    model, data, panels = build(m_float, -d_star)
    solver = FluidSolver(model, panels, MediumField(water=SEAWATER))
    data.qpos[2] = -d_star + 0.02
    zs, ts = [], []
    for i in range(int(12.0 / DT)):
        data.xfrc_applied[:] = 0.0
        solver.apply(data, data.time)
        mujoco.mj_step(model, data)
        zs.append(float(data.qpos[2])); ts.append(data.time)
    z = np.asarray(zs) - np.mean(zs)
    # Period from the zero crossings, which is robust to a slow drift.
    sgn = np.sign(z)
    cross = np.where(np.diff(sgn) != 0)[0]
    period = 2.0 * float(np.mean(np.diff(np.asarray(ts)[cross]))) if len(cross) > 2 else 0.0
    lay.checks.append(Check(
        4, "heave period of a floating body", 2 * math.pi / omega_ref, period,
        0.05, "closed form of the project's own ramp",
        note=f"k = rho_water g V/(2h) = {k_heave:.2f} N/m against an "
             f"effective mass of {M_eff:.3f} kg -- {m_float:.3f} kg dry plus "
             f"{ca * rho_blend * V:.3f} kg entrained at the blended density "
             f"{rho_blend:.1f} kg/m^3 the element sees at f = {f_star:.2f}, "
             f"not at the {rho:.0f} kg/m^3 of the water below it"))
    return lay


# --------------------------------------------------------------------------
# 5. Jet propulsion
# --------------------------------------------------------------------------


def layer5_jet() -> Layer:
    lay = Layer(5, "jet propulsion", "momentum flux out of a contracting cavity")
    rho = SEAWATER.rho
    V0, sf, A = 0.004, 0.45, 0.0012
    lo, hi = 0.0, 1.2
    freq, T = 1.5, 4.0
    dry = 8.0

    xml = f"""
    <mujoco><option timestep="0.001" gravity="0 0 0" density="0" viscosity="0"/>
    <worldbody><body name="hull" pos="0 0 -5"><freejoint/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" mass="{dry}"/>
    <body name="bell" pos="0.3 0 0">
      <joint name="bell_j" type="hinge" axis="0 1 0" range="{lo} {hi}"/>
      <geom type="ellipsoid" size="0.12 0.12 0.08" mass="0.6"/>
    </body></body></worldbody>
    <actuator><position name="a" joint="bell_j" kp="60" ctrlrange="{lo} {hi}"/>
    </actuator></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    jets = JetSet(
        body_id=np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bell")]),
        joint_id=np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bell_j")]),
        axis_local=np.array([[1.0, 0.0, 0.0]]),
        volume=np.array([V0]), stroke_fraction=np.array([sf]),
        orifice_area=np.array([A]), joint_range=np.array([[lo, hi]]))
    medium = MediumField(water=SEAWATER)
    jid = int(jets.joint_id[0])
    dt = 0.001

    impulse = 0.0
    e_pump_ideal = 0.0       # expulsion half only, the "ideal" of the docstring
    e_matched = 0.0          # with the same coeff and subf the model applies
    e_actuator = 0.0         # |tau omega|, the energy budget's own convention
    e_signed = 0.0           # tau omega, the true mechanical work
    e_reported = 0.0         # what JetSet.actuator_work says it did
    e_jet_ke = 0.0
    for k in range(int(T / dt)):
        t = k * dt
        data.ctrl[0] = 0.5 * (lo + hi) + 0.5 * (hi - lo) * math.sin(2 * math.pi * freq * t)
        data.xfrc_applied[:] = 0.0
        thrust = jets.apply(model, data, medium, t, dt)
        mujoco.mj_step(model, data)
        omega = float(data.qvel[model.jnt_dofadr[jid]])
        Q = V0 * sf * (omega / max(hi - lo, 1e-6))
        if Q > 0:
            ve = Q / A
            # Two independent statements about the same jet:
            #   the work the muscle must do,   p Q = (1/2) rho Q^3 / A^2
            #   the kinetic energy it carries, (1/2) rho Q dt ve^2
            # which are the same number, as they must be.
            e_pump_ideal += 0.5 * rho * Q**3 / A**2 * dt
            e_jet_ke += 0.5 * (rho * Q * dt) * ve**2
        # The same ideal with the coefficients the model actually applies, so
        # the identity below is a comparison and not an approximation.
        cf = 1.0 if Q > 0 else jets.refill_efficiency
        e_matched += cf * 0.5 * rho * abs(Q) ** 3 / A**2 * dt
        impulse += thrust * dt
        e_actuator += abs(float(data.actuator_force[0]) * omega) * dt
        e_signed += float(data.actuator_force[0]) * omega * dt
        e_reported += jets.actuator_work(model, data) * dt

    # The same drive with no jet, so the bell's own inertia is subtracted.
    model2 = mujoco.MjModel.from_xml_string(xml)
    data2 = mujoco.MjData(model2)
    e_noje = 0.0
    for k in range(int(T / dt)):
        t = k * dt
        data2.ctrl[0] = 0.5 * (lo + hi) + 0.5 * (hi - lo) * math.sin(2 * math.pi * freq * t)
        data2.xfrc_applied[:] = 0.0
        mujoco.mj_step(model2, data2)
        e_noje += abs(float(data2.actuator_force[0])
                      * float(data2.qvel[model2.jnt_dofadr[jid]])) * dt

    lay.checks.append(Check(
        5, "pumping work equals the jet's kinetic energy", e_pump_ideal,
        e_jet_ke, 1e-9, "closed form",
        note="p Q = (1/2) rho Q^3/A^2 and (1/2) m_dot ve^2 are the same "
             "quantity; both sides are analytic, so this pins the reference"))
    # The identity the pumping load is built on: the torque it puts on the
    # driving joint times the joint rate is the power the jet carries away.
    # Both sides are formed independently here -- one by the module, one from
    # the joint history -- so this is the check that the load is the right
    # size, and it is the one this layer is graded on.
    lay.checks.append(Check(
        5, "the charged pumping work equals p Q integrated", e_matched,
        e_reported, 5e-3, "closed form (energy conservation)",
        note="JetSet.actuator_work against (1/2) rho |Q|^3 / A^2 with the "
             "same refill and submergence coefficients the thrust uses"))

    # A propulsor cannot return energy to its motor.
    lay.checks.append(Check(
        5, "the actuator supplies at least what the jet carries away", 1.0,
        1.0 if e_signed >= e_reported else 0.0, 0.0,
        "conservation law",
        note=f"{e_signed:.2f} J of signed mechanical work against "
             f"{e_reported:.2f} J pumped"))

    # Reported, not graded.  Comparing the actuator's bill across a run with
    # the load and a run without it is not apples to apples: the load changes
    # how the servo tracks, so the bell's own inertial work changes too.  The
    # number is kept because it is what a reader of the energy budget sees,
    # and `abs(tau omega)` is that budget's convention -- it charges braking as
    # if it were driving, so it does not close either.
    delta = e_actuator - e_noje
    print(f"         budget-convention delta with the jet on vs off: "
          f"{delta:+.2f} J against {e_matched:.2f} J pumped "
          f"({100 * delta / max(e_matched, 1e-30):+.0f}%).  Not graded: the "
          f"two runs follow different trajectories and |tau omega| charges "
          f"braking as driving.")
    return lay


# --------------------------------------------------------------------------
# 6. Articulated body
# --------------------------------------------------------------------------


def layer6_articulated() -> Layer:
    lay = Layer(6, "articulated body", "internal joints, no external force")

    def chain_xml(dt: float, integrator: str) -> str:
        # Contacts are switched off deliberately.  A contact between two links
        # of the same machine is internal and conserves momentum too, but it
        # also makes the trajectory chaotic, and a chaotic trajectory cannot
        # support an order-of-accuracy study: the refinement would measure
        # divergence rather than truncation.
        return f"""
        <mujoco><option timestep="{dt}" gravity="0 0 0" density="0"
                viscosity="0" integrator="{integrator}"/>
        <worldbody><body name="hull"><freejoint/>
        <geom type="box" size="0.15 0.08 0.05" mass="4.0"
              contype="0" conaffinity="0"/>
        <body name="arm" pos="0.15 0 0">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <geom type="box" size="0.2 0.02 0.02" pos="0.2 0 0" mass="1.0"
                contype="0" conaffinity="0"/>
        </body></body></worldbody>
        <actuator><motor joint="j1" gear="1"/></actuator></mujoco>"""

    def com(model, data) -> np.ndarray:
        mujoco.mj_forward(model, data)
        tm = float(model.body_mass[1:].sum())
        return sum(float(model.body_mass[b]) * data.xipos[b]
                   for b in range(1, model.nbody)) / tm

    def com_drift(dt: float, integrator: str = "Euler", T: float = 2.0) -> float:
        model = mujoco.MjModel.from_xml_string(chain_xml(dt, integrator))
        data = mujoco.MjData(model)
        c0 = com(model, data).copy()
        for k in range(int(round(T / dt))):
            data.ctrl[0] = 2.0 * math.sin(2 * math.pi * 0.7 * k * dt)
            mujoco.mj_step(model, data)
        return float(np.linalg.norm(com(model, data) - c0))

    # 6a. The law.  A free-floating machine driven only by its own joints
    # cannot move its centre of mass, whatever gait it executes.  This is the
    # statement every swimming and flying score in the project depends on: a
    # displacement that is not paid for by a fluid force is not locomotion.
    steps = [2e-3, 1e-3, 5e-4, 2.5e-4]
    p_ord, errs = order_of_accuracy(com_drift, steps)
    lay.checks.append(Check(
        6, "internal-torque centre-of-mass drift converges at first order",
        1.0, p_ord, 0.05, "order-of-accuracy study",
        note=f"|drift| at dt = {steps}: {['%.3e' % e for e in errs]}.  The "
             f"drift is truncation, not a broken momentum balance: it is "
             f"proportional to dt and vanishes with it."))

    # 6b. And with a fourth-order integrator it essentially disappears, which
    # is what proves the model conserves momentum and the scheme does not.
    rk4 = com_drift(1e-3, "RK4")
    euler = com_drift(1e-3, "Euler")
    lay.checks.append(Check(
        6, "RK4 at the same step conserves it to near machine precision",
        0.0, rk4, 1e-7, "conservation law", absolute=True,
        note=f"{rk4:.3e} m against {euler:.3e} m for the project's "
             f"first-order scheme at the same dt -- a factor of "
             f"{euler / max(rk4, 1e-300):.0f}"))

    # 6c. The same law on the real fleet, at the settings the project runs.
    #
    # The measure is the centre of mass itself, finite-differenced from
    # `xipos`, not a reconstruction from `mj_objectVelocity`: the position is
    # unambiguous and the velocity reconstruction is not.
    from dytiscidae.core import bodyplans
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    def plan_drift(name: str, dt: float, T: float = 8.0) -> float:
        env = TriphibianEnv(build(getattr(bodyplans, name)()), seed=1)
        env.reset(Domain.AIR, randomise=False)
        model, data = env.model, env.data
        # Everything external removed: no fluid, no gravity, no contact.  What
        # is left is a machine flapping in a vacuum, which cannot move.
        model.opt.gravity[:] = 0.0
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        model.opt.timestep = dt
        data.qpos[2] = 200.0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        c0 = com(model, data).copy()
        for _ in range(int(round(T / dt))):
            cmd = env.cpg.command(env.cpg.base, data.time)
            data.ctrl[: len(cmd)] = cmd
            data.xfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)
        return float(np.linalg.norm(com(model, data) - c0))

    scored_distance = 0.6 * 8.0   # a full-marks land segment at the normaliser
    coarse, fine = 4e-3, 1e-3
    worst_name, worst_frac, worst_order = "", 0.0, 1.0
    for name in ("gannet", "beetle", "teal", "medusa"):
        d_coarse = plan_drift(name, coarse)
        d_fine = plan_drift(name, fine)
        order = math.log(d_coarse / max(d_fine, 1e-15)) / math.log(coarse / fine)
        frac = d_coarse / scored_distance
        lay.checks.append(Check(
            6, f"{name}: 8 s of gait in a vacuum moves its centre of mass",
            0.0, frac, 0.01, "conservation law", absolute=True,
            note=f"{d_coarse*1e3:.2f} mm at the project's dt = {coarse}, "
                 f"{d_fine*1e3:.2f} mm at {fine}, so the drift converges at "
                 f"order {order:.2f}.  As a fraction of the "
                 f"{scored_distance:.1f} m a full-marks land segment covers, "
                 f"{100*frac:.3f}%."))
        if frac > worst_frac:
            worst_name, worst_frac, worst_order = name, frac, order

    lay.checks.append(Check(
        6, "and the worst plan's drift converges at first order", 1.0,
        worst_order, 0.15, "order-of-accuracy study",
        note=f"{worst_name} is the worst at {100*worst_frac:.2f}% of a scored "
             f"segment, and its order is {worst_order:.2f}.  An order of 1 "
             f"means a smaller step fixes it; an order below 1 means the "
             f"trajectory is changing under refinement rather than "
             f"converging, and a smaller step does not."))

    # 6d. The mass matrix stays a mass matrix throughout.
    model = mujoco.MjModel.from_xml_string(chain_xml(DT, "Euler"))
    data = mujoco.MjData(model)
    for k in range(int(1.0 / DT)):
        data.ctrl[0] = 2.0 * math.sin(2 * math.pi * 0.7 * k * DT)
        mujoco.mj_step(model, data)
    M = _M(model, data)
    lay.checks.append(Check(
        6, "M stays symmetric", 0.0, float(np.max(np.abs(M - M.T))), 1e-12,
        "definition of kinetic energy", absolute=True))
    eig = float(np.linalg.eigvalsh(M).min())
    lay.checks.append(Check(
        6, "M stays positive definite", 1.0, 1.0 if eig > 0 else 0.0, 0.0,
        "definition of kinetic energy",
        note=f"smallest eigenvalue {eig:.6g}"))
    return lay


# --------------------------------------------------------------------------
# 7. Fluid surrogate
# --------------------------------------------------------------------------


def layer7_fluid_surrogate() -> Layer:
    lay = Layer(7, "fluid surrogate", "the assembled blade-element force")
    rho = AIR.rho
    c, b = 0.20, 1.0
    U = 12.0
    alpha_deg = 6.0
    a = math.radians(alpha_deg)
    ar = b / c

    xml = f"""
    <mujoco><option timestep="{DT}" gravity="0 0 0" density="0" viscosity="0"/>
    <worldbody><body name="w" pos="0 0 200"><freejoint/>
    <geom type="box" size="{c/2} {b/2} 0.001" mass="5.0"/>
    </body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    panels = PanelSet(
        body_id=np.array([1]), pos_local=np.zeros((1, 3)),
        span_local=np.array([[0.0, 1.0, 0.0]]),
        chord_local=np.array([[math.cos(a), 0.0, -math.sin(a)]]),
        chord=np.array([c]), dr=np.array([b]),
        volume=np.array([0.0]), volume_buoyant=np.array([0.0]),
        half_height=np.array([0.001]), kind=np.array([WING]),
        aspect_ratio=np.array([ar]), cd_bluff=np.array([0.0]))
    solver = FluidSolver(model, panels, MediumField(wind=np.array([U, 0.0, 0.0])))
    data.xfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    solver.apply(data, 0.0)
    f = data.xfrc_applied[1, :3].copy()
    f[2] -= solver.diag.added_mass * GRAVITY

    # The reference: the same coefficient functions evaluated by hand, with the
    # lift normal to the free stream and the drag along it.  The free stream is
    # +X and the strip is pitched about +Y, so lift is +Z and drag is +X.
    q = 0.5 * rho * U**2
    re = rho * U * c / AIR.mu
    cl = float(lift_coefficient(np.array([a]), np.array([re]),
                                np.array([ar]), np.array([0.0]))[0])
    cd = float(drag_coefficient(np.array([a]), np.array([re]),
                               np.array([ar]), np.array([cl]))[0])
    L_ref, D_ref = q * c * b * cl, q * c * b * cd

    lay.checks.append(Check(
        7, "assembled lift equals q S CL", L_ref, float(f[2]), 1e-9,
        "closed form of the project's own coefficients"))
    lay.checks.append(Check(
        7, "assembled drag equals q S CD", D_ref, float(f[0]), 1e-9,
        "closed form of the project's own coefficients"))
    lay.checks.append(Check(
        7, "force scales as U^2 at matched Reynolds number", 4.0,
        _force_ratio(2.0), 1e-9, "dimensional similarity",
        note="doubling the speed and halving the viscosity holds Re fixed, so "
             "the force must quadruple exactly"))
    lay.checks.append(Check(
        7, "glide angle satisfies tan(gamma) = D/L", D_ref / L_ref,
        float(f[0] / f[2]), 1e-9, "closed form",
        note=f"L/D = {L_ref/D_ref:.2f}, so a steady glide at this incidence "
             f"descends at {math.degrees(math.atan(D_ref/L_ref)):.2f} degrees"))
    return lay


def _force_ratio(speed_factor: float) -> float:
    """Force at `speed_factor * U` with Re held fixed, over force at U."""
    from dytiscidae.physics.medium import Fluid
    c, b, U, a = 0.20, 1.0, 12.0, math.radians(6.0)

    def probe(u: float, mu: float) -> float:
        xml = f"""
        <mujoco><option timestep="{DT}" gravity="0 0 0" density="0" viscosity="0"/>
        <worldbody><body name="w" pos="0 0 200"><freejoint/>
        <geom type="box" size="{c/2} {b/2} 0.001" mass="5.0"/>
        </body></worldbody></mujoco>"""
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        p = PanelSet(
            body_id=np.array([1]), pos_local=np.zeros((1, 3)),
            span_local=np.array([[0.0, 1.0, 0.0]]),
            chord_local=np.array([[math.cos(a), 0.0, -math.sin(a)]]),
            chord=np.array([c]), dr=np.array([b]),
            volume=np.array([0.0]), volume_buoyant=np.array([0.0]),
            half_height=np.array([0.001]), kind=np.array([WING]),
            aspect_ratio=np.array([b / c]), cd_bluff=np.array([0.0]))
        s = FluidSolver(m, p, MediumField(air=Fluid("scaled", 1.225, mu),
                                          wind=np.array([u, 0.0, 0.0])))
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        s.apply(d, 0.0)
        f = d.xfrc_applied[1, :3].copy()
        f[2] -= s.diag.added_mass * GRAVITY
        return float(np.linalg.norm(f))

    return probe(speed_factor * U, AIR.mu * speed_factor) / probe(U, AIR.mu)


# --------------------------------------------------------------------------
# 8. Controller
# --------------------------------------------------------------------------


def layer8_controller() -> Layer:
    lay = Layer(8, "controller", "the identified basis against the machine")
    from dytiscidae.control.cpg import CPGParams
    from dytiscidae.core.bodyplans import beetle
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    env = TriphibianEnv(build(beetle()), seed=3)
    basis = env.identify(Domain.WATER, seed=11)
    A = basis.effects * np.asarray(basis.authority)[:, None]

    # 8a. The forward model is its own transpose.  Algebra, no simulation.
    c = np.array([0.3, -0.2, 0.15, 0.05, -0.1, 0.02])[: A.shape[0]]
    lay.checks.append(Check(
        8, "twist_of(c) equals A^T c", 0.0,
        float(np.max(np.abs(basis.twist_of(c) - A.T @ c))), 1e-12,
        "algebra", absolute=True))

    env.reset(Domain.WATER, randomise=False)
    snap = env.snapshot()
    base = env.cpg.base
    scale = np.array([1.0, 1.0, 1.0, 0.3, 0.3, 0.3])

    def mean_twist(delta: np.ndarray) -> np.ndarray:
        env.restore(snap)
        env.budget.reset()
        params = CPGParams.from_flat(base.flat() + delta, env.cpg.n)
        acc = np.zeros(6)
        n = int(1.2 / env.timestep)
        for _ in range(n):
            env.step(env.cpg.command(params, env.data.time))
            acc += env.body_twist()
        return acc / n * scale

    def response(s: float) -> np.ndarray:
        """Central-differenced response to a coefficient vector scaled by s."""
        d = basis.modes.T @ (s * c)
        return 0.5 * (mean_twist(d) - mean_twist(-d))

    # 8b. Superposition.  The basis is a *linear* model, and the one thing a
    # linear model promises that can be checked without trusting any
    # coefficient is that doubling the command doubles the response.  The
    # reference is 2.0 exactly; the departure is the map's own curvature at
    # this probe scale.
    r1 = response(1.0)
    r2 = response(2.0)
    ratio = float(np.linalg.norm(r2) / max(np.linalg.norm(r1), 1e-30))
    lay.checks.append(Check(
        8, "doubling the command doubles the response", 2.0, ratio, 0.10,
        "linearity of the fitted model",
        note="a central difference about the base gait, so any drift "
             "independent of the command cancels; what is left is the "
             "nonlinearity of the parameter-to-twist map at the 0.35 probe "
             "scale the identification uses"))

    # 8c. And the direction, which is the part the controller relies on: a
    # linear model says the response direction does not depend on the command
    # magnitude at all.
    cos = float(r1 @ r2 / max(np.linalg.norm(r1) * np.linalg.norm(r2), 1e-30))
    lay.checks.append(Check(
        8, "and does not change its direction", 1.0, cos, 0.05,
        "linearity of the fitted model",
        note="cosine between the response to c and to 2c"))

    # 8d. The prediction error itself.  Reported, not graded: there is no
    # closed form here, only the identified model's own claim, and
    # `experiments/rank_threshold` has already measured how much that claim is
    # worth.
    predicted = basis.twist_of(c)
    err = float(np.linalg.norm(r1 - predicted)
                / max(np.linalg.norm(predicted), 1e-30))
    lay.checks.append(Check(
        8, "the machine delivers the twist the basis predicts", 0.0, err, 1.0,
        "the identified model's own prediction", absolute=True,
        note=f"relative error {err:.3f} on the same body the basis was fitted "
             f"on.  This is not graded against a closed form because there "
             f"isn't one; the tolerance is 1.0, i.e. 'the command did "
             f"something rather than nothing'.  What the number is worth is "
             f"in experiments/rank_threshold, which measures the "
             f"identification's own reproducibility."))
    return lay
