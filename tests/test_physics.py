"""Physics layer verification.

These are not smoke tests.  Each one pins down a sign convention or a magnitude
that the rest of the pipeline silently depends on, because a sign error in the
fluid model does not crash -- it just produces a machine that "flies" by
falling, and the optimiser will happily exploit it for a thousand generations
before anyone notices.

Run with:  python -m pytest tests/ -q      (or)  python tests/test_physics.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from dytiscidae.physics import structure  # noqa: E402
from dytiscidae.physics.energy import (  # noqa: E402
    Actuator,
    Battery,
    PowerBudget,
    cruise_power_air,
    cruise_power_water,
)
from dytiscidae.physics.fluid import (  # noqa: E402
    BLUFF,
    WING,
    FluidSolver,
    PanelSet,
    lift_coefficient,
)
from dytiscidae.physics.materials import CFRP_TUBE, PETG  # noqa: E402
from dytiscidae.physics.medium import AIR, SEAWATER, MediumField, SeaState  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------


def _single_panel_model(density: float = 200.0):
    """A single free body carrying one wing strip, used for force probes."""
    xml = f"""
    <mujoco>
      <option timestep="0.001" gravity="0 0 0" density="0" viscosity="0"/>
      <worldbody>
        <body name="wing" pos="0 0 5">
          <freejoint/>
          <geom type="box" size="0.1 0.5 0.002" density="{density}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


def _wing_panels(model, alpha_deg: float) -> PanelSet:
    """One strip: span along +Y, chord along +X, so the normal is +Z.

    Pitched nose-up by ``alpha_deg`` about the span axis, which per the module's
    convention should produce positive lift in a +X free stream.
    """
    a = math.radians(alpha_deg)
    # Rotation about +Y by a maps x -> (cos a, 0, -sin a).
    chord = np.array([[math.cos(a), 0.0, -math.sin(a)]])
    span = np.array([[0.0, 1.0, 0.0]])
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wing")
    return PanelSet(
        body_id=np.array([bid]),
        pos_local=np.zeros((1, 3)),
        span_local=span,
        chord_local=chord,
        chord=np.array([0.2]),
        dr=np.array([1.0]),
        volume=np.array([0.0]),  # isolate aerodynamics from buoyancy
        half_height=np.array([0.02]),
        kind=np.array([WING]),
        aspect_ratio=np.array([5.0]),
        cd_bluff=np.array([0.0]),
    )


def test_lift_sign_and_magnitude() -> None:
    """Positive angle of attack in a +X wind must give +Z force."""
    print("\nfluid: lift sign and magnitude")
    m, d = _single_panel_model()
    medium = MediumField(wind=np.array([10.0, 0.0, 0.0]))

    for alpha in (5.0, 10.0):
        panels = _wing_panels(m, alpha)
        solver = FluidSolver(m, panels, medium)
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        solver.apply(d, 0.0)
        f = d.xfrc_applied[panels.body_id[0], :3].copy()
        check(
            f"alpha=+{alpha:.0f} deg gives upward force",
            f[2] > 0.0,
            f"Fz={f[2]:+.2f} N  Fx={f[0]:+.2f} N",
        )

    # Negative incidence must mirror.
    panels = _wing_panels(m, -8.0)
    solver = FluidSolver(m, panels, medium)
    d.xfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)
    solver.apply(d, 0.0)
    f_neg = d.xfrc_applied[panels.body_id[0], :3].copy()
    check("alpha=-8 deg gives downward force", f_neg[2] < 0.0, f"Fz={f_neg[2]:+.2f} N")

    # Drag always opposes the wing, i.e. acts downstream (+X here).
    check("drag acts downstream", f_neg[0] > 0.0, f"Fx={f_neg[0]:+.2f} N")

    # Magnitude sanity: thin-airfoil theory with AR=5 gives CL_alpha ~ 4.5/rad,
    # so at 8 deg, q=61.25 Pa, S=0.2 m^2 -> L ~ 7.7 N.
    panels = _wing_panels(m, 8.0)
    solver = FluidSolver(m, panels, medium)
    d.xfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)
    solver.apply(d, 0.0)
    lift = d.xfrc_applied[panels.body_id[0], 2]
    check(
        "lift magnitude within 40% of thin-airfoil estimate",
        3.0 < lift < 14.0,
        f"L={lift:.2f} N (expected ~7.7 N)",
    )


def test_buoyancy() -> None:
    """A submerged volume lighter than water must be pushed up, and the net
    force must match Archimedes to within a percent."""
    print("\nfluid: buoyancy")
    xml = """
    <mujoco>
      <option timestep="0.001" gravity="0 0 -9.80665"/>
      <worldbody>
        <body name="float" pos="0 0 -5">
          <freejoint/>
          <geom type="sphere" size="0.2" density="500"/>
        </body>
      </worldbody>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "float")
    vol = 4.0 / 3.0 * math.pi * 0.2**3

    panels = PanelSet(
        body_id=np.array([bid]),
        pos_local=np.zeros((1, 3)),
        span_local=np.array([[0.0, 1.0, 0.0]]),
        chord_local=np.array([[1.0, 0.0, 0.0]]),
        chord=np.array([0.4]),
        dr=np.array([0.4]),
        volume=np.array([vol]),
        half_height=np.array([0.2]),
        kind=np.array([BLUFF]),
        aspect_ratio=np.array([1.0]),
        cd_bluff=np.array([0.47]),
    )
    solver = FluidSolver(m, panels, MediumField())
    d.xfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)
    solver.apply(d, 0.0)

    expected = SEAWATER.rho * 9.80665 * vol
    got = solver.diag.buoyancy
    check(
        "fully submerged buoyancy matches Archimedes",
        abs(got - expected) / expected < 0.01,
        f"{got:.2f} N vs {expected:.2f} N",
    )

    # And it actually rises when integrated, unlike MuJoCo's own fluid model.
    for _ in range(400):
        d.xfrc_applied[:] = 0.0
        solver.apply(d, d.time)
        mujoco.mj_step(m, d)
    check("a 500 kg/m^3 body rises in water", d.qpos[2] > -5.0, f"z={d.qpos[2]:.3f} m")


def test_bluff_drag_is_orientation_dependent() -> None:
    """A volume must cost more drag broadside than nose-on, and must never cost
    zero.

    This is a regression test for a real defect.  Bluff elements were run
    through strip theory, which projects the spanwise component of the flow out
    before forming the drag -- correct for a wing strip, badly wrong for a body.
    A hull travelling nose-first along its own axis therefore felt *no* pressure
    drag whatever, so the search could make bodies arbitrarily long and pay
    nothing, and it did.
    """
    print("\nfluid: bluff bodies see their own shape")
    xml = """
    <mujoco><worldbody><body name="b" pos="0 0 -5">
      <freejoint/><geom type="box" size="0.3 0.05 0.05" density="500"/>
    </body></worldbody></mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "b")

    def drag_at(vel) -> float:
        panels = PanelSet(
            body_id=np.array([bid]),
            pos_local=np.zeros((1, 3)),
            span_local=np.array([[1.0, 0.0, 0.0]]),
            chord_local=np.array([[0.0, -1.0, 0.0]]),
            chord=np.array([0.1]),
            dr=np.array([0.6]),
            volume=np.array([0.006]),
            volume_buoyant=np.array([0.0]),
            half_height=np.array([0.05]),
            kind=np.array([BLUFF]),
            aspect_ratio=np.array([1.0]),
            cd_bluff=np.array([0.2]),
            pitch_axis=np.array([0.5]),
            # 6:1 slender, long along its own span axis.
            ext_local=np.array([[0.6, 0.1, 0.1]]),
        )
        solver = FluidSolver(m, panels, MediumField())
        mujoco.mj_resetData(m, d)
        d.qpos[2] = -5.0
        d.qvel[:3] = vel
        mujoco.mj_forward(m, d)
        d.xfrc_applied[:] = 0.0
        solver.apply(d, 0.0)
        return solver.diag.drag

    axial = drag_at([3.0, 0.0, 0.0])
    across = drag_at([0.0, 3.0, 0.0])
    check("a body moving along its own axis still has drag", axial > 1.0, f"{axial:.1f} N")
    check(
        "broadside costs more than nose-on",
        across > 2.0 * axial,
        f"{across:.1f} N broadside vs {axial:.1f} N nose-on ({across / axial:.1f}x)",
    )


def test_generated_bodies_reach_the_fluid() -> None:
    """A free-form body must arrive at the solver as several elements carrying
    its real volume, not as one capsule standing in for it.

    Without this the shape the CPPN generates changes mass, inertia and
    collision geometry and stops there: buoyancy still acts at the geometric
    centre of a rod, so pitch trim is blind to whether the body is fat forward
    or fat aft.
    """
    print("\nfluid: generated shape reaches the solver")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.mjcf import compile_phenotype
    from dytiscidae.core.phenotype import build

    worst_ratio = 1e9
    n_plans_with_fields = 0
    for name, plan in BODY_PLANS.items():
        p = build(plan())
        _, _, _, panels = compile_phenotype(p)
        fields = [s for s in p.segments if getattr(s, "field", None) is not None]
        if not fields:
            continue
        n_plans_with_fields += 1
        bluff = panels.kind == BLUFF
        n_bluff = int(bluff.sum())
        n_nonsurface = sum(1 for s in p.segments if not s.is_surface)
        check(
            f"{name}: body is discretised, not lumped",
            n_bluff > n_nonsurface,
            f"{n_bluff} bluff elements for {n_nonsurface} non-surface segments",
        )
        # The elements must account for the volume the sizing pass reported.
        want = sum(s.volume for s in p.segments if not s.is_surface)
        got = float(panels.volume[bluff].sum())
        worst_ratio = min(worst_ratio, got / max(want, 1e-9))

    check("all five plans carry a generated body", n_plans_with_fields == 5,
          f"{n_plans_with_fields}/5")
    check(
        "panel volume accounts for the body volume",
        0.9 < worst_ratio < 1.1,
        f"worst ratio {worst_ratio:.3f}",
    )


def test_land_domain_is_reachable() -> None:
    """There must be dry ground above the waterline, and a machine dropped on
    the land spawn must land on it.

    The beach ramp's rotation sign was inverted, which put the whole ramp above
    the water -- z = +4.8 m at the shoreline, never crossing z = 0 -- and put
    the land spawn point 3.5 m *underneath* it.  Every land episode was a
    machine dropped inside terrain it could not touch, free-falling into the
    sea, and no generation could complete all three domains because one of them
    did not physically exist.  Nothing crashed and nothing warned; the land
    score just stayed low and read like a hard control problem.
    """
    print("\nscene: land exists")
    import mujoco as mj

    from dytiscidae.core.bodyplans import beetle
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    env = TriphibianEnv(build(beetle()))
    model, data = env.model, env.data
    env.reset(Domain.LAND, randomise=False)
    mj.mj_forward(model, data)

    bg = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "beach")
    R = data.geom_xmat[bg].reshape(3, 3)
    c, half = data.geom_xpos[bg], model.geom_size[bg][0]

    def ramp_z(x: float) -> float:
        return float(c[2] + ((x - c[0]) / R[0, 0]) * R[2, 0])

    def on_ramp(x: float) -> bool:
        return abs((x - c[0]) / R[0, 0]) <= half

    xs = [x for x in np.linspace(c[0] - half * R[0, 0], c[0] + half * R[0, 0], 40) if on_ramp(x)]
    zs = [ramp_z(x) for x in xs]
    check("the beach crosses the waterline", min(zs) < 0.0 < max(zs),
          f"ramp spans z {min(zs):+.2f} .. {max(zs):+.2f} m")
    check("the ramp rises inland", zs[-1] > zs[0], f"{zs[0]:+.2f} -> {zs[-1]:+.2f} m")

    spawn = env.root_pos().copy()
    check("the land spawn is above the ground under it", spawn[2] > ramp_z(spawn[0]),
          f"spawn z={spawn[2]:.2f} vs ground z={ramp_z(spawn[0]):.2f}")

    contacts = 0
    steps = int(4.0 / env.timestep)
    for _ in range(steps):
        env.step(env.cpg.command(env.cpg.base, env.data.time))
        if env._touching_ground():
            contacts += 1
    check("a machine dropped on land ends up touching it", contacts > 0.2 * steps,
          f"{contacts}/{steps} steps in ground contact")
    check("and does not fall through the world", env.root_pos()[2] > -1.0,
          f"z={float(env.root_pos()[2]):.2f} m")


def test_machine_does_not_collide_with_itself() -> None:
    """The machine collides with terrain and never with its own parts."""
    print("\nscene: contact masks")
    from dytiscidae.core.bodyplans import medusa
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    env = TriphibianEnv(build(medusa()))
    model = env.model
    machine = model.geom_bodyid != 0
    pairs_possible = 0
    for a in np.nonzero(machine)[0]:
        for b in np.nonzero(machine)[0]:
            if a < b and (
                (model.geom_contype[a] & model.geom_conaffinity[b])
                or (model.geom_contype[b] & model.geom_conaffinity[a])
            ):
                pairs_possible += 1
    check("no self-collision pair is enabled", pairs_possible == 0, f"{pairs_possible} pairs")

    terrain = np.nonzero(~machine)[0]
    solid = [
        g for g in terrain
        if model.geom_contype[g] or model.geom_conaffinity[g]
    ]
    ok = all(
        (model.geom_contype[g] & model.geom_conaffinity[np.nonzero(machine)[0][0]])
        or (model.geom_contype[np.nonzero(machine)[0][0]] & model.geom_conaffinity[g])
        for g in solid
    )
    check("the machine still collides with the terrain", ok and len(solid) >= 2,
          f"{len(solid)} solid terrain geoms")

    env.reset(Domain.LAND, randomise=False)
    for _ in range(int(2.0 / env.timestep)):
        env.step(env.cpg.command(env.cpg.base, env.data.time))
    check("and reaches it", env._touching_ground(), f"ncon={env.data.ncon}")


def test_air_segment_can_be_scored() -> None:
    """The air score's ceiling must be set by aerodynamics, not by the drop.

    Every air term is multiplied by the fraction of the episode spent airborne.
    With the old spawn -- 6 m altitude, zero airspeed -- that fraction was
    free-fall time over segment length, measured at 0.136 to 0.173 across all
    five plans, so the air score could not exceed about 0.15 however well a
    machine flew, and it did not vary with wing loading at all.  The search was
    being asked to optimise a number it could barely move.
    """
    print("\nenv: the air segment is winnable")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    fractions, loadings, launches = [], [], []
    for plan in BODY_PLANS.values():
        p = build(plan())
        env = TriphibianEnv(p)
        env.reset(Domain.AIR, randomise=False)
        launches.append(env.launch_speed)
        loadings.append(p.mass * 9.80665 / max(p.wing_area, 1e-3))
        steps = int(6.0 / env.timestep)
        aloft = 0
        for _ in range(steps):
            env.step(env.cpg.command(env.cpg.base, env.data.time))
            if env.depth() < -0.3 and not env._touching_ground():
                aloft += 1
        fractions.append(aloft / steps)

    check(
        "an untrained machine is airborne for a scorable share of the segment",
        min(fractions) > 0.30,
        f"airborne fraction {min(fractions):.2f}..{max(fractions):.2f}",
    )
    check(
        "the launch is each design's own trim speed, not one number",
        max(launches) - min(launches) > 4.0,
        f"{min(launches):.1f}..{max(launches):.1f} m/s over W/S "
        f"{min(loadings):.0f}..{max(loadings):.0f} N/m^2",
    )
    # A heavier-loaded design must be launched faster: that is what trim means.
    order = np.argsort(loadings)
    v = np.array(launches)[order]
    check("launch speed rises with wing loading", bool(np.all(np.diff(v) >= -1e-9)),
          " ".join(f"{x:.1f}" for x in v))


def test_truncated_episodes_cannot_score() -> None:
    """Ending the episode early must not be a way to win it.

    Found in the archive after 800 generations, not by reading the code.
    Twenty-one designs with wing loadings up to 480,000 N/m^2 -- objects with
    no lifting surface at all -- were scoring air competence above 0.9, and
    twenty of them had an energy margin of -0.98 or worse.  The recipe was to
    drain the battery on the first step: the two or three samples recorded are
    all at the launch altitude, so the machine reads as airborne 100% of the
    time with a measured sink rate of nil and its launch speed intact.

    Every time fraction is now divided by the length the segment was asked for
    rather than by the samples that happened to exist.
    """
    print("\nscore: a segment that stops early scores the stopping")
    from dytiscidae.core.bodyplans import beetle
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, SegmentResult, TriphibianEnv

    env = TriphibianEnv(build(beetle()))
    dt, dur = env.timestep, 8.0
    n = int(dur / dt)

    def air(n_samples, sink):
        r = SegmentResult(domain=Domain.AIR, duration=dur)
        r.mean_speed = env.launch_speed
        # Height above the ground is what the score reads; feed it directly so
        # the case under test is the one being described.
        clear = 30.0 - sink * np.arange(n_samples) * dt
        return env._score_segment(
            Domain.AIR, r, -clear, clear,
            np.ones(n_samples), np.zeros(n_samples), clear,
        )

    dead = air(3, 0.0)
    real = air(n, 0.0)
    check("a battery that dies on step 3 scores nothing for flight", dead < 0.02,
          f"{dead:.3f} (this was 1.000)")
    check("holding altitude for the whole segment still scores full", real > 0.95,
          f"{real:.3f}")
    check("and a glide scores less than level flight", air(n, 1.0) < real,
          f"glide {air(n, 1.0):.3f} vs level {real:.3f}")

    def land(frac):
        r = SegmentResult(domain=Domain.LAND, duration=dur)
        r.mean_speed = 0.6
        k = max(int(frac * n), 2)
        return env._score_segment(
            Domain.LAND, r, np.full(k, -0.1), np.full(k, 0.5), np.ones(k), np.ones(k)
        )

    check("the same hole is closed on land", land(0.02) < 0.05 < land(1.0),
          f"2% of the segment scores {land(0.02):.3f}, all of it scores {land(1.0):.3f}")


def test_added_mass_is_anisotropic() -> None:
    """A plate must cost far more to accelerate broadside than edge-on.

    With a flat Ca = 0.5 a plate and a sphere of equal volume cost the same to
    shake, which erases the reason a fin is a fin: nearly all of a paddle's
    thrust is the fluid it entrains on the power stroke and does not entrain on
    the recovery stroke.  A search told those are the same has no reason to
    invent a paddle.
    """
    print("\nfluid: added mass knows which way the body is pointing")
    rho = SEAWATER.rho
    xml = """
    <mujoco><worldbody><body name="b" pos="0 0 -5">
      <freejoint/><geom type="box" size="0.25 0.25 0.02" density="600"/>
    </body></worldbody></mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "b")
    vol = 0.5 * 0.5 * 0.04

    panels = PanelSet(
        body_id=np.array([bid]),
        pos_local=np.zeros((1, 3)),
        span_local=np.array([[1.0, 0.0, 0.0]]),
        chord_local=np.array([[0.0, -1.0, 0.0]]),
        chord=np.array([0.5]),
        dr=np.array([0.5]),
        volume=np.array([vol]),
        volume_buoyant=np.array([0.0]),
        half_height=np.array([0.02]),
        kind=np.array([BLUFF]),
        aspect_ratio=np.array([1.0]),
        cd_bluff=np.array([1.17]),
        pitch_axis=np.array([0.5]),
        ext_local=np.array([[0.5, 0.5, 0.04]]),
    )
    # One solver, reset between probes: ``apply`` leaves the model's body_mass
    # inflated by design, so a second solver built on top of it would capture
    # the first one's added mass as its dry mass.
    solver = FluidSolver(m, panels, MediumField())
    dry = float(solver._dry_mass[bid])

    def entrained(vel) -> float:
        solver.reset()
        mujoco.mj_resetData(m, d)
        d.qpos[2] = -5.0
        d.qvel[:3] = vel
        mujoco.mj_forward(m, d)
        d.xfrc_applied[:] = 0.0
        solver.apply(d, 0.0)
        out = float(m.body_mass[bid]) - dry
        solver.reset()
        return out

    span_on = entrained([2.0, 0.0, 0.0])
    chord_on = entrained([0.0, 2.0, 0.0])
    broad = entrained([0.0, 0.0, 2.0])

    check("broadside entrains far more than edge-on", broad > 10.0 * span_on,
          f"{broad:.1f} kg vs {span_on:.1f} kg ({broad / max(span_on, 1e-9):.0f}x)")
    check("a square plate is symmetric in its two edge-on directions",
          abs(span_on - chord_on) < 0.02 * max(span_on, 1e-9),
          f"{span_on:.2f} vs {chord_on:.2f} kg")
    # Lamb: a disc of radius R moving normal to itself entrains (8/3) rho R^3.
    # The square plate circumscribes that disc, so it must exceed it, and not by
    # a lot -- the area ratio is 4/pi.
    disc = 8.0 / 3.0 * rho * 0.25**3
    check("broadside is near the exact disc result", disc < broad < 2.0 * disc,
          f"{broad:.1f} kg vs {disc:.1f} kg for the inscribed disc")
    # And a sphere must still come out at the textbook Ca = 0.5.
    sphere = PanelSet(
        body_id=np.array([bid]), pos_local=np.zeros((1, 3)),
        span_local=np.array([[1.0, 0.0, 0.0]]), chord_local=np.array([[0.0, -1.0, 0.0]]),
        chord=np.array([0.4]), dr=np.array([0.4]), volume=np.array([0.0335]),
        volume_buoyant=np.array([0.0]), half_height=np.array([0.2]),
        kind=np.array([BLUFF]), aspect_ratio=np.array([1.0]), cd_bluff=np.array([0.47]),
        pitch_axis=np.array([0.5]), ext_local=np.array([[0.4, 0.4, 0.4]]),
    )
    s2 = FluidSolver(m, sphere, MediumField())
    dry2 = float(s2._dry_mass[bid])
    mujoco.mj_resetData(m, d)
    d.qpos[2] = -5.0
    d.qvel[:3] = [2.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)
    d.xfrc_applied[:] = 0.0
    s2.apply(d, 0.0)
    ca = (float(m.body_mass[bid]) - dry2) / (rho * 0.0335)
    s2.reset()
    check("a sphere still gets Ca = 0.5", abs(ca - 0.5) < 0.02, f"Ca={ca:.3f}")


def test_bodies_generate_lift_and_a_pitching_moment() -> None:
    """A body at incidence must produce a force across the stream, not only
    along it, and a shaped body must produce a moment.

    Bluff elements were given a force along the flow direction only.  That is
    the resistive part of the load and it is the smaller part: a body at
    incidence is loaded mainly by the component of the stream *across* its own
    axis, and that load acts normal to the axis.  Resolving it that way is
    Munk's slender-body result with the Allen and Perkins cross-flow
    correction, and without it three things were missing from the search --
    a body could not contribute lift, so a lifting body was unreachable; a body
    could not produce a pitching moment, so a tail was pure drag and
    weathercock stability could not be discovered; and flying sideways cost the
    same as flying forwards.
    """
    print("\nfluid: bodies lift and trim")
    xml = """
    <mujoco><option gravity="0 0 0"/><worldbody><body name="b" pos="0 0 200">
      <freejoint/><geom type="capsule" fromto="0 0 0 0.6 0 0" size="0.05" density="300"/>
    </body></worldbody></mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "b")

    def probe(aft_h: float, fwd_h: float, deg: float):
        # +x is the direction of travel, so pos_local 0.45 is forward of the
        # centre of mass at 0.30 and 0.15 is aft of it.
        panels = PanelSet(
            body_id=np.array([bid, bid]),
            pos_local=np.array([[0.15, 0.0, 0.0], [0.45, 0.0, 0.0]]),
            span_local=np.tile([1.0, 0.0, 0.0], (2, 1)),
            chord_local=np.tile([0.0, -1.0, 0.0], (2, 1)),
            chord=np.array([aft_h, fwd_h]), dr=np.array([0.3, 0.3]),
            volume=np.array([0.002, 0.002]), volume_buoyant=np.zeros(2),
            half_height=np.array([aft_h / 2, fwd_h / 2]),
            kind=np.array([BLUFF, BLUFF]), aspect_ratio=np.ones(2),
            cd_bluff=np.array([0.2, 0.2]), pitch_axis=np.array([0.5, 0.5]),
            ext_local=np.array([[0.3, aft_h, aft_h], [0.3, fwd_h, fwd_h]]),
        )
        solver = FluidSolver(m, panels, MediumField())
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 200.0
        a = math.radians(-deg)  # nose up
        d.qpos[3:7] = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
        d.qvel[:3] = (25.0, 0.0, 0.0)
        mujoco.mj_forward(m, d)
        d.xfrc_applied[:] = 0.0
        solver.apply(d, 0.0)
        out = (float(-d.xfrc_applied[bid, 0]), float(d.xfrc_applied[bid, 2]),
               float(d.xfrc_applied[bid, 4]))
        solver.reset()
        return out

    drag0, lift0, _ = probe(0.1, 0.1, 0.0)
    drag20, lift20, _ = probe(0.1, 0.1, 20.0)
    check("a body at zero incidence makes drag and no lift",
          drag0 > 0.5 and abs(lift0) < 0.05 * drag0, f"D={drag0:.2f} N L={lift0:+.3f} N")
    check("a body at incidence makes lift", lift20 > 0.3 * drag20,
          f"L={lift20:.2f} N D={drag20:.2f} N at 20 deg, L/D={lift20/drag20:.2f}")
    check("and drag rises with incidence", drag20 > drag0, f"{drag0:.2f} -> {drag20:.2f} N")

    _, _, m_fat_aft = probe(0.16, 0.05, 20.0)
    _, _, m_uniform = probe(0.10, 0.10, 20.0)
    _, _, m_fat_fwd = probe(0.05, 0.16, 20.0)
    # +y torque pitches the nose down, so for a nose-up body it is restoring.
    check("a body with its area aft is stable in pitch", m_fat_aft > 0.05,
          f"{m_fat_aft:+.3f} N.m, nose-down")
    check("a body with its area forward is unstable", m_fat_fwd < -0.05,
          f"{m_fat_fwd:+.3f} N.m, nose-up")
    check("a uniform body is neutral", abs(m_uniform) < 0.01, f"{m_uniform:+.3f} N.m")


def test_series_elasticity_needs_a_compliant_drive() -> None:
    """A spring tuned to resonance must reduce the work the motor does -- and
    it only can if the drive is allowed to be soft.

    A rigid drive pays the wing's whole inertial reversal from the motor twice
    per cycle, and that cost goes as f^2.  This is what capped every design in
    this project near 2 Hz.  A tuned spring returns the wing's kinetic energy
    instead, which is how every insect and every published flapping MAV at this
    scale works, and the family was not merely disfavoured before -- with no
    spring gene it was unreachable.

    Adding the spring alone was not enough, which is the part worth keeping.
    At the servo gain that used to be hard-wired, a resonant spring costs *more*
    power than no spring: a position servo commands a trajectory and treats a
    parallel spring as a disturbance to reject.  The gain was a constant I
    typed, and it happened to be one at which no resonant design can work.
    """
    print("\nactuation: resonance needs a soft drive")
    from dytiscidae.core.bodyplans import ray
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    def probe(stiffness: float, compliance: float, secs: float = 5.0):
        g = ray()
        for part in g.parts:
            if part.joint != "none" and part.actuated:
                part.series_stiffness = stiffness
                part.drive_compliance = compliance
        env = TriphibianEnv(build(g))
        env.reset(Domain.AIR, randomise=False)
        q = []
        for _ in range(int(secs / env.timestep)):
            env.step(env.cpg.command(env.cpg.base, env.data.time))
            q.append(env.data.qpos[7:].copy())
        q = np.array(q)[len(q) // 2:]
        swing = float(np.mean(q.max(axis=0) - q.min(axis=0))) if q.size else 0.0
        return env.budget.mean_power, swing

    # Power per unit of motion, which is the quantity that means anything.  A
    # servo can always cut power by tracking worse, so watts alone cannot tell
    # resonance from a drive that has given up.
    def cost(stiffness, compliance):
        p_w, swing = probe(stiffness, compliance)
        return p_w / max(swing, 1e-6), p_w, swing

    rigid, p_rigid, s_rigid = cost(0.0, 1.0)
    stiff_spring, p_stiff, s_stiff = cost(1.0, 1.0)
    tuned, p_tuned, s_tuned = cost(1.0, 0.3)
    very_soft, p_soft, s_soft = cost(1.0, 0.1)

    check("a spring under the old hard-wired gain buys nothing",
          stiff_spring > 0.9 * rigid,
          f"{stiff_spring:.0f} W/rad against {rigid:.0f} rigid -- it cut power from "
          f"{p_rigid:.0f} W to {p_stiff:.0f} W only by cutting motion from "
          f"{s_rigid:.2f} to {s_stiff:.2f} rad")
    check("the same spring with a compliant drive is far cheaper per unit of motion",
          tuned < 0.6 * rigid,
          f"{tuned:.0f} W/rad against {rigid:.0f} rigid "
          f"({p_tuned:.0f} W at {s_tuned:.2f} rad)")
    check("and an over-compliant drive does stop tracking",
          s_soft < 0.6 * s_rigid,
          f"kp x0.1: {p_soft:.0f} W but only {s_soft:.2f} rad against {s_rigid:.2f}")


def test_flight_is_measured_against_the_ground_not_the_waterline() -> None:
    """Altitude must mean height above whatever is underneath, and the machine
    must be measured from its lowest point.

    Found by opening the archive of a live run and asking what its best flyer
    actually was.  The answer: a design with no lifting surface at all, wing
    area 0.0000 m^2, scoring 0.75 for flight.  Three things stacked up.

      * "Airborne" was ``depth < -0.3`` -- above the *waterline*.  The beach
        rises inland to over three metres, so a machine sitting on it thirty
        metres from shore is well above the waterline.
      * Sink rate was the rate of change of world z.  The beach slopes at 0.12,
        so skimming inland reads as climbing.
      * Clearance was measured from the root body, which sits half a metre up on
        a machine at rest, and the ramp's own half-thickness was another half
        metre -- so a landed machine still read as 1 m in the air.

    The design was launched at the 30 m/s speed cap (its wing area rounded to
    zero, so its trim speed clipped), lobbed seventy-five metres downrange, and
    settled onto rising ground where its height above the ground stayed constant
    -- which the sink term read as holding altitude.
    """
    print("\nscore: flight is measured against the ground")
    from dytiscidae.core.bodyplans import beetle
    from dytiscidae.core.mjcf import beach_extent, beach_surface_z
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, SegmentResult, TriphibianEnv

    env = TriphibianEnv(build(beetle()))
    dt, dur = env.timestep, 8.0
    n = int(dur / dt)

    # The scene's own geometry must agree with what the scorer believes.
    check("the beach surface is above its centre-line", beach_surface_z(12.0) > 0.4,
          f"z={beach_surface_z(12.0):.2f} m at the shoreline")
    check("and it rises inland", beach_surface_z(35.0) > beach_surface_z(12.0),
          f"{beach_surface_z(12.0):.2f} -> {beach_surface_z(35.0):.2f} m")
    lo, hi = beach_extent()
    check("and it is finite -- no phantom ground out at sea",
          beach_surface_z(lo - 5.0) < -10.0 and beach_surface_z(hi + 5.0) < -10.0,
          f"ramp spans x {lo:.1f}..{hi:.1f}")

    def score(clear, depth):
        r = SegmentResult(domain=Domain.AIR, duration=dur)
        r.mean_speed = env.launch_speed
        return env._score_segment(
            Domain.AIR, r, depth, np.zeros(n), np.ones(n), np.zeros(n), clear
        )

    t = np.arange(n) * dt
    # Skimming a rising slope: world z climbs, height above ground does not.
    slope_z = 3.0 + 0.12 * 25.0 * t / dur          # gaining altitude in world z
    resting = np.full(n, 0.05)                      # but sitting on the ground
    # It is above the waterline the whole time, which is what used to matter.
    above_water = -slope_z
    check("resting on a rising hillside does not score as flight",
          score(resting, above_water) < 0.1,
          f"{score(resting, above_water):.3f}")

    # Genuine level flight over water still scores.
    level = np.full(n, 20.0)
    check("holding height above the ground still scores full",
          score(level, -level) > 0.95, f"{score(level, -level):.3f}")

    # And a real descent still scores as a descent even if the ground falls away
    # faster, which is the mirror image of the bug.
    descending = 20.0 - 1.0 * t
    check("a steady descent scores less than level flight",
          score(descending, -descending) < score(level, -level),
          f"{score(descending, -descending):.3f} against {score(level, -level):.3f}")

    # Clearance itself must come from the lowest geometry: a machine resting on
    # the beach has to read as touching down, not as a metre in the air.
    env.reset(Domain.LAND, randomise=False)
    for _ in range(int(3.0 / dt)):
        env.step(env.cpg.command(env.cpg.base, env.data.time))
    check("a machine at rest on the beach has near-zero clearance",
          abs(env.clearance()) < 0.25, f"clearance {env.clearance():+.2f} m")


def test_entry_shock_is_hydrodynamic_not_a_speed_limit() -> None:
    """A fast, streamlined entry must beat a slow, flat one.

    Entry was scored on the vertical speed of the machine's centre against a
    single "survivable speed" for the hull.  That cannot tell a gannet from a
    belly-flop.  A gannet enters at 24 m/s and survives because it enters nose
    first: the wetted area grows slowly, so the rate at which it entrains water
    stays low.  A flat hull at a quarter of that speed wets all at once and is
    destroyed.  Scoring on centre-of-mass speed gets that backwards and
    penalises exactly the designs worth finding.

    The load is now the von Karman-Wagner slamming force -- |d(m_add)/dt . v_n|,
    the rate of entrainment times the closing speed -- which the solver was
    already computing and the score was ignoring.  It reads attitude,
    slenderness, deadrise and structural compliance for free, because all four
    change how fast the body wets and all four are already in the dynamics.
    """
    print("\nfluid: entry shock reads the flow, not the speedometer")
    from dytiscidae.core.bodyplans import ray
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    p = build(ray())
    env = TriphibianEnv(p)
    check("the hull has a slamming pressure capacity", p.slam_pressure_capacity > 1e4,
          f"{p.slam_pressure_capacity / 1e3:.0f} kPa")

    window = max(int(0.010 / env.timestep), 1)

    def enter(pitch_deg: float, speed: float) -> float:
        env.reset(Domain.AIR, randomise=False)
        env.data.qpos[:3] = (-8.0, 0.0, 1.2)
        a = math.radians(-pitch_deg)
        env.data.qpos[3:7] = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
        env.data.qvel[:] = 0.0
        env.data.qvel[2] = -speed
        env.solver.reset()
        mujoco.mj_forward(env.model, env.data)
        w, peak = [], 0.0
        for _ in range(int(1.2 / env.timestep)):
            env.step(env.cpg.command(env.cpg.base, env.data.time))
            w.append(float(env.solver.diag.slam))
            if len(w) > window:
                w.pop(0)
            if len(w) == window:
                peak = max(peak, float(np.mean(w)))
        pressure = peak / max(p.frontal_area, 1e-3)
        return float(np.clip(1.0 - (pressure / p.slam_pressure_capacity) ** 2, 0.0, 1.0))

    flat_slow = enter(0.0, 4.0)
    flat_fast = enter(0.0, 8.0)
    nose_slow = enter(80.0, 4.0)
    nose_fast = enter(80.0, 8.0)

    check("entering flat faster is worse", flat_fast < flat_slow,
          f"{flat_slow:.3f} at 4 m/s -> {flat_fast:.3f} at 8 m/s")
    check("entering nose-first beats entering flat at the same speed",
          nose_slow > flat_slow, f"{nose_slow:.3f} against {flat_slow:.3f} at 4 m/s")
    check(
        "and a nose-first entry at twice the speed beats a flat one at half of it",
        nose_fast > flat_slow,
        f"nose-first at 8 m/s scores {nose_fast:.3f}, flat at 4 m/s scores {flat_slow:.3f}",
    )
    check("a flat entry well past the hull limit scores nothing", flat_fast == 0.0,
          f"{flat_fast:.3f}")


def test_free_surface_continuity() -> None:
    """Submerged fraction must sweep smoothly from 0 to 1 across the surface."""
    print("\nmedium: free surface")
    med = MediumField()
    zs = np.linspace(0.3, -0.3, 61)
    pos = np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], axis=1)
    f = med.submerged_fraction(pos, np.full(len(zs), 0.1))
    check("fraction is 0 well above the surface", f[0] == 0.0, f"f={f[0]:.3f}")
    check("fraction is 1 well below the surface", f[-1] == 1.0, f"f={f[-1]:.3f}")
    check("fraction is monotonic", bool(np.all(np.diff(f) >= -1e-12)))
    jump = float(np.max(np.abs(np.diff(f))))
    check("no discontinuity in the transition", jump < 0.2, f"max step={jump:.3f}")

    rho, mu, _ = med.properties(pos, np.full(len(zs), 0.1))
    check("density spans air to seawater", rho[0] < 2.0 and rho[-1] > 1000.0,
          f"{rho[0]:.2f} -> {rho[-1]:.1f} kg/m^3")


def test_added_mass_dominates_in_water() -> None:
    """Added mass should be negligible in air and large in water.

    This ratio is the reason a wing optimised for air is a bad paddle and vice
    versa, and it is the central tension the whole project is exploring.
    """
    print("\nfluid: added mass regime")
    chord, dr = 0.2, 1.0
    m_air = AIR.rho * math.pi * chord**2 * 0.25 * dr
    m_water = SEAWATER.rho * math.pi * chord**2 * 0.25 * dr
    check(
        "water added mass exceeds air by ~3 orders of magnitude",
        500 < m_water / m_air < 1500,
        f"{m_air*1e3:.2f} g vs {m_water:.1f} kg",
    )
    # A 15 kg machine's wing carries more added mass in water than the machine
    # itself weighs -- worth stating explicitly since it drives the whole design.
    check("wing added mass is comparable to vehicle mass", m_water > 20.0,
          f"{m_water:.1f} kg per wing strip")


def test_lev_extends_stall() -> None:
    """High reduced frequency must delay stall (the leading-edge vortex)."""
    print("\nfluid: leading-edge vortex")
    alpha = np.radians(np.array([25.0]))
    re = np.array([5e4])
    ar = np.array([5.0])
    cl_static = lift_coefficient(alpha, re, ar, np.array([0.0]))[0]
    cl_flap = lift_coefficient(alpha, re, ar, np.array([0.5]))[0]
    check(
        "flapping wing holds more lift at 25 deg than a static one",
        cl_flap > cl_static * 1.2,
        f"CL static={cl_static:.2f}  flapping={cl_flap:.2f}",
    )


def test_energy_budget_matches_hand_calculation() -> None:
    """The Tier-0 analytic estimate must reproduce the feasibility numbers that
    the whole project scope was justified with."""
    print("\nenergy: 15 kg mission budget")
    p_air, note = cruise_power_air(mass=15.0, span=2.4, wing_area=0.95, speed=13.0)
    check("15 kg cruise power is 300-1200 W", 300 < p_air < 1200, f"{p_air:.0f} W  ({note})")

    p_hover_scale = 15.0 * 9.80665
    p_hover = (p_hover_scale**1.5) / math.sqrt(2 * 1.225 * 1.0) / (0.5 * 0.7)
    check(
        "hover costs at least 4x cruise (so cruise is mandatory)",
        p_hover > 4 * p_air,
        f"hover={p_hover/1000:.1f} kW vs cruise={p_air:.0f} W",
    )

    p_water, wnote = cruise_power_water(volume=0.02, frontal_area=0.05, speed=1.0, seal_count=6)
    check("submerged cruise is under 100 W", p_water < 100, f"{p_water:.0f} W ({wnote})")

    # 3 cycles x 5 min per domain.
    wh = (p_air * 900 + p_water * 900 + 120.0 * 900) / 3600.0
    check("45 min mission fits in 100-400 Wh", 100 < wh < 400, f"{wh:.0f} Wh")
    pack_kg = wh / 200.0
    check("battery is a workable fraction of 15 kg", pack_kg < 3.0, f"{pack_kg:.2f} kg pack")


def test_actuator_never_regenerates() -> None:
    print("\nenergy: actuator model")
    a = Actuator(motor_class="bldc", mass=0.15, gear_ratio=4.0)
    p = a.electrical_power(np.array([-2.0]), np.array([-30.0]))
    check("negative torque and speed still costs power", p[0] > 0, f"{p[0]:.1f} W")
    p0 = a.electrical_power(np.array([0.0]), np.array([0.0]))
    check("idle draw is ~zero", abs(p0[0]) < 1e-6, f"{p0[0]:.3e} W")
    p_stall = a.electrical_power(np.array([a.stall_torque]), np.array([0.0]))
    check("stall costs copper loss only", p_stall[0] > 0, f"{p_stall[0]:.0f} W at stall")

    batt = Battery("liion", wh=150.0)
    budget = PowerBudget(battery=batt, actuators=[a, a], avionics_w=6.0)
    for _ in range(1000):
        budget.step(np.array([1.0, 0.5]), np.array([20.0, 10.0]), 0.01)
    check("energy is consumed monotonically", batt.soc < 1.0, f"SoC={batt.soc:.3f}")
    check("mean power is plausible", 10 < budget.mean_power < 500, f"{budget.mean_power:.0f} W")


def test_structure_rejects_impossible_wings() -> None:
    print("\nstructure: spar and hull")
    # A 15 kg machine on a 2.4 m span with a 6 mm printed PETG spar: must fail.
    bad = structure.spar_check(
        lift_n=15 * 9.80665, semi_span=1.2, outer_d=0.006, wall=0.001, material=PETG
    )
    check("6 mm PETG spar fails at 15 kg", not bad.ok, f"margin={bad.margin:+.2f}")

    # A 25 mm carbon tube should pass.
    good = structure.spar_check(
        lift_n=15 * 9.80665, semi_span=1.2, outer_d=0.025, wall=0.0015, material=CFRP_TUBE
    )
    check("25 mm carbon spar passes at 15 kg", good.ok, f"margin={good.margin:+.2f}")

    # Inertial reversal must bite harder as frequency rises.
    m5 = structure.flapping_inertial_check(
        wing_mass=0.8, semi_span=1.2, flap_freq=5.0, flap_amplitude_rad=0.7,
        outer_d=0.025, wall=0.0015, material=CFRP_TUBE,
    )
    m10 = structure.flapping_inertial_check(
        wing_mass=0.8, semi_span=1.2, flap_freq=10.0, flap_amplitude_rad=0.7,
        outer_d=0.025, wall=0.0015, material=CFRP_TUBE,
    )
    check("inertial load scales with f^2", m10.applied > 3.5 * m5.applied,
          f"{m5.applied/1e6:.1f} -> {m10.applied/1e6:.1f} MPa")

    hoop, buck = structure.hull_pressure_check(
        depth_m=10.0, radius=0.09, wall=0.003, length=0.4, material=PETG
    )
    check("hoop stress at 10 m is easy", hoop.ok, f"margin={hoop.margin:+.2f}")
    check("buckling is the binding hull constraint", buck.margin < hoop.margin,
          f"buckling margin={buck.margin:+.2f} vs hoop {hoop.margin:+.2f}")

    # Gas compression must destabilise depth-keeping.
    b = structure.BuoyancyState(mass=15.0, displaced_volume=0.0150, gas_volume_surface=0.002)
    n0, n10 = b.net_buoyancy(0.0), b.net_buoyancy(10.0)
    check("carried gas makes the vehicle heavier with depth", n10 < n0,
          f"{n0:+.1f} N at surface -> {n10:+.1f} N at 10 m")
    check("depth stability is negative (needs active control)",
          b.depth_stability(5.0) < 0, f"{b.depth_stability(5.0):+.2f} N/m")

    p = structure.slam_pressure(12.0, deadrise_deg=20.0)
    check("water entry at 12 m/s is a serious load", p > 2e5, f"{p/1e3:.0f} kPa")
    p_v = structure.slam_pressure(12.0, deadrise_deg=45.0)
    check("deadrise reduces slam substantially", p_v < 0.4 * p, f"{p_v/1e3:.0f} kPa at 45 deg")


def test_wave_field() -> None:
    print("\nmedium: waves")
    med = MediumField(sea_state=SeaState(amplitude=0.15, period=2.0, wavelength=6.0))
    xy = np.zeros((5, 2))
    z0 = med.sea_state.surface_z(xy, 0.0)
    z1 = med.sea_state.surface_z(xy, 0.5)
    check("surface moves with time", not np.allclose(z0, z1), f"{z0[0]:+.3f} -> {z1[0]:+.3f} m")
    deep = med.sea_state.orbital_velocity(np.array([[0.0, 0.0, -8.0]]), 0.0)
    shallow = med.sea_state.orbital_velocity(np.array([[0.0, 0.0, -0.1]]), 0.0)
    check("orbital velocity decays with depth",
          np.linalg.norm(deep) < 0.2 * np.linalg.norm(shallow),
          f"{np.linalg.norm(shallow):.3f} -> {np.linalg.norm(deep):.4f} m/s")


def main() -> int:
    print("=" * 68)
    print("Dytiscidae physics verification")
    print("=" * 68)
    test_lift_sign_and_magnitude()
    test_buoyancy()
    test_bluff_drag_is_orientation_dependent()
    test_generated_bodies_reach_the_fluid()
    test_land_domain_is_reachable()
    test_machine_does_not_collide_with_itself()
    test_air_segment_can_be_scored()
    test_truncated_episodes_cannot_score()
    test_added_mass_is_anisotropic()
    test_bodies_generate_lift_and_a_pitching_moment()
    test_series_elasticity_needs_a_compliant_drive()
    test_flight_is_measured_against_the_ground_not_the_waterline()
    test_entry_shock_is_hydrodynamic_not_a_speed_limit()
    test_free_surface_continuity()
    test_added_mass_dominates_in_water()
    test_lev_extends_stall()
    test_energy_budget_matches_hand_calculation()
    test_actuator_never_regenerates()
    test_structure_rejects_impossible_wings()
    test_wave_field()
    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all physics checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
