"""Physics layer verification.

These are not smoke tests.  Each one pins down a sign convention or a magnitude
that the rest of the pipeline silently depends on, because a sign error in the
fluid model does not crash -- it just produces a machine that "flies" by
falling, and the optimiser will happily exploit it for a thousand generations
before anyone notices.

Run with:  python -m pytest tests/ -q      (or)  python tests/test_physics.py
"""

from __future__ import annotations

import copy
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
SKIPPED: list[str] = []

#: Errors that mean "this machine lacks the hardware", not "the code is wrong".
#: Explicit rather than a bare ``except``: a new kind of environmental breakage
#: should be reported as a failure, not quietly absorbed into the skip count.
BLOCKED_MARKERS = (
    "GPU fluid extension not importable",
    "No module named 'full_pipeline'",
    # The extension imports and every construction fails -- a driver whose
    # kernel module does not match its userspace library.  Measured 2026-09-19:
    # NVRM 580.173.02 loaded under libnvidia-ml.so.580.178.04 raised
    # "Failed to initialize NVML: 18" from FullPipeline() while the import
    # succeeded, so this arrived as a traceback where the contract is a skip.
    "Failed to initialize NVML",
    "cannot be constructed",
)


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def run_all(functions) -> None:
    """Run every test, and never let one of them stop the rest.

    ``main()`` used to be a flat list of calls, so the first function that
    raised ended the file and everything after it silently never ran -- the
    console showed a traceback, not a list of what had been lost.  On a machine
    without the Mojo GPU fluid extension that is exactly what happens, and
    ``tools/suite_probe.py`` measured the cost: **36 of this file's 37 test
    functions pass here and precisely one is blocked**, yet the abort took four
    more down with it.

    A blocked function is reported as ``[skip]`` and counted separately, because
    "did not run" and "passed" must not share a line in the summary.
    """
    import traceback

    for fn in functions:
        mark = len(FAILURES)
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
            text = f"{type(exc).__name__}: {exc}"
            if any(m in text for m in BLOCKED_MARKERS):
                # The function stopped partway.  Whatever it printed before it
                # stopped is not evidence about the code -- several of these
                # check a run that never got to evaluate anything -- so its
                # checks are withdrawn rather than counted as failures.  They
                # are not counted as passes either: the function is skipped.
                partial = len(FAILURES) - mark
                del FAILURES[mark:]
                SKIPPED.append(fn.__name__)
                note = f", {partial} partial checks withdrawn" if partial else ""
                print(f"  [skip] {fn.__name__}{note}  "
                      f"-- {text.splitlines()[0][:100]}")
            else:
                FAILURES.append(f"{fn.__name__} raised")
                print(f"  [FAIL] {fn.__name__} raised")
                traceback.print_exc()


def report(label: str) -> int:
    """The summary.  A skip is never folded into the success line."""
    print("\n" + "=" * 68)
    if SKIPPED:
        print(f"{len(SKIPPED)} SKIPPED — needs the Mojo GPU fluid extension "
              f"(`cd mojo && pixi run build-all`), not a defect: "
              f"{', '.join(SKIPPED)}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    # Unqualified only when nothing was skipped, so the string a reader greps
    # for cannot appear on a run that did not run everything.
    print(label if not SKIPPED else f"{label}, {len(SKIPPED)} skipped")
    return 0


# --------------------------------------------------------------------------



def on_task_xy(env, domain, n: int, dur: float = 8.0) -> np.ndarray:
    """A horizontal trace that does exactly what the default task asks.

    Since 2026-09-21 every medium is scored on following a commanded phase
    script (``dytiscidae.envs.tasks``), and horizontal motion comes from the
    recorded positions.  A test that fabricates altitude or depth traces to
    probe a *gate* has to supply the motion too, or the task scores zero and
    the gate under test is never reached.  ``n`` samples of a segment asked to
    last ``dur`` seconds, so a truncated trace stops where the episode did.
    """
    from dytiscidae.envs.tasks import schedule_for

    t = schedule_for(domain, trim_speed=float(env.launch_speed))
    n_want = max(int(round(dur / env.timestep)), 1)
    xy = np.zeros((n, 2))
    for k, ((lo, hi), ph) in enumerate(zip(t.bounds(n_want), t.phases)):
        v = (ph.speed * np.array([np.cos(ph.heading), np.sin(ph.heading)])
             if ph.moving else np.zeros(2))
        for i in range(lo, min(hi, n)):
            xy[i] = (xy[i - 1] if i else 0.0) + v * env.timestep
    return xy

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

    check("every plan carries a generated body",
          n_plans_with_fields == len(BODY_PLANS),
          f"{n_plans_with_fields}/{len(BODY_PLANS)}")
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


def test_a_bilateral_pair_flaps_together_rather_than_rolling() -> None:
    """One command to a mirrored joint pair must move both sides the same way.

    A rotation axis is a pseudovector: reflect a hinge and the same commanded
    angle turns the mirror image the *other* way.  So a bilateral pair needs
    mirrored axes, or a symmetric command is a roll input.

    This one is a regression from fixing the surface reflection.  The old broken
    frame happened to give the right hinge behaviour and the wrong aerodynamics;
    correcting the aerodynamics swapped which of the two was wrong, and the two
    halves of a wing pair holding station settled at 0.239 and 0.172 rad under
    identical load.  Both bugs are the same mistake -- treating a mirror as
    though it were a rotation -- which is why fixing one exposed the other.
    """
    print("\nmjcf: a wing pair flaps, it does not roll")
    from dytiscidae.core.bodyplans import gannet
    from dytiscidae.core.mjcf import compile_phenotype
    from dytiscidae.core.phenotype import build

    p = build(gannet())
    m, d, acts, _panels = compile_phenotype(p)
    by_part: dict[int, list[tuple[bool, np.ndarray]]] = {}
    for name in acts:
        seg = next((s for s in p.segments if s.name == name[:-2]), None)
        if seg is None:
            continue
        jid = m.jnt_bodyid.tolist().index(m.body(seg.name).id) \
            if m.body(seg.name).id in m.jnt_bodyid.tolist() else None
        if jid is None:
            continue
        by_part.setdefault(seg.part_index, []).append((seg.mirrored, m.jnt_axis[jid]))

    pairs = [(k, v) for k, v in by_part.items() if len(v) == 2 and {a for a, _ in v} == {True, False}]
    check("the plan has mirrored joint pairs to check", len(pairs) >= 2,
          f"{len(pairs)} bilateral pairs")

    M = np.diag([1.0, -1.0, 1.0])
    ok = True
    detail = []
    for _pi, v in pairs:
        a = next(ax for mir, ax in v if not mir)
        b = next(ax for mir, ax in v if mir)
        # Local axes must be diag(-1,-1,1) of each other, which is what makes
        # the *world* axes come out as -M(world axis of the original).
        ok &= bool(np.allclose(a * np.array([-1.0, -1.0, 1.0]), b, atol=1e-9))
        detail.append(f"{np.round(a,2)}/{np.round(b,2)}")
    check("and their hinge axes are mirrored, not copied", ok, "; ".join(detail))
    del M

    # And it shows up in the dynamics: hold station and the two sides must sit
    # at the same angle.
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    env = TriphibianEnv(p)
    env.reset(Domain.AIR, randomise=False)
    for _ in range(int(1.0 / env.timestep)):
        env.step(env.cpg.command(env.cpg.base, env.data.time))
    q = env.data.qpos[7:]
    worst = 0.0
    for _pi, v in pairs:
        names = [s.name for s in p.segments if s.part_index == _pi]
        idx = [i for i, n in enumerate(acts) if n[:-2] in names]
        if len(idx) == 2 and max(idx) < len(q):
            worst = max(worst, abs(float(q[idx[0]] - q[idx[1]])))
    check("so a symmetric command holds both sides at the same angle",
          worst < 0.02, f"worst left/right difference {worst:.4f} rad")


def test_each_plan_moves_the_way_it_says_it_does() -> None:
    """A hinge axis is written in the part's own frame, and the part's frame is
    not the world's.  Every plan's headline mechanism has to survive that.

    ``joint_axis`` is expressed in the child's frame, whose +X is the limb's own
    span or length.  So ``[1,0,0]`` on a wing is a *feathering* hinge -- it
    changes incidence and never strokes -- and ``[0,1,0]`` is the stroke.  This
    is easy to get backwards and nothing caught it: the beetle, documented here
    and in its own docstring as a bilateral flapper, had a feathering hinge and
    could not flap.  Giving it the stroke axis moved air 0.118 -> 0.166 and
    water 0.467 -> 0.658.  The gannet's tailplane had the same mistake in
    reverse: a fold axis where an elevator belongs, so the whole range of the
    surface moved glide ratio by 0.02 rather than from 0.26 to 4.91.

    Checked as a world-frame direction rather than by matching the literal, so
    it keeps meaning something if the placement changes.
    """
    print("\nbodyplans: each plan moves the way it says it does")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build

    def world_axis(p, part_index: int):
        for s in p.segments:
            if s.part_index == part_index and not s.mirrored:
                a = np.asarray(s.part.joint_axis, float)
                a = a / max(float(np.linalg.norm(a)), 1e-9)
                return s.rotation @ a
        return None

    # part index, which world axis the mechanism needs, and what to call it.
    # 0 = fore-aft (stroke/fold), 1 = lateral (incidence, since span is lateral),
    # 2 = vertical (sweep).
    wanted = {
        "beetle": (1, 0, "the flapper's shoulder strokes"),
        "bat": (1, 2, "the bat's digit sweeps to change area"),
        "gannet": (1, 0, "the gannet's shoulder folds"),
    }
    for name, (pi, axis, why) in wanted.items():
        p = build(BODY_PLANS[name]())
        w = world_axis(p, pi)
        got = int(np.argmax(np.abs(w))) if w is not None else -1
        check(why, got == axis,
              f"world axis {np.round(w, 2)} -> "
              f"{['fore-aft', 'lateral', 'vertical'][got] if got >= 0 else 'missing'}")

    p = build(BODY_PLANS["gannet"]())
    w = world_axis(p, 2)
    check("and its tailplane is an elevator, not a second fold",
          w is not None and int(np.argmax(np.abs(w))) == 1,
          f"world axis {np.round(w, 2)} -- along the span, so the hinge is incidence")


def test_a_surface_can_be_told_to_hold_still() -> None:
    """``stroke_amplitude`` and ``neutral`` must reach the pattern generator.

    The generator beat every joint at 0.45 of half-travel about the midpoint of
    its range, and neither number was a gene.  A machine therefore could not
    hold a surface still: the only way to stop a wing being shaken was to leave
    it unactuated, and an unactuated wing cannot fold for a dive or take weight
    on land.  Every fixed-wing and every folding-wing configuration was outside
    the search space -- not disfavoured, unreachable -- which is the same defect
    the phase gene had, and it is why nothing in this project glided.
    """
    print("\ncontrol: a wing can be told to hold still")
    from dytiscidae.core.bodyplans import gannet
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    p = build(gannet())
    env = TriphibianEnv(p)
    wings = [i for i, n in enumerate(env.act_names)
             if any(s.name == n[:-2] and s.kind == "wing" for s in p.segments)]
    legs = [i for i in range(len(env.act_names)) if i not in wings]
    check("the trim surfaces are given no stroke",
          len(wings) > 0 and float(np.abs(env.cpg.base.amplitude[wings]).max()) < 1e-9,
          f"{len(wings)} wing joints, max amplitude "
          f"{float(np.abs(env.cpg.base.amplitude[wings]).max()) if wings else -1:.4f} rad")
    check("while the limbs still stroke",
          len(legs) > 0 and float(np.abs(env.cpg.base.amplitude[legs]).max()) > 0.1,
          f"{len(legs)} limb joints, max amplitude "
          f"{float(np.abs(env.cpg.base.amplitude[legs]).max()) if legs else -1:.3f} rad")
    # ``neutral`` = 1.0 means "sit at the top of the travel", not the middle.
    # Only the shoulder asks for that; the tail sits mid-range, so this has to
    # select on the gene rather than on the part kind -- the tail is a WING too.
    shoulders = [i for i in wings
                 if any(s.name == env.act_names[i][:-2]
                        and getattr(s.part, "neutral", 0.5) > 0.99
                        for s in p.segments)]
    hi = env.cpg.hi[shoulders] if shoulders else np.zeros(0)
    at_top = float(np.abs(env.cpg.base.offset[shoulders] - hi).max()) if shoulders else 1.0
    check("and a folding shoulder rests extended, not half folded",
          len(shoulders) > 0 and at_top < 1e-9,
          f"{len(shoulders)} shoulder joints at "
          f"{np.round(env.cpg.base.offset[shoulders], 3)} against upper limit "
          f"{np.round(hi, 3)}")

    # Held still means held at the *commanded* angle, not at wherever it
    # started: every joint travels to its neutral in the first moments, and
    # measuring drift from the start pose counts that as motion.
    env.reset(Domain.AIR, randomise=False)
    worst = 0.0
    for _ in range(int(2.0 / env.timestep)):
        u = np.asarray(env.cpg.command(env.cpg.base, env.data.time), float)
        env.step(u)
        worst = max(worst, float(np.abs(env.data.qpos[7:][wings] - u[wings]).max()))
    check("and holds that angle against real aerodynamic load when flown",
          worst < 0.25 if wings else False,
          f"worst departure from the commanded angle {worst:.3f} rad "
          f"({math.degrees(worst):.0f} deg) over 2 s")


def test_the_seeds_include_something_that_flies() -> None:
    """At least one starting plan must be able to stay in the air.

    The module docstring in ``bodyplans`` claimed the beetle was "good in air".
    Measured on the air segment it scores 0.118, and so does everything else
    there: 0.116 to 0.129 across all five, against 1.000 for a fixed-wing
    machine of the same mass.  The search was being asked to cross from 0.12 to
    1.0 with no foothold anywhere on the far side, which is precisely the deep
    valley that file exists to bridge -- and the five plans were all flapping or
    undulating solutions, so there was no bridge to build from.

    The bar here is deliberately not 1.0.  A seed is a foothold, not an answer;
    what has to be true is that one of them stays up and the others do not, so
    that flight is something the archive can hold on to and improve.
    """
    print("\nbodyplans: at least one seed can stay in the air")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.evaluate import evaluate_tier1
    from dytiscidae.envs.triphibian import MissionSpec

    # Medians over three scatter seeds, because the claim is about the plan
    # and not about one draw of its initial conditions.  Measured 2026-09-23,
    # once the servo feed-forward made the gannet's 0.6 Hz flap happen as
    # commanded: seeds 1-4 were unchanged, seed 0 went 0.213 -> 0.022 and seed
    # 5 0.012 -> 0.001, and seed 3 falls either way.  The test was sitting on
    # seed 0 alone.
    spec = MissionSpec()
    air = {}
    for name, fn in BODY_PLANS.items():
        rows = []
        for seed in (0, 1, 2):
            r = evaluate_tier1(build(fn()), spec=spec, seed=seed, segment_seconds=8.0)
            seg = r.segments.get("air")
            rows.append((seg.competence if seg else 0.0,
                         (seg.measurements if seg else {}).get("airborne_fraction", 0.0),
                         (seg.measurements if seg else {}).get("sink_rate", 99.0)))
        air[name] = tuple(float(np.median([row[i] for row in rows])) for i in range(3))
    best = max(air, key=lambda k: air[k][0])
    score, frac, sink = air[best]
    others = sorted(v[0] for k, v in air.items() if k != best)
    check("one seed stays airborne for the whole segment",
          frac > 0.95, f"{best}: airborne {frac:.2f} of the segment")
    check("and descends slowly enough to be flying rather than falling",
          sink < 3.0, f"{best}: {sink:.2f} m/s sink against 10-14 for the flappers")
    check("it is clearly ahead of the plans that cannot",
          score > 2.0 * others[-1],
          f"{best} {score:.3f} against next best {others[-1]:.3f}")


def test_the_render_shows_the_shape_the_solver_reads() -> None:
    """Lifting surfaces must draw with their taper, twist and dihedral, and
    drawing them must not change the dynamics or cost the search anything.

    Every surface used to render as one flat box at the mean chord while the
    solver read a chord, twist, camber, thickness and dihedral *per station*.
    The picture and the physics were different objects, and the picture is the
    one a human looks at -- which is how the reflected half of every symmetric
    wing pair kept its leading edge at the back for as long as it did.  Both
    halves drew as identical flat plates, so there was nothing to see.

    The strips carry no mass and no collision, and they are off unless asked
    for: MuJoCo still places every geom each step, and a ray goes from 24 geoms
    to 114 and 37 to 46 microseconds.  Search runs without them, rendering with.
    """
    print("\nmjcf: the render shows what the solver reads")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.mjcf import compile_phenotype
    from dytiscidae.core.phenotype import build

    plain_geoms, detail_geoms, worst_mass = {}, {}, 0.0
    for name, fn in BODY_PLANS.items():
        p = build(fn())
        m0, _d0, _a0, _p0 = compile_phenotype(p)
        m1, _d1, _a1, _p1 = compile_phenotype(p, detail=True)
        plain_geoms[name], detail_geoms[name] = m0.ngeom, m1.ngeom
        worst_mass = max(worst_mass, abs(float(m0.body_mass.sum())
                                         - float(m1.body_mass.sum())))
        if name == "beetle":
            # The extra geoms must all be inert: no mass, no contacts.
            extra = [i for i in range(m1.ngeom)
                     if m1.geom_contype[i] == 0 and m1.geom_conaffinity[i] == 0]
            collide0 = sum(1 for i in range(m0.ngeom)
                           if m0.geom_contype[i] or m0.geom_conaffinity[i])
            collide1 = sum(1 for i in range(m1.ngeom)
                           if m1.geom_contype[i] or m1.geom_conaffinity[i])
            beetle_quats = m1.geom_quat[extra]

    check("detail adds geometry and plain does not",
          all(detail_geoms[k] > plain_geoms[k] for k in plain_geoms),
          ", ".join(f"{k} {plain_geoms[k]}->{detail_geoms[k]}" for k in plain_geoms))
    check("the extra geometry never collides", collide1 == collide0,
          f"{collide0} colliding geoms either way")
    check("and never carries mass", worst_mass < 1e-9,
          f"worst mass difference {worst_mass:.3e} kg")
    # A washed-out wing must show its washout: at least one strip rotated away
    # from identity about the span axis.
    off_axis = float(np.abs(beetle_quats[:, 1]).max())
    check("a twisted wing draws twisted",
          off_axis > 0.01,
          f"largest strip rotation about the span axis: {2*math.degrees(math.asin(off_axis)):.1f} deg")


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
    #
    # Held *within* an airframe, not across them.  Sorting all the plans by W/S
    # and requiring the measured launch speeds to rise with it assumes every
    # design reaches the same lift coefficient, and ``launch_speed`` exists
    # precisely because they do not -- its docstring is about how effective CL
    # sits near 0.35 and varies with whatever dihedral and twist the CPPN gave
    # the surface.  Two plans can therefore share a wing loading and trim tens
    # of percent apart: gannet at 85.6 N/m^2 launches at 13.3 m/s and bat at
    # 86.3 launches at 15.8.  That pair was already a near-tie ordered by luck,
    # and the cross-plan check duly broke on the seventh plan while both of its
    # own numbers were correct.
    #
    # Adding battery changes mass and leaves the wing alone, so within one
    # airframe CL is fixed and the relation is the clean physical one.
    for name in ("gannet", "ray", "beetle"):
        light = BODY_PLANS[name]()
        heavy = BODY_PLANS[name]()
        heavy.battery_wh = light.battery_wh * 2.5
        pl, ph = build(light), build(heavy)
        el, eh = TriphibianEnv(pl), TriphibianEnv(ph)
        el.reset(Domain.AIR, randomise=False)
        eh.reset(Domain.AIR, randomise=False)
        wl = pl.mass * 9.80665 / max(pl.wing_area, 1e-3)
        wh = ph.mass * 9.80665 / max(ph.wing_area, 1e-3)
        check(f"launch speed rises with wing loading ({name})",
              eh.launch_speed >= el.launch_speed - 1e-9,
              f"W/S {wl:.0f} -> {wh:.0f} N/m^2 gives "
              f"{el.launch_speed:.1f} -> {eh.launch_speed:.1f} m/s")


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
            xys=on_task_xy(env, Domain.AIR, n_samples, dur),
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
            Domain.LAND, r, np.full(k, -0.1), np.full(k, 0.5), np.ones(k), np.ones(k),
            xys=on_task_xy(env, Domain.LAND, k, dur),
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


def test_a_mirrored_wing_is_a_mirror_image() -> None:
    """Both halves of a symmetric wing pair must meet the flow the same way,
    and incidence and camber must both add lift.

    Half of every symmetric wing in this project was flying backwards.  The
    reflection in ``expand`` negated the roll and the azimuth, which is a 180
    degree *roll* of the attachment rather than a mirror in the fore-aft plane.
    Two consequences, both measured on a hand-built glider at 15 m/s:

      * the mirrored surface's chord axis pointed forward -- leading edge at the
        back -- so at identical geometric incidence the two halves reported
        -2.81 and +17.58 degrees of alpha;
      * camber therefore helped one side and hurt the other and cancelled, so
        the camber gene, which the CPPN had been generating all along, moved
        total lift by nothing.

    A mirror is not a rotation.  The frame a reflected limb needs is
    ``M R diag(1,1,-1)`` with ``M = diag(1,-1,1)``, which is ``_look_at(M d,
    pi - roll)`` with the surface's twist and camber negated to follow its
    flipped normal.  Then the two sides are true mirror images: equal and
    opposite alpha, lift in the same direction, camber adding on both.
    """
    print("\nphenotype: bilateral symmetry")
    import math

    from dytiscidae.core.bodyplans import BIAS, CAMBER, CHORD, THICK, TWIST, _cppn, _fusiform
    from dytiscidae.core.genome import HULL, WING as WING_KIND, Edge, Genome, Part
    from dytiscidae.core.mjcf import compile_phenotype
    from dytiscidae.core.phenotype import build
    from dytiscidae.physics.medium import GRAVITY, MediumField

    def glider(twist: float, camber: float) -> Genome:
        g = Genome()
        g.cppns = [_cppn({(BIAS, CHORD): 0.30, (BIAS, TWIST): twist,
                          (BIAS, CAMBER): camber, (BIAS, THICK): -0.4})]
        g.body_cppns = [_fusiform(taper=0.8, flatten=1.4)]
        g.parts = [
            Part(kind=HULL, length=0.7, radius=0.08, material="petg", joint="none",
                 actuated=False, sealed=True, dry_fraction=0.9, body_cppn=0),
            Part(kind=WING_KIND, span=1.4, root_chord=0.40, radius=0.014,
                 material="cfrp", surface_cppn=0, joint="none", actuated=False,
                 sealed=True, dry_fraction=0.3),
        ]
        g.edges = [Edge(parent=0, child=1, pos_u=0.45, reflect=True)]
        g.battery_wh = 200.0
        return g

    def probe(twist: float, camber: float, v: float = 15.0):
        p = build(glider(twist, camber))
        m, d, _acts, panels = compile_phenotype(p)
        solver = FluidSolver(m, panels, MediumField())
        solver.record_state = True
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = (0.0, 0.0, 60.0)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[:] = 0.0
        d.qvel[0] = v
        solver.reset()
        mujoco.mj_forward(m, d)
        d.xfrc_applied[:] = 0.0
        solver.apply(d, 0.0)
        fz = float(d.xfrc_applied[:, 2].sum()) - solver.diag.added_mass * GRAVITY
        alpha = solver.last_state["alpha"]
        by_seg = {}
        for s in p.segments:
            if s.is_surface:
                sel = panels.body_id == m.body(s.name).id
                if sel.any():
                    by_seg[s.mirrored] = float(np.degrees(alpha[sel]).mean())
        solver.reset()
        return fz / (p.mass * GRAVITY), by_seg

    lw, alphas = probe(0.3, 0.9)
    left, right = alphas.get(False, 0.0), alphas.get(True, 0.0)
    check("the two halves meet the flow at the same incidence",
          abs(abs(left) - abs(right)) < 0.05 and left * right < 0.0,
          f"{left:+.2f} deg and {right:+.2f} deg -- mirrored normals, so opposite signs")

    up = probe(0.3, 0.0)[0]
    down = probe(-0.3, 0.0)[0]
    check("positive incidence lifts and negative incidence pushes down",
          up > 0.5 and down < -0.5, f"L/W {up:+.2f} at +twist, {down:+.2f} at -twist")

    flat = probe(0.0, 0.0)[0]
    cambered = probe(0.0, 0.9)[0]
    check("camber lifts an untwisted wing",
          cambered - flat > 0.5,
          f"L/W {flat:+.2f} uncambered -> {cambered:+.2f} cambered")
    # Measured against camber's own standalone lift rather than against a fixed
    # increment, because the increment is a statement about the lift model as
    # much as about the geometry: it was 0.44 under the logistic stall blend
    # and is 0.33 under the compactly supported one (F-05), on an unchanged
    # wing.  The fraction is well below 1 for a physical reason -- the twisted
    # wing already sits near stall at 17.6 degrees, where camber's marginal
    # contribution saturates -- so what separates "adds" from "cancels" is that
    # it stays a substantial part of that standalone lift, not that it matches
    # it.
    camber_alone = cambered - flat
    check("camber adds to incidence rather than cancelling across the pair",
          lw - up > 0.25 * camber_alone,
          f"L/W {up:+.2f} twist alone -> {lw:+.2f} with camber, so camber adds "
          f"{lw - up:+.2f} of the {camber_alone:+.2f} it lifts on its own "
          f"({(lw - up) / max(camber_alone, 1e-9):.0%})")

    # And the reflection itself: mirrored limbs must land on the far side of the
    # fore-aft plane at the same height, not one up and one down.
    g = glider(0.0, 0.0)
    g.edges[0] = Edge(parent=0, child=1, pos_u=0.45, azimuth=0.4, roll=0.2, reflect=True)
    rot = {s.mirrored: s.rotation for s in build(g).segments if s.is_surface}
    M = np.diag([1.0, -1.0, 1.0])
    a, b = rot[False][:, 0], rot[True][:, 0]
    check("a reflected limb is the mirror of its partner, not its 180 deg roll",
          np.allclose(M @ a, b, atol=1e-9),
          f"span axes {np.round(a, 3)} and {np.round(b, 3)}")
    chord = np.array([0.0, -1.0, 0.0])
    ca, cb = rot[False] @ chord, rot[True] @ chord
    check("and it keeps its leading edge forward",
          np.allclose(M @ ca, cb, atol=1e-9),
          f"chord axes {np.round(ca, 3)} and {np.round(cb, 3)} in world")
    del math


def test_flight_is_expressible_in_the_genome() -> None:
    """Some genome must produce a machine that glides.  Built by hand, flown
    with the controls dead.

    This is the check the mirrored-wing bug needed and did not have.  For a long
    time nothing in this project flew, and the population failing at something
    is not evidence about the population -- three times now it has been evidence
    about the test.  Distinguishing the two takes a design whose flight does not
    depend on the search finding anything: a fixed wing, a tail, no actuation,
    released in trim, integrated with zero control input.  If *that* falls, the
    representation or the physics cannot express flight and no amount of search
    will help.

    It glides at L/D of about 6, in a badly damped phugoid -- it porpoises
    between 30 m and sea level over roughly 200 m of ground.  Damping that is a
    control problem, which is what the actuators and the learning environment
    are for.  The bar here is only that the machine trades height for distance
    rather than falling: the point of the test is the difference between "cannot
    fly" and "flies badly", and it is checked with the controls dead so it stays
    a statement about the airframe.
    """
    print("\nphenotype: flight is expressible")
    from dytiscidae.core.bodyplans import BIAS, CAMBER, CHORD, THICK, TWIST, U, _cppn, _fusiform
    from dytiscidae.core.genome import HULL, WING as WING_KIND, Edge, Genome, Part
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    g = Genome()
    g.cppns = [
        # Main wing: cambered, mild washout.
        _cppn({(BIAS, CHORD): 0.2, (U, CHORD): -0.5, (BIAS, TWIST): 0.20,
               (U, TWIST): -0.17, (BIAS, CAMBER): 0.7, (BIAS, THICK): -0.4}),
        # Lifting tail on a long arm.
        _cppn({(BIAS, CHORD): -0.2, (BIAS, TWIST): 0.35, (BIAS, THICK): -0.6}),
    ]
    g.body_cppns = [_fusiform(taper=0.85, flatten=1.2)]
    g.parts = [
        Part(kind=HULL, length=1.0, radius=0.07, material="petg", joint="none",
             actuated=False, sealed=True, dry_fraction=0.9, body_cppn=0),
        Part(kind=WING_KIND, span=1.5, root_chord=0.42, radius=0.015, material="cfrp",
             surface_cppn=0, joint="none", actuated=False, sealed=True, dry_fraction=0.3),
        Part(kind=WING_KIND, span=0.60, root_chord=0.22, radius=0.010, material="cfrp",
             surface_cppn=1, joint="none", actuated=False, sealed=True, dry_fraction=0.3),
    ]
    g.edges = [Edge(parent=0, child=1, pos_u=0.44, reflect=True),
               Edge(parent=0, child=2, pos_u=1.00, reflect=True)]
    g.battery_wh = 150.0

    def glide(genome, speed=None) -> tuple[float, float, float]:
        """Release with the controls dead and report (airborne, L/D, mass)."""
        ph = build(genome)
        e = TriphibianEnv(ph)
        e.reset(Domain.AIR, randomise=False)
        if speed is not None:
            e.data.qvel[0] = speed
        dead_u = np.zeros(e.model.nu)
        x0, z0 = float(e.data.qpos[0]), float(e.data.qpos[2])
        t = 0.0
        for _ in range(int(18.0 / e.timestep)):
            e.step(dead_u)
            if e.data.qpos[2] < 2.0:
                break
            t = float(e.data.time)
        dx = float(e.data.qpos[0]) - x0
        dz = z0 - float(e.data.qpos[2])
        return t, dx / max(dz, 1e-6), ph.mass

    p = build(g)
    v_trim = TriphibianEnv(p).launch_speed
    airborne, ld, mass = glide(g)

    # The control: the same hull, released at the same speed, with the wings
    # taken off.  An absolute glide-ratio threshold is a statement about the
    # airframe's *weight* as much as its surfaces -- it moved from 5.31 to 2.93
    # when the hull wall was corrected for buckling, on an unchanged wing --
    # and this test is about whether the surfaces do anything.  Against a
    # control they measurably do, at either wall.
    bare = copy.deepcopy(g)
    bare.parts = bare.parts[:1]
    bare.edges = []
    bare_airborne, bare_ld, bare_mass = glide(bare, speed=v_trim)

    check("the hand-built glider is released at a flying speed",
          6.0 < v_trim < 20.0, f"{v_trim:.1f} m/s")
    check("its wings are what keeps it up",
          airborne > 2.0 * bare_airborne,
          f"{airborne:.1f} s airborne against {bare_airborne:.1f} s with the "
          f"wings removed ({mass:.2f} kg against {bare_mass:.2f} kg)")
    check("trading height for distance rather than falling",
          ld > 3.0 * bare_ld,
          f"glide ratio {ld:.2f} against {bare_ld:.2f} wingless -- "
          f"{ld/max(bare_ld, 1e-6):.1f}x")


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
        q, c = [], []
        for _ in range(int(secs / env.timestep)):
            u = env.cpg.command(env.cpg.base, env.data.time)
            env.step(u)
            q.append(env.data.qpos[7:].copy())
            c.append(np.asarray(u, float).copy())
        half = len(q) // 2
        q = np.array(q)[half:]
        c = np.array(c)[half:]
        if not q.size:
            return env.budget.mean_power, 0.0, 0.0
        swing = float(np.mean(q.max(axis=0) - q.min(axis=0)))
        # Tracking error against the angle actually commanded.  Amplitude alone
        # cannot say whether the drive is following: the first version of this
        # test used swing, and once the wing mirroring was fixed the softest
        # drive produced the *largest* swing -- the joints were flopping between
        # their stops, which is exactly the failure the check was meant to catch.
        k = min(q.shape[1], c.shape[1])
        err = float(np.sqrt(np.mean((q[:, :k] - c[:, :k]) ** 2))) if k else 0.0
        return env.budget.mean_power, swing, err

    # Power per unit of motion, which is the quantity that means anything.  A
    # servo can always cut power by tracking worse, so watts alone cannot tell
    # resonance from a drive that has given up.
    def cost(stiffness, compliance):
        p_w, swing, err = probe(stiffness, compliance)
        return p_w / max(swing, 1e-6), p_w, swing, err

    rigid, p_rigid, s_rigid, e_rigid = cost(0.0, 1.0)
    stiff_spring, p_stiff, s_stiff, e_stiff = cost(1.0, 1.0)
    tuned, p_tuned, s_tuned, e_tuned = cost(1.0, 0.3)
    very_soft, p_soft, s_soft, e_soft = cost(1.0, 0.05)

    # "Little", not "nothing", since the servo feeds its rate forward and since
    # the implicit damping split: 185 against 209 W/rad (2026-09-26).  Those
    # figures came from a ray the missing entrainment reaction had driven
    # 4.5 m under water after its air spawn (ROADMAP AK); flapping at the
    # surface it is 148 against 171 (2026-09-30).
    check("a spring under the old hard-wired gain buys little",
          stiff_spring > 0.8 * rigid,
          f"{stiff_spring:.0f} W/rad against {rigid:.0f} rigid -- power "
          f"{p_rigid:.0f} W -> {p_stiff:.0f} W, motion {s_rigid:.2f} -> "
          f"{s_stiff:.2f} rad")
    # 0.49x on 2026-09-26 (it read 0.79x between the feed-forward and the root
    # being taken out of the damping split).
    check("the same spring with a compliant drive is far cheaper per unit of motion",
          tuned < 0.6 * rigid,
          f"{tuned:.0f} W/rad against {rigid:.0f} rigid "
          f"({p_tuned:.0f} W at {s_tuned:.2f} rad)")
    # The cost of a soft drive shows up on a surface whose job is to *hold*, not
    # on one that is being swung: an oscillating fin tracks a sine wave about as
    # badly at every gain, which is why measuring it on the ray said nothing.  A
    # trim surface carrying real aerodynamic load is where servo stiffness is
    # the whole story, and that only became testable once a surface could be
    # told to hold still at all.
    def _gannet_with_compliance(compliance: float):
        from dytiscidae.core.bodyplans import gannet

        g = gannet()
        for part in g.parts:
            if part.joint != "none" and part.actuated:
                part.drive_compliance = compliance
        return g

    def hold_error(compliance: float, secs: float = 3.0) -> tuple:
        from dytiscidae.core.bodyplans import gannet

        g = gannet()
        for part in g.parts:
            if part.joint != "none" and part.actuated:
                part.drive_compliance = compliance
        ph = build(g)
        e = TriphibianEnv(ph)
        wings = [i for i, n in enumerate(e.act_names)
                 if any(x.name == n[:-2] and x.kind == "wing" for x in ph.segments)]
        e.reset(Domain.AIR, randomise=False)
        q, c = [], []
        for _ in range(int(secs / e.timestep)):
            u = np.asarray(e.cpg.command(e.cpg.base, e.data.time), float)
            e.step(u)
            q.append(e.data.qpos[7:][wings].copy())
            c.append(u[wings].copy())
        # Steady state, over the second half.  The worst error over the whole
        # run is not the servo at all: every joint starts at zero and is
        # commanded somewhere else, so the maximum is the initial slew and comes
        # out identical at every gain.  Measured that way this looked like a
        # gene with no cost, which it is not.
        q, c = np.array(q), np.array(c)
        h = len(q) // 2
        err = float(np.abs(q[h:].mean(axis=0) - c[h:].mean(axis=0)).max())
        return err, e.budget.mean_power

    # The cost of a soft drive shows up on a surface whose job is to *hold*, and
    # it shows up in the steady state.  The trade is monotone in both
    # directions: softening the drive costs incidence and saves power, which is
    # a real choice for the search to make rather than a free lunch.
    #
    # How much power it saves depends on how heavy the machine is, so the
    # threshold here is a floor and not a calibration.  On the gannet the
    # saving measured 46% (12.5 W -> 6.7 W) while the hull carried the
    # 8x-too-large buckling allowable, and 24% (13.4 W -> 10.2 W) once the wall
    # was corrected and the machine gained 18% of its mass: a heavier wing sags
    # further under a soft drive (5.09 deg -> 6.32 deg) and the servo pays for
    # holding it there.  The claim being tested is that the saving exists, not
    # that it has a particular size.
    hold_stiff, p_hold_stiff = hold_error(1.0)
    hold_soft, p_hold_soft = hold_error(0.05)
    check("softening the drive saves real power",
          p_hold_soft < 0.85 * p_hold_stiff,
          f"{p_hold_soft:.1f} W at kp x0.05 against {p_hold_stiff:.1f} W at "
          f"full gain -- {100*(1 - p_hold_soft/p_hold_stiff):.0f}% saved")
    check("and it is paid for in incidence the surface does not keep",
          hold_soft > 1.5 * hold_stiff,
          f"a trim wing settles {math.degrees(hold_soft):.1f} deg off its command at "
          f"kp x0.05 against {math.degrees(hold_stiff):.1f} deg at full gain")


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
            Domain.AIR, r, depth, np.zeros(n), np.ones(n), np.zeros(n), clear,
            xys=on_task_xy(env, Domain.AIR, n, dur),
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
    # Per entry: (label, sinking speed the step before first wetting, fastest
    # sinking speed after it), and the peak pressure in kPa.
    sinking: list[tuple[str, float, float]] = []
    kpa: dict[tuple, float] = {}

    def enter(pitch_deg: float, speed: float, dz: float = 0.0,
              hold: bool = False) -> float:
        env.reset(Domain.AIR, randomise=False)
        env.data.qpos[:3] = (-8.0, 0.0, 1.2 + dz)
        a = math.radians(-pitch_deg)
        env.data.qpos[3:7] = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
        env.data.qvel[:] = 0.0
        env.data.qvel[2] = -speed
        env.solver.reset()
        mujoco.mj_forward(env.model, env.data)
        w, peak = [], 0.0
        v_before, v_hit, v_after = -float(env.data.qvel[2]), None, 0.0
        for _ in range(int(1.2 / env.timestep)):
            u = env.cpg.command(env.cpg.base, env.data.time)
            env.step(np.zeros_like(u) if hold else u)
            w.append(float(env.solver.diag.slam))
            if len(w) > window:
                w.pop(0)
            if len(w) == window:
                peak = max(peak, float(np.mean(w)))
            v = -float(env.data.qvel[2])
            if v_hit is None:
                if env.solver.diag.max_submerged > 0.0:
                    v_hit = v_before
                else:
                    v_before = v
            else:
                v_after = max(v_after, v)
        sinking.append((f"{pitch_deg:.0f} deg at {speed:.1f} m/s",
                        v_hit if v_hit is not None else float("nan"), v_after))
        pressure = peak / max(p.frontal_area, 1e-3)
        kpa[(pitch_deg, speed, dz, hold)] = pressure / 1e3
        return float(np.clip(1.0 - (pressure / p.slam_pressure_capacity) ** 2, 0.0, 1.0))

    flat_slow = enter(0.0, 4.0)
    flat_fast = enter(0.0, 8.0)
    # "Well past the hull limit" is a statement about the hull, so it is solved
    # for rather than written down: the speed is raised until the score is zero.
    # It was 16 m/s while the hull carried an 8x-too-large buckling allowable
    # and a wall sized to match; correcting that doubled the wall and with it
    # the slam capacity, which is linear in wall thickness, so a fixed 16 m/s
    # stopped being past the limit at all.
    v_dead = 4.0
    while v_dead < 200.0 and enter(0.0, v_dead) > 0.0:
        v_dead *= 1.25
    flat_dead = enter(0.0, v_dead)
    nose_slow = enter(80.0, 4.0)
    nose_fast = enter(80.0, 8.0)
    # The gannet: faster than the flat entry that destroys the hull, and it
    # survives, because what breaks a hull is the rate at which it wets and not
    # how fast it is going.
    gannet = enter(80.0, 20.0)

    check("entering flat faster is worse", flat_fast < flat_slow,
          f"{flat_slow:.3f} at 4 m/s -> {flat_fast:.3f} at 8 m/s")
    # The two orderings this test used to assert -- nose-first beats flat at
    # the same 4 m/s, and nose-first at 8 m/s beats flat at 4 -- were demoted
    # to prints on 2026-09-26 (ROADMAP AK).  Two defects stood behind them.
    #
    #   * The mass matrix carried the entrained water without its reaction
    #     (`entrainment_reaction`): every step the added mass grew created
    #     ``dm v`` of momentum.  Nose-first at 8 m/s, the hull *accelerated*
    #     to 21.9 m/s downward and peaked at 1292 kPa, above the 760 kPa at
    #     20 m/s.
    #   * The implicit damping was formed on its 4-step cadence only, so first
    #     contact could meet up to three steps of water loads with a B formed
    #     in air (`ImplicitAeroDamping.due`).  A strut joint was thrown to
    #     100 rad/s and the peak depended on which step contact fell on.
    #
    # With both fixed, peak kPa over six start heights 0-7.5 cm apart, i.e.
    # over two steps of travel (`experiments/ray_entry/phase.py`):
    #
    #                    flat 4    flat 8    nose 4    nose 8    nose 20
    #   joints driven   157-198   257-354   202-286   174-257   244-293
    #   joints held     130-156   245-367    93-120   300-450   183-285
    #
    # So the orderings are not the ray's, and it is not noise: at 4 m/s flat
    # beats nose-first in all six driven entries and loses all six held, and
    # nose-first at 8 m/s loses to flat at 4 in eleven of twelve.  The ray is
    # not streamlined for entry -- its membranes and struts set the load on
    # contact, driven or not.  What is true of it, driven, in all six:
    check("at 8 m/s, nose-first loads the hull less than flat",
          nose_fast > flat_fast,
          f"{kpa[(80.0, 8.0, 0.0, False)]:.0f} kPa nose-first against "
          f"{kpa[(0.0, 8.0, 0.0, False)]:.0f} flat")
    # ...and the reason the gannet survives: a flat entry's load grows with
    # speed and a nose-first one's does not (4 -> 8 m/s: flat x1.48-2.01,
    # nose-first x0.83-1.02 over the six).
    grow_flat = kpa[(0.0, 8.0, 0.0, False)] / kpa[(0.0, 4.0, 0.0, False)]
    grow_nose = kpa[(80.0, 8.0, 0.0, False)] / kpa[(80.0, 4.0, 0.0, False)]
    check("a flat entry's load grows with speed faster than a nose-first one's",
          grow_flat > 1.3 * grow_nose,
          f"4 -> 8 m/s: flat x{grow_flat:.2f}, nose-first x{grow_nose:.2f}")
    # Which step of the damping's cycle contact lands on is an accident of the
    # start height and must not decide the load.  Four heights 4 cm apart put
    # contact on each of the four steps (~9.4 m/s at contact).  Held, because
    # that is where the stale damping bit hardest: 279-1307 kPa (4.7x) on the
    # cadence alone, 372-418 (1.12x) refreshed at the surface.
    cycle = []
    for dz in (0.0, 0.04, 0.08, 0.12):
        enter(80.0, 8.0, dz, hold=True)
        cycle.append(kpa[(80.0, 8.0, dz, True)])
    check("a held nose-first entry loads the hull the same whichever step it lands on",
          max(cycle) < 2.0 * min(cycle),
          " / ".join(f"{c:.0f}" for c in cycle) + " kPa over start heights 0-12 cm")
    # Entering water takes momentum from a machine; nothing about it can give
    # the machine more.  10% over the speed at contact is room for gravity
    # (worst 0.97 over the sixty entries of `phase.py`, driven and held; 2.38
    # without the reaction).
    worst = max(sinking, key=lambda r: r[2] / max(r[1], 1e-6))
    check("no entry leaves the machine sinking faster than it hit the water",
          all(after <= 1.1 * hit for _, hit, after in sinking),
          f"worst {worst[0]}: {worst[1]:.1f} m/s at contact, {worst[2]:.1f} after")
    check("a flat entry well past the hull limit scores nothing",
          flat_dead == 0.0 and v_dead < 200.0,
          f"{flat_dead:.3f} at {v_dead:.1f} m/s flat, against a slam capacity "
          f"of {p.slam_pressure_capacity/1e3:.0f} kPa")
    check("while a nose-first entry at a comparable speed survives it",
          gannet > 0.2,
          f"{gannet:.3f} nose-first at 20 m/s against {flat_dead:.3f} flat at "
          f"{v_dead:.1f}")


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


def test_jet_thrust_matches_momentum_flux() -> None:
    """Pulsed-jet thrust is rho Q^2/A opposite the orifice, in water only.

    The model existed unwired: a medusa passed the has_propulsor gate and then
    could not produce a newton, so a whole body-plan family was admitted to the
    search and dynamically unwinnable.  This pins the wiring: magnitude from
    momentum flux, direction opposite the expelled flow, nothing in air, refill
    charged at the reduced coefficient.
    """
    print("\njet: thrust is momentum flux, water only, refill discounted")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.mjcf import compile_phenotype
    from dytiscidae.core.phenotype import build, build_jets
    from dytiscidae.physics.jet import JetSet

    def bell_model(z: float):
        xml = f"""
        <mujoco>
          <option timestep="0.004" gravity="0 0 0" density="0" viscosity="0"/>
          <worldbody>
            <body name="bell" pos="0 0 {z}">
              <joint name="bell_j" type="hinge" axis="0 1 0" range="0 1"
                     limited="true"/>
              <geom type="sphere" size="0.06" density="400"/>
            </body>
          </worldbody>
        </mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        return model, data

    medium = MediumField()
    volume, stroke_frac, orifice = 1e-3, 0.6, 2e-4
    jets = JetSet(
        body_id=np.array([1]), joint_id=np.array([0]),
        axis_local=np.array([[1.0, 0.0, 0.0]]),
        volume=np.array([volume]), stroke_fraction=np.array([stroke_frac]),
        orifice_area=np.array([orifice]), joint_range=np.array([[0.0, 1.0]]),
    )

    # Contracting at omega = 2 rad/s over a 1 rad range sweeps
    # Q = V * stroke * omega / span = 1.2e-3 m^3/s.
    model, data = bell_model(-2.0)
    data.qpos[0], data.qvel[0] = 0.0, 2.0
    mujoco.mj_forward(model, data)
    jets.apply(model, data, medium, 0.0, model.opt.timestep)
    q = volume * stroke_frac * 2.0
    expected = medium.water.rho * q * q / orifice
    fx = float(data.xfrc_applied[1, 0])
    check("thrust magnitude is rho Q^2 / A",
          np.isclose(-fx, expected, rtol=1e-9),
          f"{-fx:.3f} N against {expected:.3f} N by hand")
    check("and it points opposite the expelled flow", fx < 0.0, f"fx={fx:.3f}")

    model, data = bell_model(-2.0)
    data.qpos[0], data.qvel[0] = 0.0, -2.0
    mujoco.mj_forward(model, data)
    jets.apply(model, data, medium, 0.0, model.opt.timestep)
    fx_refill = float(data.xfrc_applied[1, 0])
    check("refill is charged at the reduced coefficient, reversed",
          np.isclose(fx_refill, jets.refill_efficiency * expected, rtol=1e-9),
          f"{fx_refill:.3f} N against {jets.refill_efficiency * expected:.3f}")

    model, data = bell_model(10.0)
    data.qpos[0], data.qvel[0] = 0.0, 2.0
    mujoco.mj_forward(model, data)
    jets.apply(model, data, medium, 0.0, model.opt.timestep)
    check("a bell out of the water produces nothing",
          abs(float(data.xfrc_applied[1, 0])) < 1e-12,
          f"{float(data.xfrc_applied[1, 0]):.2e} N in air")

    # The wiring end: the medusa plan's own bells must reach the dynamics.
    p = build(BODY_PLANS["medusa"]())
    model, _data, _names, _panels = compile_phenotype(p)
    built = build_jets(p, model)
    check("the medusa's bells reach the jet model", built.n > 0,
          f"{built.n} bells")
    check("and at least one of them is driven by a real joint",
          bool((built.joint_id >= 0).any()),
          f"joint ids {built.joint_id.tolist()}")


def test_a_failing_sweep_gates_only_the_last_swimmer() -> None:
    """A wing that cannot sweep in water is fatal only when nothing else swims.

    Tier-0 rejected on ``min_margin`` over every check, and the hydrodynamic
    sweep check was 40 of arch24's 81 rejects -- nearly every newly grown wing
    died at the door for a load case the machine could avoid by holding that
    wing still and swimming on its paddles, which is what a real gannet does.
    The check stays in the report and in feasibility; it stops gating when
    other water propulsion remains.
    """
    print("\nstructure: a failing sweep gates only the last swimmer")
    from dytiscidae.core.bodyplans import gannet
    from dytiscidae.core.genome import WING as PART_WING
    from dytiscidae.core.phenotype import build

    g = gannet()
    for part in g.parts:
        if part.kind == PART_WING:
            part.span *= 1.6
    p = build(g)
    sweeps = [c for c in p.report.checks if c.name == "sweep_load_in_water"]
    failing = [c for c in sweeps if not c.ok]
    check("the oversized wing fails its sweep check", len(failing) >= 1,
          f"{len(failing)} of {len(sweeps)} sweep checks failing")
    check("while another surface still passes", any(c.ok for c in sweeps))
    check("the failing sweep no longer gates",
          all(not c.gating for c in failing))
    check("so the gate margin recovers while the true margin does not",
          p.report.gate_margin > 0.0 > p.report.min_margin,
          f"gate {p.report.gate_margin:+.2f} vs true {p.report.min_margin:+.2f}")
    check("and feasibility still tells the truth", not p.report.ok)

    # The reference gannet is untouched: everything passes, both margins agree.
    ref = build(gannet())
    check("a passing design's two margins agree",
          abs(ref.report.gate_margin - ref.report.min_margin) < 1e-12,
          f"{ref.report.gate_margin:+.2f}")


def test_the_power_budget_vectorises_without_changing_the_answer() -> None:
    """The batched budget must equal the per-actuator loop it replaced.

    ``PowerBudget.step`` called ``electrical_power`` and ``thermal_overload``
    once per actuator with a *scalar*, so every timestep paid a numpy round
    trip per actuator per quantity.  Profiled on real evolved bodies that was
    33-36% of a batched step, against 10% for ``mj_step``.

    The trap this pins: ``electrical_power`` adds shaft-seal friction and
    ``thermal_overload`` deliberately does not, so the two cannot share one
    torque.  Folding the seal into both moved the overload by 1.8e-2 -- small
    enough to look like rounding, large enough to shift the exploit threshold
    at ``max_actuator_overload > 3.0``.
    """
    print("\nenergy: the vectorised budget equals the loop it replaced")
    rng = np.random.default_rng(0)
    acts = [
        Actuator(motor_class=str(c), mass=float(m), gear_ratio=float(g),
                 sealed=bool(s))
        for c, m, g, s in zip(rng.choice(["bldc", "coreless", "geared"], 13),
                              rng.uniform(0.04, 0.3, 13),
                              rng.uniform(1.0, 12.0, 13),
                              rng.random(13) < 0.4)
    ]

    def scalar(budget, tq, sp):
        p, overload = budget.avionics_w, 0.0
        for i, a in enumerate(budget.actuators):
            if i >= len(tq):
                break
            p += float(a.electrical_power(tq[i], sp[i]))
            overload = max(overload, float(a.thermal_overload(tq[i], sp[i])))
        return p, overload

    worst_p = worst_o = 0.0
    for _ in range(200):
        tq, sp = rng.normal(0, 3, 13), rng.normal(0, 20, 13)
        b = PowerBudget(battery=Battery(wh=260), actuators=acts)
        ref_p, ref_o = scalar(b, tq, sp)
        b.step(tq, sp, 0.004)
        worst_p = max(worst_p, abs(b.total_j / 0.004 - ref_p) / max(abs(ref_p), 1e-9))
        worst_o = max(worst_o, abs(b.max_overload - ref_o) / max(abs(ref_o), 1e-9))

    check("power matches the scalar loop to floating point",
          worst_p < 1e-12, f"max relative error {worst_p:.2e}")
    check("and the overload matches exactly, seal friction excluded",
          worst_o == 0.0, f"max relative error {worst_o:.2e}")

    # Fewer actuators than torques, and none at all, are both real cases.
    short = PowerBudget(battery=Battery(wh=260), actuators=acts[:3])
    short.step(rng.normal(0, 3, 13), rng.normal(0, 20, 13), 0.004)
    check("a short actuator list charges only its own actuators",
          short.total_j > 0.0)
    empty = PowerBudget(battery=Battery(wh=260), actuators=[])
    empty.step(np.zeros(0), np.zeros(0), 0.004)
    check("a machine with no actuators still pays the hotel load",
          np.isclose(empty.total_j, empty.avionics_w * 0.004),
          f"{empty.total_j:.6f} J")


def test_training_states_are_a_distribution_not_a_pose() -> None:
    """Rollouts must not all begin from the same state.

    Every rollout the policy learned from started at one spawn pose per domain
    with a few centimetres of noise, identity attitude, no velocity, a full
    battery and a stroke phase of exactly zero -- and the crossings had no noise
    at all.  Two of the channels added to the observation were therefore
    constant across the whole training set: a battery that is always full and a
    stroke that always starts at the same point cannot be learned from.
    """
    print("\nsensing: training states are a distribution, not a pose")
    from dytiscidae.core.bodyplans import beetle
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    env = TriphibianEnv(build(beetle()))
    tilt, stroke, battery = [], [], []
    for k in range(60):
        env.reset(Domain.WATER)
        env.scatter(np.random.default_rng(k))
        o = env.observation(Domain.WATER)
        tilt.append(math.degrees(math.acos(min(1.0, max(-1.0, -o[8])))))
        battery.append(o[16])
        stroke.append(o[17])

    check("attitude is perturbed, but not into a tumble",
          2.0 < float(np.mean(tilt)) < 20.0 and max(tilt) < 45.0,
          f"mean {np.mean(tilt):.1f} deg, max {max(tilt):.1f} deg")
    check("the battery channel actually varies",
          float(np.std(battery)) > 0.05,
          f"sd {np.std(battery):.3f}, range "
          f"{min(battery):.2f}-{max(battery):.2f}")
    check("and so does the stroke phase",
          float(np.std(stroke)) > 1e-3, f"sd {np.std(stroke):.4f}")

    # Same generator, same state.  ``randomise=False`` isolates the scatter
    # from reset's own spawn noise, which draws from the env's generator and
    # therefore advances between calls -- the first version of this check
    # conflated the two and failed on the wrong thing.
    env.reset(Domain.WATER, randomise=False)
    env.scatter(np.random.default_rng(11))
    a = env.observation(Domain.WATER).copy()
    env.reset(Domain.WATER, randomise=False)
    env.scatter(np.random.default_rng(11))
    check("the same draw gives the same state",
          np.allclose(a, env.observation(Domain.WATER)))


def test_the_controller_senses_what_the_mission_scores() -> None:
    """The four senses added for the mission's own quantities behave.

    ``tanh(d/5)`` reads 0.96 at the 10 m target -- the one place the depth
    channel must discriminate is the one place it saturated, so depth-hold had
    to be inferred from reward alone.  The error channel is zero exactly at
    the target.  Battery, contact and stroke phase existed in the simulation
    and were invisible to the controller.
    """
    print("\nsensing: the controller senses what the mission scores")
    from dytiscidae.core.bodyplans import beetle
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    env = TriphibianEnv(build(beetle()))
    env.reset(Domain.WATER)
    obs = env.observation(Domain.WATER)
    check("the observation matches its declared width",
          len(obs) == TriphibianEnv.OBS_DIM,
          f"{len(obs)} vs OBS_DIM={TriphibianEnv.OBS_DIM}")

    # Teleport the root to the mission's target depth: the error channel must
    # read zero there, where the absolute channel is already saturated.
    env.data.qpos[2] = -TriphibianEnv.TARGET_DEPTH
    mujoco.mj_forward(env.model, env.data)
    at_target = env.observation(Domain.WATER)
    check("the depth-error channel is zero at the target",
          abs(at_target[14]) < 0.05, f"error channel {at_target[14]:+.3f}")
    check("where the absolute depth channel is nearly blind",
          at_target[9] > 0.9, f"tanh(d/5) = {at_target[9]:.3f}")

    env.reset(Domain.WATER)
    before = env.observation(Domain.WATER)[16]
    env.rollout(0.5, domain=Domain.WATER)
    after = env.observation(Domain.WATER)[16]
    check("the battery channel drains as energy is spent", after < before,
          f"{before:.5f} -> {after:.5f}")
    check("stroke phase and rate stay in their declared ranges",
          -1.0 <= after and abs(env.observation(Domain.WATER)[17]) <= 1.0
          and abs(env.observation(Domain.WATER)[18]) <= 3.0)


def test_flap_frequency_is_commandable_in_the_loop() -> None:
    """The identified basis can move flap frequency, so resonance is reachable.

    ``resonance_seek`` on the skill bench argues that a compliant wing driven at
    resonance costs a fraction of the power, and that the resonance *moves* when
    the machine enters water.  The bench's weights cannot transfer to a vehicle
    -- different observation and action widths over different dynamics -- so the
    question that matters is whether a mission policy can do the same thing in
    the loop.  It needs three things, and this pins the one that was never
    checked: frequency is the last entry of ``CPGParams.flat()``, and the
    mobility basis spans it.
    """
    print("\ncontrol: flap frequency is commandable through the mobility basis")
    from dytiscidae.control.cpg import CPGParams
    from dytiscidae.core.bodyplans import beetle
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.batchroll import identify_batch
    from dytiscidae.envs.triphibian import (
        MORPHOLOGY_DIM, Domain, TriphibianEnv)

    env = TriphibianEnv(build(beetle()))
    n = env.cpg.n
    check("the CPG parameter vector carries frequency last",
          env.cpg.n_params == 3 * n + 1, f"n_params={env.cpg.n_params}, n={n}")

    basis = identify_batch([env], Domain.WATER, seed=1)[0]
    freq_weight = float(np.max(np.abs(basis.modes[:, -1])))
    check("the identified water basis spans the frequency dimension",
          freq_weight > 0.05,
          f"largest frequency component across modes: {freq_weight:.3f}")

    # And commanding it actually changes the rhythm, rather than the component
    # being present in the matrix and discarded downstream.
    base = env.cpg.base
    mode = int(np.argmax(np.abs(basis.modes[:, -1])))
    coeffs = np.zeros(basis.modes.shape[0])
    coeffs[mode] = 1.0
    moved = basis.command_params(base, coeffs, n)
    check("commanding that mode moves the flap frequency",
          abs(moved.frequency - base.frequency) > 1e-6,
          f"{base.frequency:.4f} -> {moved.frequency:.4f} Hz")
    check("and the observation carries the stroke phase to time it against",
          TriphibianEnv.OBS_DIM == 25 + MORPHOLOGY_DIM,
          f"19 sensed + 6 task + {MORPHOLOGY_DIM} morphology = {TriphibianEnv.OBS_DIM}")
    del CPGParams


def test_an_auto_reset_rollout_is_not_trusted() -> None:
    """A segment whose physics blew up must fail, not be scored.

    MuJoCo 3.x auto-resets the state to the initial pose when qacc goes
    non-finite, so a blowup never trips the position-divergence guard: the
    machine teleports to spawn mid-rollout and keeps being scored as if the
    trajectory were real.  arch30's resumed population produced 18 such events
    in three generations with zero rollouts marked diverged -- an entirely
    silent score-corruption channel.
    """
    print("\nstability: an auto-reset rollout is marked unstable")
    from dytiscidae.core.bodyplans import beetle
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    env = TriphibianEnv(build(beetle()))
    env.reset(Domain.AIR)
    clean = env.rollout(0.1, domain=Domain.AIR)
    check("a clean rollout records no bad-qacc events",
          clean.bad_qacc == 0 and clean.failure != "unstable",
          f"bad_qacc={clean.bad_qacc}")

    bad_enum = mujoco.mjtWarning.mjWARN_BADQACC
    original = env.step

    def step_with_warning(angles):
        env.data.warning[bad_enum].number += 1
        return original(angles)

    env.reset(Domain.AIR)
    env.step = step_with_warning
    res = env.rollout(0.1, domain=Domain.AIR)
    env.step = original
    check("a rollout with bad-qacc events is marked unstable",
          (not res.survived) and res.failure == "unstable",
          f"survived={res.survived} failure={res.failure!r}")
    check("and the event count is recorded", res.bad_qacc > 0,
          f"bad_qacc={res.bad_qacc}")


def test_the_air_score_measures_flight() -> None:
    """Four ways an object that is not flying used to score for flight.

    Measured across arch33: winged designs scored 0.144 in air and wingless
    0.136, and both fell at about 11 m/s.  The score was not measuring flight,
    it was measuring having been thrown.
    """
    print("\nair score: what a machine did, not what it was given")
    import numpy as np
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import (
        MAX_SPIN_RATE, MAX_WING_LOADING, Domain, SegmentResult, TriphibianEnv,
        _turn_authority, airworthiness,
    )

    class Body:
        def __init__(self, area, loading):
            self.wing_area, self.wing_loading = area, loading

    # 1. The Tier-0 gates.  arch33's mission champion: 12 kg, wing_area 0.0000,
    #    wing loading 1,178,337 N/m^2.
    check("a design with no lifting surface is gated",
          airworthiness(Body(0.0, 1_178_337.0)) == ["no lifting surface"],
          f"{airworthiness(Body(0.0, 1_178_337.0))}")
    over = airworthiness(Body(0.05, MAX_WING_LOADING * 2))
    check("and so is one loaded past any speed that could carry it",
          len(over) == 1 and "wing loading" in over[0], f"{over}")
    check("but a real wing is not",
          airworthiness(Body(0.51, 85.6)) == [],
          "the gannet, 0.51 m^2 at 86 N/m^2")

    env = TriphibianEnv(build(BODY_PLANS["gannet"]()))
    dt, dur = env.timestep, 8.0
    n = int(dur / dt)

    def air(sink, *, spins=None, commands=None, responses=None, gates=(),
            clear=None, seconds=None):
        secs = dur if seconds is None else seconds
        m = int(secs / dt)
        r = SegmentResult(domain=Domain.AIR, duration=secs)
        r.mean_speed = 20.0
        if clear is None:
            clear = 30.0 - sink * np.arange(m) * dt
        env.air_gates = list(gates)
        try:
            s = env._score_segment(
                Domain.AIR, r, -clear, clear, np.ones(m), np.zeros(m), clear,
                spins=spins, commands=commands, responses=responses,
                xys=on_task_xy(env, Domain.AIR, m, secs))
        finally:
            env.air_gates = []
        return s, r.measurements

    # 2. Holding height is not the same as losing it slowly.
    _s, held = air(0.0)
    _s, creep = air(0.4)
    check("holding a height satisfies station keeping",
          held["station_keeping"] > 0.95, f"{held['station_keeping']:.2f}")
    check("and a shallow glide does not, though it clears the sink bar",
          creep["sink_rate"] < 0.5 and creep["station_keeping"] < 0.6,
          f"sink {creep['sink_rate']:.2f} m/s, station "
          f"{creep['station_keeping']:.2f}")

    # 2b. And a machine that drops and comes back reads as holding height.
    #
    # `sink_rate` is the difference between the two endpoints of the late half
    # of the airborne window, so a path that leaves its height and returns to it
    # scores zero sink and clears `holds_height` while holding nothing.  arch37
    # measured the consequence over 173 real segments: a straight descent at
    # their measured sink rates would have scored 0.704 median
    # `station_keeping`, and they scored 0.062 -- a tenth.  Every account of
    # *how* those paths were not straight was an inference, because nothing
    # published the shape.
    #
    # This is the fixture that inference describes, and `altitude_excursion` is
    # what now sees it.
    t_s = np.arange(n) * dt
    bounce = 30.0 - 3.0 * np.sin(np.pi * np.clip((t_s - 4.0) / 4.0, 0.0, 1.0))
    _s, bob = air(0.0, clear=bounce)
    check("a path that drops 3 m and returns still reads as holding height",
          abs(bob["sink_rate"]) < 0.5,
          f"sink {bob['sink_rate']:+.3f} m/s over a 3 m excursion")
    check("and the excursion is what says otherwise",
          bob["altitude_excursion"] > 2.5 and bob["excursion_ratio"] > 2.0,
          f"excursion {bob['altitude_excursion']:.2f} m = "
          f"{bob['excursion_ratio']:.1f} station bands")
    check("while a straight glide's excursion is just its drift",
          creep["altitude_excursion"] < 0.4 * dur / 2 + 1e-6,
          f"{creep['altitude_excursion']:.2f} m over a {dur / 2:.0f} s late "
          f"window at 0.4 m/s")
    check("and holding height leaves almost none",
          held["altitude_excursion"] < 1e-6, f"{held['altitude_excursion']:.3g} m")

    # 2c. The measurements must mean the same thing at any segment length.
    #
    # `station_keeping` is the fraction of its window spent within a band, so
    # for a steady descent at v it is `band / (v * T)` -- which makes the rung
    # threshold 0.6 an encoding of a *sink rate that depends on T*.  At 8 s it
    # asks for <= 0.21 m/s and at 24 s it would ask for <= 0.069 m/s under the
    # same name.  A ladder whose rungs change meaning with a config flag cannot
    # do the job this one is for, and the fix is not to keep the segment short:
    # a design that only holds height for four seconds is not holding height,
    # and a longer segment reveals that rather than causing it.
    #
    # So the segment length and the measurement window are separated.  Survival
    # is read by `airborne_fraction` and does get harder as the segment grows.
    # Holding a height is read over a fixed window and does not.
    _s, creep8 = air(0.4, seconds=8.0)
    _s, creep24 = air(0.4, seconds=24.0)
    check("a steady descent scores the same station at 8 s and at 24 s",
          abs(creep8["station_keeping"] - creep24["station_keeping"]) < 0.02,
          f"{creep8['station_keeping']:.3f} against "
          f"{creep24['station_keeping']:.3f}")
    check("and the same sink rate",
          abs(creep8["sink_rate"] - creep24["sink_rate"]) < 0.02,
          f"{creep8['sink_rate']:.3f} against {creep24['sink_rate']:.3f} m/s")

    # And the oscillation amplitude is window-invariant too, which the
    # excursion above is not -- `altitude_excursion` for a descent is the drift,
    # and drift grows with the window.  Subtracting the fitted trend leaves the
    # part that is not a descent at all.
    def wobbler(secs, amp=2.0, period=4.0, sink=0.4):
        tt = np.arange(int(secs / dt)) * dt
        return 30.0 - sink * tt + amp * np.sin(2 * np.pi * tt / period)

    _s, w8 = air(0.0, clear=wobbler(8.0), seconds=8.0)
    _s, w24 = air(0.0, clear=wobbler(24.0), seconds=24.0)
    check("an oscillation reads the same amplitude at 8 s and at 24 s",
          abs(w8["altitude_wobble"] - w24["altitude_wobble"]) < 0.25,
          f"{w8['altitude_wobble']:.2f} m against "
          f"{w24['altitude_wobble']:.2f} m, both against a 2.0 m amplitude")
    # The two are published together because they answer different questions,
    # not because one is window-invariant and the other is not -- capping the
    # window made both of those.  `altitude_excursion` reads the fixed tail
    # window and includes the drift within it; `altitude_wobble` reads the whole
    # airborne stretch and has the trend removed.  So a machine that thrashes
    # early and settles is invisible to the first and obvious to the second,
    # which is the case a segment long enough to have an early and a late part
    # makes possible at all.
    early = np.concatenate([
        30.0 + 3.0 * np.sin(2 * np.pi * np.arange(int(12.0 / dt)) * dt / 4.0),
        30.0 * np.ones(int(12.0 / dt)),
    ])
    _s, thrash = air(0.0, clear=early, seconds=24.0)
    check("a machine that thrashes early and settles is caught by the wobble",
          thrash["altitude_wobble"] > 1.0 and thrash["altitude_excursion"] < 0.1,
          f"wobble {thrash['altitude_wobble']:.2f} m, tail excursion "
          f"{thrash['altitude_excursion']:.3f} m -- the tail window alone "
          f"cannot see the first twelve seconds")
    check("and a straight descent has drift but no wobble",
          creep24["altitude_wobble"] < 0.05
          and creep24["altitude_excursion"] > 1.0,
          f"wobble {creep24['altitude_wobble']:.3f} m against excursion "
          f"{creep24['altitude_excursion']:.2f} m")

    # And the gate that decides whether episode measurements are published at
    # all must be window-invariant too, for the same reason and with a sharper
    # consequence.  It was `0.35 * res.duration`; at `--segment-seconds 24` that
    # became 8.4 s instead of 2.8 s and **1,571 of arch38's first 1,601 air
    # segments took the short-hop exit** -- nine of fourteen air rungs, the
    # thrust rungs among them, unreachable for 99.6% of the population, with the
    # share falling because selection could not see what it was no longer being
    # shown.
    slow8 = 30.0 - 0.4 * np.arange(int(8.0 / dt)) * dt
    slow24 = 30.0 - 0.4 * np.arange(int(24.0 / dt)) * dt
    _s, m8 = air(0.0, clear=slow8, seconds=8.0)
    _s, m24 = air(0.0, clear=slow24, seconds=24.0)
    check("a machine measurable at 8 s is still measurable at 24 s",
          "sink_rate" in m8 and "sink_rate" in m24,
          "both reach the full measurement path")
    # 4 s airborne clears the 2.8 s gate at either segment length; as a fraction
    # of 24 s it is 0.167 and would not have cleared 0.35.
    brief = np.concatenate([30.0 - 0.4 * np.arange(int(4.0 / dt)) * dt,
                            np.zeros(int(20.0 / dt))])
    _s, mb = air(0.0, clear=brief, seconds=24.0)
    check("and four seconds aloft is measurable however long the segment is",
          "sink_rate" in mb,
          "0.167 of a 24 s window, and 4 s of flight either way")

    # An unmeasurable thrust margin is absent, not zero.
    #
    # The first version returned 0.0 for "no actuated joints", "no drag to divide
    # by", "the pose could not be set" *and* "the flapping produces exactly
    # nothing" -- and the rung `flaps_forward` sat at `>= 0.0`, so the three
    # unmeasurable states cleared it.  Over 80 re-scored arch38 elites, 8 read
    # exactly 0.0 with a median `flap_travel` of 0.8998 rad: flapping nearly a
    # radian, and the number was a guard rather than a result.
    env.p._measured_thrust = None
    try:
        _s, unmeasured = air(0.0)
        check("an unmeasurable thrust margin is absent rather than zero",
              "thrust_margin" not in unmeasured and "flap_travel" in unmeasured,
              "a missing metric stops `rung_reached` where it stands, which is "
              "what 'unknown' means")
    finally:
        del env.p._measured_thrust

    # And the stroke is reported separately, so "did not flap" and "flapped to no
    # effect" are distinguishable in the record.
    check("the stroke travel is published beside the margin",
          held.get("flap_travel", 0.0) > 0.0,
          f"{held.get('flap_travel', 0.0):.3f} rad of joint travel per cycle")

    # The property that makes this safe to add mid-programme.  `rung_reached`
    # stops at the first rung whose metric is missing, and arch37's first launch
    # was voided by exactly that: a metric published on one of the air branch's
    # three exits and read by a rung.  These are published on one exit too --
    # and nothing reads them.
    from dytiscidae.evolution.judge import LADDER
    read_by_rungs = {m for rungs in LADDER.values() for _n, m, _t in rungs}
    check("no rung reads the new diagnostics, so a branch that skips them is safe",
          not read_by_rungs & {"altitude_excursion", "excursion_ratio"},
          "excursion metrics are diagnostics, not rungs")

    # 3. A tumble is not a commanded turn.
    rng = np.random.default_rng(0)
    cmd = [rng.normal(size=3) for _ in range(80)]
    follows = [np.zeros(3)] + [0.8 * c for c in cmd[:-1]]
    tumbles = [rng.normal(size=3) * 3.0 for _ in range(80)]
    a_follow, _ = _turn_authority(cmd, follows)
    a_tumble, _ = _turn_authority(tumbles, tumbles[::-1])
    check("a machine that turns when told to has command authority",
          a_follow > 0.9, f"corr {a_follow:+.2f}")
    check("and one that spins on its own does not",
          abs(a_tumble) < 0.3, f"corr {a_tumble:+.2f}")
    _s, uncommanded = air(0.0, commands=None, responses=None)
    check("so an uncommanded rotation scores no manoeuvring",
          uncommanded["turn_rate_held"] == 0.0,
          "the arch33 champion logged 31.4 rad/s with zero actuated DOF")

    # 4. Spinning faster than a revolution a second is not flight.
    fast = np.full(n, MAX_SPIN_RATE * 5.0)
    spun, m_spun = air(0.0, spins=fast)
    level, _ = air(0.0)
    check("a machine tumbling at five revolutions a second is not flying",
          spun < 0.1 * level, f"{spun:.3f} against {level:.3f} level")
    check("and it cannot climb the ladder on its sink rate either",
          m_spun["sink_rate"] > 9.0 and m_spun["measured_sink_rate"] == 0.0,
          f"ladder sees {m_spun['sink_rate']:.1f}, record keeps "
          f"{m_spun['measured_sink_rate']:.1f}")

    # 5. The launch is no longer inversely earned.
    lo, hi = __import__("dytiscidae.envs.triphibian", fromlist=["x"]).LAUNCH_SPEED_RANGE
    speed, _pitch, _margin = env._measure_trim_speed(lo, hi)
    check("a design that can fly is launched inside the band",
          lo <= speed <= hi, f"gannet launches at {speed:.1f} m/s")
    # The "no trim speed anywhere in the band" branch, forced by asking for a
    # band nothing can fly in.  This used to return the *top* of the range, so
    # the machines that could not fly at all were the ones thrown hardest.
    none_speed, _, _ = env._measure_trim_speed(1.0, 2.0)
    check("and one that cannot fly anywhere in the band is dropped, not thrown",
          abs(none_speed - 1.0) < 1e-9,
          f"released at {none_speed:.1f} m/s, the floor, not the 2.0 m/s cap")

    # 6. Being dropped no longer buys rungs.
    #
    # `airborne_fraction` is satisfied by falling: the segment releases the
    # machine at 30 m and a body dropped there is airborne for the 2.5 s it
    # takes to arrive, which is why 53% of arch36 scored `leaves_surface` and
    # `stays_up` with a population median sink rate of 9.9 m/s.  130 of its 179
    # elites (72.6%) could not lift their own weight at any speed in the band.
    #
    # `lift_margin` is a property of the airframe, its four rungs sit below the
    # ones that read where the machine happened to be, and `rung_reached` stops
    # at the first unmet rung -- so the ordering is the gate and no separate
    # condition is needed.
    from dytiscidae.evolution.judge import LADDER, rung_reached

    perfect_episode = {"airborne_fraction": 1.0, "sink_rate": 0.0,
                       "station_keeping": 1.0, "turn_rate_held": 1.0}
    grounded = rung_reached("air", {**perfect_episode, "lift_margin": 0.0})
    flying = rung_reached("air", {**perfect_episode, "lift_margin": 1.5})
    check("a body that makes no lift scores nothing in air, however long it "
          "stays up", grounded == 0,
          f"rung {grounded} on a whole segment airborne at zero sink")
    check("and the same episode from a body that can carry itself does score",
          flying > grounded, f"rung {flying} against rung {grounded}")
    check("the lift rungs sit below the ones that read the episode",
          [n for n, _, _ in LADDER["air"]][:4]
          == ["makes_lift", "carries_a_third", "nearly_flies", "carries_itself"],
          " -> ".join(n for n, _, _ in LADDER["air"][:5]))
    # Set from arch36's measured distribution: min -1.105, median 0.333,
    # p90 5.591.  A bar above the population reads zero and carries no gradient.
    partial = rung_reached("air", {**perfect_episode, "lift_margin": 0.35})
    check("and a body halfway there outscores one that makes none",
          0 < partial < flying, f"margin 0.35 -> rung {partial}")

    # 7. Every path out of the air branch publishes it.
    #
    # It has three, and the first attempt at this changed one of them: a grep
    # for "airborne_fraction" said there was a single producer, and the branch
    # that handles a hop too short to measure a sink rate over does not publish
    # that key at all.  50 of arch37's first 180 evaluations went through it and
    # scored air rung 0 whatever their airframe could do -- and that branch is
    # the one *every falling design* takes, because free fall from the 30 m
    # spawn lasts 2.5 s, which is 31% of an 8 s segment and under its 35% bar.
    #
    # So this asserts the property rather than one instance of it.
    # The branch tests read `frac` and `airborne_seconds` against `n_want`,
    # which is `duration / timestep` -- not the array length.  A duration that
    # does not match the sample count dilutes every fraction and silently sends
    # all three fixtures down the same path, which is what the first version of
    # this did.
    # Long enough that "airborne throughout" clears
    # `MEASURABLE_AIR_SECONDS`.  It used to be 96 samples -- 0.384 s -- which
    # was fine while the gate was a fraction of the segment and stopped being
    # fine the moment it became an absolute duration: the fixture quietly fell
    # into the short-hop branch and this test found it, which is what the
    # "three different paths" check below exists for.
    n_air = int(2.0 * TriphibianEnv.MEASURABLE_AIR_SECONDS / env.timestep)
    dur_air = n_air * env.timestep
    ones_a, zeros_a = np.ones(n_air), np.zeros(n_air)
    paths = {
        # never clear of the surface -> `frac < 0.05`
        "never airborne": dict(clearances=zeros_a, contacts=ones_a),
        # clear briefly -> the short-hop early return
        # airborne, but for less than `MEASURABLE_AIR_SECONDS`
        "brief hop": dict(
            clearances=np.where(np.arange(n_air) < n_air // 8, 3.0, 0.0),
            contacts=np.where(np.arange(n_air) < n_air // 8, 0.0, 1.0)),
        # clear throughout -> the full measurement
        "airborne throughout": dict(clearances=ones_a * 5.0, contacts=zeros_a),
        # the rollout went unstable -> the `not res.survived` early return,
        # which used to publish nothing at all.  243 of arch37's 14,092 air
        # segments left by it and scored rung 0 for a `bad_qacc` rather than for
        # an airframe.
        "diverged": dict(clearances=ones_a * 5.0, contacts=zeros_a,
                         survived=False),
    }
    missing, seen = [], {}
    for name, kw in paths.items():
        r = SegmentResult(domain=Domain.AIR, duration=dur_air)
        r.mean_speed = 0.0
        r.survived = kw.get("survived", True)
        env._score_segment(Domain.AIR, r, zeros_a, ones_a * 0.9, ones_a,
                           kw["contacts"], clearances=kw["clearances"],
                           vzs=zeros_a)
        m = r.measurements
        # Both of the airframe properties, not just the one that caused the
        # incident.  `lift_margin` was published on one exit of three and read
        # by a rung, which voided arch37's first launch; `thrust_margin` is the
        # same shape of quantity added the same way, so it gets the same check
        # rather than the same accident.
        if "lift_margin" not in m or "thrust_margin" not in m:
            missing.append(name + " (" + ", ".join(
                k for k in ("lift_margin", "thrust_margin") if k not in m) + ")")
        # Each branch leaves a different fingerprint, so this records *which*
        # one ran.  Without it the three fixtures could all be falling down the
        # same path and the check would pass while testing one third of what it
        # claims.
        seen[name] = ("diverged" if not r.survived
                      else "full" if "airborne_fraction" in m
                      else "short-hop" if "airborne_seconds" in m
                      else "never-airborne")
    check("every path out of the air branch publishes the airframe properties",
          not missing,
          "missing on: " + ", ".join(missing) if missing
          else ", ".join(f"{k}={v}" for k, v in seen.items()))
    check("and the four fixtures really do take four different paths",
          len(set(seen.values())) == 4, ", ".join(sorted(set(seen.values()))))


def test_takeoff_is_measured_where_the_machine_starts_on_the_ground() -> None:
    """The air segment is a launch, so nothing measured leaving the ground.

    arch34 ran 14,006 evaluations and its telemetry could not answer whether any
    design can take off, because every air score was earned from a 30 m spawn at
    trim speed.  A probe over its final archive put 96 elites on the beach for
    30 s: median peak clearance 0.050 m -- exactly the spawn value, so most
    never moved upward at all -- and best 0.133 m against a 0.50 m bar.

    So the measurement is the projectile estimate, not peak clearance: a
    threshold above 0.133 m reads zero for the whole population and carries no
    gradient, while ``clearance + max(0, vz)^2 / 2g`` is dense from the first
    upward velocity and does not require leaving the ground.
    """
    import numpy as np

    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, SegmentResult, TriphibianEnv
    from dytiscidae.evolution.judge import LADDER, rung_reached

    env = TriphibianEnv(build(BODY_PLANS["beetle"]()))
    n = 120
    ones, zeros = np.ones(n), np.zeros(n)

    # Airborne and upright at a constant height: it left the ground but gained
    # nothing, so the measure is a gain and reads zero.
    res = SegmentResult(domain=Domain.LAND, duration=1.0)
    res.mean_speed = 0.0
    env._score_segment(Domain.LAND, res, zeros, ones * 0.9, ones,
                       zeros, clearances=ones * 0.05, vzs=zeros)
    flat = res.measurements["takeoff_height"]
    check("height is measured as a gain, so holding one reads zero",
          abs(flat) < 1e-9, f"{flat:.4f} m")

    # The same machine with one upward impulse it never converts to height:
    # 2 m/s straight up is an apex 0.204 m above where it started.
    vz = zeros.copy()
    vz[10] = 2.0
    res2 = SegmentResult(domain=Domain.LAND, duration=1.0)
    res2.mean_speed = 0.0
    env._score_segment(Domain.LAND, res2, zeros, ones * 0.9, ones,
                       zeros, clearances=ones * 0.05, vzs=vz)
    impulse = res2.measurements["takeoff_height"]
    check("an upward impulse scores before any height is gained",
          impulse > flat, f"{flat:.4f} -> {impulse:.4f} m with 2 m/s of vz")
    check("and it scores exactly the apex that impulse implies",
          abs(impulse - 4.0 / (2 * 9.81)) < 1e-6,
          f"{impulse:.4f} m against v^2/2g = {4.0 / 19.62:.4f}")

    # The ladder has to put arch34's best inside it rather than at zero.
    check("arch34's best elite lands on a rung, not on the floor",
          rung_reached("takeoff", {"takeoff_height": 0.133}) >= 1,
          f"0.133 m -> rung {rung_reached('takeoff', {'takeoff_height': 0.133})}")
    check("and a machine that never rose does not",
          rung_reached("takeoff", {"takeoff_height": 0.0}) == 0)
    check("there is a slope above it rather than one bar",
          len(LADDER["takeoff"]) >= 3
          and rung_reached("takeoff", {"takeoff_height": 2.0}) > rung_reached(
              "takeoff", {"takeoff_height": 0.3}),
          f"{len(LADDER['takeoff'])} rungs")

    # Both gates, and both were put there by measurement rather than taste.
    # Ungated over 64 arch34 elites the estimate read p90 0.378 m from a
    # population whose best controlled hop is a tenth of that: it was scoring
    # the rebound of a machine falling over.
    fallen = SegmentResult(domain=Domain.LAND, duration=1.0)
    fallen.mean_speed = 0.0
    rising = zeros.copy()
    rising[10] = 3.0
    env._score_segment(Domain.LAND, fallen, zeros, ones * 0.9,
                       ones * 0.1,                      # on its side
                       zeros, clearances=ones * 0.05, vzs=rising)
    check("a machine on its side scores no take-off however fast it is rising",
          fallen.measurements["takeoff_height"] == 0.0)
    grounded = SegmentResult(domain=Domain.LAND, duration=1.0)
    grounded.mean_speed = 0.0
    env._score_segment(Domain.LAND, grounded, zeros, ones * 0.9, ones,
                       ones,                             # never off the ground
                       clearances=ones * 0.05, vzs=rising)
    check("and neither does one that never leaves the ground",
          grounded.measurements["takeoff_height"] == 0.0)

    # The estimate is dense and the achievement is not, so both are recorded.
    # The estimate is an apex and the measurement is a height actually held, so
    # ungated the first is the larger of the two.
    check("the height actually reached is kept beside the estimate",
          "measured_takeoff_height" in res2.measurements
          and res2.measurements["measured_takeoff_height"] <= res2.measurements[
              "takeoff_height"] + 1e-9)

    # --- arch36's two gates ------------------------------------------------
    #
    # Posture across the whole segment, not only at the apex.  A body flung by
    # a contact impulse is upright on its way through, so the per-sample gate
    # above passes it; over arch35 the designs below this bar scored 0.79-0.95x
    # the mission of those above it once the take-off multiplier was divided
    # out, while being paid for take-off.
    flung = SegmentResult(domain=Domain.LAND, duration=1.0)
    flung.mean_speed = 0.0
    tumbling = ones * 0.2          # on its side for almost the whole segment
    tumbling[9:12] = 1.0           # upright for the three samples around apex
    lofted = ones * 0.05           # and it really does get high, briefly
    lofted[9:12] = 0.9
    env._score_segment(Domain.LAND, flung, zeros, ones * 0.9, tumbling,
                       zeros, clearances=lofted, vzs=rising)
    check("a body upright only at its apex scores no take-off",
          flung.measurements["takeoff_height"] == 0.0,
          f"segment-mean posture {float(np.mean(tumbling)):.2f} "
          f"against the 0.7 bar `stays_upright` already uses")
    check("but the height it reached is still recorded, so the next instance "
          "of this is still findable",
          flung.measurements["measured_takeoff_height"] > 0.5,
          f"{flung.measurements['measured_takeoff_height']:.4f} m measured "
          f"against {flung.measurements['takeoff_height']:.4f} m scored")

    # A departure is a flight claim, and `airworthiness` already says which
    # designs may not make one.  arch35 had 31 wingless segments at `clears`
    # and 13 at `climbs_out`, one reaching 4.15 m measured.
    wingless = TriphibianEnv(build(BODY_PLANS["beetle"]()))
    wingless.p.wing_area = 0.0
    big = zeros.copy()
    big[10] = 6.0                                   # apex ~1.8 m, well past 0.80
    soar = SegmentResult(domain=Domain.LAND, duration=1.0)
    soar.mean_speed = 0.0
    wingless._score_segment(Domain.LAND, soar, zeros, ones * 0.9, ones,
                            zeros, clearances=ones * 0.05, vzs=big)
    winged = SegmentResult(domain=Domain.LAND, duration=1.0)
    winged.mean_speed = 0.0
    env._score_segment(Domain.LAND, winged, zeros, ones * 0.9, ones,
                       zeros, clearances=ones * 0.05, vzs=big)
    # Rungs are counted, so `clears` is rung 3 of unweights/hops/clears/
    # climbs_out and "cannot reach clears" is `< 3`, not `< 2`.
    check("a design with no lifting surface cannot reach `clears`",
          rung_reached("takeoff", soar.measurements) < 3,
          f"{soar.measurements['takeoff_height']:.3f} m scored from an apex of "
          f"{winged.measurements['takeoff_height']:.3f} m, rung "
          f"{rung_reached('takeoff', soar.measurements)}")
    check("and the same rollout with a wing does",
          rung_reached("takeoff", winged.measurements) >= 3,
          f"rung {rung_reached('takeoff', winged.measurements)}")
    check("but it keeps `hops`, because being thrown does leave the ground",
          rung_reached("takeoff", soar.measurements) >= 2,
          f"rung {rung_reached('takeoff', soar.measurements)}")

    # Locomotion: a lurch of half a metre inside one second, then nothing.
    lurch = SegmentResult(domain=Domain.LAND, duration=8.0)
    lurch.mean_speed = 0.05
    xy = np.zeros((n, 2))
    hop = n // 8
    xy[hop:2 * hop, 0] = np.linspace(0.0, 0.5, hop)
    xy[2 * hop:, 0] = 0.5
    env._score_segment(Domain.LAND, lurch, zeros, ones * 0.9, ones, ones,
                       clearances=ones * 0.05, vzs=zeros, xys=xy)
    # The ratio, not the factor 8.  That 8 was calibrated against the fixture's
    # hand-set `mean_speed` of 0.05, which used to *be* `land_speed`; the metric
    # is now the gated mean of the same posture-masked windows the peak is taken
    # over, so it is computed rather than stipulated.  For a 0.5 m lurch
    # occupying one second of an eight-second segment the arithmetic ceiling on
    # peak/mean is about 8 and the measured value is 7.0 -- the claim being made
    # is that the peak sees a lurch the mean spreads out, which is a several-fold
    # difference, not a specific constant.
    check("peak speed sees motion the segment mean averages away",
          lurch.measurements["land_peak_speed"]
          > 5 * lurch.measurements["land_speed"],
          f"gated mean {lurch.measurements['land_speed']:.3f} m/s vs peak "
          f"{lurch.measurements['land_peak_speed']:.3f} m/s, "
          f"{lurch.measurements['land_peak_speed'] / max(lurch.measurements['land_speed'], 1e-9):.1f}x")
    sliding = SegmentResult(domain=Domain.LAND, duration=8.0)
    sliding.mean_speed = 0.05
    env._score_segment(Domain.LAND, sliding, zeros, ones * 0.9,
                       ones * 0.1, ones,                # sliding on its side
                       clearances=ones * 0.05, vzs=zeros, xys=xy)
    check("but not a body sliding down the beach on its side",
          sliding.measurements["land_peak_speed"] == 0.0)

    # The new land rung goes in front of `moves`, where the population is.
    names = [r[0] for r in LADDER["land"]]
    check("the land ladder gains a rung below the one 61.6% were stuck at",
          names.index("stirs") < names.index("moves"), " -> ".join(names))

    # Take-off multiplies the mission rather than joining min(competences),
    # which would multiply the whole population by its own zero.
    from dytiscidae.envs.evaluate import TAKEOFF_FLOOR, TAKEOFF_FULL

    check("a design that never rises keeps the floor, not zero",
          TAKEOFF_FLOOR > 0.0, f"floor {TAKEOFF_FLOOR}")
    check("and one that clears the bar is worth the full factor",
          float(np.clip(TAKEOFF_FULL / TAKEOFF_FULL, 0, 1)) == 1.0
          and TAKEOFF_FULL / TAKEOFF_FLOOR >= 4.0,
          f"{TAKEOFF_FULL} m full, {1 / TAKEOFF_FLOOR:.0f}x the floor")


def test_depth_is_a_gain_not_a_spawn() -> None:
    """The water ladder had the air ladder's defect and outlived its fix.

    `SPAWN[Domain.WATER]` releases the machine four metres under, so the
    **minimum** `max_depth` over arch37's 14,058 water segments is 3.34 m and
    the first two water rungs -- `submerges` at 0.5 m, `dives` at 3.0 m
    absolute -- were cleared by 100% of every evaluation this project has ever
    run, by being dropped.  Exactly what `leaves_surface` and `stays_up` were
    doing in air, three runs after that was fixed.
    """
    print("\ndepth: a gain over where the machine was released")
    import numpy as np
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, SegmentResult, TriphibianEnv

    env = TriphibianEnv(build(BODY_PLANS["eel"]()))
    dt, dur = env.timestep, 8.0
    n = int(dur / dt)

    def water(depths):
        r = SegmentResult(domain=Domain.WATER, duration=dur)
        r.mean_speed = 0.5
        r.max_depth = float(np.max(depths))
        c = env._score_segment(
            Domain.WATER, r, np.asarray(depths, float), -np.asarray(depths, float),
            np.ones(n), np.zeros(n), np.zeros(n), vzs=np.zeros(n),
            xys=on_task_xy(env, Domain.WATER, n, dur))
        return c, r.measurements

    from dytiscidae.envs.tasks import schedule_for
    target = float(schedule_for(Domain.WATER).phases[0].depth)
    ramp = np.clip(np.arange(n) / (n / 8), 0, 1)
    held = 4.0 * np.ones(n)                                  # released and idle
    dove = 4.0 + (target - 4.0) * ramp                       # to the commanded depth
    plunged = 4.0 + 9.0 * np.clip(np.arange(n) / (n / 2), 0, 1)  # 4 m -> 13 m
    rose = 4.0 - 0.6 * np.clip(np.arange(n) / (n / 2), 0, 1)  # floats up

    c_held, m_held = water(held)
    c_dove, m_dove = water(dove)
    c_plunged, m_plunged = water(plunged)
    c_rose, m_rose = water(rose)

    check("a machine that stays where it was dropped gained nothing",
          abs(m_held["depth_gain"]) < 1e-6 and m_held["max_depth"] >= 3.9,
          f"max_depth {m_held['max_depth']:.2f} m, gain "
          f"{m_held['depth_gain']:+.3f} m")
    # `max_depth` is a maximum, so the gain is floored at zero by construction
    # and a machine that only ever rises scores 0 rather than a negative.  That
    # is the right floor -- "went no deeper than it was put" -- and it is worth
    # a check because the number that motivated these thresholds was computed
    # as `max_depth - 4.0` against a spawn that is randomised by 0.2 m, which
    # produced apparent negatives that the real definition cannot have.
    check("and one that only ever rises scores the floor, not a negative",
          m_rose["depth_gain"] == 0.0 and c_rose < c_dove,
          f"gain {m_rose['depth_gain']:+.3f} m, competence {c_rose:.4f} "
          f"against {c_dove:.4f}")
    check("while a real dive is measured from the release depth",
          abs(m_plunged["depth_gain"] - 9.0) < 1e-6,
          f"4 m -> 13 m reads {m_plunged['depth_gain']:+.2f} m, not "
          f"{m_plunged['max_depth']:.2f}")
    check("the competence no longer pays for the spawn",
          c_dove > c_held, f"idle {c_held:.4f} against diving {c_dove:.4f}")
    # Depth is a command now, not a thing to maximise: going to the depth
    # asked for and staying is the task, and plunging past it is not.
    check("and reaching the commanded depth beats plunging past it",
          c_dove > c_plunged,
          f"to {target:.1f} m {c_dove:.4f} against to 13 m {c_plunged:.4f}")

    # `max_depth` stays published unchanged, so every number measured before
    # this change is still re-derivable from the record.
    check("and the absolute depth is still on the record",
          abs(m_held["max_depth"] - 4.0) < 1e-6, f"{m_held['max_depth']:.2f} m")


def test_each_phase_is_scored_on_its_own_purpose() -> None:
    """A segment asks one thing at a time, and is scored on what it asked.

    Raised by the user 2026-09-21: "if the purpose is to go forward, forward
    should score; if it is to hover, it should hover then, and moving should be
    penalised -- not one reward applied everywhere."  Until then the controller
    was told which medium it was in and never what to do there, and each
    medium's score blended goals that fought: ``corr(headway,
    depth_station_keeping)`` was -0.264 over arch40's 6,288 water segments, and
    the water formula paid a perfect hover 0.20 against a perfect mover 0.80.

    So each segment is two phases with one purpose each (``envs/tasks.py``),
    the controller observes the purpose, and each phase is scored on its own
    objective.  This asserts the four properties that make that true:

      1. the observation carries the command, and it switches at the phase
         boundary;
      2. a cruise phase pays for the commanded velocity and nothing else --
         not standing still, not drifting across the heading, not overshooting;
      3. a hold phase pays for being at the commanded depth *and* not moving --
         not for sitting where the machine was released, not for sinking
         through the target on the way past;
      4. a machine with its actuators held still scores ~nothing in any medium,
         on the real rollout path, with no passive twin to subtract.
    """
    print("\ncompetence: each phase is scored on its own purpose")
    import inspect

    from dytiscidae.control.cpg import CPGParams
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.tasks import (CRUISE, HOLD, STOP, TASK_SPEED, Phase,
                                       TaskSchedule, schedule_for)
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    check("rollout runs one segment, not two",
          "passive_control" not in inspect.signature(TriphibianEnv.rollout).parameters)

    env = TriphibianEnv(build(BODY_PLANS["eel"]()), seed=0)
    dt = env.timestep
    n = int(8.0 / dt)

    # --- 1. the command is observed, and it switches ------------------------
    env.reset(Domain.WATER)
    check("outside a segment the task channels are silent, as in a transition",
          not np.any(env.task_channels()), f"{env.task_channels()}")
    env._arm_task(Domain.WATER, 8.0)
    first = env.task_channels().copy()
    obs_first = env.observation(Domain.WATER).copy()
    env.data.time = env._seg_t0 + 5.0          # into the second phase
    second = env.task_channels().copy()
    obs_second = env.observation(Domain.WATER).copy()
    env._active_task = None
    # What the *controller* receives, not what a helper can compute: the
    # channels have to be in the observation vector, in the state block ahead
    # of the morphology, and they have to change when the command does.  A
    # first version of this test checked `task_channels()` alone, and the
    # mutation that dropped them from the observation survived it.
    from dytiscidae.envs.triphibian import MORPHOLOGY_DIM
    lo = TriphibianEnv.OBS_DIM - MORPHOLOGY_DIM - 6
    check("the controller's observation carries the command",
          np.allclose(obs_first[lo:lo + 6], first) and np.allclose(obs_second[lo:lo + 6], second),
          f"obs {obs_first[lo:lo + 6]} vs channels {first}")
    check("and what it observes changes when the command changes",
          not np.allclose(obs_first[lo:lo + 6], obs_second[lo:lo + 6]))
    check("the default water task holds first, then cruises",
          first[1] == 1.0 and first[0] == 0.0 and second[0] == 1.0 and second[1] == 0.0,
          f"first {first[:2]}, second {second[:2]}")
    check("and the commanded speed appears only while cruising",
          first[2] == 0.0 and abs(second[2] - 1.0) < 1e-9, f"{first[2]} / {second[2]}")
    obs = env.observation(Domain.WATER)
    check("the observation is as wide as it says", obs.shape == (TriphibianEnv.OBS_DIM,),
          f"{obs.shape} vs {TriphibianEnv.OBS_DIM}")

    # --- 2 & 3. each phase on fabricated traces --------------------------------
    depth = 6.5
    sched = TaskSchedule("water", (Phase(HOLD, 0.0, depth=depth),
                                   Phase(CRUISE, 0.5, heading=0.0,
                                         speed=TASK_SPEED["water"], depth=depth)))

    def water(depths, vel2, vel1=0.0):
        """Score a water segment whose halves move at ``vel1`` and ``vel2``."""
        xy = np.zeros((n, 2))
        for i in range(1, n):
            xy[i] = xy[i - 1] + (vel2 if i >= n // 2 else vel1) * dt
        env._active_task = sched
        try:
            return env._task_scores(Domain.WATER, np.asarray(depths, float),
                                    -np.asarray(depths, float), xy, n)
        finally:
            env._active_task = None

    at_depth = np.concatenate([np.linspace(4.0, depth, n // 8), np.full(n - n // 8, depth)])
    v = TASK_SPEED["water"]
    good = water(at_depth, np.array([v, 0.0]))
    ph = {r["kind"]: r["score"] for r in good["phases"]}
    check("going to the commanded depth, holding it, then cruising as told scores full",
          ph[HOLD] > 0.99 and ph[CRUISE] > 0.99, f"{ph}")
    idle = water(np.full(n, 4.0), np.zeros(2))
    check("a hull that sits where it was released scores nothing on either phase",
          idle["task"] == 0.0, f"{[r['score'] for r in idle['phases']]}")
    sinking = np.linspace(4.0, 8.4, n)          # the still eel's 0.55 m/s
    through = water(sinking, np.array([v, 0.0]))
    hold_through = next(r for r in through["phases"] if r["kind"] == HOLD)
    check("sinking through the target on the way past does not count as holding it",
          hold_through["score"] < 0.1,
          f"hold {hold_through['score']:.3f}, depth error {hold_through['depth_error']:.2f} m, "
          f"moved {hold_through['drift']:.2f} m")
    # arch42's held-still eel: 0.094 m/s, a third of the old band, reaching the
    # commanded depth in the middle of the hold window.  It scored 0.63.
    t_s = np.arange(n) * dt
    slow = water(depth + 0.1 * (t_s - 0.375 * n * dt), np.array([v, 0.0]))
    hold_slow = next(r for r in slow["phases"] if r["kind"] == HOLD)
    check("sinking through it slowly does not count either",
          hold_slow["score"] < 0.05,
          f"hold {hold_slow['score']:.3f}, net {hold_slow['sink']:.3f} m/s")
    # At the depth and level, but drifting across it: the vertical term cannot
    # see this, so the stillness term has to.
    sideways = water(at_depth, np.array([v, 0.0]), vel1=np.array([0.0, v]))
    check("drifting sideways at the commanded depth is not holding it",
          next(r for r in sideways["phases"] if r["kind"] == HOLD)["score"] < 0.05,
          f"{next(r for r in sideways['phases'] if r['kind'] == HOLD)['score']:.3f}")
    across = water(at_depth, np.array([0.0, v]))
    check("cruising across the commanded heading makes no progress",
          next(r for r in across["phases"] if r["kind"] == CRUISE)["progress"] < 0.01,
          f"{next(r for r in across['phases'] if r['kind'] == CRUISE)['progress']:.4f}")
    double = water(at_depth, np.array([2 * v, 0.0]))
    # Forward is forward: the user's rule scores progress, so overshooting the
    # commanded speed is not a penalty -- the command sets what full marks is.
    check("going faster than asked is still full progress",
          next(r for r in double["phases"] if r["kind"] == CRUISE)["progress"] > 0.99)
    check("and a segment that holds, then cruises as told, scores full",
          good["task"] > 0.99, f"{good['task']:.4f}")
    drifting = water(at_depth, np.array([0.1, 0.0]))
    # Drifting at 0.4 m/s while told to hold: most of ``HOLD_DRIFT_SPEED``.
    hold_drift = water(np.concatenate([at_depth[: n // 4], np.linspace(depth, depth + 2.4, n - n // 4)]),
                       np.array([v, 0.0]))
    check("moving while told to hold is the penalty",
          next(r for r in hold_drift["phases"] if r["kind"] == HOLD)["score"]
          < next(r for r in good["phases"] if r["kind"] == HOLD)["score"] - 0.3,
          f"{next(r for r in hold_drift['phases'] if r['kind'] == HOLD)['score']:.3f}")
    check("and a slow cruise earns part of it, so there is a gradient to climb",
          0.0 < next(r for r in drifting["phases"] if r["kind"] == CRUISE)["score"] < 0.5)

    # Land: stopping is free for a rock, so it qualifies the walk instead of
    # adding to it.
    lsched = TaskSchedule("land", (Phase(CRUISE, 0.0, heading=0.0, speed=TASK_SPEED["land"]),
                                   Phase(STOP, 0.5)))

    def land(v1, v2):
        xy = np.zeros((n, 2))
        for i in range(1, n):
            xy[i] = xy[i - 1] + (v1 if i < n // 2 else v2) * dt
        env._active_task = lsched
        try:
            return env._task_scores(Domain.LAND, np.zeros(n), np.full(n, 0.3), xy, n)["task"]
        finally:
            env._active_task = None

    w = np.array([TASK_SPEED["land"], 0.0])
    rock, walker, runs_on = land(np.zeros(2), np.zeros(2)), land(w, np.zeros(2)), land(w, w)
    check("on land a rock scores nothing, though it stops perfectly", rock == 0.0, f"{rock}")
    check("walking as told and stopping as told scores full", walker > 0.99, f"{walker:.3f}")
    check("walking on when told to stop scores half", abs(runs_on - 0.5) < 0.05,
          f"{runs_on:.3f}")

    # Air: a turn is scored as a response to the command -- the change in
    # velocity across the old heading, toward the new one.  A body slowing
    # down on its launch heading used to project onto a wide turn and score.
    asched = TaskSchedule("air", (Phase(CRUISE, 0.0, heading=0.0, speed=10.0),
                                  Phase(CRUISE, 0.5, heading=np.radians(120.0), speed=10.0)))

    def air(v1, v2, airborne=None):
        xy = np.zeros((n, 2))
        for i in range(1, n):
            xy[i] = xy[i - 1] + (v1 if i < n // 2 else v2) * dt
        env._active_task = asched
        try:
            return env._task_scores(Domain.AIR, -np.full(n, 20.0), np.full(n, 20.0), xy, n,
                                    airborne=airborne)
        finally:
            env._active_task = None

    slowing = air(np.array([10.0, 0.0]), np.array([4.0, 0.0]))
    turned = air(np.array([10.0, 0.0]), 10.0 * np.array([np.cos(np.radians(120)), np.sin(np.radians(120))]))
    check("slowing down on the launch heading is not a turn",
          slowing["measurements"]["turn_response"] == 0.0,
          f"{slowing['measurements']['turn_response']:.3f}")
    check("turning as commanded is", turned["measurements"]["turn_response"] > 0.8,
          f"{turned['measurements']['turn_response']:.3f}")
    # arch42's held-still gannets: down in the sea, sliding away from the
    # commanded side, then stopped.  Stopping is a velocity change toward it.
    stopped = air(np.array([10.0, -3.0]), np.zeros(2))
    check("stopping after a slide away from the new heading is not a turn",
          stopped["measurements"]["turn_response"] == 0.0,
          f"{stopped['measurements']['turn_response']:.3f}")
    # Reversing along the launch line changes velocity toward *every* turn more
    # than 90 degrees off; only the normal to the old heading says it is none.
    reversed_ = air(np.array([10.0, 0.0]), np.array([-10.0, 0.0]))
    check("reversing along the launch line is not a turn either",
          reversed_["measurements"]["turn_response"] == 0.0,
          f"{reversed_['measurements']['turn_response']:.3f}")
    wet = air(np.array([10.0, 0.0]), 10.0 * np.array([np.cos(np.radians(120)), np.sin(np.radians(120))]),
              airborne=np.arange(n) < n // 2)
    check("and a turn made on the water is not a turn in the air",
          wet["measurements"]["turn_response"] == 0.0,
          f"{wet['measurements']['turn_response']:.3f}")

    # --- 4. held still, on the real rollout path ---------------------------------
    # Averaged over task draws, as a run draws them.  Progress may be passive --
    # a hull that glides forward is the user's "passive forward motion can
    # score" -- so it is reported, not barred.  What must be chosen may not be:
    # holding at a commanded depth, and responding to a stop.
    from dytiscidae.envs.tasks import task_seed
    for plan in ("beetle", "eel"):
        for dom in (Domain.WATER, Domain.LAND):
            e = TriphibianEnv(build(BODY_PLANS[plan]()), seed=3)
            chosen, prog = [], []
            for draw in range(8):
                e.reset(dom)
                e.scatter(np.random.default_rng(11 + draw))
                e.task = schedule_for(dom, np.random.default_rng(task_seed(11 + draw)))
                b = e.cpg.base
                still = CPGParams(amplitude=np.zeros(e.cpg.n), phase=np.asarray(b.phase, float),
                                  offset=np.asarray(e.data.qpos[e._act_qadr], float),
                                  frequency=float(b.frequency))
                m = e.rollout(8.0, params=still, domain=dom).measurements
                pr = float(m.get("cruise_progress" if dom is Domain.WATER else "walk_progress", 0.0))
                prog.append(pr)
                # Water: the hold must be chosen.  Land: stopping only qualifies
                # the walk, so a still body can score no more than its own
                # passive progress.
                chosen.append(float(m.get("hold_score", 0.0)) if dom is Domain.WATER
                              else max(0.0, float(m.get("task_score", 0.0)) - pr))
            check(f"{plan} held still in {dom.value}: what must be chosen scores ~nothing",
                  float(np.mean(chosen)) < 0.02,
                  f"{'hold' if dom is Domain.WATER else 'beyond passive progress'} mean "
                  f"{np.mean(chosen):.4f}, passive progress mean {np.mean(prog):.3f} (allowed)")
    check("and the task is re-drawn per generation, not fixed",
          schedule_for(Domain.WATER, np.random.default_rng(1))
          != schedule_for(Domain.WATER, np.random.default_rng(2)))

def test_the_first_air_reset_is_like_every_other() -> None:
    """Measuring a body's trim must not disturb the simulation it is measured for.

    ``launch_speed`` is computed the first time it is read -- inside ``reset``,
    for the air, after the spawn pose and its random offset are written -- and
    until 2026-09-21 it probed on the live ``MjData``.  So the first air reset
    of every phenotype object came out at the exact spawn, with the last
    probe's velocities, and every later one did not: an air segment's start
    depended on whether that phenotype's trim had been computed before.
    """
    print("\nreset: the first air reset of a body is like every other")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    cold = TriphibianEnv(build(BODY_PLANS["beetle"]()), seed=17)
    warm = TriphibianEnv(build(BODY_PLANS["beetle"]()), seed=17)
    _ = warm.launch_speed                       # trim computed before any reset
    cold.reset(Domain.AIR)
    warm.reset(Domain.AIR)
    check("a body whose trim is computed inside reset starts where a warmed one does",
          np.array_equal(cold.data.qpos, warm.data.qpos)
          and np.array_equal(cold.data.qvel, warm.data.qvel),
          f"cold {np.round(cold.data.qpos[:3], 4)} warm {np.round(warm.data.qpos[:3], 4)}")
    check("and it carries the spawn's random offset, not the bare spawn point",
          not np.allclose(cold.data.qpos[:3], TriphibianEnv.SPAWN[Domain.AIR]),
          f"{np.round(cold.data.qpos[:3], 4)}")


def test_the_gait_gain_can_stop_a_machine() -> None:
    """Stopping has to be in the action space before any reward can teach it.

    The policy commands ``base + modes^T c`` with ``c`` in [-1, 1]; measured on
    arch42's top elites the best such ``c`` left 74% of the base amplitude on
    land, so a walker told to stop could not.  The gain channel scales the
    commanded amplitude, and at 1 must change nothing at all.
    """
    print("\ncontrol: the gait gain")
    from dytiscidae.control.cpg import Policy, split_command
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.tasks import CRUISE, STOP, Phase, TaskSchedule
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    p = build(BODY_PLANS["beetle"]())
    env = TriphibianEnv(p, seed=3)
    b = env.identify(Domain.LAND, seed=3)
    base = env.cpg.base
    c = np.linspace(-0.5, 0.5, b.modes.shape[0])
    check("a gain of one is exactly the gait it always was",
          np.array_equal(b.command_params(base, c, env.cpg.n).flat(),
                         b.command_params(base, c, env.cpg.n, gain=1.0).flat()))
    check("a gain of zero holds the posture: no amplitude left",
          float(np.abs(b.command_params(base, c, env.cpg.n, gain=0.0).amplitude).max()) == 0.0)
    raw6, raw7 = np.zeros(6), np.append(np.zeros(6), -1.0)
    check("a policy without the channel drives at gain one; with it, it reads the last output",
          split_command(raw6, 6)[1] == 1.0 and split_command(raw7, 6)[1] == 0.0)

    # On the real rollout: told to stop, with the gain output saturated low.
    def stop_score(bias):
        pol = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=6, hidden=0, gain=True)
        pol.weights[-1] = bias
        e = TriphibianEnv(p, seed=3)
        e.reset(Domain.LAND)
        e.task = TaskSchedule("land", (Phase(CRUISE, 0.0, heading=0.0, speed=0.04),
                                       Phase(STOP, 0.5)))
        r = e.rollout(4.0, domain=Domain.LAND, params=e.cpg.base, policy=pol, basis=b)
        return r.measurements

    flapping, held = stop_score(0.0), stop_score(-10.0)
    check("a walker whose gain stays at one does not stop",
          flapping.get("stop_score", 0.0) < 0.5, f"stop {flapping.get('stop_score', 0.0):.3f}")
    check("and one that turns its gain down does", held.get("stop_score", 0.0) > 0.9,
          f"stop {held.get('stop_score', 0.0):.3f}")
    # And the run can see which it was, per phase.
    check("the commanded gain is published per phase",
          abs(flapping.get("gain_stop", -1) - 1.0) < 1e-9 and held.get("gain_stop", 1) < 0.01,
          f"gain_stop {flapping.get('gain_stop')} / {held.get('gain_stop')}")
    e = TriphibianEnv(p, seed=3)
    e.reset(Domain.LAND)
    blind = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=6, hidden=0)
    m = e.rollout(1.0, domain=Domain.LAND, params=e.cpg.base, policy=blind, basis=b).measurements
    check("and not at all for a controller without the channel",
          not any(k.startswith("gain_") for k in m), f"{sorted(k for k in m if k.startswith('gain_'))}")


def test_chattering_commands_are_measured() -> None:
    """A controller's command rate and reversals are published, and can be charged.

    ROADMAP item Y: the filmed arch40 machine, sitting on sand, reversed its
    command direction on 72% of 40 ms control steps -- a loop through its
    body-rate channels.  `command_rate` and `command_reversal` are published on
    every segment with a controller so a run can measure their distribution;
    `action_rate_penalty` (default 0) charges the rate once a weight is set
    from it.
    """
    print("\ncommands: chatter is measured, and chargeable")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.tasks import TASK_SPEED
    from dytiscidae.envs.triphibian import (Domain, SegmentResult, TriphibianEnv,
                                            command_statistics)

    smooth = [np.array([0.01 * k, 0.0, 0.0]) for k in range(50)]
    chatter = [np.array([0.3 * (-1) ** k, 0.0, 0.0]) for k in range(50)]
    s_rate, s_rev = command_statistics(smooth)
    c_rate, c_rev = command_statistics(chatter)
    check("a steadily ramping command never reverses", s_rev == 0.0, f"{s_rev}")
    check("an alternating one reverses on every step", c_rev == 1.0, f"{c_rev}")
    check("and its rate is the size of the step", abs(c_rate - 0.6) < 1e-12, f"{c_rate}")
    check("too few decisions to say is None, not zero", command_statistics(smooth[:2]) is None)

    env = TriphibianEnv(build(BODY_PLANS["eel"]()), seed=0)
    n = int(8.0 / env.timestep)
    depths = np.concatenate([np.linspace(4.0, 5.75, n // 8), np.full(n - n // 8, 5.75)])
    xy = np.zeros((n, 2))
    for i in range(1, n):
        xy[i] = xy[i - 1] + (np.array([0.0, TASK_SPEED["water"]]) if i >= n // 2 else 0.0) * env.timestep

    def score(commands, weight):
        env.action_rate_penalty = weight
        res = SegmentResult(domain=Domain.WATER, duration=8.0, survived=True)
        res.distance = float(np.linalg.norm(xy[-1]))
        res.mean_speed = res.distance / 8.0
        c = env._score_segment(Domain.WATER, res, depths, -depths, np.ones(n), np.zeros(n),
                               -depths, commands=commands, spins=np.zeros(n),
                               vzs=np.zeros(n), xys=xy)
        return c, res.measurements

    base, m = score(chatter, 0.0)
    check("the statistics are published on the segment",
          m.get("command_reversal") == 1.0 and abs(m.get("command_rate", 0) - 0.6) < 1e-12,
          f"{m.get('command_rate')} / {m.get('command_reversal')}")
    check("at the default weight they change nothing", base == score(smooth, 0.0)[0] and base > 0.0,
          f"{base:.4f}")
    charged = score(chatter, 1.0)[0]
    check("with a weight, chattering costs and smooth commands do not",
          charged < base and score(smooth, 1.0)[0] > charged,
          f"chatter {charged:.4f} smooth {score(smooth, 1.0)[0]:.4f} unweighted {base:.4f}")
    env.action_rate_penalty = 0.0


def test_a_film_is_the_evaluation() -> None:
    """Attaching a camera to the Tier-1 evaluation must not change what it measures.

    `viz/film.py` films an elite by re-running ``evaluate_tier1`` -- the function
    that produces a score -- with ``detail=True`` (the lifting surfaces drawn as
    the shape the fluid solver reads) and a per-step ``on_step`` hook.  Both are
    meant to be invisible to the physics: the detail strips are massless and
    collisionless, and the hook only reads.  If either leaked, every film would
    be a slightly different experiment from the score printed on it, which is
    the failure the film module exists to end.
    """
    print("\nfilm: a camera on the evaluation changes nothing it measures")
    from dytiscidae.control.cpg import Policy
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.evaluate import Controller, evaluate_tier1
    from dytiscidae.envs.triphibian import TriphibianEnv

    p = build(BODY_PLANS["beetle"]())
    rng = np.random.default_rng(3)
    pol = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=6, hidden=0)
    pol.weights = rng.normal(0.0, 0.5, pol.n_weights)
    base = evaluate_tier1(p, controller=Controller(params=None, policy=pol),
                          segment_seconds=2.0, seed=17, identify_axes=True)
    bases = {k.value if hasattr(k, "value") else k: v for k, v in base.mobility.items()}

    def run(detail, hook):
        seen = []
        r = evaluate_tier1(p, controller=Controller(params=None, policy=pol, bases=dict(bases)),
                           segment_seconds=2.0, seed=17, identify_axes=False, detail=detail,
                           on_step=(lambda d, e, i: seen.append(i)) if hook else None)
        return r, len(seen)

    plain, _ = run(False, False)
    filmed, calls = run(True, True)
    same = all(
        plain.segments[d].competence == filmed.segments[d].competence
        and plain.segments[d].distance == filmed.segments[d].distance
        for d in plain.segments)
    check("detail geometry and a per-step hook leave every segment bit-identical",
          same, " ".join(f"{getattr(d, 'value', d)} {plain.segments[d].distance:.6f}/"
                         f"{filmed.segments[d].distance:.6f}" for d in plain.segments))
    check("and the hook was actually called on every step of every segment",
          calls == 3 * int(2.0 / TriphibianEnv(p).timestep), f"{calls} calls")


def _strip_force(wind, alpha_deg, *, camber=0.0, omega_y=0.0):
    """Force on the single test strip in a uniform wind, body at rest except
    for a pitch rate ``omega_y`` about the span axis (+Y)."""
    m, d = _single_panel_model()
    panels = _wing_panels(m, alpha_deg)
    panels.camber = np.array([camber])
    solver = FluidSolver(m, panels, MediumField(wind=np.asarray(wind, float)))
    d.qvel[:] = 0.0
    d.qvel[4] = omega_y           # free joint: angular part is body-local
    d.xfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)
    solver.apply(d, 0.0)
    f = d.xfrc_applied[panels.body_id[0], :3].copy()
    # Less the entrained fluid's weight cancellation (`finish_bodies`), which is
    # not a flow force.  It was taken off as the still-air force, which stopped
    # being the same number when the added mass became direction-dependent
    # (F-03): at rest the tensor takes the mean of its three axes.
    b = panels.body_id[0]
    f[2] -= (m.body_mass[b] - solver._dry_mass[b]) * 9.80665
    return f


def test_a_strip_moves_with_its_hinge() -> None:
    """A strip's velocity is its hinge's rotation times its distance from the
    *hinge*, not from the centre of mass.

    mj_objectVelocity on mjOBJ_BODY reports the linear velocity at ``xipos``
    and the strip lever arm was taken from ``xpos``; for a wing hinged at its
    root that counted the hinge-to-centre-of-mass arm twice, and flapping strip
    speeds came out 1.8-3.6x too high in U^2 S on the seed plans (2026-09-23).
    """
    print("\nfluid: a flapping strip moves with its hinge")
    xml = """
    <mujoco><option timestep="0.001" gravity="0 0 0" density="0" viscosity="0"/>
      <worldbody><body name="wing" pos="0 0 5">
        <joint type="hinge" axis="1 0 0"/>
        <geom type="box" size="0.1 0.5 0.002" pos="0 0.5 0" density="200"/>
      </body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wing")
    panels = PanelSet(
        body_id=np.array([bid]), pos_local=np.array([[0.0, 1.0, 0.0]]),
        span_local=np.array([[0.0, 1.0, 0.0]]),
        chord_local=np.array([[1.0, 0.0, 0.0]]),
        chord=np.array([0.2]), dr=np.array([0.1]), volume=np.array([0.0]),
        half_height=np.array([0.02]), kind=np.array([WING]),
        aspect_ratio=np.array([5.0]), cd_bluff=np.array([0.0]))
    solver = FluidSolver(m, panels, MediumField())
    solver.record_state = True
    d.qvel[0] = 4.0
    mujoco.mj_forward(m, d)
    solver.apply(d, 0.0)
    u = float(solver.last_state["speed"][0])
    check("a strip 1 m from the hinge at 4 rad/s moves at 4 m/s",
          abs(u - 4.0) < 1e-6,
          f"U = {u:.4f} m/s (the centre of mass is 0.5 m out; the old arm gave 6.0)")


def test_reversed_flow_reverses_the_force() -> None:
    """A plate swept backwards at a fixed pitch pushes the other way.

    Incidence was folded with ``atan2(sin, |cos|)``, a mirror, so flow over the
    trailing edge at 171 deg read as +9 deg and not -9 deg; a strip moving
    forward and one moving backward at the same pitch got the same lift (probe:
    -3.770 N both ways).  A cambered arc, being fore-aft symmetric, still lifts
    toward its convex side from either end.
    """
    print("\nfluid: reversed flow reverses the force")
    fwd = _strip_force([10.0, 0, 0], 10.0)
    back = _strip_force([-10.0, 0, 0], 10.0)
    check("a flat plate at +10 deg lifts up in one direction and down in the other",
          fwd[2] > 0.0 and back[2] < 0.0 and abs(fwd[2] + back[2]) < 1e-6 * abs(fwd[2]) + 1e-9,
          f"Fz forward {fwd[2]:+.3f} N, backward {back[2]:+.3f} N")
    cf = _strip_force([10.0, 0, 0], 0.0, camber=0.06)
    cb = _strip_force([-10.0, 0, 0], 0.0, camber=0.06)
    # n = s x c is -Z for this strip, so the convex side is below it.
    check("a cambered strip at zero pitch lifts toward its convex side both ways",
          cf[2] < 0.0 and cb[2] < 0.0 and abs(cf[2] - cb[2]) < 1e-6 * abs(cf[2]) + 1e-9,
          f"Fz forward {cf[2]:+.3f} N, backward {cb[2]:+.3f} N")


def test_pitching_nose_up_adds_lift() -> None:
    """The Kramer (rotational) force adds lift while the wing pitches nose-up.

    d(alpha)/dt = -omega_s in this module's frame, and the term was written
    with +omega_s, so it opposed lift during pitch-up (2026-09-23).  The LEV
    term reads |pitch rate| and is symmetric, so the difference between the two
    signs isolates Kramer.
    """
    print("\nfluid: pitching nose-up adds lift")
    up = _strip_force([10.0, 0, 0], 10.0, omega_y=+5.0)
    down = _strip_force([10.0, 0, 0], 10.0, omega_y=-5.0)
    check("nose-up pitching lifts more than nose-down at the same rate",
          up[2] > down[2], f"Fz nose-up {up[2]:+.3f} N, nose-down {down[2]:+.3f} N")


def test_the_servo_reaches_a_fast_stroke() -> None:
    """A commanded stroke is reached where the motor has the torque for it.

    The position servo damps the joint's absolute velocity, so alone it lags by
    kv/kp = 75 ms and the gannet reached 27% of a 7.3 Hz stroke with its torque
    never saturating.  The env feeds the reference rate forward; this drives the
    gannet's joints through ``env.step`` with the hull held and reads the
    reached stroke.
    """
    print("\nactuation: the servo reaches a fast stroke")
    from dytiscidae.control.cpg import CPGParams
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import TriphibianEnv

    env = TriphibianEnv(build(BODY_PLANS["gannet"]()), seed=0)
    m, d = env.model, env.data
    m.opt.gravity[:] = 0.0
    env.solver.lift_scale = env.solver.cd_scale = 0.0
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    q0 = d.qpos.copy()
    n = env.cpg.n
    base = env.cpg.base
    amp = np.minimum(0.5, 0.9 * (env.cpg.hi - env.cpg.lo) / 2)
    p = CPGParams(amp, np.zeros(n), 0.5 * (env.cpg.hi + env.cpg.lo), 7.3)
    del base
    jq = [m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)]
    env.cpg.reset()
    rec = []
    t_end = d.time + 3.0
    while d.time < t_end:
        env.step(env.cpg.command(p, d.time))
        for j in free:
            a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
            d.qpos[a:a + 7] = q0[a:a + 7]
            d.qvel[v:v + 6] = 0.0
        if t_end - d.time < 1.5:
            rec.append(d.qpos[jq].copy())
    rec = np.array(rec)
    reach = float(np.median((rec.max(0) - rec.min(0)) / (2 * amp[: m.nu])))
    check("the gannet reaches at least 85% of a 7.3 Hz stroke",
          reach > 0.85, f"median reached/commanded {reach:.2f} (0.27 without feed-forward)")


def test_a_frequency_change_does_not_jump_the_stroke() -> None:
    """Changing the flap frequency changes the rhythm, not the stroke position.

    The phase was 2 pi f t on the absolute clock, so 0.1 Hz at t = 4 s jumped
    the stroke by 2.5 rad in one control step.
    """
    print("\ncontrol: a frequency change keeps the stroke continuous")
    from dytiscidae.control.cpg import CPG, CPGParams

    c = CPG(1)
    p1 = CPGParams(np.array([0.5]), np.zeros(1), np.zeros(1), 2.0)
    p2 = CPGParams(np.array([0.5]), np.zeros(1), np.zeros(1), 2.1)
    dt = 0.004
    a0 = c.command(p1, 4.0)
    a1 = c.command(p2, 4.0 + dt)
    bound = 0.5 * 2 * np.pi * 2.1 * dt * 1.01
    check("one step after a 0.1 Hz change at t = 4 s the stroke moves by one step's worth",
          abs(float(a1[0] - a0[0])) <= bound,
          f"|delta| = {abs(float(a1[0] - a0[0])):.4f} rad, bound {bound:.4f}")
    c2 = CPG(1)
    same = [float(c2.command(p1, k * dt)[0]) for k in range(500)]
    closed = [float(np.zeros(1)[0] + 0.5 * np.sin(2.0 * np.pi * 2.0 * (k * dt) + 0.0 + 0.0))
              for k in range(500)]
    check("and at constant frequency it is the closed form exactly",
          max(abs(a - b) for a, b in zip(same, closed)) == 0.0)


def test_a_fall_scores_no_flight() -> None:
    """A body that reaches the sea from the launch faster than anything flying
    could scores zero for flight.

    The short-exit branch paid ``credit * frac * 0.10`` -- 0.033 across arch42
    -- while a body that stayed up longer but sank faster than 6 m/s scored 0,
    so air competence correlated -0.44 with time aloft.
    """
    print("\nair score: a fall is not flight")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, SegmentResult, TriphibianEnv

    env = TriphibianEnv(build(BODY_PLANS["gannet"]()))
    n = int(2.0 * TriphibianEnv.MEASURABLE_AIR_SECONDS / env.timestep)
    ones, zeros = np.ones(n), np.zeros(n)
    k = int(0.9 * TriphibianEnv.MEASURABLE_AIR_SECONDS / env.timestep)
    clr = np.where(np.arange(n) < k, np.linspace(30.0, 0.0, n) * 2, 0.0)
    r = SegmentResult(domain=Domain.AIR, duration=n * env.timestep)
    r.mean_speed = 0.0
    s = env._score_segment(Domain.AIR, r, zeros, ones * 0.9, ones,
                           np.where(np.arange(n) < k, 0.0, 1.0),
                           clearances=clr, vzs=zeros)
    check("a body in the sea after 2.5 s scores 0 in air",
          s == 0.0 and "airborne_seconds" in r.measurements,
          f"score {s}, airborne {r.measurements.get('airborne_seconds')}")


def test_the_rotor_model_matches_a_measured_propeller() -> None:
    """BEMT against the UIUC APC 10x4.7 SF, and a mirrored pair cancels in yaw."""
    print("\nrotor: blade-element momentum theory against a measured propeller")
    from dytiscidae.core.bodyplans import BODY_PLANS, REFERENCE_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv, airworthiness, rotor_lift_ratio
    from dytiscidae.physics.medium import AIR
    from dytiscidae.physics.rotor import RotorSpec, bemt

    D, rpm = 0.254, 5018.0
    spec = RotorSpec(radius=D / 2, pitch=4.7 * 0.0254, blades=2)
    n = rpm / 60.0
    meas = [(0.115, 0.0476), (0.262, 0.0442), (0.408, 0.0383), (0.519, 0.0312), (0.576, 0.0268)]
    err = []
    for J, cp in meas:
        T, Q = bemt(spec, 2 * math.pi * n, J * n * D, 0.0, AIR.rho, AIR.mu)
        err.append(Q * 2 * math.pi * n / (AIR.rho * n**3 * D**5) - cp)
    rms = math.sqrt(sum(e * e for e in err) / len(err))
    check("power coefficient over J = 0.1-0.6 within 0.006 rms of the measured one",
          rms < 0.006, f"rms {rms:.4f} (camber is fitted to static CT only)")
    T0, _ = bemt(spec, 2 * math.pi * 4029 / 60, 0.0, 0.0, AIR.rho, AIR.mu)
    ct0 = T0 / (AIR.rho * (4029 / 60) ** 2 * D**4)
    check("static thrust coefficient within 25% of the measured 0.1158",
          abs(ct0 / 0.1158 - 1.0) < 0.25, f"CT0 {ct0:.4f}")

    check("the quadrotor is a reference, not a seed", "quad" not in BODY_PLANS)
    p = build(REFERENCE_PLANS["quad"]())
    check("a machine its rotors can lift is airworthy without a wing",
          airworthiness(p) == [] and rotor_lift_ratio(p) > 1.0,
          f"rotor lift ratio {rotor_lift_ratio(p):.2f}")
    env = TriphibianEnv(p, seed=0)
    env.reset(Domain.AIR, randomise=False)
    env.data.qvel[:] = 0.0
    for k in env.rotors.dof:
        env.data.qvel[k] = 600.0
    mujoco.mj_forward(env.model, env.data)
    env.data.xfrc_applied[:] = 0.0
    env.rotors.apply(env.model, env.data, env.medium, 0.0)
    F = env.data.xfrc_applied[:, :3].sum(0)
    Tq = env.data.xfrc_applied[:, 3:].sum(0)
    check("its mirrored pairs are opposite-handed: thrust up, yaw torques cancel",
          F[2] > 0 and abs(F[0]) + abs(F[1]) < 1e-9 * F[2] and abs(Tq[2]) < 1e-9,
          f"F {np.round(F, 3)}, net aero yaw {Tq[2]:.2e}")


def test_a_universal_joint_flaps_and_feathers() -> None:
    """ROADMAP AB: a "universal" joint is a stroke hinge and a feathering hinge
    about the span, each with its own motor -- it was built as a plain hinge."""
    print("\nmjcf: a universal joint flaps and feathers")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.core.mjcf import compile_phenotype

    g = BODY_PLANS["gannet"]()
    plain = build(BODY_PLANS["gannet"]())
    n_wing = 0
    for part in g.parts:
        if part.joint == "hinge" and part.actuated and part.is_surface:
            part.joint = "universal"
            n_wing += 1
    p = build(g)
    m, d, acts, _ = compile_phenotype(p)
    feather = [a for a in acts if a.endswith("_f")]
    check("every universal surface gets a feathering actuator",
          n_wing > 0 and len(feather) == sum(1 for s in p.segments if s.part.joint == "universal"),
          f"{len(feather)} feathering actuators")
    j = m.actuator_trnid[acts.index(feather[0]), 0]
    check("the feathering hinge turns about the span (local +/-X)",
          abs(abs(m.jnt_axis[j][0]) - 1.0) < 1e-12, f"axis {m.jnt_axis[j]}")
    check("and its motor is a second motor, with its own mass",
          p.budget.actuators > plain.budget.actuators + 1e-6,
          f"{plain.budget.actuators:.3f} -> {p.budget.actuators:.3f} kg of motors")


def test_the_wagner_slam_pressure() -> None:
    """S-02: Wagner's spray-root peak, 0.5 rho (pi V / (2 tan b))^2."""
    print("\nstructure: Wagner's slam pressure")
    from dytiscidae.physics.medium import SEAWATER
    from dytiscidae.physics.structure import slam_pressure
    b = math.radians(20.0)
    want = 0.5 * SEAWATER.rho * (math.pi / (2 * math.tan(b))) ** 2
    got = slam_pressure(1.0, 20.0)
    check("at 20 degrees deadrise the peak is (pi/(2 tan b))^2 of 0.5 rho V^2",
          abs(got / want - 1.0) < 1e-12, f"{got:.1f} Pa against {want:.1f}")


def test_lift_and_drag_are_integrated_implicitly() -> None:
    """F-03: the damping split changes nothing at the current state, and it is
    what keeps the corrected added mass stable."""
    print("\nfluid: implicit lift and drag")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    env = TriphibianEnv(build(BODY_PLANS["ray"]()), seed=0)
    env.reset(Domain.AIR, randomise=False)
    for _ in range(50):
        env.step(env.cpg.command(env.cpg.base, env.data.time))
    # Formed on the state it is checked against: `step` moves qvel afterwards.
    env.data.xfrc_applied[:] = 0.0
    env.solver.apply(env.data, env.data.time)
    dmp = env.solver._damping
    B = dmp.last_b
    check("the solver ships with the split on, and it is live on a flapping machine",
          env.solver.implicit_damping and env.solver.wing_added_mass_tensor
          and float(B.max()) > 0.0,
          f"implicit {env.solver.implicit_damping}, tensor "
          f"{env.solver.wing_added_mass_tensor}, max B {float(B.max()):.3g}")
    check("dof_damping is the dry value plus the split",
          np.allclose(env.model.dof_damping - dmp.base, B, atol=1e-12) or env.jets.n > 0,
          f"max B {B.max():.3g} N s/m")
    check("and qfrc_applied carries +B qdot, so the force at this state is unchanged",
          np.allclose(env.data.qfrc_applied, B * env.data.qvel, atol=1e-9))
    # The plan that runs away moved as the fluid model did (ray in air before
    # the unsteady terms, teal in water after): the sweep is
    # experiments/flight_audit/stability_probe.py, 2026-09-26.
    big = []
    for implicit in (True, False):
        e = TriphibianEnv(build(BODY_PLANS["teal"]()), seed=0)
        e.solver.implicit_damping = implicit
        e.reset(Domain.WATER, randomise=False)
        q = []
        for _ in range(int(5.0 / e.timestep)):
            e.step(e.cpg.command(e.cpg.base, e.data.time))
            q.append(np.abs(e.data.qpos[7:]).max())
        big.append(float(np.max(q)))
    check("the teal in water, with the added-mass tensor, stays bounded implicitly "
          "and runs away explicitly",
          big[0] < 5.0 and big[1] > 20.0,
          f"largest joint angle over 5 s: {big[0]:.2f} rad implicit, {big[1]:.1f} explicit")


def test_the_limiter_and_the_weight_cancellation() -> None:
    """F-04 and F-14 in `finish_bodies`."""
    print("\nfluid: the per-machine limiter and the entrained weight")
    from dytiscidae.physics.fluid import finish_bodies
    fb = np.array([[0, 0, 0, 0, 0, 0], [30.0, 0, 40.0, 1.0, 2.0, 3.0], [0, 60.0, 80.0, 0, 0, 0.5]])
    fs = np.array([0.0, 50.0, 100.0])
    ref = fb.copy()
    clamped = finish_bodies(fb, fs, np.zeros(3), 75.0)
    check("above the limit every body is scaled by the same limit / sum|F|",
          clamped and np.allclose(fb, ref * 0.5), f"{fb[1]}")
    fb2 = np.zeros((2, 6))
    finish_bodies(fb2, np.zeros(2), np.array([0.0, 2.0]), 1e9)
    check("the entrained fluid's weight is cancelled at the centre of mass: force, no torque",
          abs(fb2[1, 2] - 2.0 * 9.80665) < 1e-12 and not fb2[:, 3:].any())


def test_the_inflow_has_both_limits() -> None:
    """F-13: Glauert's disc is Rankine-Froude in hover and lifting-line's
    induced angle in forward flight."""
    print("\nfluid: the momentum inflow")
    from dytiscidae.physics.fluid import InducedFlow
    f = InducedFlow(span=1.0)
    T, rho = 10.0, 1.225
    for _ in range(4000):
        w = f.update(np.array([0.0, 0.0, T]), np.zeros(3), rho, 0.004)
    want = math.sqrt(T / (2 * rho * f.area))
    check("hover: w = sqrt(T / 2 rho A)", abs(-w[2] / want - 1.0) < 1e-3,
          f"{-w[2]:.4f} against {want:.4f} m/s")
    f = InducedFlow(span=1.0)
    V = 20.0
    for _ in range(4000):
        w = f.update(np.array([0.0, 0.0, T]), np.array([-V, 0.0, 0.0]), rho, 0.004)
    want = T / (2 * rho * f.area * V)
    check("forward flight: w = T / (2 rho A V), lifting-line's induced velocity",
          abs(-w[2] / want - 1.0) < 1e-3, f"{-w[2]:.5f} against {want:.5f} m/s")


def test_the_circulation_has_a_history() -> None:
    """F-02 and F-13: Wagner's half at the start, the LEV's two chords, and the
    revolving wing's Rossby number."""
    print("\nfluid: the unsteady history")
    from dytiscidae.physics.fluid import UnsteadyState
    st = UnsteadyState(1)
    a = np.array([0.1]); rev = np.array([False]); U = np.array([10.0]); c = np.array([0.1])
    om = np.zeros((1, 3)); sh = np.array([[0.0, 1.0, 0.0]])
    ae0, lev0 = st.update(a, rev, U, c, om, sh, 1e-6)
    check("an impulsively started strip carries half its incidence (Wagner)",
          abs(ae0[0] / a[0] - 0.5) < 1e-3, f"alpha_e / alpha = {ae0[0] / a[0]:.4f}")
    for _ in range(2000):
        ae, lev = st.update(a, rev, U, c, om, sh, 0.004)
    check("and all of it after many chords, with the translating LEV shed",
          abs(ae[0] / a[0] - 1.0) < 1e-3 and lev[0] == 0.0, f"{ae[0] / a[0]:.4f}, lev {lev[0]}")
    ae, lev = st.update(a, np.array([True]), U, c, om, sh, 0.004)
    # 0.4 chords are travelled in the reversal step itself, so a little over
    # half: 0.577.
    check("a reversal restarts both", lev[0] == 1.0 and ae[0] / a[0] < 0.6,
          f"lev {lev[0]}, alpha_e / alpha {ae[0] / a[0]:.3f}")
    st2 = UnsteadyState(1)
    for _ in range(2000):
        _, lev = st2.update(a, rev, U, c, np.array([[0.0, 0.0, 40.0]]), sh, 0.004)
    check("a strip revolving at Ro = U/(w c) = 2.5 keeps its LEV however far it goes",
          lev[0] == 1.0, f"lev {lev[0]}")


def test_the_model_reproduces_the_robofly() -> None:
    """ROADMAP AF: a revolving wing at Re 136 against Dickinson et al. 1999."""
    print("\nfluid: the robofly")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "robofly", Path(__file__).resolve().parents[1] / "experiments/robofly/run.py")
    rf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rf)
    rows = []
    for a in (9.0, 27.0, 45.0, 63.0, 81.0):
        cl, cd = rf.revolve(a)
        clr, cdr = rf.dickinson(a)
        rows.append((cl - float(clr), cd - float(cdr)))
    e_cl = math.sqrt(sum(r[0] ** 2 for r in rows) / len(rows))
    e_cd = math.sqrt(sum(r[1] ** 2 for r in rows) / len(rows))
    check("CL within 0.25 rms of the robofly (it was 0.47 before the LEV was keyed right)",
          e_cl < 0.25, f"rms {e_cl:.3f}")
    check("CD within 0.20 rms (it was 0.68)", e_cd < 0.20, f"rms {e_cd:.3f}")


def test_a_drop_and_recovery_is_not_holding_height() -> None:
    """ROADMAP AE: the air height term reads the worst second, not only the
    endpoints; and curriculum stage 1 counts air progress only while airborne."""
    print("\nair score: a drop and a recovery is not holding height")
    from types import SimpleNamespace

    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.tasks import CRUISE, Phase, TaskSchedule
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.evolution.curriculum import stage_score

    env = TriphibianEnv(build(BODY_PLANS["gannet"]()))
    # This is about the shape of the trajectory, so the level-flight gate (AD)
    # is held open; `test_holding_height_is_flight_only_if_the_actuators_can`
    # is the gate's own test.
    env.level_margin = lambda: 1.0
    dt = env.timestep
    n = int(8.0 / dt)
    sched = TaskSchedule("air", (Phase(CRUISE, 0.0, heading=0.0, speed=10.0),
                                 Phase(CRUISE, 0.5, heading=0.0, speed=10.0)))
    xy = np.zeros((n, 2))
    xy[:, 0] = 10.0 * dt * np.arange(n)
    t = np.arange(n) * dt

    def hold(clr):
        env._active_task = sched
        try:
            return env._task_scores(Domain.AIR, -clr, clr, xy, n,
                                    airborne=np.ones(n, bool))["measurements"]["height_hold"]
        finally:
            env._active_task = None

    level = hold(np.full(n, 20.0))
    # 5 m down and back up in each half, the shape of arch42's tumblers
    dip = 20.0 - 5.0 * np.sin(np.pi * np.clip((t % 4.0) - 1.5, 0.0, 1.0)) ** 2
    dipped = hold(dip)
    check("a body that drops 5 m and climbs back scores well below one that holds",
          dipped < 0.5 * level, f"{dipped:.3f} against {level:.3f}")

    def result(progress, airborne):
        m = {"cruise_progress": progress}
        if airborne is not None:
            m["airborne_fraction"] = airborne
        return SimpleNamespace(segments={"air": SimpleNamespace(competence=0.0, measurements=m)})

    fell = stage_score(1, result(0.8, 0.0))
    flew = stage_score(1, result(0.8, 1.0))
    check("stage 1 does not pay air progress made after falling into the sea",
          fell == 0.0 and abs(flew - 0.8) < 1e-12, f"fell {fell}, flew {flew}")


def test_holding_height_is_flight_only_if_the_actuators_can() -> None:
    """ROADMAP AD: the air task's flight term, and the ladder's flight rungs,
    are gated on `level_margin >= LEVEL_GATE`; the glide term is not."""
    print("\nair score: the level-flight gate")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.tasks import CRUISE, Phase, TaskSchedule
    from dytiscidae.envs.triphibian import LEVEL_GATE, Domain, TriphibianEnv
    from dytiscidae.evolution.judge import LADDER, rung_reached

    env = TriphibianEnv(build(BODY_PLANS["gannet"]()))
    dt = env.timestep
    n = int(8.0 / dt)
    sched = TaskSchedule("air", (Phase(CRUISE, 0.0, heading=0.0, speed=10.0),
                                 Phase(CRUISE, 0.5, heading=0.0, speed=10.0)))
    xy = np.zeros((n, 2))
    xy[:, 0] = 10.0 * dt * np.arange(n)
    level = np.full(n, 20.0)
    gliding = 20.0 - 2.0 * dt * np.arange(n)          # 2 m/s down: a glide

    def hold(clr, margin):
        env.level_margin = lambda: margin
        env._active_task = sched
        try:
            return env._task_scores(Domain.AIR, -clr, clr, xy, n,
                                    airborne=np.ones(n, bool))["measurements"]["height_hold"]
        finally:
            env._active_task = None

    flies, cannot, unmeasured = (hold(level, LEVEL_GATE + 0.1),
                                 hold(level, LEVEL_GATE - 0.1), hold(level, None))
    check("holding height pays flight only to a machine whose actuators can hold it",
          flies > 0.9 and cannot < 0.5 * flies,
          f"{flies:.3f} over the gate, {cannot:.3f} under it")
    check("and a margin the rig could not measure does not clear the gate",
          unmeasured == cannot, f"{unmeasured:.3f} against {cannot:.3f}")
    g_over, g_under = hold(gliding, LEVEL_GATE + 0.1), hold(gliding, LEVEL_GATE - 0.1)
    check("a glide still scores under the gate: it is a capability, not flight",
          g_under > 0.0 and abs(g_over - g_under) < 1e-12,
          f"{g_under:.3f} under, {g_over:.3f} over")

    names = [r[0] for r in LADDER["air"]]
    top = {"lift_margin": 5.0, "airborne_fraction": 1.0, "sink_rate": -1.0,
           "thrust_margin": 1.0, "station_keeping": 1.0, "turn_response": 1.0}
    check("the air ladder stops at `flies_level` without a level margin",
          rung_reached("air", top) == names.index("flies_level"),
          f"rung {rung_reached('air', top)} of {names}")
    check("and under the gate, and climbs past it over the gate",
          rung_reached("air", {**top, "level_margin": LEVEL_GATE - 0.01})
          == names.index("flies_level")
          and rung_reached("air", {**top, "level_margin": LEVEL_GATE}) == len(names))
    check("the ladder's rung and the task's gate are the same number",
          dict((r[0], r[2]) for r in LADDER["air"])["flies_level"] == LEVEL_GATE)


def test_a_transition_can_start_back_from_its_interface() -> None:
    """ROADMAP Y/O: a probe's start steps back from the interface, the air
    launch steps down, and 0 / None are the placements every run used."""
    print("\ntransitions: starting back from the interface")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.transitions import _place_for, run_transition
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    env = TriphibianEnv(build(BODY_PLANS["beetle"]()), seed=0)

    def at(kind, back):
        _place_for(env, kind, back)
        return env.data.qpos[:3].copy(), float(env.clearance())

    (a0, _), (a3, _) = at("air_to_water", 0.0), at("air_to_water", 3.0)
    (w0, _), (w3, _) = at("water_to_air", 0.0), at("water_to_air", 3.0)
    (s0, _), (s3, _) = at("water_to_land", 0.0), at("water_to_land", 3.0)
    (l0, c0), (l3, c3) = at("land_to_water", 0.0), at("land_to_water", 3.0)
    check("air_to_water starts higher, water_to_air deeper, by the distance asked",
          abs((a3[2] - a0[2]) - 3.0) < 1e-9 and abs((w0[2] - w3[2]) - 3.0) < 1e-9,
          f"air {a0[2]:.2f} -> {a3[2]:.2f}, water {w0[2]:.2f} -> {w3[2]:.2f}")
    check("water_to_land starts further seaward, land_to_water further up the beach",
          abs((s0[0] - s3[0]) - 3.0) < 1e-9 and abs((l3[0] - l0[0]) - 3.0) < 1e-9,
          f"x {s0[0]:.2f} -> {s3[0]:.2f}; {l0[0]:.2f} -> {l3[0]:.2f}")
    check("and up the beach it is set down on the ground, not inside it",
          c3 > -0.01, f"clearance {c3:.3f} (at the old start {c0:.3f})")

    env.air_launch_height = 12.0
    env.reset(Domain.AIR, randomise=False)
    z12 = float(env.data.qpos[2])
    env.air_launch_height = None
    env.reset(Domain.AIR, randomise=False)
    check("the air launch height is honoured, and None is the 30 m spawn",
          z12 == 12.0 and float(env.data.qpos[2]) == TriphibianEnv.SPAWN[Domain.AIR][2],
          f"{z12} / {float(env.data.qpos[2])}")

    tr = run_transition(env, "water_to_air", Controller(params=env.cpg.base),
                        duration=0.2, back=2.0)
    check("a transition records where it started", tr.start_back == 2.0,
          f"{tr.start_back}")


def test_the_level_margin_measures_what_the_actuators_deliver() -> None:
    """ROADMAP AG: min(<Fz>/W, 1 + <Fx>/W) on a rig, with the real actuators."""
    print("\nair: the level-flight margin")
    from dytiscidae.core.bodyplans import BODY_PLANS, REFERENCE_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import TriphibianEnv

    quad = TriphibianEnv(build(REFERENCE_PLANS["quad"]()), seed=0)
    j0 = quad.budget.actuator_j
    lq = quad.level_margin()
    check("measuring it does not spend the battery",
          quad.budget.actuator_j == j0, f"{j0} -> {quad.budget.actuator_j}")
    eel = TriphibianEnv(build(BODY_PLANS["eel"]()), seed=0).level_margin()
    gannet = TriphibianEnv(build(BODY_PLANS["gannet"]()), seed=0).level_margin()
    check("a multirotor at its rest throttle nearly holds itself up; a wingless eel does not",
          lq is not None and eel is not None and lq > 0.7 and eel < 0.2,
          f"quad {lq:.3f}, eel {eel:.3f}")
    check("and a glider reads short of 1: it has lift and no thrust",
          gannet is not None and 0.5 < gannet < 1.0, f"gannet {gannet:.3f}")


def test_the_rotor_table_is_the_rotor_model() -> None:
    """The lookup table `RotorSet` uses against `bemt` it is built from."""
    print("\nrotor: the lookup table")
    from dytiscidae.physics.medium import AIR, SEAWATER
    from dytiscidae.physics.rotor import RotorSpec, bemt, rotor_forces
    s = RotorSpec(radius=0.127, pitch=4.7 * 0.0254)
    et, eq = 0.0, 0.0
    for om in (250.0, 450.0, 700.0, 1100.0):
        T0, Q0 = bemt(s, om, 0.0, 0.0, AIR.rho, AIR.mu)
        for vax in (0.0, 1.5, 4.0, 7.0, 12.0):
            for vip in (0.0, 2.0, 5.0, 10.0):
                a = bemt(s, om, vax, vip, AIR.rho, AIR.mu)
                b = rotor_forces(s, om, vax, vip, 0.0, AIR, SEAWATER)
                et = max(et, abs(a[0] - b[0]) / T0)
                eq = max(eq, abs(a[1] - b[1]) / Q0)
    check("thrust within 5% of static thrust, torque within 15% of static torque, off-grid",
          et < 0.05 and eq < 0.15, f"worst {et:.3f} thrust, {eq:.3f} torque")


def test_the_search_can_build_rotorcraft() -> None:
    """ROADMAP AI: `mut_rotor` adds, removes and retunes, and what it makes builds."""
    print("\nsearch: rotorcraft are reachable")
    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.genome import MUTATION_OPERATORS, mut_rotor
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import TriphibianEnv

    rng = np.random.default_rng(3)
    g = BODY_PLANS["beetle"]()
    check("the operator is registered", MUTATION_OPERATORS.get("rotor") is mut_rotor)
    fired = mut_rotor(g, rng)
    n1 = sum(p.rotor_radius > 1e-3 for p in g.parts)
    check("on a design with no propeller it adds one", fired and n1 == 1, f"{n1}")
    env = TriphibianEnv(build(g), seed=0)
    check("and the design builds with a spinning rotor on a speed servo",
          env.rotors.n >= 1 and any(a.endswith("_r") for a in env.act_names),
          f"{env.rotors.n} rotors, actuators {env.act_names[-2:]}")
    counts = []
    for _ in range(60):
        mut_rotor(g, rng)
        counts.append(sum(p.rotor_radius > 1e-3 for p in g.parts))
    check("repeated, it both adds and removes", max(counts) > 1 and min(counts) < max(counts),
          f"rotor count ranged {min(counts)}-{max(counts)}")


def test_the_rotor_batch_is_the_per_rotor_loop() -> None:
    """`RotorBatch` steps every rotor of a shard as one array computation
    (2026-10-03: the per-rotor loop was 198 us a rotor-step, 82% of a
    rotor-heavy evaluation).  It must be that loop to the bit -- thrust, torque,
    the damping split and the end-of-step speed -- with a different table per
    rotor, rotors in air, in water and across the surface, idle ones and
    reversed ones, on a calm sea and on waves, one machine and two at once."""
    print("\nrotor: the vectorised rotors are the per-rotor loop")
    from dytiscidae.core.bodyplans import REFERENCE_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.physics.medium import SeaState
    from dytiscidae.physics.rotor import RotorBatch, RotorSet, RotorSpec

    rng = np.random.default_rng(11)

    def machine(sea, k0):
        env = TriphibianEnv(build(REFERENCE_PLANS["quad"]()), seed=0, sea_state=sea)
        env.reset(Domain.WATER, randomise=False)
        names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, b)
                 for b in env.rotors.body]
        # Four different rotors, so a lookup that reads one rotor's table
        # for another's cannot agree by accident.
        specs = {nm: RotorSpec(radius=0.08 + 0.03 * ((j + k0) % 4),
                               pitch=0.06 + 0.025 * ((j + 2 * k0) % 3),
                               blades=2 + (j % 2), handed=(1 if j % 2 else -1))
                 for j, nm in enumerate(names)}
        env.rotors = RotorSet(env.model, specs)
        return env

    def place(env, depth_of_first):
        d, m = env.data, env.model
        d.qpos[3:7] = [math.cos(0.2), math.sin(0.2), 0.0, 0.0]       # rolled 23 deg
        mujoco.mj_forward(m, d)
        z = d.xipos[env.rotors.body, 2]
        d.qpos[2] += -depth_of_first - z[0]
        d.qvel[:] = rng.normal(0.0, 2.0, m.nv)
        for j, k in enumerate(env.rotors.dof):
            d.qvel[k] = (0.0, -450.0, 700.0, 300.0)[j % 4] * (1.0 + 0.1 * rng.random())
        mujoco.mj_forward(m, d)
        d.xfrc_applied[:] = rng.normal(0.0, 1.0, d.xfrc_applied.shape)
        d.qfrc_applied[:] = rng.normal(0.0, 1.0, m.nv)

    def state(env):
        return (env.data.xfrc_applied.copy(), env.data.qfrc_applied.copy(),
                env.model.dof_damping.copy())

    def restore(env, s):
        env.data.xfrc_applied[:], env.data.qfrc_applied[:], env.model.dof_damping[:] = s

    worst, cases, unequal, fracs = 0.0, 0, 0, []
    for sea in (None, SeaState(amplitude=0.3, period=2.0, wavelength=5.0, direction=0.6)):
        envs = [machine(sea, 0), machine(sea, 1)]
        for depth in (0.004, -0.5, 0.3, 0.0):
            for env in envs:
                place(env, depth)
            fracs += [float(f) for env in envs for f in env.medium.properties(
                env.data.xipos[env.rotors.body], np.full(env.rotors.n, 0.01),
                env.data.time)[2]]
            before = [state(e) for e in envs]
            ref = []
            for e, s in zip(envs, before):
                tot = e.rotors.apply_per_rotor(e.model, e.data, e.medium, e.data.time)
                ref.append((*state(e), e.rotors.last_thrust.copy(),
                            e.rotors.last_torque.copy(), np.array([tot])))
                restore(e, s)
            # One machine at a time (the single path), then both as one batch.
            one = []
            for e in envs:
                tot = e.rotors.apply(e.model, e.data, e.medium, e.data.time)
                one.append((*state(e), e.rotors.last_thrust.copy(),
                            e.rotors.last_torque.copy(), np.array([tot])))
            for e, s in zip(envs, before):
                restore(e, s)
            batch = RotorBatch([e.rotors for e in envs])
            tots = batch.apply([(i, e.model, e.data, e.medium) for i, e in enumerate(envs)],
                               envs[0].data.time)
            both = [(*state(e), e.rotors.last_thrust.copy(), e.rotors.last_torque.copy(),
                     np.array([tots[i]])) for i, e in enumerate(envs)]
            for got in (one, both):
                for g, r in zip(got, ref):
                    cases += 1
                    for a, b in zip(g, r):
                        if not np.array_equal(a, b):
                            unequal += 1
                            worst = max(worst, float(np.nanmax(np.abs(a - b))))
            for e, s in zip(envs, before):
                restore(e, s)
    fracs = np.array(fracs)
    check("the fixture has rotors in air, in water and across the surface",
          (fracs == 0).any() and (fracs == 1).any() and ((fracs > 0) & (fracs < 1)).any(),
          f"submerged fractions {np.round(np.unique(fracs), 3).tolist()}")
    check("every force, torque, damping, thrust and total is the loop's to the bit",
          cases == 32 and unequal == 0,
          f"{cases} cases, {unequal} arrays differ, worst difference {worst:.3g}")
    check("and the rotors make force: the comparison is not of zeros",
          np.abs(ref[0][3]).max() > 0.1 and np.abs(ref[0][4]).max() > 0.0,
          f"thrusts {np.round(ref[0][3], 3).tolist()}")

    # The tables themselves: `rotor_table` builds its grid through
    # `bemt_many` (28x faster, 2026-10-03); the grid must be the scalar
    # loop's, and `bemt_many` must be `bemt` off the grid too, windmilling
    # and edgewise included.
    from dytiscidae.physics.medium import AIR, SEAWATER
    from dytiscidae.physics.rotor import _build_table, bemt, bemt_many
    sp = RotorSpec(radius=0.09, pitch=0.11, blades=3)
    same = all(np.array_equal(x, y) for fl in (AIR, SEAWATER)
               for x, y in zip(_build_table(sp, fl.rho, fl.mu, scalar=True),
                               _build_table(sp, fl.rho, fl.mu)))
    om = rng.uniform(50.0, 1500.0, 40) * rng.choice([-1.0, 1.0], 40)
    vax, vip = rng.normal(0.0, 6.0, 40), np.abs(rng.normal(0.0, 4.0, 40))
    T, Q = bemt_many(sp, om, vax, vip, SEAWATER.rho, SEAWATER.mu)
    one = [bemt(sp, o, a, b, SEAWATER.rho, SEAWATER.mu) for o, a, b in zip(om, vax, vip)]
    check("the lookup tables, and bemt at 40 random points, are the scalar loop's to the bit",
          same and np.array_equal(T, [x[0] for x in one]) and np.array_equal(Q, [x[1] for x in one]),
          f"tables {same}, worst thrust difference "
          f"{np.abs(T - np.array([x[0] for x in one])).max():.3g}")


def test_a_propeller_can_go_under_water() -> None:
    """arch43 was stopped at gen 3: 5.1% of rollouts diverged, rotor designs
    crossing into water.  A rotor spinning at 550 rad/s put under water reached
    |qvel| 787,158 -- its drag torque, ~1000x in water, meets a tiny inertia.
    Backward Euler on the drag and an implicit split keep it finite."""
    print("\nrotor: a spinning propeller put under water")
    from dytiscidae.core.bodyplans import REFERENCE_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    env = TriphibianEnv(build(REFERENCE_PLANS["quad"]()), seed=0)
    env.reset(Domain.WATER, randomise=False)
    for k in env.rotors.dof:
        env.data.qvel[k] = 550.0
    w0 = int(env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    worst = 0.0
    for _ in range(int(1.0 / env.timestep)):
        env.step(env.cpg.command(env.cpg.base, env.data.time))
        worst = max(worst, float(np.abs(env.data.qvel).max()))
    bad = int(env.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) - w0
    check("no divergence, and nothing faster than the spin it started with",
          bad == 0 and worst < 600.0, f"{bad} bad-qacc, largest |qvel| {worst:.1f}")


def test_the_batched_power_budget_is_the_power_budget() -> None:
    """`BatchedPower` charges a whole batch in one pass (2026-09-27, 7% of a
    batched evaluation was per-machine numpy dispatch in `PowerBudget.step`).
    It must be the same model to the bit, or the two evaluation paths drift."""
    print("\nenergy: the batched power budget is the power budget")
    import copy

    from dytiscidae.core.bodyplans import BODY_PLANS
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import TriphibianEnv
    from dytiscidae.physics.energy import BatchedPower

    envs = [TriphibianEnv(build(BODY_PLANS[k]())) for k in ("beetle", "teal", "ray")]
    envs[1].budget.battery.energy_j = 20.0          # goes flat mid-fixture
    ref = [copy.deepcopy(e.budget) for e in envs]
    rng = np.random.default_rng(7)
    power = BatchedPower(envs)
    active = np.ones(len(envs), dtype=bool)
    alive_ref = [True] * len(envs)
    for _ in range(50):
        for i, e in enumerate(envs):
            e.data.actuator_force[:] = rng.normal(0.0, 3.0, e.model.nu)
            e.data.actuator_velocity[:] = rng.normal(0.0, 20.0, e.model.nu)
            if alive_ref[i]:
                alive_ref[i] = ref[i].step(np.abs(e.data.actuator_force),
                                           np.abs(e.data.actuator_velocity), e.timestep)
        power.step(envs, active)
    fields = ("total_j", "actuator_j", "avionics_j", "max_overload", "t")
    same = all(getattr(e.budget, f) == getattr(r, f) for e, r in zip(envs, ref) for f in fields)
    same &= all(e.budget.battery.energy_j == r.battery.energy_j for e, r in zip(envs, ref))
    check("every field bit-identical to PowerBudget.step over 50 steps", same,
          f"{[(e.budget.total_j, r.total_j) for e, r in zip(envs, ref)]}")
    check("and the fixture draws real power", all(r.actuator_j > 1.0 for r in ref),
          f"{[r.actuator_j for r in ref]}")
    check("a flat pack leaves the batch as it left the loop, and one did",
          [bool(a) for a in active] == alive_ref and not alive_ref[1] and alive_ref[0],
          f"{list(active)} vs {alive_ref}")


def main() -> int:
    print("=" * 68)
    print("Dytiscidae physics verification")
    print("=" * 68)
    run_all([
        test_a_strip_moves_with_its_hinge,
        test_reversed_flow_reverses_the_force,
        test_pitching_nose_up_adds_lift,
        test_the_servo_reaches_a_fast_stroke,
        test_a_frequency_change_does_not_jump_the_stroke,
        test_a_fall_scores_no_flight,
        test_the_rotor_model_matches_a_measured_propeller,
        test_a_universal_joint_flaps_and_feathers,
        test_the_wagner_slam_pressure,
        test_lift_and_drag_are_integrated_implicitly,
        test_the_limiter_and_the_weight_cancellation,
        test_the_inflow_has_both_limits,
        test_the_circulation_has_a_history,
        test_the_model_reproduces_the_robofly,
        test_a_drop_and_recovery_is_not_holding_height,
        test_holding_height_is_flight_only_if_the_actuators_can,
        test_the_level_margin_measures_what_the_actuators_deliver,
        test_a_transition_can_start_back_from_its_interface,
        test_the_rotor_table_is_the_rotor_model,
        test_the_search_can_build_rotorcraft,
        test_the_rotor_batch_is_the_per_rotor_loop,
        test_a_propeller_can_go_under_water,
        test_the_batched_power_budget_is_the_power_budget,
        test_each_phase_is_scored_on_its_own_purpose,
        test_the_first_air_reset_is_like_every_other,
        test_chattering_commands_are_measured,
        test_the_gait_gain_can_stop_a_machine,
        test_a_film_is_the_evaluation,
        test_lift_sign_and_magnitude,
        test_buoyancy,
        test_bluff_drag_is_orientation_dependent,
        test_generated_bodies_reach_the_fluid,
        test_land_domain_is_reachable,
        test_a_bilateral_pair_flaps_together_rather_than_rolling,
        test_each_plan_moves_the_way_it_says_it_does,
        test_a_surface_can_be_told_to_hold_still,
        test_the_seeds_include_something_that_flies,
        test_the_render_shows_the_shape_the_solver_reads,
        test_machine_does_not_collide_with_itself,
        test_air_segment_can_be_scored,
        test_truncated_episodes_cannot_score,
        test_added_mass_is_anisotropic,
        test_a_mirrored_wing_is_a_mirror_image,
        test_flight_is_expressible_in_the_genome,
        test_bodies_generate_lift_and_a_pitching_moment,
        test_series_elasticity_needs_a_compliant_drive,
        test_flight_is_measured_against_the_ground_not_the_waterline,
        test_entry_shock_is_hydrodynamic_not_a_speed_limit,
        test_free_surface_continuity,
        test_added_mass_dominates_in_water,
        test_lev_extends_stall,
        test_energy_budget_matches_hand_calculation,
        test_actuator_never_regenerates,
        test_structure_rejects_impossible_wings,
        test_wave_field,
        test_jet_thrust_matches_momentum_flux,
        test_a_failing_sweep_gates_only_the_last_swimmer,
        test_the_power_budget_vectorises_without_changing_the_answer,
        test_training_states_are_a_distribution_not_a_pose,
        test_the_controller_senses_what_the_mission_scores,
        test_flap_frequency_is_commandable_in_the_loop,
        test_an_auto_reset_rollout_is_not_trusted,
        test_the_air_score_measures_flight,
        test_takeoff_is_measured_where_the_machine_starts_on_the_ground,
        test_depth_is_a_gain_not_a_spawn,
    ])
    return report("all physics checks passed")


if __name__ == "__main__":
    raise SystemExit(main())
