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
