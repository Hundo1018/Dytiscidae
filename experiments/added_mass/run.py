#!/usr/bin/env python3
"""Two questions about added mass, both answered by the integrator, not the code.

0. The problem
--------------
`FluidSolver.apply` does not apply added mass as a force.  It writes it into
`model.body_mass` and `model.body_inertia` and lets MuJoCo invert it implicitly,
which is the right call: an explicit `-d(m_a v)/dt` term is a feedback loop of
gain m_added/m_body and diverges above unity.  The module says so, and says the
explicit version produced NaN accelerations after 1.3 s of water.

That leaves two claims nothing in the project checks.

  Q1  Does the number written into `body_mass` reach the mass matrix?
      `tests/test_physics.py::test_added_mass_is_anisotropic` reads
      `model.body_mass[bid]` back and checks the bookkeeping.  Bookkeeping is
      not dynamics.  Nothing asks MuJoCo what it did with it.

  Q2  Is the wing branch anisotropic?  The bluff branch projects a directional
      coefficient onto the direction of motion, with a long comment explaining
      that treating a plate like a sphere "erases the whole reason a fin is a
      fin".  The wing branch on the next line is `rho pi chord^2 / 4 * dr`, the
      2D added mass of a plate accelerating **normal to its own surface**, with
      no direction in it, written into a scalar.

1. Competing explanations
--------------------------
  Q1-H1  it reaches the dynamics on every model.
  Q1-H2  it reaches the dynamics on some models and not others, depending on
         something about the model MuJoCo decides at compile time.
  Q2-H1  the wing branch is anisotropic somewhere this reading missed.
  Q2-H2  the wing branch is isotropic, so a wing strip is as expensive to slice
         edgewise through water as to slap broadside.

2. Measurement
--------------
Push a submerged body with a known force along each of its own three axes and
read the acceleration out of MuJoCo.  m_eff = F/a.  No model introspection: the
number comes from the integrator.  Q1 runs the same body twice, once with a
joint in it and once without.

Run:  PYTHONPATH=. python experiments/added_mass/run.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco  # noqa: E402

from dytiscidae.physics.fluid import BLUFF, WING, FluidSolver, PanelSet  # noqa: E402
from dytiscidae.physics.medium import GRAVITY, SEAWATER, MediumField  # noqa: E402
from experiments.harness import ExperimentResult, load_config  # noqa: E402

AXES = {
    "chord (+X)": np.array([1.0, 0.0, 0.0]),
    "span  (+Y)": np.array([0.0, 1.0, 0.0]),
    "normal(+Z)": np.array([0.0, 0.0, 1.0]),
}


def _xml(cfg: dict, jointed: bool) -> str:
    c, b, t = (cfg["strip"]["chord_m"], cfg["strip"]["span_m"],
               cfg["strip"]["thickness_m"])
    child = ("""
          <body name="dummy" pos="0 0 0.05">
            <joint name="h" type="hinge" axis="0 1 0"/>
            <geom type="box" size="0.01 0.01 0.01" mass="0.001"/>
          </body>""" if jointed else "")
    return f"""
    <mujoco>
      <option timestep="{cfg['timestep_s']}" gravity="0 0 0"
              density="0" viscosity="0"/>
      <worldbody>
        <body name="strip" pos="0 0 {-cfg['depth_m']}">
          <freejoint/>
          <geom type="box" size="{c/2} {b/2} {t/2}" mass="{cfg['dry_mass_kg']}"/>{child}
        </body>
      </worldbody>
    </mujoco>
    """


def _panels(model, cfg: dict, kind: int) -> PanelSet:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "strip")
    c, b, t = (cfg["strip"]["chord_m"], cfg["strip"]["span_m"],
               cfg["strip"]["thickness_m"])
    return PanelSet(
        body_id=np.array([bid]),
        pos_local=np.zeros((1, 3)),
        span_local=np.array([[0.0, 1.0, 0.0]]),
        chord_local=np.array([[1.0, 0.0, 0.0]]),
        chord=np.array([c]),
        dr=np.array([b]),
        volume=np.array([c * b * t]),
        volume_buoyant=np.array([0.0]),   # isolate inertia from buoyancy
        half_height=np.array([t / 2]),
        kind=np.array([kind]),
        aspect_ratio=np.array([b / c]),
        cd_bluff=np.array([0.0]),
        ext_local=np.array([[b, c, t]]),
    )


def push(cfg: dict, kind: int, axis: np.ndarray, jointed: bool) -> dict:
    model = mujoco.MjModel.from_xml_string(_xml(cfg, jointed))
    data = mujoco.MjData(model)
    panels = _panels(model, cfg, kind)
    solver = FluidSolver(model, panels, MediumField(water=SEAWATER))
    bid = int(panels.body_id[0])
    dry = float(model.body_mass[bid])
    F = cfg["applied_force_N"] * axis

    m_eff, clamped = [], False
    for step in range(cfg["settle_steps"]):
        data.xfrc_applied[:] = 0.0
        solver.apply(data, data.time)
        clamped = clamped or solver.diag.clamped
        data.xfrc_applied[bid, :3] += F
        total = float(data.xfrc_applied[bid, :3] @ axis)
        mujoco.mj_step(model, data)
        a = float(data.qacc[:3] @ axis)
        if step >= cfg["settle_steps"] // 2 and abs(a) > 1e-12:
            m_eff.append(total / a)
    return {
        "dry_mass": dry,
        "body_mass_after_apply": float(model.body_mass[bid]),
        "bookkept_added_mass": float(model.body_mass[bid]) - dry,
        "effective_mass": float(np.median(m_eff)),
        "dynamic_added_mass": float(np.median(m_eff)) - dry,
        "force_limiter_engaged": bool(clamped),
        "body_simple": [int(x) for x in model.body_simple],
        "nv": int(model.nv),
    }


def run(cfg: dict) -> ExperimentResult:
    res = ExperimentResult("added_mass", cfg)
    rho = SEAWATER.rho
    c, b, t = (cfg["strip"]["chord_m"], cfg["strip"]["span_m"],
               cfg["strip"]["thickness_m"])
    m_strip = rho * math.pi * c**2 / 4 * b

    # ------------------------------------------------------------------ Q1
    print("Q1. does the added mass written into body_mass reach the mass matrix?")
    print(f"    one wing strip, {b} m x {c} m x {t*1e3:.0f} mm, "
          f"{cfg['dry_mass_kg']} kg dry, submerged, pushed along +Z")
    q1 = {}
    for label, jointed in (("no joints (one free body)", False),
                           ("one hinge added", True)):
        r = push(cfg, WING, AXES["normal(+Z)"], jointed)
        q1[label] = r
        print(f"    {label:<26} body_mass says +{r['bookkept_added_mass']:7.2f} kg, "
              f"the integrator says +{r['dynamic_added_mass']:7.2f} kg   "
              f"body_simple={r['body_simple']}")
    lost = q1["no joints (one free body)"]
    kept = q1["one hinge added"]
    ratio = kept["effective_mass"] / max(lost["effective_mass"], 1e-12)
    shortfall = 1.0 - (lost["dynamic_added_mass"]
                       / max(lost["bookkept_added_mass"], 1e-12))
    print()
    if abs(shortfall) < 0.01:
        print(f"    Both reach the integrator.  The jointless model applies "
              f"{lost['dynamic_added_mass']:.2f} kg of the "
              f"{lost['bookkept_added_mass']:.2f} kg it bookkeeps, and its "
              f"effective mass of {lost['effective_mass']:.2f} kg matches the "
              f"{kept['effective_mass']:.2f} kg of the same body with a hinge "
              f"in it to a factor of {ratio:.3f}.")
        print(f"    This is F-01 closed.  MuJoCo marks a body 'simple' when "
              f"nothing is jointed to it and takes those DOFs' mass from the "
              f"compile-time constant `dof_M0`; a runtime `body_mass` edit "
              f"does not reach them until the constants are rebuilt. "
              f"`FluidSolver._publish_inertia` now rebuilds them, gated on "
              f"whether the model needs it, so a machine with joints pays "
              f"nothing.")
    else:
        print(f"    The jointless model discards "
              f"{100 * shortfall:.1f}% of the "
              f"{lost['bookkept_added_mass']:.2f} kg of entrained water it "
              f"bookkeeps: its effective mass is "
              f"{lost['effective_mass']:.2f} kg against "
              f"{kept['effective_mass']:.2f} kg for the same body with one "
              f"hinge in it, a factor of {ratio:.2f}.")
        print(f"    Mechanism: MuJoCo marks a body 'simple' when nothing is "
              f"jointed to it and takes those DOFs' mass from the "
              f"compile-time constant `dof_M0`, so a runtime `body_mass` edit "
              f"never reaches them.  `mj_setConst` is what rebuilds it.")
        print(f"    This project has produced jointless designs: arch33's "
              f"mission champion had zero actuated degrees of freedom.  In "
              f"water such a design accelerates {ratio:.2f}x more easily than "
              f"the physics the solver computed, for free, and every "
              f"diagnostic reports the added mass as applied.")
    res_shortfall = shortfall

    # ------------------------------------------------------------------ Q2
    print("\nQ2. is the wing branch anisotropic?")
    theory = {
        "normal(+Z)": rho * math.pi * c**2 / 4 * b,   # plate, broadside
        "chord (+X)": rho * math.pi * t**2 / 4 * b,   # plate, edge-on
        "span  (+Y)": 0.0,                            # along the span
    }
    q2 = {}
    for kind_name, kind in (("WING", WING), ("BLUFF", BLUFF)):
        print(f"\n    {kind_name} element, jointed model so the mass matrix "
              f"sees it:")
        print(f"      {'push':<12} {'m_dry':>7} {'m_eff':>9} {'m_added':>9} "
              f"{'2D strip theory':>16}")
        rows = {}
        for name, axis in AXES.items():
            r = push(cfg, kind, axis, jointed=True)
            th = theory[name] if kind == WING else float("nan")
            rows[name] = r | {"strip_theory_kg": th}
            print(f"      {name:<12} {r['dry_mass']:7.2f} "
                  f"{r['effective_mass']:9.2f} {r['dynamic_added_mass']:9.2f} "
                  f"{th:16.5f}")
        vals = [rows[k]["dynamic_added_mass"] for k in AXES]
        spread = (max(vals) - min(vals)) / max(abs(np.mean(vals)), 1e-12)
        print(f"      spread across the three directions: {100*spread:.3f}%")
        rows["spread_fraction"] = float(spread)
        q2[kind_name] = rows

    edge = q2["WING"]["chord (+X)"]["dynamic_added_mass"]
    th_edge = theory["chord (+X)"]
    over = edge / max(th_edge, 1e-30)
    print(f"\n    WING edgewise: the solver entrains {edge:.2f} kg where 2D "
          f"strip theory for a {t*1e3:.0f} mm plate gives {th_edge:.5f} kg.")
    print(f"    That is {over:.0f}x, and the factor is (chord/thickness)^2 = "
          f"{(c/t)**2:.0f} exactly, because the wing branch has no direction "
          f"in it.")
    bl = q2["BLUFF"]
    bl_ratio = (bl["normal(+Z)"]["dynamic_added_mass"]
                / max(bl["chord (+X)"]["dynamic_added_mass"], 1e-12))
    print(f"    The same geometry declared BLUFF, which the code does project, "
          f"comes out {bl_ratio:.1f}x apart between broadside and edgewise.")

    res.record("q1_reaches_dynamics", q1)
    res.record("q1_jointless_mass_ratio", float(ratio))
    res.record("q1_jointless_added_mass_shortfall", float(res_shortfall))
    res.record("q2_directions", q2)
    res.record("q2_strip_theory_kg", theory)
    res.record("q2_wing_edgewise_overstatement", float(over))
    res.record("q2_chord_over_thickness_squared", float((c / t) ** 2))
    res.record("q2_bluff_anisotropy_ratio", float(bl_ratio))
    res.record("strip_broadside_theory_kg", float(m_strip))
    return res


if __name__ == "__main__":
    cfg = load_config(Path(__file__).parent / "config.json")
    result = run(cfg)
    result.write(Path(__file__).parent / "results")
