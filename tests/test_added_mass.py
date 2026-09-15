"""Added mass has to reach the mass matrix, not just the bookkeeping.

`FluidSolver` folds entrained fluid into `model.body_mass` rather than applying
it as a force, which is right: an explicit `-d(m_a v)/dt` term is a feedback
loop of gain `m_a / m_body` and diverges above unity, and in water that ratio is
routinely three or more.

`tests/test_physics.py::test_added_mass_is_anisotropic` reads
`model.body_mass[bid]` back afterwards and checks the number. That is the
bookkeeping. These tests check the *dynamics*: whether

    M(q) qdd + C(q, qd) qd = F

actually carries it. It did not, for a machine with no joints -- MuJoCo marks
such a body `simple` and takes those DOFs' mass from `dof_M0`, a constant
computed when the model was compiled, so a runtime `body_mass` edit updated
`cinert` and never reached `M`. Measured: +32.20 kg bookkept, +0.00 kg applied.

Run:  PYTHONPATH=. python tests/test_added_mass.py
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

from dytiscidae.physics.fluid import WING, FluidSolver, PanelSet  # noqa: E402
from dytiscidae.physics.medium import GRAVITY, SEAWATER, MediumField  # noqa: E402

FAILURES: list[str] = []

CHORD, SPAN, THICK = 0.20, 1.0, 0.002
DRY = 12.0
DT = 0.001


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def _plate(jointed: bool, *, scale: float = 1.0):
    """One free body carrying one wing strip, optionally with a joint on it."""
    child = ('<body name="d" pos="0 0 0.05">'
             '<joint type="hinge" axis="0 1 0"/>'
             '<geom type="box" size="0.005 0.005 0.005" mass="1e-4"/></body>'
             if jointed else "")
    xml = f"""
    <mujoco>
      <option timestep="{DT}" gravity="0 0 0" density="0" viscosity="0"
              integrator="implicitfast"/>
      <worldbody>
        <body name="b" pos="0 0 -8"><freejoint/>
          <geom type="box" size="{CHORD/2} {SPAN/2} {THICK/2}" mass="{DRY}"/>
          {child}
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    panels = PanelSet(
        body_id=np.array([1]), pos_local=np.zeros((1, 3)),
        span_local=np.array([[0.0, 1.0, 0.0]]),
        chord_local=np.array([[1.0, 0.0, 0.0]]),
        chord=np.array([CHORD]), dr=np.array([SPAN]),
        volume=np.array([CHORD * SPAN * THICK]),
        volume_buoyant=np.array([0.0]),
        half_height=np.array([THICK / 2]), kind=np.array([WING]),
        aspect_ratio=np.array([SPAN / CHORD]), cd_bluff=np.array([0.0]),
        ext_local=np.array([[SPAN, CHORD, THICK]]))
    solver = FluidSolver(model, panels, MediumField(water=SEAWATER),
                         added_mass_scale=scale)
    return model, data, panels, solver


def _mass_matrix(model, data) -> np.ndarray:
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)
    return full


def _push(model, data, solver, axis: np.ndarray, force: float = 40.0) -> float:
    """Effective mass along `axis`, read out of the integrator."""
    vals = []
    for step in range(80):
        data.xfrc_applied[:] = 0.0
        solver.apply(data, data.time)
        # Gravity is off, so the solver's compensation for the weight MuJoCo
        # would otherwise apply to entrained fluid is unbalanced here.
        data.xfrc_applied[1, 2] -= solver.diag.added_mass * GRAVITY
        data.xfrc_applied[1, :3] += force * axis
        total = float(data.xfrc_applied[1, :3] @ axis)
        mujoco.mj_step(model, data)
        a = float(data.qacc[:3] @ axis)
        if step >= 40 and abs(a) > 1e-12:
            vals.append(total / a)
    return float(np.median(vals))


def test_the_mass_matrix_itself_carries_the_added_mass() -> None:
    """Not `body_mass`. `M` in `M qdd + C qd = F`, read with `mj_fullM`."""
    print("\nadded mass: the mass matrix carries it, for jointed and jointless")
    strip = SEAWATER.rho * math.pi * CHORD**2 / 4 * SPAN
    for jointed in (True, False):
        tag = "jointed  " if jointed else "jointless"
        model, data, panels, solver = _plate(jointed)
        mujoco.mj_forward(model, data)
        dry_M = _mass_matrix(model, data)[0, 0]
        data.xfrc_applied[:] = 0.0
        solver.apply(data, 0.0)
        mujoco.mj_forward(model, data)
        wet_M = _mass_matrix(model, data)[0, 0]
        check(f"{tag}: M[0,0] rises by the entrained mass",
              abs((wet_M - dry_M) - strip) < 1e-6 * strip,
              f"{dry_M:.3f} -> {wet_M:.3f} kg against a strip-theory "
              f"{strip:.3f} kg")
        check(f"{tag}: and the bookkeeping agrees with the matrix",
              abs(float(model.body_mass[1]) - DRY - strip) < 1e-9,
              "body_mass and M must not be allowed to disagree again")


def test_changing_the_added_mass_changes_the_trajectory() -> None:
    """Same body, same initial condition, same external force, two masses.

    The trajectory must differ, and by the amount the two masses imply. This is
    the check the old test could not make: it read a number back out of the
    model rather than asking the integrator what it did with it.
    """
    print("\nadded mass: scaling it changes the trajectory, by the right amount")
    strip = SEAWATER.rho * math.pi * CHORD**2 / 4 * SPAN
    for jointed in (True, False):
        tag = "jointed  " if jointed else "jointless"
        travelled = {}
        for scale in (0.0, 1.0):
            model, data, panels, solver = _plate(jointed, scale=scale)
            # Isolate inertia.  With drag left on, the lighter run reaches a
            # higher speed and loses more to it, so the distances stop obeying
            # the constant-force kinematics this check is built on.
            #
            # Note `c_rot` has to be zeroed by hand: the Kramer rotational term
            # is the one circulatory force `cd_scale` and `lift_scale` do not
            # reach, which is MATH_AUDIT F-07 and is why the auditor cannot
            # perturb it either.  It is zero here anyway -- the plate does not
            # rotate -- and is zeroed so this test does not quietly depend on
            # that staying true.
            solver.cd_scale = 0.0
            solver.lift_scale = 0.0
            solver.c_rot = np.zeros_like(solver.c_rot)
            for _ in range(600):
                data.xfrc_applied[:] = 0.0
                solver.apply(data, data.time)
                data.xfrc_applied[1, 2] -= solver.diag.added_mass * GRAVITY
                data.xfrc_applied[1, 2] += 40.0
                mujoco.mj_step(model, data)
            travelled[scale] = float(data.qpos[2] + 8.0)
        # Constant force from rest: s = F t^2 / (2 m), so the ratio of
        # distances is the inverse ratio of the effective masses.
        expect = (DRY + 0.0) / (DRY + strip)
        got = travelled[1.0] / travelled[0.0]
        check(f"{tag}: distance ratio matches the mass ratio",
              abs(got / expect - 1.0) < 2e-3,
              f"{travelled[1.0]*1e3:.2f} mm against {travelled[0.0]*1e3:.2f} mm "
              f"-- ratio {got:.5f} against {expect:.5f}")
        check(f"{tag}: and the two trajectories are measurably different",
              abs(travelled[1.0] - travelled[0.0]) > 1e-3,
              f"{abs(travelled[1.0]-travelled[0.0])*1e3:.2f} mm apart after 0.6 s")


def test_the_effective_mass_matches_strip_theory() -> None:
    """`m_eff = F/a` from the integrator, against the closed form."""
    print("\nadded mass: the effective mass the integrator uses")
    strip = SEAWATER.rho * math.pi * CHORD**2 / 4 * SPAN
    for jointed in (True, False):
        tag = "jointed  " if jointed else "jointless"
        model, data, panels, solver = _plate(jointed)
        m_eff = _push(model, data, solver, np.array([0.0, 0.0, 1.0]))
        check(f"{tag}: pushing normal to the plate",
              abs(m_eff - (DRY + strip)) < 0.02 * (DRY + strip),
              f"{m_eff:.3f} kg against {DRY + strip:.3f} kg")


def test_the_model_constants_are_restored_on_reset() -> None:
    """A previous episode's entrained water must not survive into the next."""
    print("\nadded mass: reset restores the dry inertia and the constants")
    for jointed in (True, False):
        tag = "jointed  " if jointed else "jointless"
        model, data, panels, solver = _plate(jointed)
        mujoco.mj_forward(model, data)
        dry_M = _mass_matrix(model, data)[0, 0]
        dry_M0 = float(model.dof_M0[0])
        data.xfrc_applied[:] = 0.0
        solver.apply(data, 0.0)
        solver.reset()
        mujoco.mj_forward(model, data)
        check(f"{tag}: M returns to the dry value",
              abs(_mass_matrix(model, data)[0, 0] - dry_M) < 1e-12,
              f"{_mass_matrix(model, data)[0, 0]:.6f} against {dry_M:.6f}")
        check(f"{tag}: and so does dof_M0",
              abs(float(model.dof_M0[0]) - dry_M0) < 1e-12)


def test_the_rebuild_only_runs_where_it_is_needed() -> None:
    """`mj_setConst` is not free, so it is gated on detection.

    No panel-carrying body of any seed plan is `simple`, so a real machine must
    pay nothing for this.
    """
    print("\nadded mass: the constant rebuild is gated on the model needing it")
    _, _, _, jointed = _plate(True)
    _, _, _, jointless = _plate(False)
    check("a jointed model does not trigger the rebuild",
          jointed._needs_const is False)
    check("a jointless one does", jointless._needs_const is True)

    from dytiscidae.core import bodyplans
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv

    for name in ("beetle", "gannet", "medusa"):
        env = TriphibianEnv(build(getattr(bodyplans, name)()), seed=1)
        env.reset(Domain.WATER, randomise=False)
        check(f"seed plan {name} does not trigger it",
              env.solver._needs_const is False,
              f"{int(np.sum(env.model.body_simple[np.unique(env.panels.body_id)]))} "
              f"of its {len(np.unique(env.panels.body_id))} panel-carrying "
              f"bodies are simple")


def test_the_scratch_data_protects_the_simulation_state() -> None:
    """`mj_setConst(model, data)` resets `data` to qpos0.  It must not get ours.

    This is why the rebuild is handed a throwaway `MjData`: called on the live
    one it would teleport the machine back to its spawn every step, silently.
    """
    print("\nadded mass: the constant rebuild does not touch the live state")
    model, data, panels, solver = _plate(False)
    data.qpos[:3] = [1.5, -2.0, 3.0]
    data.qvel[:3] = [0.3, 0.0, -0.2]
    mujoco.mj_forward(model, data)
    q, v = data.qpos.copy(), data.qvel.copy()
    data.xfrc_applied[:] = 0.0
    solver.apply(data, 0.0)
    check("qpos survives the rebuild",
          float(np.max(np.abs(data.qpos - q))) < 1e-15,
          f"{q[:3]} -> {data.qpos[:3]}")
    check("and so does qvel", float(np.max(np.abs(data.qvel - v))) < 1e-15)

    # And the control: the live data would have been reset.
    scratch = mujoco.MjData(model)
    scratch.qpos[:3] = [1.5, -2.0, 3.0]
    mujoco.mj_setConst(model, scratch)
    check("a direct call on the live data would have reset it",
          float(np.max(np.abs(scratch.qpos[:3] - np.array([1.5, -2.0, 3.0])))) > 1e-6,
          "which is the trap this arrangement exists to avoid")


def main() -> int:
    test_the_mass_matrix_itself_carries_the_added_mass()
    test_changing_the_added_mass_changes_the_trajectory()
    test_the_effective_mass_matches_strip_theory()
    test_the_model_constants_are_restored_on_reset()
    test_the_rebuild_only_runs_where_it_is_needed()
    test_the_scratch_data_protects_the_simulation_state()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} added-mass checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("added mass reaches the mass matrix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
