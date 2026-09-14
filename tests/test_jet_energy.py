"""The jet's momentum flux and the work that pays for it must agree.

`JetSet` produces thrust `rho Q^2 / A` by expelling water.  Expelling water
costs work, and the only honest test of whether the model charges for it is to
compare the work it reports against the work the jet's own momentum flux
implies -- computed here from the definition, not from the module under test.

Run:  PYTHONPATH=. python tests/test_jet_energy.py
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

from dytiscidae.physics.jet import JetSet  # noqa: E402
from dytiscidae.physics.medium import SEAWATER, MediumField  # noqa: E402

FAILURES: list[str] = []

V0, SF, AREA = 0.004, 0.45, 0.0012
LO, HI = 0.0, 1.2
DT, FREQ, SECONDS = 0.001, 1.5, 4.0


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        FAILURES.append(name)


def _bell(*, depth: float = 5.0, integrator: str = "implicitfast"):
    xml = f"""
    <mujoco>
      <option timestep="{DT}" gravity="0 0 0" density="0" viscosity="0"
              integrator="{integrator}"/>
      <worldbody>
        <body name="hull" pos="0 0 {-depth}"><freejoint/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" mass="8.0"/>
          <body name="bell" pos="0.3 0 0">
            <joint name="bell_j" type="hinge" axis="0 1 0"
                   range="{LO} {HI}"/>
            <geom type="ellipsoid" size="0.12 0.12 0.08" mass="0.6"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position name="a" joint="bell_j" kp="60" ctrlrange="{LO} {HI}"/>
      </actuator>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    jets = JetSet(
        body_id=np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bell")]),
        joint_id=np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bell_j")]),
        axis_local=np.array([[1.0, 0.0, 0.0]]),
        volume=np.array([V0]), stroke_fraction=np.array([SF]),
        orifice_area=np.array([AREA]), joint_range=np.array([[LO, HI]]))
    return model, data, jets


def _drive(model, data, jets, medium, seconds: float = SECONDS) -> dict:
    jid = int(jets.joint_id[0])
    out = {"reported": 0.0, "ideal": 0.0, "signed": 0.0, "impulse": 0.0,
           "peak_ve": 0.0, "peak_thrust": 0.0}
    rho = SEAWATER.rho
    for k in range(int(seconds / DT)):
        t = k * DT
        data.ctrl[0] = 0.5 * (LO + HI) + 0.5 * (HI - LO) * math.sin(
            2 * math.pi * FREQ * t)
        data.xfrc_applied[:] = 0.0
        thrust = jets.apply(model, data, medium, t, DT)
        mujoco.mj_step(model, data)
        omega = float(data.qvel[model.jnt_dofadr[jid]])
        Q = V0 * SF * (omega / (HI - LO))
        cf = 1.0 if Q > 0 else jets.refill_efficiency
        out["ideal"] += cf * 0.5 * rho * abs(Q) ** 3 / AREA**2 * DT
        out["reported"] += jets.actuator_work(model, data) * DT
        out["signed"] += float(data.actuator_force[0]) * omega * DT
        out["impulse"] += thrust * DT
        out["peak_ve"] = max(out["peak_ve"], abs(Q) / AREA)
        out["peak_thrust"] = max(out["peak_thrust"], thrust)
    return out


def test_the_work_reported_is_the_work_the_jet_implies() -> None:
    """`tau * omega == p * Q`, which is the identity the load is built on."""
    print("\njet: the reported pumping work is the work the momentum flux implies")
    model, data, jets = _bell()
    r = _drive(model, data, jets, MediumField(water=SEAWATER))
    rel = abs(r["reported"] - r["ideal"]) / max(r["ideal"], 1e-30)
    check("JetSet.actuator_work equals (1/2) rho |Q|^3 / A^2 integrated",
          rel < 5e-3,
          f"{r['reported']:.4f} J reported against {r['ideal']:.4f} J implied, "
          f"relative {rel:.3e}")
    check("and it is not zero", r["reported"] > 1.0,
          f"{r['reported']:.2f} J over {SECONDS} s -- this returned a bare 0.0 "
          f"under a comment claiming the driving actuator's torque accounted "
          f"for it, with no torque anywhere in the module")


def test_the_actuator_supplies_at_least_what_the_jet_carries() -> None:
    """P_actuator >= P_jet.  A propulsor cannot return energy to its motor."""
    print("\njet: the actuator supplies at least what the jet carries away")
    model, data, jets = _bell()
    r = _drive(model, data, jets, MediumField(water=SEAWATER))
    check("the actuator's mechanical work covers the pumping work",
          r["signed"] >= r["reported"],
          f"{r['signed']:.2f} J supplied against {r['reported']:.2f} J pumped")


def test_the_load_is_real_and_bounds_the_jet() -> None:
    """Charging for the water has to slow the stroke down.

    A load that changes nothing is not a load.  The same drive against the same
    servo must reach a lower jet velocity once the pressure is being resisted.
    """
    print("\njet: charging for the water slows the stroke")
    model, data, jets = _bell()
    loaded = _drive(model, data, jets, MediumField(water=SEAWATER))

    # The same machine with the pumping load switched off, by handing `apply`
    # a model whose damping it cannot reach.
    model2, data2, jets2 = _bell()
    jets2._dofadr = np.array([-1])   # no driven joint -> no load applied
    free = _drive(model2, data2, jets2, MediumField(water=SEAWATER))

    check("the loaded stroke reaches a lower jet velocity",
          loaded["peak_ve"] < free["peak_ve"],
          f"{loaded['peak_ve']:.2f} m/s loaded against {free['peak_ve']:.2f} "
          f"m/s unloaded")
    check("and a lower peak thrust",
          loaded["peak_thrust"] < free["peak_thrust"],
          f"{loaded['peak_thrust']:.0f} N against {free['peak_thrust']:.0f} N")
    check("the unloaded case reports no pumping work at all",
          free["reported"] == 0.0,
          "which is what the module did before this load existed")


def test_a_bell_out_of_the_water_costs_nothing() -> None:
    """No water, no jet, no bill.  The same `subf` that gates the thrust."""
    print("\njet: a bell in air pumps nothing and is charged nothing")
    model, data, jets = _bell(depth=-50.0)   # well above the surface
    r = _drive(model, data, jets, MediumField(water=SEAWATER), seconds=1.0)
    check("no thrust in air", abs(r["impulse"]) < 1e-9,
          f"impulse {r['impulse']:.3e} N.s")
    check("and no pumping work in air", r["reported"] < 1e-9,
          f"{r['reported']:.3e} J")


def test_the_load_is_stable_against_the_timestep() -> None:
    """The reason it is damping and not an explicit torque.

    `tau = -c(omega) omega` with `c` running into hundreds of N.m.s/rad against
    a bell inertia of a few thousandths of a kg.m^2.  Applied explicitly that
    diverges -- measured at a jet velocity of 7.4e6 m/s within four seconds.
    Written into `dof_damping`, MuJoCo's implicit integrator inverts it with
    the rest of the dynamics.
    """
    print("\njet: the pumping load is stable at every timestep tested")
    ves = []
    for dt_scale in (1.0, 2.0, 4.0):
        model, data, jets = _bell()
        model.opt.timestep = DT * dt_scale
        r = _drive(model, data, jets, MediumField(water=SEAWATER))
        ves.append(r["peak_ve"])
        check(f"stable at dt = {DT * dt_scale:g}",
              math.isfinite(r["peak_ve"]) and r["peak_ve"] < 200.0,
              f"peak jet velocity {r['peak_ve']:.2f} m/s")
    # The damping coefficient carries |omega| from the previous state, so the
    # load lags by a step and a coarser step feels a weaker one.  Recorded
    # rather than hidden: the *stability* is unconditional, the *magnitude*
    # is not converged at the project's own timestep.
    print(f"         peak jet velocity {ves[0]:.1f} -> {ves[1]:.1f} -> "
          f"{ves[2]:.1f} m/s as dt goes {DT:g} -> {DT*2:g} -> {DT*4:g}.  The "
          f"load's magnitude is first-order in dt because its coefficient "
          f"reads last step's joint rate; the project runs at 0.004, where "
          f"the load is the weakest of the three.")


def test_the_model_damping_is_restored_between_episodes() -> None:
    """A bell must not carry its pumping load into the next episode."""
    print("\njet: the pumping load is rebuilt, never accumulated")
    model, data, jets = _bell()
    dof = int(model.jnt_dofadr[int(jets.joint_id[0])])
    dry = float(model.dof_damping[dof])
    _drive(model, data, jets, MediumField(water=SEAWATER), seconds=0.5)
    during = float(model.dof_damping[dof])
    jets.reset(model)
    check("the load is present while driving", during > dry,
          f"{dry:.4g} dry -> {during:.4g} N.m.s/rad under load")
    check("and reset restores the model's own damping",
          abs(float(model.dof_damping[dof]) - dry) < 1e-15,
          f"{float(model.dof_damping[dof]):.4g} against {dry:.4g}")


def main() -> int:
    test_the_work_reported_is_the_work_the_jet_implies()
    test_the_actuator_supplies_at_least_what_the_jet_carries()
    test_the_load_is_real_and_bounds_the_jet()
    test_a_bell_out_of_the_water_costs_nothing()
    test_the_load_is_stable_against_the_timestep()
    test_the_model_damping_is_restored_between_episodes()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} jet-energy checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("jet energy accounting checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
