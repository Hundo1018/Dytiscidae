#!/usr/bin/env python3
"""Who pays for the water a pulsed jet expels?

0. The problem
--------------
`JetSet.apply` produces thrust `rho Q^2 / A` and injects it into
`data.xfrc_applied` as a pure force.  `JetSet.actuator_work` returns

    return 0.0  # accounted through the driving actuator's torque

and `TriphibianEnv.step` charges the battery from

    self.budget.step(|data.actuator_force|, |data.actuator_velocity|, dt)

So the claim is that expelling the water shows up as extra torque on the joint
that drives the bell.  Nothing in `JetSet.apply` writes a joint torque, a
`qfrc_applied`, or a body torque -- it writes a force at the body CoM and
returns.  If the claim is false, a bell produces thrust and is charged only for
swinging its own hinge, which is free thrust in water.

1. Competing explanations
--------------------------
  H1  MuJoCo's constraint solver reflects the thrust back into the joint, so
      the actuator does feel it.  Plausible: the force acts at the body CoM,
      which is not on the joint axis, so it exerts a moment about the joint.
  H2  The pumping work is simply not charged, and the moment in H1 is the
      thrust's lever arm rather than the pressure the pump works against --
      a different quantity, of a different size, and of either sign.

2. What the pumping work actually is
-------------------------------------
The jet leaves at `v_e = Q / A`.  The cavity pressure needed to drive it is the
stagnation pressure `p = (1/2) rho v_e^2`, so the mechanical power the muscle
must supply is

    P_jet = p * Q = (1/2) rho Q^3 / A^2 = (1/2) * thrust * v_e.        (1)

This is the ideal minimum -- no nozzle loss, no leakage, no refill cost.  The
measurement below compares it against what the actuator was charged.

3. Measurement
--------------
One free body carrying one bell, submerged, driven by a position servo through
a sinusoid over the joint's full range.  Integrate (1) over the run, integrate
`|tau * omega|` from `data.actuator_force` and `data.actuator_velocity` over the
same run, and print both.

Run:  PYTHONPATH=. python experiments/jet_energy/run.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco  # noqa: E402

from dytiscidae.physics.jet import JetSet  # noqa: E402
from dytiscidae.physics.medium import SEAWATER, MediumField  # noqa: E402
from experiments.harness import ExperimentResult, load_config  # noqa: E402


def build(cfg: dict):
    lo, hi = cfg["bell"]["joint_range_rad"]
    xml = f"""
    <mujoco>
      <option timestep="{cfg['drive']['timestep_s']}" gravity="0 0 -9.80665"
              density="0" viscosity="0"/>
      <worldbody>
        <body name="hull" pos="0 0 {-cfg['body']['depth_m']}">
          <freejoint/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.05"
                mass="{cfg['body']['dry_mass_kg']}"/>
          <body name="bell" pos="0.3 0 0">
            <joint name="bell_j" type="hinge" axis="0 1 0"
                   range="{lo} {hi}"/>
            <geom type="ellipsoid" size="0.12 0.12 0.08" mass="0.6"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position name="bell_a" joint="bell_j" kp="60" ctrlrange="{lo} {hi}"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bell")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bell_j")
    jets = JetSet(
        body_id=np.array([bid]), joint_id=np.array([jid]),
        axis_local=np.array([[1.0, 0.0, 0.0]]),
        volume=np.array([cfg["bell"]["volume_m3"]]),
        stroke_fraction=np.array([cfg["bell"]["stroke_fraction"]]),
        orifice_area=np.array([cfg["bell"]["orifice_area_m2"]]),
        joint_range=np.array([[lo, hi]]),
    )
    return model, data, jets, jid


def run(cfg: dict) -> ExperimentResult:
    res = ExperimentResult("jet_energy", cfg)
    model, data, jets, jid = build(cfg)
    medium = MediumField(water=SEAWATER)
    dt = cfg["drive"]["timestep_s"]
    n = int(cfg["drive"]["seconds"] / dt)
    lo, hi = cfg["bell"]["joint_range_rad"]
    f = cfg["drive"]["frequency_hz"]
    rho = SEAWATER.rho
    V0 = cfg["bell"]["volume_m3"]
    sf = cfg["bell"]["stroke_fraction"]
    A = cfg["bell"]["orifice_area_m2"]

    e_jet_ideal = 0.0     # integral of (1/2) rho Q^3 / A^2
    e_actuator = 0.0      # integral of |tau * omega|
    impulse = 0.0         # integral of thrust
    peak_thrust = 0.0
    peak_ve = 0.0

    for k in range(n):
        t = k * dt
        target = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.sin(2 * np.pi * f * t)
        data.ctrl[0] = target
        data.xfrc_applied[:] = 0.0
        thrust = jets.apply(model, data, medium, t, dt)
        mujoco.mj_step(model, data)

        omega = float(data.qvel[model.jnt_dofadr[jid]])
        span = max(hi - lo, 1e-6)
        Q = V0 * sf * (omega / span)          # positive while expelling
        if Q > 0:
            e_jet_ideal += 0.5 * rho * Q**3 / A**2 * dt
            peak_ve = max(peak_ve, Q / A)
        e_actuator += abs(float(data.actuator_force[0]) * omega) * dt
        impulse += thrust * dt
        peak_thrust = max(peak_thrust, thrust)

    ratio = e_jet_ideal / max(e_actuator, 1e-30)
    print(f"\n{cfg['drive']['seconds']} s at {f} Hz, bell {V0*1e3:.1f} L, "
          f"stroke {sf:.2f}, orifice {A*1e4:.1f} cm^2, submerged")
    print(f"  peak thrust                                  {peak_thrust:10.2f} N")
    print(f"  peak jet velocity                            {peak_ve:10.2f} m/s")
    print(f"  thrust impulse                               {impulse:10.2f} N.s")
    print(f"  ideal pumping work, (1/2) rho Q^3 / A^2 dt   {e_jet_ideal:10.2f} J")
    print(f"  actuator mechanical work charged             {e_actuator:10.2f} J")
    print(f"\n  the jet does {ratio:.1f}x the work the actuator was charged for.")

    # H1: does the thrust reach the joint at all?  Run the same drive with the
    # jet off and compare the actuator work.
    model2, data2, jets2, jid2 = build(cfg)
    e_noje = 0.0
    for k in range(n):
        t = k * dt
        data2.ctrl[0] = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.sin(2 * np.pi * f * t)
        data2.xfrc_applied[:] = 0.0
        mujoco.mj_step(model2, data2)
        omega = float(data2.qvel[model2.jnt_dofadr[jid2]])
        e_noje += abs(float(data2.actuator_force[0]) * omega) * dt
    charged = e_actuator - e_noje
    frac = charged / max(e_jet_ideal, 1e-30)
    print(f"\n  the same drive with the jet switched off costs "
          f"{e_noje:.4f} J against {e_actuator:.4f} J with it on.")
    print(f"  Switching the jet on changes the actuator's bill by "
          f"{charged:+.4f} J, which is {100*frac:+.3f}% of the "
          f"{e_jet_ideal:.2f} J the jet actually did.")
    print(f"  So {100*(1-abs(frac)):.1f}% of the pumping work is uncharged.  "
          f"H1 is rejected: the thrust's moment about the joint is a "
          f"different quantity from the pressure the pump works against, of a "
          f"different size, and of either sign.")
    print(f"\n  Separately: peak thrust is {peak_thrust:.0f} N on a "
          f"{cfg['body']['dry_mass_kg'] + 0.6:.1f} kg machine -- "
          f"{peak_thrust/((cfg['body']['dry_mass_kg']+0.6)*9.80665):.0f} g.  "
          f"`thrust = rho Q^2 / A` with Q taken straight from the joint rate "
          f"has no upper bound and no limiter, where `FluidSolver` clamps its "
          f"own forces at 60x vehicle weight and raises `diag.clamped`.")

    res.record("peak_thrust_N", peak_thrust)
    res.record("peak_jet_velocity_m_s", peak_ve)
    res.record("thrust_impulse_Ns", impulse)
    res.record("ideal_pumping_work_J", e_jet_ideal)
    res.record("actuator_work_charged_J", e_actuator)
    res.record("actuator_work_jet_off_J", e_noje)
    res.record("uncharged_work_ratio", ratio)
    res.record("actuator_delta_from_jet_J", float(e_actuator - e_noje))
    res.record("charged_fraction_of_pumping_work",
               float((e_actuator - e_noje) / max(e_jet_ideal, 1e-30)))
    res.record("peak_thrust_in_g",
               float(peak_thrust / ((cfg["body"]["dry_mass_kg"] + 0.6) * 9.80665)))
    return res


if __name__ == "__main__":
    cfg = load_config(Path(__file__).parent / "config.json")
    result = run(cfg)
    result.write(Path(__file__).parent / "results")
