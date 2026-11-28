#!/usr/bin/env python3
"""Probe: what the reference quad is, as built (no rollout).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python \
        experiments/triphibian_diagnosis/existence/probes/p00_b0_facts.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import mujoco  # noqa: E402

from dytiscidae.core.bodyplans import REFERENCE_PLANS  # noqa: E402
from dytiscidae.core.phenotype import build  # noqa: E402
from dytiscidae.envs.triphibian import (  # noqa: E402
    Domain, MissionSpec, TriphibianEnv, airworthiness, evaluate_tier0, rotor_lift_ratio)
from dytiscidae.physics.medium import AIR, GRAVITY, SEAWATER  # noqa: E402
from dytiscidae.physics.rotor import bemt  # noqa: E402

g = REFERENCE_PLANS["quad"]()
p = build(g)
print("summary:", p.summary())
print("mass budget:", {k: round(v, 4) for k, v in p.budget.as_dict().items()})
print("buoyant_volume L:", p.buoyant_volume * 1e3, "displaced L:", p.displaced_volume * 1e3,
      "gas L:", p.gas_volume * 1e3, "density_ratio:", p.density_ratio)
for s in p.segments:
    print(f"  seg {s.name}: kind={s.kind} L={s.length:.3f} r={s.radius:.3f} mass={s.mass:.4f} "
          f"vol={s.volume*1e3:.3f}L buoy={s.volume_buoyant*1e3:.3f}L mirrored={s.mirrored} "
          f"rotor={'yes' if s.rotor is not None else 'no'}")
print("checks:")
for c in p.report.checks:
    print(f"  {c.name}: margin={c.margin:+.3f} gating={c.gating} note={c.note}")
t0 = evaluate_tier0(p, MissionSpec())
print("gate_margin:", p.report.gate_margin, "tier0.mission_fraction:", t0.mission_fraction,
      "-> search Tier-0 gate passes:",
      not (p.report.gate_margin < -0.85 or t0.mission_fraction <= 0.0))
print("tier0 notes:", t0.notes)
print("airworthiness gates:", airworthiness(p), "rotor_lift_ratio:", rotor_lift_ratio(p))
print("battery: wh", g.battery_wh, g.battery_chem, "mass", p.battery.mass, "p_max W", p.battery.p_max)
for a in p.actuators:
    print(f"  actuator: class={a.motor_class} mass={a.mass:.3f} stall_torque={a.stall_torque:.4f} "
          f"p_cont={a.p_cont:.1f} max_speed={a.max_speed:.1f} km={a.km:.4f}")

env = TriphibianEnv(p, seed=0)
m, d = env.model, env.data
print("act_names:", env.act_names, "nu", m.nu, "nq", m.nq, "nv", m.nv, "timestep", m.opt.timestep)
print("cpg lo/hi:", env.cpg.lo, env.cpg.hi, "base offset:", env.cpg.base.offset)
spec = env.rotors.spec[0]
for name, fl in (("air", AIR), ("water", SEAWATER)):
    for om in (100.0, 300.0, 600.0, 1000.0):
        T, Q = bemt(spec, om, 0.0, 0.0, fl.rho, fl.mu)
        print(f"  bemt {name} om={om:6.0f}: T={T:9.3f} N Q={Q:8.4f} Nm  k_t={T/om**2:.3e} kappa={Q/max(T,1e-9):.4f}")
print("weight N:", p.mass * GRAVITY, "launch_speed:", env.launch_speed, "launch_pitch:", env.launch_pitch)
print("flies_level:", env.flies_level(), "level_margin:", env.level_margin(), "lift_margin:", env.lift_margin)
print("max_entry_speed:", p.max_entry_speed, "slam cap:", p.slam_pressure_capacity,
      "frontal_area:", p.frontal_area, "max_span:", p.max_span)
# geometry of geoms
for gi in range(m.ngeom):
    b = m.geom_bodyid[gi]
    if b == 0:
        continue
    print("  geom", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi), "body",
          mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b), "type", m.geom_type[gi],
          "size", np.round(m.geom_size[gi], 4), "contype", m.geom_contype[gi], m.geom_conaffinity[gi])
env.reset(Domain.LAND, randomise=False)
print("land reset: root z", d.qpos[2], "clearance", env.clearance(), "ground z at x", env.ground_height(float(d.qpos[0])))
for b in range(1, m.nbody):
    print("  body", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b), "xpos", np.round(d.xpos[b], 3),
          "mass", round(float(m.body_mass[b]), 4))
print("subtree com (rel root):", np.round(d.subtree_com[env.root_body] - d.xpos[env.root_body], 4))
