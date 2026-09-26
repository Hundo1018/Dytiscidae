#!/usr/bin/env python3
"""The control: a quadrotor flies the real air segment, scored as a flapper is.

Same spawn, scatter, task and `_score_segment` as `evaluate_tier1`'s air
segment.  One substitution: the command.  A multirotor is unstable without
feedback, so a textbook cascade (velocity -> tilt, attitude PD, altitude PD,
then a mixer) turns the task's commanded heading and speed into four rotor
speeds, in place of the CPG's open-loop output.  Everything downstream of the
command -- the rotor model, the airframe, the battery, the score -- is the
search's own.

What it answers: whether this simulator and this score can represent and
recognise flight when a machine has thrust.  If the quad scores and no flapper
does, the gap is the flapping, not the physics or the reward.

    PYTHONPATH=. python experiments/rotor/fly.py [seeds...]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco  # noqa: E402

from dytiscidae.core.bodyplans import REFERENCE_PLANS  # noqa: E402
from dytiscidae.core.phenotype import build  # noqa: E402
from dytiscidae.envs.evaluate import _scatter_seed  # noqa: E402
from dytiscidae.envs.tasks import schedule_for, task_seed  # noqa: E402
from dytiscidae.envs.triphibian import Domain, TriphibianEnv  # noqa: E402
from dytiscidae.physics.medium import AIR, GRAVITY  # noqa: E402
from dytiscidae.physics.rotor import bemt  # noqa: E402


def _fullM(m, d, M):
    """Dense mass matrix (this MuJoCo's signature, as tests/test_added_mass.py)."""
    mujoco.mj_fullM(m, d, M)


def controller(env):
    m, d = env.model, env.data
    rot_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, a[:-2] + "_rot")
                  for a in env.act_names]
    spec = env.rotors.spec[0]
    T0, Q0 = bemt(spec, 600.0, 0.0, 0.0, AIR.rho, AIR.mu)
    k_t, kappa = T0 / 600.0**2, Q0 / T0
    om_max = float(env.cpg.hi[0])
    state = {"z0": None}
    M = np.zeros((m.nv, m.nv))

    def command(params, t):
        _fullM(m, d, M)
        mass = float(M[0, 0])
        inertia = np.diag(M[3:6, 3:6]).mean()
        R = d.xmat[env.root_body].reshape(3, 3)
        pos = d.xipos[env.root_body]
        v = d.qvel[:3].copy()
        w_body = d.qvel[3:6].copy()                       # free joint: local frame
        if state["z0"] is None:
            state["z0"] = float(pos[2])
        ph = env._phase_now()
        head = float(ph.heading) if ph is not None else 0.0
        speed = float(ph.speed) if (ph is not None and ph.speed) else 0.0
        v_des = np.array([speed * math.cos(head), speed * math.sin(head), 0.0])
        a = 1.2 * (v_des - v)
        a[2] = 3.0 * (state["z0"] - pos[2]) - 2.5 * v[2]
        a = a + np.array([0.0, 0.0, GRAVITY])
        # at most 35 degrees of tilt
        h = np.linalg.norm(a[:2]); lim = a[2] * math.tan(math.radians(35))
        if h > lim > 0:
            a[:2] *= lim / h
        f = mass * a
        z_des = f / np.linalg.norm(f)
        # desired frame: body z along z_des, body x toward the commanded heading
        x_c = np.array([math.cos(head), math.sin(head), 0.0])
        y_d = np.cross(z_des, x_c); y_d /= np.linalg.norm(y_d)
        x_d = np.cross(y_d, z_des)
        Rd = np.stack([x_d, y_d, z_des], 1)
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        e_r = np.array([E[2, 1], E[0, 2], E[1, 0]])
        wn = 7.0
        tau = inertia * (-(wn**2) * e_r - 2 * 0.8 * wn * w_body)   # body frame
        thrust = float(f @ R[:, 2])
        # mixer in the body frame: sum t_i, r_i x (t_i z), and -kappa s_i t_i
        A = np.zeros((4, len(rot_bodies)))
        for i, b in enumerate(rot_bodies):
            r = R.T @ (d.xipos[b] - pos)
            ax_b = R.T @ (d.xmat[b].reshape(3, 3) @ m.jnt_axis[m.body_jntadr[b]])
            s = float(np.sign(ax_b[2]))                   # spin sense seen from above
            A[:, i] = [1.0, r[1], -r[0], -kappa * s]
        t_i = np.linalg.lstsq(A, np.array([thrust, tau[0], tau[1], tau[2]]), rcond=None)[0]
        return np.clip(np.sqrt(np.maximum(t_i, 0.0) / k_t), 0.0, om_max)

    return command


def fly(seed: int) -> dict:
    env = TriphibianEnv(build(REFERENCE_PLANS["quad"]()), seed=seed)
    env.reset(Domain.AIR)
    env.scatter(np.random.default_rng(_scatter_seed(seed, Domain.AIR)))
    env.task = schedule_for(Domain.AIR, np.random.default_rng(task_seed(_scatter_seed(seed, Domain.AIR))))
    env.cpg.command = controller(env)
    z0 = float(env.root_pos()[2])
    seg = env.rollout(8.0, domain=Domain.AIR)
    mm = seg.measurements
    return {"seed": seed, "competence": seg.competence, "z_start": z0,
            "z_end": float(env.root_pos()[2]),
            "energy_wh": float(env.budget.actuator_j + env.budget.avionics_j) / 3600.0,
            **{k: mm[k] for k in ("height_hold", "turn_response", "cruise_tracking",
                                  "turn_tracking", "measured_sink_rate", "airborne_fraction",
                                  "task_score", "spin_rate") if k in mm}}


def main() -> int:
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    rows = []
    for s in seeds:
        r = fly(s)
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    Path(__file__).with_name("results").mkdir(exist_ok=True)
    (Path(__file__).with_name("results") / "fly.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
