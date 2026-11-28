"""M1 (standing take-off from the beach): shared helpers over `mission.Machine`.

Nothing here changes the harness: a machine is made by `mission.Machine`, placed
with its own `place`/`drop_to_ground` plus a yaw written into the free joint's
quaternion, and read through MuJoCo's data.  `PREREG_M1_takeoff.md`.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

G = 9.80665
# K1, the take-off point of W6 / M0: 4.5 Hz, stroke 85 deg peak to peak, feather
# 50 deg a quarter beat ahead, body at 90 deg.
K1 = dict(f=4.5, Phi=math.radians(85.0), Psi=math.radians(50.0))


def machine(x=17.0, yaw_deg=0.0, J_r=2e-5, gear=24.0, feather_gear=16.0, **kw):
    """The reference machine lying on its belly on the beach at ``x``, nose toward
    ``yaw_deg`` (0: +x, up the beach), settled by nothing yet."""
    from mission import Machine
    motors = dict(stroke=dict(J_r=J_r, N=gear), feather=dict(J_r=J_r, N=feather_gear))
    mc = Machine("land", motors=motors, **kw)
    mc.place(x=x, y=0.0, z=5.0, pitch_deg=0.0)
    set_attitude(mc, yaw_deg, 0.0)
    mc.drop_to_ground(0.01)
    return mc


def set_attitude(mc, yaw_deg, pitch_deg):
    mj = mc.mj
    qy = np.array([math.cos(math.radians(yaw_deg) / 2), 0, 0, math.sin(math.radians(yaw_deg) / 2)])
    a = math.radians(pitch_deg)
    qp = np.array([math.cos(-a / 2), 0.0, math.sin(-a / 2), 0.0])
    q = np.zeros(4)
    mj.mju_mulQuat(q, qy, qp)
    mc.d.qpos[3:7] = q
    mj.mj_forward(mc.m, mc.d)


def com(mc) -> np.ndarray:
    """Centre of the machine's dry (weighing) mass, world frame."""
    dm = mc.env.solver._dry_mass
    return (dm[:, None] * mc.d.xipos).sum(0) / dm.sum()


def com_vel(mc) -> np.ndarray:
    mj, m, d = mc.mj, mc.m, mc.d
    dm = mc.env.solver._dry_mass
    v = np.zeros(3)
    tmp = np.zeros(6)
    for b in range(1, m.nbody):
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, b, tmp, 0)
        v += dm[b] * tmp[3:]
    return v / dm.sum()


def world_contacts(mc):
    """(number of machine-world contacts, names of the machine bodies touching)."""
    m, d = mc.m, mc.d
    n, bodies = 0, set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 == 0) == (b2 == 0):
            continue
        n += 1
        bodies.add(b2 if b1 == 0 else b1)
    return n, sorted(bodies)


def axes(mc):
    """The hull's nose (x), left-wing (y) and back (z) axes in the world."""
    R = mc.d.xmat[mc.hull].reshape(3, 3)
    return R[:, 0], R[:, 1], R[:, 2]


def k1_targets(mc, t, amp=1.0, bias=0.0, f=None, Phi=None, Psi=None, psi0=0.0,
               amp_lr=0.0, psi0_lr=0.0, phase0=0.0):
    """Joint targets and rates by role for the K1 stroke (both wings alike unless
    ``amp_lr`` / ``psi0_lr``: left gets +half, right -half).  Stroke positive is up
    for both wings (checked on the built model, M1 probe P0)."""
    f = K1["f"] if f is None else f
    Phi = K1["Phi"] if Phi is None else Phi
    Psi = K1["Psi"] if Psi is None else Psi
    w = 2 * math.pi * f
    ph = w * t + phase0
    q = np.zeros(len(mc.aid))
    qd = np.zeros(len(mc.aid))
    left = np.array([("seg1" in mc.mj.mj_id2name(mc.m, mc.mj.mjtObj.mjOBJ_ACTUATOR, int(a))) for a in mc.aid])
    for n, r in enumerate(mc.role):
        s = 1.0 if left[n] else -1.0
        if r == "stroke":
            A = Phi / 2 * amp * (1.0 + 0.5 * s * amp_lr)
            q[n] = bias + A * math.sin(ph)
            qd[n] = A * w * math.cos(ph)
        elif r == "feather":
            q[n] = -Psi * math.cos(ph) + psi0 + 0.5 * s * psi0_lr
            qd[n] = Psi * w * math.sin(ph)
    return q, qd
