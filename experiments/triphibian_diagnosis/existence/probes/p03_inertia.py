#!/usr/bin/env python3
"""Probe: what the rotational block of mj_fullM is for the free root (about the
origin or about the COM), checked against a direct sum over bodies."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triquad as tq  # noqa: E402
import mujoco  # noqa: E402
from dytiscidae.envs.triphibian import Domain  # noqa: E402

for body in ("B0", "B1"):
    p, env, facts = tq.make_env(body, 0)
    env.reset(Domain.AIR, randomise=False)
    m, d = env.model, env.data
    M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, d, M)
    R = d.xmat[env.root_body].reshape(3, 3)
    o = d.xpos[env.root_body]; com = d.subtree_com[env.root_body]
    def inertia_about(pt):
        I = np.zeros((3, 3))
        for b in range(1, m.nbody):
            Rb = d.ximat[b].reshape(3, 3)
            Ib = Rb @ np.diag(m.body_inertia[b]) @ Rb.T
            r = d.xipos[b] - pt
            I += Ib + m.body_mass[b] * (float(r @ r) * np.eye(3) - np.outer(r, r))
        return R.T @ I @ R
    print(body, "mass", M[0, 0], "com-origin (body frame)", np.round(R.T @ (com - o), 4))
    print("  M rot diag      ", np.round(np.diag(M[3:6, 3:6]), 5))
    print("  about origin    ", np.round(np.diag(inertia_about(o)), 5))
    print("  about COM       ", np.round(np.diag(inertia_about(com)), 5))
    print("  M[0:3,3:6]\n", np.round(M[0:3, 3:6], 4))
