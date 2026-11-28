#!/usr/bin/env python3
"""Probe: the mixer's inputs at the first steps of the B0 air segment (seed 1)."""
import sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triquad as tq  # noqa: E402
import mujoco  # noqa: E402
from dytiscidae.envs.triphibian import Domain  # noqa: E402

p, env, facts = tq.make_env("B0", 1)
ctrl, hand = tq.controller_for(env, "hand", False)
print("kv", hand.kv, "tau_max", hand.tau_max, "kt_a", hand.kt_a, "kq_a", hand.kq_a, "kt_w", hand.kt_w, "kq_w", hand.kq_w,
      "mass", hand.mass, "vol_b", hand.vol_b, "om_max", hand.om_max)
lst = np.linalg.lstsq
n = [0]
def dbg(A, b, rcond=None):
    out = lst(A, b, rcond=rcond)
    if n[0] < 60 and n[0] % 8 == 0:
        d = env.data
        print(f"t={d.time:.3f} rhs(thrust,tau)={np.round(b,4)} sub={env.solver.diag.mean_submerged:.3f}\n A=\n{np.round(A,4)}\n t_i={np.round(out[0],3)} rank={out[2]}")
    n[0] += 1
    return out
np.linalg.lstsq = dbg
row = tq.run_seg_item(env, ctrl, Domain.AIR, 1)
