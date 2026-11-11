#!/usr/bin/env python3
"""Probe: what the outer loop asks of the cascade in the air segment (B0 seed 1)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triquad as tq  # noqa: E402
from dytiscidae.envs.triphibian import Domain  # noqa: E402

p, env, facts = tq.make_env(sys.argv[1] if len(sys.argv) > 1 else "B0", 1)
ctrl, hand = tq.controller_for(env, "hand", False)
casc = hand._cascade
n = [0]
def dbg(S, a_fb, yaw, tilt_deg, **kw):
    out = casc(S, a_fb, yaw, tilt_deg, **kw)
    if n[0] % 50 == 0 and S.t < 4.3:
        ph = env._phase_now()
        print(f"t={S.t:.3f} ph=({ph.kind},{ph.heading:.2f},{ph.speed:.2f}) v={np.round(S.v,2)} a_fb={np.round(a_fb,2)} "
              f"yaw_des={yaw:.2f} yaw={S.yaw:.2f} R2={np.round(S.R[:,2],3)} Rx={np.round(S.R[:,0],3)} thrust_meas={np.round(env.rotors.last_thrust,2)}")
    n[0] += 1
    return out
hand._cascade = dbg
row = tq.run_seg_item(env, ctrl, Domain.AIR, 1)
print(row["score"], row["m"])
