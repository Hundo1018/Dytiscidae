"""Share of a commanded sinusoidal stroke each servo actually reaches.

Hull welded in place (gravity off, no fluid), each actuated joint driven with
amp*sin(2 pi f t); reports achieved/commanded peak-to-peak over the last two
seconds.  "lead" adds (kv/kp)*d(q_ref)/dt to the command -- the velocity
feed-forward a trajectory-tracking servo applies.
"""
import sys
import numpy as np
import mujoco
from dytiscidae.core.bodyplans import beetle, gannet
from dytiscidae.core.mjcf import compile_phenotype
from dytiscidae.core.phenotype import build

AMP = 0.5
def run(plan, f, lead):
    p = build(plan())
    m, d, acts, _ = compile_phenotype(p)
    m.opt.gravity[:] = 0
    # weld the root: zero its free joint by huge mass is simpler -> fix via eq? just clamp qpos each step
    fj = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    qadr = [m.jnt_qposadr[j] for j in fj]; vadr = [m.jnt_dofadr[j] for j in fj]
    q0 = d.qpos.copy()
    jids = [m.actuator_trnid[a, 0] for a in range(m.nu)]
    tau = m.actuator_biasprm[:, 2] / np.minimum(m.actuator_biasprm[:, 1], -1e-9)  # kv/kp (biasprm = [0,-kp,-kv])
    T = 4.0; n = int(T / m.opt.timestep); rec = []; sat = 0
    for i in range(n):
        t = d.time
        ref = AMP * np.sin(2 * np.pi * f * t)
        dref = AMP * 2 * np.pi * f * np.cos(2 * np.pi * f * t)
        d.ctrl[:] = ref + (tau * dref if lead else 0.0)
        mujoco.mj_step(m, d)
        for a, v in zip(qadr, vadr):
            d.qpos[a:a + 7] = q0[a:a + 7]; d.qvel[v:v + 6] = 0
        if t > T - 2.0:
            rec.append([d.qpos[m.jnt_qposadr[j]] for j in jids])
            sat += np.mean(np.abs(d.actuator_force) >= 0.999 * m.actuator_forcerange[:, 1])
    rec = np.array(rec)
    ratio = (rec.max(0) - rec.min(0)) / (2 * AMP)
    return float(np.median(ratio)), sat / len(rec), float(np.median(tau))

for name, plan in (("beetle", beetle), ("gannet", gannet)):
    for f in (2.2, 4.0, 7.3, 11.0):
        r0, s0, tau = run(plan, f, False); r1, s1, _ = run(plan, f, True)
        print(f"{name:7s} {f:5.1f} Hz  plain {r0:.2f} (sat {s0:.0%})   lead {r1:.2f} (sat {s1:.0%})   kv/kp {tau*1000:.0f} ms")
