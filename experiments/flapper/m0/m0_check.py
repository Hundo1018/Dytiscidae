import sys, math, numpy as np
sys.path.insert(0,"experiments/flapper")
from mission import Machine
# 1. land at rest, wings level
mc = Machine("land"); mc.drop_to_ground()
z0 = mc.d.qpos[2]
for k in range(8000): mc.step(np.zeros(len(mc.aid)))
print("land rest: mass %.3f roles %s dz %.4f load %.3f pitch %.1f qacc %d" % (mc.mass, list(mc.role), mc.d.qpos[2]-z0, mc.contact_load(), mc.pitch_deg(), mc.bad_qacc))
# 2. held in air at 90 deg, the take-off stroke (4.5 Hz, 85, feather 50 a quarter beat ahead)
f, Phi, Psi = 4.5, math.radians(85), math.radians(50)
w = 2*math.pi*f
for gear in (24, 32):
    mc = Machine("air", motors=dict(stroke=dict(N=gear)), pose=dict(z=30, pitch_deg=90.0, vx=0.0))
    q0 = mc.d.qpos[:7].copy(); az=[]
    sign = np.where(np.array([("seg2" in n) for n in mc.env.act_names]), 1.0, 1.0)
    nb = int(round(1/(f*mc.dt)))
    for k in range(6*nb):
        t = mc.t
        q = np.zeros(len(mc.aid)); qd = np.zeros(len(mc.aid))
        st = mc.role=="stroke"; fe = mc.role=="feather"
        q[st] = Phi/2*math.sin(w*t); qd[st] = Phi/2*w*math.cos(w*t)
        q[fe] = -Psi*math.cos(w*t); qd[fe] = Psi*w*math.sin(w*t)
        vb = mc.d.qvel[:3].copy()
        mc.step(q, qd)
        if k >= 2*nb: az.append((mc.d.qvel[:3]-vb)/mc.dt)
        mc.d.qpos[:7] = q0; mc.d.qvel[:6] = 0.0
    a = np.mean(az, axis=0)
    st_q = mc.d.qpos[mc.qad[mc.role=="stroke"]]
    print("held take-off gear %d: lift/W %.3f  ax %.2f  elec %s  limit %s qacc %d" % (gear, (a[2]+9.80665)/9.80665, a[0], {k:round(v,1) for k,v in mc.summary()["elec_W"].items()}, {k:round(v,2) for k,v in mc.summary()["at_limit"].items()}, mc.bad_qacc))
