import sys, math, numpy as np
sys.path.insert(0,"experiments/flapper")
from mission import Machine
mc = Machine("land")
d,m=mc.d,mc.m
print("start z %.3f ground %.3f x %.2f"%(d.qpos[2], mc.env.ground_height(d.qpos[0]), d.qpos[0]))
for k in range(8000): mc.step(np.zeros(len(mc.aid)))
print("after 2 s: z %.3f ncon %d load %.3f"%(d.qpos[2], d.ncon, mc.contact_load()))
for i in range(min(d.ncon,4)):
    c=d.contact[i]; print("  contact bodies", m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2], "geom types", m.geom_type[c.geom1], m.geom_type[c.geom2])
f, Phi, Psi = 4.5, math.radians(85), math.radians(50); w=2*math.pi*f
for wn, dt in ((150,2.5e-4),(300,1e-4),(600,1e-4)):
    mc = Machine("air", wn=wn, dt=dt, pose=dict(z=30, pitch_deg=90.0, vx=0.0))
    q0 = mc.d.qpos[:7].copy(); az=[]; err=[]
    nb = int(round(1/(f*mc.dt))); st=mc.role=="stroke"; fe=mc.role=="feather"
    for k in range(6*nb):
        t=mc.t; q=np.zeros(len(mc.aid)); qd=np.zeros(len(mc.aid))
        q[st]=Phi/2*math.sin(w*t); qd[st]=Phi/2*w*math.cos(w*t); q[fe]=-Psi*math.cos(w*t); qd[fe]=Psi*w*math.sin(w*t)
        vb=mc.d.qvel[:3].copy(); mc.step(q,qd)
        if k>=2*nb: az.append((mc.d.qvel[:3]-vb)/mc.dt); err.append(np.abs(mc.d.qpos[mc.qad]-q)[:4].max())
        mc.d.qpos[:7]=q0; mc.d.qvel[:6]=0
    s=mc.summary(); a=np.mean(az,axis=0)
    print("wn %d dt %.0e: lift/W %.3f err %.2f deg shaft %s elec %s"%(wn,dt,(a[2]+9.80665)/9.80665,math.degrees(max(err)),{k:round(v,1) for k,v in s["shaft_W"].items()},{k:round(v,1) for k,v in s["elec_W"].items()}))
