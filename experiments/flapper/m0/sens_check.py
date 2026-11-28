import sys, math, numpy as np
sys.path.insert(0,"experiments/flapper")
from mission import Machine
from sensors import Sensors
def run(mc, S, stroke_deg, f, secs, label):
    w=2*math.pi*f; Phi=math.radians(stroke_deg); st=mc.role=="stroke"; ep=[]; er=[]
    for k in range(int(secs/mc.dt)):
        t=mc.t; q=np.zeros(len(mc.aid)); qd=np.zeros(len(mc.aid))
        q[st]=Phi/2*math.sin(w*t); qd[st]=Phi/2*w*math.cos(w*t)
        mc.step(q,qd); r=S.sample()
        if r is not None and mc.t>1.0:
            from sensors import _quat_to_R
            Rt=mc.d.xmat[mc.hull].reshape(3,3); Re=_quat_to_R(r["quat"])
            ut=Rt.T@[0,0,1]; ue=Re.T@[0,0,1]
            ep.append(math.degrees(math.acos(max(-1,min(1,float(ut@ue)))))); er.append(math.degrees(math.acos(max(-1,min(1,float(Rt[2,2]))))))
    print("%s: tilt estimate error mean %.2f max %.2f deg | true tilt from upright mean %.1f max %.1f" % (label, np.mean(ep), np.max(ep), np.mean(er), np.max(er)))
mc = Machine("land", pose=dict(x=17, z=5)); mc.drop_to_ground(0.01); run(mc, Sensors(mc, seed=1), 0, 1, 5, "rest on beach")
mc = Machine("land", pose=dict(x=17, z=5)); mc.drop_to_ground(0.01); run(mc, Sensors(mc, seed=1), 20, 2.5, 5, "flapping 20 deg on the beach")
mc = Machine("air", pose=dict(z=30, pitch_deg=10, vx=8.0)); run(mc, Sensors(mc, seed=1), 40, 2.5, 4, "free, flapping 40 deg at 8 m/s")
