import sys, math, numpy as np
sys.path.insert(0, "experiments/triphibian_diagnosis/existence")
import stateless as st
from dytiscidae.envs import mission as M
start, seed, t0, t1, every = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
arm = sys.argv[6] if len(sys.argv) > 6 else "teach"
_, env, _ = st.tq.make_env("L1", seed)
sp = st.Split(env, arm, seed); sp.log = []
read = M.evaluate_mission(env, sp.ctrl, st.Domain(start), seed)
print("legs", [(d.value, round(s,2)) for d, s in read.schedule], "held", read.held, "shares", [round(x,2) for x in read.shares], read.failure)
for i, (t, o, u) in enumerate(sp.log):
    if t0 <= t <= t1 and i % every == 0:
        v = 5*o[0:3]; depth = 5*math.atanh(max(min(o[9],0.999999999),-0.999999999))
        print(f"t={t:6.2f} cmd={int(np.argmax(o[11:14]))} depth={depth:6.2f} clr={o[35]:6.2f} dw={o[36]:6.2f} vb=({v[0]:5.2f},{v[1]:5.2f},{v[2]:5.2f}) gz={o[8]:5.2f} wet={np.round(o[39:43],1)} con={o[15]:.0f} mem={o[45]:.0f} | a=({u[0]:5.1f},{u[1]:5.1f},{u[2]:5.1f}) tilt={u[3]:.0f} on={u[5]:.0f} dr={u[6]:.1f} tu={u[7]:.1f} m={u[9]:.0f}")
