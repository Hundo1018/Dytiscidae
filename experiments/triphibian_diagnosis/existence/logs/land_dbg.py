import sys, json, math
sys.path.insert(0, "experiments/triphibian_diagnosis/existence")
import landgait as lg, numpy as np
tq = lg.tq
body, seed = sys.argv[1], int(sys.argv[2]); gait = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
every = int(sys.argv[4]) if len(sys.argv) > 4 else 50
p, env, facts = tq.make_env(body, seed)
ctrl, hand, law = lg.controller_for(env, "walk", gait)
step = env.step; n = [0]
def st(target):
    a = step(target)
    if n[0] % every == 0:
        d = env.data; R = d.xmat[env.root_body].reshape(3, 3)
        ph = env._phase_now()
        com = (env.model.body_mass[1:, None] * d.xipos[1:]).sum(0) / env.model.body_mass[1:].sum()
        print(f"t={d.time:5.2f} {ph.kind if ph else '-':6s} hd={ph.heading if ph else 0:5.2f} yaw={math.atan2(R[1,0],R[0,0]):5.2f} up={R[2,2]:.2f} root=({d.qpos[0]:.3f},{d.qpos[1]:.3f},{d.qpos[2]:.3f}) com=({com[0]:.3f},{com[1]:.3f}) Fz/mg={lg.ground_fz(env)/(env.model.body_mass[1:].sum()*9.81):.2f} ncon={d.ncon} legs a={np.round(d.qpos[env._act_qadr[4::2]],2)} f={np.round(d.qpos[env._act_qadr[5::2]],2)}")
    n[0] += 1
    return a
env.step = st
row = tq.run_seg_item(env, ctrl, lg.Domain.LAND, seed)
print(row["score"], row["m"])
