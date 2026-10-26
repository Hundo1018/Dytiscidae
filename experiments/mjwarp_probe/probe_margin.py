# Beetle from the project fixture path. Host contacts at rest with the production margin, and whether
# MJWarp accepts that margin once MULTICCD and NATIVECCD are disabled on the device copy.
import copy, mujoco, numpy as np, mujoco_warp as mjw, warp as wp
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import TriphibianEnv
e = TriphibianEnv(build(BODY_PLANS["beetle"]()), seed=0)
mjm = e.model
print("host disableflags MULTICCD:", bool(mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD),
      "NATIVECCD:", bool(mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD),
      "margin>0 geoms:", int((mjm.geom_margin != 0).sum()), flush=True)
d = mujoco.MjData(mjm)
for _ in range(400): mujoco.mj_step(mjm, d)
print("host ncon after 400 steps at rest:", int(d.ncon), flush=True)
dev = copy.deepcopy(mjm)
dev.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD | mujoco.mjtDisableBit.mjDSBL_NATIVECCD
try:
    m = mjw.put_model(dev); print("device put_model accepted margin with MULTICCD+NATIVECCD off", flush=True)
except Exception as x:
    print("device put_model refused:", str(x)[:160], flush=True)

def run_host(model, steps=400, tail=100):
    dd = mujoco.MjData(model); n = []
    for k in range(steps):
        mujoco.mj_step(model, dd)
        if k >= steps - tail: n.append(int(dd.ncon))
    return float(np.mean(n)), int(n[-1])

def run_device(model_dev, steps=400, tail=100):
    dd = mujoco.MjData(model_dev)
    md = mjw.put_model(model_dev); dv = mjw.put_data(model_dev, dd, nworld=1)
    n = []
    for k in range(steps):
        mjw.step(md, dv)
        if k >= steps - tail: n.append(int(dv.nacon.numpy()[0]))
    return float(np.mean(n)), int(n[-1])

zero = copy.deepcopy(mjm); zero.geom_margin[:] = 0.0
print("host margin kept   mean/last ncon (last 100 steps):", run_host(mjm), flush=True)
print("host margin zeroed mean/last ncon:", run_host(zero), flush=True)
print("device margin kept, MULTICCD+NATIVECCD off mean/last nacon:", run_device(dev), flush=True)
