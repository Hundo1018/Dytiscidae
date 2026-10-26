# Is body_subtreemass a batchable (*) field in MuJoCo Warp 3.15? put_model with a per-world batch of it.
import mujoco, mujoco_warp as mjw
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import TriphibianEnv
e = TriphibianEnv(build(BODY_PLANS["ray"]()), seed=0)
import copy
dev = copy.deepcopy(e.model)
dev.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD | mujoco.mjtDisableBit.mjDSBL_NATIVECCD
try:
    m = mjw.put_model(dev, batch_sizes={"body_subtreemass": 2})
    print("batched body_subtreemass shape", m.body_subtreemass.shape, flush=True)
except Exception as x:
    print("REFUSED", type(x).__name__, str(x)[:200], flush=True)
