import sys, math, numpy as np
sys.path.insert(0,"experiments/flapper")
from mission import Machine
from dytiscidae.core.mjcf import SHORE_X, BEACH_SLOPE
print("shore", SHORE_X, "slope", BEACH_SLOPE)
for x in (SHORE_X+5, SHORE_X+20):
    mc = Machine("land", pose=dict(x=x, z=5.0, pitch_deg=0.0)); mc.drop_to_ground(0.01)
    z0=mc.d.qpos[2]
    for k in range(20000): mc.step(np.zeros(len(mc.aid)))
    print("x %.1f ground %.3f: dz %.4f dx %.4f ncon %d load %.3f pitch %.2f qacc %d" % (x, mc.env.ground_height(x), mc.d.qpos[2]-z0, mc.d.qpos[0]-x, mc.d.ncon, mc.contact_load(), mc.pitch_deg(), mc.bad_qacc))
