"""M2: the signs of the joints, read by forward kinematics on the built model (no rollout)."""
import sys, math, numpy as np
sys.path.insert(0, "experiments/flapper")
from m2_common import build, draw
mc = build("depth", draw(100), gains="here")
m, d, mj = mc.m, mc.d, mc.mj
d.qpos[3:7] = (1, 0, 0, 0); mj.mj_forward(m, d)
def geo(b):  # world corners of each wing's box: centre and a point at +x (forward) of the box
    g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == b][0]
    R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; sz = m.geom_size[g]
    return c, c + R[:, 0] * sz[0], c + R[:, 1] * sz[1]
names = [m.body(i).name for i in range(m.nbody)]
for n, j in enumerate(mc.jid):
    b = int(m.jnt_bodyid[j])
    for val in (0.0, 0.3):
        d.qpos[mc.qad] = 0.0; d.qpos[mc.qad[n]] = val; mj.mj_forward(m, d)
        c, ax0, ax1 = geo(b)
        print(names[b], mc.role[n], "q=%.1f" % val, "centre", (c - d.xpos[mc.hull]).round(3), "box +x end", (ax0 - d.xpos[mc.hull]).round(3), "box +y end", (ax1 - d.xpos[mc.hull]).round(3))
