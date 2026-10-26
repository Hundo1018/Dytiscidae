# Case (d): does MJWarp read per-world body_mass for a jointless (welded) body?
# Measured 2026-10-26 (mujoco 3.15.0, mujoco_warp 3.15.0, warp 1.18.0, cuda:0):
# qM world0 vs world1 max abs diff 4.71 when the fin mass is x10 in world 1 -> yes, it reads per world.
import mujoco, mujoco_warp as mjw, warp as wp, numpy as np
XML = """<mujoco><worldbody><body name="base"><joint type="free"/><geom size=".1"/>
<body name="fin" pos="0 0 .2"><geom size=".05"/></body></body></worldbody></mujoco>"""
mjm = mujoco.MjModel.from_xml_string(XML); mjd = mujoco.MjData(mjm)
fin = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "fin")
m = mjw.put_model(mjm, batch_sizes={"body_mass": 2, "body_inertia": 2})
d = mjw.put_data(mjm, mjd, nworld=2)
bm = m.body_mass.numpy(); print("body_mass shape", bm.shape)
bm[1, fin] *= 10.0; m.body_mass.assign(bm)
mjw.forward(m, d)
qM = d.M.numpy(); print("qM world0 vs world1 max abs diff:", float(np.abs(qM[0]-qM[1]).max()))
