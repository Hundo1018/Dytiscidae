# N9 step-1 feasibility probe (<=10 lines). Scratch venv only: ../mjwarp-venv/bin/python probe10.py <xml> <device>
import sys, time, mujoco, warp as wp, mujoco_warp as mjw
wp.set_device(sys.argv[2])
mjm = mujoco.MjModel.from_xml_path(sys.argv[1]); mjd = mujoco.MjData(mjm)
if len(sys.argv) > 3 and sys.argv[3] == 'noccd': mjm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD  # workaround 1
if len(sys.argv) > 3 and sys.argv[3] == 'margin0': mjm.geom_margin[:] = 0.0  # workaround 2: the project's geom margin=0.001 is rejected
m = mjw.put_model(mjm); d = mjw.put_data(mjm, mjd, nworld=24)
mjw.step(m, d); wp.synchronize()                      # compile
t = time.perf_counter()
for _ in range(300): mjw.step(m, d)
wp.synchronize(); print("24 worlds x 300 steps", time.perf_counter() - t, "s")
