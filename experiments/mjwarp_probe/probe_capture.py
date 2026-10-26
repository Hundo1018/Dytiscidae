# D-A3 (IMPL_1010_SPEC §4): capture wall of an unrolled N-step graph vs one per-step graph launched M times; run per XML.
import sys, time, mujoco, mujoco_warp as mjw, warp as wp
path, nworld, N, M = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
mjm = mujoco.MjModel.from_xml_path(path); mjm.geom_margin[:] = 0.0; mjm.opt.gravity[2] = 0.0
m = mjw.put_model(mjm); d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=nworld, njmax=512)
mjw.step(m, d); wp.synchronize()                                # compile outside the timings
def cap(n):
    t = time.perf_counter()
    with wp.ScopedCapture() as c:
        for _ in range(n): mjw.step(m, d)
    wp.synchronize(); return c.graph, time.perf_counter() - t
g1, t1 = cap(1); t = time.perf_counter()
for _ in range(M): wp.capture_launch(g1)
wp.synchronize(); tl1 = time.perf_counter() - t
gN, tN = cap(N); t = time.perf_counter(); wp.capture_launch(gN); wp.synchronize(); tlN = time.perf_counter() - t
print(f"{path.split('/')[-1]} nworld={nworld} capture_1={t1*1e3:.2f}ms launch_1x{M}={tl1*1e3:.1f}ms "
      f"capture_{N}={tN*1e3:.1f}ms launch_{N}={tlN*1e3:.1f}ms", flush=True)
