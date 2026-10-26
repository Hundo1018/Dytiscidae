# D-A1 (IMPL_1010_SPEC §4): cost of DeviceBody.fluid_and_scatter (no mjw.step) as one CUDA graph, W = 24/100/730/1600,
# float32 and float64, median (beetle) and rotor fixtures. "fluid" = launches of fluid_wp kernels; "scatter" = the rest
# (servo, damping, jets, rotors, added mass into model, diagnostics, writes). Fluid-only graph = same step with every
# non-fluid_wp launch skipped; scatter = full - fluid. Run: MUJOCO_GL=disable PYTHONPATH=. python probe_d_a1.py
import gc, statistics as st, sys, time
import numpy as np, mujoco, warp as wp
from dytiscidae.core.bodyplans import BODY_PLANS
from dytiscidae.core.phenotype import build
from dytiscidae.envs.triphibian import Domain, TriphibianEnv
from dytiscidae.envs.devroll.model_wp import DeviceBody
from dytiscidae.envs.devroll import fluid_wp
from tests.test_devroll import _rotor_machine, _place_rotor_machine, _round_to_float32

WS, REPS, INNER, WARM = (24, 100, 730, 1600), 7, 50, 30

def median_env():
    rng = np.random.default_rng(1)
    e = TriphibianEnv(build(BODY_PLANS["beetle"]()), seed=0)
    e.reset(Domain.WATER, randomise=False)
    e.data.qvel[:] = rng.normal(0.0, 1.5, e.model.nv)
    mujoco.mj_forward(e.model, e.data); e.data.qpos[2] += -0.3 - e.data.xpos[1, 2]
    mujoco.mj_forward(e.model, e.data)
    return e
def rotor_env():
    e = _rotor_machine(0); _place_rotor_machine(e, -0.5, np.random.default_rng(67)); return e

real_launch = wp.launch
def capture(body, fluid_only):
    fl = {id(k) for k in fluid_wp.make_kernels(body.dtype).values() if hasattr(k, "key")}
    n = [0]
    def patched(kernel, *a, **kw):
        if fluid_only and id(kernel) not in fl: return None
        n[0] += 1; return real_launch(kernel, *a, **kw)
    wp.launch = patched
    try:
        with wp.ScopedCapture() as c: body.fluid_and_scatter()
    finally: wp.launch = real_launch
    wp.synchronize(); return c.graph, n[0]

def time_graph(g):
    for _ in range(WARM): wp.capture_launch(g)
    wp.synchronize(); out = []
    for _ in range(REPS):
        t = time.perf_counter()
        for _ in range(INNER): wp.capture_launch(g)
        wp.synchronize(); out.append((time.perf_counter() - t) / INNER * 1e3)
    return out

rows = {}
for fx, make in (("median", median_env), ("rotor", rotor_env)):
    env = make(); _round_to_float32([env])
    for dt, dn in ((wp.float32, "f32"), (wp.float64, "f64")):
        for W in WS:
            key = (fx, dn, W)
            try:
                body = DeviceBody([env] * W, dtype=dt)
                body.load_state([e.data for e in body.envs]); body.step(); body.reset()
                body.load_state([e.data for e in body.envs]); body.gather()
                npan = body.npan
                gf, nf = capture(body, True); gall, na = capture(body, False)
                tf, ta = time_graph(gf), time_graph(gall)
                rows[key] = dict(npan=npan, nf=nf, na=na, fl=st.median(tf), fl_rng=(min(tf), max(tf)),
                                 all=st.median(ta), all_rng=(min(ta), max(ta)))
            except Exception as ex:
                rows[key] = dict(err=f"{type(ex).__name__}: {str(ex)[:120]}")
            body = gf = gall = None; gc.collect(); wp.synchronize()
            print("done", key, rows[key], file=sys.stderr, flush=True)

print("\nfixture prec    W  npan/w launches(f/all) fluid_ms [min-max]  total_ms [min-max]  scatter_ms scat/fluid  us/panel(total)")
for (fx, dn, W), r in rows.items():
    if "err" in r: print(f"{fx:6s} {dn} {W:5d}  NOT RUN: {r['err']}"); continue
    sc = r["all"] - r["fl"]
    print(f"{fx:6s} {dn} {W:5d} {r['npan']:5d}  {r['nf']:3d}/{r['na']:3d}  {r['fl']:8.4f} [{r['fl_rng'][0]:.4f}-{r['fl_rng'][1]:.4f}]  "
          f"{r['all']:8.4f} [{r['all_rng'][0]:.4f}-{r['all_rng'][1]:.4f}]  {sc:8.4f}  {sc/r['fl']:5.2f}  {r['all']*1e3/(W*r['npan']):.4f}")
print("\nmarginal us/panel between W=730 and 1600 (fluid-only / total), and us per launch at W=24 (total)")
for fx in ("median", "rotor"):
    for dn in ("f32", "f64"):
        a, b = rows.get((fx, dn, 730), {}), rows.get((fx, dn, 1600), {})
        if "err" in a or "err" in b or not a or not b: print(fx, dn, "n/a"); continue
        dW = (1600 - 730) * a["npan"]
        s = rows[(fx, dn, 24)]
        print(f"{fx} {dn} marginal fluid {(b['fl']-a['fl'])*1e3/dW:.4f}  total {(b['all']-a['all'])*1e3/dW:.4f}  "
              f"launch@W24 {s['all']*1e3/s['na']:.1f} us/launch")
