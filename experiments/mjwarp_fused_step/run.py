#!/usr/bin/env python3
"""Q2 -- the fused rigid + control step in MuJoCo Warp (ROADMAP 2026-10-10, later: an evaluation that
never leaves the device; probe Q2).

Frozen prediction (verbatim from docs/ROADMAP.md): "*Prediction:* `a` <= 0.5 ms, `b` <= 2 us per
world. *Falsified* if `a` > 1 ms or `b` > 5 us."  T_step(W) = a + b*W is fitted over W in (24, 100, 730).
Classification used here, per variant: CONFIRMED if a <= 0.5 ms and b <= 2 us; REFUTED if a > 1 ms or
b > 5 us; NEITHER otherwise (the ROADMAP leaves the middle unnamed; it is reported, not decided).

Runs in the SCRATCH venv (never the project .venv for Warp):

    ../mjwarp-venv/bin/python experiments/mjwarp_fused_step/run.py --selftest           # CPU, no GPU
    ../mjwarp-venv/bin/python experiments/mjwarp_fused_step/run.py                      # GPU, writes JSON + Q2_result.md

What one control callback does (ONE kernel launch per step, inside mjw.step, between fwd_velocity and
fwd_actuation; mujoco_warp 3.15.0 forward.py:2081):
  (a) per world, assemble a 33-float observation from device fields: root cvel (6, rot:lin), gravity
      direction in the root frame (3, from xmat of body 1), then 12 joint qpos and 12 joint qvel (guarded);
  (b) 33-64-64-6 tanh MLP, random weights, fixed seed, weights in device arrays shared by all worlds;
  (c) CPG: ctrl[a] = amp[a]*sin(2 pi freq[a] t + phase[w,a]) + 0.5*mlp[a % 6], written to d.ctrl;
  (d) placeholder body-frame force for EVERY body: xfrc_applied[w,b] = (-0.01*cvel_lin, -0.001*cvel_ang)
      (force first, torque second: support.py _apply_ft reads spatial_top as force).
Two MLP implementations of the same maths: `tile` (wp.tile / tile_matmul, one block of BLK threads =
BLK worlds) and `loop` (one thread per world, plain loops over global weights). `--mlp auto` (default)
uses tile and falls back to loop if the tile kernel fails to build or launch; the one used is recorded.

Variants per (fixture, W), all 2000 steps, median of 3 after a warm-up replay:
  cb_perstep   callback installed, one step captured in a graph, replayed 2000 times (N9's method)
  cb_rollout   callback installed, the whole rollout in ONE graph launch (wp.capture_while on a device
               counter; falls back to a single unrolled 2000-step graph, recorded in `rollout_mode`)
  nocb_perstep / nocb_rollout   the same without the callback (N9's number), for the delta
"""
import argparse, json, math, os, statistics, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PROBE = os.path.join(REPO, "experiments", "mjwarp_probe")
FIXTURES = {"median": os.path.join(PROBE, "elite_median_dof.xml"),
            "rotor_median": os.path.join(PROBE, "elite_rotor_median_dof.xml")}
WORLDS = (24, 100, 730)
STEPS = 2000
REPS = 3
NJMAX = 512          # N9: MJWarp's default (64 per world) overflows once the body lands
DIM_IN, DIM_H, DIM_OUT = 33, 64, 6
SEED = 20261010
FROZEN = "*Prediction:* `a` <= 0.5 ms, `b` <= 2 us per world. *Falsified* if `a` > 1 ms or `b` > 5 us."

ap = argparse.ArgumentParser()
ap.add_argument("--selftest", action="store_true", help="CPU device, W=2, 3 steps, no timing, no files")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--fixtures", default="median,rotor_median")
ap.add_argument("--worlds", default=",".join(map(str, WORLDS)))
ap.add_argument("--steps", type=int, default=STEPS)
ap.add_argument("--reps", type=int, default=REPS)
ap.add_argument("--mlp", choices=("auto", "tile", "loop"), default="auto")
ap.add_argument("--blk", type=int, default=32, help="threads per block of the tile kernel (worlds per block)")
ap.add_argument("--gravity", type=int, default=0, help="0 = flight, no contacts (default); 1 = N9's falling-onto-seabed case")
ap.add_argument("--out", default=os.path.join(HERE, "results_q2.json"))
ap.add_argument("--report", default=os.path.join(REPO, "runs", "analysis_1010_failure_theory", "Q2_result.md"))
ap.add_argument("--command", default="", help="exact command line, for the report (default: sys.argv)")
args = ap.parse_args()

import warp as wp
import mujoco
import mujoco_warp as mjw

BLK = args.blk
wp.config.enable_cpu_blocks = True   # CPU tile blocks > 1 thread are opt-in (context.py:_resolve_launch_block_dim); only matters for --selftest
vec_in = wp.types.vector(length=DIM_IN, dtype=float)
vec_h = wp.types.vector(length=DIM_H, dtype=float)
vec_out = wp.types.vector(length=DIM_OUT, dtype=float)
NIN = wp.constant(DIM_IN)
NH = wp.constant(DIM_H)
NOUT = wp.constant(DIM_OUT)
NBLK = wp.constant(BLK)


@wp.func
def tanh_f(x: float):
    return wp.tanh(x)


@wp.func
def build_obs(w: int, nq: int, nv: int, cvel: wp.array2d[wp.spatial_vector], xmat: wp.array2d[wp.mat33],
              qpos: wp.array2d[float], qvel: wp.array2d[float]):
    """33 floats: root cvel (6), gravity dir in root frame (3), 12 qpos[7:], 12 qvel[6:] (zero past the end)."""
    o = vec_in()
    cv = cvel[w, 1]
    for i in range(6):
        o[i] = cv[i]
    R = xmat[w, 1]
    # R^T * (0,0,-1) = -(row 2 of R)
    o[6] = -R[2, 0]
    o[7] = -R[2, 1]
    o[8] = -R[2, 2]
    for k in range(12):
        j = 7 + k
        if j < nq:
            o[9 + k] = qpos[w, j]
        j2 = 6 + k
        if j2 < nv:
            o[21 + k] = qvel[w, j2]
    return o


@wp.func
def post(w: int, nu: int, nbody: int, out: vec_out, time_w: float, amp: wp.array[float], freq: wp.array[float],
         phase: wp.array2d[float], cvel: wp.array2d[wp.spatial_vector], ctrl: wp.array2d[float],
         xfrc: wp.array2d[wp.spatial_vector]):
    for a in range(nu):
        ctrl[w, a] = amp[a] * wp.sin(6.2831853 * freq[a] * time_w + phase[w, a]) + 0.5 * out[a % NOUT]
    for b in range(nbody):
        cv = cvel[w, b]
        xfrc[w, b] = wp.spatial_vector(-0.01 * cv[3], -0.01 * cv[4], -0.01 * cv[5],
                                       -0.001 * cv[0], -0.001 * cv[1], -0.001 * cv[2])


@wp.kernel(module="unique")   # own module: tile shapes are fixed at codegen by the launch block_dim
def policy_tile(nworld: int, nu: int, nq: int, nv: int, nbody: int,
                cvel: wp.array2d[wp.spatial_vector], xmat: wp.array2d[wp.mat33],
                qpos: wp.array2d[float], qvel: wp.array2d[float], time: wp.array[float],
                amp: wp.array[float], freq: wp.array[float], phase: wp.array2d[float],
                W0: wp.array2d[float], B0: wp.array2d[float], W1: wp.array2d[float], B1: wp.array2d[float],
                W2: wp.array2d[float], B2: wp.array2d[float],
                ctrl: wp.array2d[float], xfrc: wp.array2d[wp.spatial_vector]):
    tid = wp.tid()
    w = wp.min(tid, nworld - 1)        # padded threads of the last block recompute world nworld-1, never write
    f = wp.tile(build_obs(w, nq, nv, cvel, xmat, qpos, qvel))                      # (33, BLK)
    w0 = wp.tile_load(W0, shape=(NH, NIN))
    b0 = wp.tile_load(B0, shape=(NH, 1))
    z = wp.tile_map(tanh_f, wp.tile_matmul(w0, f) + wp.tile_broadcast(b0, shape=(NH, NBLK)))
    w1 = wp.tile_load(W1, shape=(NH, NH))
    b1 = wp.tile_load(B1, shape=(NH, 1))
    z = wp.tile_map(tanh_f, wp.tile_matmul(w1, z) + wp.tile_broadcast(b1, shape=(NH, NBLK)))
    w2 = wp.tile_load(W2, shape=(NOUT, NH))
    b2 = wp.tile_load(B2, shape=(NOUT, 1))
    o = wp.tile_map(tanh_f, wp.tile_matmul(w2, z) + wp.tile_broadcast(b2, shape=(NOUT, NBLK)))
    out = wp.untile(o)
    if tid < nworld:
        post(w, nu, nbody, out, time[w], amp, freq, phase, cvel, ctrl, xfrc)


@wp.kernel
def policy_loop(nworld: int, nu: int, nq: int, nv: int, nbody: int,
                cvel: wp.array2d[wp.spatial_vector], xmat: wp.array2d[wp.mat33],
                qpos: wp.array2d[float], qvel: wp.array2d[float], time: wp.array[float],
                amp: wp.array[float], freq: wp.array[float], phase: wp.array2d[float],
                W0: wp.array2d[float], B0: wp.array2d[float], W1: wp.array2d[float], B1: wp.array2d[float],
                W2: wp.array2d[float], B2: wp.array2d[float],
                ctrl: wp.array2d[float], xfrc: wp.array2d[wp.spatial_vector]):
    w = wp.tid()
    o = build_obs(w, nq, nv, cvel, xmat, qpos, qvel)
    h1 = vec_h()
    for i in range(NH):
        s = B0[i, 0]
        for j in range(NIN):
            s += W0[i, j] * o[j]
        h1[i] = wp.tanh(s)
    h2 = vec_h()
    for i in range(NH):
        s = B1[i, 0]
        for j in range(NH):
            s += W1[i, j] * h1[j]
        h2[i] = wp.tanh(s)
    out = vec_out()
    for i in range(NOUT):
        s = B2[i, 0]
        for j in range(NH):
            s += W2[i, j] * h2[j]
        out[i] = wp.tanh(s)
    post(w, nu, nbody, out, time[w], amp, freq, phase, cvel, ctrl, xfrc)


@wp.kernel
def dec_counter(counter: wp.array[int]):
    counter[0] = counter[0] - 1


def make_params(nworld, nu, device):
    rng = np.random.default_rng(SEED)
    def lin(nout, nin):
        s = 1.0 / math.sqrt(nin)
        return (rng.uniform(-s, s, (nout, nin)).astype(np.float32), rng.uniform(-s, s, (nout, 1)).astype(np.float32))
    host = {}
    host["W0"], host["B0"] = lin(DIM_H, DIM_IN)
    host["W1"], host["B1"] = lin(DIM_H, DIM_H)
    host["W2"], host["B2"] = lin(DIM_OUT, DIM_H)
    host["amp"] = np.full(nu, 0.4, np.float32)
    host["freq"] = (1.0 + 0.1 * np.arange(nu)).astype(np.float32)
    host["phase"] = rng.uniform(0, 2 * math.pi, (nworld, nu)).astype(np.float32)   # different per world
    return host, {k: wp.array(v, dtype=float, device=device) for k, v in host.items()}


class Policy:
    """Holds the device arrays and installs/uninstalls the control callback on a Model."""

    def __init__(self, mjm, nworld, device, mlp):
        self.host, self.dev = make_params(nworld, mjm.nu, device)
        self.nu, self.nq, self.nv, self.nbody = int(mjm.nu), int(mjm.nq), int(mjm.nv), int(mjm.nbody)
        self.nworld, self.device, self.mlp = nworld, device, mlp
        self.nlaunch = 0

    def kernel_inputs(self, d):
        P = self.dev
        return [self.nworld, self.nu, self.nq, self.nv, self.nbody, d.cvel, d.xmat, d.qpos, d.qvel, d.time,
                P["amp"], P["freq"], P["phase"], P["W0"], P["B0"], P["W1"], P["B1"], P["W2"], P["B2"],
                d.ctrl, d.xfrc_applied]

    def callback(self, m, d):
        self.nlaunch += 1
        if self.mlp == "tile":
            wp.launch(policy_tile, dim=int(math.ceil(self.nworld / BLK)) * BLK, block_dim=BLK,
                      inputs=self.kernel_inputs(d), device=self.device)
        else:
            wp.launch(policy_loop, dim=self.nworld, inputs=self.kernel_inputs(d), device=self.device)

    def install(self, m):
        m.callback.control = self.callback

    @staticmethod
    def uninstall(m):
        m.callback.control = None


def numpy_reference(host, d, nu, nq, nv):
    """Independent numpy evaluation of obs -> MLP -> ctrl from the device fields (after mjw.forward)."""
    cvel, xmat = d.cvel.numpy(), d.xmat.numpy()
    qpos, qvel, t = d.qpos.numpy(), d.qvel.numpy(), d.time.numpy()
    nw = cvel.shape[0]
    obs = np.zeros((nw, DIM_IN), np.float64)
    obs[:, 0:6] = cvel[:, 1]
    obs[:, 6:9] = -xmat[:, 1, 2, :]
    for k in range(12):
        if 7 + k < nq:
            obs[:, 9 + k] = qpos[:, 7 + k]
        if 6 + k < nv:
            obs[:, 21 + k] = qvel[:, 6 + k]
    h = np.tanh(obs @ host["W0"].T + host["B0"][:, 0])
    h = np.tanh(h @ host["W1"].T + host["B1"][:, 0])
    out = np.tanh(h @ host["W2"].T + host["B2"][:, 0])
    a = np.arange(nu)
    ctrl = host["amp"] * np.sin(2 * math.pi * host["freq"] * t[:, None] + host["phase"]) + 0.5 * out[:, a % DIM_OUT]
    return ctrl


def load_model(path, gravity):
    mjm = mujoco.MjModel.from_xml_path(path)
    mjm.geom_margin[:] = 0.0          # N9: MJWarp 3.15 rejects geom margin 0.001 on box/mesh pairs
    mjm.opt.gravity[2] = -9.80665 if gravity else 0.0
    return mjm


def build(mjm, nworld, device):
    mjd = mujoco.MjData(mjm)
    with wp.ScopedDevice(device):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=nworld, njmax=NJMAX)
    return m, d


def reset(m, d, mjm, nworld, with_const_ctrl):
    mjw.reset_data(m, d)
    if with_const_ctrl:
        d.ctrl.assign(np.full((nworld, mjm.nu), 0.3, dtype=np.float32))


def checks(d, pol, mjm):
    """Cheap correctness evidence after a rollout: ctrl finite and different across worlds, xfrc nonzero."""
    ctrl, xf = d.ctrl.numpy(), d.xfrc_applied.numpy()
    qpos = d.qpos.numpy()
    nw = ctrl.shape[0]
    return {"ctrl_finite": bool(np.isfinite(ctrl).all()),
            "ctrl_std_across_worlds_max": float(np.nan_to_num(ctrl.std(axis=0)).max()),
            "ctrl_differs_across_worlds": bool(nw > 1 and float(np.nan_to_num(ctrl.std(axis=0)).max()) > 1e-6),
            "xfrc_nonzero_fraction": float(np.mean(np.abs(xf) > 0)),
            "xfrc_finite": bool(np.isfinite(xf).all()),
            "worlds_with_finite_qpos": int(np.isfinite(qpos).all(axis=1).sum()), "nworld": int(nw)}


def timed(f, device):
    wp.synchronize_device(device)
    t = time.perf_counter()
    f()
    wp.synchronize_device(device)
    return time.perf_counter() - t


def run_variant(mjm, nworld, device, mlp, callback, rollout, steps, reps, blk_loaded):
    """Returns (record, mlp_used)."""
    m, d = build(mjm, nworld, device)
    pol = None
    if callback:
        pol = Policy(mjm, nworld, device, mlp)
        pol.install(m)
    with wp.ScopedDevice(device):
        reset(m, d, mjm, nworld, not callback)
        mjw.step(m, d)                                   # warm-up, compiles/loads every module outside capture
        wp.synchronize_device(device)
        reset(m, d, mjm, nworld, not callback)
        mode = "per-step graph"
        if not rollout:
            with wp.ScopedCapture() as cap:
                mjw.step(m, d)
            def replay():
                for _ in range(steps):
                    wp.capture_launch(cap.graph)
        else:
            counter = wp.array([steps], dtype=wp.int32, device=device)
            err = None
            try:
                def body():
                    mjw.step(m, d)
                    wp.launch(dec_counter, dim=1, inputs=[counter], device=device)
                with wp.ScopedCapture() as cap:
                    wp.capture_while(counter, body)
                def replay():
                    counter.fill_(steps)
                    wp.capture_launch(cap.graph)
                mode = "capture_while"
                replay(); wp.synchronize_device(device)   # force instantiation errors to surface here
            except Exception as e:                        # noqa: BLE001 -- recorded, then fall back
                err = f"{type(e).__name__}: {e}"
                print(f"   capture_while failed ({err}); falling back to one unrolled {steps}-step graph", flush=True)
                reset(m, d, mjm, nworld, not callback)
                with wp.ScopedCapture() as cap:
                    for _ in range(steps):
                        mjw.step(m, d)
                def replay():
                    wp.capture_launch(cap.graph)
                mode = f"unrolled {steps}-step graph (capture_while failed: {err})"
        reset(m, d, mjm, nworld, not callback); replay()          # warm-up replay
        ts = []
        for _ in range(reps):
            reset(m, d, mjm, nworld, not callback)
            ts.append(timed(replay, device))
        rec = {"wall_s": ts, "median_s": statistics.median(ts), "step_us": statistics.median(ts) / steps * 1e6,
               "world_steps_per_s": nworld * steps / statistics.median(ts), "mode": mode}
        if callback:
            rec["checks"] = checks(d, pol, mjm)
    return rec


def fit(Ws, step_us):
    """T = a + b W by least squares; a in ms, b in us per world."""
    A = np.vstack([np.ones(len(Ws)), np.array(Ws, float)]).T
    coef, res, *_ = np.linalg.lstsq(A, np.array(step_us, float), rcond=None)
    pred = A @ coef
    ss_res = float(((np.array(step_us) - pred) ** 2).sum())
    ss_tot = float(((np.array(step_us) - np.mean(step_us)) ** 2).sum())
    return {"a_ms": float(coef[0]) / 1e3, "b_us": float(coef[1]), "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else None,
            "points_us": list(map(float, step_us)), "W": list(Ws)}


def classify(a_ms, b_us):
    if a_ms > 1.0 or b_us > 5.0:
        return "REFUTED"
    if a_ms <= 0.5 and b_us <= 2.0:
        return "CONFIRMED"
    return "NEITHER"


def pick_mlp(mjm, device, requested):
    """tile unless it fails to compile/launch on this device (probe on W=BLK, outside any capture)."""
    if requested != "auto":
        return requested, None
    try:
        m, d = build(mjm, BLK, device)
        pol = Policy(mjm, BLK, device, "tile")
        with wp.ScopedDevice(device):
            wp.load_module(module=policy_tile.module, device=device, block_dim=BLK)
            mjw.forward(m, d)
            pol.callback(m, d)
            wp.synchronize_device(device)
        return "tile", None
    except Exception as e:                                 # noqa: BLE001
        return "loop", f"{type(e).__name__}: {e}"


def selftest():
    wp.init()
    device = "cpu"
    mjm = load_model(FIXTURES["median"], gravity=0)
    results = {}
    nworld, nsteps = 2, 3
    for mlp in ("tile", "loop"):
        m, d = build(mjm, nworld, device)
        pol = Policy(mjm, nworld, device, mlp)
        pol.install(m)
        with wp.ScopedDevice(device):
            wp.load_module(module=policy_tile.module, device=device, block_dim=BLK)
            mjw.reset_data(m, d)
            for _ in range(nsteps):
                mjw.step(m, d)
            mjw.forward(m, d)                            # callback on the final state, so the reference can be rebuilt
            ctrl = d.ctrl.numpy().copy()
            ref = numpy_reference(pol.host, d, mjm.nu, mjm.nq, mjm.nv)
            xf = d.xfrc_applied.numpy()
            cvel = d.cvel.numpy()
            exp_f = -0.01 * cvel[:, :, 3:6]
        c = checks(d, pol, mjm)
        err = float(np.abs(ctrl - ref).max())
        ferr = float(np.abs(xf[:, :, 0:3] - exp_f).max())
        results[mlp] = dict(c, max_abs_ctrl_err_vs_numpy=err, max_abs_xfrc_err=ferr, callback_launches=pol.nlaunch, ctrl=ctrl)
        print(f"  mlp={mlp}: launches={pol.nlaunch} ctrl_finite={c['ctrl_finite']} differs_across_worlds={c['ctrl_differs_across_worlds']} "
              f"xfrc_nonzero_frac={c['xfrc_nonzero_fraction']:.2f} max|ctrl-numpy|={err:.2e} max|xfrc-expected|={ferr:.2e}", flush=True)
    both = float(np.abs(results["tile"]["ctrl"] - results["loop"]["ctrl"]).max())
    ok = all(r["ctrl_finite"] and r["ctrl_differs_across_worlds"] and r["xfrc_nonzero_fraction"] > 0 and r["xfrc_finite"]
             and r["max_abs_ctrl_err_vs_numpy"] < 1e-4 and r["max_abs_xfrc_err"] < 1e-5 for r in results.values()) and both < 1e-4
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}: device=cpu (MJWarp runs on the Warp CPU device, so physics is mjw.step, "
          f"not plain mujoco), W={nworld}, steps={nsteps}, fixture=median, tile-vs-loop max|dctrl|={both:.2e}, "
          f"tile_max_err_vs_numpy={results['tile']['max_abs_ctrl_err_vs_numpy']:.2e}, "
          f"loop_max_err_vs_numpy={results['loop']['max_abs_ctrl_err_vs_numpy']:.2e}", flush=True)
    return 0 if ok else 1


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:                                      # noqa: BLE001
        return "unknown"


def write_report(res, path):
    sha = res["commit"]
    L = []
    L.append(f"# Q2 — the fused rigid + control step in MJWarp, {res['date']}, commit {sha}\n")
    L.append("## 6 What ran")
    L.append(f"command:        {res['command']}")
    L.append(f"commit:         {sha} (working tree may carry uncommitted changes; the script is experiments/mjwarp_fused_step/run.py)")
    L.append(f"seed(s):        weights/CPG phases numpy default_rng({SEED}); same for every variant")
    L.append("inputs:         experiments/mjwarp_probe/elite_median_dof.xml, elite_rotor_median_dof.xml (N9 fixtures; geom margin zeroed, njmax 512)")
    L.append(f"wall:           {res['wall_s']:.0f}")
    L.append(f"effective config: device {res['device']}; gravity {'on (falls onto seabed)' if res['gravity'] else 'off (flight, no contacts)'}; "
             f"MLP kernel actually used: {res['mlp_used']}"
             + (f" (tile failed: {res['mlp_fallback_reason']})" if res.get("mlp_fallback_reason") else "")
             + f"; rollout graph mode: {', '.join(sorted(set(res['rollout_modes'])))}; steps {res['steps']}, reps {res['reps']}, BLK {res['blk']}\n")
    L.append("## 7 Data quality")
    nrec = sum(1 for _ in res["records"])
    L.append(f"n:              {nrec} (fixture, W, variant) cells, each the median of {res['reps']} timed {res['steps']}-step replays after one warm-up replay")
    L.append(f"dropped:        {res['dropped']}")
    bad = [f"{r['fixture']} W={r['W']} {r['variant']}" for r in res["records"] if r.get("checks") and not (
        r["checks"]["ctrl_finite"] and r["checks"]["ctrl_differs_across_worlds"] and r["checks"]["xfrc_nonzero_fraction"] > 0)]
    nonfin = [f"{r['fixture']} W={r['W']} {r['variant']}: {r['checks']['worlds_with_finite_qpos']}/{r['checks']['nworld']}"
              for r in res["records"] if r.get("checks") and r["checks"]["worlds_with_finite_qpos"] < r["checks"]["nworld"]]
    L.append(f"artifacts:      correctness checks (ctrl finite, differs across worlds, xfrc nonzero) failed in: {bad or 'none'}; "
             f"cells with non-finite qpos in some world: {nonfin or 'none'}\n")
    L.append("## 8 Numbers")
    Wl = res["worlds"]
    L.append("| fixture | variant | " + " | ".join(f"W={w} us/step" for w in Wl) + " | a (ms) | b (us/world) | R^2 | class |")
    L.append("|---|---|" + "---|" * len(Wl) + "---|---|---|---|")
    for fx, fv in res["fits"].items():
        for var, f in fv.items():
            p = f["points_us"]
            L.append(f"| {fx} | {var} | " + " | ".join(f"{x:.0f}" for x in p) + f" | {f['a_ms']:.3f} | {f['b_us']:.3f} | "
                     f"{f['r2'] if f['r2'] is None else round(f['r2'], 4)} | {f.get('class', '')} |")
    L.append("\ncallback delta (cb - nocb), same graph mechanism:\n")
    L.append("| fixture | mechanism | a_delta (ms) | b_delta (us/world) |")
    L.append("|---|---|---|---|")
    for fx, dv in res["delta_fits"].items():
        for mech, f in dv.items():
            L.append(f"| {fx} | {mech} | {f['a_ms']:.3f} | {f['b_us']:.3f} |")
    L.append("\nspread:         every cell is the median of the reps listed in the JSON (`wall_s`); min/max per cell:")
    for r in res["records"]:
        L.append(f"  {r['fixture']} W={r['W']} {r['variant']}: {min(r['wall_s'])/res['steps']*1e6:.0f}..{max(r['wall_s'])/res['steps']*1e6:.0f} us/step")
    L.append("\n## 9 Against the frozen prediction (quote it verbatim, then one word)")
    L.append(f"prediction:     \"{FROZEN}\"")
    L.append(f"outcome:        {res['outcome']}")
    L.append(f"by:             {res['by']}\n")
    L.append("## 12 Reproducer")
    L.append("```bash")
    L.append(res["command"])
    L.append("```")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    wp.init()
    if args.selftest:
        return selftest()
    t_all = time.perf_counter()
    device = args.device
    Ws = [int(x) for x in args.worlds.split(",")]
    res = {"date": time.strftime("%Y-%m-%d"), "commit": git_sha(), "command": args.command or " ".join(["../mjwarp-venv/bin/python"] + sys.argv),
           "device": str(wp.get_device(device)), "warp": wp.config.version, "mujoco": mujoco.__version__,
           "mujoco_warp": getattr(mjw, "__version__", "3.15.0"), "gravity": args.gravity, "steps": args.steps, "reps": args.reps,
           "worlds": Ws, "blk": BLK, "records": [], "fits": {}, "delta_fits": {}, "dropped": "none", "rollout_modes": []}
    mlp_used = None
    for fx in args.fixtures.split(","):
        mjm = load_model(FIXTURES[fx], args.gravity)
        res.setdefault("fixture_dims", {})[fx] = {"nbody": int(mjm.nbody), "nq": int(mjm.nq), "nv": int(mjm.nv), "nu": int(mjm.nu)}
        if mlp_used is None:
            mlp_used, why = pick_mlp(mjm, device, args.mlp)
            res["mlp_used"], res["mlp_fallback_reason"] = mlp_used, why
            print(f"MLP kernel: {mlp_used}" + (f" (tile failed: {why})" if why else ""), flush=True)
        tab = {}
        for W in Ws:
            for var, (cb, ro) in {"cb_perstep": (True, False), "cb_rollout": (True, True),
                                  "nocb_perstep": (False, False), "nocb_rollout": (False, True)}.items():
                try:
                    rec = run_variant(mjm, W, device, mlp_used, cb, ro, args.steps, args.reps, True)
                except Exception as e:                    # noqa: BLE001
                    rec = None
                    res["dropped"] = (res["dropped"] if res["dropped"] != "none" else "") + f"{fx} W={W} {var}: {type(e).__name__}: {e}; "
                    print(f"{fx} W={W} {var}: FAILED {type(e).__name__}: {e}", flush=True)
                if rec is None:
                    continue
                if ro:
                    res["rollout_modes"].append(rec["mode"])
                rec.update({"fixture": fx, "W": W, "variant": var})
                res["records"].append(rec)
                tab[(W, var)] = rec["step_us"]
                print(f"{fx} W={W:4d} {var:13s}: {rec['step_us']:8.1f} us/step (reps {[round(x / args.steps * 1e6, 1) for x in rec['wall_s']]})"
                      f"  [{rec['mode']}]" + (f"  checks {rec['checks']}" if rec.get('checks') else ""), flush=True)
        res["fits"][fx] = {}
        for var in ("cb_perstep", "cb_rollout", "nocb_perstep", "nocb_rollout"):
            pts = [(W, tab[(W, var)]) for W in Ws if (W, var) in tab]
            if len(pts) >= 2:
                f = fit([p[0] for p in pts], [p[1] for p in pts])
                f["class"] = classify(f["a_ms"], f["b_us"])
                res["fits"][fx][var] = f
                print(f"FIT {fx} {var}: a = {f['a_ms']:.3f} ms, b = {f['b_us']:.3f} us/world, R2 = {f['r2']}  -> {f['class']}", flush=True)
        res["delta_fits"][fx] = {}
        for mech in ("perstep", "rollout"):
            pts = [(W, tab[(W, "cb_" + mech)] - tab[(W, "nocb_" + mech)]) for W in Ws if (W, "cb_" + mech) in tab and (W, "nocb_" + mech) in tab]
            if len(pts) >= 2:
                res["delta_fits"][fx][mech] = fit([p[0] for p in pts], [p[1] for p in pts])
    res["wall_s"] = time.perf_counter() - t_all
    # headline: the callback variants, per fixture; per-step captured graph first (the frozen text: "inside the captured graph")
    cl = {(fx, v): f["class"] for fx, fv in res["fits"].items() for v, f in fv.items() if v.startswith("cb_")}
    heads = [cl[(fx, "cb_perstep")] for fx in res["fits"] if (fx, "cb_perstep") in cl]
    res["outcome"] = "REFUTED" if "REFUTED" in heads else ("NEITHER" if "NEITHER" in heads else ("CONFIRMED" if heads else "NOT COMPARABLE"))
    res["by"] = "; ".join(f"{fx} {v}: a = {f['a_ms']:.3f} ms, b = {f['b_us']:.3f} us -> {f['class']}"
                          for fx, fv in res["fits"].items() for v, f in fv.items() if v.startswith("cb_"))
    res["outcome_rule"] = ("headline = the per-step captured-graph callback variant on both fixtures (worst class wins); "
                           "the rollout variant is reported beside it")
    json.dump(res, open(args.out, "w"), indent=1)
    write_report(res, args.report)
    print("wrote", args.out, "and", args.report, "\nOUTCOME:", res["outcome"], "|", res["by"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
