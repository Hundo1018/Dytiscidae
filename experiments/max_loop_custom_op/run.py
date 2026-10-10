#!/usr/bin/env python3
"""Q8 -- a custom Mojo op inside `ops.while_loop` (ROADMAP 2026-10-10, "an
evaluation that never leaves the device", section 8, probe Q8).

Frozen text (verbatim, line breaks as in docs/ROADMAP.md):

    **Q8 (MAX graph, ~half a day): a custom Mojo op inside `ops.while_loop`.**
    The existing fluid kernel (or a stub of its shape: a thread-per-panel kernel
    over a `(W*B, panels)` buffer) registered as a MAX custom op and called once
    per iteration of a 2000-step `while_loop` beside a `(W*B) x 33 @ 33 x 64`
    matmul, buffers device-resident; W*B = 1,600. *Prediction:* <= 150 us per
    iteration (fixed ~50 us plus the two ops). *Falsified* above 500 us, or if a
    custom op cannot be placed inside the loop. If Q8 holds and Q2 says A is
    > 3x today's wall, B is built on MAX graph + Mojo custom ops rather than on
    hand-launched kernels.

What this script builds (all float32, device-resident, one `ops.while_loop`):
    inputs   obs (WB, 33), W (33, 64), state (WB, 50, 8)
    body     h  = tanh(obs @ W)                       (the matmul)
             h0 = h[:, 0:1]
             state' = stub_panel_op(state, h0)        (thread-per-panel; atan2f)
             obs'   = obs + onehot_col0 * 1e-3 * h0   (the one-element data
                                                       dependency: row i's obs[i,0])
    variants "base"    same loop, no custom op (state passed through)  -> 51 us shape
             "custom"  functional custom op, loop-carried state          -> the verdict
             "inplace" ops.inplace_custom on a BufferType input          -> secondary
Outputs stay on device; timing is 2000 iterations after one warm-up execute,
host wall time around execute() + device synchronize, repeated --reps times.

Readings applied (the request's, not changes to the text):
  * "a rotation of a 3-vector by a 3x3 stored in the state": a panel row has 8
    floats, which cannot hold a vector (3) and a 3x3 (9), so the row holds the
    vector, a unit quaternion and the output; R(q) is formed in registers.
  * "writing it back in place": the verdict variant is the functional op whose
    result the loop carries (MAX owns the buffers); the in-place op is measured
    too and reported beside it. The kernel itself reads and writes one row.
  * Outcome: us/iteration of "custom" <= 150 CONFIRMED, > 500 REFUTED, between
    NEITHER (written as "NOT COMPARABLE"-style word NEITHER, as the request said).
    "custom op cannot be placed in the loop" -> REFUTED when graph construction
    or loading of BOTH custom variants fails with a loop/custom-op error; any
    other failure is reported as NOT COMPARABLE (it says nothing about placement).

Usage (run from the mojo/ pixi env; never touches mojo/build/):

    cd mojo && pixi run python ../experiments/max_loop_custom_op/run.py --selftest   # CPU, ~1 min
    cd mojo && pixi run python ../experiments/max_loop_custom_op/run.py              # GPU, ~4 min predicted, timeout 900 s
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
KERNELS = HERE / "kernels"
DEFAULT_REPORT = REPO / "runs" / "analysis_1010_failure_theory" / "Q8_result.md"
DEFAULT_RESULTS = HERE / "results_q8.json"

FROZEN_FALLBACK = (
    "<= 150 us per iteration (fixed ~50 us plus the two ops). *Falsified* above 500 us, "
    "or if a custom op cannot be placed inside the loop."
)
CONFIRM_US = 150.0
REFUTE_US = 500.0
OBS_DIM, HID, PANELS, FEAT = 33, 64, 50, 8
EPS = 1e-3


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--selftest", action="store_true", help="CPU device, WB=4, 3 iterations, numpy reference check; no GPU")
    p.add_argument("--wb", type=int, default=1600)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--reps", type=int, default=5, help="timed executions per variant (after 1 warm-up)")
    p.add_argument("--variants", default="base,custom,inplace")
    p.add_argument("--check-iters", type=int, default=3, help="iterations of the GPU correctness graph")
    p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--report", default=str(DEFAULT_REPORT))
    p.add_argument("--results", default=str(DEFAULT_RESULTS))
    return p.parse_args(argv)


# ------------------------------------------------------------- numpy reference
def atan_unit_np(x):
    z = x * x
    f = np.float32
    return x * (f(0.9998660) + z * (f(-0.3302995) + z * (f(0.1801410) + z * (f(-0.0851330) + z * f(0.0208351)))))


def atan2f_np(y, x):
    """float32 port of mojo/src/mathx.mojo atan2f (the same polynomial and folds)."""
    f = np.float32
    pi, hpi = f(np.pi), f(np.pi / 2)
    ax, ay = np.abs(x), np.abs(y)
    swap = ay > ax
    num = np.where(swap, ax, ay)
    den = np.where(swap, ay, ax)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den != 0, num / np.where(den != 0, den, f(1)), f(0))
    r = np.where(den != 0, atan_unit_np(ratio.astype(np.float32)), f(0)).astype(np.float32)
    r = np.where(swap, hpi - r, r)
    r = np.where(x < 0, pi - r, r)
    r = np.where(y < 0, -r, r)
    return r.astype(np.float32)


def panel_step_np(state, gain):
    """state (WB, P, 8) float32, gain (WB, 1) float32 -> new state; same math as kernels/stub_panel_op.mojo."""
    f = np.float32
    v = state[..., 0:3]
    qw, qx, qy, qz = (state[..., 3], state[..., 4], state[..., 5], state[..., 6])
    one, two = f(1), f(2)
    R = np.stack([
        np.stack([one - two * (qy * qy + qz * qz), two * (qx * qy - qz * qw), two * (qx * qz + qy * qw)], -1),
        np.stack([two * (qx * qy + qz * qw), one - two * (qx * qx + qz * qz), two * (qy * qz - qx * qw)], -1),
        np.stack([two * (qx * qz - qy * qw), two * (qy * qz + qx * qw), one - two * (qx * qx + qy * qy)], -1),
    ], -2).astype(np.float32)
    w = np.einsum("...ij,...j->...i", R, v).astype(np.float32)
    c = np.cross(w, v).astype(np.float32)
    aoa = atan2f_np(w[..., 2], w[..., 0])
    out = (np.sqrt((c * c).sum(-1)) * aoa + gain).astype(np.float32)
    new = state.copy()
    new[..., 0:3] = w
    new[..., 7] = out
    return new, aoa


def reference_loop(obs, W, state, iters):
    obs, state = obs.copy(), state.copy()
    mask = np.zeros((1, OBS_DIM), np.float32)
    mask[0, 0] = 1.0
    for _ in range(iters):
        h = np.tanh(obs @ W).astype(np.float32)
        h0 = h[:, 0:1]
        state, _ = panel_step_np(state, h0)
        obs = (obs + mask * np.float32(EPS) * h0).astype(np.float32)
    return obs, state


def make_inputs(wb, seed):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((wb, OBS_DIM)).astype(np.float32)
    W = (rng.standard_normal((OBS_DIM, HID)) / np.sqrt(OBS_DIM)).astype(np.float32)
    state = np.zeros((wb, PANELS, FEAT), np.float32)
    state[..., 0:3] = rng.standard_normal((wb, PANELS, 3))
    q = rng.standard_normal((wb, PANELS, 4))
    state[..., 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return obs, W, state


# ----------------------------------------------------------------------- graph
def build_graph(variant, device, wb, iters):
    from max.dtype import DType
    from max.graph import BufferType, DeviceRef, Graph, TensorType, ops

    dref = DeviceRef.from_device(device)
    t_obs = TensorType(DType.float32, [wb, OBS_DIM], device=dref)
    t_w = TensorType(DType.float32, [OBS_DIM, HID], device=dref)
    t_state = TensorType(DType.float32, [wb, PANELS, FEAT], device=dref)
    inputs = [t_obs, t_w, BufferType(DType.float32, [wb, PANELS, FEAT], device=dref) if variant == "inplace" else t_state]
    mask_np = np.zeros((1, OBS_DIM), np.float32)
    mask_np[0, 0] = EPS

    with Graph(f"q8_{variant}", input_types=inputs, custom_extensions=[KERNELS]) as g:
        obs0, w0, state0 = g.inputs
        # the loop counter lives on the CPU (as in G's 50.9 us/iteration probe)
        i0 = ops.constant(0, DType.int32, device=DeviceRef.CPU())

        def predicate(i, *_):
            return i < iters

        def body_values(obs, w, state):
            h = ops.tanh(ops.matmul(obs, w))                 # (WB, 64)
            h0 = h[:, 0:1]                                   # (WB, 1)
            mask = ops.constant(mask_np, DType.float32, device=dref)
            obs_n = obs + mask * h0                          # one column moves: loop data dependency
            return h0, obs_n

        if variant == "inplace":
            state_buf = state0

            def body(i, obs, w):
                h0, obs_n = body_values(obs, w, None)
                ops.inplace_custom("stub_panel_op_inplace", device=dref, values=[state_buf, h0])
                return i + 1, obs_n, w

            r = ops.while_loop([i0, obs0, w0], predicate, body)
            g.output(r[1])
        else:
            def body(i, obs, w, state):
                h0, obs_n = body_values(obs, w, state)
                if variant == "custom":
                    state_n = ops.custom(
                        "stub_panel_op", device=dref, values=[state, h0], out_types=[t_state]
                    )[0].tensor
                else:
                    state_n = state
                return i + 1, obs_n, w, state_n

            r = ops.while_loop([i0, obs0, w0, state0], predicate, body)
            g.output(r[1], r[3])
    return g


def to_np(x):
    from max.driver import CPU

    return x.to(CPU()).to_numpy()


def run_variant(variant, device, session, wb, iters, reps, seed, check=True):
    """Build, compile, warm up, time. Returns a dict; never raises (the failure stage is recorded)."""
    from max.driver import Buffer

    res = {"variant": variant, "ok": False, "stage": "graph-build", "error": None}
    obs, W, state = make_inputs(wb, seed)

    def fresh():
        o = Buffer.from_numpy(obs).to(device)
        w = Buffer.from_numpy(W).to(device)
        s = Buffer.from_numpy(state.copy()).to(device)
        return o, w, s

    try:
        t = time.perf_counter()
        g = build_graph(variant, device, wb, iters)
        res["build_s"] = time.perf_counter() - t
        res["stage"] = "load/compile"
        t = time.perf_counter()
        model = session.load(g)
        res["compile_s"] = time.perf_counter() - t
        res["stage"] = "execute"
        o, w, s = fresh()
        model.execute(o, w, s)                      # warm-up (one full 2000-iteration execution)
        device.synchronize()
        times = []
        for _ in range(reps):
            o, w, s = fresh()
            device.synchronize()
            t = time.perf_counter()
            out = model.execute(o, w, s)
            device.synchronize()
            times.append(time.perf_counter() - t)
        res["total_s"] = times
        per = [1e6 * x / iters for x in times]
        res["us_per_iter"] = per
        res["us_median"] = statistics.median(per)
        res["us_min"], res["us_max"] = min(per), max(per)
        res["output_devices"] = [str(getattr(x, "device", "?")) for x in out]
        res["ok"], res["stage"] = True, "done"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:600]}"
        res["traceback_tail"] = traceback.format_exc()[-800:]
    return res


def check_variant(variant, device, session, wb, iters, seed):
    """Run a short graph and compare the on-device result with the numpy reference."""
    from max.driver import Buffer

    obs, W, state = make_inputs(wb, seed)
    out = {"variant": variant, "iters": iters, "wb": wb}
    try:
        model = session.load(build_graph(variant, device, wb, iters))
        o = Buffer.from_numpy(obs).to(device)
        w = Buffer.from_numpy(W).to(device)
        s = Buffer.from_numpy(state.copy()).to(device)
        r = model.execute(o, w, s)
        device.synchronize()
        got_obs = to_np(r[0])
        got_state = to_np(r[1]) if variant != "inplace" else to_np(s)
        ref_obs, ref_state = reference_loop(obs, W, state, iters)
        out["max_abs_state"] = float(np.abs(got_state - ref_state).max())
        out["max_abs_force"] = float(np.abs(got_state[..., 7] - ref_state[..., 7]).max())
        out["max_abs_obs"] = float(np.abs(got_obs - ref_obs).max())
        out["force_scale"] = float(np.abs(ref_state[..., 7]).max())
        out["finite"] = bool(np.isfinite(got_state).all())
        out["state_changed"] = bool(np.abs(got_state - state).max() > 1e-3)
        out["ok"] = bool(out["finite"] and out["state_changed"] and out["max_abs_state"] < 2e-3 and out["max_abs_obs"] < 2e-3)
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:600]}"
    return out


def atan2_error_vs_true(wb, seed):
    obs, W, state = make_inputs(wb, seed)
    _, aoa = panel_step_np(state, np.zeros((wb, 1), np.float32))
    v = state[..., 0:3]
    R_state, _ = panel_step_np(state, np.zeros((wb, 1), np.float32))
    w = R_state[..., 0:3]
    true = np.arctan2(w[..., 2].astype(np.float64), w[..., 0].astype(np.float64))
    return float(np.abs(aoa.astype(np.float64) - true).max())


# --------------------------------------------------------------------- verdict
def classify(res):
    """res: {'custom': variant dict, 'inplace': variant dict or None}. Returns (word, why)."""
    c = res.get("custom")
    if c is None or not c.get("ok"):
        errs = [(v or {}).get("error") or "not run" for v in (res.get("custom"), res.get("inplace"))]
        stage = (c or {}).get("stage")
        text = " | ".join(errs).lower()
        if stage in ("graph-build", "load/compile") and re.search(r"loop|while|custom|capture|buffer|chain", text):
            return "REFUTED", f"custom op could not be placed in the loop (stage {stage}): {errs[0][:200]}"
        return "NOT COMPARABLE", f"the custom variant did not run (stage {stage}): {errs[0][:200]}"
    us = c["us_median"]
    if us <= CONFIRM_US:
        return "CONFIRMED", f"custom-op loop {us:.1f} us/iteration <= {CONFIRM_US:.0f}"
    if us > REFUTE_US:
        return "REFUTED", f"custom-op loop {us:.1f} us/iteration > {REFUTE_US:.0f}"
    return "NEITHER", f"custom-op loop {us:.1f} us/iteration is between {CONFIRM_US:.0f} and {REFUTE_US:.0f}"


# ---------------------------------------------------------------------- report
def frozen_prediction():
    try:
        text = (REPO / "docs" / "ROADMAP.md").read_text()
        m = re.search(r"\*\*Q8 \(MAX graph.*?\*Prediction:\* (.*?)\s+If Q8 holds", text, re.S)
        if m:
            return " ".join(m.group(1).split())
    except OSError:
        pass
    return FROZEN_FALLBACK


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def fmt(x, nd=1):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}" if abs(x) < 1e6 else f"{x:.6g}"
    return str(x)


def render_report(R, args, wall, cmd, date, sha, selftest=False):
    word, why = classify(R["variants"])
    V = R["variants"]
    L = []
    A = L.append
    A(f"# Q8 — a custom Mojo op inside ops.while_loop, {date}, commit {sha}" + ("  [SELFTEST: synthetic numbers, not a measurement]" if selftest else ""))
    A("")
    A("## 6 What ran")
    A(f"command:        {cmd}")
    A(f"commit:         {sha} (probe files are new and uncommitted: experiments/max_loop_custom_op/)")
    A(f"seed(s):        {args.seed} (numpy default_rng for obs, W, panel state)")
    A("inputs:         experiments/max_loop_custom_op/kernels/stub_panel_op.mojo (compiled by MAX at session.load via custom_extensions); no stored data")
    A(f"wall:           {fmt(wall)}")
    A(f"effective config: WB={args.wb}, panels={PANELS}, state row={FEAT} floats, {args.iters} iterations, float32, 1 warm-up execute then {args.reps} timed executes per variant; "
      "a panel row holds vector(3)+quaternion(4)+out(1), so the 3x3 rotation is formed from the stored quaternion (8 floats cannot hold a vector and a 3x3); "
      "the verdict variant is the functional custom op with loop-carried state, the in-place op is reported beside it; atan2f copied from mojo/src/mathx.mojo")
    A("")
    A("## 7 Data quality")
    ok_variants = [k for k, v in V.items() if v and v.get("ok")]
    A(f"n:              {args.reps} timed executes x {args.iters} iterations per variant; variants that ran: {', '.join(ok_variants) or 'none'}")
    bad = {k: (v or {}).get("error") for k, v in V.items() if not (v or {}).get("ok")}
    A("output devices: " + "; ".join(f"{k}: {v.get('output_devices')}" for k, v in V.items() if v and v.get('ok')))
    A("dropped:        " + ("none" if not bad else "; ".join(f"{k}: stage {V[k].get('stage') if V.get(k) else 'not run'}: {e}" for k, e in bad.items())))
    ck = R.get("checks", {})
    A("artifacts:      host wall time around execute() + device synchronize (includes launch of the whole while_loop, not a GPU-event time); the loop counter is a CPU int32 as in the 50.9 us probe; "
      "first execute (warm-up) excluded; compile time excluded (reported below). Correctness graph (" + str(args.check_iters) + " iterations, same WB): "
      + ("; ".join(f"{k}: max|state - numpy| {fmt(c.get('max_abs_state'), 6)}, force {fmt(c.get('max_abs_force'), 6)} (scale {fmt(c.get('force_scale'), 3)}), obs {fmt(c.get('max_abs_obs'), 6)}, finite {fmt(c.get('finite'))}, state changed {fmt(c.get('state_changed'))}, ok {fmt(c.get('ok'))}" + (f", error {c.get('error')}" if c.get('error') else '') for k, c in ck.items()) or "not run")
      + f". atan2f polynomial vs true atan2 on the panel angles: max {fmt(R.get('atan2_err'), 7)} rad.")
    A("")
    A("## 8 Numbers (the table the ROADMAP item asked for, nothing else)")
    A("| variant | us / iteration (median) | min - max | total ms (median) | compile s |")
    A("|---|---|---|---|---|")
    for k in ("base", "custom", "inplace"):
        v = V.get(k)
        if v and v.get("ok"):
            A(f"| {k} | {fmt(v['us_median'])} | {fmt(v['us_min'])} - {fmt(v['us_max'])} | {fmt(1e3 * statistics.median(v['total_s']))} | {fmt(v.get('compile_s'))} |")
        else:
            A(f"| {k} | n/a | n/a | n/a | n/a |")
    A("")
    b, c, ip = V.get("base"), V.get("custom"), V.get("inplace")
    delta = (c["us_median"] - b["us_median"]) if (b and b.get("ok") and c and c.get("ok")) else None
    delta_ip = (ip["us_median"] - b["us_median"]) if (b and b.get("ok") and ip and ip.get("ok")) else None
    A(f"delta (custom op, functional) vs base: {fmt(delta)} us/iteration; in-place vs base: {fmt(delta_ip)} us/iteration; G's matmul-only 51 us shape was 50.9 us/iteration (4096x6x6 bmm).")
    A("")
    A("spread:         per-variant min - max and the per-execute list: " + "; ".join(
        f"{k}: [{', '.join(fmt(x) for x in v['us_per_iter'])}]" for k, v in V.items() if v and v.get("ok")) + f" us/iteration over {args.reps} timed executes; one process, one session.")
    A("")
    A("## 9 Against the frozen prediction (quote it verbatim, then one word)")
    A(f'prediction:     "{frozen_prediction()}"')
    A(f"outcome:        {word}")
    A(f"by:             {why}; thresholds as frozen (<= {CONFIRM_US:.0f} us CONFIRMED, > {REFUTE_US:.0f} us or not placeable REFUTED, otherwise NEITHER).")
    A("")
    A("## 12 Reproducer")
    A("```bash")
    A("cd /home/hundo/Projects/Dytiscidae/dytiscidae/mojo")
    A("pixi run python ../experiments/max_loop_custom_op/run.py > ../experiments/max_loop_custom_op/run_q8.log 2>&1")
    A("```")
    return "\n".join(L) + "\n", word


# ------------------------------------------------------------------------ main
def get_device_and_session(selftest):
    from max.driver import CPU, Accelerator, accelerator_count
    from max.engine import InferenceSession

    if selftest or accelerator_count() == 0:
        if not selftest:
            raise SystemExit("no accelerator found; use --selftest for the CPU check")
        dev = CPU()
    else:
        dev = Accelerator()
    return dev, InferenceSession(devices=[dev])


def gpu_main(a):
    t0 = time.time()
    dev, sess = get_device_and_session(False)
    R = {"variants": {}, "checks": {}}
    for k in a.variants.split(","):
        k = k.strip()
        print(f"[{k}] build + compile + time ...", flush=True)
        R["variants"][k] = run_variant(k, dev, sess, a.wb, a.iters, a.reps, a.seed)
        print(f"[{k}]", json.dumps({x: y for x, y in R["variants"][k].items() if x != "us_per_iter" and x != "total_s"}, default=str)[:700], flush=True)
    for k in a.variants.split(","):
        k = k.strip()
        if k == "base":
            continue
        print(f"[{k}] correctness graph ({a.check_iters} iterations) ...", flush=True)
        R["checks"][k] = check_variant(k, dev, sess, a.wb, a.check_iters, a.seed)
        print(f"[{k}] check", R["checks"][k], flush=True)
    R["atan2_err"] = atan2_error_vs_true(a.wb, a.seed)
    wall = time.time() - t0
    cmd = "cd mojo && pixi run python ../experiments/max_loop_custom_op/run.py"
    text, word = render_report(R, a, wall, cmd, time.strftime("%Y-%m-%d"), git_sha())
    Path(a.results).write_text(json.dumps(R, indent=1, default=str))
    Path(a.report).write_text(text)
    print(f"report -> {a.report}\nOUTCOME: {word}")


def selftest(a):
    """CPU device (MAX CPU target), WB=4, 3 iterations; numpy reference; report rendering with synthetic numbers."""
    f = np.float32
    # 1. numpy reference sanity: atan2f port vs true atan2, rotation preserves |v|
    rng = np.random.default_rng(1)
    y, x = rng.standard_normal(10000).astype(f), rng.standard_normal(10000).astype(f)
    e = float(np.abs(atan2f_np(y, x) - np.arctan2(y, x)).max())
    assert e < 2e-5, e
    obs, W, state = make_inputs(4, a.seed)
    s1, _ = panel_step_np(state, np.zeros((4, 1), f))
    assert np.allclose(np.linalg.norm(s1[..., 0:3], axis=-1), np.linalg.norm(state[..., 0:3], axis=-1), atol=1e-5)
    print(f"[ok] numpy reference: atan2f port max err {e:.2e} rad; rotation preserves |v|")

    # 2. the graph on the CPU target
    dev, sess = get_device_and_session(True)
    wb, iters = 4, 3
    outcome = {}
    for k in ("custom", "inplace"):
        c = check_variant(k, dev, sess, wb, iters, a.seed)
        outcome[k] = c
        print(f"[{'ok' if c.get('ok') else 'FAIL'}] CPU graph '{k}' WB={wb} iters={iters}: " + (
            f"max|state-numpy| {c['max_abs_state']:.2e}, force {c['max_abs_force']:.2e}, obs {c['max_abs_obs']:.2e}, finite {c['finite']}, changed {c['state_changed']}" if "max_abs_state" in c else c.get("error", "?")), flush=True)
    base = run_variant("base", dev, sess, wb, iters, 2, a.seed)
    print(f"[{'ok' if base['ok'] else 'FAIL'}] CPU graph 'base' runs: {base.get('error') or ('%.1f us/iter (CPU, meaningless)' % base['us_median'])}")

    # 3. report writer with synthetic numbers
    def synth(us):
        return {"variant": "x", "ok": True, "stage": "done", "us_per_iter": [us] * 3, "total_s": [us * 2e-3] * 3, "us_median": us, "us_min": us, "us_max": us, "compile_s": 1.0}
    for us, want in ((60.0, "CONFIRMED"), (300.0, "NEITHER"), (900.0, "REFUTED")):
        w, _ = classify({"custom": synth(us)})
        assert w == want, (us, w, want)
    w, _ = classify({"custom": {"ok": False, "stage": "graph-build", "error": "ValueError: custom op in while loop body rejected"}})
    assert w == "REFUTED", w
    w, _ = classify({"custom": {"ok": False, "stage": "load/compile", "error": "mojo: error: expected ')'"}})
    assert w == "NOT COMPARABLE", w
    R = {"variants": {"base": synth(51.0), "custom": synth(80.0), "inplace": synth(85.0)}, "checks": {k: v for k, v in outcome.items()}, "atan2_err": 1.2e-5}
    a2 = argparse.Namespace(**vars(a))
    text, word = render_report(R, a2, 1.0, "selftest", "2026-10-10", git_sha(), selftest=True)
    for hdr in ("## 6 What ran", "## 7 Data quality", "## 8 Numbers", "## 9 Against the frozen prediction", "## 12 Reproducer"):
        assert hdr in text, hdr
    pred = frozen_prediction()
    assert pred.startswith("<= 150 us per iteration") and "*Falsified* above 500 us" in pred, pred
    assert pred in text and word == "CONFIRMED"
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "r.md").write_text(text)
    print("[ok] report writer: sections present, prediction quoted verbatim, classify() thresholds")
    allok = outcome["custom"].get("ok") and base["ok"]
    print("SELFTEST " + ("PASS" if allok else "FAIL") + f" (custom ok={outcome['custom'].get('ok')}, inplace ok={outcome['inplace'].get('ok')})")
    return 0 if allok else 1


if __name__ == "__main__":
    args = parse_args()
    sys.exit(selftest(args) if args.selftest else (gpu_main(args) or 0))
