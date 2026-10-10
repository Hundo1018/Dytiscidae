#!/usr/bin/env python3
"""Q1 -- the Mojo fluid kernel's marginal cost (ROADMAP 2026-10-10, "an
evaluation that never leaves the device", section 4, probe Q1).

Frozen text (verbatim, line breaks as in docs/ROADMAP.md):

    **Q1 (GPU, ~10 min): the fluid kernel's marginal cost.** Time
    `FullPipeline.step` (the Mojo kernel, as the batched path calls it) at 190,
    5,000 and 50,000 panels built from copies of one arch49 body, back to back,
    graph-free as today. *Prediction (T7):* <= 0.5 us per panel marginal above
    the ~90 us launch, so 50,000 panels (1,000 worlds of a 50-panel body) cost
    <= 25 ms per step. *Falsified* above 2 us per panel. Decides whether the fluid
    side can carry hundreds of draws at all; if it cannot, none of A, B, C pays.

Outcome rule applied (the request's reading of the text above, not a change to
it): fit t = a + b * panels by least squares on the per-size median step time;
b <= 0.5 us per panel -> CONFIRMED, b > 2 us per panel -> REFUTED, otherwise
NEITHER.  The 25 ms at 50,000 panels is reported beside it, not used to decide.

What is timed: ``FullPipeline.step(desc)`` of mojo/src/full_pipeline.mojo (its
``step`` is ``launch`` then ``wait``: input upload, the panel kernel, the body
gather kernel, the output download, one synchronize), called with the very
descriptor ``BatchedFluid`` builds (dytiscidae/envs/batchroll.py:446 is the
``self.pipe.launch(self._desc)`` of the batched path, :456 its ``self.pipe.wait()``
in ``finish``; experiments/perf/gpu_contention.py:29-31 times the same ``step`` by
wrapping ``BatchedFluid.pipe``).  No MuJoCo stepping: the host input block is
filled once by ``BatchedFluid.launch`` and the same bytes are sent every step,
so the number is the kernel's launch + per-panel cost plus the PCIe copy of the
fixed-size blocks.  The panel set is N copies of ONE arch49 elite (the median-dof
body experiments/mjwarp_probe/elite_median_dof.json names), the same env object
repeated N times in the ``BatchedFluid`` envs list (rebasing of body ids and
machine ids is what BatchedFluid does for N distinct machines).

Capacity: the pipeline's capacity is fixed at the first allocation
(batchroll.py:192-217: MIN_CAP_PANELS 16384, MIN_CAP_BODIES 8192, MIN_CAP_MACHINES
256; a second FullPipeline in one process hangs, :172-181; an oversize batch raises
RuntimeError at :212).  The two largest sizes exceed 16384 panels and 256 machines,
so this script makes the first ``_get_pipeline`` call itself with the largest
size, which is the documented way to take a bigger capacity (``cap = max(n,
MIN_CAP_*)``).  The kernel launch grid is ``ceildiv(n, 128)`` and does not read
the capacity, so a larger allocation does not change the timed work.

    # GPU free:
    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/fluid_kernel_scaling/run.py
    # no GPU:
    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/fluid_kernel_scaling/run.py --selftest
    # rewrite the report from a results file:
    PYTHONPATH=. .venv/bin/python experiments/fluid_kernel_scaling/run.py --report-from experiments/fluid_kernel_scaling/results_arch49.json
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ELITE_JSON = ROOT / "experiments" / "mjwarp_probe" / "elite_median_dof.json"
RUN = "runs/arch49"
DEFAULT_PANELS = (190, 1000, 5000, 20000, 50000)
DEFAULT_OUT = HERE / "results_arch49.json"
DEFAULT_REPORT = ROOT / "runs" / "analysis_1010_failure_theory" / "Q1_result.md"
WARMUP, STEPS = 20, 200
SEED = 20261010

PREDICTION = (
    "<= 0.5 us per panel marginal above the ~90 us launch, so 50,000 panels "
    "(1,000 worlds of a 50-panel body) cost <= 25 ms per step. Falsified above 2 us "
    "per panel.")
CONFIRM_US, REFUTE_US, FIFTY_K_MS = 0.5, 2.0, 25.0


# --------------------------------------------------------------------------
# Pure arithmetic (no simulator): the plan, the fit, the outcome
# --------------------------------------------------------------------------


def plan_copies(targets, per_body):
    """Whole-body copies for each requested panel count (nearest, at least 1).

    ``round`` is half-up so 190 of a 126-panel body is 2 copies (252 panels), not
    1 (126), whose distance to 190 is 64 vs 62.  Returns [(target, copies,
    panels)], duplicates in copies collapsed."""
    out, seen = [], set()
    for t in targets:
        c = max(1, int(math.floor(t / per_body + 0.5)))
        if c in seen:
            continue
        seen.add(c)
        out.append((int(t), c, c * per_body))
    return out


def fit_line(panels, t_us):
    """Least squares t = a + b * panels.  Returns (a, b, r2)."""
    x, y = np.asarray(panels, float), np.asarray(t_us, float)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), r2


def outcome_of(b_us):
    if not np.isfinite(b_us):
        return "NOT COMPARABLE"
    if b_us <= CONFIRM_US:
        return "CONFIRMED"
    if b_us > REFUTE_US:
        return "REFUTED"
    return "NEITHER"


def summarise(samples_us):
    s = np.sort(np.asarray(samples_us, float))
    return {"median": float(np.median(s)), "p90": float(np.percentile(s, 90)),
            "mean": float(s.mean()), "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "q25": float(np.percentile(s, 25)), "q75": float(np.percentile(s, 75)),
            "min": float(s[0]), "max": float(s[-1]), "n": int(len(s))}


def bootstrap_b(sample_sets, panels, rng, reps=400):
    """Spread of b: resample each size's steps, refit the medians."""
    bs = []
    for _ in range(reps):
        meds = [np.median(rng.choice(s, size=len(s), replace=True)) for s in sample_sets]
        bs.append(fit_line(panels, meds)[1])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyse(sizes):
    """``sizes``: list of dicts with panels and step_us (list).  Adds the fit."""
    sizes = sorted(sizes, key=lambda r: r["panels"])
    panels = [r["panels"] for r in sizes]
    for r in sizes:
        r["stats"] = summarise(r["step_us"])
    med = [r["stats"]["median"] for r in sizes]
    p90 = [r["stats"]["p90"] for r in sizes]
    a, b, r2 = fit_line(panels, med)
    a9, b9, r29 = fit_line(panels, p90)
    lo, hi = bootstrap_b([np.asarray(r["step_us"]) for r in sizes], panels,
                         np.random.default_rng(SEED))
    pair = [{"from": sizes[i]["panels"], "to": sizes[i + 1]["panels"],
             "us_per_panel": (med[i + 1] - med[i]) / (sizes[i + 1]["panels"] - sizes[i]["panels"])}
            for i in range(len(sizes) - 1)]
    big = sizes[-1]
    return {"fit_median": {"a_us": a, "b_us_per_panel": b, "r2": r2,
                           "b_ci95_bootstrap": [lo, hi]},
            "fit_p90": {"a_us": a9, "b_us_per_panel": b9, "r2": r29},
            "pairwise_marginal": pair,
            "largest": {"panels": big["panels"], "median_ms": big["stats"]["median"] / 1e3,
                        "p90_ms": big["stats"]["p90"] / 1e3},
            "outcome": outcome_of(b)}


# --------------------------------------------------------------------------
# Building the shard
# --------------------------------------------------------------------------


def commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_elite_env():
    """The named arch49 elite, as a TriphibianEnv ready to stand for N machines.

    Loaded as experiments/mjwarp_probe/export_mjcf.py loads it: the island's
    archive from ``load_run_archive``, matched on genome_id."""
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs.triphibian import TriphibianEnv
    from dytiscidae.ops.run import load_run_archive
    meta = json.loads(ELITE_JSON.read_text())
    arc, _ = load_run_archive(str(ROOT / RUN), meta["island"])
    hit = [e for e in arc.cells.values()
           if str(getattr(e.genome, "genome_id", "")) == meta["genome_id"]]
    if not hit:
        raise SystemExit(f"{meta['genome_id']} not in {RUN} island {meta['island']}")
    env = TriphibianEnv(build(hit[0].genome), seed=1)
    return env, meta


def moving_state(env):
    """A flying, non-degenerate state: launch speed from ``reset(AIR)`` plus
    random joint velocities, forward-kinematics'd so xpos, xmat, xipos and cvel
    are what ``BatchedFluid.launch`` reads.  Zero relative velocity would let a
    kernel branch out early and understate the per-panel cost."""
    import mujoco
    from dytiscidae.envs.triphibian import Domain
    env.reset(Domain.AIR, randomise=False)
    rng = np.random.default_rng(SEED)
    nv = env.model.nv
    env.data.qvel[6:nv] += rng.normal(0.0, 1.0, nv - 6)
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_comVel(env.model, env.data)
    return {"qvel_norm": float(np.linalg.norm(env.data.qvel)),
            "cvel_finite": bool(np.isfinite(env.data.cvel).all()),
            "xpos_finite": bool(np.isfinite(env.data.xpos).all())}


def counts(env):
    return int(len(env.solver.panels.body_id)), int(env.model.nbody)


def gpu_state():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        u, m = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"util_pct": u, "mem_mib": m}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def measure(args):
    from dytiscidae.envs import batchroll
    pf = time.perf_counter
    t_start = pf()
    ok, why = batchroll.usable()
    if not ok:
        raise SystemExit(f"batched evaluator unusable: {why}")
    env, meta = load_elite_env()
    state = moving_state(env)
    per_p, per_b = counts(env)
    plan = plan_copies(args.panels, per_p)
    nmax = max(c for _, c, _ in plan)
    gpu0 = gpu_state()
    if (gpu0.get("util_pct", 0) or 0) > args.busy_pct and not args.allow_busy:
        raise SystemExit(f"GPU is busy ({gpu0}); a timing taken now includes contention. "
                         f"Rerun when free, or pass --allow-busy to record it anyway.")
    # Take the capacity for the largest size on the first (and only) allocation.
    batchroll._get_pipeline(nmax * per_p, nmax * per_b, nmax)
    cap = batchroll._POOL["cap"]
    order = [p for p in plan] + [plan[0]]          # last entry repeats the first: drift check
    rows = []
    for k, (target, copies, panels) in enumerate(order):
        bf = batchroll.BatchedFluid([env] * copies)
        assert bf.n == panels, (bf.n, panels)
        # Prime as the batched path does: the first launch resets the unsteady
        # history, the second is a steady step (sc[21] = 0), and both are waited.
        bf.reset_slam()
        for t in (0.0, env.model.opt.timestep):
            bf.launch(t)
            bf.pipe.wait()
            batchroll._POOL["outstanding"] = None
        out = bf.out
        finite = bool(np.isfinite(out["force"]).all() and np.isfinite(out["xfrc"]).all())
        mag = float(np.abs(out["force"]).mean())
        pipe, desc = bf.pipe, bf._desc
        for _ in range(WARMUP):
            pipe.step(desc)
        step = np.empty(STEPS)
        for i in range(STEPS):
            t0 = pf()
            pipe.step(desc)
            step[i] = (pf() - t0) * 1e6
        launch, wait = np.empty(STEPS), np.empty(STEPS)
        for i in range(STEPS):                      # same two calls `step` makes, timed apart
            t0 = pf()
            pipe.launch(desc)
            t1 = pf()
            pipe.wait()
            t2 = pf()
            launch[i], wait[i] = (t1 - t0) * 1e6, (t2 - t1) * 1e6
        row = {"target": target, "copies": copies, "panels": int(bf.n), "bodies": int(bf.nb),
               "machines": int(bf.nm), "step_us": step.tolist(),
               "launch_us_median": float(np.median(launch)),
               "wait_us_median": float(np.median(wait)),
               "output_finite": finite, "mean_abs_panel_force": mag,
               "drift_check": k == len(order) - 1}
        rows.append(row)
        s = summarise(step)
        print(f"{bf.n:>6} panels ({copies:>3} copies, {bf.nb} bodies): step median "
              f"{s['median']:.1f} us p90 {s['p90']:.1f} us | launch {row['launch_us_median']:.1f} "
              f"wait {row['wait_us_median']:.1f} | finite {finite}", flush=True)
        del bf
    main_rows = [r for r in rows if not r["drift_check"]]
    res = analyse(main_rows)
    drift = rows[-1]
    first = next(r for r in main_rows if r["panels"] == drift["panels"])
    res["drift_check"] = {"panels": drift["panels"],
                          "median_us_first": summarise(first["step_us"])["median"],
                          "median_us_last": summarise(drift["step_us"])["median"]}
    res.update({
        "item": "Q1", "commit": commit(), "date": time.strftime("%Y-%m-%d"),
        "command": "PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python experiments/fluid_kernel_scaling/run.py"
                   + ("" if list(args.panels) == list(DEFAULT_PANELS)
                      else " --panels " + " ".join(map(str, args.panels))),
        "elite": meta, "run": RUN, "panels_per_body": per_p, "bodies_per_machine": per_b,
        "plan": [{"target": t, "copies": c, "panels": p} for t, c, p in plan],
        "pipeline_capacity_panels_bodies_machines": list(cap),
        "state": state, "warmup": WARMUP, "steps": STEPS, "seed": SEED,
        "gpu_before": gpu0, "gpu_after": gpu_state(),
        "kernel_dir": str(batchroll._BUILD_DIR),
        "sizes": [{k: v for k, v in r.items() if k != "step_us"} | {"stats": summarise(r["step_us"])}
                  for r in main_rows],
        "wall_s": pf() - t_start,
        "samples_us": {str(r["panels"]): r["step_us"] for r in main_rows},
    })
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")
    return res


# --------------------------------------------------------------------------
# The report, in the shape of runs/analysis_1010_failure_theory/REPORT_FORMAT.md
# --------------------------------------------------------------------------


def write_report(res, path):
    fm, f9 = res["fit_median"], res["fit_p90"]
    lo, hi = fm["b_ci95_bootstrap"]
    big = res["largest"]
    out = res["outcome"]
    sizes = res["sizes"]
    lines = []
    A = lines.append
    A(f"# Q1 — the fluid kernel's marginal cost, {res['date']}, commit {res['commit']}")
    A("")
    A("## 6 What ran")
    A(f"command:        {res['command']}")
    A(f"commit:         {res['commit']} (script experiments/fluid_kernel_scaling/run.py; uncommitted if not in git log)")
    A(f"seed(s):        {res['seed']} (joint-velocity draw of the shared state; bootstrap); no search seed, no MuJoCo stepping")
    A(f"inputs:         {res['run']} island {res['elite']['island']} genome {res['elite']['genome_id']} "
      f"(dof {res['elite']['dof']}, n_rotors {res['elite']['n_rotors']}; named by experiments/mjwarp_probe/elite_median_dof.json); "
      f"kernel {res['kernel_dir']}")
    A(f"wall:           {res['wall_s']:.1f}")
    A("effective config: FullPipeline.step (launch + wait) on a BatchedFluid of N copies of the one body "
      f"({res['panels_per_body']} panels, {res['bodies_per_machine']} bodies each), same input block every step; "
      f"{res['warmup']} warm-up then {res['steps']} timed steps per size. Requested panel counts "
      f"{[p['target'] for p in res['plan']]} become whole-body copies: "
      f"{[(p['target'], p['copies'], p['panels']) for p in res['plan']]} (target, copies, panels). "
      f"Pipeline capacity (panels, bodies, machines) raised to {tuple(res['pipeline_capacity_panels_bodies_machines'])} "
      "by the first _get_pipeline call (batchroll default 16384/8192/256 cannot hold the two largest sizes: "
      "batchroll.py:204-217); nothing else differs from the batched path. GPU before/after: "
      f"{res['gpu_before']} / {res['gpu_after']}.")
    A("")
    A("## 7 Data quality")
    A(f"n:              {len(sizes)} sizes x {res['steps']} steps (+ one repeat of the smallest size after the largest)")
    A(f"dropped:        0; all {res['steps']} timed steps of every size are used (no trimming)")
    nonfin = [s['panels'] for s in sizes if not s.get('output_finite', True)]
    d = res["drift_check"]
    A(f"artifacts:      first launch with the unsteady-history reset and one steady launch are done before warm-up; "
      f"outputs finite at every size: {not nonfin}{'' if not nonfin else ' (NOT finite at ' + str(nonfin) + ')'}; "
      f"drift check, {d['panels']} panels re-timed after the largest: median {d['median_us_first']:.1f} us first, "
      f"{d['median_us_last']:.1f} us last.")
    A("")
    A("## 8 Numbers (the table the ROADMAP item asked for, nothing else)")
    A("| panels | copies | bodies | median us/step | p90 us/step | SD us | launch (median us) | wait (median us) |")
    A("|---|---|---|---|---|---|---|---|")
    for s in sizes:
        st = s["stats"]
        A(f"| {s['panels']} | {s['copies']} | {s['bodies']} | {st['median']:.1f} | {st['p90']:.1f} | "
          f"{st['sd']:.1f} | {s['launch_us_median']:.1f} | {s['wait_us_median']:.1f} |")
    A("")
    A(f"fit t = a + b * panels on the medians: a = {fm['a_us']:.1f} us, b = {fm['b_us_per_panel']:.4f} us per panel "
      f"(R^2 {fm['r2']:.4f}); on the p90s: a = {f9['a_us']:.1f} us, b = {f9['b_us_per_panel']:.4f} us per panel.")
    A("consecutive-size marginal (us per panel): "
      + "; ".join(f"{p['from']}->{p['to']}: {p['us_per_panel']:.4f}" for p in res["pairwise_marginal"]))
    A(f"largest size ({big['panels']} panels): {big['median_ms']:.2f} ms median, {big['p90_ms']:.2f} ms p90 "
      f"(prediction's 25 ms at 50,000 panels: {'within' if big['panels'] >= 50000 and big['median_ms'] <= FIFTY_K_MS else 'not within' if big['panels'] >= 50000 else 'n/a, largest size is below 50,000'}).")
    A("")
    A(f"spread:         b 95% bootstrap CI over the {res['steps']} steps per size [{lo:.4f}, {hi:.4f}] us per panel; "
      "per-size SD and p90 in the table.")
    A("")
    A("## 9 Against the frozen prediction (quote it verbatim, then one word)")
    A(f"prediction:     \"{PREDICTION}\"")
    A(f"outcome:        {out}" + ("" if out != "NEITHER" else
                                  " (not CONFIRMED, not REFUTED; the frozen text puts 0.5-2 us per panel in neither)"))
    A(f"by:             b = {fm['b_us_per_panel']:.4f} us per panel (thresholds: <= {CONFIRM_US} confirmed, > {REFUTE_US} refuted), "
      f"a = {fm['a_us']:.1f} us against the predicted ~90 us launch, "
      f"{big['panels']} panels at {big['median_ms']:.2f} ms median.")
    A("")
    A("## 12 Reproducer")
    A("```bash")
    A("cd /home/hundo/Projects/Dytiscidae/dytiscidae")
    A(res["command"])
    A("```")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# Selftest: CPU only, no pipeline is constructed
# --------------------------------------------------------------------------


def selftest(args):
    # arithmetic
    assert plan_copies([190, 1000, 5000, 20000, 50000], 126) == [
        (190, 2, 252), (1000, 8, 1008), (5000, 40, 5040), (20000, 159, 20034), (50000, 397, 50022)]
    assert plan_copies([100, 130], 126) == [(100, 1, 126)]                 # duplicates collapse
    a, b, r2 = fit_line([100, 1000, 10000], [90 + 0.3 * x for x in (100, 1000, 10000)])
    assert abs(a - 90) < 1e-6 and abs(b - 0.3) < 1e-9 and abs(r2 - 1) < 1e-9, (a, b, r2)
    assert [outcome_of(x) for x in (0.1, 0.5, 0.51, 2.0, 2.01, float("nan"))] == [
        "CONFIRMED", "CONFIRMED", "NEITHER", "NEITHER", "REFUTED", "NOT COMPARABLE"]
    # analyse + report on synthetic samples
    rng = np.random.default_rng(0)
    syn = [{"panels": p, "copies": p // 126, "bodies": p // 9, "machines": p // 126,
            "step_us": (90 + 0.4 * p + rng.normal(0, 3, STEPS)).tolist(),
            "launch_us_median": 60.0, "wait_us_median": 30.0, "output_finite": True}
           for p in (252, 1008, 5040, 20034, 50022)]
    an = analyse(syn)
    assert an["outcome"] == "CONFIRMED" and abs(an["fit_median"]["b_us_per_panel"] - 0.4) < 0.01, an
    tmp = Path(args.out).with_suffix(".selftest.md")
    res = dict(an, item="Q1", commit="selftest", date="1970-01-01", command="selftest", elite=json.loads(ELITE_JSON.read_text()),
               run=RUN, panels_per_body=126, bodies_per_machine=14,
               plan=[{"target": t, "copies": c, "panels": p} for t, c, p in plan_copies(DEFAULT_PANELS, 126)],
               pipeline_capacity_panels_bodies_machines=[50022, 8192, 397], warmup=WARMUP, steps=STEPS,
               seed=SEED, gpu_before={}, gpu_after={}, kernel_dir="-", wall_s=0.0,
               sizes=[{k: v for k, v in r.items() if k != "step_us"} | {"stats": summarise(r["step_us"])} for r in syn],
               drift_check={"panels": 252, "median_us_first": 1.0, "median_us_last": 1.0})
    write_report(res, tmp)
    txt = tmp.read_text()
    assert PREDICTION in txt and "## 12 Reproducer" in txt
    tmp.unlink()
    # arguments
    assert list(parse(["--panels", "10", "20"]).panels) == [10, 20]
    assert list(parse([]).panels) == list(DEFAULT_PANELS)
    # the real elite, the real env, the state the GPU run would send (CPU only)
    from dytiscidae.envs import batchroll
    env, meta = load_elite_env()
    state = moving_state(env)
    assert state["cvel_finite"] and state["xpos_finite"] and state["qvel_norm"] > 1.0, state
    per_p, per_b = counts(env)
    plan = plan_copies(args.panels, per_p)
    nmax = max(c for _, c, _ in plan)
    need = (nmax * per_p, nmax * per_b, nmax)
    cap = (max(need[0], batchroll.MIN_CAP_PANELS), max(need[1], batchroll.MIN_CAP_BODIES),
           max(need[2], batchroll.MIN_CAP_MACHINES))
    # the interface the timed call relies on
    iface = {"AVAILABLE": bool(batchroll.AVAILABLE),
             "BatchedFluid.launch": hasattr(batchroll.BatchedFluid, "launch"),
             "_get_pipeline": callable(batchroll._get_pipeline)}
    if batchroll.AVAILABLE:
        iface["FullPipeline.step"] = all(hasattr(batchroll._fp.FullPipeline, m)
                                         for m in ("step", "launch", "wait", "upload_static"))
    assert all(iface.values()) or not batchroll.AVAILABLE, iface
    print(f"selftest ok: elite {meta['genome_id']} {per_p} panels/{per_b} bodies per copy; "
          f"plan {[(c, p) for _, c, p in plan]}; needed capacity {need} vs default "
          f"({batchroll.MIN_CAP_PANELS},{batchroll.MIN_CAP_BODIES},{batchroll.MIN_CAP_MACHINES}) -> allocate {cap}; "
          f"interface {iface}; GPU not touched")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panels", type=int, nargs="+", default=list(DEFAULT_PANELS),
                    help="target panel counts (rounded to whole copies of the elite)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--report-from", default=None, help="rewrite the report from a results json; no GPU")
    ap.add_argument("--selftest", action="store_true", help="CPU-only import/argument/arithmetic checks")
    ap.add_argument("--allow-busy", action="store_true", help="time even if the GPU is in use")
    ap.add_argument("--busy-pct", type=float, default=20.0)
    return ap.parse_args(argv)


def main():
    args = parse()
    if args.selftest:
        selftest(args)
    elif args.report_from:
        write_report(json.loads(Path(args.report_from).read_text()), args.report)
    else:
        res = measure(args)
        write_report(res, args.report)
        print(f"OUTCOME {res['outcome']}  a={res['fit_median']['a_us']:.1f} us  "
              f"b={res['fit_median']['b_us_per_panel']:.4f} us/panel")


if __name__ == "__main__":
    main()
