"""Pass S0: rows (PREREG_S0_continuation.md, results/S0_plan.md).

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/flapper/s0_run.py SPECS.json OUT.jsonl [WORKERS]

Each spec is one row: ``run_row(spec)`` re-runs it bit-exactly.  spec keys:
id, scenario, leg, rung, cfg (s0_machine.configure), kind (cruise | entry |
takeoff | land | water) and the kind's own arguments, all passed to the
frozen M-series runners unchanged:

- cruise, entry: ``half`` = an `m4_lib.run_half` spec (law "A4" swaps
  `m4_lib.EntryFlight` for `s0_laws.HoverEntry`); entry ``again``: "takeoff"
  (V+ from the landed state, `m1_takeoff.run` on the same machine) or "exit"
  (M2b's XS from the floating state, 3 s held first, as M2b's float start);
- takeoff: ``arm``, ``entry`` for `m1_takeoff.run`;
- land: ``m3b`` = an `m3b_land.run` spec;
- water: ``water`` = an `s0_laws.water_row` spec.

Rows go to OUT with the sha256 of every file used (first line of each
invocation); traces to results/S0_traces/.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TR = HERE / "results" / "S0_traces"
FILES = ["mission.py", "m1_takeoff.py", "m1_lib.py", "m4_common.py", "m4_lib.py", "m4_read.py", "m3_land.py",
         "m3b_land.py", "m3b_sweep.py", "m3_read.py", "m3b_read.py", "m2_common.py", "m2_laws.py", "m2b_laws.py",
         "m2b_fly.py", "m2b_run.py", "m2_read.py", "m2b_read.py", "c1_pilot.py", "body.py", "rig.py",
         "s0_machine.py", "s0_laws.py", "s0_run.py", "s0_read.py"]
FREQ = dict(cruise=None, takeoff=4.5, exit=4.5)


def shas():
    return {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:16] for f in FILES if (HERE / f).exists()}


def _peaks(mc):
    return mc.inst.peaks()


def _save_trace(rid, T):
    TR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TR / f"{rid}.npz", **{k: (np.array(v) if k != "phase" else np.array(v, dtype="U10"))
                                             for k, v in T.items()})


def _xs_again(mc, mode, T=40.0):
    """M2b's XS from where the machine is (3 s held at 0 first, as M2b's float start)."""
    from m2_common import run_law, settle
    from m2b_laws import make2
    import m2b_run
    mc.env.solver.entrainment = mode
    settle(mc, 3.0)
    law = make2(dict(m2b_run.FROZEN["XS"]))
    t_start = mc.t
    rec = run_law(mc, law, T, {})
    return dict(t_start=t_start, summary=rec.summary(), trace=rec.trace, log=getattr(law, "log", None))


def _takeoff_again(mc, arm="V+"):
    import m1_takeoff as T
    orig = T.make
    T.make = lambda *a, **k: mc
    t_start = mc.t
    try:
        r = T.run(arm, (float(mc.com()[0]), 0.0), trace=True)
    finally:
        T.make = orig
    r["t_start"] = t_start
    r["passed"] = T.passed(r)
    return r


def run_row(spec):
    import s0_machine as S
    S.configure(spec.get("cfg", {}))
    t0 = time.time()
    kind = spec["kind"]
    out = dict(id=spec["id"], spec=spec)
    if kind in ("cruise", "entry"):
        import m4_lib
        import s0_laws
        half = dict(spec["half"])
        orig = m4_lib.EntryFlight
        if half.get("law") == "A4":
            m4_lib.EntryFlight = s0_laws.HoverEntry
        try:
            r, T = m4_lib.run_half(half)
        finally:
            m4_lib.EntryFlight = orig
        r.pop("snaps", None)
        _save_trace(spec["id"], T)
        mc = S.LAST[-1]
        fl_log = None
        out.update(result=r, peaks_leg1=_peaks(mc))
        if spec.get("again") == "takeoff":
            out["again"] = _takeoff_again(mc)
        elif spec.get("again") == "exit":
            out["again"] = _xs_again(mc, half.get("ent", "wetting"), spec.get("again_T", 40.0))
        out["fl_log"] = fl_log
    elif kind == "takeoff":
        import m1_takeoff as T
        r = T.run(spec["arm"], tuple(spec["entry"]), trace=True)
        r["passed"] = T.passed(r)
        mc = S.LAST[-1]
        out.update(result=r, x2=mc.inst.x2(T.T_C, T.T_C + T.T_RUN))
    elif kind == "land":
        import m3b_land
        m3b_land.TRACE_DIR = TR / "land"
        r = m3b_land.run(dict(spec["m3b"]))
        mc = S.LAST[-1]
        out.update(result=r)
    elif kind == "water":
        import s0_laws
        mc, r = s0_laws.water_row(spec["water"])
        out.update(result=r)
        if spec["water"]["law"].get("law") == "stay":
            from m2b_read import gate_gx30
            g = gate_gx30(dict(trace=r["trace"]))
            if g.get("t_star") is not None:
                ts = r["t_start"] + g["t_star"]
                out["x2"] = mc.inst.x2(ts, ts + 30.0)
    else:
        raise ValueError(kind)
    if "again" in out and spec.get("again") == "exit":
        from m2b_read import gate_gx30
        a = out["again"]
        g = gate_gx30(dict(trace=a["trace"]))
        if g.get("t_star") is not None:
            ts = a["t_start"] + g["t_star"]
            a["x2"] = mc.inst.x2(ts, ts + 30.0)
    if "again" in out and spec.get("again") == "takeoff":
        import m1_takeoff as T
        a = out["again"]
        a["x2"] = mc.inst.x2(a["t_start"] + T.T_C, a["t_start"] + T.T_C + T.T_RUN)
    out.update(peaks=_peaks(mc), mass=mc.mass, s0=mc.s0, bad_qacc=int(mc.bad_qacc), wall_s=round(time.time() - t0, 1))
    return out


def job(spec):
    try:
        return run_row(spec)
    except Exception:
        return dict(id=spec["id"], error=traceback.format_exc()[-2500:], spec=spec)


def main(argv):
    specs = json.load(open(argv[0]))
    outp = Path(argv[1]).resolve()
    nw = int(argv[2]) if len(argv) > 2 else 4
    done = set()
    if outp.exists():
        for x in open(outp):
            if x.strip():
                r = json.loads(x)
                if "error" not in r and "id" in r and r.get("kind") != "hash":
                    done.add(r["id"])
    specs = [s for s in specs if s["id"] not in done]
    print(f"{len(specs)} rows to run", flush=True)
    with open(outp, "a") as fh:
        fh.write(json.dumps(dict(kind="hash", sha=shas(), n=len(specs), t=time.time(), specs=argv[0])) + "\n")
    with mp.get_context("spawn").Pool(nw, maxtasksperchild=1) as pool, open(outp, "a") as fh:
        for r in pool.imap_unordered(job, specs):
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            if "error" in r:
                print(r["id"], "ERROR", r["error"][-600:], flush=True)
            else:
                print(r["id"], "done wall %.0f s qacc %d mass %.3f" % (r["wall_s"], r["bad_qacc"], r["mass"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
