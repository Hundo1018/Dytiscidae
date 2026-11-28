"""D1 (S0b_probes.md, D1-D2 and B14-B17): the kicked hover on T5, from the float and at z = 10 m in air, both
entrainment modes.  Uses s0b_t5_probe.mission unchanged (the kick is its `fly` phase's `kick`); ``z_float``
moves the start (hull origin height) so that nothing touches the water.  Rows -> results/S0b_probes4.jsonl.

    MUJOCO_GL=disable PYTHONPATH=. python experiments/flapper/s0b_t5b_d1.py [procs]
"""
from __future__ import annotations
import json, multiprocessing as mp, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "S0b_probes4.jsonl"


def _run(job):
    import s0b_t5_probe as P
    P.Z_FLOAT = job.get("z_float", P.Z_FLOAT)
    row = P._dispatch(job)
    return row


def jobs():
    import s0b_t5_probe as P
    law = P.law_cfg(gains=dict(wn=(8.0, 8.0, 4.0), zeta=0.8, k_v=0.5, k_p=0.1))
    J = []
    for where, zf in (("float", None), ("air10", 10.165)):
        for kick in (True, False):
            if where == "float" and not kick:
                continue
            for mode, ent in P._modes():
                fly = dict(kind="fly", H=2.0, hold_s=10.0, land=True, after=5.0)
                if kick:
                    fly["kick"] = {"roll_rate": 1.0, "vy": 0.3}
                j = dict(fn="mission", probe="D1", id=f"d1_{where}_{'kick' if kick else 'nokick'}_{mode}", mode=mode,
                         entrainment=ent, start="float", law=law, motor="x4",
                         phases=[dict(kind="settle", T=3.0 if where == "float" else 0.0), fly])
                if zf:
                    j["z_float"] = zf
                J.append(j)
    return J


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with mp.get_context("spawn").Pool(n) as pool:
        for r in pool.imap_unordered(_run, jobs(), chunksize=1):
            with open(OUT, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
            print(r["id"], r.get("abort"), [s for s in r["summary"]][-1], r["wall_s"], flush=True)
