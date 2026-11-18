"""Pass M2: checks ADDED after the scored rows (RESULT_M2_water.md, disclosed).

1. XV on seeds 1-5 under the solver's default entrainment (C9 read only seed 0,
   where XV did not leave the water): rows to results/M2_checks.jsonl with
   extra {"entrainment": true}, by m2_run.run_row, unchanged.
2. When the solver's 60-weight force limiter engages in XV (seeds 0 and 3, the
   primary mode): the same row re-run, the limiter's step count read at every
   10 Hz sample; rows of kind "clamp_times" to results/M2_checks.jsonl.

    m2_extra.py
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "M2_checks.jsonl"


def clamp_times(seed):
    import m2_run as R
    from m2_common import build, draw, run_law, settle
    from m2_laws import make
    law_spec, start, gains, vb, T = R.spec_for("XV")
    dr = draw(seed)
    mc = build(start, dr, gains=gains, vb=vb, entrainment="wetting")
    settle(mc, 3.0)
    law = make(law_spec)
    marks = []

    def on(rec):
        x = rec.trace[-1]
        marks.append((x["t"], rec.clamped_steps, x["wet"], x["com"][2]))
        return False
    rec = run_law(mc, law, T, {}, on_sample=on)
    return dict(kind="clamp_times", arm="XV", seed=seed, marks=marks, exit_log=law.log,
                summary=rec.summary())


def job(j):
    if j[0] == "clamp":
        return clamp_times(j[1])
    import m2_run as R
    return R.run_row(("XV", j[1], "X", dict(entrainment=True)))


if __name__ == "__main__":
    import m2_run as R
    J = [("ent", s) for s in range(1, 6)] + [("clamp", 0), ("clamp", 3)]
    with Pool(4) as p:
        for row in p.imap_unordered(job, J):
            row["sha"] = R.shas()
            with open(OUT, "a") as fh:
                fh.write(json.dumps(row, default=float) + "\n")
            print(row["kind"], row.get("seed"), row["summary"]["clamped_steps"], flush=True)
