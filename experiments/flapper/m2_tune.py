"""Pass M2: one tuning rollout per spec, appended to results/M2_trials.jsonl.

    m2_tune.py '<json spec>' ['<json spec>' ...]   (runs them on up to 4 processes)

spec: {"note", "law", "g", "seed", "start": "depth"|"float", "gains": "here"|"air",
       "T", "cmd": {"h": rad or "draw" or "draw+pi", "z": depth or "deeper"/"shallower"},
       "vb": bool, "settle": s}
Tuning seeds only (100-103), section 1.
"""
from __future__ import annotations

import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TRIALS = HERE / "results" / "M2_trials.jsonl"


def resolve_cmd(cmd, dr):
    out = {}
    for k, v in (cmd or {}).items():
        if k == "h":
            out["h"] = dr["h"] if v == "draw" else (dr["h"] + math.pi if v == "draw+pi" else float(v))
        elif k == "z":
            out["z"] = {"deeper": 3.0 + dr["delta"], "shallower": 3.0 - dr["delta"]}.get(v, v)
        else:
            out[k] = v
    return out


def one(spec):
    import numpy as np
    from m2_common import build, draw, run_law, settle
    from m2_laws import make
    assert int(spec["seed"]) >= 100, "tuning seeds only"
    t0 = time.time()
    dr = draw(spec["seed"])
    mc = build(spec.get("start", "depth"), dr, gains=spec.get("gains", "here"), vb=spec.get("vb", False),
               entrainment=spec.get("entrainment", "wetting"))
    if spec.get("start") == "float":
        settle(mc, spec.get("settle", 3.0))
    law = make(spec)
    law.phi = dr["phase"]
    cmd = resolve_cmd(spec.get("cmd"), dr)
    rec = run_law(mc, law, spec["T"], cmd)
    tr = rec.trace
    com = np.array([r["com"] for r in tr])
    out = dict(kind="trial", spec=spec, cmd=cmd, draw=dr, summary=rec.summary(), wall_s=time.time() - t0,
               start_com=com[0].tolist(), end_com=com[-1].tolist(),
               trace=tr[:: max(1, int(spec.get("keep_every", 5)))])
    with open(TRIALS, "a") as fh:
        fh.write(json.dumps(out, default=float) + "\n")
    return out


def brief(o):
    s = o["summary"]
    tr = o["trace"]
    import numpy as np
    com = np.array([r["com"] for r in tr])
    h = o["cmd"].get("h")
    along = ""
    if h is not None:
        dxy = com[-1, :2] - com[0, :2]
        along = " along %.2f m" % (dxy[0] * math.cos(h) + dxy[1] * math.sin(h))
    yaw = np.unwrap([math.atan2(r["nose"][1], r["nose"][0]) for r in tr])
    along += " dyaw %.0f deg roll_end %.0f" % (math.degrees(yaw[-1] - yaw[0]), math.degrees(math.atan2(-0.0, 1.0)))
    ts = [r["t"] for r in tr]
    best, run0 = 0.0, None
    for t_, r in zip(ts, tr):
        if r["wet"] == 0:
            run0 = t_ if run0 is None else run0
            best = max(best, t_ - run0)
        else:
            run0 = None
    along += " clear_max %.1f s" % best
    return ("%s seed %d: d=(%.2f, %.2f, %.2f) m%s  pitch_end %.0f  wet_end %d  maxz %.2f  P %.1f W  peak %.0f W  lim %s  qacc %d  wall %.0f s"
            % (o["spec"].get("note", ""), o["spec"]["seed"], *(com[-1] - com[0]), along,
               math.degrees(math.asin(max(-1, min(1, tr[-1]["nose"][2])))), tr[-1]["wet"], com[:, 2].max(),
               s["P_elec_mean_W"], s["P_peak_50ms_all_W"], {k: round(v, 2) for k, v in s["at_limit"].items()},
               s["bad_qacc"], o["wall_s"]))


if __name__ == "__main__":
    specs = [json.loads(a) for a in sys.argv[1:]]
    with Pool(min(4, len(specs))) as p:
        for o in p.imap(one, specs):
            print(brief(o), flush=True)
