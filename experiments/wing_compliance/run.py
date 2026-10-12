"""What does a passive pitch hinge (``Part.pitch_compliance``) do to a wing?

Two measurements, numpy path, open loop, no learner:

1. **Fixed rig** (``experiments/flight_audit/fixed_rig.py``): the airframe held
   at its trim speed and attitude, the joints running their real dynamics.
   Per body plan, per flap frequency, per compliance: mean fluid force along
   (Fx) and across (Fz) the path, flapping and with the joints held at the
   gait's mean, and the hinges' deflection.  ``dyn_tm`` is fixed_rig's dynamic
   thrust margin, (Fx_flap - Fx_held) / drag_held.
2. **Air segment** (``experiments/flight_audit/feather_fly.fly``): the real
   rollout and ``_score_segment``, the plan's own gait and the still machine
   (``held_still_params``, CLAUDE.md rules 1, 6, 7), on three seeds.  A hinge
   that pays a falling or a still machine is a defect, not flexibility.

Compliance is set on every WING and MEMBRANE surface whose joint is not
universal.  Read the JSON, not the stdout, for the numbers:

    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/wing_compliance/run.py \\
        --out experiments/wing_compliance/results.json
    # the still-machine read on more seeds:
    MUJOCO_GL=disable PYTHONPATH=. .venv/bin/python experiments/wing_compliance/run.py \\
        --plans gannet --skip-rig --seeds 10 --out experiments/wing_compliance/still_gannet.json
"""
import argparse
import json
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, "experiments/flight_audit")
from feather_fly import SEEDS, fly  # noqa: E402
from fixed_rig import rig  # noqa: E402

from dytiscidae.core.bodyplans import BODY_PLANS  # noqa: E402
from dytiscidae.core.genome import MEMBRANE, WING  # noqa: E402
from dytiscidae.core.phenotype import build  # noqa: E402
from dytiscidae.envs.triphibian import TriphibianEnv  # noqa: E402

PLANS = ("gannet", "teal", "bat", "beetle")
COMPLIANCE = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
#: None is the plan's own gait frequency; 7 Hz is inside the band where
#: 2026-09-26 lost level flight to actuator torque.
FREQS = (None, 7.0)
STOP = 0.95  # rad; FEATHER_RANGE is 1.0


def compliant(name, c):
    g = BODY_PLANS[name]()
    n = 0
    for part in g.parts:
        if part.kind in (WING, MEMBRANE) and part.joint != "universal":
            part.pitch_compliance = float(c)
            n += 1
    return g, n


def hinge_qadr(env):
    import mujoco
    out = []
    for j in range(env.model.njnt):
        nm = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.endswith("_jp"):
            out.append(int(env.model.jnt_qposadr[j]))
    return np.array(out, dtype=int)


def rig_row(name, c, f):
    g, n_surf = compliant(name, c)
    env = TriphibianEnv(build(g), seed=0)
    base = env.cpg.base if f is None else replace(env.cpg.base, frequency=f)
    defl = []

    def sample(e):
        qa = hinge_qadr(e)
        if len(qa):
            defl.append(e.data.qpos[qa].copy())

    fx_h, fz_h, W, _ = rig(g, base, hold=True)
    fx_f, fz_f, W, stroke = rig(g, base, on_sample=sample)
    drag = -fx_h
    row = {"plan": name, "compliance": c, "freq": round(float(base.frequency), 3),
           "surfaces": n_surf, "weight_N": round(W, 3),
           "Fx_held": round(fx_h, 4), "Fz_held": round(fz_h, 4),
           "Fx_flap": round(fx_f, 4), "Fz_flap": round(fz_f, 4),
           "dyn_tm": round((fx_f - fx_h) / max(drag, 1e-9), 4),
           "lift_over_W": round(fz_f / W, 4), "stroke_reached": round(stroke, 3)}
    if defl:
        a = np.abs(np.concatenate(defl))
        row.update(defl_mean=round(float(a.mean()), 4), defl_p95=round(float(np.percentile(a, 95)), 4),
                   at_stop=round(float((a > STOP).mean()), 4))
    return row


def air_rows(name, c, seeds=SEEDS):
    g, _ = compliant(name, c)
    still = TriphibianEnv(build(g), seed=0).held_still_params()
    out = []
    for seed in seeds:
        out.append({"plan": name, "compliance": c, "seed": seed, "arm": "own_gait",
                    **fly(g, None, seed)})
        out.append({"plan": name, "compliance": c, "seed": seed, "arm": "still",
                    **fly(g, still, seed)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default=",".join(PLANS))
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-air", action="store_true")
    ap.add_argument("--skip-rig", action="store_true")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="air-segment seeds 0..N-1 per arm (default: feather_fly's three)")
    args = ap.parse_args()
    plans = args.plans.split(",")
    res = {"rig": [], "air": []}
    t0 = time.time()
    for name in plans:
        for f in () if args.skip_rig else FREQS:
            for c in COMPLIANCE:
                r = rig_row(name, c, f)
                res["rig"].append(r)
                print(json.dumps(r), flush=True)
        if not args.skip_air:
            for c in (0.0, 1.0):
                for r in air_rows(name, c, tuple(range(args.seeds))):
                    res["air"].append(r)
                    print(json.dumps(r), flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
    print(f"wall {res['wall_s']} s")


if __name__ == "__main__":
    main()
