"""N0: the anchor's neighbourhood under the unchanged framework.

No learning.  See `existence/PREREG_neighbourhood.md` (frozen first).

    PYTHONPATH=. MUJOCO_GL=disable .venv/bin/python \
        experiments/triphibian_diagnosis/scripts/neighbourhood.py OUT.jsonl --jobs 8
"""
import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

DEPTHS, PER_DEPTH, SEEDS, STARTS = (1, 2, 4, 8), 24, (200, 201), ("air", "water", "land")


def draw():
    from dytiscidae.core.bodyplans import ANCHOR_PLANS
    from dytiscidae.core.genome import mutate

    parent = ANCHOR_PLANS["legged_quad"]()
    out = []
    for d in DEPTHS:
        rng = np.random.default_rng([20261112, d])
        redrawn = 0
        while sum(1 for c in out if c[0] == d) < PER_DEPTH:
            child, ops = mutate(parent.copy(), rng, n_ops=d)
            # No operator applied is the parent again.  (A child whose operators
            # all drew a no-op is kept and shows as one that keeps the mission.)
            if not ops:
                redrawn += 1
                continue
            out.append((d, sum(1 for c in out if c[0] == d), child, list(ops), redrawn))
    return out


def judge(args):
    d, k, genome, ops, redrawn = args
    from dytiscidae.control.outer import teacher_controller
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import mission as M
    from dytiscidae.envs.triphibian import Domain, MissionSpec, TriphibianEnv, evaluate_tier0
    from dytiscidae.evolution.loop import SearchConfig

    t0 = time.time()
    row = {"depth": d, "k": k, "ops": ops, "redrawn_before": redrawn}
    try:
        p = build(genome)
        t = evaluate_tier0(p, MissionSpec())
        names = TriphibianEnv(p, seed=0).act_names
        row.update({"mass": float(p.mass), "n_rotors": sum(a.endswith("_r") for a in names),
                    "n_limb_act": sum(a.endswith("_a") for a in names), "n_act": len(names),
                    "density_ratio": float(p.mass / (1000.0 * max(p.buoyant_volume, 1e-9))),
                    "gate_margin": float(p.report.gate_margin),
                    "tier0": bool(p.report.gate_margin >= SearchConfig().tier0_gate
                                  and t.mission_fraction > 0.0)})
    except Exception as exc:
        row.update({"tier0": False, "build_error": repr(exc)[:200]})
        return row
    if not row["tier0"]:
        return row
    missions = []
    for start in STARTS:
        for seed in SEEDS:
            m = {"start": start, "seed": seed}
            try:
                env = TriphibianEnv(p, seed=seed)
                with teacher_controller(env).installed(env) as sc:
                    read = M.evaluate_mission(env, sc.controller, Domain(start), seed)
                m.update({"completed": read.completed, "held": read.held, "failure": read.failure,
                          "legs": [x.value for x, _ in read.schedule]})
            except Exception as exc:
                m.update({"completed": False, "controller_error": repr(exc)[:200]})
            missions.append(m)
    row["missions"] = missions
    row["controller_error"] = next((m["controller_error"] for m in missions
                                    if "controller_error" in m), "")
    row["completed"] = sum(bool(m["completed"]) for m in missions)
    row["keeps"] = row["completed"] == len(missions)
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    children = draw()
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        rows = pool.map(judge, children, chunksize=1)
    with open(Path(a.out).resolve(), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    for d in DEPTHS:
        v = [r for r in rows if r["depth"] == d]
        t0 = [r for r in v if r["tier0"]]
        err = [r for r in t0 if r.get("controller_error")]
        flown = [r for r in t0 if not r.get("controller_error")]
        print(f"depth {d}: {len(v)} children, Tier-0 passes {len(t0)}, controller cannot be built "
              f"{len(err)}, flown {len(flown)}, keep the mission {sum(r['keeps'] for r in flown)}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
