"""B': turn authority on bodies that glide (README.md, B').

  authority.py --run runs/arch49 --out experiments/air_turn_bar/authority.json [--bodies 12] [--policies 32]
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/arch49")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bodies", type=int, default=12)
    ap.add_argument("--policies", type=int, default=32)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--screen-seed", type=int, default=2026101920)
    ap.add_argument("--repl-seeds", default="2026101921,2026101922,2026101923,2026101924")
    args = ap.parse_args()
    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, plain_controller
    from dytiscidae.control.cpg import Policy
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.viz.film import run_provenance

    run = Path(args.run)
    ok, why = batchroll.usable(); assert ok, why
    cfg = run_provenance(run).get("config") or {}
    seg = float(cfg.get("segment_seconds") or 8.0); n_modes = int(cfg.get("n_modes") or 6)
    gain = bool(cfg.get("gait_gain", False))
    elites = load_elites(run)
    pool = [(i, e) for i, e in enumerate(elites)
            if (e.meta or {}).get("body_plan") == "gannet" and not (e.meta or {}).get("n_rotors")
            and ((e.meta or {}).get("rungs") or {}).get("air", 0) >= 6]
    pool.sort(key=lambda t: -float((t[1].meta or {}).get("air") or 0.0))
    pool = pool[: args.bodies]
    print(f"{len(elites)} elites; {len(pool)} gannet gliders (rung >= 6, no rotors) measured", flush=True)
    spec = MissionSpec(); t0 = time.time(); rows = []; summary = []
    repl_seeds = [int(s) for s in args.repl_seeds.split(",")]

    def air_row(res):
        a = res.segments.get("air"); mm = (a.measurements or {}) if a is not None else {}
        return {"air": float(a.competence) if a is not None else None,
                "pair_turn_mean": mm.get("pair_turn_mean"), "pair_turn": mm.get("pair_turn"),
                "factor": mm.get("pair_turn_factor"), "turn_response": mm.get("turn_response"),
                "airborne_fraction": mm.get("airborne_fraction"), "spin": mm.get("spin_rate"),
                "bad_qacc": int(a.bad_qacc) if a is not None else None}

    for bi, (i, e) in enumerate(pool):
        p = build(e.genome)
        base_ctrl, _ = plain_controller(run, e, p, int((e.meta or {}).get("eval_seed") or 0))
        if base_ctrl is None:
            print(f"body {i}: no controller, skipped", flush=True); continue
        n_obs = TriphibianEnv.OBS_DIM
        rng = np.random.default_rng(1000 + i)
        pols = []
        for k in range(args.policies):
            pol = Policy(n_obs=n_obs, n_modes=n_modes, hidden=0, gain=gain)
            pol.weights = rng.normal(0.0, args.sigma, pol.n_weights)
            pols.append(pol)
        env = TriphibianEnv(p, seed=args.screen_seed)
        still = Controller(params=env.held_still_params(), policy=None, bases=base_ctrl.bases)
        del env
        ctrls = [Controller(params=base_ctrl.params, policy=pol, bases=base_ctrl.bases) for pol in pols] + [still]
        out = batchroll.evaluate_tier1_batch([p] * len(ctrls), spec=spec, controllers=ctrls, segment_seconds=seg,
                                             identify_axes=False, seed=[args.screen_seed] * len(ctrls),
                                             shared=None, n_modes=n_modes)
        scr = [air_row(r) for r in out]
        for k, r in enumerate(scr):
            rows.append({"body": i, "phase": "screen", "policy": k if k < args.policies else "still",
                         "seed": args.screen_seed, **r})
        meas = [(k, r["pair_turn_mean"]) for k, r in enumerate(scr[:-1])
                if (r["factor"] or 0) > 0 and r["pair_turn_mean"] is not None]
        meas.sort(key=lambda t: -t[1])
        top = [k for k, _ in meas[: args.top]]
        del out, ctrls
        # replicate the top policies and the still machine at fresh draws
        ctrls = [copy.deepcopy(Controller(params=base_ctrl.params, policy=pols[k], bases=base_ctrl.bases))
                 for k in top for _ in repl_seeds]
        seeds = [s for _ in top for s in repl_seeds]
        env = TriphibianEnv(p, seed=repl_seeds[0])
        for s in repl_seeds:
            ctrls.append(Controller(params=env.held_still_params(), policy=None, bases=base_ctrl.bases)); seeds.append(s)
        del env
        rep = {}
        if ctrls:
            out = batchroll.evaluate_tier1_batch([p] * len(ctrls), spec=spec, controllers=ctrls, segment_seconds=seg,
                                                 identify_axes=False, seed=seeds, shared=None, n_modes=n_modes)
            labels = [k for k in top for _ in repl_seeds] + ["still"] * len(repl_seeds)
            for lab, s, r in zip(labels, seeds, out):
                rr = air_row(r); rows.append({"body": i, "phase": "repl", "policy": lab, "seed": s, **rr})
                rep.setdefault(lab, []).append(rr)
            del out, ctrls
        def mean_meas(rs):
            xs = [r["pair_turn_mean"] for r in rs if (r["factor"] or 0) > 0 and r["pair_turn_mean"] is not None]
            return (float(np.mean(xs)) if xs else None, len(xs))
        best = None
        for k in top:
            m, n = mean_meas(rep.get(k, []))
            if m is not None and (best is None or m > best[1]):
                best = (k, m, n)
        still_vals = [r["pair_turn_mean"] for r in rep.get("still", []) if (r["factor"] or 0) > 0 and r["pair_turn_mean"] is not None]
        summary.append({"body": i, "recorded_air": (e.meta or {}).get("air"), "rung": ((e.meta or {}).get("rungs") or {}).get("air"),
                        "screen_measurable": len(meas), "screen_best": meas[0][1] if meas else None,
                        "screen_positive": sum(1 for _, v in meas if v > 0),
                        "top": top, "repl_best": best, "still_repl_max": max(still_vals, default=None),
                        "still_measurable": len(still_vals)})
        print(f"body {bi + 1}/{len(pool)} idx {i}: screen measurable {len(meas)}/{args.policies} best "
              f"{meas[0][1] if meas else None} -> repl best {best}; still max {max(still_vals, default=None)}; "
              f"wall {time.time() - t0:.0f}s", flush=True)
        gc.collect()
    out_p = Path(args.out)
    out_p.with_suffix(".rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    n_auth = sum(1 for s in summary if s["repl_best"] and s["repl_best"][1] > 0.1)
    res = {"run": args.run, "bodies": len(summary), "policies": args.policies, "sigma": args.sigma,
           "wall_s": time.time() - t0, "bodies_with_replicated_mean_gt_0.1": n_auth, "summary": summary}
    out_p.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "summary"}, indent=1))


if __name__ == "__main__":
    main()
