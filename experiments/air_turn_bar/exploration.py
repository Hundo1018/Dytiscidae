"""Air list item 6, the learner's half (pre-registered in README.md, "Exploration read"):
how much of the shared policy's sampling noise keeps a gannet aloft through the command?

  exploration.py --run runs/arch49 --out experiments/air_turn_bar/exploration.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/arch49"); ap.add_argument("--out", required=True)
    ap.add_argument("--copies", type=int, default=32); ap.add_argument("--seed", type=int, default=2026102130)
    ap.add_argument("--log-stds", default="-0.5,-1.5,-2.5,-4.0")
    args = ap.parse_args()
    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, plain_controller
    import torch
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.learning.ppo import RolloutBuffer, SharedPolicy
    from dytiscidae.viz.film import run_provenance
    run = Path(args.run); ok, why = batchroll.usable(); assert ok, why
    cfg = run_provenance(run).get("config") or {}
    seg = float(cfg.get("segment_seconds") or 8.0); n_modes = int(cfg.get("n_modes") or 6)
    elites = load_elites(run)
    pool = [(i, e) for i, e in enumerate(elites) if (e.meta or {}).get("body_plan") == "gannet"
            and not (e.meta or {}).get("n_rotors") and ((e.meta or {}).get("rungs") or {}).get("air", 0) >= 6]
    pool.sort(key=lambda t: -float((t[1].meta or {}).get("air") or 0.0)); pool = pool[:8]
    net = SharedPolicy(TriphibianEnv.OBS_DIM, 6, hidden=16)
    with torch.no_grad():
        last = [m for m in net.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
        last.weight.zero_(); last.bias.zero_()
    spec = MissionSpec(); t0 = time.time(); out = {"bodies": [], "by_log_std": {}}
    stds = [float(x) for x in args.log_stds.split(",")]
    agg = {s: [0, 0, 0] for s in stds}  # measurable, positive, total
    for i, e in pool:
        p = build(e.genome)
        base, _ = plain_controller(run, e, p, int((e.meta or {}).get("eval_seed") or 0))
        if base is None: continue
        row = {"body": i, "recorded_air": (e.meta or {}).get("air")}
        for s in stds:
            with torch.no_grad(): net.log_std.fill_(s)
            buf = RolloutBuffer(shaping=0.0)
            ctrls = [Controller(params=base.params, policy=None, bases=base.bases) for _ in range(args.copies)]
            res = batchroll.evaluate_tier1_batch([p] * args.copies, spec=spec, controllers=ctrls, segment_seconds=seg,
                                                 identify_axes=False, seed=[args.seed] * args.copies, shared=net, buffer=buf, n_modes=n_modes)
            ms = [(r.segments["air"].measurements or {}) for r in res]
            meas = [m for m in ms if (m.get("pair_turn_factor") or 0) > 0.5 and m.get("pair_turn_mean") is not None]
            pos = sum(1 for m in meas if m["pair_turn_mean"] > 0)
            row[str(s)] = {"measurable": len(meas), "positive": pos, "n": len(ms)}
            agg[s][0] += len(meas); agg[s][1] += pos; agg[s][2] += len(ms)
            del res, buf, ctrls
        out["bodies"].append(row)
        print(f"body {i}: " + " ".join(f"std{s}: {row[str(s)]['measurable']}/{row[str(s)]['n']} meas, {row[str(s)]['positive']} pos" for s in stds) + f"  wall {time.time()-t0:.0f}s", flush=True)
    out["by_log_std"] = {str(s): {"measurable": a[0], "positive": a[1], "n": a[2], "measurable_share": (a[0] / a[2] if a[2] else None)} for s, a in agg.items()}
    out["wall_s"] = time.time() - t0
    Path(args.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out["by_log_std"], indent=1))

if __name__ == "__main__":
    main()
