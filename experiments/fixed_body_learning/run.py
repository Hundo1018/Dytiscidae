"""Does the shared policy learn on a body that does not change?  (ROADMAP M4.)

See README.md for the method, the pre-registered predictions and the decision
rule.  Three arch48 elites (best water, land, air), 3 learner seeds, 30 PPO
updates of 8 rollouts each through the search's own batched path and
``ppo_update``; held-out deterministic evaluations every 10 updates against a
still machine and the base gait.

    python experiments/fixed_body_learning/run.py --run runs/arch48 \
        --out experiments/fixed_body_learning/results_arch48.json
    python experiments/fixed_body_learning/run.py --from-rows <out>.rows.jsonl --out <out>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MEDIA = ("air", "water", "land")
HELD_OUT = [9001 + 17 * j for j in range(8)]


def competences(results):
    """Mean competence per medium over a list of MissionResults."""
    return {m: float(np.mean([r.segments[m].competence if m in r.segments else 0.0
                              for r in results])) for m in MEDIA}


def analyse(rows):
    out = {}
    for body in sorted({r["body"] for r in rows}):
        rs = [r for r in rows if r["body"] == body]
        m = rs[0]["medium"]
        first = [r["curve"][0]["held_out"][m] for r in rs]
        last = [r["curve"][-1]["held_out"][m] for r in rs]
        gains = [b - a for a, b in zip(first, last)]
        spread = max(last) - min(last)
        out[body] = {
            "medium": m, "seeds": len(rs),
            "update0": first, "final": last, "gain_mean": float(np.mean(gains)),
            "seed_spread_final": spread,
            "learned": bool(np.mean(gains) > spread and np.mean(gains) > 0),
            "still": rs[0]["still"][m], "base_gait": rs[0]["base_gait"][m],
            "above_base_gait": bool(np.mean(last) > rs[0]["base_gait"][m]),
            "curve_mean": [float(np.mean([r["curve"][i]["held_out"][m] for r in rs]))
                           for i in range(len(rs[0]["curve"]))],
            "updates": [c["update"] for c in rs[0]["curve"]],
        }
    learned = sum(v["learned"] for v in out.values())
    return {"bodies": out, "P1_holds": learned < 2 if len(out) >= 3 else None,
            "decision": ("learner is the wall: M5 next" if learned < 2
                         else "coupling is the wall: timescale split next")}


def measure(args, rows_path):
    import torch

    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, plain_controller
    from dytiscidae.core.phenotype import build
    from dytiscidae.control.cpg import TWIST_DIM
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import MissionSpec, TriphibianEnv
    from dytiscidae.evolution.loop import SearchConfig
    from dytiscidae.learning import ppo as _ppo
    from dytiscidae.viz.film import run_provenance

    ok, why = batchroll.usable()
    assert ok, why
    run = Path(args.run)
    rcfg = run_provenance(run).get("config") or {}
    seg = float(rcfg.get("segment_seconds") or 8.0)
    n_modes = int(rcfg.get("n_modes") or 6)
    cfg = SearchConfig()
    elites = load_elites(run)
    picks = {}
    for m in MEDIA:
        best = max(range(len(elites)), key=lambda i: float((elites[i].meta or {}).get(m) or 0.0))
        picks[m] = best
    spec = MissionSpec()
    fh = open(rows_path, "w")
    t0 = time.time()
    for medium, idx in picks.items():
        e = elites[idx]
        m = e.meta or {}
        pheno = build(e.genome)
        c, _ = plain_controller(run, e, pheno, int(m.get("eval_seed") or 0))
        bases = c.bases
        env = TriphibianEnv(pheno, seed=0)

        def held_out(policy_shared, params=None):
            ctrls = [Controller(params=params, policy=None, bases=bases) for _ in HELD_OUT]
            res = batchroll.evaluate_tier1_batch(
                [pheno] * len(HELD_OUT), spec=spec, controllers=ctrls,
                segment_seconds=seg, identify_axes=False, seed=list(HELD_OUT),
                shared=policy_shared, n_modes=n_modes)
            return competences(res)

        print(f"{medium}: elite {idx} loaded ({time.time() - t0:.0f}s)", flush=True)
        still = held_out(None, env.held_still_params())
        base_gait = held_out(None, None)
        print(f"{medium}: still {still} base gait {base_gait} ({time.time() - t0:.0f}s)", flush=True)
        for s in range(args.seeds):
            torch.manual_seed(1000 + s)
            pol = _ppo.SharedPolicy(TriphibianEnv.OBS_DIM, TWIST_DIM, hidden=cfg.shared_hidden)
            opt = torch.optim.Adam(pol.parameters(), lr=cfg.shared_lr, eps=1e-5)
            lrng = np.random.default_rng(2000 + s)
            drng = np.random.default_rng([idx, s, 0xF1])
            curve = [{"update": 0, "held_out": held_out(pol)}]
            infos = []
            for u in range(1, args.updates + 1):
                buf = _ppo.RolloutBuffer(shaping=cfg.reward_shaping)
                draws = [int(x) for x in drng.integers(1 << 30, size=args.batch)]
                ctrls = [Controller(params=None, policy=None, bases=bases) for _ in draws]
                batchroll.evaluate_tier1_batch(
                    [pheno] * args.batch, spec=spec, controllers=ctrls,
                    segment_seconds=seg, identify_axes=False, seed=draws,
                    shared=pol, buffer=buf, n_modes=n_modes,
                    streams=list(range(args.batch)))
                info = _ppo.ppo_update(
                    pol, buf, lr=cfg.shared_lr, epochs=cfg.shared_epochs,
                    minibatch=cfg.shared_minibatch, target_kl=cfg.shared_target_kl,
                    ent_coef=cfg.shared_ent_coef, lr_fraction=1.0, optimiser=opt, rng=lrng)
                infos.append({k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                              for k, v in (info or {}).items()
                              if isinstance(v, (int, float, np.floating))})
                if u % args.every == 0:
                    curve.append({"update": u, "held_out": held_out(pol)})
                    print(f"{medium} elite {idx} seed {s} update {u}: "
                          f"{curve[-1]['held_out']}  ({time.time() - t0:.0f}s)", flush=True)
            fh.write(json.dumps({"body": f"{medium}:{idx}", "medium": medium, "index": idx,
                                 "seed": s, "curve": curve, "still": still,
                                 "base_gait": base_gait, "updates_info": infos}) + "\n")
            fh.flush()
    fh.close()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/arch48")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--from-rows")
    args = ap.parse_args()
    out = Path(args.out)
    rows_path = Path(args.from_rows) if args.from_rows else Path(str(out) + ".rows.jsonl")
    wall = float("nan") if args.from_rows else measure(args, rows_path)
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    result = {"run": args.run, "wall_s": wall, "held_out_draws": HELD_OUT,
              **analyse(rows)}
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
