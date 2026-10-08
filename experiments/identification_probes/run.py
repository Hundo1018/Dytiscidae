"""Can identification spend fewer probes?  See README.md (ROADMAP G, after M6)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/arch48")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--probes", default="24,12,8")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, plain_controller
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.batchroll import identify_batch
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import Domain, MissionSpec, TriphibianEnv
    from dytiscidae.viz.film import control_laws, run_provenance

    ok, why = batchroll.usable()
    assert ok, why
    run = Path(args.run)
    cfg = run_provenance(run).get("config") or {}
    seg = float(cfg.get("segment_seconds") or 8.0)
    n_modes = int(cfg.get("n_modes") or 6)
    elites = load_elites(run)
    order = sorted(range(len(elites)), key=lambda i: -int((elites[i].meta or {}).get("gen") or 0))
    phenos, ctrls, picked, seeds, like = [], [], [], [], None
    for i in order:
        if len(picked) >= args.n:
            break
        m = elites[i].meta or {}
        p = build(elites[i].genome)
        c, sh = plain_controller(run, elites[i], p, int(m.get("eval_seed") or 0))
        if c is None:
            continue
        like = like or sh
        phenos.append(p)
        ctrls.append(c)
        picked.append(i)
        seeds.append(int(m.get("eval_seed") or 0))
    _, net = next(iter(control_laws(elites[picked[0]], run, like)))

    out = {"run": str(run), "elites": picked, "by_probes": {}}
    scores = {}
    for n_p in [int(x) for x in args.probes.split(",")]:
        envs = [TriphibianEnv(p, seed=s) for p, s in zip(phenos, seeds)]
        bases = [{} for _ in envs]
        t0 = time.perf_counter()
        for dom in (Domain.AIR, Domain.WATER):
            found = identify_batch(envs, dom, seed=seeds, n_probes=n_p, max_modes=n_modes)
            for b, f in zip(bases, found):
                b[dom.value] = f
        wall = time.perf_counter() - t0
        cs = [Controller(params=c.params, policy=c.policy, bases=b) for c, b in zip(ctrls, bases)]
        res = batchroll.evaluate_tier1_batch(
            phenos, spec=MissionSpec(), controllers=cs, segment_seconds=seg,
            identify_axes=False, seed=seeds, shared=net, n_modes=n_modes)
        scores[n_p] = [float(sum(s.competence for s in r.segments.values())) for r in res]
        out["by_probes"][n_p] = {"identify_wall_s": wall, "scores": scores[n_p]}
        print(f"probes {n_p}: identify {wall:.1f}s, mean summed competence "
              f"{np.mean(scores[n_p]):.4f}", flush=True)
    ref = np.array(scores[24])
    for n_p, v in out["by_probes"].items():
        d = np.abs(np.array(v["scores"]) - ref)
        v["median_abs_diff_vs_24"] = float(np.median(d))
        v["max_abs_diff_vs_24"] = float(d.max())
    v12 = out["by_probes"].get(12)
    if v12:
        out["P1"] = bool(v12["median_abs_diff_vs_24"] < 0.005)
        out["decision"] = ("n_probes 12 by default" if out["P1"] and v12["max_abs_diff_vs_24"] <= 0.044
                           else "24 stays; G closed for probes")
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
