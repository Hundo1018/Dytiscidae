"""What share of a batched Tier-1 evaluation's wall is mobility identification?

ROADMAP M6 (2026-10-08).  Identification is 70% of the main path's physics
*steps* in arch48 and has been batched on the GPU since ``identify_batch``, so
its share of *seconds* is unmeasured.  Decision rule (ROADMAP M6, written
before the data): under a third of the main evaluation's wall, item G is
closed as a throughput lever.

The ``--n`` latest elites of a run (rotor-heavy late bodies), one batch, the
network that scored the first of them, the run's segment length: timed with
``identify_axes=True`` and ``False``, ``--repeats`` times each, alternating,
on a quiet machine.  Single process, no pool: the share, not the wall, is the
read.

    PYTHONPATH=. .venv/bin/python experiments/identification_share/run.py \
        --run runs/arch48 --out experiments/identification_share/results_arch48.json
"""
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
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, plain_controller
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import MissionSpec
    from dytiscidae.viz.film import control_laws, run_provenance

    ok, why = batchroll.usable()
    assert ok, why
    run = Path(args.run)
    cfg = run_provenance(run).get("config") or {}
    seg = float(cfg.get("segment_seconds") or 8.0)
    n_modes = int(cfg.get("n_modes") or 6)
    elites = load_elites(run)
    order = sorted(range(len(elites)), key=lambda i: -int((elites[i].meta or {}).get("gen") or 0))
    phenos, ctrls, picked, like = [], [], [], None
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
    _, net = next(iter(control_laws(elites[picked[0]], run, like)))
    rotors = [int((elites[i].meta or {}).get("n_rotors") or 0) for i in picked]

    def once(identify):
        cs = [Controller(params=c.params, policy=c.policy, bases=c.bases) for c in ctrls]
        t0 = time.perf_counter()
        batchroll.evaluate_tier1_batch(phenos, spec=MissionSpec(), controllers=cs,
                                       segment_seconds=seg, identify_axes=identify,
                                       seed=[int((elites[i].meta or {}).get("eval_seed") or 0)
                                             for i in picked],
                                       shared=net, n_modes=n_modes)
        return time.perf_counter() - t0

    once(False)                                   # warm the kernel and caches
    walls = {"with": [], "without": []}
    for _ in range(args.repeats):
        walls["with"].append(once(True))
        walls["without"].append(once(False))
    w, wo = float(np.median(walls["with"])), float(np.median(walls["without"]))
    out = {"run": str(run), "elites": picked, "n_rotors": rotors, "segment_seconds": seg,
           "walls": walls, "median_with": w, "median_without": wo,
           "identification_s": w - wo, "identification_share": (w - wo) / w,
           "decision": ("G closed: under a third" if (w - wo) / w < 1 / 3
                        else "G open: identification is at least a third")}
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
