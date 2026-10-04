"""Does the shared policy carry weight?  (ROADMAP R; decides N, GRPO, and an entropy sweep.)

Each elite of a finished run is re-scored twice through the batched Tier-1
evaluator -- the path the search scores with -- at its own recorded
``eval_seed`` and ``segment_seconds``:

    with     the network that scored it (``scoring_networks/gen<gen>.npz``)
    without  ``shared=None``: its own stored policy alone

Nothing else differs: same body, same stored mobility basis, same own policy,
same scatter and task draws.  The paired difference is the shared network's
contribution to the score, and it is the only thing varied.

    python experiments/shared_policy_value/rescore.py --run runs/arch46 \
        --out experiments/shared_policy_value/results_arch46.json
    python experiments/shared_policy_value/rescore.py --run runs/arch46 --n 60   # stratified 60

Run under a memory cap on the shared machine:
    systemd-run --user --scope -q -p MemoryMax=4G env MUJOCO_GL=disable PYTHONPATH=. python ...

A third column, ``recorded``, is the competence the run stored when it scored
the elite.  ``with`` reproducing it says the re-score is the experiment that
scored the elite (a film is evidence only if it reproduces its printed score);
the paired delta does not depend on it, because both arms run today's code.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments")
MEDIA = ("air", "water", "land")
TOL = 0.005


def load_elites(run):
    from dytiscidae.ops.run import load_run_archive
    a, _ = load_run_archive(str(run))
    return list(a.cells.values())


def stratified(elites, n, seed=0):
    """Top by each medium (n/6 each), then random fill to ``n``."""
    if n >= len(elites):
        return list(range(len(elites)))
    chosen = []
    per = max(1, n // 6)
    for m in MEDIA:
        order = sorted(range(len(elites)), key=lambda i: -float((elites[i].meta or {}).get(m) or 0.0))
        chosen += [i for i in order if i not in chosen][:per]
    rest = [i for i in np.random.default_rng(seed).permutation(len(elites)) if i not in chosen]
    return chosen + rest[: n - len(chosen)]


def plain_controller(run, elite, p, seed):
    """The elite's own policy + stored bases as a plain Controller, no shared half.

    The batched evaluator takes the shared network as an argument and sums it
    in itself; the SummedPolicy that ``controller_for_elite`` returns is the
    single-machine form of the same thing, and passing both would count it twice.
    """
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.ops.run import controller_for_elite
    full = controller_for_elite(str(run), elite, p, seed, log=lambda *a, **k: None)
    if full is None:
        return None, None
    own = getattr(full.policy, "own", None) if full.policy is not None else None
    shared = getattr(full.policy, "shared", None)
    return Controller(params=full.params, policy=own, bases=full.bases), shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = every elite, else a stratified n")
    ap.add_argument("--max-groups", type=int, default=0, help="stop after this many (gen, seed) groups")
    args = ap.parse_args()

    from experiments.harness import paired_delta
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.triphibian import Domain, MissionSpec
    from dytiscidae.viz.film import control_laws, run_provenance

    run = Path(args.run)
    ok, why = batchroll.usable()
    assert ok, f"batched evaluator unusable: {why}"
    prov = run_provenance(run)
    seg = float((prov.get("config") or {}).get("segment_seconds") or 8.0)
    n_modes = int((prov.get("config") or {}).get("n_modes") or 6)
    elites = load_elites(run)
    pick = stratified(elites, args.n) if args.n else list(range(len(elites)))

    groups = collections.defaultdict(list)
    for i in pick:
        m = elites[i].meta or {}
        groups[(int(m.get("gen") or 0), int(m.get("eval_seed") or 0))].append(i)
    keys = sorted(groups)
    if args.max_groups:
        keys = keys[: args.max_groups]
    print(f"{len(elites)} elites in the archive, scoring {sum(len(groups[k]) for k in keys)} "
          f"in {len(keys)} (gen, seed) groups; segment_seconds {seg}", flush=True)

    rows, t0 = [], time.time()
    for gi, key in enumerate(keys):
        idx = groups[key]
        phenos, ctrls, like = [], [], None
        for i in idx:
            p = build(elites[i].genome)
            c, sh = plain_controller(run, elites[i], p, key[1])
            like = like or sh
            phenos.append(p)
            ctrls.append(c)
        desc, net = next(iter(control_laws(elites[idx[0]], run, like)))
        out = {}
        for arm, sh in (("with", net), ("without", None)):
            out[arm] = batchroll.evaluate_tier1_batch(
                phenos, spec=MissionSpec(), controllers=ctrls, segment_seconds=seg,
                identify_axes=False, seed=key[1], shared=sh, n_modes=n_modes)
        for slot, i in enumerate(idx):
            m = elites[i].meta or {}
            row = {"index": int(i), "gen": key[0], "island": m.get("island"),
                   "net": "scored" if "was not kept" not in desc else "final-network fallback"}
            for arm in ("with", "without"):
                r = out[arm][slot]
                row[arm] = {md: float(r.segments[Domain(md)].competence) for md in MEDIA
                            if Domain(md) in r.segments}
                row[arm]["mission_fraction"] = float(r.mission_fraction)
            row["recorded"] = {md: m.get(md) for md in MEDIA}
            row["recorded"]["mission_fraction"] = m.get("mission_fraction")
            rows.append(row)
        print(f"group {gi + 1}/{len(keys)} gen {key[0]} n={len(idx)} "
              f"wall {time.time() - t0:.0f}s", flush=True)

    # ---- summary ----------------------------------------------------------
    def col(arm, k, sel=None):
        return np.array([r[arm][k] for r in rows if sel is None or sel(r)], float)

    summary = {}
    for k in (*MEDIA, "mission_fraction"):
        w, wo = col("with", k), col("without", k)
        d = paired_delta(wo, w)                          # with - without
        d["mean_with"], d["mean_without"] = float(w.mean()), float(wo.mean())
        d["n_improved"] = int((w - wo > 1e-9).sum())
        d["n_worse"] = int((w - wo < -1e-9).sum())
        d["n_changed_over_0.01"] = int((np.abs(w - wo) > 0.01).sum())
        d["max_abs_delta"] = float(np.abs(w - wo).max())
        summary[k] = d
    repro = {}
    for k in MEDIA:
        have = [r for r in rows if r["recorded"][k] is not None]
        repro[k] = {"n": len(have),
                    "within_tol": int(sum(abs(r["with"][k] - r["recorded"][k]) <= TOL for r in have)),
                    "max_abs_gap": float(max((abs(r["with"][k] - r["recorded"][k]) for r in have), default=0.0))}
    # The same deltas on the elites whose "with" arm reproduced all three recorded
    # competences: for those the re-score is demonstrably the scoring experiment.
    def reproduced(r):
        return all(r["recorded"][k] is None or abs(r["with"][k] - r["recorded"][k]) <= TOL for k in MEDIA)
    subset = [r for r in rows if reproduced(r)]
    on_repro = {"n": len(subset)}
    for k in (*MEDIA, "mission_fraction"):
        if len(subset) > 1:
            d = paired_delta([r["without"][k] for r in subset], [r["with"][k] for r in subset])
            on_repro[k] = {"mean_delta": d["mean_delta"], "se": d["se"], "t": d["t"],
                           "n_improved": int(sum(r["with"][k] - r["without"][k] > 1e-9 for r in subset))}
    by_island = {}
    for isl in sorted({r["island"] for r in rows}):
        sel = lambda r, isl=isl: r["island"] == isl                          # noqa: E731
        by_island[isl] = {"n": int(sum(1 for r in rows if sel(r))),
                          **{k: float((col("with", k, sel) - col("without", k, sel)).mean())
                             for k in (*MEDIA, "mission_fraction")}}
    result = {"run": str(run), "segment_seconds": seg, "tolerance": TOL,
              "elites": len(rows), "of_archive": len(elites), "wall_s": time.time() - t0,
              "summary_with_minus_without": summary, "reproduction_of_recorded": repro,
              "summary_on_reproducing_elites": on_repro,
              "by_island_mean_delta": by_island, "rows": rows}
    Path(args.out).write_text(json.dumps(result, indent=1))
    for k, d in summary.items():
        print(f"{k:17s} with {d['mean_with']:.4f}  without {d['mean_without']:.4f}  "
              f"delta {d['mean_delta']:+.4f} +/- {d['se']:.4f}  t {d['t']:+.2f}  "
              f"improved {d['n_improved']}/{d['n']}  worse {d['n_worse']}  |d|>0.01 {d['n_changed_over_0.01']}")
    print("reproduction of recorded:", {k: f"{v['within_tol']}/{v['n']}" for k, v in repro.items()})
    print("on elites that reproduce all three:", json.dumps(on_repro))


if __name__ == "__main__":
    main()
