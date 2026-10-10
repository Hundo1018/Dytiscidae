"""R2 (ROADMAP "R1, the open read"): why do 25 of 58 land-competent arch49 elites not
reproduce their recorded land score on the batched path?  Three rivals, land only.

  swap   (a) one stored input replaced at a time: final network / freshly identified
             basis / no stored policy.  Grouping identical to R1 (by scoring network, gen).
  batch  (b) each target alone and inside 3 random batches of 8 other arch49 elites.
  chaos  (c) one batch [t, t, t with qpos[0] += 1e-9 after scatter].

Targets: the 25 batched non-reproducers of R1 plus the first 10 reproducers by index.

  flock /home/hundo/.cache/dytiscidae-gpu.lock env MUJOCO_GL=disable PYTHONPATH=$PWD \
     DYTISCIDAE_KERNEL_DIR=<main>/mojo/build <main>/.venv/bin/python -u \
     experiments/contact_model_dependence/r2.py --phase swap|batch|chaos --run <main>/runs/arch49
  ... --phase summarise
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, "experiments/shared_policy_value")
D = Path("experiments/contact_model_dependence")
R1 = D / "results_arch49_r1_batched.json.rows.jsonl"
LAND_MIN, TOL, PERT = 0.012, 0.005, 1e-9


def setup(run):
    import torch
    torch.set_num_threads(1)
    from rescore import load_elites
    from dytiscidae.viz.film import run_provenance
    cfg = run_provenance(Path(run)).get("config") or {}
    elites = load_elites(Path(run))
    r1 = {r["index"]: r for r in map(json.loads, R1.read_text().splitlines())}
    pick = sorted(r1)
    non = [i for i in pick if abs(r1[i]["comp"]["land"] - r1[i]["recorded_land"]) > TOL]
    rep = [i for i in pick if i not in non][:10]
    return (elites, float(cfg.get("segment_seconds") or 8.0), int(cfg.get("n_modes") or 6),
            r1, pick, non, rep)


def mk(run, elites, i):
    from rescore import plain_controller
    from dytiscidae.core.phenotype import build
    from dytiscidae.viz.film import control_laws
    el = elites[i]
    seed = int((el.meta or {}).get("eval_seed") or 0)
    p = build(el.genome)
    c, final = plain_controller(run, el, p, seed)
    desc, net = next(iter(control_laws(el, run, final)))
    return dict(i=i, p=p, c=c, seed=seed, net=net, final=final, desc=desc,
                gen=int((el.meta or {}).get("gen") or 0))


def call(items, seg, nm, *, shared, identify=False, perts=()):
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.evaluate import Controller
    cs = [Controller(params=x["c"].params, policy=x["c"].policy, bases=x["c"].bases) for x in items]
    for k, x in enumerate(items):
        x["p"]._r2_pert = k in perts
    res = batchroll.evaluate_tier1_batch(
        [x["p"] for x in items], controllers=cs, segment_seconds=seg, identify_axes=identify,
        seed=[x["seed"] for x in items], shared=shared, n_modes=nm)
    return [dict(land=float(r.segments["land"].competence), air=float(r.segments["air"].competence),
                 water=float(r.segments["water"].competence),
                 bad=int(r.segments["land"].bad_qacc)) for r in res]


def patch():
    from dytiscidae.envs import triphibian as tp
    orig = tp.TriphibianEnv.scatter

    def scatter(self, rng, **k):
        orig(self, rng, **k)
        if getattr(self.p, "_r2_pert", False):
            self.data.qpos[0] += PERT
    tp.TriphibianEnv.scatter = scatter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=0, help="smoke: first n targets")
    a = ap.parse_args()
    if a.phase == "summarise":
        return summarise()
    patch()
    run = a.run
    elites, seg, nm, r1, pick, non, rep = setup(run)
    targets = (non + rep)[: a.n] if a.n else non + rep
    out = open(D / f"r2_{a.phase}.rows.jsonl", "w")
    t0 = time.time()

    def emit(**r):
        out.write(json.dumps(r) + "\n"); out.flush()

    if a.phase == "swap":
        groups = {}
        for i in pick:
            x = mk(run, elites, i)
            groups.setdefault((x["desc"], x["gen"]), []).append(x)
        for arm in ("final", "basis", "nopolicy"):
            for gi, (key, items) in enumerate(sorted(groups.items(), key=lambda kv: kv[0][1])):
                if not any(x["i"] in targets for x in items):
                    continue
                its, sh, ident = items, items[0]["net"], False
                if arm == "final":
                    sh = items[0]["final"]
                elif arm == "basis":
                    ident = True
                else:
                    its = [dict(x, c=type(x["c"])(params=x["c"].params, policy=None, bases=x["c"].bases))
                           for x in items]
                res = call(its, seg, nm, shared=sh, identify=ident)
                for x, r in zip(items, res):
                    emit(arm=arm, index=x["i"], gen=x["gen"], law=key[0], **r)
                print(f"{arm} group {gi} gen {key[1]} n={len(items)} {time.time()-t0:.0f}s", flush=True)
    elif a.phase == "batch":
        others_all = [i for i in range(len(elites))]
        for n, t in enumerate(targets):
            x = mk(run, elites, t)
            r = call([x], seg, nm, shared=x["net"])[0]
            emit(arm="alone", index=t, **r)
            for s in (101, 102, 103):
                rng = np.random.default_rng([s, t])
                oth = [int(j) for j in rng.choice([j for j in others_all if j != t], 8, replace=False)]
                pos = int(rng.integers(0, 9))
                members = [mk(run, elites, j) for j in oth]
                members.insert(pos, mk(run, elites, t))
                r = call(members, seg, nm, shared=x["net"])[pos]
                emit(arm=f"rand{s}", index=t, pos=pos, others=oth, **r)
            print(f"[{n+1}/{len(targets)}] elite {t} {time.time()-t0:.0f}s", flush=True)
    elif a.phase == "chaos":
        for n, t in enumerate(targets):
            items = [mk(run, elites, t) for _ in range(3)]
            res = call(items, seg, nm, shared=items[0]["net"], perts=(2,))
            for k, r in enumerate(res):
                emit(arm=("rep1", "rep2", "pert")[k], index=t, **r)
            print(f"[{n+1}/{len(targets)}] elite {t} {time.time()-t0:.0f}s", flush=True)
    out.close()


def summarise():
    elites_r1 = {}
    _e, _s, _n, r1, pick, non, rep = setup_light()
    rec = {i: r1[i]["recorded_land"] for i in pick}
    base = {i: r1[i]["comp"]["land"] for i in pick}
    S = {"n_nonrepro": len(non), "n_control": len(rep), "nonrepro": non, "control": rep}
    rows = lambda ph: [json.loads(l) for l in (D / f"r2_{ph}.rows.jsonl").read_text().splitlines()] \
        if (D / f"r2_{ph}.rows.jsonl").exists() else []
    sw = rows("swap"); a = {}
    for arm in ("final", "basis", "nopolicy"):
        v = {r["index"]: r["land"] for r in sw if r["arm"] == arm}
        restored = [i for i in non if i in v and abs(v[i] - rec[i]) <= TOL]
        broke = [i for i in rep if i in v and abs(v[i] - rec[i]) > TOL]
        moved = [i for i in non if i in v and abs(v[i] - base[i]) > TOL]
        a[arm] = {"restored_of_25": len(restored), "restored": restored, "control_broken_of_10": len(broke),
                  "nonrepro_moved_from_R1": len(moved), "n": len([i for i in non if i in v])}
    S["a_swaps"] = a
    b = rows("batch"); bb = {}
    for r in b:
        bb.setdefault(r["index"], {})[r["arm"]] = r["land"]
    rng_ = {i: max(v.values()) - min(v.values()) for i, v in bb.items() if len(v) == 4}
    S["b_batch"] = {"n": len(rng_), "nonrepro_range_gt_0.005": sum(rng_[i] > TOL for i in non if i in rng_),
                    "control_range_gt_0.005": sum(rng_[i] > TOL for i in rep if i in rng_),
                    "median_range_nonrepro": float(np.median([rng_[i] for i in non if i in rng_])) if rng_ else None,
                    "alone_equals_R1_within_0.005": sum(abs(bb[i]["alone"] - base[i]) <= TOL for i in bb),
                    "alone_reproduces_record": sum(abs(bb[i]["alone"] - rec[i]) <= TOL for i in non if i in bb),
                    "per_elite_range": {str(i): rng_[i] for i in rng_}}
    c = rows("chaos"); cc = {}
    for r in c:
        cc.setdefault(r["index"], {})[r["arm"]] = r["land"]
    ident = [i for i, v in cc.items() if v["rep1"] == v["rep2"]]
    dp = {i: abs(v["pert"] - v["rep1"]) for i, v in cc.items()}
    S["c_chaos"] = {"n": len(cc), "repeats_identical": len(ident),
                    "pert_changed_gt_0.005_nonrepro": sum(dp[i] > TOL for i in non if i in dp),
                    "pert_changed_gt_0.005_control": sum(dp[i] > TOL for i in rep if i in dp),
                    "median_abs_change_nonrepro": float(np.median([dp[i] for i in non if i in dp])) if dp else None,
                    "median_record_gap_nonrepro": float(np.median([abs(base[i] - rec[i]) for i in non])),
                    "rep1_equals_alone": sum(abs(cc[i]["rep1"] - bb[i]["alone"]) <= 1e-12 for i in cc if i in bb)}
    (D / "results_arch49_r2.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: v for k, v in S.items() if k not in ("nonrepro", "control")}, indent=1)[:4000])


def setup_light():
    r1 = {r["index"]: r for r in map(json.loads, R1.read_text().splitlines())}
    pick = sorted(r1)
    non = [i for i in pick if abs(r1[i]["comp"]["land"] - r1[i]["recorded_land"]) > TOL]
    rep = [i for i in pick if i not in non][:10]
    return None, None, None, r1, pick, non, rep


if __name__ == "__main__":
    main()
