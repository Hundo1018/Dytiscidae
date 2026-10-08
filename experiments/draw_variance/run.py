"""How much of a Tier-1 score is the draw?  (ROADMAP M2 / C2.)

See README.md for the question, the pre-registered predictions and the decision
rule.  Every elite of a finished run is scored through
``batchroll.evaluate_tier1_batch`` with the network that scored it, at its
recorded draw ``s`` and at ``K`` fresh draws of its own, one batch per
(gen, seed) group with one seed per machine.

    systemd-run --user --unit draw-variance -p MemoryMax=3500M --same-dir \
        env MUJOCO_GL=disable PYTHONPATH=. \
        .venv/bin/python experiments/draw_variance/run.py \
        --run runs/arch48 --out experiments/draw_variance/results_arch48.json

    python experiments/draw_variance/run.py --selftest
    python experiments/draw_variance/run.py --from-rows <out>.rows.jsonl --out <out>
"""
import argparse
import collections
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

MEDIA = ("air", "water", "land")
PLACEMENT = ("island", "stage")
SCORES = MEDIA + PLACEMENT
MIN_NONZERO = 20
CHUNK = 64


def fresh_seed(s: int, index: int, j: int) -> int:
    """Fresh draw ``j`` for one elite, used only by this experiment."""
    return int(np.random.default_rng([int(s), int(index), int(j), 0xC2]).integers(1 << 30))


# --------------------------------------------------------------------------
# Rows -> the table (numpy only, no simulator)
# --------------------------------------------------------------------------


def _ranks(x):
    """Average ranks, ties shared (many scores are exactly zero)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size)
    r[order] = np.arange(x.size, dtype=float)
    _, inv = np.unique(x, return_inverse=True)
    sums = np.bincount(inv, weights=r)
    counts = np.bincount(inv)
    return sums[inv] / counts[inv]


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def components(fresh):
    """``fresh``: (n designs, K draws).  Draw and design variance, and R."""
    fresh = np.asarray(fresh, float)
    n, k = fresh.shape
    draw_var = float(np.mean(np.var(fresh, axis=1, ddof=1)))
    design_var = max(float(np.var(fresh.mean(axis=1), ddof=1)) - draw_var / k, 0.0)
    ratio = draw_var / design_var if design_var > 0 else float("inf")
    return draw_var, design_var, ratio


def read(s, fresh, n_boot=2000, seed=0):
    """One score's table entry.  ``s``: (n,), ``fresh``: (n, K)."""
    s, fresh = np.asarray(s, float), np.asarray(fresh, float)
    n, k = fresh.shape
    draw_var, design_var, ratio = components(fresh)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(components(fresh[idx])[2])
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    icc = design_var / (design_var + draw_var) if (design_var + draw_var) > 0 else float("nan")
    nonzero = int((fresh > 0).any(axis=1).sum())
    mean_s, mean_f = float(s.mean()), float(fresh.mean())
    return {
        "n": int(n), "k": int(k), "nonzero_any_fresh": nonzero,
        "underpowered": nonzero < MIN_NONZERO,
        "mean_s": mean_s, "mean_fresh": mean_f,
        "curse": (1.0 - mean_f / mean_s) if mean_s > 0 else float("nan"),
        "draw_var": draw_var, "design_var": design_var,
        "R": ratio, "R_ci": [lo, hi],
        "icc1": icc, "k80": 4.0 * ratio,
        "rank": spearman(fresh[:, 0], fresh[:, 1:].mean(axis=1)),
    }


def verdict(entry):
    if entry["underpowered"]:
        return "underpowered"
    lo, hi = entry["R_ci"]
    if lo > 1.0:
        return "draw dominates: sequential evaluation earns a build"
    if hi < 1.0:
        return "design dominates: one draw per candidate is enough"
    return f"undecided: k80 = {entry['k80']:.1f}, build nothing"


def analyse(rows, n_boot=2000):
    out = {}
    for name in SCORES:
        s = [r["score"]["s"][name] for r in rows]
        fresh = [[f[name] for f in r["score"]["fresh"]] for r in rows]
        e = read(s, fresh, n_boot=n_boot)
        e["verdict"] = verdict(e)
        out[name] = e
    return out


def predictions(table):
    def ge(name, key, bar):
        v = table[name][key]
        return None if table[name]["underpowered"] or v != v else bool(v >= bar)
    return {
        "P1": {m: ge(m, "R", 1.0) for m in ("water", "land")},
        "P2": {m: ge(m, "curse", 0.20) for m in ("water", "land")},
        "P3": {m: (None if table[m]["underpowered"] else bool(table[m]["rank"] < 0.7))
               for m in ("water", "land")},
        "P4": ge("stage", "R", 1.0),
    }


def selftest():
    rng = np.random.default_rng(1)
    n, k = 300, 6
    # Design variance 1, draw variance 4 -> R = 4, icc 0.2, k80 16.
    design = rng.normal(0.0, 1.0, n)
    fresh = design[:, None] + rng.normal(0.0, 2.0, (n, k))
    e = read(design, fresh, n_boot=200)
    assert 3.0 < e["R"] < 5.5, e
    assert 0.12 < e["icc1"] < 0.3, e
    assert e["R_ci"][0] > 1.0 and verdict(e).startswith("draw dominates"), e
    # Draws identical -> no draw variance; design dominates.
    flat = np.repeat(design[:, None] + 5.0, k, axis=1)
    e2 = read(design + 5.0, flat, n_boot=200)
    assert e2["draw_var"] < 1e-20 and e2["R"] < 1e-20, e2
    assert verdict(e2).startswith("design dominates"), e2
    # A winner's curse: s is the best of the draws.
    best = fresh.max(axis=1) + 10.0
    e3 = read(best, fresh + 10.0, n_boot=50)
    assert e3["curse"] > 0.0, e3
    # Ties: Spearman on mostly zeros is finite and bounded.
    a = np.r_[np.zeros(50), np.arange(10.0)]
    assert abs(spearman(a, a) - 1.0) < 1e-12
    # Underpowered: fewer than MIN_NONZERO nonzero designs.
    z = np.zeros((100, k))
    z[:5, 0] = 0.1
    e4 = read(np.zeros(100), z, n_boot=20)
    assert e4["underpowered"] and verdict(e4) == "underpowered", e4
    # analyse/predictions on rows.
    rows = [{"score": {"s": {m: float(best[i]) for m in SCORES},
                       "fresh": [{m: float(fresh[i, j] + 10.0) for m in SCORES} for j in range(k)]}}
            for i in range(n)]
    t = analyse(rows, n_boot=50)
    p = predictions(t)
    assert p["P1"]["water"] is True and p["P4"] is True, p
    print("draw_variance selftest passed")


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


def measure(args, rows_path):
    import importlib.util
    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, plain_controller, stratified
    spec_ = importlib.util.spec_from_file_location(
        "refine_criterion_run", "experiments/refine_criterion/run.py")
    rc = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(rc)
    criteria_of = rc.criteria_of
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.evaluate import Controller
    from dytiscidae.envs.triphibian import MissionSpec
    from dytiscidae.viz.film import control_laws, run_provenance

    run = Path(args.run)
    ok, why = batchroll.usable()
    assert ok, f"batched evaluator unusable: {why}"
    cfg = run_provenance(run).get("config") or {}
    seg = float(cfg.get("segment_seconds") or 8.0)
    n_modes = int(cfg.get("n_modes") or 6)
    elites = load_elites(run)
    pick = stratified(elites, args.n) if args.n else list(range(len(elites)))
    groups = collections.defaultdict(list)
    for i in pick:
        m = elites[i].meta or {}
        groups[(int(m.get("gen") or 0), int(m.get("eval_seed") or 0))].append(i)
    keys = sorted(groups)
    K = args.k
    print(f"{len(elites)} elites, measuring {sum(len(groups[k]) for k in keys)} in "
          f"{len(keys)} (gen, seed) groups; K={K}, segment_seconds {seg}", flush=True)

    spec = MissionSpec()
    t0, skipped = time.time(), {"no_controller": []}
    done = set()
    if rows_path.exists() and args.resume:
        done = {json.loads(ln)["index"] for ln in rows_path.read_text().splitlines()}
    fh = open(rows_path, "a" if args.resume else "w")
    for gi, key in enumerate(keys):
        phenos, ctrls, keep, like = [], [], [], None
        for i in groups[key]:
            if i in done:
                continue
            p = build(elites[i].genome)
            c, sh = plain_controller(run, elites[i], p, key[1])
            if c is None:
                skipped["no_controller"].append(int(i))
                continue
            like = like or sh
            phenos.append(p)
            ctrls.append(c)
            keep.append(i)
        if not keep:
            continue
        _, net = next(iter(control_laws(elites[keep[0]], run, like)))
        # Slot layout: draw-major, [s for every elite, f1 for every elite, ...].
        all_p, all_c, all_s = [], [], []
        for j in range(K + 1):
            for slot, i in enumerate(keep):
                c = ctrls[slot]
                all_p.append(phenos[slot])
                all_c.append(Controller(params=c.params, policy=c.policy, bases=c.bases))
                all_s.append(key[1] if j == 0 else fresh_seed(key[1], i, j))
        res = []
        for a in range(0, len(all_p), CHUNK):
            res += batchroll.evaluate_tier1_batch(
                all_p[a:a + CHUNK], spec=spec, controllers=all_c[a:a + CHUNK],
                segment_seconds=seg, identify_axes=False, seed=all_s[a:a + CHUNK],
                shared=net, n_modes=n_modes)
        n = len(keep)
        for slot, i in enumerate(keep):
            m = elites[i].meta or {}
            island = m.get("island") or "generalist"
            stage = int(m.get("stage") or 0)

            def score(r):
                out = {d: float(r.segments[d].competence) if d in r.segments else 0.0
                       for d in MEDIA}
                crit = criteria_of(r, island, stage)
                out.update({c: crit[c] for c in PLACEMENT})
                return out

            mine = [res[j * n + slot] for j in range(K + 1)]
            assert all(int(r.eval_seed) == int(all_s[j * n + slot])
                       for j, r in enumerate(mine)), "a result does not carry its seed"
            rec = {"index": int(i), "gen": key[0], "seed": key[1],
                   "fresh_seeds": [int(all_s[j * n + slot]) for j in range(1, K + 1)],
                   "island": island, "stage": stage, "plan": m.get("body_plan") or "?",
                   "recorded": {d: m.get(d) for d in MEDIA},
                   "score": {"s": score(mine[0]), "fresh": [score(r) for r in mine[1:]]},
                   "bad_qacc": int(sum(int(getattr(r.segments[d], "bad_qacc", 0))
                                       for r in mine for d in MEDIA if d in r.segments))}
            fh.write(json.dumps(rec) + "\n")
        fh.flush()
        del res, all_p, all_c, phenos, ctrls
        gc.collect()
        print(f"group {gi + 1}/{len(keys)} gen {key[0]} n={n} wall {time.time() - t0:.0f}s",
              flush=True)
    fh.close()
    return time.time() - t0, skipped, len(elites)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=0, help="0 = every elite, else a stratified n")
    ap.add_argument("--k", type=int, default=6, help="fresh draws per elite")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--from-rows")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    out = Path(args.out)
    rows_path = Path(args.from_rows) if args.from_rows else Path(str(out) + ".rows.jsonl")
    if args.from_rows:
        wall, skipped, n_archive = float("nan"), {}, 0
    else:
        wall, skipped, n_archive = measure(args, rows_path)
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines()]
    table = analyse(rows)
    result = {"run": args.run, "n_elites": len(rows), "of_archive": n_archive,
              "k": args.k, "wall_s": wall, "skipped": skipped, "table": table,
              "predictions": predictions(table),
              "bad_qacc_rows": int(sum(r["bad_qacc"] > 0 for r in rows)),
              "islands": dict(collections.Counter(r["island"] for r in rows))}
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
