"""What a generation cost in a finished run, read from its telemetry.

    python experiments/perf/gen_cost.py runs/arch44/generations.jsonl [--skip 6]

Per generation: wall seconds (difference of consecutive ``elapsed``), the
designs it evaluated, Tier-0 rejections and promotions (differences of the
cumulative counters), and which periodic duties fell on it (Tier-2, audit,
checkpoint, migration -- periods from the run's own ``run_start`` config).
Tier-2 and the audit count *visits to the generation's island*, not
generations (``SearchState.island_visits``), and fire on the first visit;
checkpoint and migration count generations.

A run written after 2026-10-01 also carries ``cost`` on every generation line
(``GenerationCost``): seconds per phase, physics steps per evaluation call and
part, and the size of the designs that reached Tier 1. When present, the
medians of those are printed per window of generations.

Prints the distribution of wall seconds, the steady-state median of the
generations no periodic duty fell on, the extra median cost of each duty, and
a least-squares fit of wall seconds on the per-generation counts.
"""
import argparse
import json

import numpy as np


def load(path):
    cfg, gens = {}, []
    for line in open(path):
        d = json.loads(line)
        if d.get("kind") == "run_start":
            cfg = d.get("config", {})
        elif "generation" in d and "elapsed" in d:
            gens.append(d)
    return cfg, gens


def rows(cfg, gens):
    per_visit = {"tier2": cfg.get("tier2_every", 0), "audit": cfg.get("audit_every", 0)}
    per_gen = {"checkpoint": cfg.get("checkpoint_every", 0),
               "migrate": cfg.get("migrate_every", 0)}
    visits = {}
    seen = {}
    for d in gens:
        isl = d.get("island")
        seen[d["generation"]] = visits.get(isl, 0)
        visits[isl] = visits.get(isl, 0) + 1
    out = []
    for prev, cur in zip(gens, gens[1:]):
        g = cur["generation"]
        r = {"gen": g, "dt": cur["elapsed"] - prev["elapsed"], "island": cur.get("island"),
             "evaluated": cur.get("evaluated", 0) - prev.get("evaluated", 0),
             "tier0_rejected": cur.get("tier0_rejected", 0) - prev.get("tier0_rejected", 0),
             "promotions": cur.get("promotions", 0) - prev.get("promotions", 0),
             "rollouts": cur.get("rollouts", 0),
             "diverged": cur.get("diverged_rollouts", 0)}
        for k, every in per_visit.items():
            r[k] = int(bool(every) and seen[g] % every == 0)
        # The checkpoint written at generation g is paid between g's line and
        # g+1's, so it lands in g+1's difference.
        r["checkpoint"] = int(bool(per_gen["checkpoint"]) and (g - 1) % per_gen["checkpoint"] == 0)
        r["migrate"] = int(bool(per_gen["migrate"]) and g > 0 and g % per_gen["migrate"] == 0)
        r["cost"] = cur.get("cost")
        out.append(r)
    return out, list(per_visit) + list(per_gen)


def cost_windows(rs, width):
    """Median of each ``cost`` field per window of ``width`` generations."""
    have = [r for r in rs if r["cost"]]
    if not have:
        return
    print(f"\ncost fields, median per {width} generations:")
    lo = have[0]["gen"] // width * width
    while lo <= have[-1]["gen"]:
        w = [r["cost"] for r in have if lo <= r["gen"] < lo + width]
        lo += width
        if not w:
            continue
        flat = {}
        for c in w:
            for k, v in c.get("seconds", {}).items():
                flat.setdefault(f"s.{k}", []).append(v)
            for k, v in c.get("steps", {}).items():
                flat.setdefault(f"steps.{k}", []).append(v)
            for k, v in (c.get("designs") or {}).items():
                flat.setdefault(f"designs.{k}", []).append(v)
        print(f"  gen {lo - width}-{lo - 1} (n {len(w)}): " + ", ".join(
            f"{k} {np.median(v):.4g}" for k, v in sorted(flat.items())))


def q(x):
    x = np.asarray(x, float)
    return (f"n {len(x):4d}  median {np.median(x):7.1f}  p10 {np.percentile(x, 10):7.1f}  "
            f"p90 {np.percentile(x, 90):7.1f}  mean {x.mean():7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--skip", type=int, default=6,
                    help="drop generations below this (the verification burst)")
    ap.add_argument("--window", type=int, default=50,
                    help="generations per window for the cost fields")
    a = ap.parse_args()

    cfg, gens = load(a.path)
    allrows, duties = rows(cfg, gens)
    burst = [r for r in allrows if r["gen"] < a.skip]
    rs = [r for r in allrows if r["gen"] >= a.skip]
    print(f"{len(gens)} generations, last elapsed {gens[-1]['elapsed']:.0f} s "
          f"({gens[-1]['elapsed'] / 3600:.1f} h)")
    if burst:
        print(f"burst  gen 1..{a.skip - 1}: total {sum(r['dt'] for r in burst):.0f} s")
    print("all    " + q([r["dt"] for r in rs]))
    plain = [r for r in rs if not any(r[k] for k in duties)]
    print("plain  " + q([r["dt"] for r in plain]))
    base = float(np.median([r["dt"] for r in plain]))
    for k in duties:
        hit = [r["dt"] for r in rs if r[k]]
        if hit:
            print(f"{k:10s} n {len(hit):3d}  median {np.median(hit):7.1f}  "
                  f"extra over plain {np.median(hit) - base:+7.1f}")

    print("\nper-generation counts (steady state):")
    for k in ("evaluated", "tier0_rejected", "promotions", "rollouts", "diverged"):
        v = np.array([r[k] for r in rs], float)
        print(f"  {k:15s} median {np.median(v):6.1f}  min {v.min():5.0f}  max {v.max():5.0f}")

    by_island = {}
    for r in plain:
        by_island.setdefault(r["island"], []).append(r["dt"])
    print("\nplain generations by island:")
    for k, v in sorted(by_island.items()):
        print(f"  {str(k):14s} {q(v)}")

    cost_windows(rs, a.window)

    # `evaluated` is left out: the batch is fixed, so it is the intercept.
    names = ["promotions"] + duties
    X = np.array([[1.0] + [r[k] for k in names] for r in rs])
    y = np.array([r["dt"] for r in rs])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"\nleast squares, dt = c0 + sum c_k * count_k   (R^2 {r2:.3f}):")
    for k, c in zip(["intercept"] + names, coef):
        print(f"  {k:15s} {c:+8.2f} s")


if __name__ == "__main__":
    main()
