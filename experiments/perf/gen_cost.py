"""What a generation cost in a finished run, read from its telemetry.

    python experiments/perf/gen_cost.py runs/arch44/generations.jsonl [--skip 6]

Per generation: wall seconds (difference of consecutive ``elapsed``), the
designs it evaluated, Tier-0 rejections and promotions (differences of the
cumulative counters), and which periodic duties fell on it (Tier-2, audit,
checkpoint, migration -- periods from the run's own ``run_start`` config).

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
    duties = {"tier2": cfg.get("tier2_every", 0), "audit": cfg.get("audit_every", 0),
              "checkpoint": cfg.get("checkpoint_every", 0),
              "migrate": cfg.get("migrate_every", 0)}
    out = []
    for prev, cur in zip(gens, gens[1:]):
        g = cur["generation"]
        r = {"gen": g, "dt": cur["elapsed"] - prev["elapsed"], "island": cur.get("island"),
             "evaluated": cur.get("evaluated", 0) - prev.get("evaluated", 0),
             "tier0_rejected": cur.get("tier0_rejected", 0) - prev.get("tier0_rejected", 0),
             "promotions": cur.get("promotions", 0) - prev.get("promotions", 0),
             "rollouts": cur.get("rollouts", 0),
             "diverged": cur.get("diverged_rollouts", 0)}
        for k, every in duties.items():
            r[k] = int(bool(every) and g > 0 and g % every == 0)
        out.append(r)
    return out, list(duties)


def q(x):
    x = np.asarray(x, float)
    return (f"n {len(x):4d}  median {np.median(x):7.1f}  p10 {np.percentile(x, 10):7.1f}  "
            f"p90 {np.percentile(x, 90):7.1f}  mean {x.mean():7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--skip", type=int, default=6,
                    help="drop generations below this (the verification burst)")
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

    names = ["evaluated", "promotions"] + duties
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
