"""Aggregate a run's telemetry into the report page's data payload.

Reads `<run>/generations.jsonl` and `<run>/events.jsonl`, writes
`<run>/report.html` from `template.html`. Every section degrades to absent
rather than to a guess: a run with no shared policy has no PPO events and its
PPO charts are dropped from the page rather than drawn empty, and the "not in
this report" section says so.

Usage:  python report.py <run-dir> [--title "..."]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path

BINS = 45


def read(path: Path) -> list:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def corr(pairs) -> float:
    if len(pairs) < 3:
        return 0.0
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx, my = st.fmean(xs), st.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in pairs)
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else 0.0


def binned(items, key, n_gens, n=BINS, gk="gen"):
    span = max(n_gens, 1) / n
    buckets = collections.defaultdict(list)
    for e in items:
        g = e.get(gk)
        if g is None:
            continue
        v = key(e) if callable(key) else e.get(key)
        if v is None:
            continue
        buckets[min(int(g // span), n - 1)].append(v)
    return [{"g": round((i + 0.5) * span), "v": st.fmean(buckets[i]), "n": len(buckets[i])}
            for i in sorted(buckets) if buckets[i]]


def build(run: Path) -> dict:
    G = [r for r in read(run / "generations.jsonl") if "generation" in r]
    EV = read(run / "events.jsonl")
    if not G:
        raise SystemExit(f"no generations in {run}")
    start = next((e for e in EV if e.get("kind") == "run_start"), {})
    cfg = start.get("config", {})
    E = [e for e in EV if e.get("kind") == "evaluate"]
    P = [e for e in EV if e.get("kind") == "promote"]
    PPO = [e for e in EV if e.get("kind") == "ppo"]
    REJ = [e for e in EV if e.get("kind") == "tier0_reject"]
    n_gens = max(r["generation"] for r in G) + 1
    d = [b["elapsed"] - a["elapsed"] for a, b in zip(G, G[1:])]

    out = {"meta": {
        "generations": n_gens, "evaluations": len(E), "promotions": len(P),
        "hours": round(G[-1]["elapsed"] / 3600, 1), "ppo_updates": len(PPO),
        "seed": cfg.get("seed"), "batch": cfg.get("batch"),
        "segment_seconds": cfg.get("segment_seconds"),
        "sec_per_gen": round(st.median(d)) if d else 0,
        "islands": len(cfg.get("islands") or []) or 6,
        "has_ppo": bool(PPO)}}

    out["train_return"] = {"mission": binned(E, "mission_fraction", n_gens),
                           "fitness": binned(E, "fitness", n_gens)}

    T = [(p["tier1_fraction"], p["tier2_fraction"]) for p in P
         if p.get("tier1_fraction") is not None and p.get("tier2_fraction") is not None]
    out["eval_return"] = {
        "tier1": binned([p for p in P if p.get("tier1_fraction") is not None],
                        "tier1_fraction", n_gens, 30),
        "tier2": binned([p for p in P if p.get("tier2_fraction") is not None],
                        "tier2_fraction", n_gens, 30),
        "corr_all": round(corr(T), 3),
        "scatter": [[round(a, 4), round(b, 4)] for a, b in T]}

    span = max(n_gens, 1) / BINS
    tot, feas, gate = collections.Counter(), collections.Counter(), collections.Counter()
    for e in E:
        g = e.get("gen")
        if g is None:
            continue
        i = min(int(g // span), BINS - 1)
        tot[i] += 1
        if (e.get("energy_margin") if e.get("energy_margin") is not None else -1) >= 0:
            feas[i] += 1
        if e.get("air_gates"):
            gate[i] += 1
    out["success"] = {
        "energy": [{"g": round((i + .5) * span), "v": 100 * feas[i] / tot[i]} for i in sorted(tot)],
        "gated": [{"g": round((i + .5) * span), "v": 100 * gate[i] / tot[i]} for i in sorted(tot)],
        "diverged": [{"g": r["generation"],
                      "v": 100 * r.get("diverged_rollouts", 0) / max(r.get("rollouts", 1), 1)}
                     for r in G]}

    R = [(p["gen"], p["tier1_5_short"], p["tier1_5_retention"]) for p in P
         if p.get("tier1_5_retention") is not None and p.get("tier1_5_short")]
    if R:
        r = [x for _, _, x in R]
        med = st.median([s for _, s, _ in R])
        edges = [(0, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1), (1, 1.5), (1.5, 2), (2, 1e9)]
        out["retention"] = {
            "n": len(R), "median": round(st.median(r), 3), "mean": round(st.fmean(r), 3),
            "hist": [sum(1 for x in r if lo <= x < hi) for lo, hi in edges],
            "hi_med": round(st.median([x for _, s, x in R if s >= med]), 3),
            "lo_med": round(st.median([x for _, s, x in R if s < med]), 3),
            "hi_n": sum(1 for _, s, _ in R if s >= med),
            "lo_n": sum(1 for _, s, _ in R if s < med)}
    else:
        out["retention"] = None

    if PPO:
        keys = ("pi_loss", "v_loss", "entropy", "kl", "clipfrac", "lr")
        step = max(1, len(PPO) // 300)
        out["ppo"] = {k: [{"g": e["gen"], "v": e[k]} for e in PPO if e.get(k) is not None][::step]
                      for k in keys}
        tags = collections.defaultdict(list)
        for e in PPO[-60:]:
            for tag, v in (e.get("reward_by_tag") or {}).items():
                tags[tag].append(v[0] if isinstance(v, (list, tuple)) else v)
        out["ppo"]["reward_by_tag"] = {k: round(st.fmean(v), 4) for k, v in tags.items()}
    else:
        out["ppo"] = None

    mf = [e["mission_fraction"] for e in E if e.get("mission_fraction") is not None]
    edges = [(0, .005), (.005, .01), (.01, .02), (.02, .04), (.04, .08), (.08, .15),
             (.15, .25), (.25, 1e9)]
    out["dist"] = {"mission_hist": [sum(1 for x in mf if lo <= x < hi) for lo, hi in edges],
                   "mission_max": round(max(mf), 4) if mf else 0,
                   "mission_median": round(st.median(mf), 5) if mf else 0}
    out["failures"] = {
        "gates": collections.Counter(x.split(":")[0][:40] for e in E
                                     for x in (e.get("air_gates") or [])).most_common(4),
        "tier0_rejects": len(REJ), "evaluations": len(E),
        "energy_infeasible": sum(1 for e in E if (e.get("energy_margin") or 0) < 0)}

    refits = [e for e in EV if e.get("kind") == "descriptor_refit"]
    out["archive"] = {
        "coverage": [{"g": r["generation"], "v": 100 * r["coverage"]} for r in G],
        "qd": [{"g": r["generation"], "v": r["qd_score"]} for r in G],
        "refits": [e["gen"] for e in refits],
        "refit_before": [e.get("before") for e in refits if e.get("before") is not None],
        "islands": dict(collections.Counter(p.get("island") for p in P))}

    byp = collections.defaultdict(lambda: [0, 0])
    for e in E:
        p = e.get("n_parts")
        if p and e.get("energy_margin") is not None:
            k = min(p, 10)
            byp[k][0] += 1 if e["energy_margin"] < 0 else 0
            byp[k][1] += 1
    out["parts"] = {
        "infeasible": [{"p": k, "v": 100 * a / b, "n": b} for k, (a, b) in sorted(byp.items())
                       if b > 50],
        "trend": binned(E, "n_parts", n_gens)}
    out["lineage"] = [{"g": r["generation"], "v": r.get("scout", {}).get("depth_mean", 0)}
                      for r in G]
    out["corr_series"] = [{"g": r["generation"], "v": r.get("mission_corr", 0)} for r in G]
    best = max(E, key=lambda e: e.get("mission_fraction") or 0) if E else {}
    out["best"] = {k: best.get(k) for k in
                   ("gen", "island", "body_plan", "n_parts", "dof", "mass", "span",
                    "wing_area", "wing_loading", "air", "water", "land",
                    "mission_fraction", "energy_margin")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    data = build(run)
    title = a.title or f"{run.name} — training report"
    m = data["meta"]
    sub = (f"seed {m['seed']} · batch {m['batch']} · {m['segment_seconds']} s segments"
           if m["seed"] is not None else "configuration not recorded")
    tpl = (Path(__file__).parent / "template.html").read_text()
    html = (tpl.replace("__PAYLOAD__", json.dumps(data))
               .replace("__TITLE__", title)
               .replace("__SUBTITLE__", sub))
    out = Path(a.out) if a.out else run / "report.html"
    out.write_text(html)
    print(f"{out}  ({out.stat().st_size // 1024} KB)")
    print(f"  {m['generations']} generations, {m['evaluations']} evaluations, "
          f"{m['hours']} h, {m['sec_per_gen']} s/gen")
    if not m["has_ppo"]:
        print("  no PPO events: the learner sections will be empty for this run")
    if data["retention"] is None:
        print("  no tier1_5 retentions: section 4 will be empty for this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
