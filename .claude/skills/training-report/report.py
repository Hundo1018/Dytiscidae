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
    GEN = read(run / "generations.jsonl")
    G = [r for r in GEN if "generation" in r]
    EV = read(run / "events.jsonl")
    if not G:
        raise SystemExit(f"no generations in {run}")
    # The run's configuration is written to `generations.jsonl`, not to
    # `events.jsonl`.  Looking only in the latter found nothing and the header
    # rendered "batch null · null s segments" for every run ever reported.
    # The run's configuration is the header record of `generations.jsonl`, and
    # it carries no "generation" key, so it is not in `G`.  Looking for it only
    # in `events.jsonl` found nothing and every report ever produced rendered
    # "batch null · null s segments".
    start = next((e for e in GEN + EV if e.get("config")), {})
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

    out["events"] = event_marks(run, EV)
    out["comparable"] = comparable_series(E, G, REJ, n_gens)
    out["tree"] = lineage_tree(run)
    out["bodies"] = body_panel(run, E)
    return out


# --------------------------------------------------------------------------
# Events.  Every line chart in this report is drawn against generation, and
# three machines move those lines without the search doing anything: the
# descriptor refit merges cells every 400 *evaluations*, the auditor deletes
# designs, and the judge raises its bars.  A reader who does not know where
# those fired will read their effects as search behaviour -- which is exactly
# how arch35's `mission_corr` dip was first misread.  So they are marked.

MARK_KINDS = {
    "descriptor_refit": ("refit", "descriptor refit — cells merged, "
                                  "coverage and archive size step down"),
    "audit": ("invalidated", "auditor removed a design as model-dependent"),
    "judge_tighten": ("bars", "judge raised a competence bar"),
    "migrate": ("migration", "designs moved between islands"),
}


def event_marks(run: Path, EV: list) -> dict:
    """Generation numbers for each kind of thing that moves a line by itself."""
    out = {}
    for kind, (short, why) in MARK_KINDS.items():
        gens = sorted({e["gen"] for e in EV
                       if e.get("kind") == kind and e.get("gen") is not None
                       and (kind != "audit" or e.get("invalid"))})
        if gens:
            out[short] = {"gens": gens, "why": why, "n": len(gens)}
    ex = sorted({e["gen"] for e in read(run / "exploits.jsonl")
                 if e.get("gen") is not None})
    if ex:
        out["exploit"] = {"gens": ex, "n": len(ex),
                          "why": "a design was caught scoring for something "
                                 "the mission did not ask for"}
    # The refit cadence, stated in the units it actually uses.
    refits = out.get("refit", {}).get("gens") or []
    if len(refits) > 1:
        step = st.fmean([b - a for a, b in zip(refits, refits[1:])])
        out["refit"]["cadence"] = (
            f"every 400 evaluations — about {step:.0f} generations here, "
            f"not every 400 generations")
    return out


# --------------------------------------------------------------------------
# Baselines.  A number with nothing beside it cannot be read, so every series
# that *can* carry an earlier run's line gets one.  The ones that cannot are
# named, with the reason, rather than quietly omitted.
#
# Comparisons are computed from raw measurements against fixed thresholds, not
# from ladder rung indices: arch35 inserted `stirs` into the land ladder, so
# rung 3 means different things in the two runs and comparing the indices is a
# category error.  `land_speed >= 0.1` is the same question in both.

COMPARABLE = [
    ("moves", "fraction of evaluations reaching land_speed >= 0.1 m/s",
     lambda e: _lm(e, "land", "land_speed") is not None,
     lambda e: (_lm(e, "land", "land_speed") or 0) >= 0.1),
    ("upright", "fraction holding upright >= 0.7",
     lambda e: _lm(e, "land", "upright") is not None,
     lambda e: (_lm(e, "land", "upright") or 0) >= 0.7),
    ("submerges", "fraction reaching max_depth >= 0.5 m",
     lambda e: e.get("max_depth") is not None,
     lambda e: (e.get("max_depth") or 0) >= 0.5),
    ("energy_feasible", "fraction with energy_margin >= 0",
     lambda e: e.get("energy_margin") is not None,
     lambda e: (e.get("energy_margin") or -1) >= 0),
]

#: Named, with the reason, instead of being drawn against a baseline that would
#: make the comparison look meaningful.  CLAUDE.md: say "not comparable" rather
#: than shrinking the difference.
NOT_COMPARABLE = [
    ("mission_fraction", "arch35 multiplies it by max(takeoff_fraction, 0.05); "
                         "the term was redefined at this boundary"),
    ("fitness", "carries the mission term, so it moved with mission_fraction"),
    ("air scores", "redefined at arch33→arch34 and again at arch34→arch35"),
    ("takeoff_height", "did not exist before arch35 — a baseline would be a "
                       "line at zero, which reads as a measured zero"),
    ("land_peak_speed", "did not exist before arch35, same reason"),
    ("land ladder rungs", "arch35 inserted `stirs`, so rung N is a different "
                          "rung in the two runs; the metrics above are the "
                          "comparable form of the same question"),
]


def _lm(e, domain, key):
    v = ((e.get("ladder_measurements") or {}).get(domain) or {}).get(key)
    return v if isinstance(v, (int, float)) else None


def comparable_series(E, G, REJ, n_gens, bins=18):
    """Per-band rates that mean the same thing across runs."""
    span = max(n_gens, 1) / bins
    out = {}
    for name, label, has, hit in COMPARABLE:
        tot, got = collections.Counter(), collections.Counter()
        for e in E:
            g = e.get("gen")
            if g is None or not has(e):
                continue
            i = min(int(g // span), bins - 1)
            tot[i] += 1
            got[i] += 1 if hit(e) else 0
        if sum(tot.values()) > 50:
            out[name] = {"label": label,
                         "pts": [{"g": round((i + .5) * span),
                                  "v": 100 * got[i] / tot[i]}
                                 for i in sorted(tot) if tot[i] >= 20]}
    d = [b["elapsed"] - a["elapsed"] for a, b in zip(G, G[1:])]
    d = [x for x in d if 0 < x < 600]
    out["_scalars"] = {
        "sec_per_gen": round(st.median(d)) if d else 0,
        "tier0_reject_rate": round(100 * len(REJ) / max(len(E) + len(REJ), 1), 1),
        "n_parts": round(st.fmean([e["n_parts"] for e in E
                                   if e.get("n_parts")]), 2) if E else 0,
        "evaluations": len(E),
        "generations": n_gens,
    }
    out["_not_comparable"] = NOT_COMPARABLE
    return out


def load_comparable(run: Path):
    """The same rates for a baseline run, or None if it is not there."""
    try:
        G = [r for r in read(run / "generations.jsonl") if "generation" in r]
        EV = read(run / "events.jsonl")
        if not G:
            return None
        E = [e for e in EV if e.get("kind") == "evaluate"]
        REJ = [e for e in EV if e.get("kind") == "tier0_reject"]
        n = max(r["generation"] for r in G) + 1
        out = comparable_series(E, G, REJ, n)
        out["_name"] = run.name
        return out
    except Exception:
        return None


# --------------------------------------------------------------------------
# Lineage.  The scout keeps the whole parent->child graph and it is pickled
# with the search state, so the family tree does not need new telemetry.

def lineage_tree(run: Path, keep: int = 160) -> dict | None:
    """Who descended from whom, and whether the line paid off."""
    # Unpickling the scout needs the `dytiscidae` package on the path, and when
    # this file is run as a script `sys.path[0]` is the skill directory, not the
    # repo -- so the import fails, the except swallows it, and the section
    # silently disappears.  Put the repo root on the path first.
    try:
        import pickle
        import sys
        root = str(Path(__file__).resolve().parents[3])
        if root not in sys.path:
            sys.path.insert(0, root)
        with open(run / "search_state.pkl", "rb") as fh:
            state = pickle.load(fh)
        nodes = getattr(state.get("scout"), "nodes", None)
        if not nodes:
            return None
    except Exception as exc:                                   # noqa: BLE001
        print(f"  no lineage section: {type(exc).__name__}: {exc}")
        return None

    def f(n, k, default=None):
        return getattr(n, k, default)

    kids = collections.defaultdict(list)
    for nid, n in nodes.items():
        if f(n, "parent_id"):
            kids[f(n, "parent_id")].append(nid)

    depths = collections.Counter(f(n, "depth", 0) for n in nodes.values())
    # Did a line improve on the design it started from?  `best_descendant` is
    # the best fitness anywhere below a node; comparing it to the node's own
    # fitness is the honest form of "did continuing here pay".
    paid, tried = collections.Counter(), collections.Counter()
    for n in nodes.values():
        d = f(n, "depth", 0)
        tried[d] += 1
        if (f(n, "best_descendant") or 0) > (f(n, "fitness") or 0) + 1e-9:
            paid[d] += 1

    # The whole family the winner came from -- not just its ancestors.  A spine
    # plus siblings shows the path and hides the competition; what the structure
    # is actually made of is one root's entire descent, most of which died.  So:
    # walk down from the winner's root, best-first, until `keep` nodes.
    best_id = max(nodes, key=lambda k: f(nodes[k], "fitness") or 0)
    spine, cur = [], best_id
    while cur:
        spine.append(cur)
        cur = f(nodes[cur], "parent_id")
    spine = list(reversed(spine))
    spine_set = set(spine)

    root = spine[0]
    picked, queue = [], [root]
    while queue and len(picked) < keep:
        nid = queue.pop(0)
        picked.append(nid)
        # Best-first, but the winner's line is never dropped for want of room.
        queue.extend(sorted(kids.get(nid, []),
                            key=lambda k: (k not in spine_set,
                                           -(f(nodes[k], "best_descendant") or 0))))

    def row(nid):
        n = nodes[nid]
        return {"id": nid, "p": f(n, "parent_id"),
                "g": f(n, "generation", 0), "d": f(n, "depth", 0),
                "fit": round(f(n, "fitness") or 0, 4),
                "best": round(f(n, "best_descendant") or 0, 4),
                "isl": f(n, "island", ""),
                "kids": len(kids.get(nid, [])),
                "spine": nid in spine_set}

    roots = [nid for nid, n in nodes.items() if not f(n, "parent_id")]
    top = sorted(roots, key=lambda k: -(f(nodes[k], "best_descendant") or 0))[:10]
    return {
        "n_nodes": len(nodes), "n_roots": len(roots),
        "max_depth": max(depths) if depths else 0,
        "depth_hist": [{"k": str(d), "v": depths[d]} for d in sorted(depths)],
        "paid_by_depth": [{"k": str(d), "v": round(100 * paid[d] / tried[d], 1)}
                          for d in sorted(tried) if tried[d] >= 20],
        "branching": round(st.fmean([len(v) for v in kids.values()]), 2) if kids else 0,
        "dead_ends": sum(1 for nid in nodes if not kids.get(nid)),
        "subtree": [row(nid) for nid in picked],
        "best_id": best_id,
        "top_roots": [{"id": r, "isl": f(nodes[r], "island", ""),
                       "best": round(f(nodes[r], "best_descendant") or 0, 4)}
                      for r in top],
    }


# --------------------------------------------------------------------------
# Bodies.  A distribution of scores with no machines beside it says nothing
# about what a score buys.  `bodies.py` renders a still for each percentile
# and drops it in <run>/media/thumbs; this embeds whatever is there.

def body_panel(run: Path, E: list) -> dict | None:
    import base64
    manifest = run / "media" / "thumbs" / "manifest.json"
    if not manifest.exists():
        return None
    try:
        rows = json.loads(manifest.read_text())
    except Exception:
        return None
    for r in rows:
        p = run / "media" / "thumbs" / r.get("file", "")
        if p.exists() and p.stat().st_size < 900_000:
            r["img"] = ("data:image/png;base64,"
                        + base64.b64encode(p.read_bytes()).decode())
    return {"rows": [r for r in rows if r.get("img")]}


def _previous_run(run: Path):
    """The newest sibling run that has telemetry, for use as a default baseline.

    Picked by mtime rather than by name: run names are not ordered (arch34_4w,
    arch34_aborted, gputest2), and the newest finished run is nearly always the
    one a reader wants the new numbers held against.
    """
    parent = run.resolve().parent
    cands = [p for p in parent.glob("*/generations.jsonl")
             if p.parent.resolve() != run.resolve()]
    cands = [p for p in cands if p.stat().st_size > 20_000]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default=None,
                    help="an earlier run to draw as a baseline on every series "
                         "that means the same thing in both (default: the most "
                         "recent other run with telemetry, if there is one)")
    ap.add_argument("--no-baseline", action="store_true")
    a = ap.parse_args()
    run = Path(a.run)
    data = build(run)

    base = None
    if not a.no_baseline:
        cand = Path(a.baseline) if a.baseline else _previous_run(run)
        if cand is not None and cand.resolve() != run.resolve():
            base = load_comparable(cand)
            if base is None and a.baseline:
                print(f"  baseline {cand} has no usable telemetry — drawing none")
    data["baseline"] = base
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
