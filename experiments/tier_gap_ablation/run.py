#!/usr/bin/env python3
"""Which difference between a Tier-1 segment and a Tier-2 leg loses the credit?

arch48 (experiments/tier_gap): 35 of 36 promotions with Tier-1 water >= 0.15 scored
< 0.15 on the Tier-2 water leg; Spearman(Tier-1, Tier-2) is -0.09 (water), 0.02 (land).
Either Tier-1 pays for something Tier-2 shows is not there, or Tier-2 measures
something else.  This walks from "Tier-1 as the search scored it" to "the Tier-2
leg", one difference at a time, per medium.

Rungs (cumulative, in this order; each row of the table adds one thing):

  R0   the search's path (batchroll.evaluate_tier1_batch's per-domain loop), the elite's
       own stored policy + mobility basis, the network that scored its generation,
       its recorded ``eval_seed``, 8 s, the generation's task draw, ``scatter``,
       no disturbances.  Compared against the competence the archive recorded.
  R0s  = R0 at fresh seeds (three).  Everything a later rung does is at fresh
       seeds, so this is the baseline for them, and its gap to R0 is the seed
       noise of a Tier-1 score (and the winner's curse of the archive keeping the
       best draw).
  R1   offline path: ``TriphibianEnv.rollout`` one machine at a time (what Tier-2 uses),
       with SummedPolicy.
  R2   + leg length 25 s  (``spec.seconds_per_domain / time_compression`` = 300 / 12)
  R3   + the default task schedule (``env.task`` unset: ``_arm_task`` -> ``schedule_for(domain)``)
  R4   + no ``env.scatter``
  R5   + Tier-2's disturbances: SeaState, current, wind drawn exactly as ``evaluate_tier2``
  R6   + the controller Tier-2 gets: the run's final network instead of the one that
       scored the generation (``_with_shared(state, ctrl2)`` is the network at promotion
       time), and the promoted policy (``policy_promoted``) where the elite has one.
  R7   the real ``evaluate_tier2(label_all_media=True)`` with the R6 controller, as
       a check that R6 *is* the verifier.

What was actually run (names as in ``ALL``; ``stageB.sh``..``stageE.sh`` are the launch record):

  R0, R0s, R1                 n = 16 passes + 16 controls per medium
  everything else             a fixed random 10 + 10 per medium (``--sub 10``)
  R2b..R6b                    R2..R6 on the *batched* path ("b"): R1 vs R0s showed the path
                              does not matter (rho 0.98), and the offline wall (7 s per 25 s leg)
                              did not fit.  R7 (real ``evaluate_tier2``, offline) is the check.
  S_task S_scatter S_env      R0 with one part of the seed fresh (the other two the generation's)
  T_heading T_order T_depth   R0 with one part of the task draw replaced by the default's
  A_*                         each difference alone from R0s (fresh seeds)
  B_*, R0p                    each difference alone from R0 (the generation's own seed kept);
                              ``R0p`` = R0 through the ``prime`` path, to show it reproduces R0
  R0o                         R0 through the offline path at the same seed
  H0..H7                      R0 with the cruise heading set to eight fixed values
  X:still                     the same rung with ``env.held_still_params()``, no policy

Every rung that has randomness runs at SEEDS fresh seeds; the table uses the median
over seeds per elite.  Run under a memory cap:

    systemd-run --user --scope -q -p MemoryMax=3500M env MUJOCO_GL=disable PYTHONPATH=. \
        .venv/bin/python experiments/tier_gap_ablation/run.py --run runs/arch48 ...
"""
import argparse
import collections
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments/shared_policy_value")
sys.path.insert(0, "experiments")

MEDIA = ("air", "water", "land")
BAR = 0.15
FRESH = (910011, 910022, 910033)


@dataclass(frozen=True)
class Cfg:
    path: str = "batch"        # batch | offline
    seconds: float = 8.0
    task: str = "gen"          # gen | default
    scatter: bool = True
    dist: bool = False
    ctrl: str = "scored"       # scored | final
    seeds: str = "fresh"       # eval | fresh
    #: with ``seeds="eval"``: the one draw that is fresh (task | scatter | env), the
    #: other two stay the generation's.  "" = all of them move together.
    part: str = ""
    #: with ``task="gen"``: one component of the task takes the *default* schedule's
    #: value (heading | order | depth), the other two stay the generation's draw.
    task_part: str = ""
    #: eval seed only: run just the requested medium, but ``reset`` (and scatter) the domains
    #: before it in the cycle so the env's one reset-noise stream is where the full cycle
    #: leaves it (``rng`` is read only in ``reset``).  Checked against R0 by ``R0p``.
    prime: bool = False
    #: cruise heading in radians, replacing the task's (NaN = the task's own).  The eval seed is kept.
    heading: float = float("nan")


R0 = Cfg(seeds="eval")
R0S = Cfg()
RUNGS = {
    "R0": R0,
    "R0s": R0S,
    "R1": replace(R0S, path="offline"),
}
RUNGS["R2"] = replace(RUNGS["R1"], seconds=25.0)
RUNGS["R3"] = replace(RUNGS["R2"], task="default")
RUNGS["R4"] = replace(RUNGS["R3"], scatter=False)
RUNGS["R5"] = replace(RUNGS["R4"], dist=True)
RUNGS["R6"] = replace(RUNGS["R5"], ctrl="final")
ALONE = {
    "A_leg": replace(R0S, seconds=25.0),
    "A_task": replace(R0S, task="default"),
    "A_scatter": replace(R0S, scatter=False),
    "A_dist": replace(R0S, dist=True),
    "A_ctrl": replace(R0S, ctrl="final"),
}
#: What a changed seed is made of: the task draw (heading, hold/cruise order, depth),
#: the scatter draw (attitude, velocity), the env's reset noise.  Each moved alone
#: from R0.
SEEDPARTS = {
    "S_task": replace(R0, part="task"),
    "S_scatter": replace(R0, part="scatter"),
    "S_env": replace(R0, part="env"),
}
#: The same cumulative ladder on the batched path ("b"), for the rungs where the
#: path is known not to matter (R1 vs R0s) and the offline wall (7 s per 25 s leg) does not fit.
RUNGS_B = {f"{k}b": replace(RUNGS[k], path="batch") for k in ("R2", "R3", "R4", "R5", "R6")}
#: R0 through the offline path at the same seed: the path alone, same draw.
#: What a task draw is made of: heading, hold/cruise order, hold depth.  Each set to the
#: default schedule's value alone, from R0 (the generation's own seed otherwise).
TASKPARTS = {f"T_{c}": replace(R0, task_part=c) for c in ("heading", "order", "depth")}
#: Each difference ALONE from R0 -- the generation's own seed kept, so the draw cannot be the
#: thing that moved (``A_*`` above are alone from R0s, at fresh seeds).
R0P = replace(R0, prime=True)
ALONE_R0 = {
    "R0p": R0P,
    "B_leg": replace(R0P, seconds=25.0),
    "B_task": replace(R0P, task="default"),
    "B_scatter": replace(R0P, scatter=False),
    "B_dist": replace(R0P, dist=True),
    "B_ctrl": replace(R0P, ctrl="final"),
}
#: A heading sweep at the generation's own seed: eight evenly spaced cruise headings, the hold/
#: cruise order, depth, scatter and reset noise all the draw's own.
HEADINGS = {f"H{k}": replace(R0P, heading=-np.pi + k * np.pi / 4.0) for k in range(8)}
ALL = {**RUNGS, **ALONE, **SEEDPARTS, **TASKPARTS, **RUNGS_B, **ALONE_R0, **HEADINGS,
       "R0o": replace(R0, path="offline")}


@dataclass
class Item:
    idx: int                   # position in the pool
    elite: object
    pheno: object
    gen: int
    eval_seed: int
    own: object                # the elite's own Policy (or None)
    promoted: object           # policy_promoted as a Policy, or None
    bases: dict
    params: object
    recorded: dict
    promoted_differs: bool = False


# ------------------------------------------------------------------ pool

def load_pool(run):
    """Every island archive, de-duplicated (seeding files one design on every island)."""
    import glob
    from dytiscidae.evolution.archive import Archive
    seen, pool = set(), []
    for f in sorted(glob.glob(str(Path(run) / "archive_*.pkl"))):
        a = Archive.load(f)
        for e in a.cells.values():
            m = e.meta or {}
            w = m.get("policy")
            key = (m.get("gen"), m.get("eval_seed"), round(float(m.get("mass") or 0), 6),
                   tuple(np.round(np.asarray(w, float)[:6], 6)) if w is not None else None)
            if key in seen:
                continue
            seen.add(key)
            e.meta = dict(m, _island_file=Path(f).stem)
            pool.append(e)
    return pool


def choose(pool, medium, n_max, seed=0):
    rng = np.random.default_rng(seed)
    pos = [i for i, e in enumerate(pool) if float((e.meta or {}).get(medium) or 0.0) >= BAR]
    neg = [i for i, e in enumerate(pool) if float((e.meta or {}).get(medium) or 0.0) < BAR]
    if n_max and len(pos) > n_max:
        pos = sorted(rng.choice(pos, n_max, replace=False).tolist())
    k = min(len(pos), len(neg))
    neg = sorted(rng.choice(neg, k, replace=False).tolist())
    return pos, neg


def build_items(run, pool, picks):
    from dytiscidae.core.phenotype import build
    from dytiscidae.control.cpg import Policy
    from rescore import plain_controller
    items = []
    for i in picks:
        e = pool[i]
        m = e.meta or {}
        p = build(e.genome)
        seed = int(m.get("eval_seed") or 0)
        c, final = plain_controller(run, e, p, seed)
        if c is None:
            continue
        prom, differs = None, False
        w = m.get("policy_promoted")
        if w is not None and c.policy is not None:
            prom = Policy(n_obs=c.policy.n_obs, n_modes=c.policy.n_modes,
                          hidden=c.policy.hidden, gain=bool(getattr(c.policy, "gain", False)))
            prom.weights = np.asarray(w, float).copy()
            differs = not np.array_equal(prom.weights, c.policy.weights)
        items.append(Item(i, e, p, int(m.get("gen") or 0), seed, c.policy, prom,
                          c.bases, c.params,
                          {k: m.get(k) for k in MEDIA}, differs))
    return items, final


# ------------------------------------------------------------------ one rung

def disturbances(seed):
    """Exactly the draws ``evaluate_tier2`` makes from ``default_rng(seed)``, in its order."""
    from dytiscidae.physics.medium import SeaState
    rng = np.random.default_rng(seed)
    sea = SeaState(amplitude=float(rng.uniform(0.0, 0.25)),
                   period=float(rng.uniform(1.6, 3.2)))
    cur = rng.normal(0, 0.15, 3) * np.array([1, 1, 0.2])
    wind = rng.normal(0, 1.2, 3) * np.array([1, 1, 0.3])
    return dict(sea_state=sea, current=cur, wind=wind)


def make_task(cfg, dom, s_tk):
    """The task one segment is asked: the generation's draw, or the default schedule,
    or the draw with one component replaced by the default's (``cfg.task_part``)."""
    from dataclasses import replace as _rep
    from dytiscidae.envs import tasks as T
    from dytiscidae.envs.evaluate import _scatter_seed
    if cfg.task == "default":
        return None                         # env.reset cleared it: _arm_task -> schedule_for(dom)
    t = T.schedule_for(dom, np.random.default_rng(T.task_seed(_scatter_seed(s_tk, dom))))
    if cfg.heading == cfg.heading:          # not NaN: a swept heading, everything else the draw's
        t = T.TaskSchedule(t.domain, tuple(_rep(p, heading=float(cfg.heading)) if p.kind == T.CRUISE else p
                                              for p in t.phases))
    if not cfg.task_part:
        return t
    d = T.schedule_for(dom)
    ph = {p.kind: p for p in t.phases}
    dph = {p.kind: p for p in d.phases}
    if cfg.task_part == "heading":
        ph = {k: _rep(v, heading=dph[k].heading) for k, v in ph.items()}
    elif cfg.task_part == "depth":
        ph = {k: _rep(v, depth=dph[k].depth) for k, v in ph.items()}
    order = [p.kind for p in (d.phases if cfg.task_part == "order" else t.phases)]
    return T.TaskSchedule(t.domain, tuple(T._at(ph[k], 0.0 if i == 0 else 0.5)
                                          for i, k in enumerate(order)))


def _key(d):
    """A draw triple as the seed label stored with a value: the fresh one if only one moved."""
    return d[0] if d[0] == d[1] == d[2] else max(d, key=lambda x: (d.count(x) == 1, x))


class Nets:
    """Scoring networks by generation, and the final one."""

    def __init__(self, run, final):
        self.run, self.final, self._c = Path(run), final, {}

    def scored(self, item):
        from dytiscidae.viz.film import control_laws
        if item.gen not in self._c:
            desc, net = next(iter(control_laws(item.elite, self.run, self.final)))
            self._c[item.gen] = (net, "was not kept" not in desc)
        return self._c[item.gen]


def policy_of(item, cfg):
    if cfg.ctrl == "final" and item.promoted is not None:
        return item.promoted
    return item.own


def run_cfg(items, cfg, nets, media, seeds, still=False, domains_full=False, chunk=16, log=None):
    """{item.idx: {seed: {medium: competence}}} for one configuration."""
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.batchroll import BatchedFluid, rollout_batch
    from dytiscidae.envs.evaluate import Controller, SharedController, SummedPolicy, _scatter_seed
    from dytiscidae.envs.tasks import schedule_for, task_seed
    from dytiscidae.envs.triphibian import Domain, DOMAIN_CYCLE, TriphibianEnv

    out = collections.defaultdict(dict)
    last = max(DOMAIN_CYCLE.index(Domain(m)) for m in media)
    doms = [d for d in DOMAIN_CYCLE
            if (domains_full or d.value in media or (cfg.prime and DOMAIN_CYCLE.index(d) < last))]
    def draws(it):
        """[(env seed, scatter seed, task seed)] this item is run under."""
        if cfg.seeds == "fresh":
            return [(f, f, f) for f in seeds]
        if not cfg.part:
            return [(it.eval_seed,) * 3]
        e = it.eval_seed
        return [(f if cfg.part == "env" else e, f if cfg.part == "scatter" else e,
                 f if cfg.part == "task" else e) for f in seeds]

    def net_for(it):
        if still:
            return None
        return nets.final if cfg.ctrl == "final" else nets.scored(it)[0]

    if cfg.path == "batch":
        # one batch per (generation, seed): the scoring network is per generation
        groups = collections.defaultdict(list)
        for it in items:
            for d in draws(it):
                groups[(it.gen if cfg.ctrl == "scored" else 0, d)].append(it)
        for (g, d), its in sorted(groups.items()):
            s, s_sc, s_tk = d
            for lo in range(0, len(its), chunk):
                part = its[lo:lo + chunk]
                kw = disturbances(s) if cfg.dist else {}
                envs = [TriphibianEnv(it.pheno, seed=s, **kw) for it in part]
                for e in envs:
                    e.air_launch_height = None
                bf = BatchedFluid(envs)
                if still:
                    ctrls = [Controller(params=e.held_still_params(), policy=None, bases=it.bases)
                             for e, it in zip(envs, part)]
                else:
                    ctrls = [Controller(params=it.params, policy=policy_of(it, cfg), bases=it.bases)
                             for it in part]
                net = net_for(part[0])
                for dom in doms:
                    sc = _scatter_seed(s_sc, dom)
                    task = make_task(cfg, dom, s_tk)
                    for e in envs:
                        e.reset(dom)
                        if cfg.scatter:
                            e.scatter(np.random.default_rng(sc))
                        if task is not None:
                            e.task = task
                    if cfg.prime and not domains_full and dom.value not in media:
                        continue                       # reset only: keeps the rng where the cycle leaves it
                    bf.reset_slam()
                    segs = rollout_batch(envs, bf, cfg.seconds, [c.params for c in ctrls], dom,
                                         policies=[c.policy for c in ctrls],
                                         bases=[c.basis_for(dom) for c in ctrls], shared=net)
                    for slot, it in enumerate(part):
                        if dom.value in media:
                            out[it.idx].setdefault(_key(d), {})[dom.value] = float(segs[slot].competence)
                for e in envs:
                    pass
                if log:
                    log(f"    batch gen/{g} draws {d} n={len(part)}")
        return out

    # offline: one machine at a time, exactly the evaluate_tier2 construction
    for it in items:
        for d in draws(it):
            s, s_sc, s_tk = d
            kw = disturbances(s) if cfg.dist else {}
            env = TriphibianEnv(it.pheno, seed=s, **kw)
            env.air_launch_height = None
            if still:
                ctrl = Controller(params=env.held_still_params(), policy=None, bases=it.bases)
            else:
                net = net_for(it)
                pol = SummedPolicy(own=policy_of(it, cfg), shared=net, n_modes=6)
                ctrl = SharedController(params=it.params, bases=it.bases, policy=pol)
            for dom in doms:
                env.reset(dom)
                if cfg.scatter:
                    env.scatter(np.random.default_rng(_scatter_seed(s_sc, dom)))
                task = make_task(cfg, dom, s_tk)
                if task is not None:
                    env.task = task
                if cfg.prime and not domains_full and dom.value not in media:
                    continue
                seg = env.rollout(cfg.seconds, params=ctrl.params, policy=ctrl.policy,
                                  basis=ctrl.basis_for(dom), domain=dom)
                if dom.value in media:
                    out[it.idx].setdefault(_key(d), {})[dom.value] = float(seg.competence)
    return out


def run_tier2(items, nets, media, seeds, log=None):
    """The real verifier, with the R6 controller (final network, promoted policy)."""
    from dytiscidae.envs.evaluate import SharedController, SummedPolicy, evaluate_tier2
    from dytiscidae.envs.triphibian import MissionSpec
    out = collections.defaultdict(dict)
    for it in items:
        for s in seeds:
            pol = SummedPolicy(own=policy_of(it, RUNGS["R6"]), shared=nets.final, n_modes=6)
            ctrl = SharedController(params=None, bases=it.bases, policy=pol)
            r = evaluate_tier2(it.pheno, spec=MissionSpec(), controller=ctrl, seed=s,
                               label_all_media=True)
            seg = {**r.probe_segments, **r.segments}
            out[it.idx][s] = {m: float(seg[m].competence) for m in media if m in seg}
    return out


# ------------------------------------------------------------------ analysis

def _ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None            # constant: undefined, not zero
    return float(np.corrcoef(ra, rb)[0, 1])


def per_item(raw, rung, medium):
    """{idx: median over seeds} for a rung and medium."""
    out = {}
    for row in raw:
        if row["rung"] == rung and row["medium"] == medium:
            out.setdefault(row["idx"], []).append(row["v"])
    return {k: float(np.median(v)) for k, v in out.items()}


def summarise(raw, sel):
    """Per rung and medium: median, share >= 0.15, Spearman vs rung 0 (reproduced)."""
    rows = []
    rungs = []
    for r in raw:
        if r["rung"] not in rungs:
            rungs.append(r["rung"])
    for medium in ("water", "land"):
        base = per_item(raw, "R0", medium)
        rec = {i: sel["recorded"][str(i)][medium] for i in base}
        for rung in rungs:
            v = per_item(raw, rung, medium)
            ids = [i for i in sel["sets"][medium] if i in v and i in base]
            if not ids:
                continue
            pos = [i for i in ids if sel["recorded"][str(i)][medium] >= BAR]
            neg = [i for i in ids if sel["recorded"][str(i)][medium] < BAR]
            vv = lambda s: np.array([v[i] for i in s], float)               # noqa: E731
            row = {"rung": rung, "medium": medium, "n": len(ids), "n_pos": len(pos), "n_neg": len(neg),
                   "median_all": float(np.median(vv(ids))),
                   "median_pos": float(np.median(vv(pos))) if pos else None,
                   "share_pos": float(np.mean(vv(pos) >= BAR)) if pos else None,
                   "k_pos": int(np.sum(vv(pos) >= BAR)) if pos else None,
                   "share_neg": float(np.mean(vv(neg) >= BAR)) if neg else None,
                   "spearman_vs_R0": spearman(vv(ids), [base[i] for i in ids]),
                   "spearman_vs_recorded": spearman(vv(ids), [rec[i] for i in ids])}
            rows.append(row)
    return rows


def fmt_table(rows):
    def f(x, p=3):
        return "-" if x is None else f"{x:.{p}f}"
    lines = []
    for medium in ("water", "land"):
        lines.append(f"\n{medium}: n = pos+neg; share = fraction >= {BAR}; rho vs R0 over all n")
        lines.append("| rung | n | median pos | share pos | share neg | median all | rho vs R0 |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            if r["medium"] != medium:
                continue
            lines.append(f"| {r['rung']} | {r['n_pos']}+{r['n_neg']} | {f(r['median_pos'])} | "
                         f"{f(r['share_pos'], 2)} ({r['k_pos']}/{r['n_pos']}) | {f(r['share_neg'], 2)} | "
                         f"{f(r['median_all'])} | {f(r['spearman_vs_R0'], 2)} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/arch48")
    ap.add_argument("--out", default="experiments/tier_gap_ablation/results.json")
    ap.add_argument("--raw", default="experiments/tier_gap_ablation/raw.json")
    ap.add_argument("--n-max", type=int, default=10, help="max Tier-1 passes per medium (and as many controls)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rungs", default="R0,R0s,R1,R2,R3,R4,R5,R6")
    ap.add_argument("--still", default="", help="comma list of rungs to also run held still")
    ap.add_argument("--tier2", type=int, default=0, help="R7: elites per medium through evaluate_tier2")
    ap.add_argument("--limit", type=int, default=0, help="timing: only the first k items per set")
    ap.add_argument("--sub", type=int, default=0,
                    help="run this invocation's rungs on a fixed random k pos + k neg per medium "
                         "(the selection and the recorded table stay the full ones)")
    args = ap.parse_args()

    from dytiscidae.envs import batchroll
    ok, why = batchroll.usable()
    assert ok, f"batched evaluator unusable: {why}"

    run = Path(args.run)
    pool = load_pool(run)
    sets, picks = {}, {}
    for m in ("water", "land"):
        pos, neg = choose(pool, m, args.n_max)
        if args.limit:
            pos, neg = pos[:args.limit], neg[:args.limit]
        sets[m] = pos + neg
        picks[m] = (pos, neg)
    union = sorted(set(sets["water"]) | set(sets["land"]))
    print(f"pool {len(pool)}; water {len(picks['water'][0])}+{len(picks['water'][1])}, "
          f"land {len(picks['land'][0])}+{len(picks['land'][1])}, union {len(union)}", flush=True)
    items, final = build_items(run, pool, union)
    by_idx = {it.idx: it for it in items}
    nets = Nets(run, final)
    sel = {"sets": {m: [i for i in sets[m] if i in by_idx] for m in sets},
           "recorded": {str(i): {k: float(v or 0.0) for k, v in by_idx[i].recorded.items()} for i in by_idx},
           "gen": {str(i): by_idx[i].gen for i in by_idx},
           "promoted": {str(i): by_idx[i].promoted is not None for i in by_idx},
           "promoted_differs": {str(i): by_idx[i].promoted_differs for i in by_idx},
           "island_file": {str(i): by_idx[i].elite.meta.get("_island_file") for i in by_idx}}
    seeds = list(FRESH[:args.seeds])
    active = {m: list(sets[m]) for m in sets}
    if args.sub:
        rs = np.random.default_rng(1)
        for m in sets:
            pos, neg = picks[m]
            active[m] = (sorted(rs.choice(pos, min(args.sub, len(pos)), replace=False).tolist())
                         + sorted(rs.choice(neg, min(args.sub, len(neg)), replace=False).tolist()))
    raw_path = Path(args.raw)
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else []
    done = {r["rung"] for r in raw}
    walls = {}
    wall_path = raw_path.with_suffix(".walls.json")
    if wall_path.exists():
        walls = json.loads(wall_path.read_text())

    def add(rung, out, media):
        for idx, per_seed in out.items():
            for s, vals in per_seed.items():
                for m in media:
                    if m in vals:
                        raw.append({"rung": rung, "medium": m, "idx": int(idx), "seed": int(s), "v": vals[m]})

    def flush():
        raw_path.write_text(json.dumps(raw))
        wall_path.write_text(json.dumps(walls))
        rows = summarise(raw, sel) if any(r["rung"] == "R0" for r in raw) else []
        Path(args.out).write_text(json.dumps({"args": vars(args), "seeds": seeds, "walls_s": walls,
                                              "sel": sel, "rows": rows}, indent=1))
        return rows

    still_for = set(x for x in args.still.split(",") if x)
    for rung in [r for r in args.rungs.split(",") if r] + [f"{r}:still" for r in sorted(still_for)]:
        if rung in done:
            print(f"{rung}: already in {raw_path}, skipped", flush=True)
            continue
        base, still = (rung.split(":")[0], rung.endswith(":still"))
        cfg = ALL[base]
        t0 = time.time()
        if cfg.seeds == "eval" and not cfg.part and cfg.prime:
            for m in ("water", "land"):
                its = [by_idx[i] for i in active[m] if i in by_idx]
                out = run_cfg(its, cfg, nets, (m,), seeds, still=still, log=None)
                add(rung, out, (m,))
        elif cfg.seeds == "eval" and not cfg.part:
            # The generation's own seed: all three domains in the cycle's order, because the
            # env's reset noise is one stream that air consumes before water.
            ids = sorted(set(active["water"]) | set(active["land"])) if (base != "R0" or still) else union
            out = run_cfg([by_idx[i] for i in ids if i in by_idx], cfg, nets, MEDIA, seeds,
                          still=still, domains_full=True, log=None)
            add(rung, out, ("water", "land", "air"))
        else:
            for m in ("water", "land"):
                its = [by_idx[i] for i in active[m] if i in by_idx]
                out = run_cfg(its, cfg, nets, (m,), seeds, still=still, log=None)
                add(rung, out, (m,))
        walls[rung] = round(time.time() - t0, 1)
        rows = flush()
        print(f"{rung} done, wall {walls[rung]} s; total {sum(walls.values()):.0f} s", flush=True)
        for r in rows:
            if r["rung"] == rung:
                print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

    if args.tier2:
        t0 = time.time()
        out7 = {}
        for m in ("water", "land"):
            its = [by_idx[i] for i in active[m][:args.tier2] + active[m][-args.tier2:] if i in by_idx]
            o = run_tier2(its, nets, (m,), seeds[:1])
            add("R7", o, (m,))
        walls["R7"] = round(time.time() - t0, 1)
        flush()
        print(f"R7 done, wall {walls['R7']} s", flush=True)

    rows = flush()
    print(fmt_table(rows))
    r0 = [r for r in raw if r["rung"] == "R0"]
    err = {}
    for m in MEDIA:
        e = [abs(r["v"] - sel["recorded"][str(r["idx"])][m]) for r in r0 if r["medium"] == m]
        if e:
            err[m] = {"n": len(e), "median_abs": float(np.median(e)), "max_abs": float(np.max(e)),
                      "within_0.005": int(sum(x <= 0.005 for x in e))}
    print("R0 reproduction error vs recorded:", json.dumps(err))


if __name__ == "__main__":
    main()
