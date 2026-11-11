"""P0.2 on the evolved populations: what the 2026-11-11 crossing gates do to stored elites.

No learning.  Every elite of the final island archives of the named runs is
replayed through ``transitions.run_transition`` (single path, default entry
state, the elite's ``eval_seed``) for the four crossings, under three arms

  own    the control law the archive says scored it: its stored policy plus the
         run's shared network of its generation (``scoring_networks/genNNNNN.npz``,
         as ``viz.film.control_laws``), shared half at its mean (``SummedPolicy``)
  base   ``Controller(params=env.cpg.base, policy=None, bases=None)``
  still  ``Controller(params=env.held_still_params(), policy=None, bases=None)``

each with ``DYTISCIDAE_CROSSING_GATES`` 0 and 1 (set before the tracker is
constructed).  One JSON row per rollout.  A design filed on several islands with
the same genome, seed, generation and policy is replayed once; the row names
every (island, index) that holds it in ``members``.

    PYTHONPATH=. MUJOCO_GL=disable <main>/.venv/bin/python \
        experiments/triphibian_diagnosis/scripts/population_gates.py run \
        --runs-root <main>/runs --out experiments/triphibian_diagnosis/results/population_gates.jsonl
        --sample 60                                   # stratum A: results/population_gates.jsonl
    ... population_gates.py run --runs-root <main>/runs --arms own \
        --exclude <stratum A jsonl> --out results/population_gates_own_rest.jsonl   # stratum B
    ... population_gates.py qacc  --runs-root <main>/runs --rows <A> <B> --out <qacc jsonl>
    ... population_gates.py report --rows <A> --own-rest <B> --qacc <qacc jsonl> \
        --sizes results/population_gates_sizes.json \
        --notes scripts/population_gates_notes.md --out reports/P02_population_gates.md

As run on 2026-11-11: stratum A 31 min, stratum B 53 min, qacc 3 min, 4 processes.
"""
import argparse
import collections
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

RUNS = ("arch49", "arch51", "arch56")
KINDS = ("air_to_water", "water_to_air", "water_to_land", "land_to_air")
ARMS = ("own", "base", "still")
KEEP = ("crossed", "failure", "bad_qacc", "hold", "hold_drift", "end_dwell", "min_upright",
        "exit_upright", "approach", "started_in", "shore_progress", "go_peak_clearance",
        "aloft_fraction", "exit_depth")
SAMPLE_SEED = 20261111
PRE_COMMIT = "b652788"      # the parent of 7c59754: run_transition before the bad_qacc read


# --------------------------------------------------------------------------
# Units of work
# --------------------------------------------------------------------------


def stored_credit(meta):
    """``(completed, graded)``: whether the archive recorded a completed crossing
    (``crossed_fraction`` > 0) and whether it recorded any graded approach
    (the ladder's ``crossed`` component mean > 0)."""
    lm = ((meta or {}).get("ladder_measurements") or {}).get("transition") or {}
    return (bool(float(lm.get("crossed_fraction") or 0.0) > 0.0),
            bool(float(lm.get("crossed") or 0.0) > 0.0))


def units_of(runs_root, run):
    """``[unit]`` for one run: unique (genome, seed, gen, policy) over all islands."""
    from dytiscidae.ops.run import load_run_archive

    _, names = load_run_archive(Path(runs_root) / run)
    by = collections.OrderedDict()
    for island in names:
        arch, _ = (load_run_archive(Path(runs_root) / run, island) if island != "default"
                   else load_run_archive(Path(runs_root) / run))
        for idx, (cell, el) in enumerate(arch.cells.items()):
            m = el.meta or {}
            key = hashlib.sha1(pickle.dumps(
                (pickle.dumps(el.genome), m.get("eval_seed"), m.get("gen"),
                 tuple(m.get("policy") or ()), m.get("scored_with_shared_policy")))).hexdigest()[:16]
            u = by.setdefault(key, {"run": run, "unit": key, "island": island, "index": idx,
                                    "members": [], "credit": False, "graded": False,
                                    "stored_crossed_fraction": 0.0, "stored_approach": 0.0})
            u["members"].append([island, idx])
            done, graded = stored_credit(m)
            lm = (m.get("ladder_measurements") or {}).get("transition") or {}
            u["credit"], u["graded"] = u["credit"] or done, u["graded"] or graded
            u["stored_crossed_fraction"] = float(lm.get("crossed_fraction") or 0.0)
            u["stored_approach"] = float(lm.get("crossed") or 0.0)
    return list(by.values())


def load_elite(runs_root, unit):
    from dytiscidae.ops.run import load_run_archive

    run_dir = Path(runs_root) / unit["run"]
    isl = unit["island"]
    arch, _ = load_run_archive(run_dir, isl) if isl != "default" else load_run_archive(run_dir)
    return list(arch.cells.values())[unit["index"]]


# --------------------------------------------------------------------------
# The control law
# --------------------------------------------------------------------------

_NETS = {}


def own_controller(run_dir, elite, p, seed):
    """``(controller, description)``: the law the archive says scored the elite."""
    from dytiscidae.ops.run import controller_for_elite
    from dytiscidae.viz.film import _load_network

    meta = elite.meta or {}
    ctrl = controller_for_elite(str(run_dir), elite, p, seed, log=lambda *a, **k: None)
    if ctrl is None:
        return None, "no stored policy and no shared network recoverable"
    summed = ctrl.policy
    shared = getattr(summed, "shared", None)
    flag = meta.get("scored_with_shared_policy")
    gen = int(meta.get("gen") or 0)
    own = "own policy" if getattr(summed, "own", summed) is not None else "no own policy"
    if shared is None or not hasattr(summed, "shared"):
        return ctrl, f"{own} alone (no shared network recoverable)"
    if flag is True:
        path = Path(run_dir) / "scoring_networks" / f"gen{gen:05d}.npz"
        k = str(path)
        if k not in _NETS:
            _NETS[k] = _load_network(path, shared)
        if _NETS[k] is not None:
            summed.shared = _NETS[k]
            return ctrl, f"{own} + shared network of gen {gen} (scoring_networks), mean action"
        return ctrl, f"{own} + shared FINAL network (gen {gen} network not kept), mean action"
    if flag is False:
        summed.shared = None
        return ctrl, f"{own} alone, as recorded"
    return ctrl, f"{own} + shared FINAL network (record does not say), mean action"


# --------------------------------------------------------------------------
# One unit: 4 kinds x 3 arms x 2 gate settings
# --------------------------------------------------------------------------


def rollout(T, p, seed, kind, make_ctrl, gates):
    """One fresh environment, one crossing.  Returns the KEEP fields."""
    from dytiscidae.envs.triphibian import TriphibianEnv

    os.environ["DYTISCIDAE_CROSSING_GATES"] = "1" if gates else "0"
    env = TriphibianEnv(p, seed=seed)
    r = T.run_transition(env, kind, make_ctrl(env))
    out = {}
    for k in KEEP:
        v = getattr(r, k, None)
        out[k] = v if (v is None or isinstance(v, (str, bool))) else (
            int(v) if isinstance(v, (int, np.integer)) else float(v))
    out["crossed"] = bool(r.crossed)
    return out


def arm_factory(arm, run_dir, elite, p, seed):
    """``(make_ctrl(env) -> Controller, description)``."""
    from dytiscidae.envs.evaluate import Controller

    if arm == "base":
        return (lambda env: Controller(params=env.cpg.base, policy=None, bases=None)), "base gait"
    if arm == "still":
        return (lambda env: Controller(params=env.held_still_params(), policy=None,
                                       bases=None)), "held still"
    ctrl, desc = own_controller(run_dir, elite, p, seed)
    if ctrl is None:
        raise RuntimeError(desc)
    return (lambda env: ctrl), desc


def work(job):
    """All rollouts of one unit.  Never raises: a failure is a row."""
    runs_root, unit, arms, kinds = job
    os.environ.setdefault("MUJOCO_GL", "disable")
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import transitions as T

    rows, t0 = [], time.time()
    head = {k: unit[k] for k in ("run", "unit", "island", "index", "members", "credit", "graded",
                                 "stored_crossed_fraction", "stored_approach")}
    try:
        elite = load_elite(runs_root, unit)
        meta = elite.meta or {}
        seed = int(meta.get("eval_seed") or 0)
        p = build(elite.genome)
        head.update({
            "eval_seed": seed, "seed_stored": meta.get("eval_seed") is not None,
            "gen": meta.get("gen"), "plan": meta.get("body_plan"),
            "n_rotors": int(meta.get("n_rotors") or 0),
            "rotor": bool(int(meta.get("n_rotors") or 0) > 0),
            "mass": float(meta.get("mass") or getattr(p, "mass", float("nan"))),
            "dof": meta.get("dof"), "stored_island": meta.get("island"),
        })
    except Exception as exc:                                      # noqa: BLE001
        err = f"exception: {exc!r} | {traceback.format_exc(limit=2)}"
        for kind in kinds:
            for arm in arms:
                for g in (0, 1):
                    rows.append({**head, "kind": kind, "arm": arm, "gates": g, "exception": err})
        return rows, time.time() - t0
    for arm in arms:
        try:
            make, desc = arm_factory(arm, Path(runs_root) / unit["run"], elite, p, seed)
            arm_err = None
        except Exception as exc:                                  # noqa: BLE001
            make, desc, arm_err = None, "", f"exception: {exc!r}"
        for kind in kinds:
            for g in (0, 1):
                row = {**head, "kind": kind, "arm": arm, "gates": g, "law": desc}
                if arm_err is not None:
                    row["exception"] = arm_err
                else:
                    try:
                        row.update(rollout(T, p, seed, kind, make, g))
                    except Exception as exc:                      # noqa: BLE001
                        row["exception"] = f"exception: {exc!r} | {traceback.format_exc(limit=3)}"
                rows.append(row)
    return rows, time.time() - t0


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def pick_units(runs_root, runs, sample):
    """Every unit, or (``sample`` > 0) every unit with stored crossing credit plus a
    seeded sample of ``sample`` others per run."""
    rng = np.random.default_rng(SAMPLE_SEED)
    out, sizes = [], {}
    for run in runs:
        us = units_of(runs_root, run)
        entries = sum(len(u["members"]) for u in us)
        if sample > 0:
            cred = [u for u in us if u["credit"]]
            rest = [u for u in us if not u["credit"]]
            take = sorted(rng.choice(len(rest), size=min(sample, len(rest)), replace=False))
            chosen = cred + [rest[i] for i in take]
        else:
            cred, chosen = [u for u in us if u["credit"]], us
        sizes[run] = {"island_entries": entries, "unique": len(us), "with_credit": len(cred),
                      "replayed": len(chosen)}
        out += chosen
    return out, sizes


def cmd_run(args):
    import multiprocessing as mp

    out = Path(args.out)
    units, sizes = pick_units(args.runs_root, args.runs, args.sample)
    print(json.dumps(sizes), flush=True)
    if args.limit:
        units = units[: args.limit]
    done = set()
    if out.exists() and args.resume:
        cnt = collections.Counter(json.loads(ln)["run"] + ":" + json.loads(ln)["unit"]
                                  for ln in out.read_text().splitlines() if ln.strip())
        done = {k for k, n in cnt.items() if n >= len(args.arms) * len(KINDS) * 2}
    for other in args.exclude:
        done |= {json.loads(ln)["run"] + ":" + json.loads(ln)["unit"]
                 for ln in Path(other).read_text().splitlines() if ln.strip()}
    todo = [u for u in units if u["run"] + ":" + u["unit"] not in done]
    print(f"{len(units)} units, {len(done)} done, {len(todo)} to do, "
          f"{len(todo) * len(args.arms) * len(KINDS) * 2} rollouts, {args.procs} processes", flush=True)
    jobs = [(args.runs_root, u, tuple(args.arms), KINDS) for u in todo]
    t0, n = time.time(), 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a" if args.resume else "w") as fh:
        if args.procs <= 1:
            it = map(work, jobs)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(args.procs, maxtasksperchild=25)
            it = pool.imap_unordered(work, jobs)
        for rows, wall in it:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            n += 1
            if n % 10 == 0 or n == len(jobs) or args.procs <= 1:
                el = time.time() - t0
                print(f"unit {n}/{len(jobs)} ({rows[0]['run']} {rows[0]['island']}:{rows[0]['index']}) "
                      f"unit wall {wall:.0f}s elapsed {el:.0f}s eta {el / n * (len(jobs) - n):.0f}s",
                      flush=True)
        if pool is not None:
            pool.close()
            pool.join()
    (out.parent / (out.stem + "_sizes.json")).write_text(json.dumps(
        {"sample": args.sample, "sample_seed": SAMPLE_SEED, "sizes": sizes}, indent=1))
    return 0


def pre_commit_transitions():
    """``transitions.py`` as it was before 7c59754 (no bad_qacc read in
    ``run_transition``, no gates), loaded from git beside the current one."""
    import types

    src = subprocess.run(["git", "show", f"{PRE_COMMIT}:dytiscidae/envs/transitions.py"],
                         capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parent).stdout
    mod = types.ModuleType("dytiscidae.envs.transitions_pre")
    mod.__package__ = "dytiscidae.envs"
    mod.__file__ = f"<git {PRE_COMMIT}:dytiscidae/envs/transitions.py>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def qacc_work(job):
    runs_root, unit, cases = job
    os.environ.setdefault("MUJOCO_GL", "disable")
    from dytiscidae.core.phenotype import build

    P = pre_commit_transitions()
    out = []
    try:
        elite = load_elite(runs_root, unit)
        seed = int((elite.meta or {}).get("eval_seed") or 0)
        p = build(elite.genome)
    except Exception as exc:                                      # noqa: BLE001
        return [{"run": unit["run"], "unit": unit["unit"], "kind": k, "arm": a,
                 "exception": f"exception: {exc!r}"} for k, a in cases]
    made = {}
    for kind, arm in cases:
        row = {"run": unit["run"], "unit": unit["unit"], "kind": kind, "arm": arm}
        try:
            if arm not in made:
                made[arm] = arm_factory(arm, Path(runs_root) / unit["run"], elite, p, seed)[0]
            got = rollout(P, p, seed, kind, made[arm], 0)
            row.update({"pre_" + k: got.get(k) for k in ("crossed", "failure", "hold", "approach")})
        except Exception as exc:                                  # noqa: BLE001
            row["exception"] = f"exception: {exc!r}"
        out.append(row)
    return out


def cmd_qacc(args):
    """Replay every row the bad_qacc read touched through the pre-commit
    ``run_transition``: what the row was before the read."""
    import multiprocessing as mp

    rows = [r for f in args.rows for r in _read(f) if "exception" not in r]
    cases, units = collections.defaultdict(list), {}
    for r in rows:
        if r["gates"] == 0 and int(r.get("bad_qacc") or 0) > 0:
            k = (r["run"], r["unit"])
            cases[k].append((r["kind"], r["arm"]))
            units[k] = {x: r[x] for x in ("run", "unit", "island", "index")}
    jobs = [(args.runs_root, units[k], cases[k]) for k in cases]
    print(f"{sum(len(c) for c in cases.values())} rows with bad_qacc > 0 in {len(jobs)} units",
          flush=True)
    with open(args.out, "w") as fh, mp.get_context("spawn").Pool(args.procs, maxtasksperchild=25) as pool:
        for i, out in enumerate(pool.imap_unordered(qacc_work, jobs)):
            for r in out:
                fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0:
                print(f"unit {i + 1}/{len(jobs)}", flush=True)
    return 0


# --------------------------------------------------------------------------
# Report (pure Python over the rows)
# --------------------------------------------------------------------------

ABORTS = ("battery exhausted mid-transition", "diverged", "unstable")
START_OF = {"air_to_water": "air", "water_to_air": "water", "water_to_land": "water",
            "land_to_air": "land"}


def _read(path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def _quart(v):
    if not v:
        return "n=0"
    q = np.percentile(v, [0, 25, 50, 75, 100])
    return (f"n={len(v)}: min {q[0]:.3f}, p25 {q[1]:.3f}, median {q[2]:.3f}, "
            f"p75 {q[3]:.3f}, max {q[4]:.3f}")


def _pairs(rows):
    """``{(run, unit, kind, arm): {0: row, 1: row}}``."""
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["run"], r["unit"], r["kind"], r["arm"])][r["gates"]] = r
    return by


def cmd_report(args):
    sample = _read(args.rows)
    rest = _read(args.own_rest) if args.own_rest and Path(args.own_rest).exists() else []
    qacc = _read(args.qacc) if args.qacc and Path(args.qacc).exists() else None
    sizes = json.loads(Path(args.sizes).read_text()) if args.sizes else {}
    allrows = sample + rest
    runs = [r for r in RUNS if any(x["run"] == r for x in allrows)]
    L = []
    w = L.append

    exc = [r for r in allrows if "exception" in r]
    good = [r for r in allrows if "exception" not in r]
    pairs = {k: v for k, v in _pairs(good).items() if 0 in v and 1 in v}
    nondet = sum(any(v[0][f] != v[1][f] for f in
                     ("hold_drift", "min_upright", "exit_upright", "bad_qacc", "end_dwell", "exit_depth"))
                 for v in pairs.values())

    def units(rows, run=None):
        return {(r["run"], r["unit"]) for r in rows if run is None or r["run"] == run}

    w("# P0.2 on the evolved populations: what the 2026-11-11 crossing gates do to stored elites")
    w("")
    w("Generated by `experiments/triphibian_diagnosis/scripts/population_gates.py report` from the "
      "rows files; every number below is counted from rows, none is typed.  No training of any "
      "kind: stored elites replayed through `transitions.run_transition` at commit 7c59754.")
    w("")
    w("## What was run")
    w("")
    w("- Single path (`run_transition`), default entry state (`back=0`, the fixed "
      "`transition_scatter_seed`), 6 s, a fresh `TriphibianEnv(p, seed=eval_seed)` per rollout; "
      "every replayed elite stores an `eval_seed`.")
    w("- Arms: `own` (stored policy + the run's shared network of the elite's generation from "
      "`scoring_networks/`, shared half at its mean via `SummedPolicy`, stored mobility bases), "
      "`base` (`Controller(params=env.cpg.base, policy=None, bases=None)`), `still` "
      "(`Controller(params=env.held_still_params(), policy=None, bases=None)`).")
    w("- Each (elite, kind, arm) twice: `DYTISCIDAE_CROSSING_GATES=0` and `=1`, set before the "
      "tracker is constructed.")
    w("- Elites are island-archive entries: `(run, island, index in that island's archive)`.  No "
      "two entries shared genome, seed, generation and policy, so none was merged.")
    w("")
    w("### Sample rule and sizes")
    w("")
    w("The full job (1,115 entries x 24 rollouts at ~1.35 s each) was measured at ~2.5 h on 4 "
      "processes, so it was sampled, in two strata:")
    w("")
    w(f"- **Stratum A, all three arms**: every entry whose archive record has a completed crossing "
      f"(`ladder_measurements.transition.crossed_fraction > 0`), plus a seeded random sample "
      f"(`np.random.default_rng({SAMPLE_SEED})`, runs drawn in the order arch49, arch51, arch56) "
      f"of 60 other entries per run.")
    w("- **Stratum B, own arm only**: every remaining entry, so the own arm (items 1 and 3) "
      "covers the whole population; base and still (item 2) are stratum A only.")
    w("")
    w("| run | island entries | stored completed crossing | stratum A (3 arms) | stratum B (own only) "
      "| own arm total |")
    w("|---|---|---|---|---|---|")
    for run in runs:
        s = (sizes.get("sizes") or {}).get(run, {})
        a, b = len(units(sample, run)), len(units(rest, run))
        w(f"| {run} | {s.get('island_entries', '?')} | {s.get('with_credit', '?')} | {a} | {b} | {a + b} |")
    w("")
    w("\"Stored crossing credit\" was read as a completed crossing.  The graded approach "
      "(`ladder_measurements.transition.crossed > 0`) is recorded on "
      f"{len({(r['run'], r['unit']) for r in allrows if r.get('graded')})} of "
      f"{len(units(allrows))} replayed entries and is too common to be a stratum.")
    w("")
    w(f"Rows: {len(allrows)} ({len(sample)} stratum A, {len(rest)} stratum B); "
      f"{len(exc)} rows are exceptions (kept as rows, never as 0).  Determinism: of {len(pairs)} "
      f"(elite, kind, arm) pairs, {nondet} differ between gates off and on in any raw field "
      "(hold_drift, min_upright, exit_upright, bad_qacc, end_dwell, exit_depth), so a gates-on row "
      "is the same trajectory scored differently.")
    laws = collections.Counter()
    for r in good:
        if r["arm"] == "own" and r["gates"] == 0 and r["kind"] == KINDS[0]:
            law = r.get("law", "")
            laws[" ".join("N" if t.isdigit() else t for t in law.split())] += 1
    w("")
    w("Own-arm control law, by entry: " + "; ".join(f"{k}: {v}" for k, v in laws.most_common()) + ".")
    w("")

    # ---- item 1 ---------------------------------------------------------
    own = {k: v for k, v in pairs.items() if k[3] == "own"}
    w("## 1. Own arm: crossings with the gates off and on")
    w("")
    w("Denominator: every replayed entry of the run (strata A and B) whose pair of rollouts ran.")
    w("")
    w("| run | kind | crossed, gates off | crossed, gates on | lost | gained |")
    w("|---|---|---|---|---|---|")
    lost = []
    for run in runs:
        for kind in KINDS:
            v = [p for k, p in own.items() if k[0] == run and k[2] == kind]
            off = sum(p[0]["crossed"] for p in v)
            on = sum(p[1]["crossed"] for p in v)
            ls = [p for p in v if p[0]["crossed"] and not p[1]["crossed"]]
            gn = [p for p in v if p[1]["crossed"] and not p[0]["crossed"]]
            lost += ls
            w(f"| {run} | {kind} | {off}/{len(v)} | {on}/{len(v)} | {len(ls)} | {len(gn)} |")
    w("")
    tot_off = sum(p[0]["crossed"] for p in own.values())
    tot_on = sum(p[1]["crossed"] for p in own.values())
    w(f"All runs and kinds: {tot_off}/{len(own)} own-arm crossings with the gates off, "
      f"{tot_on}/{len(own)} with them on; {len(lost)} lost.")
    w("")

    def describe(p):
        o, g = p[0], p[1]
        station = o["hold"] >= HOLD_PASS_ and g["hold"] < HOLD_PASS_
        over = (g.get("exit_upright") is not None and g["exit_upright"] <= 0.0
                and o["kind"] in KINDS)
        which = " + ".join(x for x, t in (("station", station), ("upright", over)) if t) or "?"
        return (f"| {o['run']} | {o['island']}:{o['index']} | {o['kind']} | {o.get('plan')} | "
                f"{'rotor x' + str(o.get('n_rotors')) if o.get('rotor') else 'no rotor'} | "
                f"{o.get('mass'):.3f} | {o['hold_drift']:.3f} | {o['hold']:.2f} -> {g['hold']:.2f} | "
                f"{o['exit_upright']:.3f} | {o['min_upright']:.3f} | "
                f"{'-' if o.get('end_dwell') is None else format(o['end_dwell'], '.2f')} | "
                f"{which} | {g['failure']} |")

    head = ("| run | island:index | kind | plan | rotor | mass kg | hold_drift m | hold off -> on | "
            "exit_upright | min_upright | end_dwell | gate | failure with gates on |")
    sep = "|" + "---|" * 13
    w("### Every own-arm crossing lost to the gates")
    w("")
    w("`station`: the hold fell below HOLD_PASS (0.5), i.e. the root drifted more than 0.50 m in "
      "the 1.5 s hold.  `upright`: `exit_upright <= 0`.  Both can apply; the tracker reports the "
      "hold first.  An air start (`air_to_water`) is launched at speed and its hold is "
      "energy height, not station: its `hold_drift` of tens of metres is not read by any gate.")
    w("")
    if lost:
        w(head)
        w(sep)
        for p in sorted(lost, key=lambda p: (p[0]["run"], p[0]["kind"], p[0]["island"], p[0]["index"])):
            w(describe(p))
    else:
        w("None.")
    w("")
    kept = [p for p in own.values() if p[0]["crossed"] and p[1]["crossed"]]
    w("### Every own-arm crossing kept with the gates on")
    w("")
    if kept:
        w(head)
        w(sep)
        for p in sorted(kept, key=lambda p: (p[0]["run"], p[0]["kind"], p[0]["island"], p[0]["index"])):
            w(describe(p).replace("| ? |", "| kept |"))
    else:
        w("None.")
    w("")
    by_gate = collections.Counter()
    for p in lost:
        st = p[0]["hold"] >= HOLD_PASS_ and p[1]["hold"] < HOLD_PASS_
        ov = p[1]["exit_upright"] <= 0.0
        by_gate[("station" if st else "") + ("+" if st and ov else "") + ("upright" if ov else "")] += 1
    w("Lost, by gate: " + (", ".join(f"{k or '?'} {v}" for k, v in sorted(by_gate.items())) or "none")
      + f".  Lost, with a rotor: {sum(bool(p[0].get('rotor')) for p in lost)}/{len(lost)}; "
      f"kept, with a rotor: {sum(bool(p[0].get('rotor')) for p in kept)}/{len(kept)}.")
    w("")
    cred = {(r["run"], r["unit"]) for r in allrows if r.get("credit")}
    cross_units = {k[:2] for k, p in own.items() if p[0]["crossed"]}
    w(f"Reproduction of the archive: {len(cred)} replayed entries store a completed crossing; "
      f"{len(cred & cross_units)} of them complete any crossing here with the gates off, and "
      f"{len(cross_units - cred)} entries cross here that store none.")
    w("")
    ap_off = [p[0]["approach"] for p in own.values()]
    ap_on = [p[1]["approach"] for p in own.values()]
    ch = sum(abs(a - b) > 1e-12 for a, b in zip(ap_off, ap_on))
    w(f"Graded approach (own arm, {len(own)} pairs): changed by the gates in {ch}; mean "
      f"{float(np.mean(ap_off)) if ap_off else float('nan'):.4f} off, "
      f"{float(np.mean(ap_on)) if ap_on else float('nan'):.4f} on; pairs with approach > 0: "
      f"{sum(a > 0 for a in ap_off)} off, {sum(a > 0 for a in ap_on)} on.")
    w("")

    # ---- item 2 ---------------------------------------------------------
    w("## 2. Base and still arms: crossings with the gates off and on (stratum A)")
    w("")
    w("| run | kind | base off | base on | still off | still on |")
    w("|---|---|---|---|---|---|")
    null_cross = []
    for run in runs:
        for kind in KINDS:
            cells = []
            for arm in ("base", "still"):
                v = [p for k, p in pairs.items() if k[0] == run and k[2] == kind and k[3] == arm]
                cells += [f"{sum(p[0]['crossed'] for p in v)}/{len(v)}",
                          f"{sum(p[1]['crossed'] for p in v)}/{len(v)}"]
                null_cross += [p for p in v if p[0]["crossed"] or p[1]["crossed"]]
            w(f"| {run} | {kind} | " + " | ".join(cells) + " |")
    w("")
    for arm in ("base", "still"):
        v = [p for k, p in pairs.items() if k[3] == arm]
        w(f"- {arm}, all runs and kinds: {sum(p[0]['crossed'] for p in v)}/{len(v)} off, "
          f"{sum(p[1]['crossed'] for p in v)}/{len(v)} on.")
    w("")
    if null_cross:
        w("Every base or still crossing, either setting:")
        w("")
        w("| arm | crossed off / on " + head)
        w("|---|---" + sep)
        for p in sorted(null_cross, key=lambda p: (p[0]["arm"], p[0]["run"], p[0]["kind"])):
            w(f"| {p[0]['arm']} | {p[0]['crossed']} / {p[1]['crossed']} " + describe(p))
        w("")
        w("For each of these, the same entry's own arm on the same crossing:")
        w("")
        for p in sorted(null_cross, key=lambda p: (p[0]["arm"], p[0]["run"], p[0]["kind"])):
            o = own.get((p[0]["run"], p[0]["unit"], p[0]["kind"], "own"))
            w(f"- {p[0]['arm']} {p[0]['run']} {p[0]['island']}:{p[0]['index']} {p[0]['kind']}: own "
              + ("not run" if o is None else
                 f"crossed {o[0]['crossed']} off / {o[1]['crossed']} on, hold_drift "
                 f"{o[0]['hold_drift']:.3f}, exit_upright {o[0]['exit_upright']:.3f}"))
        w("")

    # ---- item 3 ---------------------------------------------------------
    w("## 3. hold_drift, by start medium")
    w("")
    w(f"STATION_FULL 0.25 m, STATION_ZERO 0.75 m; the hold passes (>= 0.5) at a drift of at most "
      "0.50 m.  Only rollouts that started in the start medium and were not aborted (battery, "
      "diverged, unstable), gates-off rows; `hold_drift` is the furthest horizontal distance of the "
      "root from where the hold began, over the 1.5 s hold.  Water start: `water_to_air` and "
      "`water_to_land` (two rows per entry).  Land start: `land_to_air`.")
    w("")

    def drift_rows(arm, start, run=None):
        out = []
        for k, p in pairs.items():
            r = p[0]
            if k[3] != arm or START_OF[r["kind"]] != start or (run and r["run"] != run):
                continue
            if r["started_in"] != start or r["failure"] in ABORTS:
                continue
            out.append(r)
        return out

    def band(v):
        if not v:
            return "-"
        a = np.asarray(v)
        return (f"<=0.25: {int((a <= 0.25).sum())}, 0.25-0.50: {int(((a > 0.25) & (a <= 0.5)).sum())}, "
                f"0.50-0.75: {int(((a > 0.5) & (a < 0.75)).sum())}, >=0.75: {int((a >= 0.75).sum())}")

    for arm in ARMS:
        w(f"### {arm} arm" + (" (whole population)" if arm == "own" else " (stratum A)"))
        w("")
        w("| start | run | crossed (gates off) | hold_drift m | by band |")
        w("|---|---|---|---|---|")
        for start in ("water", "land"):
            for run in runs + [None]:
                rs = drift_rows(arm, start, run)
                for lab, sel in (("crossed", True), ("not crossed", False)):
                    v = [r["hold_drift"] for r in rs if bool(r["crossed"]) is sel]
                    w(f"| {start} | {run or 'all'} | {lab} | {_quart(v)} | {band(v)} |")
        w("")
    for start in ("water", "land"):
        n_all = sum(1 for k, p in pairs.items() if k[3] == "own" and START_OF[p[0]["kind"]] == start)
        n_in = len(drift_rows("own", start))
        w(f"- own arm, {start} start: {n_in} of {n_all} rows read; {n_all - n_in} excluded "
          "(aborted, or did not start in the start medium).")
    w("")
    w("Own arm against the still arm on the same entries and crossings (stratum A, both rows "
      "read): how often the controlled machine drifts less than the same body held still.")
    w("")
    for start in ("water", "land"):
        o = {(r["run"], r["unit"], r["kind"]): r["hold_drift"] for r in drift_rows("own", start)}
        s = {(r["run"], r["unit"], r["kind"]): r["hold_drift"] for r in drift_rows("still", start)}
        both = sorted(set(o) & set(s))
        if both:
            d = np.array([o[k] - s[k] for k in both])
            w(f"- {start} start, n={len(both)}: own drifts less in {int((d < 0).sum())}, more in "
              f"{int((d > 0).sum())}; own - still: {_quart(list(d))}.")
    w("")

    # ---- item 4 ---------------------------------------------------------
    w("## 4. Rows changed only by the bad_qacc read")
    w("")
    off = [r for r in good if r["gates"] == 0]
    bq = [r for r in off if int(r.get("bad_qacc") or 0) > 0]
    un = [r for r in bq if r["failure"] == "unstable"]
    w(f"Gates-off rows: {len(off)}.  With `bad_qacc > 0`: {len(bq)} "
      f"({', '.join(f'{a} {sum(r['arm'] == a for r in bq)}' for a in ARMS)}).  Of those, "
      f"{len(un)} carry `failure == 'unstable'` (the read set it); the other {len(bq) - len(un)} "
      "already had an earlier failure: "
      + (", ".join(f"{k!r} {v}" for k, v in collections.Counter(
          r["failure"] for r in bq if r["failure"] != "unstable").items()) or "none") + ".")
    w("")
    if qacc is None:
        w("The pre-commit replay was not run.")
    else:
        cur = {(r["run"], r["unit"], r["kind"], r["arm"]): r for r in bq}
        qe = [q for q in qacc if "exception" in q]
        qg = [q for q in qacc if "exception" not in q]
        was_cross = [q for q in qg if q["pre_crossed"]]
        was_appr = [q for q in qg if not q["pre_crossed"] and (q["pre_approach"] or 0) > 0
                    and cur[(q["run"], q["unit"], q["kind"], q["arm"])]["approach"] == 0]
        was_hold = [q for q in qg if (q["pre_hold"] or 0) > 0
                    and cur[(q["run"], q["unit"], q["kind"], q["arm"])]["hold"] == 0]
        same = [q for q in qg if not q["pre_crossed"]
                and abs((q["pre_approach"] or 0) - cur[(q["run"], q["unit"], q["kind"], q["arm"])]["approach"]) < 1e-12]
        w(f"Each of the {len(bq)} rows was replayed through `run_transition` as it was at "
          f"{PRE_COMMIT} (the parent of 7c59754: no bad_qacc read, no gates), same body, seed and "
          f"controller ({len(qg)} replayed, {len(qe)} exceptions):")
        w("")
        w(f"- crossed before the read, not after: **{len(was_cross)}** "
          f"({', '.join(f'{a} {sum(q['arm'] == a for q in was_cross)}' for a in ARMS)})")
        w(f"- not crossed either way, graded approach > 0 before and 0 after: **{len(was_appr)}** "
          f"({', '.join(f'{a} {sum(q['arm'] == a for q in was_appr)}' for a in ARMS)})")
        w(f"- hold > 0 before and 0 after (includes the rows above): {len(was_hold)}")
        w(f"- crossed and approach unchanged (only the failure text and `bad_qacc` differ): {len(same)}")
        w("")
        if was_cross:
            w("| run | island:index | kind | arm | plan | rotor | mass kg | bad_qacc | pre-commit failure |")
            w("|---|---|---|---|---|---|---|---|---|")
            for q in was_cross:
                c = cur[(q["run"], q["unit"], q["kind"], q["arm"])]
                w(f"| {c['run']} | {c['island']}:{c['index']} | {c['kind']} | {c['arm']} | {c.get('plan')} | "
                  f"{'rotor x' + str(c.get('n_rotors')) if c.get('rotor') else 'no rotor'} | "
                  f"{c.get('mass'):.3f} | {c['bad_qacc']} | {q['pre_failure']!r} |")
            w("")
    w("")
    w("## Failures, gates off (all arms)")
    w("")
    w("| arm | failure | rows |")
    w("|---|---|---|")
    fc = collections.Counter((r["arm"], r["failure"] or "(crossed)") for r in off)
    for (arm, f), n in sorted(fc.items(), key=lambda kv: (ARMS.index(kv[0][0]), -kv[1])):
        w(f"| {arm} | {f} | {n} |")
    w("")
    if exc:
        w("## Exceptions")
        w("")
        for t, n in collections.Counter((r["arm"], r["exception"][:160]) for r in exc).most_common():
            w(f"- {n} rows, arm {t[0]}: `{t[1]}`")
        w("")
    extra = Path(args.notes).read_text() if args.notes and Path(args.notes).exists() else ""
    if extra:
        w(extra.rstrip())
        w("")
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


HOLD_PASS_ = 0.5    # transitions.HOLD_PASS, repeated so the report needs no simulator


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--rows", required=True)
    rp.add_argument("--own-rest", default="")
    rp.add_argument("--qacc", default="")
    rp.add_argument("--sizes", default="")
    rp.add_argument("--notes", default="", help="a markdown file appended verbatim (item 5)")
    rp.add_argument("--out", required=True)
    a = sub.add_parser("run")
    a.add_argument("--runs-root", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--runs", nargs="+", default=list(RUNS))
    a.add_argument("--arms", nargs="+", default=list(ARMS))
    a.add_argument("--sample", type=int, default=0,
                   help="0 = every unit; else all with stored credit + this many others per run")
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--procs", type=int, default=4)
    a.add_argument("--resume", action="store_true")
    a.add_argument("--exclude", nargs="*", default=[],
                   help="rows files whose units are skipped (the rest of the population)")
    q = sub.add_parser("qacc")
    q.add_argument("--runs-root", required=True)
    q.add_argument("--rows", required=True, nargs="+")
    q.add_argument("--out", required=True)
    q.add_argument("--procs", type=int, default=4)
    s = sub.add_parser("sizes")
    s.add_argument("--runs-root", required=True)
    s.add_argument("--runs", nargs="+", default=list(RUNS))
    s.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "report":
        return cmd_report(args)
    if args.cmd == "run":
        assert args.procs <= 4, "at most 4 processes"
        return cmd_run(args)
    if args.cmd == "qacc":
        assert args.procs <= 4, "at most 4 processes"
        return cmd_qacc(args)
    _, sizes = pick_units(args.runs_root, args.runs, args.sample)
    print(json.dumps(sizes, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
