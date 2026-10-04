"""Crossing rate by probe start distance, and air competence by launch height (ROADMAP Y/O).

Sets the placeholder numbers of ``evolution.curriculum.DistanceCurriculum``
(step 0.5 m, advance_share 0.5, window 200, max_back 20, launch_step 2 m to a
4 m floor, launch_bar 0.1) from measured rates instead of typing them.

For every elite of a finished run, through the batched evaluator the search
scores with (``run_transition_batch`` / ``rollout_batch``), under the network
that scored it (``scoring_networks/gen<gen>.npz``) and its stored own policy and
mobility basis:

  * each crossing kind at start distance ``back`` in BACKS (``_place_for``);
  * the air segment at launch height in HEIGHTS (``env.air_launch_height``).

Every cell is also run with the actuators held still -- amplitude zero, phase
and offset kept, no policy, no shared network -- because a crossing rate is
evidence only above its still-machine column (CLAUDE.md, the eighth instance).

    systemd-run --user --scope -q -p MemoryMax=3500M env MUJOCO_GL=disable PYTHONPATH=. \
        DYTISCIDAE_KERNEL_DIR=<main>/mojo/build .venv/bin/python \
        experiments/transition_distance/run.py --run runs/arch46 \
        --out experiments/transition_distance/results.json
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

KINDS = ("air_to_water", "water_to_air", "water_to_land", "land_to_water")
BACKS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
HEIGHTS = (30.0, 20.0, 12.0, 8.0, 4.0)
MEDIA = ("air", "water", "land")


def still_params(env):
    """The base gait with every amplitude zeroed (experiments/still_transitions).

    **Rotors keep spinning** at their throttle: a rotor's channel is a speed held
    at its offset.  This was the "still" arm of the 2026-10-04 run; it is kept as
    the ``rotors_on`` arm, and the still arm is now ``env.held_still_params()``.
    """
    from dytiscidae.control.cpg import CPGParams
    b = env.cpg.base
    return CPGParams(amplitude=np.zeros(env.cpg.n), phase=np.asarray(b.phase, float),
                     offset=np.asarray(b.offset, float), frequency=float(b.frequency))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="0 = every elite, else a stratified n")
    ap.add_argument("--max-groups", type=int, default=0)
    # Subsetting (2026-10-04 re-measure): which elites, kinds, distances and arms;
    # and whether to run the air-by-height cells at all.
    ap.add_argument("--indices", default="", help="comma-separated elite indices; empty = all")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--backs", default=",".join(str(b) for b in BACKS))
    ap.add_argument("--arms", default="elite,still",
                    help="elite, still (every actuator still, rotors stopped), rotors_on")
    ap.add_argument("--no-air", action="store_true")
    args = ap.parse_args()
    kinds = tuple(k for k in args.kinds.split(",") if k)
    backs = tuple(float(b) for b in args.backs.split(",") if b)
    want_arms = tuple(a for a in args.arms.split(",") if a)

    sys.path.insert(0, "experiments/shared_policy_value")
    from rescore import load_elites, plain_controller, stratified
    from dytiscidae.core.phenotype import build
    from dytiscidae.envs import batchroll
    from dytiscidae.envs.batchroll import BatchedFluid, rollout_batch, run_transition_batch
    from dytiscidae.envs.evaluate import Controller, _scatter_seed
    from dytiscidae.envs.tasks import schedule_for, task_seed
    from dytiscidae.envs.transitions import HOLD_PASS
    from dytiscidae.envs.triphibian import Domain, TriphibianEnv
    from dytiscidae.viz.film import control_laws, run_provenance

    run = Path(args.run)
    ok, why = batchroll.usable()
    assert ok, f"batched evaluator unusable: {why}"
    seg = float((run_provenance(run).get("config") or {}).get("segment_seconds") or 8.0)
    elites = load_elites(run)
    pick = stratified(elites, args.n) if args.n else list(range(len(elites)))
    if args.indices:
        pick = [int(i) for i in args.indices.split(",")]
    groups = collections.defaultdict(list)
    for i in pick:
        m = elites[i].meta or {}
        groups[(int(m.get("gen") or 0), int(m.get("eval_seed") or 0))].append(i)
    keys = sorted(groups)[: args.max_groups or None]
    print(f"{len(elites)} elites, measuring {sum(len(groups[k]) for k in keys)} in {len(keys)} "
          f"groups; kinds {kinds}, back {backs}, arms {want_arms}, "
          f"heights {() if args.no_air else HEIGHTS}", flush=True)

    trows, arows, t0 = [], [], time.time()
    for gi, key in enumerate(keys):
        idx = groups[key]
        phenos = [build(elites[i].genome) for i in idx]
        ctrls, like = [], None
        for i, p in zip(idx, phenos):
            c, sh = plain_controller(run, elites[i], p, key[1])
            ctrls.append(c)
            like = like or sh
        _, net = next(iter(control_laws(elites[idx[0]], run, like)))
        envs = [TriphibianEnv(p, seed=key[1]) for p in phenos]
        stills = [Controller(params=e.held_still_params(), policy=None, bases=c.bases)
                  for e, c in zip(envs, ctrls)]
        spun = [Controller(params=still_params(e), policy=None, bases=c.bases)
                for e, c in zip(envs, ctrls)]
        bf = BatchedFluid(envs)
        arms = tuple(a for a in (("elite", ctrls, net), ("still", stills, None),
                                 ("rotors_on", spun, None)) if a[0] in want_arms)

        for kind in kinds:
            for back in backs:
                for arm, cs, sh in arms:
                    trs = run_transition_batch(envs, bf, kind, cs, shared=sh, back=back)
                    for slot, i in enumerate(idx):
                        t = trs[slot]
                        trows.append({"index": int(i), "island": (elites[i].meta or {}).get("island"),
                                      "kind": kind, "back": back, "arm": arm,
                                      "crossed": bool(t.crossed), "hold": float(t.hold),
                                      "hold_pass": bool(t.hold >= HOLD_PASS),
                                      "started_in": t.started_in, "failure": t.failure,
                                      "approach": float(t.approach)})

        dom = Domain.AIR
        scatter_seed = _scatter_seed(key[1], dom)
        task = schedule_for(dom, np.random.default_rng(task_seed(scatter_seed)))
        for h in (() if args.no_air else HEIGHTS):
            for arm, cs, sh in arms:
                for e in envs:
                    e.air_launch_height = None if h >= 30.0 else float(h)
                    e.reset(dom)
                    e.scatter(np.random.default_rng(scatter_seed))
                    e.task = task
                bf.reset_slam()
                segs = rollout_batch(envs, bf, seg, [c.params for c in cs], dom,
                                     policies=[c.policy for c in cs],
                                     bases=[c.basis_for(dom) for c in cs], shared=sh)
                for slot, i in enumerate(idx):
                    arows.append({"index": int(i), "island": (elites[i].meta or {}).get("island"),
                                  "height": h, "arm": arm,
                                  "competence": float(segs[slot].competence),
                                  "recorded": (elites[i].meta or {}).get("air")})
        for e in envs:
            e.air_launch_height = None
        print(f"group {gi + 1}/{len(keys)} gen {key[0]} n={len(idx)} wall {time.time() - t0:.0f}s",
              flush=True)

    # ---- aggregate --------------------------------------------------------
    def share(rows, f):
        return float(np.mean([f(r) for r in rows])) if rows else float("nan")

    table = {}
    for kind in kinds:
        for back in backs:
            cell = {}
            for arm in want_arms:
                rs = [r for r in trows if r["kind"] == kind and r["back"] == back and r["arm"] == arm]
                cell[arm] = {"n": len(rs),
                             "crossed": share(rs, lambda r: r["crossed"]),
                             "hold_pass": share(rs, lambda r: r["hold_pass"]),
                             "started_in_right_medium": share(
                                 rs, lambda r, k=kind: r["started_in"] == (
                                     "air" if k.startswith("air") else
                                     "water" if k.startswith("water") else "land")),
                             "failed": share(rs, lambda r: bool(r["failure"]))}
            table[f"{kind}|{back}"] = cell
    air = {}
    for h in (() if args.no_air else HEIGHTS):
        cell = {}
        for arm in want_arms:
            rs = [r for r in arows if r["height"] == h and r["arm"] == arm]
            c = np.array([r["competence"] for r in rs])
            cell[arm] = {"n": len(rs), "mean": float(c.mean()), "share_ge_0.1": float((c >= 0.1).mean()),
                         "share_ge_0.3": float((c >= 0.3).mean())}
        air[str(h)] = cell
    ref = [r for r in arows if r["height"] == 30.0 and r["arm"] == "elite" and r["recorded"] is not None]
    repro = {"n": len(ref), "within_0.005": int(sum(abs(r["competence"] - r["recorded"]) <= 0.005
                                                     for r in ref))}
    out = {"run": str(run), "segment_seconds": seg, "elites": len({r["index"] for r in trows}),
           "of_archive": len(elites), "backs": backs, "kinds": kinds, "arms": want_arms,
           "heights": () if args.no_air else HEIGHTS,
           "wall_s": time.time() - t0, "transitions": table, "air_by_height": air,
           "air_30m_reproduces_recorded": repro, "transition_rows": trows, "air_rows": arows}
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"{'kind':14s} {'back':>4s}  " + "   ".join(f"{a} crossed (n) / hold-pass" for a in want_arms))
    for kind in kinds:
        for back in backs:
            c = table[f"{kind}|{back}"]
            print(f"{kind:14s} {back:4.1f}  " + "   ".join(
                f"{c[a]['crossed']:.3f} ({round(c[a]['crossed'] * c[a]['n'])}/{c[a]['n']}) / "
                f"{c[a]['hold_pass']:.3f}" for a in want_arms))
    for h in (() if args.no_air else HEIGHTS):
        a = air[str(h)]
        print(f"air launch {h:4.0f} m  elite mean {a['elite']['mean']:.3f} (>=0.1: {a['elite']['share_ge_0.1']:.3f})"
              f"   still mean {a['still']['mean']:.3f} (>=0.1: {a['still']['share_ge_0.1']:.3f})")
    print("air at 30 m reproduces recorded:", repro)


if __name__ == "__main__":
    main()
