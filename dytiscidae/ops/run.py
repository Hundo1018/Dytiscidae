"""Command-line entry point.

    python -m dytiscidae.ops.run verify                 physics self-check
    python -m dytiscidae.ops.run reference              inspect the hand design
    python -m dytiscidae.ops.run search   [options]     run the design search
    python -m dytiscidae.ops.run skills   [options]     train the actuator skills
    python -m dytiscidae.ops.run dashboard --run DIR    regenerate the dashboard
    python -m dytiscidae.ops.run render   --run DIR     render archive elites

Every long-running command checkpoints as it goes and can be resumed, because
the environments this is meant to run in are frequently reclaimed without
warning.  Nothing here needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_run_archive(run_dir):
    """The archive of a run, however that run stored it.

    A single-population run writes ``archive.pkl``.  An archipelago writes one
    per island and no combined file, so every downstream command that hard-coded
    ``archive.pkl`` -- render, cohort, and anything filming a result -- has been
    exiting with "no archive" for every run since the islands landed.  The
    search was reachable and its output was not.

    Islands are merged by taking each island's archive in turn and keeping the
    better occupant of any cell two islands both filled.  That is not a
    principled cross-island comparison -- the islands score on different
    objectives, which is the whole point of having them.  For picking something
    to film, "the best thing anywhere" is the right question; for anything else,
    read the per-island archives.

    Each elite carries ``meta["island"]``, the island whose copy won the cell,
    and ``meta["islands"]``, every island that held that cell at all.  Both are
    needed and the second is the honest one: seeding files the same design on
    every island, so for a seed the singular tag is whichever archive happened
    to be read first, and reading it as "where this design lives" is wrong.  It
    is written down because that is exactly how it was misread an hour after
    being introduced -- the flyer looked like it was on the land island and
    absent from the air one, and it was on all six.
    """
    from ..evolution.archive import Archive

    run_dir = Path(run_dir)
    single = run_dir / "archive.pkl"
    if single.exists():
        return Archive.load(single), ["default"]

    parts = sorted(run_dir.glob("archive_*.pkl"))
    if not parts:
        return None, []

    merged, islands = None, []
    for path in parts:
        name = path.stem[len("archive_"):]
        islands.append(name)
        a = Archive.load(path)
        for e in a.cells.values():
            e.meta = dict(e.meta or {})
            e.meta.setdefault("island", name)
            e.meta["islands"] = [name]
        if merged is None:
            merged = a
            continue
        for cell, e in a.cells.items():
            held = merged.cells.get(cell)
            if held is None:
                merged.cells[cell] = e
                continue
            # The cell exists on more than one island; record that before
            # deciding which copy to keep, or the fact is lost.
            seen = list(held.meta.get("islands", [])) + [name]
            if e.fitness > held.fitness:
                e.meta["islands"] = seen
                merged.cells[cell] = e
            else:
                held.meta["islands"] = seen
        merged.tainted.update(a.tainted)
        merged.generation = max(merged.generation, a.generation)
    return merged, islands


def cmd_verify(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.test_physics import main as physics_main

    return physics_main()


def cmd_reference(args) -> int:
    from ..core.mjcf import build_model_xml
    from ..core.phenotype import build
    from ..core.reference import reference_genome
    from ..envs.triphibian import evaluate_tier0

    p = build(reference_genome())
    print(p.summary())
    print()
    print("mass budget (kg):")
    for k, v in p.budget.as_dict().items():
        print(f"  {k:16s} {v:8.3f}")
    print()
    print(f"volumes: envelope={p.displaced_volume*1e3:.1f} L  "
          f"buoyant={p.buoyant_volume*1e3:.1f} L  gas={p.gas_volume*1e3:.1f} L  "
          f"ballast={p.ballast_volume*1e3:.1f} L")
    print(f"survivable water entry: {p.max_entry_speed:.1f} m/s   "
          f"depth instability: {p.depth_instability:+.2f} N/m")
    print()
    print("structural checks (worst first):")
    for c in sorted(p.report.checks, key=lambda c: c.margin):
        print(f"  {c.margin:+7.2f}  {c.name:26s} {c.note}")
    print()
    r0 = evaluate_tier0(p)
    print(f"tier-0: feasible={r0.feasible} mission_fraction={r0.mission_fraction:.3f} "
          f"E_req={r0.energy_required_wh:.0f}Wh E_avail={r0.energy_available_wh:.0f}Wh")
    for n in r0.notes:
        print(f"        {n}")
    if args.write_xml:
        xml, names = build_model_xml(p)
        Path(args.write_xml).write_text(xml)
        print(f"\nwrote MJCF to {args.write_xml} ({len(names)} actuators)")
    return 0


def cmd_search(args) -> int:
    from ..evolution.loop import SearchConfig, run_search
    from ..envs.triphibian import MissionSpec
    from ..viz.dashboard import build_dashboard

    cfg = SearchConfig(
        generations=args.generations,
        batch=args.batch,
        seed=args.seed,
        segment_seconds=args.segment_seconds,
        run_dir=args.run,
        tier2_every=args.tier2_every,
        n_reference_seeds=args.reference_seeds,
        n_random_seeds=args.random_seeds,
        checkpoint_every=args.checkpoint_every,
        learned_axes=not args.fixed_axes,
        descriptor_refit_every=args.descriptor_refit_every,
        migrate_every=args.migrate_every,
        use_critic=not args.no_critic,
        audit_every=args.audit_every,
        use_scout=not args.no_scout,
        scout_reserve=args.scout_reserve,
        **({"islands": tuple(x.strip() for x in args.islands.split(","))}
           if args.islands else {}),
    )
    spec = MissionSpec(
        cycles=args.cycles,
        seconds_per_domain=args.seconds_per_domain,
        target_depth=args.depth,
    )

    def report(state, r):
        line = (
            f"gen{r['generation']:<4d} {r.get('island','-')[:9]:<9s} "
            f"{r['regime']:<12s} elites={r['filled']:<4d} "
            f"cov={r['coverage']*100:5.2f}% qd={r['qd_score']:7.2f} "
            f"best={r.get('best_fitness', 0):.3f} "
            f"stage{r.get('curriculum', {}).get('reached', 0)} "
            f"crit={r.get('critic', {}).get('calibration', 0):.2f} "
            f"inv={r.get('auditor', {}).get('invalidated', 0):<3d} "
            f"scout={r.get('scout', {}).get('calibration', 0):.2f}/"
            f"{r.get('scout', {}).get('protected', 0):<2d} "
            f"evals={r['evaluated']:<5d} {r['elapsed']:6.0f}s"
        )
        print(line, flush=True)
        if r["generation"] % 5 == 0:
            try:
                state.archive.export_json(Path(cfg.run_dir) / "archive.json")
                build_dashboard(cfg.run_dir)
            except Exception as exc:
                print(f"  (dashboard refresh failed: {exc})", flush=True)

    state = run_search(cfg, spec, on_generation=report)
    out = build_dashboard(cfg.run_dir)
    print(f"\ndashboard: {out}")

    best = state.archive.best
    if best is not None:
        print("\nbest design:")
        for k, v in best.meta.items():
            if k in ("policy", "mobility_axes"):
                continue
            print(f"  {k:18s} {v}")
        for medium, lines in (best.meta.get("mobility_axes") or {}).items():
            print(f"  discovered axes ({medium}):")
            for ln in lines:
                print(f"      {ln}")
    return 0


def cmd_skills(args) -> int:
    from ..envs.skills import SKILL_TASKS, baseline_score, train_skill

    out_dir = Path(args.run)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = args.tasks.split(",") if args.tasks else list(SKILL_TASKS)
    results = {}
    for name in names:
        if name not in SKILL_TASKS:
            print(f"unknown task {name!r}; known: {', '.join(SKILL_TASKS)}")
            continue
        base = baseline_score(name)
        print(f"\n{name}  --  {SKILL_TASKS[name].why}")
        print(f"  do-nothing baseline: {base:.3f}")

        def progress(it, h, _n=name):
            if it % 5 == 0 or it == args.iterations - 1:
                print(f"  iter {it:3d}  best={h['best']:.3f} mean={h['mean']:.3f} "
                      f"sigma={h['sigma']:.3f}", flush=True)

        r = train_skill(name, iterations=args.iterations,
                        episodes_per_eval=args.episodes, seed=args.seed,
                        on_iteration=progress)
        r["baseline"] = base
        results[name] = r
        print(f"  trained: {r['best_score']:.3f}  (gain {r['best_score']-base:+.3f})")
        print(f"  detail: {r['final_detail']}")

    path = out_dir / "skills.json"
    path.write_text(json.dumps(results, indent=1, default=float))
    print(f"\nwrote {path}")
    return 0


def cmd_train(args) -> int:
    """Train a controller for one design and render what it learned."""
    import pickle

    from ..control.train import train_controller
    from ..core.phenotype import build
    from ..core.reference import reference_genome
    from ..viz.render import render_design, render_learning_comparison

    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)

    if args.from_archive:
        from ..evolution.archive import Archive

        archive = Archive.load(Path(args.from_archive) / "archive.pkl")
        elites = sorted(archive.cells.values(), key=lambda e: -e.fitness)
        if not elites:
            print("archive is empty")
            return 1
        genome = elites[min(args.rank, len(elites) - 1)].genome
        stem = f"elite{args.rank}"
    else:
        genome = reference_genome()
        stem = "reference"

    p = build(genome)
    print(p.summary())
    print()

    def progress(it, r):
        print(f"  iter {it:3d}  best={r['best']:.4f} mean={r['mean']:.4f} "
              f"sigma={r['sigma']:.3f}  {r['elapsed']:.0f}s", flush=True)

    controller, result = train_controller(
        p, iterations=args.iterations, popsize=args.popsize,
        segment_seconds=args.segment_seconds, seed=args.seed, on_iteration=progress,
    )
    print()
    print(f"  {result.summary()}")
    for medium, basis in result.bases.items():
        print(f"  discovered axes ({medium}), rank {basis.rank}:")
        for line in basis.describe()[: basis.rank or 1]:
            print(f"      {line}")

    with open(out / f"{stem}_controller.pkl", "wb") as f:
        pickle.dump({"weights": result.policy_weights,
                     "hidden": getattr(result, "policy_hidden", 0),
                     "bases": result.bases,
                     "score": result.score, "baseline": result.baseline_score,
                     "per_domain": result.per_domain,
                     "baseline_per_domain": result.baseline_per_domain,
                     "history": result.history}, f)
    print(f"\nsaved {out / (stem + '_controller.pkl')}")

    if args.render:
        print("rendering...")
        made = render_learning_comparison(p, controller, out / "media", stem=stem,
                                          duration=args.duration)
        made += render_design(p, out / "media", stem=stem, duration=args.duration,
                              controller=controller, turntable=True)
        for m in made:
            print(f"  {m}")
    return 0


def cmd_cohort(args) -> int:
    """Approve a cohort from an archive, and optionally film each member."""
    import json

    from ..control.train import train_controller
    from ..core.phenotype import build
    from ..evolution.archive import Archive
    from ..evolution.curator import Curator
    from ..viz.showcase import render_mission

    run = Path(args.run)
    archive, islands = load_run_archive(run)
    if archive is None:
        print(f"no archive in {run}")
        return 1
    print(f"{len(archive.cells)} elites from {len(islands)} island(s): "
          f"{', '.join(islands)}")
    curator = Curator(archive, seed=args.seed)
    cohort = curator.select_cohort(args.n)
    if not cohort:
        print("no feasible, untainted elites to approve")
        return 1

    rows = curator.cohort_report(cohort)
    print(f"approved cohort of {len(cohort)} from {len(archive.cells)} elites\n")
    hdr = f"{'#':>2} {'fitness':>8} {'mass':>7} {'span':>6} {'rho':>5} " \
          f"{'air':>5} {'water':>5} {'land':>5} {'dof':>4} {'tier':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['rank']:>2} {r['fitness']:>8.4f} {r['mass_kg']:>6.1f}kg "
              f"{r['span_m']:>5.2f}m {r['density_ratio']:>5.2f} {r['air']:>5.2f} "
              f"{r['water']:>5.2f} {r['land']:>5.2f} {r['dof']:>4} {r['tier']:>4}")

    (run / "cohort.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"\nwrote {run / 'cohort.json'}")

    if args.render:
        for i, elite in enumerate(cohort[: args.render_top]):
            p = build(elite.genome)
            controller = None
            if args.train:
                print(f"\ntraining controller for cohort member {i}...")
                controller, res = train_controller(
                    p, iterations=args.iterations, popsize=args.popsize,
                    segment_seconds=args.segment_seconds, seed=args.seed,
                    continuous=True,
                )
                print(f"  {res.summary()}")
            path, mission = render_mission(
                p, controller, run / "media" / f"cohort{i}_mission.mp4",
                leg_seconds=args.leg_seconds, cycles=args.cycles, seed=args.seed,
                label=f"cohort#{i}  {p.mass:.1f}kg",
            )
            print(f"  {path}")
            print(f"  {mission.summary()}")
    return 0


def cmd_showcase(args) -> int:
    """Train a controller and film one continuous mission with flow and stress."""
    import pickle

    from ..control.cpg import Policy
    from ..control.train import train_controller
    from ..core.phenotype import build
    from ..core.reference import reference_genome
    from ..envs.evaluate import Controller
    from ..envs.triphibian import TriphibianEnv
    from ..viz.showcase import render_mission

    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)

    # The showcase filmed the hand-built reference and nothing else, so the one
    # video this project exists to produce could not show a design the search
    # found.  ``--design`` names a run whose archive to take the best elite
    # from, optionally restricted to one island.
    if args.design:
        archive, islands = load_run_archive(args.design)
        if archive is None:
            print(f"no archive in {args.design}")
            return 1
        pool = [e for e in archive.cells.values()
                if not args.island or (e.meta or {}).get("island") == args.island]
        if not pool:
            print(f"no elites in {args.design}"
                  + (f" from island {args.island!r} (have: {', '.join(islands)})"
                     if args.island else ""))
            return 1
        elite = max(pool, key=lambda e: e.fitness)
        p = build(elite.genome)
        print(f"filming the best of {len(pool)} elites from {args.design}"
              f"{f' (island {args.island})' if args.island else ''}: "
              f"fitness {elite.fitness:.4f}, island "
              f"{(elite.meta or {}).get('island', '-')}, tier {elite.tier}")
    elif args.plan:
        from ..core.bodyplans import BODY_PLANS

        if args.plan not in BODY_PLANS:
            print(f"unknown plan {args.plan!r}; known: {', '.join(BODY_PLANS)}")
            return 1
        p = build(BODY_PLANS[args.plan]())
        print(f"filming the {args.plan} body plan")
    else:
        p = build(reference_genome())
    print(p.summary())

    controller = None
    if args.controller and Path(args.controller).exists():
        d = pickle.load(open(args.controller, "rb"))
        # Width comes from the pickle: a controller trained before the default
        # changed must still load with the shape it was trained at.
        pol = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=4,
                     hidden=int(d.get("hidden", 16)))
        pol.weights = d["weights"]
        env = TriphibianEnv(p, seed=args.seed)
        controller = Controller(params=env.cpg.base, policy=pol, bases=d["bases"])
        print(f"loaded controller from {args.controller} "
              f"(score {d.get('baseline', 0):.3f} -> {d.get('score', 0):.3f})")
    elif args.train:
        def prog(it, r):
            print(f"  iter {it:3d} best={r['best']:.4f} mean={r['mean']:.4f} "
                  f"{r['elapsed']:.0f}s", flush=True)

        controller, res = train_controller(
            p, iterations=args.iterations, popsize=args.popsize,
            segment_seconds=args.segment_seconds, seed=args.seed, continuous=True,
            on_iteration=prog,
        )
        print(f"  {res.summary()}")

    path, mission = render_mission(
        p, controller, out / "mission.mp4", leg_seconds=args.leg_seconds,
        cycles=args.cycles, seed=args.seed, show_wake=args.wake,
        show_stress=args.stress,
    )
    print(f"\n{path}")
    print(mission.summary())
    for leg in mission.legs:
        print(f"  leg {leg.index+1} {leg.commanded.value:<6s} "
              f"on-task {leg.on_task_fraction*100:3.0f}%  "
              f"entry {'-' if leg.entry_time is None else f'{leg.entry_time:.1f}s'}  "
              f"P {leg.mean_power:.0f} W")
    return 0


def cmd_dashboard(args) -> int:
    from ..viz.dashboard import build_dashboard

    print(build_dashboard(args.run))
    return 0


def cmd_render(args) -> int:
    from ..viz.render import render_elites

    archive, islands = load_run_archive(args.run)
    if archive is None:
        print(f"no archive in {args.run}")
        return 1
    print(f"{len(archive.cells)} elites from {len(islands)} island(s): "
          f"{', '.join(islands)}")
    made = render_elites(archive, args.run, top=args.top, duration=args.duration)
    if not made:
        print("nothing rendered (no GL context and no matplotlib?)")
        return 1
    for m in made:
        print(m)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dytiscidae", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("verify", help="run the physics self-check")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("reference", help="inspect the hand-designed reference")
    p.add_argument("--write-xml", default=None, help="also write the MJCF here")
    p.set_defaults(fn=cmd_reference)

    p = sub.add_parser("search", help="run the design search")
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--segment-seconds", type=float, default=8.0,
                   help="Tier-1 episode length; the main cost/fidelity dial")
    p.add_argument("--run", default="runs/latest")
    p.add_argument("--tier2-every", type=int, default=15)
    p.add_argument("--reference-seeds", type=int, default=12)
    p.add_argument("--random-seeds", type=int, default=8)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--seconds-per-domain", type=float, default=300.0)
    p.add_argument("--depth", type=float, default=10.0)
    p.add_argument("--fixed-axes", action="store_true",
                   help="use the hand-picked archive axes instead of learning them")
    p.add_argument("--descriptor-refit-every", type=int, default=400,
                   help="episodes between refits of the learned archive axes")
    p.add_argument("--islands", default=None,
                   help="comma-separated islands to run (default: all)")
    p.add_argument("--migrate-every", type=int, default=60)
    p.add_argument("--no-critic", action="store_true",
                   help="run without the learned critic")
    p.add_argument("--audit-every", type=int, default=30)
    p.add_argument("--no-scout", action="store_true",
                   help="run without the potential predictor (greedy selection)")
    p.add_argument("--scout-reserve", type=float, default=0.15,
                   help="share of each archive protected on predicted potential")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("skills", help="train the actuator skill library")
    p.add_argument("--tasks", default=None, help="comma-separated subset")
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default="runs/skills")
    p.set_defaults(fn=cmd_skills)

    p = sub.add_parser("train", help="train a controller for one design and film it")
    p.add_argument("--from-archive", default=None,
                   help="run directory to take a design from; default is the reference")
    p.add_argument("--rank", type=int, default=0, help="which elite, 0 = best")
    p.add_argument("--iterations", type=int, default=18)
    p.add_argument("--popsize", type=int, default=12)
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--duration", type=float, default=10.0, help="video length")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default="runs/trained")
    p.add_argument("--no-render", dest="render", action="store_false")
    p.set_defaults(fn=cmd_train, render=True)

    p = sub.add_parser("cohort", help="approve N designs from an archive and film them")
    p.add_argument("--run", default="runs/latest")
    p.add_argument("-n", type=int, default=None, help="cohort size; default 6")
    p.add_argument("--render", action="store_true")
    p.add_argument("--render-top", type=int, default=3)
    p.add_argument("--train", action="store_true", help="train a controller per member")
    p.add_argument("--iterations", type=int, default=18)
    p.add_argument("--popsize", type=int, default=10)
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--leg-seconds", type=float, default=7.0)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_cohort)

    p = sub.add_parser("showcase",
                       help="film one continuous mission with wake and stress overlays")
    p.add_argument("--run", default="runs/showcase")
    p.add_argument("--design", default=None,
                   help="run directory to take the best archive elite from "
                        "(default: the hand-built reference)")
    p.add_argument("--island", default=None,
                   help="restrict --design to one island's archive")
    p.add_argument("--plan", default=None,
                   help="film a named body plan instead (beetle, gannet, ...)")
    p.add_argument("--controller", default=None, help="load a trained controller pickle")
    p.add_argument("--train", action="store_true", help="train one first")
    p.add_argument("--iterations", type=int, default=22)
    p.add_argument("--popsize", type=int, default=10)
    p.add_argument("--segment-seconds", type=float, default=5.0)
    p.add_argument("--leg-seconds", type=float, default=8.0)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wake", dest="wake", action="store_false")
    p.add_argument("--no-stress", dest="stress", action="store_false")
    p.set_defaults(fn=cmd_showcase, wake=True, stress=True)

    p = sub.add_parser("dashboard", help="regenerate the dashboard for a run")
    p.add_argument("--run", default="runs/latest")
    p.set_defaults(fn=cmd_dashboard)

    p = sub.add_parser("render", help="render archive elites to video")
    p.add_argument("--run", default="runs/latest")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--duration", type=float, default=8.0)
    p.set_defaults(fn=cmd_render)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
