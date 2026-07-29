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
    )
    spec = MissionSpec(
        cycles=args.cycles,
        seconds_per_domain=args.seconds_per_domain,
        target_depth=args.depth,
    )

    def report(state, r):
        line = (
            f"gen{r['generation']:<4d} {r['regime']:<13s} elites={r['filled']:<4d} "
            f"cov={r['coverage']*100:5.2f}% qd={r['qd_score']:7.2f} "
            f"best={r.get('best_fitness', 0):.3f} evals={r['evaluated']:<5d} "
            f"rej={r['tier0_rejected']:<4d} {r['elapsed']:6.0f}s"
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
        pickle.dump({"weights": result.policy_weights, "bases": result.bases,
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


def cmd_dashboard(args) -> int:
    from ..viz.dashboard import build_dashboard

    print(build_dashboard(args.run))
    return 0


def cmd_render(args) -> int:
    from ..evolution.archive import Archive
    from ..viz.render import render_elites

    path = Path(args.run) / "archive.pkl"
    if not path.exists():
        print(f"no archive at {path}")
        return 1
    archive = Archive.load(path)
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
