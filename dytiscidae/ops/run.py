"""Command-line entry point.

    python -m dytiscidae.ops.run verify                 physics self-check
    python -m dytiscidae.ops.run reference              inspect the hand design
    python -m dytiscidae.ops.run search   [options]     run the design search
    python -m dytiscidae.ops.run skills   [options]     train the actuator skills
    python -m dytiscidae.ops.run dashboard --run DIR    regenerate the dashboard
    python -m dytiscidae.ops.run render   --run DIR     render archive elites

Every long-running command checkpoints as it goes and can be resumed, because
the environments this is meant to run in are frequently reclaimed without
warning.

`search` uses the GPU when the Mojo extension is importable, evaluating a whole
generation of candidates in one set of kernel launches.  It falls back to the
numpy solver when the extension is missing, which is correct but several times
slower; see docs/CPU_LEGACY.md for what else was decided under that assumption
and has not been revisited.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_run_archive(run_dir, island: str | None = None):
    """The archive of a run, however that run stored it.

    ``island`` reads that island's archive **alone**.  Filtering the merged
    archive by ``meta["island"]`` is not the same thing, and it is what
    ``showcase --island`` did: the merge keeps one occupant per cell across all
    islands, so an island's elite that lost its cell to another island's
    occupant was gone before the filter ran.  Measured on arch39's
    generation-160 checkpoint, the air island's archive held 78 elites and the
    filtered merge held 34 of them -- and the air island's best flier was not
    among the 34.

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
    names = [q.stem[len("archive_"):] for q in parts]
    if island is not None:
        if island not in names:
            return None, names
        a = Archive.load(run_dir / f"archive_{island}.pkl")
        for e in a.cells.values():
            e.meta = dict(e.meta or {})
            e.meta.setdefault("island", island)
            e.meta["islands"] = [island]
        return a, [island]

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

    shared = args.shared_policy
    if shared is None:
        try:
            import torch  # noqa: F401
            shared = True
        except ImportError:
            shared = False
            print("[warn] torch not importable: running without the shared "
                  "policy (pass --no-shared-policy to silence)", file=sys.stderr)
    if not shared and args.refine_steps <= 0:
        print("[warn] no shared policy and --refine-steps 0: a candidate whose "
              "own weights are zero commands nothing", file=sys.stderr)
    cfg = SearchConfig(
        generations=args.generations,
        batch=args.batch,
        seed=args.seed,
        workers=args.workers,
        min_shard=args.min_shard,
        pool_per_worker=args.pool_per_worker,
        pool_balance=bool(args.pool_balance),
        controller_refine_funnel=args.refine_funnel,
        distance_curriculum=bool(args.distance_curriculum),
        action_rate_penalty=args.action_rate_penalty,
        descriptor_keep_if_overlap=args.descriptor_keep_if_overlap,
        tier2_label_all_media=not args.no_tier2_label_all_media,
        gait_gain=bool(args.gait_gain),
        segment_seconds=args.segment_seconds,
        controller_refine_steps=args.refine_steps,
        controller_refine_sigma=args.refine_sigma,
        promotion_refine_steps=args.promotion_refine_steps,
        policy_hidden=args.policy_hidden,
        use_shared_policy=bool(shared),
        shared_hidden=args.shared_hidden,
        shared_lr=args.shared_lr,
        shared_epochs=args.shared_epochs,
        shared_target_kl=args.shared_target_kl,
        shared_ent_coef=args.shared_ent_coef,
        shared_learner=args.shared_learner,
        grpo_group=args.grpo_group,
        grpo_bodies=args.grpo_bodies,
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
        resume=bool(getattr(args, "resume", False)),
        memory_ceiling_mb=args.memory_ceiling_mb,
        scout_reserve=args.scout_reserve,
        mission_weight=args.mission_weight,
        descriptor_bins=args.descriptor_bins,
        reward_shaping=args.reward_shaping,
        n_modes=args.n_modes,
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
            f"lin={r.get('scout', {}).get('depth_mean', 0):.1f} "
            f"stage{r.get('curriculum', {}).get('typical', 0)}"
            f"/{r.get('curriculum', {}).get('reached', 0)} "
            f"crit={r.get('critic', {}).get('calibration', 0):.2f} "
            f"inv={r.get('auditor', {}).get('invalidated', 0):<3d} "
            f"scout={r.get('scout', {}).get('calibration', 0):.2f}/"
            f"{r.get('scout', {}).get('protected', 0):<2d} "
            f"div={100.0 * r.get('diverged_rollouts', 0) / max(r.get('rollouts', 0), 1):4.1f}% "
            f"mf={r.get('mission_best', 0.0):.3f} "
            f"corr={r.get('mission_corr', 0.0):+.2f} "
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
    if getattr(args, "postrun", True):
        launch_postrun(cfg.run_dir)

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


def child_env() -> dict:
    """The environment for a child Python: ``os.environ``, passed explicitly.

    Importing the Mojo kernel (``full_pipeline``) calls C ``setenv`` for
    ``PYTHONEXECUTABLE`` (a pyenv shim), ``PYTHONPATH=":"`` and
    ``MOJO_PYTHON_LIBRARY``.  ``os.environ`` never sees them, but a child
    started with ``env=None`` inherits the C environment, and
    ``PYTHONEXECUTABLE`` turns ``sys.executable`` into the system interpreter
    with no venv.  Every automatic post-run since at least arch47 died on
    ``import numpy`` that way.
    """
    import os

    env = dict(os.environ)
    for k in ("PYTHONEXECUTABLE", "MOJO_PYTHON_LIBRARY"):
        env.pop(k, None)
    return env


def launch_postrun(run_dir, *, timeout: float = 3600.0) -> int:
    """Run ``postrun`` for a finished run, in its own process.

    Its own process so that a renderer that cannot get a GL context, or a
    report that throws, cannot take the search's exit status with it -- the
    run is finished and checkpointed by the time this is called.  Blocking, so
    that "the search process has exited" means "the report and the films
    exist", which is what the watcher reads.
    """
    import subprocess
    import sys

    log = Path(run_dir) / "postrun.log"
    print(f"\npost-run: report and films -> {log}", flush=True)
    try:
        with open(log, "w") as f:
            rc = subprocess.run([sys.executable, "-m", "dytiscidae.ops.run", "postrun",
                                 "--run", str(run_dir)], stdout=f, stderr=subprocess.STDOUT,
                                env=child_env(),
                                timeout=timeout).returncode
    except Exception as exc:                                  # noqa: BLE001
        print(f"  post-run failed to start: {exc}", flush=True)
        return 1
    print(f"  post-run exited {rc}", flush=True)
    return rc


def cmd_postrun(args) -> int:
    """Everything a finished run should leave behind, in one call.

    1. The chart report (`.claude/skills/training-report/report.py`).
    2. The films (`viz/film.py`): each medium filmed *as it was evaluated*,
       stamped with the recorded and the reproduced score, beside the
       continuous mission -- which is never shown alone, because on
       2026-09-21 it was, and every judgement drawn from it was wrong.

    ``ops.run search`` calls this when a run finishes, and so does the job
    layer's search trainer, so nobody has to remember to.
    """
    import subprocess
    import sys

    run_dir = Path(args.run)
    report = Path(__file__).resolve().parents[2] / ".claude" / "skills" / "training-report" / "report.py"
    rc = 0
    if report.exists():
        print("== report ==", flush=True)
        rc |= subprocess.run([sys.executable, str(report), str(run_dir)],
                             env=child_env()).returncode
    print("== films ==", flush=True)
    from ..viz.film import film_run

    manifest = film_run(run_dir, by=args.by)
    if manifest:
        # Once more, now that the manifest exists, so the report embeds it.
        if report.exists():
            subprocess.run([sys.executable, str(report), str(run_dir)],
                           stdout=subprocess.DEVNULL, env=child_env())
    return rc if manifest else 1


def cmd_film(args) -> int:
    from ..viz.film import film_run

    m = film_run(args.run, by=args.by, island=args.island)
    return 0 if m else 1


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
        d = basis.diagnostics()
        print(f"  discovered axes ({medium}), control rank {d['control_rank']}"
              f" of {d['numerical_rank']} numerical, cond {d['condition']:.1f}"
              f", {d['n_probes']} probes for {d['n_params']} parameters"
              f"{' -- UNDERDETERMINED' if d['underdetermined'] else ''}:")
        for line in basis.describe()[: basis.control_rank or 1]:
            print(f"      {line}")

    with open(out / f"{stem}_controller.pkl", "wb") as f:
        pickle.dump({"weights": result.policy_weights,
                     "hidden": getattr(result, "policy_hidden", 0),
                     "n_modes": getattr(result, "n_modes", None),
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


class _Cached(Exception):
    """Control flow only: the run's shared network is already loaded."""


#: The shared network of a run, by run directory.  Rebuilding it costs a torch
#: import and a checkpoint read, and a probe over an archive asks for it once per
#: elite.
_SHARED_NETWORK_CACHE: dict = {}


def controller_for_elite(design_dir, elite, p, seed: int, *, log=print):
    """The control law an archived elite's scores were earned under, rebuilt.

    Lifted out of ``cmd_showcase`` unchanged so that anything re-measuring an
    elite drives it the way the search did -- its own stored policy, the run's
    shared network, and the mobility basis it was scored against -- rather than
    the pattern generator open-loop, which is a different experiment.  Returns
    None, having said so, when neither policy is recoverable.
    """
    from ..control.cpg import Policy  # noqa: F401  (kept for parity with the original)
    from ..envs.evaluate import Controller
    from ..envs.triphibian import TriphibianEnv

    import pickle as _pickle

    from ..control.cpg import Policy as _Policy
    from ..envs.evaluate import SharedController, SummedPolicy
    from ..envs.triphibian import Domain as _Dom

    import numpy as _np

    from ..control.cpg import MobilityBasis as _Basis

    env0 = TriphibianEnv(p, seed=seed)
    own = None
    w = (elite.meta or {}).get("policy")
    # Either width: a run with the gait-gain channel stores one more output.
    for _gain in (False, True):
        pol = _Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=6, hidden=0, gain=_gain)
        if w is not None and len(w) == pol.n_weights:
            pol.weights = _np.asarray(w, float).copy()
            own = pol
            break

    # The basis the score was earned against, if the run recorded it.
    #
    # Re-identifying is not the same experiment: `env.identify` is seeded,
    # and until `eval_seed` was recorded there was no way to ask for the
    # draw the score used.  So prefer what is stored and fall back to a
    # fresh identification, saying which happened.
    basis_src = "stored"
    stored = (elite.meta or {}).get("mobility_basis") or {}
    bases = _Basis.bases_from_record(stored)
    for dom_name in sorted(set(stored) - set(bases)):
        log(f"  stored basis for {dom_name} unusable")
    if not bases:
        basis_src = "re-identified"
        seed = (elite.meta or {}).get("eval_seed")
        seed = int(seed) if isinstance(seed, (int, float)) else seed
        for dom in (_Dom.AIR, _Dom.WATER):
            try:
                bases[dom.value] = env0.identify(dom, seed=seed, max_modes=6)
            except Exception as exc:                          # noqa: BLE001
                log(f"  mobility identification failed in "
                      f"{dom.value}: {exc}")
    shared = _SHARED_NETWORK_CACHE.get(str(design_dir))

    # The portable checkpoint first: it carries the network as arrays and
    # needs no project class unpickled.  `search_state.pkl` is the fallback
    # for runs made before it existed.
    try:
        if shared is not None:
            raise _Cached
        from . import checkpoint as _ckmod
        _ck = _ckmod.read(design_dir)
        shared, _miss, _unex = _ckmod.load_network(_ck)
        log(f"  network from {_ck.path.name} "
              f"(gen {_ck.generation}, git "
              f"{(_ck.meta.get('provenance') or {}).get('git', '')[:8]})")
    except _Cached:
        pass
    except Exception:                                         # noqa: BLE001
        shared = None
    sp = Path(design_dir) / "search_state.pkl"
    if shared is None and sp.exists():
        try:
            d = _pickle.load(open(sp, "rb"))
            sd, shape = d.get("shared_state"), d.get("shared_shape")
            if sd and shape:
                import torch as _torch

                from ..learning import ppo as _ppo
                # Inferred from the stored tensors, not assumed from a flag:
                # a width that does not match loads into the wrong shape, and
                # `--shared-hidden` is not even an argument of this
                # subcommand.  The first 2-D weight's row count is the trunk
                # width the run actually used.
                hid = next((int(v.shape[0]) for _k, v in sd.items()
                            if getattr(v, "ndim", 0) == 2), 64)
                shared = _ppo.SharedPolicy(int(shape[0]), int(shape[1]),
                                           hidden=hid)
                shared.load_state_dict(
                    {k: _torch.as_tensor(v) for k, v in sd.items()})
                shared.eval()
        except Exception as exc:                              # noqa: BLE001
            log(f"  shared policy not recoverable: {exc}")
    if shared is not None:
        _SHARED_NETWORK_CACHE[str(design_dir)] = shared
    controller = None
    if own is not None or shared is not None:
        if shared is not None:
            controller = SharedController(
                params=env0.cpg.base, bases=bases,
                policy=SummedPolicy(own=own, shared=shared, n_modes=6))
        else:
            controller = Controller(params=env0.cpg.base, policy=own,
                                    bases=bases)
        log(f"  driving as evaluated: own policy "
              f"{'yes' if own is not None else 'NO'}, shared policy "
              f"{'yes' if shared is not None else 'NO'}, bases "
              f"{sorted(bases)} ({basis_src})")
    else:
        log("  WARNING: neither a stored policy nor a shared network was "
              "recoverable; this film is the pattern generator open-loop "
              "and is not evidence about the design")
    return controller


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
    if args.design and not args.controller and not args.train:
        # A design from a run is filmed as it was scored, beside the continuous
        # mission -- never the mission alone (ROADMAP item Y; `viz/film.py`).
        from ..viz.film import film_run

        m = film_run(args.design, by=args.by, island=args.island,
                     out_dir=Path(args.run))
        return 0 if m else 1
    if args.design:
        archive, islands = load_run_archive(args.design, island=args.island)
        if archive is None:
            print(f"no archive in {args.design}"
                  + (f" for island {args.island!r} (have: {', '.join(islands)})"
                     if args.island and islands else ""))
            return 1
        pool = [e for e in archive.cells.values()
                if not args.island or (e.meta or {}).get("island") == args.island]
        if not pool:
            print(f"no elites in {args.design}"
                  + (f" from island {args.island!r} (have: {', '.join(islands)})"
                     if args.island else ""))
            return 1
        # "Best" is ambiguous and the two answers disagree.  Fitness is what the
        # curator ranks by; `mission_fraction` is what the machine achieves on
        # the mission, and arch34 ended with corr(fitness, mission) = +0.70 --
        # close, and not close enough.  Filming the fitness-best elite showed a
        # machine that walked and never left the ground while the run's actual
        # best flew at a 95% airborne fraction, which is how this option came
        # to exist.
        if args.by == "mission":
            key = lambda e: (e.meta or {}).get("mission_fraction") or 0.0
        elif args.by == "island":
            # Best at the island's own domains, which fitness is not: stage 0
            # of the curriculum reads the best medium anywhere, so an island's
            # fitness champion can be a specialist in someone else's medium.
            from ..evolution.islands import own_domain_score

            def key(e):
                meta = e.meta or {}
                return own_domain_score(args.island or meta.get("island", ""), meta)
        else:
            key = lambda e: e.fitness
        elite = max(pool, key=key)
        p = build(elite.genome)
        print(f"filming the best of {len(pool)} elites from {args.design}"
              f"{f' (island {args.island})' if args.island else ''} by {args.by}: "
              f"fitness {elite.fitness:.4f}, mission_fraction "
              f"{(elite.meta or {}).get('mission_fraction') or 0:.4f}, island "
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
    # The control law the elite's scores were earned under, rebuilt from the run.
    #
    # `run_continuous` (mission.py) reads
    # `params = controller.params if controller is not None else env.cpg.base`
    # and `basis = ... else None`, so a showcase with no controller drove the
    # machine **open-loop from the raw pattern generator, with no policy and no
    # mobility basis**.  That is the failure ROADMAP finding 3 is about, and it
    # is why five runs of footage showed machines that do not move: the filmed
    # machine was not the machine that earned the score.  The elite filmed for
    # arch38 records `land_speed` 0.788 m/s, `takeoff_height` 2.288 m and
    # `max_depth` 9.60 m, and its film reported 0.0 m of depth and no motion.
    #
    # `SummedPolicy`'s own docstring names the showcase as one of the paths that
    # needs it.  The pieces were all present and nothing wired them together:
    # the elite's stored 168-weight policy, the run's shared network from
    # `search_state.pkl`, and a mobility basis identified on this body the way
    # `evaluate_tier1` does it.
    if args.design and not args.controller and not args.train:
        controller = controller_for_elite(args.design, elite, p, args.seed)

    if args.controller and Path(args.controller).exists():
        d = pickle.load(open(args.controller, "rb"))
        # Width comes from the pickle: a controller trained before the default
        # changed must still load with the shape it was trained at.  The mode
        # count said that in a comment and then hard-coded 4, so a controller
        # trained at any other width loaded into the wrong shape.
        pol = Policy(n_obs=TriphibianEnv.OBS_DIM,
                     n_modes=int(d.get("n_modes") or 4),
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
    """The top elites, each filmed as it was evaluated.

    This used to drive every elite open-loop -- no policy, no basis -- at seed
    0, without the evaluation's scatter, for 10 s, and print no score, so its
    clips were a different experiment from the numbers beside them and nothing
    on screen said so.  Now each is ``viz/film.py``'s evaluated film, with the
    recorded and reproduced score on every clip.
    """
    from ..viz.film import MEDIA, evaluate_on_film, _banner, _verdict, _write

    archive, islands = load_run_archive(args.run)
    if archive is None:
        print(f"no archive in {args.run}")
        return 1
    elites = sorted(archive.cells.values(),
                    key=lambda e: -((e.meta or {}).get("mission_fraction") or 0.0))[:args.top]
    out = Path(args.run) / "media"
    for rank, e in enumerate(elites):
        ev = evaluate_on_film(e, args.run)
        for m in MEDIA:
            r = ev["media"][m]
            text, colour = _verdict(r)
            frames = [_banner(f, [f"SCORED -- elite {rank}, {m}, as evaluated", text,
                                  f"control: {ev['control_law']}"], colour, 640)
                      for f in r["frames"]]
            path = _write(frames, out / f"elite{rank}_{m}.mp4", 25)
            print(f"  elite{rank} {m:5s} {text}  {path or '(no footage)'}")
    return 0


def cmd_distill(args) -> int:
    """Is a shared controller reachable at all?  Answered from stored policies."""
    from ..learning.distill import distil

    res = distil(args.run, n_states=args.states, hidden=args.hidden,
                 epochs=args.epochs, held_out=args.held_out, seed=args.seed,
                 refined_only=args.refined_only)
    print(f"distillation study of {args.run}")
    print("-" * 60)
    print("\n".join(res.lines()))
    if res.per_body:
        print("\nworst held-out bodies (name, R2, tier-1 mission fraction):")
        for row in res.per_body[:5]:
            print(f"  {row[0]:<28} {row[1]:+.3f}  {row[2]:.4f}")
    return 0 if res.scores else 1


def build_parser() -> argparse.ArgumentParser:
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
    p.add_argument("--batch", type=int, default=16,
                   help="designs per generation. 16 is what arch34-arch48 ran "
                        "and what the pool shape was swept at (4x4 31.7 s, "
                        "8x2 queue 0.875x on rotor-heavy bodies)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                   help="worker processes stepping machines in parallel; 4 "
                        "(capped at the core count) because worker count is "
                        "bounded by memory before cores. 1 is the "
                        "single-process path. "
                        "Measured: 16 workers of 8 machines return 12.7x the "
                        "throughput of one. A shard smaller than --min-shard "
                        "is not worth a batch's fixed cost, so the pool never "
                        "makes more than --batch // --min-shard of them: to "
                        "use more cores, raise --batch.")
    p.add_argument("--pool-per-worker", type=float, default=2.0,
                   help="shards per worker; above 1 the pool becomes a queue "
                        "that hands the next shard to whichever worker is free "
                        "(ROADMAP AJ). 2 since 2026-10-03: 0.875x the wall of "
                        "4x4 on rotor-heavy batches; 1 is the pool every run "
                        "up to arch46 used")
    p.add_argument("--distance-curriculum", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="on by default since 2026-10-05 (--no-distance-curriculum "
                        "restores the old probes); step transition starts back from their interfaces and "
                        "the air launch down from 30 m as the population learns "
                        "(ROADMAP Y/O); changes what transition and air scores mean")
    p.add_argument("--refine-funnel", type=float, default=None,
                   help="refine only candidates within this fraction of their "
                        "cell's incumbent (ROADMAP AK); unset refines all")
    p.add_argument("--pool-balance", action="store_true",
                   help="assign machines to shards by predicted cost (rotors, "
                        "envs.actors.shard_cost) "
                        "rather than by position (ROADMAP AJ)")
    p.add_argument("--min-shard", type=int, default=2,
                   help="fewest machines a worker is given at once. Measured "
                        "per machine-step: 238 us alone, 149 in a shard of "
                        "four, 105 in eight, 89 in sixteen -- so a smaller "
                        "shard spends the parallelism it gains. 4 was the swept "
                        "optimum on random bodies; 2 with --pool-per-worker 2 "
                        "is the optimum on rotor-heavy ones; 8 collapses the "
                        "pool to one shard "
                        "as soon as one machine is rejected at batch 16")
    p.add_argument("--segment-seconds", type=float, default=8.0,
                   help="Tier-1 episode length; the main cost/fidelity dial")
    p.add_argument("--refine-steps", type=int, default=0,
                   help="(1+1)-ES steps refining each candidate's own policy "
                        "on top of the shared one. Default 0 since 2026-10-08 "
                        "(ROADMAP M1): on arch48's elites an accepted step kept "
                        "8-23%% of its gain at a fresh draw and was not "
                        "separable from a random perturbation. Runs arch34-"
                        "arch47 passed 2, arch48 passed 1. The noise-free "
                        "re-score still runs at 0 whenever the shared policy "
                        "is on. Each step is one more batched Tier-1 for the "
                        "whole generation. Without the shared policy, 0 leaves "
                        "a fresh candidate's zero weights commanding nothing.")
    p.add_argument("--refine-sigma", type=float, default=0.1,
                   help="perturbation scale on policy weights during refinement")
    p.add_argument("--promotion-refine-steps", type=int, default=6,
                   help="(1+1)-ES steps spent on an elite when it is promoted "
                        "to Tier-2. Bounded by promotions (at most three per "
                        "verification round) rather than by population, so "
                        "unlike --refine-steps it is affordable by default. 0 "
                        "verifies the elite exactly as the archive stored it.")
    p.add_argument("--shared-policy", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="on by default when torch imports (what every run since "
                        "arch34 used; its worth is unmeasured, ROADMAP E1), "
                        "--no-shared-policy turns it off. "
                        "train one PPO policy across every morphology, on the "
                        "transitions the whole generation produces. Coexists "
                        "with --refine-steps: the shared policy generalises, "
                        "the per-candidate (1+1)-ES adapts, the intents are "
                        "summed. Needs torch.")
    p.add_argument("--shared-hidden", type=int, default=64,
                   help="width of the shared PPO policy")
    p.add_argument("--shared-lr", type=float, default=1e-3,
                   help="PPO learning rate. arch30 ran at 3e-4 and reported a "
                        "mean KL of 0.0008 against the 0.01-0.02 an update "
                        "normally aims for -- the policy was barely asked to "
                        "move.")
    p.add_argument("--shared-epochs", type=int, default=10,
                   help="PPO passes over each generation's batch")
    p.add_argument("--shared-target-kl", type=float, default=0.015,
                   help="stop an update once its mean KL exceeds this")
    p.add_argument("--shared-ent-coef", type=float, default=0.01,
                   help="entropy bonus of the shared policy (ROADMAP R). Until "
                        "2026-10-03 there was no flag and direct runs were "
                        "always 0.01; exploration is not comparable across a "
                        "change of it")
    p.add_argument("--shared-learner", default="ppo",
                   choices=("ppo", "grpo", "ppo+grpo"),
                   help="estimator for the shared policy (ROADMAP N). ppo is "
                        "every run to date; ppo+grpo adds --grpo-bodies x "
                        "--grpo-group learning-only rollouts per generation "
                        "with group-relative advantages; grpo trains on those "
                        "alone. Needs --shared-policy. Off by default: built "
                        "2026-10-03, to be switched on only if the shared "
                        "policy is measured to carry weight (item R)")
    p.add_argument("--grpo-group", type=int, default=4,
                   help="rollouts of one body per group (>= 2)")
    p.add_argument("--grpo-bodies", type=int, default=4,
                   help="bodies given a group each generation")
    p.add_argument("--policy-hidden", type=int, default=0,
                   help="hidden units in the policy; 0 is linear (60 weights), "
                        "16 is 308")
    p.add_argument("--run", default="runs/latest")
    p.add_argument("--tier2-every", type=int, default=5)
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
    p.add_argument("--memory-ceiling-mb", type=int, default=0,
                   help="stop cleanly and checkpoint if the search's own "
                        "resident set (parent plus workers) passes this. "
                        "0 disables. arch35 lost 405 generations to an OOM "
                        "kill it could have resumed from")
    p.add_argument("--resume", action="store_true",
                   help="continue the run in --run instead of starting over")
    p.add_argument("--no-scout", action="store_true",
                   help="run without the potential predictor (greedy selection)")
    p.add_argument("--scout-reserve", type=float, default=0.15,
                   help="share of each archive protected on predicted potential")
    p.add_argument("--mission-weight", type=float, default=0.30,
                   help="share of the archive's scalar that is the mission "
                        "itself, as a population quantile, rather than the "
                        "island/curriculum blend. At 0 -- every run before "
                        "2026-09-01 -- corr(fitness, mission_fraction) "
                        "measured 0.159 and the search spent 36%% of its energy "
                        "fraction buying 32%% more competence.")
    p.add_argument("--descriptor-bins", type=int, default=5,
                   help="bins per learned archive axis. Four axes at 8 bins is "
                        "4096 cells per island, 32,768 across eight, against "
                        "arch31's entire budget of 9,620 evaluations -- so "
                        "81.9%% of cells were never improved on. Size the map "
                        "to the budget.")
    p.add_argument("--reward-shaping", type=float, default=0.2,
                   help="weight on potential-based shaping in the PPO reward. "
                        "0 is the terminal-only reward, which delivered one "
                        "scalar per 161 decisions. Shaping of this form cannot "
                        "change the optimal policy.")
    p.add_argument("--n-modes", type=int, default=6,
                   help="mobility modes identified per body per domain, and "
                        "the width the shared policy commands through")
    p.add_argument("--descriptor-keep-if-overlap", type=float, default=0.95,
                   help="keep the learned axes when a refit would reproduce "
                        "them (subspace overlap at or above this); 0 = off, "
                        "which every run through arch45 used (ARCH46_SPEC §3)")
    p.add_argument("--no-tier2-label-all-media", action="store_true",
                   help="do not run the label-only Tier-2 legs in media the "
                        "mission never reached (ARCH46_SPEC §2b)")
    p.add_argument("--gait-gain", action="store_true",
                   help="give both policies a gait-gain output, a factor on the "
                        "commanded amplitude, so stopping and throttling are "
                        "reachable (control.cpg.GAIN_RANGE); off by default")
    p.add_argument("--action-rate-penalty", type=float, default=0.0,
                   help="weight of the command-rate penalty on competence "
                        "(ROADMAP item Y); 0 = off")
    p.add_argument("--no-postrun", dest="postrun", action="store_false",
                   help="skip the report and films a finished run makes of "
                        "itself (see `postrun`)")
    p.set_defaults(fn=cmd_search, postrun=True)

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
    p.add_argument("--by", choices=("fitness", "mission", "island"), default="mission",
                   help="which elite counts as best: the curator's fitness "
                        "ranking, the mission fraction the machine achieved, or "
                        "competence at the island's own domains (use with "
                        "--island; an island's fitness champion can be a "
                        "specialist in another medium)")
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

    p = sub.add_parser("film", help="film a run's best elite as it was scored, "
                                    "beside the continuous mission")
    p.add_argument("--run", required=True)
    p.add_argument("--by", choices=("fitness", "mission", "island"), default="mission")
    p.add_argument("--island", default=None)
    p.set_defaults(fn=cmd_film)

    p = sub.add_parser("postrun", help="the report and films a finished run "
                                       "leaves behind (run automatically)")
    p.add_argument("--run", required=True)
    p.add_argument("--by", choices=("fitness", "mission", "island"), default="mission")
    p.set_defaults(fn=cmd_postrun)

    p = sub.add_parser("dashboard", help="regenerate the dashboard for a run")
    p.add_argument("--run", default="runs/latest")
    p.set_defaults(fn=cmd_dashboard)

    p = sub.add_parser("render", help="render archive elites to video")
    p.add_argument("--run", default="runs/latest")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--duration", type=float, default=8.0)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser(
        "distill",
        help="can one conditioned network match the per-body controllers?")
    p.add_argument("--run", default="runs/latest")
    p.add_argument("--states", type=int, default=256,
                   help="observations sampled per teacher")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--held-out", type=float, default=0.3, dest="held_out",
                   help="fraction of *bodies* the student never sees")
    p.add_argument("--refined-only", action="store_true", dest="refined_only",
                   help="only policies a promotion actually fitted to their body")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_distill)

    # The job/experiment command group lives in ``adapters/cli.py``, which is a
    # driving adapter over the application layer.  It is registered here rather
    # than written here so that this module stays what it has always been --
    # the direct commands -- and so that the new ones stay free of any decision
    # the use cases should be making.
    p = sub.add_parser(
        "checkpoints", help="list a job's checkpoints, newest step last")
    p.add_argument("job_id")
    p.add_argument("--root", default=None)
    p.add_argument("--store", default=None, choices=("files", "sqlite"))
    p.add_argument("--verify", action="store_true",
                   help="fetch each one and check it against its digest")
    from ..adapters import cli as _cli
    p.set_defaults(fn=_cli.cmd_checkpoints)
    _cli.build_parsers(sub)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
