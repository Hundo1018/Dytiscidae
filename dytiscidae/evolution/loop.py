"""The co-evolution loop: morphology and control, curated.

One generation is:

    1. the curator classifies the run and sets search pressure
    2. it selects a parent from the archive, weighting fitness, curiosity,
       frontier position and recency
    3. it selects mutation operators by bandit
    4. the child is built, gated at Tier 0, and evaluated at Tier 1
    5. the child's controller is inherited and locally refined
    6. the outcome is credited back to the operators and the parent
    7. elites are promoted to Tier 2 within budget, exploits are quarantined,
       and crowded regions are thinned

Morphology and control co-evolve rather than being optimised in sequence,
because they are not separable: the mobility basis a controller commands is a
property of the *body*, so a controller is only meaningful relative to the
morphology it was measured on.  Inheriting the parent's controller and refining
it is what makes that affordable -- a child body is usually similar enough to
its parent that the parent's policy is a good starting point, and re-identifying
the axes from scratch every time would cost more than the evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..control.cpg import CPGParams, Policy
from ..core.genome import Genome, crossover, mutate, random_genome
from ..core.phenotype import build
from ..core.bodyplans import seed_population
from ..envs.evaluate import (
    BD_AXES,
    OBJECTIVE_NAMES,
    Controller,
    behaviour_descriptor,
    evaluate_tier1,
    evaluate_tier2,
    fitness,
    objectives,
)
from ..envs.triphibian import MissionSpec, TriphibianEnv, evaluate_tier0
from ..ops.telemetry import Telemetry
from .archive import Archive
from .cmaes import CMAES, Emitter
from .curator import Curator
from .descriptors import LearnedDescriptors, episode_features


@dataclass
class SearchConfig:
    """Everything adjustable about a run.

    The defaults are tuned for a four-core CPU with no GPU, which is the
    environment this was developed in.  ``segment_seconds`` is the main
    cost/fidelity dial: halving it roughly halves the run time and roughly
    doubles the variance of every Tier-1 score.
    """

    generations: int = 200
    batch: int = 4  # candidates per generation
    seed: int = 0

    # Fidelity
    segment_seconds: float = 8.0
    identify_axes_every: int = 1  # re-identify a child's axes this often
    tier0_gate: float = -0.85  # reject below this structural margin
    tier2_every: int = 15

    # Controller refinement
    controller_refine_steps: int = 0  # CMA-ES iterations per child (0 = inherit only)
    policy_hidden: int = 16
    n_modes: int = 4

    # Seeding
    n_reference_seeds: int = 20
    n_random_seeds: int = 8

    # Output
    run_dir: str = "runs/latest"
    checkpoint_every: int = 20

    #: Learn the archive's axes from behaviour instead of using the hand-picked
    #: four.  ``BD_AXES`` was a list I wrote -- log mass, density ratio, air
    #: competence, water competence -- and each entry silently decided what the
    #: search would consider a different *kind* of machine.  Two designs that
    #: differ in a way none of those axes captures collide in one cell and one
    #: is thrown away, so the axes bound what can be found in exactly the way a
    #: fixed part taxonomy bounds what can be built.
    #:
    #: With this on, a generic 16-feature behaviour vector is recorded from
    #: every episode and a projection of it is fitted from the run's own data
    #: (AURORA, Cully 2019).  The system decides what behavioural difference
    #: means, and the definition moves as the population moves.
    learned_axes: bool = True
    descriptor_refit_every: int = 400
    event_sample: int = 1


@dataclass
class SearchState:
    archive: Archive
    curator: Curator
    telemetry: Telemetry
    config: SearchConfig
    rng: np.random.Generator
    emitters: list[Emitter] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    evaluated: int = 0
    tier0_rejected: int = 0
    descriptors: LearnedDescriptors | None = None


# --------------------------------------------------------------------------


def _controller_for(pheno, genome: Genome, cfg: SearchConfig, inherited=None):
    """Build a controller, reusing the parent's weights when the shape allows.

    A mutation that adds a joint changes the CPG parameter count, so inherited
    weights only transfer when the policy's input and output shape is unchanged.
    When it is not, the child starts from zeros -- which with a tanh output means
    "command nothing", i.e. fall back to the raw pattern generator rather than to
    random flailing.
    """
    policy = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=cfg.n_modes, hidden=cfg.policy_hidden)
    if inherited is not None and len(inherited) == policy.n_weights:
        policy.weights = np.asarray(inherited, float).copy()
    return policy


def evaluate_candidate(
    genome: Genome,
    cfg: SearchConfig,
    *,
    inherited_policy=None,
    identify: bool = True,
    spec: MissionSpec | None = None,
    seed: int = 0,
):
    """Tier-0 gate then Tier-1.  Returns ``(phenotype, result, controller)``."""
    pheno = build(genome)
    t0 = evaluate_tier0(pheno, spec)
    if pheno.report.min_margin < cfg.tier0_gate or t0.mission_fraction <= 0.0:
        return pheno, t0, None

    policy = _controller_for(pheno, genome, cfg, inherited_policy)
    ctrl = Controller(params=None, policy=policy)  # params filled by the env
    result = evaluate_tier1(
        pheno,
        spec=spec,
        controller=None if identify else ctrl,
        segment_seconds=cfg.segment_seconds,
        identify_axes=identify,
        seed=seed,
    )
    return pheno, result, ctrl


def seed_archive(state: SearchState, spec: MissionSpec) -> None:
    """Populate the archive with the reference design and its neighbourhood."""
    cfg = state.config
    # Round-robin over every archetype, then random graphs.  Seeding only from
    # one plan is what produced an archive of nothing but bilateral flappers:
    # body plans are separated by valleys that mutation does not cross, so an
    # unseeded plan is not merely rare, it is unreachable.
    seeds: list[Genome] = list(seed_population(state.rng, cfg.n_reference_seeds))
    seeds += [random_genome(state.rng) for _ in range(cfg.n_random_seeds)]

    for i, g in enumerate(seeds):
        g.genome_id = f"seed{i}"
        pheno, result, ctrl = evaluate_candidate(
            g, cfg, identify=True, spec=spec, seed=int(state.rng.integers(1 << 30))
        )
        state.evaluated += 1
        state.curator.evaluations += 1
        _place(state, g, pheno, result, ctrl, parent=None, operators=["seed"])


def _meta(pheno, result, ctrl) -> dict:
    seg = result.segments
    return {
        "mass": round(pheno.mass, 3),
        "span": round(pheno.max_span, 3),
        "wing_area": round(pheno.wing_area, 4),
        "aspect_ratio": round(pheno.aspect_ratio, 2),
        "wing_loading": round(pheno.wing_loading, 1),
        "density_ratio": round(pheno.density_ratio, 3),
        "battery_wh": round(pheno.genome.battery_wh, 1),
        "flap_hz": round(pheno.genome.flap_frequency, 2),
        "dof": pheno.n_actuated,
        "body_plan": (pheno.genome.lineage[0] if pheno.genome.lineage else "?"),
        "feasible": bool(pheno.report.ok),
        "margin": round(pheno.report.min_margin, 3),
        "worst_check": pheno.report.worst.name if pheno.report.worst else "",
        "mission_fraction": round(result.mission_fraction, 4),
        "energy_margin": round(result.energy_margin, 3),
        "tier": result.tier,
        "air": round(seg["air"].competence, 3) if "air" in seg else 0.0,
        "water": round(seg["water"].competence, 3) if "water" in seg else 0.0,
        "land": round(seg["land"].competence, 3) if "land" in seg else 0.0,
        "max_depth": round(max((s.max_depth for s in seg.values()), default=0.0), 2),
        "mobility_rank": {k: v.rank for k, v in result.mobility.items()},
        "mobility_axes": {k: v.describe() for k, v in result.mobility.items()},
        "policy": (ctrl.policy.weights.tolist()
                   if ctrl is not None and ctrl.policy is not None else None),
    }


def _place(state: SearchState, genome, pheno, result, ctrl, parent, operators) -> str:
    """Score a finished candidate, file it, and credit everyone involved."""
    cfg = state.config
    if result.tier == 0:
        state.tier0_rejected += 1
        state.telemetry.event(
            {"kind": "tier0_reject", "gen": state.archive.generation,
             "margin": round(pheno.report.min_margin, 3),
             "worst": pheno.report.worst.name if pheno.report.worst else "",
             "operators": operators}
        )
        state.curator.credit(operators, "rejected", 0.0)
        state.curator.note_offspring(parent, "rejected")
        return "rejected"

    feats = episode_features(result, pheno)
    if state.descriptors is not None:
        state.descriptors.observe(feats)
    # The hand-picked descriptor stays in use until the projection has actually
    # been fitted.  An unfitted projector returns the first four raw features,
    # which are time fractions and a speed -- nothing like the ranges the
    # archive was built with -- so switching before the fit would file every
    # early design into a corner of the grid for no reason.
    if state.descriptors is not None and state.descriptors.fitted:
        bd = state.descriptors.project(feats)
    else:
        bd = behaviour_descriptor(pheno, result)
    fit = fitness(pheno, result)
    obj = objectives(pheno, result)

    if result.exploit:
        state.curator.quarantine(bd, result.exploit, genome)
        state.telemetry.exploit(
            {"gen": state.archive.generation, "reason": result.exploit,
             "descriptor": bd, "meta": _meta(pheno, result, ctrl)}
        )
        state.curator.credit(operators, "rejected", 0.0)
        state.curator.note_offspring(parent, "rejected")
        return "exploit"

    cell = state.archive.cell_of(bd)
    previous = state.archive.cells[cell].fitness if cell in state.archive.cells else 0.0
    meta = _meta(pheno, result, ctrl)
    # The raw feature vector travels with the elite.  A learned projection moves,
    # and re-binning has to re-project from the features rather than from a
    # latent coordinate that no longer means the same thing.
    meta["features"] = [float(x) for x in feats]
    meta["objectives"] = {n: round(float(v), 4) for n, v in zip(OBJECTIVE_NAMES, obj)}
    status = state.archive.add(genome, fit, bd, meta, tier=result.tier, objectives=obj)

    state.curator.observe_domains(meta)
    state.curator.credit(operators, status, fit - previous)
    state.curator.note_offspring(parent, status)
    state.telemetry.event(
        {"kind": "evaluate", "gen": state.archive.generation, "status": status,
         "fitness": round(fit, 4), "cell": list(cell), "operators": operators,
         "wall": round(result.wall_time, 2), "notes": result.notes[:3],
         **{k: v for k, v in meta.items() if k not in ("policy", "features")}}
    )
    return status


# --------------------------------------------------------------------------


def run_search(cfg: SearchConfig, spec: MissionSpec | None = None,
               on_generation=None) -> SearchState:
    """Run the whole loop.  Returns the final state, checkpointed as it goes."""
    spec = spec or MissionSpec()
    rng = np.random.default_rng(cfg.seed)
    archive = Archive(BD_AXES)
    curator = Curator(archive, seed=cfg.seed)
    telemetry = Telemetry(cfg.run_dir, event_sample=cfg.event_sample)
    learned = (
        LearnedDescriptors(n_dims=len(BD_AXES), refit_every=cfg.descriptor_refit_every)
        if cfg.learned_axes else None
    )
    state = SearchState(archive=archive, curator=curator, telemetry=telemetry,
                        config=cfg, rng=rng, descriptors=learned)

    telemetry.write("generations", {"kind": "run_start", "config": cfg.__dict__,
                                    "spec": spec.__dict__})
    seed_archive(state, spec)

    for gen in range(cfg.generations):
        archive.generation = gen
        regime = curator.update_regime()

        for _ in range(cfg.batch):
            parent = curator.select_parent()
            operators = curator.choose_operators()

            if parent is None:
                child = random_genome(rng)
                operators = ["random"]
                inherited = None
            else:
                child, operators = mutate(parent.genome, rng, operators=operators,
                                          n_ops=regime.n_mutations)
                inherited = parent.meta.get("policy")
                # Occasional crossover with a second archive member: transfers
                # energy strategy and surface shape without scrambling a working
                # kinematic tree.
                if rng.random() < 0.15:
                    other = curator.select_parent()
                    if other is not None and other is not parent:
                        child = crossover(child, other.genome, rng)
                        operators = operators + ["crossover"]

            child.generation = gen
            child.genome_id = f"g{gen}_{state.evaluated}"
            identify = (state.evaluated % max(cfg.identify_axes_every, 1)) == 0

            try:
                pheno, result, ctrl = evaluate_candidate(
                    child, cfg, inherited_policy=inherited, identify=identify,
                    spec=spec, seed=int(rng.integers(1 << 30))
                )
            except Exception as exc:
                telemetry.event({"kind": "error", "gen": gen, "error": f"{type(exc).__name__}: {exc}",
                                 "operators": operators})
                curator.credit(operators, "rejected", 0.0)
                continue

            state.evaluated += 1
            curator.evaluations += 1
            _place(state, child, pheno, result, ctrl, parent, operators)

        # --- verification, quarantine, thinning ---------------------------
        if gen % max(cfg.tier2_every, 1) == 0 and archive.cells:
            for elite in sorted(archive.cells.values(), key=lambda e: -e.fitness)[:3]:
                if not curator.should_promote(elite):
                    continue
                try:
                    p2 = build(elite.genome)
                    r2 = evaluate_tier2(p2, spec=spec, seed=int(rng.integers(1 << 30)))
                    f2 = fitness(p2, r2)
                    curator.record_promotion(elite, f2)
                    telemetry.event({"kind": "promote", "gen": gen, "cell": list(elite.cell),
                                     "tier2_fitness": round(f2, 4),
                                     "tier2_fraction": round(r2.mission_fraction, 4),
                                     "exploit": r2.exploit, "notes": r2.notes[:3]})
                    if r2.exploit:
                        curator.quarantine(elite.descriptor, r2.exploit, elite.genome)
                        telemetry.exploit({"gen": gen, "reason": r2.exploit,
                                           "cell": list(elite.cell), "stage": "tier2"})
                except Exception as exc:
                    telemetry.event({"kind": "error", "gen": gen, "stage": "tier2",
                                     "error": f"{type(exc).__name__}: {exc}"})
            curator.prune()

        # --- refit the learned descriptor axes -----------------------------
        # Infrequent on purpose.  A projection that never refits is just a
        # different set of fixed axes; one that refits constantly destroys the
        # archive it is supposed to organise, because every refit re-files every
        # elite and merges any that the new axes call the same.
        if learned is not None and learned.due_for_refit() and learned.fit():
            def _reproject(e, _d=learned):
                f = e.meta.get("features")
                return _d.project(np.asarray(f, float)) if f else None

            axes = [
                (f"latent{i}", float(lo), float(hi), 8)
                for i, (lo, hi) in enumerate(learned.bounds())
            ]
            stats = archive.rebin(axes, _reproject)
            curator.on_rebin()
            telemetry.event({
                "kind": "descriptor_refit", "gen": gen, **stats,
                **learned.report(),
            })

        # --- report -------------------------------------------------------
        report = curator.generation_report()
        report["evaluated"] = state.evaluated
        report["tier0_rejected"] = state.tier0_rejected
        report["elapsed"] = round(time.time() - state.started, 1)
        best = archive.best
        if best is not None:
            report["best"] = {k: v for k, v in best.meta.items()
                              if k not in ("policy", "mobility_axes")}
            report["best_fitness"] = round(best.fitness, 4)
        archive.history.append(archive.snapshot())
        telemetry.generation(report)
        if on_generation is not None:
            on_generation(state, report)

        if gen % max(cfg.checkpoint_every, 1) == 0:
            archive.save(Path(cfg.run_dir) / "archive.pkl")
            archive.export_json(Path(cfg.run_dir) / "archive.json")

    archive.save(Path(cfg.run_dir) / "archive.pkl")
    archive.export_json(Path(cfg.run_dir) / "archive.json")
    telemetry.close()
    return state
