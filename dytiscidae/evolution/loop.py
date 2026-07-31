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
from ..envs.transitions import TransitionSet
from .archive import Archive
from .auditor import Auditor
from .cmaes import CMAES, Emitter
from .critic import Critic, critic_features
from .curator import Curator
from .curriculum import STAGES, Curriculum
from .descriptors import LearnedDescriptors, episode_features
from .islands import ISLANDS, Archipelago, island_score
from .judge import Judge


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

    # --- the archipelago ---------------------------------------------------
    #: Which islands to run.  Each has its own archive, curator and curriculum,
    #: and its own objective; the judge, the auditor and the critic are shared,
    #: because a standard that differs per island is not a standard and a model
    #: of the gap between cheap and expensive evaluation is the same model
    #: wherever the design came from.
    islands: tuple = tuple(ISLANDS)
    migrate_every: int = 60
    n_migrants: int = 2

    # --- the judge ---------------------------------------------------------
    judge_quantile: float = 0.9
    judge_update_every: int = 50

    # --- the critic --------------------------------------------------------
    use_critic: bool = True
    critic_refit_every: int = 40

    # --- the auditor -------------------------------------------------------
    #: Designs audited per review, chosen by the critic's suspicion where it has
    #: one and by fitness where it does not.
    audit_every: int = 30
    audits_per_review: int = 2


@dataclass(eq=False)
class SearchState:
    #: The island currently being worked on.  ``archive`` and ``curator`` are
    #: kept as properties pointing at it so that every existing caller, test and
    #: report keeps working against what is now one island of several.
    telemetry: Telemetry
    config: SearchConfig
    rng: np.random.Generator
    archipelago: Archipelago = None  # type: ignore[assignment]
    curricula: dict = field(default_factory=dict)
    judge: Judge = None  # type: ignore[assignment]
    auditor: Auditor = None  # type: ignore[assignment]
    critic: Critic | None = None
    island: str = "generalist"
    emitters: list[Emitter] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    evaluated: int = 0
    tier0_rejected: int = 0
    descriptors: LearnedDescriptors | None = None
    spec: MissionSpec | None = None
    #: Bars the judge has raised since the last audit review.  Held so the
    #: auditor can veto a tightening that turned out to rest on a design it
    #: subsequently invalidated.
    judge_moves: list = field(default_factory=list)

    @property
    def archive(self) -> Archive:
        return self.archipelago.archives[self.island]

    @property
    def curator(self) -> Curator:
        return self.archipelago.curators[self.island]

    @property
    def curriculum(self) -> Curriculum:
        return self.curricula[self.island]


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


def _meta_light(pheno, result) -> dict:
    """The subset of ``_meta`` the critic reads, without the expensive parts."""
    seg = result.segments
    return {
        "mass": pheno.mass,
        "wing_loading": pheno.wing_loading,
        "aspect_ratio": pheno.aspect_ratio,
        "dof": pheno.n_actuated,
        "air": seg["air"].competence if "air" in seg else 0.0,
        "water": seg["water"].competence if "water" in seg else 0.0,
        "land": seg["land"].competence if "land" in seg else 0.0,
        "max_actuator_overload": max(
            (s.max_actuator_overload for s in seg.values()), default=0.0
        ),
    }


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
    # --- scoring -----------------------------------------------------------
    #
    # Four things now decide what a design is worth, and each does one job.
    #
    #   the judge     turns raw measurements into a rung on a fixed ladder plus
    #                 a fraction of a bar that ratchets, so the standard rises
    #                 with the population without the record becoming unreadable
    #   the island    decides which domains count at all, so a specialist is not
    #                 scored on media it has deliberately given up
    #   the critic    discounts designs whose cheap score it predicts will not
    #                 survive expensive checking; it can only ever subtract
    #   the curriculum decides which question this cell is ready to be asked,
    #                 and supplies a gradient where the mission has none
    judged = {}
    for dom, seg in result.segments.items():
        js = state.judge.score(dom, getattr(seg, "measurements", {}) or {})
        judged[dom] = js
    tset = getattr(result, "transitions", None) or TransitionSet()
    tmeas = tset.component_means()
    tmeas["crossed_fraction"] = tset.crossed_fraction
    judged["transition"] = state.judge.score("transition", tmeas)
    state.judge.observe({
        **{d: (getattr(s, "measurements", {}) or {}) for d, s in result.segments.items()},
        "transition": tmeas,
    })

    cell = state.archive.cell_of(bd)
    base = island_score(state.island, result, tset)
    sr = state.curriculum.evaluate(cell, result, tset)
    # A cell that has been promoted is asked a harder question, and the answer
    # to that question is what it is scored on.  Below the top stage the
    # curriculum's projection replaces the island's, because the island's
    # objective is the *final* question and asking it of a design three stages
    # away is the sparse-reward trap this exists to avoid.
    if sr.stage < 4:
        base = float(0.5 * base + 0.5 * sr.score)
    cfeat = critic_features(_meta_light(pheno, result), result)
    discount = state.critic.discount(cfeat) if state.critic is not None else 1.0
    fit = float(base * discount)
    obj = objectives(pheno, result)
    # The island's own objective replaces the generic mission fraction in the
    # dominance vector, so an island's Pareto front is a front over *its* task.
    obj[0] = float(base if result.feasible else -0.5 + 0.5 * base)

    if result.exploit:
        state.curator.quarantine(bd, result.exploit, genome)
        state.telemetry.exploit(
            {"gen": state.archive.generation, "reason": result.exploit,
             "descriptor": bd, "meta": _meta(pheno, result, ctrl)}
        )
        state.curator.credit(operators, "rejected", 0.0)
        state.curator.note_offspring(parent, "rejected")
        return "exploit"

    previous = state.archive.cells[cell].fitness if cell in state.archive.cells else 0.0
    meta = _meta(pheno, result, ctrl)
    # The raw feature vector travels with the elite.  A learned projection moves,
    # and re-binning has to re-project from the features rather than from a
    # latent coordinate that no longer means the same thing.
    meta["features"] = [float(x) for x in feats]
    meta["objectives"] = {n: round(float(v), 4) for n, v in zip(OBJECTIVE_NAMES, obj)}
    meta["island"] = state.island
    meta["stage"] = sr.stage
    meta["stage_name"] = STAGES[sr.stage][0]
    meta["rungs"] = {d: j["rung"] for d, j in judged.items()}
    meta["judged"] = {d: round(j["total"], 3) for d, j in judged.items()}
    meta["critic_discount"] = round(discount, 3)
    meta["critic_features"] = [float(x) for x in cfeat]
    status = state.archive.add(genome, fit, bd, meta, tier=result.tier, objectives=obj)
    state.curriculum.update(cell, sr)

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
    """Run the whole loop.  Returns the final state, checkpointed as it goes.

    The shape of one generation:

      1. pick an island and breed a batch there, scored on that island's terms,
         at the difficulty stage its cell has earned, discounted by the critic
      2. every ``tier2_every`` generations, verify the best of the current
         island against the full mission -- and label the critic with what that
         verification found, which is the only way it learns anything
      3. every ``audit_every``, audit the designs the critic is most suspicious
         of; invalidate what fails and veto any bar the judge raised on the
         strength of it
      4. every ``judge_update_every``, let the bar ratchet
      5. every ``migrate_every``, move designs between islands and cross
         specialists from different ones
    """
    spec = spec or MissionSpec()
    rng = np.random.default_rng(cfg.seed)
    telemetry = Telemetry(cfg.run_dir, event_sample=cfg.event_sample)

    archipelago = Archipelago(migrate_every=cfg.migrate_every, n_migrants=cfg.n_migrants)
    curricula: dict = {}
    for name in cfg.islands:
        a = Archive(BD_AXES)
        archipelago.register(name, a, Curator(a, seed=cfg.seed))
        curricula[name] = Curriculum()

    learned = (
        LearnedDescriptors(n_dims=len(BD_AXES), refit_every=cfg.descriptor_refit_every)
        if cfg.learned_axes else None
    )
    state = SearchState(
        telemetry=telemetry, config=cfg, rng=rng,
        archipelago=archipelago, curricula=curricula,
        judge=Judge(quantile=cfg.judge_quantile, update_every=cfg.judge_update_every),
        auditor=Auditor(),
        critic=Critic(refit_every=cfg.critic_refit_every) if cfg.use_critic else None,
        island=cfg.islands[-1] if cfg.islands else "generalist",
        descriptors=learned, spec=spec,
    )

    telemetry.write("generations", {"kind": "run_start", "config": cfg.__dict__,
                                    "spec": spec.__dict__,
                                    "islands": {k: ISLANDS[k]["note"] for k in cfg.islands
                                                if k in ISLANDS}})
    seed_archipelago(state, spec)

    order = list(cfg.islands) or ["generalist"]
    pending: list = []

    for gen in range(cfg.generations):
        state.island = order[gen % len(order)]
        archive = state.archive
        curator = state.curator
        for a in archipelago.archives.values():
            a.generation = gen
        regime = curator.update_regime()

        for _ in range(cfg.batch):
            # Immigrants and hybrids are evaluated before anything home-grown,
            # because the whole point of moving them is to find out whether they
            # are worth anything under the receiving island's objective.
            inherited = None
            if pending and pending[0]["island"] == state.island:
                item = pending.pop(0)
                child, operators, parent = item["genome"].copy(), [item["kind"]], None
            else:
                parent = curator.select_parent()
                operators = curator.choose_operators()
                if parent is None:
                    child = random_genome(rng)
                    operators = ["random"]
                else:
                    child, operators = mutate(parent.genome, rng, operators=operators,
                                              n_ops=regime.n_mutations)
                    inherited = parent.meta.get("policy")
                    if rng.random() < 0.15:
                        other = curator.select_parent()
                        if other is not None and other is not parent:
                            child = crossover(child, other.genome, rng)
                            operators = operators + ["crossover"]

            child.generation = gen
            child.genome_id = f"{state.island[:2]}{gen}_{state.evaluated}"
            identify = (state.evaluated % max(cfg.identify_axes_every, 1)) == 0

            try:
                pheno, result, ctrl = evaluate_candidate(
                    child, cfg, inherited_policy=inherited, identify=identify,
                    spec=spec, seed=int(rng.integers(1 << 30))
                )
            except Exception as exc:
                telemetry.event({"kind": "error", "gen": gen, "island": state.island,
                                 "error": f"{type(exc).__name__}: {exc}"})
                curator.credit(operators, "rejected", 0.0)
                continue

            state.evaluated += 1
            curator.evaluations += 1
            _place(state, child, pheno, result, ctrl, parent, operators)

        # --- verification, and the critic's only source of truth ------------
        if gen % max(cfg.tier2_every, 1) == 0 and archive.cells:
            _verify_and_label(state, gen, spec, rng)

        # --- the third party ------------------------------------------------
        if gen % max(cfg.audit_every, 1) == 0 and archive.cells:
            invalid = _audit(state, gen, spec, rng)
            if invalid:
                vetoed = state.auditor.review_tightening(
                    state.judge, state.judge_moves, invalid
                )
                if vetoed:
                    telemetry.event({"kind": "judge_veto", "gen": gen,
                                     "vetoed": vetoed,
                                     "invalid_designs": invalid})
            state.judge_moves = []

        # --- the judge tightens ---------------------------------------------
        moved = state.judge.maybe_tighten(gen)
        if moved:
            state.judge_moves.extend(moved)
            telemetry.event({"kind": "judge_tighten", "gen": gen, "moves": moved,
                             **state.judge.report()})

        # --- the critic refits ----------------------------------------------
        if state.critic is not None and state.critic.due() and state.critic.fit():
            telemetry.event({"kind": "critic_fit", "gen": gen, **state.critic.report()})

        # --- learned descriptor axes ----------------------------------------
        if learned is not None and learned.due_for_refit() and learned.fit():
            def _reproject(e, _d=learned):
                f = e.meta.get("features")
                return _d.project(np.asarray(f, float)) if f else None

            axes = [
                (f"latent{i}", float(lo), float(hi), 8)
                for i, (lo, hi) in enumerate(learned.bounds())
            ]
            for name, a in archipelago.archives.items():
                stats = a.rebin(axes, _reproject)
                archipelago.curators[name].on_rebin()
            telemetry.event({"kind": "descriptor_refit", "gen": gen, **stats,
                             **learned.report()})

        # --- migration and hybridisation -------------------------------------
        if archipelago.due(gen):
            pending.extend(archipelago.migrate(gen, rng, crossover=crossover))
            telemetry.event({"kind": "migrate", "gen": gen, "pending": len(pending),
                             **archipelago.report()})

        # --- report -----------------------------------------------------------
        report = curator.generation_report()
        report["island"] = state.island
        report["evaluated"] = state.evaluated
        report["tier0_rejected"] = state.tier0_rejected
        report["elapsed"] = round(time.time() - state.started, 1)
        report["curriculum"] = state.curriculum.report()
        report["judge"] = state.judge.report()
        report["auditor"] = state.auditor.report()
        if state.critic is not None:
            report["critic"] = state.critic.report()
        report["archipelago"] = archipelago.report()
        best = archive.best
        if best is not None:
            report["best"] = {k: v for k, v in best.meta.items()
                              if k not in ("policy", "mobility_axes", "features",
                                           "critic_features")}
            report["best_fitness"] = round(best.fitness, 4)
        archive.history.append(archive.snapshot())
        telemetry.generation(report)
        if on_generation is not None:
            on_generation(state, report)

        if gen % max(cfg.checkpoint_every, 1) == 0:
            for name, a in archipelago.archives.items():
                a.save(Path(cfg.run_dir) / f"archive_{name}.pkl")
                a.export_json(Path(cfg.run_dir) / f"archive_{name}.json")

    for name, a in archipelago.archives.items():
        a.save(Path(cfg.run_dir) / f"archive_{name}.pkl")
        a.export_json(Path(cfg.run_dir) / f"archive_{name}.json")
    telemetry.close()
    return state


def seed_archipelago(state: SearchState, spec: MissionSpec) -> None:
    """Seed every island from the same archetypes.

    The same seeds everywhere on purpose.  The islands differ in what they
    *reward*, not in what they start from, so any divergence between them after
    a few hundred generations is attributable to the objective rather than to
    the draw.
    """
    cfg = state.config
    seeds: list[Genome] = list(seed_population(state.rng, cfg.n_reference_seeds))
    seeds += [random_genome(state.rng) for _ in range(cfg.n_random_seeds)]

    for i, g in enumerate(seeds):
        g.genome_id = f"seed{i}"
        try:
            pheno, result, ctrl = evaluate_candidate(
                g, cfg, identify=True, spec=spec, seed=int(state.rng.integers(1 << 30))
            )
        except Exception:
            continue
        state.evaluated += 1
        # One evaluation, filed on every island: the physics is the same, only
        # the scoring differs, so re-simulating per island would buy nothing.
        for name in state.archipelago.names:
            state.island = name
            state.curator.evaluations += 1
            _place(state, g.copy(), pheno, result, ctrl, parent=None, operators=["seed"])


def _verify_and_label(state: SearchState, gen: int, spec, rng) -> None:
    """Tier-2 the best of this island, and teach the critic what it found.

    This is the critic's only ground truth.  Everything it knows about the gap
    between a cheap score and a real one comes from these pairs, which is why
    verification is worth its cost even when the archive is not being pruned.
    """
    cfg = state.config
    archive, curator = state.archive, state.curator
    for elite in sorted(archive.cells.values(), key=lambda e: -e.fitness)[:3]:
        if not curator.should_promote(elite):
            continue
        try:
            p2 = build(elite.genome)
            r2 = evaluate_tier2(p2, spec=spec, seed=int(rng.integers(1 << 30)))
            f2 = fitness(p2, r2)
            curator.record_promotion(elite, f2)
            cheap = float(elite.meta.get("mission_fraction", 0.0))
            if state.critic is not None and cheap > 1e-4:
                feats = elite.meta.get("critic_features")
                if feats:
                    state.critic.label(
                        np.asarray(feats, float),
                        float(r2.mission_fraction) / cheap,
                    )
            state.telemetry.event({"kind": "promote", "gen": gen,
                                   "island": state.island, "cell": list(elite.cell),
                                   "tier2_fitness": round(f2, 4),
                                   "tier2_fraction": round(r2.mission_fraction, 4),
                                   "tier1_fraction": round(cheap, 4),
                                   "exploit": r2.exploit, "notes": r2.notes[:3]})
            if r2.exploit:
                curator.quarantine(elite.descriptor, r2.exploit, elite.genome)
                state.telemetry.exploit({"gen": gen, "reason": r2.exploit,
                                         "cell": list(elite.cell), "stage": "tier2"})
        except Exception as exc:
            state.telemetry.event({"kind": "error", "gen": gen, "stage": "tier2",
                                   "error": f"{type(exc).__name__}: {exc}"})
    curator.prune()


def _audit(state: SearchState, gen: int, spec, rng) -> int:
    """Audit the designs the critic is most suspicious of.  Returns how many
    were invalidated.

    Suspicion is used to *aim* the expensive check, not to punish -- a critic
    that can smell an exploit is most useful for deciding where to spend an
    audit.  Where it has no opinion the audit falls back to the best designs,
    which is where an undetected exploit would be doing the most damage anyway.
    """
    cfg = state.config
    archive = state.archive
    if not archive.cells:
        return 0

    def suspicion(e):
        if state.critic is None:
            return 0.0
        f = e.meta.get("critic_features")
        return state.critic.suspicion(np.asarray(f, float)) if f else 0.0

    ranked = sorted(archive.cells.values(),
                    key=lambda e: (-suspicion(e), -e.fitness))
    invalid = 0
    for elite in ranked[: cfg.audits_per_review]:
        try:
            pheno = build(elite.genome)
        except Exception:
            continue

        def reevaluate(seed=0, perturb=None, _p=pheno):
            return evaluate_tier1(
                _p, spec=spec, segment_seconds=cfg.segment_seconds,
                identify_axes=False, seed=seed, perturb=perturb,
            )

        class _Cheap:
            mission_fraction = float(elite.meta.get("mission_fraction", 0.0))
            segments: dict = {}

        rep = state.auditor.audit(pheno, _Cheap(), reevaluate=reevaluate,
                                  name=str(elite.meta.get("body_plan", "?")))
        state.telemetry.event({"kind": "audit", "gen": gen, "island": state.island,
                               "cell": list(elite.cell), **rep.summary()})
        if rep.invalid:
            invalid += 1
            archive.remove(elite.cell)
            if state.critic is not None:
                f = elite.meta.get("critic_features")
                if f:
                    state.critic.label(np.asarray(f, float), 0.0)
    return invalid
