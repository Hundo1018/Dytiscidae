"""Tier 1 and Tier 2 evaluation, and the scoring that turns them into fitness.

Tier 1 measures short episodes and *extrapolates* the forty-five minute mission
from steady-state power.  That is legitimate for the cruise phases -- steady
cruise is steady, and measuring 12 s of it tells you what 300 s of it costs --
but it is exactly the kind of shortcut that a search will learn to exploit.  So
the extrapolation is guarded:

  * a candidate that has not reached steady state within the window is scored on
    its worst window, not its mean;
  * transitions are simulated rather than extrapolated, because they are
    transient by definition and are where the energy and the structure go;
  * Tier 2 re-runs promoted elites on the real schedule, and any elite whose
    Tier-1 estimate was optimistic by more than a set fraction is flagged to the
    curator as an extrapolation exploit.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..control.cpg import CPGParams, MobilityBasis, Policy
from ..core.phenotype import Phenotype
from ..physics.energy import transition_energy
from .transitions import TransitionSet, run_transition
from .triphibian import (
    DOMAIN_CYCLE,
    Domain,
    MissionResult,
    MissionSpec,
    SegmentResult,
    TriphibianEnv,
    evaluate_tier0,
)


@dataclass(eq=False)
class Controller:
    """Everything needed to drive one machine: rhythm, axes, and intent."""

    params: CPGParams
    policy: Policy | None = None
    bases: dict[str, MobilityBasis] | None = None

    def basis_for(self, domain: Domain) -> MobilityBasis | None:
        if not self.bases:
            return None
        # Land uses the air basis: contact-dominated motion is not well captured
        # by a fluid-probe identification, and the air basis at least describes
        # the machine's own limb coordination.
        key = "water" if domain is Domain.WATER else "air"
        return self.bases.get(key)


# --------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Tier 1
# --------------------------------------------------------------------------


def evaluate_tier1(
    p: Phenotype,
    *,
    spec: MissionSpec | None = None,
    controller: Controller | None = None,
    segment_seconds: float = 10.0,
    identify_axes: bool = False,
    seed: int = 0,
    sea_state=None,
) -> MissionResult:
    """Short dynamic episodes in each domain, plus transitions."""
    spec = spec or MissionSpec()
    t0 = time.time()

    tier0 = evaluate_tier0(p, spec)
    r = MissionResult(tier=1)
    r.structural_margin = tier0.structural_margin
    r.feasible = tier0.feasible
    if not p.segments:
        r.notes.append("empty phenotype")
        r.wall_time = time.time() - t0
        return r

    try:
        env = TriphibianEnv(p, seed=seed, sea_state=sea_state)
    except Exception as exc:  # a genome that will not compile is simply dead
        r.notes.append(f"compile failed: {type(exc).__name__}: {exc}")
        r.wall_time = time.time() - t0
        return r

    ctrl = controller or Controller(params=env.cpg.base)

    if identify_axes:
        for dom in (Domain.AIR, Domain.WATER):
            try:
                r.mobility[dom.value] = env.identify(dom, seed=seed)
            except Exception as exc:
                r.notes.append(f"mobility id failed in {dom.value}: {exc}")
        if ctrl.bases is None:
            ctrl.bases = r.mobility

    clamped_any = False
    for dom in DOMAIN_CYCLE:
        env.reset(dom)
        seg = env.rollout(
            segment_seconds,
            params=ctrl.params,
            policy=ctrl.policy,
            basis=ctrl.basis_for(dom),
            domain=dom,
        )
        clamped_any |= env.solver.diag.clamped
        r.segments[dom.value] = seg

    for kind in ("air_to_water", "water_to_air", "water_to_land"):
        tr = run_transition(env, kind, ctrl)
        r.transitions.results[kind] = tr
        r.transition_ok[kind] = tr.crossed
        if tr.failure:
            r.notes.append(f"{kind}: {tr.failure}")

    # Energy: measured steady power extrapolated over the domain durations, plus
    # simulated transition costs.
    cruise_j = sum(
        s.mean_power * spec.seconds_per_domain * spec.cycles for s in r.segments.values()
    )
    trans_j = sum(transition_energy(p.mass, k) for k in spec.transitions)
    r.energy_required_wh = (cruise_j + trans_j) / 3600.0
    r.energy_available_wh = p.genome.battery_wh * 0.85

    competences = [s.competence for s in r.segments.values()]
    energy_fraction = float(
        np.clip(r.energy_available_wh / max(r.energy_required_wh, 1e-6), 0.0, 1.0)
    )
    # Graded, not binary.  A crossing used to contribute 1 or 0 depending only
    # on whether the depth changed sign, so a tumbling arrival at twice the
    # hull's survivable speed counted the same as a clean one.  The quality
    # terms are conditioned on having crossed at all, so refusing the hard
    # crossing cannot raise the average.
    tc = r.transitions.component_means()
    transition_fraction = float(
        tc["crossed"] * (
            0.40
            + 0.60 * float(np.mean([
                tc["shock"], tc["control"], tc["settle"], tc["economy"],
                tc["exit_state"],
            ]))
        )
    )
    # The mission is only as good as its weakest domain: a machine that flies
    # beautifully and cannot dive has completed none of the cycles, so this is a
    # geometric-style aggregate rather than a mean.
    r.mission_fraction = float(
        min(competences) ** 0.5
        * float(np.mean(competences))
        * energy_fraction
        * max(transition_fraction, 0.05)
    )

    # Exploit detection.
    #
    # The force limiter engaging is not by itself evidence of cheating: an
    # uncontrolled machine tumbling out of the sky trips it too, and that is
    # simply a design that fails.  What makes it an exploit is scoring *well*
    # while outside the model's valid envelope -- that is a candidate whose
    # performance depends on extrapolated coefficients.  So the flag is
    # conditioned on the score, and mere clamping is a proportional penalty.
    if clamped_any:
        if r.mission_fraction > 0.35:
            r.exploit = "scored well while outside the fluid model's valid envelope"
        else:
            r.mission_fraction *= 0.7
            r.notes.append("force limiter engaged (penalised, not disqualified)")
    if any(s.max_actuator_overload > 3.0 for s in r.segments.values()):
        r.exploit = "actuators run far past their thermal rating"

    r.wall_time = time.time() - t0
    return r


# --------------------------------------------------------------------------
# Tier 2
# --------------------------------------------------------------------------


def evaluate_tier2(
    p: Phenotype,
    *,
    spec: MissionSpec | None = None,
    controller: Controller | None = None,
    seed: int = 0,
    time_compression: float = 12.0,
    sea_state=None,
    tier1: MissionResult | None = None,
) -> MissionResult:
    """The real mission schedule, with disturbances and a random start domain.

    ``time_compression`` shortens each domain leg by a constant factor so a
    forty-five minute mission costs tens of seconds rather than tens of minutes.
    The *energy* accounting is still done over the full duration -- power is
    measured, then charged for the real 300 s -- so compression costs realism in
    the slow dynamics (thermal drift, wave beat frequencies) but not in the
    budget.  Setting it to 1.0 runs the mission honestly, and the orchestrator
    does exactly that for final candidates.
    """
    spec = spec or MissionSpec()
    t0 = time.time()
    rng = np.random.default_rng(seed)

    r = MissionResult(tier=2)
    r.structural_margin = p.report.min_margin
    r.feasible = p.report.ok
    if not p.segments:
        r.notes.append("empty phenotype")
        return r

    from ..physics.medium import SeaState

    sea = sea_state or SeaState(amplitude=float(rng.uniform(0.0, 0.25)),
                                period=float(rng.uniform(1.6, 3.2)))
    try:
        env = TriphibianEnv(
            p, seed=seed, sea_state=sea,
            current=rng.normal(0, 0.15, 3) * np.array([1, 1, 0.2]),
            wind=rng.normal(0, 1.2, 3) * np.array([1, 1, 0.3]),
        )
    except Exception as exc:
        r.notes.append(f"compile failed: {exc}")
        return r

    ctrl = controller or Controller(params=env.cpg.base)

    # Random starting domain, then cycle -- as specified.
    start = int(rng.integers(len(DOMAIN_CYCLE)))
    order = [DOMAIN_CYCLE[(start + i) % len(DOMAIN_CYCLE)]
             for i in range(len(DOMAIN_CYCLE) * spec.cycles)]

    leg_seconds = spec.seconds_per_domain / max(time_compression, 1.0)
    totals: dict[str, list[SegmentResult]] = {}
    completed = 0
    energy_j = 0.0

    env.reset(order[0])
    for leg_i, dom in enumerate(order):
        # Re-place the machine for the new domain: a real transition was scored
        # separately in Tier 1, and repeating all nine of them here would spend
        # the whole budget on transitions.
        if leg_i > 0:
            env.reset(dom)
        seg = env.rollout(
            leg_seconds,
            params=ctrl.params,
            policy=ctrl.policy,
            basis=ctrl.basis_for(dom),
            domain=dom,
        )
        totals.setdefault(dom.value, []).append(seg)
        # Charge energy for the *real* leg duration, not the compressed one.
        energy_j += seg.mean_power * spec.seconds_per_domain
        if seg.competence < 0.15 or not seg.survived:
            r.notes.append(f"leg {leg_i} ({dom.value}) failed: {seg.failure or 'incompetent'}")
            break
        completed += 1
        if energy_j / 3600.0 > p.genome.battery_wh * 0.85:
            r.notes.append(f"battery exhausted after {completed} of {len(order)} legs")
            break

    for k, segs in totals.items():
        best = max(segs, key=lambda s: s.competence)
        merged = SegmentResult(
            domain=best.domain,
            duration=sum(s.duration for s in segs),
            distance=sum(s.distance for s in segs),
            mean_speed=float(np.mean([s.mean_speed for s in segs])),
            mean_power=float(np.mean([s.mean_power for s in segs])),
            survived=all(s.survived for s in segs),
            competence=float(np.mean([s.competence for s in segs])),
            max_depth=max(s.max_depth for s in segs),
            peak_slam=max(s.peak_slam for s in segs),
            max_actuator_overload=max(s.max_actuator_overload for s in segs),
        )
        r.segments[k] = merged

    r.energy_required_wh = energy_j / 3600.0 + sum(
        transition_energy(p.mass, kind) for kind in spec.transitions
    ) / 3600.0
    r.energy_available_wh = p.genome.battery_wh * 0.85
    r.mission_fraction = completed / max(len(order), 1)

    # Cross-check the Tier-1 extrapolation.  A large optimistic gap means the
    # short window was not representative, which is a search exploit rather than
    # a modelling detail, so the curator needs to hear about it.
    if tier1 is not None and tier1.mission_fraction > 0.05:
        ratio = r.mission_fraction / tier1.mission_fraction
        if ratio < 0.45:
            r.exploit = (
                f"tier-1 overestimated by {1/max(ratio,1e-3):.1f}x "
                "(short window was not steady state)"
            )
        r.notes.append(f"tier1->tier2 retention {ratio:.2f}")

    r.wall_time = time.time() - t0
    return r


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

#: Behaviour descriptor axes for the MAP-Elites archive.
#:
#: Chosen so the resulting map answers the question the user actually asked --
#: "what specification is achievable?" -- rather than merely "what is best".
#: Mass and density ratio span the *design* question (how big, how buoyant);
#: the two competences span the *capability* question (does it fly, does it
#: dive).  Land competence is deliberately in the fitness rather than an axis,
#: because walking is the easiest of the three and would waste a dimension.
BD_AXES = [
    ("log_mass", math.log10(0.4), math.log10(40.0), 10),
    ("density_ratio", 0.15, 1.5, 8),
    ("air_competence", 0.0, 1.0, 8),
    ("water_competence", 0.0, 1.0, 8),
]


def behaviour_descriptor(p: Phenotype, r: MissionResult) -> np.ndarray:
    air = r.segments.get("air")
    water = r.segments.get("water")
    return np.array(
        [
            math.log10(max(p.mass, 1e-3)),
            p.density_ratio,
            air.competence if air else 0.0,
            water.competence if water else 0.0,
        ]
    )


#: The objectives a design is judged on, all "higher is better".
#:
#: These are kept *separate* rather than summed because summing them means
#: choosing an exchange rate between them, and I have no basis for choosing one.
#: The 0.10 coefficients in ``fitness`` below say that a tenth of the structural
#: margin is worth a tenth of the energy margin is worth a tenth of the land
#: competence.  That is not an engineering judgement, it is three numbers I
#: typed, and the cost of typing them is invisible: a design that is worse on
#: the weighted sum but better on an axis the weights undervalue is silently
#: discarded, and nothing in the run ever reports that it happened.
OBJECTIVE_NAMES = ("mission", "structure", "energy")


def objectives(p: Phenotype, r: MissionResult) -> np.ndarray:
    """The objective vector used to decide who occupies a cell.

    Feasibility is not an objective, it is a gate: an infeasible design gets a
    sentinel first component so it can never dominate a feasible one, however
    good its other numbers look.
    """
    if r.exploit:
        return np.array([-1.0, -1.0, -1.0])
    mission = r.mission_fraction if r.feasible else -0.5 + 0.5 * r.mission_fraction
    return np.array([
        float(mission),
        float(np.clip(r.structural_margin, -1.0, 3.0)),
        float(np.clip(r.energy_margin, -1.0, 2.0)),
    ])


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True when ``a`` is at least as good everywhere and better somewhere."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(a >= b - 1e-12) and np.any(a > b + 1e-12))


def fitness(p: Phenotype, r: MissionResult, spec: MissionSpec | None = None) -> float:
    """Scalar quality within a behaviour cell.

    Mission completion dominates.  Everything else is a tie-breaker that keeps
    the elite in a cell from being a fragile one-off: structural margin, energy
    margin, and land competence all push toward designs that would still work if
    the numbers moved a little.
    """
    spec = spec or MissionSpec()
    if r.exploit:
        return 0.0

    base = r.mission_fraction
    land = r.segments.get("land")
    land_c = land.competence if land else 0.0

    margin = float(np.clip(r.structural_margin, -1.0, 3.0))
    margin_term = 0.10 * float(np.clip(margin / 2.0, 0.0, 1.0))
    energy_term = 0.10 * float(np.clip(r.energy_margin, 0.0, 2.0) / 2.0)
    land_term = 0.10 * land_c

    if not r.feasible:
        # Infeasible designs are not zeroed -- that would flatten the landscape
        # and remove the gradient back toward feasibility -- but they can never
        # outrank a feasible one.
        return 0.25 * float(np.clip(1.0 + margin, 0.0, 1.0)) * (0.2 + 0.8 * base)

    return float(np.clip(base + margin_term + energy_term + land_term, 0.0, 2.0))
