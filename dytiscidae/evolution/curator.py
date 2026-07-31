"""The curator: active management of the search, not just survival of the fittest.

Plain evolutionary search makes exactly one decision -- who survives -- and makes
it with one number.  Everything else is fixed in advance by the person who wrote
the config: which mutations to try, how often to try structural ones, who to
breed from, how much compute to spend verifying a result.  Those settings are
usually wrong, and worse, the *right* settings change during the run.

This module makes those decisions part of the system, and it makes them from
evidence the run itself produces.

Six responsibilities
--------------------

1. **Operator credit assignment.**  Every mutation operator is an arm of a
   bandit.  Reward is the archive outcome it produced.  Operators that are
   currently paying get sampled more; operators that have stopped paying decay
   back toward exploration rather than being switched off, because their value
   returns when the population moves.

2. **Parent selection.**  Not uniform over elites, and not greedy on fitness.
   Combines four signals: fitness, *curiosity* (how often this elite's offspring
   land anywhere), *frontier position* (how empty the neighbourhood is), and
   recency.  Breeding from the crowded middle of a mapped region is the single
   biggest waste of evaluations in naive MAP-Elites.

3. **Fidelity promotion.**  Tier-2 evaluation is ~10x the cost of Tier-1, so it
   is spent as a budget: on elites that are new, high-scoring, long-unverified,
   or suspiciously good.

4. **Exploit detection and quarantine.**  Candidates that beat the simulator
   rather than the task are removed *and their cell is tainted*, so the search
   does not immediately rediscover the same trick from the same place.

5. **Regime detection.**  The run is classified each generation -- exploring,
   refining, stagnant, or collapsing -- and mutation pressure, emitter mix and
   feasibility bias are set from that.

6. **Crowding control.**  When the archive gets dense in one region, marginal
   elites there are pruned so the map stays legible and selection pressure stays
   spread out.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from ..core.genome import MUTATION_OPERATORS, STRUCTURAL_OPERATORS
from .archive import Archive, Elite


@dataclass
class OperatorStats:
    """Running record for one mutation operator."""

    name: str
    tries: int = 0
    reward_sum: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=60))

    @property
    def mean_reward(self) -> float:
        if not self.recent:
            return 0.0
        return float(np.mean(self.recent))

    @property
    def lifetime_mean(self) -> float:
        return self.reward_sum / max(self.tries, 1)


class OperatorBandit:
    """UCB over mutation operators, on a sliding window.

    The window matters more than the algorithm.  Operator usefulness is strongly
    non-stationary: ``add_part`` is worth a great deal in the first hundred
    generations and close to nothing once the body plans have settled, while
    ``cppn_weights`` is the reverse.  A lifetime average would keep sampling the
    early winners long after they stopped working, so the acting statistic is
    windowed and the lifetime figure is kept only for reporting.
    """

    def __init__(self, names: list[str] | None = None, c: float = 0.6) -> None:
        names = names or list(MUTATION_OPERATORS)
        self.stats = {n: OperatorStats(n) for n in names}
        self.c = c
        self.total = 0

    def select(self, rng: np.random.Generator, *, structural_bias: float = 1.0) -> str:
        """Pick an operator by UCB, tilted by the current structural appetite."""
        self.total += 1
        scores = {}
        for name, s in self.stats.items():
            if s.tries == 0:
                scores[name] = 1e6 + rng.random()  # try everything once
                continue
            bonus = self.c * math.sqrt(math.log(self.total + 1) / s.tries)
            tilt = structural_bias if name in STRUCTURAL_OPERATORS else 1.0
            scores[name] = (s.mean_reward + bonus) * tilt
        return max(scores, key=scores.get)

    def update(self, names: list[str], reward: float) -> None:
        """Credit every operator that contributed to one child.

        Split evenly rather than assigned to the last operator applied: with two
        or three mutations per child there is no way to attribute the outcome
        exactly, and even credit is unbiased.
        """
        if not names:
            return
        share = reward / len(names)
        for n in names:
            s = self.stats.get(n)
            if s is None:
                continue
            s.tries += 1
            s.reward_sum += share
            s.recent.append(share)

    def report(self) -> list[dict]:
        return sorted(
            (
                {"operator": n, "tries": s.tries, "recent": round(s.mean_reward, 4),
                 "lifetime": round(s.lifetime_mean, 4)}
                for n, s in self.stats.items()
            ),
            key=lambda d: -d["recent"],
        )


@dataclass
class Regime:
    """The curator's read on what the run is currently doing."""

    name: str = "exploring"
    structural_bias: float = 1.4
    n_mutations: int = 2
    emitter_fraction: float = 0.3
    feasibility_bias: float = 0.5
    note: str = ""


class Curator:
    """Decides who breeds, how, and what gets verified.

    Parameters
    ----------
    tier2_budget_fraction:
        Share of the evaluation budget spent on high-fidelity verification.
    crowding_limit:
        Neighbourhood occupancy above which marginal elites become prune
        candidates.
    """

    def __init__(
        self,
        archive: Archive,
        *,
        seed: int = 0,
        tier2_budget_fraction: float = 0.08,
        crowding_limit: int = 14,
    ) -> None:
        self.archive = archive
        self.rng = np.random.default_rng(seed)
        self.bandit = OperatorBandit()
        self.tier2_budget_fraction = tier2_budget_fraction
        self.crowding_limit = crowding_limit

        self.regime = Regime()
        self.coverage_history: deque = deque(maxlen=12)
        self.qd_history: deque = deque(maxlen=12)
        self.exploits: list[dict] = []
        self.promotions = 0
        self.evaluations = 0
        self.prunes = 0
        self._verified: dict[tuple[int, ...], int] = {}
        self._famine: dict[str, int] = {}
        #: Per-domain record process: best seen, evaluations drawn, and the
        #: draw index at which the best was set.  This is what makes the
        #: intervention decision evidence-based rather than scheduled.
        self._records: dict[str, dict] = {}
        #: Set by the loop.  The curator asks it two questions: which designs
        #: deserve extra weight as parents despite their score, and which must
        #: not be pruned however low that score is.
        self.scout = None
        self.starved_domains: list[str] = []
        self.famine_events: list[dict] = []
        self.log: list[dict] = []

    # ------------------------------------------------------------ 1. operators

    def choose_operators(self) -> list[str]:
        n = max(1, self.regime.n_mutations)
        return [
            self.bandit.select(self.rng, structural_bias=self.regime.structural_bias)
            for _ in range(n)
        ]

    def credit(self, operators: list[str], status: str, gain: float) -> None:
        reward = {"new": 1.0, "improved": 0.35, "rejected": -0.03}.get(status, 0.0)
        if status == "improved":
            reward += float(np.clip(gain, 0.0, 1.0))
        self.bandit.update(operators, reward)

    # ------------------------------------------------------ 2. parent selection

    def select_parent(self) -> Elite | None:
        """Pick who to breed from, weighting four independent signals.

        Drawn from the cells' full Pareto fronts, not only from the one
        representative each cell reports.  A design kept because it trades
        mission fraction for structural margin is only worth keeping if it can
        also be bred from; sampling representatives alone would store the
        trade-off and never explore it.
        """
        cells = [
            e
            for c, front in self.archive.fronts.items()
            if c not in self.archive.tainted
            for e in front
        ]
        if not cells:
            cells = [e for c, e in self.archive.cells.items() if c not in self.archive.tainted]
        if not cells:
            return None
        if len(cells) == 1:
            return cells[0]

        fitness = np.array([e.fitness for e in cells])
        curiosity = np.array([e.curiosity for e in cells])
        density = np.array([self.archive.neighbour_density(e.cell) for e in cells], float)
        age = np.array([self.archive.generation - e.born_at for e in cells], float)

        def norm(x):
            r = x.max() - x.min()
            return (x - x.min()) / r if r > 1e-12 else np.zeros_like(x)

        # Frontier preference: fewer neighbours is better, because offspring
        # from there have empty cells to land in.
        frontier = 1.0 - norm(density)
        # Mild recency preference, so a newly discovered region gets explored
        # while it is still moving.
        recency = np.exp(-age / 25.0)

        w = (
            0.35 * norm(fitness)
            + 0.30 * curiosity
            + 0.25 * frontier
            + 0.10 * recency
        )
        if self.regime.feasibility_bias > 0:
            feasible = np.array(
                [1.0 if e.meta.get("feasible") else 0.0 for e in cells]
            )
            w = w * (1.0 - self.regime.feasibility_bias
                     + self.regime.feasibility_bias * (0.25 + 0.75 * feasible))

        # When a domain is starved, tilt hard toward whoever is least bad at it.
        w = w * self._famine_weight(cells)
        # The scout argues for designs nobody else wants.  Multiplicative and
        # never below 1, so it can raise a dark horse's chances and cannot be
        # used to suppress a rival -- which matters when the thing making the
        # argument is a learned model.
        if self.scout is not None:
            w = w * np.array([
                self.scout.selection_weight(np.asarray(f, float))
                if (f := e.meta.get("scout_features")) is not None else 1.0
                for e in cells
            ])

        w = np.maximum(w, 1e-6)
        w = w / w.sum()
        return cells[int(self.rng.choice(len(cells), p=w))]

    def note_offspring(self, parent: Elite | None, status: str) -> None:
        if parent is None:
            return
        parent.offspring += 1
        if status in ("new", "improved"):
            parent.offspring_placed += 1

    # ------------------------------------------------------- 3. fidelity tiers

    def should_promote(self, elite: Elite) -> bool:
        """Whether this elite has earned a high-fidelity re-evaluation."""
        if elite.tier >= 2:
            return False
        budget = self.tier2_budget_fraction * max(self.evaluations, 1)
        if self.promotions > budget:
            return False

        best = self.archive.best
        top = best.fitness if best else 1.0
        # Verify things that are near the top, brand new, or improbably good --
        # the last of those being the most informative, since an outlier is
        # either a genuine discovery or an exploit and both are worth knowing.
        if elite.fitness > 0.85 * top:
            return True
        if elite.improvements == 0 and elite.fitness > 0.5 * top:
            return True
        last = self._verified.get(elite.cell, -999)
        return self.archive.generation - last > 60 and elite.fitness > 0.4 * top

    def on_rebin(self) -> None:
        """Forget everything keyed by cell coordinate.

        A learned-descriptor refit re-files every elite, so a cell index from
        before the refit names a different region afterwards.  ``_verified``
        would otherwise suppress Tier-2 promotion for designs that have never
        been verified, purely because some unrelated design used to sit at those
        coordinates.
        """
        self._verified.clear()

    def record_promotion(self, elite: Elite, tier2_fitness: float) -> None:
        self.promotions += 1
        self._verified[elite.cell] = self.archive.generation
        elite.tier = 2
        # Trust the verified number: an elite whose Tier-1 score does not survive
        # contact with the full mission should not keep its inflated rank.
        elite.fitness = min(elite.fitness, tier2_fitness)

    # ------------------------------------------------------- 4. exploit control

    def quarantine(self, descriptor: np.ndarray, reason: str, genome=None) -> None:
        """Record an exploit and taint the region it came from."""
        cell = self.archive.cell_of(descriptor)
        self.archive.tainted[cell] = self.archive.tainted.get(cell, 0) + 1
        self.exploits.append(
            {"generation": self.archive.generation, "cell": list(cell), "reason": reason}
        )
        # A cell tainted repeatedly is not a fluke; drop whatever is sitting in
        # it so the search stops breeding from a known trap.
        if self.archive.tainted[cell] >= 3:
            self.archive.remove(cell)

    # -------------------------------------------------------- 5. regime control

    def update_regime(self) -> Regime:
        """Classify the run and set search pressure accordingly."""
        a = self.archive
        self.coverage_history.append(a.coverage)
        self.qd_history.append(a.qd_score)

        cov_growth = 0.0
        qd_growth = 0.0
        if len(self.coverage_history) >= 4:
            cov_growth = self.coverage_history[-1] - self.coverage_history[-4]
            prev = self.qd_history[-4]
            qd_growth = (self.qd_history[-1] - prev) / max(abs(prev), 1e-6)

        feasible = sum(1 for e in a.cells.values() if e.meta.get("feasible"))
        feasible_frac = feasible / max(len(a.cells), 1)

        r = Regime()
        if len(a.cells) < 20:
            r.name = "bootstrapping"
            r.structural_bias, r.n_mutations = 1.6, 2
            r.emitter_fraction = 0.0  # nothing worth refining yet
            r.feasibility_bias = 0.7
            r.note = "too few elites to refine; widen the net"
        elif cov_growth > 0.004:
            r.name = "exploring"
            r.structural_bias, r.n_mutations = 1.5, 2
            r.emitter_fraction = 0.25
            r.feasibility_bias = 0.4
            r.note = f"coverage climbing (+{cov_growth*100:.2f}pp)"
        elif qd_growth > 0.01:
            r.name = "refining"
            r.structural_bias, r.n_mutations = 0.7, 1
            r.emitter_fraction = 0.55
            r.feasibility_bias = 0.6
            r.note = f"quality climbing (+{qd_growth*100:.1f}%) with flat coverage"
        else:
            # Nothing is moving.  Push hard on structure: the body plans in the
            # archive have been optimised to their local limits, and only a
            # topology change gets out of that.
            r.name = "stagnant"
            r.structural_bias, r.n_mutations = 2.2, 3
            r.emitter_fraction = 0.15
            r.feasibility_bias = 0.25
            r.note = "coverage and quality both flat; forcing structural change"

        if feasible_frac < 0.15 and len(a.cells) > 30:
            r.feasibility_bias = 0.85
            r.note += "; archive mostly infeasible, biasing hard toward feasible parents"

        self.regime = r
        # Famine detection runs last so it can override the regime it just set:
        # a domain nothing can perform is a more urgent fact than whether
        # coverage happened to tick up this generation.
        self.check_famine()
        return self.regime

    # ------------------------------------------------------ 6. crowding control

    def prune(self, max_prunes: int = 3) -> int:
        """Drop marginal elites from over-dense regions.

        Only ever removes an elite that is both *crowded* and *weak relative to
        its own neighbourhood*, so this thins redundancy without ever deleting a
        region's best representative.
        """
        a = self.archive
        removed = 0
        candidates = []
        for cell, e in a.cells.items():
            d = a.neighbour_density(cell)
            if d <= self.crowding_limit:
                continue
            neigh = [
                o.fitness for c, o in a.cells.items()
                if c != cell and all(abs(x - y) <= 1 for x, y in zip(cell, c))
            ]
            if neigh and e.fitness < np.percentile(neigh, 25):
                candidates.append((e.fitness, cell))
        # The scout's reserve.  A quota rather than a threshold: a fixed share of
        # the archive is held on predicted potential regardless of how the
        # scores happen to be distributed.  A threshold would protect everything
        # early and nothing late, which is backwards -- a dark horse matters
        # more once selection has sharpened, not less.
        reserved = set()
        if self.scout is not None:
            reserved = self.scout.reserve_ids(list(a.cells.values()))
        for _, cell in sorted(candidates)[:max_prunes]:
            e = a.cells.get(cell)
            if e is not None and id(e) in reserved:
                continue
            a.remove(cell)
            removed += 1
        self.prunes += removed
        return removed

    # ------------------------------------------------- 7. domain famine rescue

    #: Significance level for calling a domain stalled.  See ``plateau_p``.
    #:
    #: This replaces two constants that were simply typed: a competence floor of
    #: 0.35 below which a domain counted as starved, and a patience of 25
    #: generations before acting on it.  Neither had a basis.  0.35 is not a
    #: property of flight, and 25 generations is not a property of anything --
    #: they were numbers that produced behaviour I found reasonable when I
    #: watched a few runs, which is precisely the hand-tuning this project is
    #: supposed to be getting rid of.
    #:
    #: What replaced them needs no floor at all.  Mission fraction is built on
    #: ``min(competences)``, so the weakest domain *is* the binding constraint by
    #: construction -- there is always exactly one worth intervening on, and the
    #: only real question is whether it is stalled or merely slow.  That question
    #: has an answer from the data rather than from me.
    PLATEAU_ALPHA = 0.25

    def domain_bests(self) -> dict[str, float]:
        """Best competence any elite achieves in each domain."""
        out = {"air": 0.0, "water": 0.0, "land": 0.0}
        for e in self.archive.cells.values():
            for k in out:
                v = (e.meta or {}).get(k)
                if isinstance(v, (int, float)):
                    out[k] = max(out[k], float(v))
        return out

    def observe_domains(self, meta: dict) -> None:
        """Record one evaluation's per-domain competences.

        Every evaluation counts, not only the ones that make it into the
        archive: the question being asked is whether the *search* is still
        finding better performances in a domain, and a design rejected for
        landing in an occupied cell is still evidence about that.
        """
        for d in ("air", "water", "land"):
            v = (meta or {}).get(d)
            if not isinstance(v, (int, float)):
                continue
            rec = self._records.setdefault(d, {"best": -1.0, "draws": 0, "at_record": 0})
            rec["draws"] += 1
            if float(v) > rec["best"]:
                rec["best"] = float(v)
                rec["at_record"] = rec["draws"]

    def plateau_p(self, domain: str) -> float:
        """How surprising the current drought is, if the search were still
        improving at the rate it has been.

        Under exchangeable draws the probability that none of the next ``m``
        beats the best of the first ``n`` is exactly ``n / (n + m)``.  So the
        drought since the last record has a likelihood under "nothing has
        changed", and that likelihood is the evidence.  Waiting three times as
        long as the run took to set its last record puts it at 0.25.

        The exchangeability assumption is not true here -- an evolutionary
        search sets records faster than chance early and slower late -- so this
        is a proxy, not a hypothesis test.  It is a proxy with a stated basis
        and no free parameter, which is the point: the alternative was the
        number 25, chosen because runs looked about right with it.
        """
        rec = self._records.get(domain)
        if not rec or rec["at_record"] < 5:
            return 1.0  # too early to call anything a drought
        n = rec["at_record"]
        m = rec["draws"] - n
        return float(n / max(n + m, 1))

    def check_famine(self) -> list[str]:
        """Detect domains no design in the archive can perform, and act.

        A whole archive that cannot fly is not a search that needs more
        generations -- it is a search whose reachable space does not contain the
        capability, and running it longer produces a better non-flyer.  The
        useful response is to notice and change something, which is what this
        does: it redirects selection and mutation toward the starved domain, and
        it says so loudly enough to be seen in the log.

        This is deliberately *not* a silent adjustment.  A run that has been
        rescuing the same domain for two hundred generations is telling you the
        design space is wrong, and that is worth surfacing rather than papering
        over.
        """
        bests = self.domain_bests()
        if not bests or not self._records:
            self.starved_domains = []
            return []

        # The weakest domain is the binding constraint: mission fraction is
        # built on min(competences), so improving anything else cannot help
        # until this one moves.  No threshold is needed to identify it.
        weakest = min(bests, key=lambda d: bests[d])
        p = self.plateau_p(weakest)
        for d in ("air", "water", "land"):
            self._famine[d] = self._famine.get(d, 0) + 1 if d == weakest else 0

        acute = [weakest] if p <= self.PLATEAU_ALPHA else []
        if acute:
            # Push hard on structure: the current body plans have been optimised
            # to their local limits in this domain, and only a topology change
            # gets out of that.
            self.regime.name = "famine"
            self.regime.structural_bias = 2.6
            self.regime.n_mutations = 3
            self.regime.emitter_fraction = 0.1
            self.regime.feasibility_bias = 0.3
            rec = self._records[weakest]
            self.regime.note = (
                f"{weakest} is the binding domain at {bests[weakest]:.2f} and has "
                f"gone {rec['draws'] - rec['at_record']} evaluations without a "
                f"new best after taking {rec['at_record']} to set the last one "
                f"(p={p:.2f}) -- forcing structural change and biasing "
                "selection toward the least-bad performers there"
            )
            self.famine_events.append(
                {"generation": self.archive.generation, "domains": acute,
                 "p": round(p, 4), "drought": rec["draws"] - rec["at_record"],
                 "record_at": rec["at_record"],
                 "bests": {k: round(v, 3) for k, v in bests.items()}}
            )
            self.starved_domains = acute
        else:
            self.starved_domains = []
        return acute

    def _famine_weight(self, cells: list[Elite]) -> np.ndarray:
        """Extra selection weight for elites least bad in a starved domain."""
        if not self.starved_domains:
            return np.ones(len(cells))
        w = np.ones(len(cells))
        for d in self.starved_domains:
            vals = np.array([float((e.meta or {}).get(d, 0.0)) for e in cells])
            if vals.max() > 1e-9:
                w *= 1.0 + 2.5 * (vals / vals.max())
        return w

    # ------------------------------------------------------ 8. cohort approval

    #: How many designs are approved per round.
    #:
    #: Six, and the number is a judgement rather than a tuning constant.
    #:
    #: The archive is a *map*, so an approved cohort should represent the map
    #: and not merely its peak.  Two slots go to the best verified designs,
    #: because that is what "best" means; the other four are chosen by
    #: farthest-point sampling in behaviour space, so the cohort spans the mass
    #: and buoyancy range that was actually found rather than clustering around
    #: one body plan.
    #:
    #: Below about four the diversity that makes quality-diversity search worth
    #: running is lost and this becomes an expensive way to do hill climbing.
    #: Above about eight, Tier-2 verification -- roughly ten times the cost of a
    #: Tier-1 evaluation -- starts to dominate the budget, and the next round
    #: gets fewer generations to improve with.
    COHORT_SIZE = 6

    def select_cohort(
        self, n: int | None = None, *, require_feasible: bool = True,
        require_verified: bool = False,
    ) -> list[Elite]:
        """Approve a cohort to carry into the next round.

        Not the top *n*.  Taking the top *n* by fitness from a
        quality-diversity archive reliably returns *n* near-copies of one
        design, because neighbouring cells hold near-identical genomes -- which
        throws away the entire reason for keeping a map.
        """
        n = n or self.COHORT_SIZE
        pool = [
            e for c, e in self.archive.cells.items()
            if c not in self.archive.tainted
            and (not require_feasible or e.meta.get("feasible"))
            and (not require_verified or e.tier >= 2)
        ]
        if not pool:
            return []
        if len(pool) <= n:
            return sorted(pool, key=lambda e: -e.fitness)

        by_fitness = sorted(pool, key=lambda e: -e.fitness)
        chosen = by_fitness[: min(2, n)]
        remaining = [e for e in pool if e not in chosen]

        # Normalise descriptors so each axis contributes comparably; otherwise
        # the mass axis (which spans two decades) dominates the distance.
        lo, hi = self.archive.lo, self.archive.hi
        rng_ = np.maximum(hi - lo, 1e-9)

        def norm(e: Elite) -> np.ndarray:
            return (np.asarray(e.descriptor, float) - lo) / rng_

        while len(chosen) < n and remaining:
            picks = np.array([norm(e) for e in remaining])
            have = np.array([norm(e) for e in chosen])
            # Farthest-point: maximise the distance to the nearest already-chosen
            # design, then break ties on fitness.
            d = np.linalg.norm(picks[:, None, :] - have[None, :, :], axis=2).min(axis=1)
            fit = np.array([e.fitness for e in remaining])
            best = int(np.argmax(d + 0.15 * fit / max(fit.max(), 1e-9)))
            chosen.append(remaining.pop(best))
        return chosen

    def cohort_report(self, cohort: list[Elite]) -> list[dict]:
        """A compact, printable description of an approved cohort."""
        rows = []
        for i, e in enumerate(cohort):
            m = e.meta or {}
            rows.append({
                "rank": i,
                "fitness": round(e.fitness, 4),
                "tier": e.tier,
                "cell": list(e.cell),
                "mass_kg": m.get("mass"),
                "span_m": m.get("span"),
                "density_ratio": m.get("density_ratio"),
                "air": m.get("air"), "water": m.get("water"), "land": m.get("land"),
                "dof": m.get("dof"),
                "feasible": m.get("feasible"),
                "margin": m.get("margin"),
            })
        return rows

    # ---------------------------------------------------------------- reporting

    def generation_report(self) -> dict:
        a = self.archive
        return {
            "generation": a.generation,
            "regime": self.regime.name,
            "regime_note": self.regime.note,
            "structural_bias": round(self.regime.structural_bias, 2),
            "n_mutations": self.regime.n_mutations,
            "emitter_fraction": round(self.regime.emitter_fraction, 2),
            "feasibility_bias": round(self.regime.feasibility_bias, 2),
            "coverage": round(a.coverage, 4),
            "qd_score": round(a.qd_score, 3),
            "filled": len(a.cells),
            "front_designs": a.front_size,
            "evaluations": self.evaluations,
            "promotions": self.promotions,
            "exploits": len(self.exploits),
            "prunes": self.prunes,
            "domain_best": {k: round(v, 3) for k, v in self.domain_bests().items()},
            "starved": list(self.starved_domains),
            "famine_events": len(self.famine_events),
            "top_operators": self.bandit.report()[:5],
        }
