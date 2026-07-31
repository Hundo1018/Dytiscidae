"""Learning a controller for one morphology.

The pieces this assembles were all present but never joined: a body has a
pattern generator, a set of empirically discovered control axes, and a small
policy that commands coefficients in those axes -- but until the policy weights
are actually optimised, every "controller" is a zero vector and the machine is
running open-loop on its raw rhythm.  That is a design being *driven*, not a
design that has *learned*, and the difference is visible the moment you watch
one.

Why CMA-ES rather than a gradient method
----------------------------------------
The policy is small on purpose.  The rhythm comes from the CPG and the
coordination comes from the mobility basis, so the policy only has to decide how
much of which axis to ask for given what the IMU and depth sensor report -- about
130 weights.  At that size a full-covariance evolution strategy converges in a
few hundred episodes on a CPU, needs no differentiable simulator, and does not
care that the fluid solver, the contacts and the energy model are all
non-smooth.  A gradient method would need all three to be differentiable and
would buy nothing at this parameter count.

What it optimises
-----------------
The mean of the three domain competences, with the *worst* domain weighted
extra.  A triphibian that flies beautifully and cannot dive has completed none
of the mission, so a plain mean is the wrong objective -- it rewards
specialising, which is exactly the behaviour the whole project is trying to
avoid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..evolution.cmaes import CMAES


@dataclass(eq=False)
class TrainingResult:
    """A trained controller and the record of how it got there."""

    policy_weights: np.ndarray
    bases: dict
    score: float
    baseline_score: float
    per_domain: dict[str, float]
    baseline_per_domain: dict[str, float]
    history: list[dict] = field(default_factory=list)
    wall_time: float = 0.0

    @property
    def gain(self) -> float:
        return self.score - self.baseline_score

    def summary(self) -> str:
        # The two objectives report different quantities, so show whichever was
        # actually measured rather than printing three zeros for the other one.
        keys = [k for k in self.per_domain if not k.startswith("_")]
        parts = "  ".join(
            f"{k}={self.baseline_per_domain.get(k, 0):.2f}->{self.per_domain.get(k, 0):.2f}"
            for k in keys
        )
        return (
            f"score {self.baseline_score:.3f} -> {self.score:.3f} "
            f"({self.gain:+.3f})   {parts}   {self.wall_time:.0f}s"
        )


def _objective(competences: dict[str, float]) -> float:
    """Collapse a competence report to one number.

    The continuous evaluator supplies its own composite under ``_score``; the
    isolated-leg evaluator gets ``0.6 * mean + 0.4 * min``.  The min term exists
    because a triphibian that flies well and cannot dive has completed none of
    the mission, and a plain mean rewards exactly the specialisation this
    project is trying to avoid.
    """
    if "_score" in competences:
        return float(competences["_score"])
    vals = list(competences.values())
    if not vals:
        return 0.0
    return 0.6 * float(np.mean(vals)) + 0.4 * float(np.min(vals))


def train_controller(
    phenotype,
    *,
    iterations: int = 15,
    popsize: int = 12,
    segment_seconds: float = 5.0,
    seed: int = 0,
    sigma0: float = 0.5,
    n_modes: int = 4,
    hidden: int = 16,
    bases: dict | None = None,
    continuous: bool = True,
    cycles: int = 1,
    on_iteration=None,
):
    """Optimise a policy for one phenotype.  Returns ``(Controller, TrainingResult)``.

    ``continuous`` selects the objective.  True scores the unbroken mission --
    the machine has to reach each commanded domain by itself, which is what the
    task actually is.  False scores each domain in isolation with a reset in
    between, which is cheaper and less noisy but rewards a competence that does
    not include getting there.
    """
    from ..control.cpg import Policy
    from ..envs.evaluate import Controller
    from ..envs.triphibian import DOMAIN_CYCLE, Domain, TriphibianEnv

    from ..envs.triphibian import MissionSpec

    t0 = time.time()
    spec_local = MissionSpec(cycles=cycles)
    env = TriphibianEnv(phenotype, seed=seed)

    # The axes have to be identified before anything can be commanded in them.
    if bases is None:
        bases = {}
        for dom in (Domain.AIR, Domain.WATER):
            bases[dom.value] = env.identify(dom, seed=seed)

    proto = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=n_modes, hidden=hidden)
    n_w = proto.n_weights

    # Structural loading has to be part of the objective, not just the picture:
    # a controller that reaches every domain by overloading its spars threefold
    # has not solved the task.
    probe = None
    if continuous:
        from ..physics.wake import attach_wake_probe
        from ..viz.showcase import StressProbe

        attach_wake_probe(env.solver)  # records the panel forces the probe reads
        probe = StressProbe(env)

    def make_policy(weights):
        if weights is None:
            return None
        pol = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=n_modes, hidden=hidden)
        pol.weights = np.asarray(weights, float)
        return pol

    def evaluate_legs(weights: np.ndarray | None) -> dict[str, float]:
        """Score each domain in isolation, resetting between them."""
        pol = make_policy(weights)
        ctrl = Controller(params=env.cpg.base, policy=pol, bases=bases)
        out: dict[str, float] = {}
        for dom in DOMAIN_CYCLE:
            env.reset(dom, randomise=False)
            seg = env.rollout(
                segment_seconds,
                params=env.cpg.base,
                policy=pol,
                basis=ctrl.basis_for(dom),
                domain=dom,
            )
            out[dom.value] = seg.competence
        return out

    def evaluate_continuous(weights: np.ndarray | None) -> dict[str, float]:
        """Score the unbroken mission: the machine must *get* to each domain.

        This is the objective that actually matches the task.  Scoring isolated
        legs rewards a machine that is competent in each domain when placed
        there, which is a different and much easier thing than one that can
        cross between them under its own control.
        """
        from ..envs.mission import build_schedule, continuous_score, run_continuous

        pol = make_policy(weights)
        ctrl = Controller(params=env.cpg.base, policy=pol, bases=bases)
        rng = np.random.default_rng(seed)
        schedule = build_schedule(spec_local, rng, leg_seconds=segment_seconds)
        res = run_continuous(env, ctrl, schedule, stress_probe=probe)
        return {
            "on_task": res.on_task,
            "transitions": res.transition_rate,
            "depth": float(np.clip(res.max_depth / 10.0, 0.0, 1.0)),
            "survival": 1.0 if res.survived else 0.2,
            "peak_stress": res.peak_stress,
            "_score": continuous_score(res, spec_local),
        }

    evaluate = evaluate_continuous if continuous else evaluate_legs

    # The open-loop reference: the same body on its raw rhythm, no policy.
    # Every reported gain is against this, because a controller that does not
    # beat "do nothing but flap" has not learned anything.
    baseline_doms = evaluate(None)
    baseline = _objective(baseline_doms)

    es = CMAES(np.zeros(n_w), sigma0=sigma0, popsize=popsize, seed=seed,
               bounds=(-4.0, 4.0))
    history: list[dict] = []
    best_doms = baseline_doms

    for it in range(iterations):
        pop = es.ask()
        scores = np.zeros(len(pop))
        doms_for = []
        for k, w in enumerate(pop):
            d = evaluate(w)
            doms_for.append(d)
            scores[k] = _objective(d)
        es.tell(pop, scores)
        top = int(np.argmax(scores))
        if scores[top] >= es.best_f - 1e-12:
            best_doms = doms_for[top]
        rec = {
            "iteration": it,
            "best": float(scores.max()),
            "mean": float(scores.mean()),
            "sigma": float(es.sigma),
            "elapsed": round(time.time() - t0, 1),
        }
        history.append(rec)
        if on_iteration is not None:
            on_iteration(it, rec)

    # Re-measure the winner rather than trusting the noisy in-loop score.
    final_doms = evaluate(es.best_x)
    final = _objective(final_doms)
    if final < baseline:
        # Optimisation can lose to the open-loop reference on a re-measure when
        # the episodes are short and the dynamics are chaotic. Keeping the worse
        # policy would be dishonest, so fall back and say so.
        final_doms, final = baseline_doms, baseline

    result = TrainingResult(
        policy_weights=es.best_x.copy(),
        bases=bases,
        score=final,
        baseline_score=baseline,
        per_domain=final_doms,
        baseline_per_domain=baseline_doms,
        history=history,
        wall_time=time.time() - t0,
    )

    pol = Policy(n_obs=TriphibianEnv.OBS_DIM, n_modes=n_modes, hidden=hidden)
    pol.weights = es.best_x.copy()
    controller = Controller(params=env.cpg.base, policy=pol, bases=bases)
    return controller, result
