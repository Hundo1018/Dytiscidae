#!/usr/bin/env python3
"""Break the code on purpose and find out which tests notice.

Why
---

A test suite's own output cannot tell you whether it is effective.  Nineteen
files, 13,308 lines and every suite green is compatible with a suite that would
stay green through a sign error in the policy loss.  The only measurement that
separates the two is to introduce the sign error and look.

So each entry below is a *defect this project could plausibly ship*: a dropped
recursion in GAE, a clip that does not clip, an optimiser step that never runs,
a reverted fix.  The harness applies one at a time to a throwaway copy of the
tree, runs the suites that ought to catch it, and reports `caught` or
`SURVIVED`.  A survivor is not a bug in the code -- the code is fine once the
mutation is reverted -- it is a **hole in the tests**, named precisely enough to
close.

What it deliberately does not do
--------------------------------

It does not mutate at random.  Random operator mutation (`+` to `-`, `<` to
`<=`) generates mostly equivalent or absurd variants and buries the signal in
triage.  Every mutation here is hand-written to model a real failure mode, and
carries the suites that *should* see it, so "caught by nothing" and "caught by
the wrong suite" are different results.

It never touches the working tree.  The tree is copied with `git ls-files`, so
uncommitted edits to tracked files are included and nothing that is not tracked
is.  A mutation that does not apply exactly once is reported as `MISAPPLIED`
rather than silently skipped -- a mutation harness whose mutations quietly fail
to apply reports a perfect score.

Usage
-----

    PYTHONPATH=. .venv/bin/python tools/mutate.py            # every mutation
    PYTHONPATH=. .venv/bin/python tools/mutate.py --only gae # by id substring
    PYTHONPATH=. .venv/bin/python tools/mutate.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).exists():                                # pragma: no cover
    PYTHON = sys.executable


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect, and who ought to notice it."""

    id: str
    path: str
    find: str
    replace: str
    #: What the mutation models, in the terms a reviewer would use.
    defect: str
    #: Suites that should fail.  The first that fails ends the run for this
    #: mutation, so put the cheapest one first.
    suites: tuple
    #: Which checklist item this is evidence for.
    item: str = ""


@dataclass
class Result:
    mutation: Mutation
    status: str                 # "caught" | "SURVIVED" | "MISAPPLIED" | "ERROR"
    by: str = ""                # the suite that caught it
    detail: str = ""
    seconds: float = 0.0
    failing_checks: list = field(default_factory=list)


# --------------------------------------------------------------------------
# The mutations.  Grouped by what they model.
# --------------------------------------------------------------------------

MUTATIONS: tuple = (

    # --- one task draw per candidate (ROADMAP M2, 2026-10-08) --------------
    Mutation(
        id="batch-shares-the-first-seed",
        path="dytiscidae/envs/batchroll.py",
        find="    seeds = _per_machine(seed, k)\n    results = [MissionResult(tier=1) for _ in range(k)]\n",
        replace="    seeds = [_per_machine(seed, k)[0]] * k\n    results = [MissionResult(tier=1) for _ in range(k)]\n",
        defect="the batched path scores every machine on the first machine's draw, "
               "the generation-wide draw M2 removed",
        suites=("test_search::test_each_candidate_faces_its_own_draw",), item="draw per candidate"),
    Mutation(
        id="scatter-from-the-first-seed",
        path="dytiscidae/envs/batchroll.py",
        find="        scatter_seeds = {i: _scatter_seed(seeds[i], dom) for i in live}\n",
        replace="        scatter_seeds = {i: _scatter_seed(seeds[live[0]], dom) for i in live}\n",
        defect="each machine is stamped with its own seed but faces the first "
               "machine's scatter and task, so the stamp does not reproduce the score",
        suites=("test_search::test_each_candidate_faces_its_own_draw",), item="draw per candidate"),
    Mutation(
        id="pool-sends-every-seed-to-every-shard",
        path="dytiscidae/envs/actors.py",
        find="                kw[\"seed\"] = [kwargs[\"seed\"][i] for i in idx]\n",
        replace="                pass\n",
        defect="a shard receives the whole generation's seed list, raises in the "
               "worker, and the pool quietly re-runs the batch in the parent",
        suites=("test_search::test_each_candidate_faces_its_own_draw",), item="draw per candidate"),
    Mutation(
        id="loop-shares-the-generation-draw",
        path="dytiscidae/evolution/loop.py",
        find="        draw = [int(seeds[i]) for i in passed]\n",
        replace="        draw = int(seeds[passed[0]])\n",
        defect="evaluate_candidates passes one seed for the generation whatever "
               "draw_per_candidate says",
        suites=("test_search::test_each_candidate_faces_its_own_draw",), item="draw per candidate"),
    Mutation(
        id="placement-keeps-the-best-draw",
        path="dytiscidae/evolution/loop.py",
        find="    return results[order[(len(results) - 1) // 2]]\n",
        replace="    return results[order[-1]]\n",
        defect="the archive receives the best of a candidate's draws, the winner's "
               "curse the median exists to remove",
        suites=("test_search::test_a_candidate_is_placed_on_its_median_draw",), item="placement draws"),
    Mutation(
        id="placement-redraws-the-same-seed",
        path="dytiscidae/evolution/loop.py",
        find="    return int(np.random.default_rng([int(seed) & 0x7FFFFFFF, int(j), 0x5D]).integers(1 << 30))\n",
        replace="    return int(seed)\n",
        defect="the extra draws repeat the first, so the median is the one draw",
        suites=("test_search::test_a_candidate_is_placed_on_its_median_draw",), item="placement draws"),
    # --- the still machine is still (PAPERS_2610 item 4, 2026-10-08) -------
    Mutation(
        id="rotor-keeps-the-travel-margin",
        path="dytiscidae/envs/triphibian.py",
        find="                self.cpg.margin[k] = 0.0\n",
        replace="                pass\n",
        defect="a stopped rotor is clipped to 2.5% of top speed, so the still "
               "machine's rotors spin",
        suites=("test_physics::test_a_still_machine_is_still_from_the_first_step",), item="still machine"),
    Mutation(
        id="scatter-poses-the-body-gait",
        path="dytiscidae/envs/triphibian.py",
        find="                self.cpg.base if pose is None else pose, 0.0)\n",
        replace="                self.cpg.base, 0.0)\n",
        defect="the still machine starts at the gait's pose and its servos snap "
               "the joints to the held offset",
        suites=("test_physics::test_a_still_machine_is_still_from_the_first_step",), item="still machine"),
    # --- the antipodal heading pair (2026-10-06) ----------------------------
    Mutation(
        id="pair-takes-the-better-half",
        path="dytiscidae/envs/triphibian.py",
        find="        along = 0.5 * (cruise[\"along\"] + p[\"along\"])\n",
        replace="        along = max(cruise[\"along\"], p[\"along\"])\n",
        defect="the pair pays the better of the two headings, so a design that "
               "goes one way scores whenever either draw points there",
        suites=("test_search::test_the_heading_pair_cancels_what_the_command_did_not_choose",), item="heading pair"),
    Mutation(
        id="antipode-keeps-the-heading",
        path="dytiscidae/envs/tasks.py",
        find="        Phase(ph.kind, ph.start, _wrap(ph.heading + math.pi), ph.speed, ph.depth)\n",
        replace="        Phase(ph.kind, ph.start, ph.heading, ph.speed, ph.depth)\n",
        defect="the second half repeats the first heading, so nothing cancels",
        suites=("test_search::test_the_heading_pair_cancels_what_the_command_did_not_choose",), item="heading pair"),
    Mutation(
        id="pair-skipped-on-the-batched-path",
        path="dytiscidae/envs/batchroll.py",
        find="        if dom.value in PAIRED_MEDIA:\n",
        replace="        if False:\n",
        defect="the search scores single headings while every verification scores "
               "the pair -- the two evaluation paths disagree again",
        suites=("test_search::test_the_heading_pair_cancels_what_the_command_did_not_choose",), item="heading pair"),
    Mutation(
        id="pair-second-half-not-rewound",
        path="dytiscidae/envs/evaluate.py",
        find="    env.rng.bit_generator.state = state\n",
        replace="",
        defect="the second half starts from another reset draw, so passive motion "
               "differs between the halves and no longer cancels",
        suites=("test_search::test_the_heading_pair_cancels_what_the_command_did_not_choose",), item="heading pair"),
    Mutation(
        id="tier2-heading-fixed",
        path="dytiscidae/envs/evaluate.py",
        find="        seg = run_segment(env, dom, leg_seconds, ctrl, task=schedule_for(dom, rng))\n",
        replace="        seg = run_segment(env, dom, leg_seconds, ctrl)\n",
        defect="Tier-2 verifies at one fixed heading, the defect it exists to catch",
        suites=("test_search::test_tier2_draws_its_headings",), item="heading pair"),
    # --- the auditor's held-out check, per medium (2026-10-06) -------------
    Mutation(
        id="audit-held-out-reads-only-the-mission",
        path="dytiscidae/evolution/auditor.py",
        find="        if base > 1e-6 or media:\n",
        replace="        if base > 1e-6:\n",
        defect="the held-out check is skipped whenever the mission is zero -- every "
               "audit of arch48 -- so a credit earned at one seed is never re-measured",
        suites=("test_search::test_an_audit_re_measures_each_credited_medium_at_unseen_seeds",),
        item="winner's curse"),
    Mutation(
        id="audit-perturbation-reads-only-the-mission",
        path="dytiscidae/evolution/auditor.py",
        find="        if media or base > 1e-6:\n",
        replace="        if base > 1e-6:\n",
        defect="the perturbation check is skipped whenever the mission is zero, so "
               "a design that needs the model exactly right is never noticed",
        suites=("test_search::test_an_audit_re_measures_each_credited_medium_at_unseen_seeds",),
        item="winner's curse"),
    Mutation(
        id="audit-retained-defaults-to-one",
        path="dytiscidae/evolution/auditor.py",
        find="    retained_fraction: float | None = None\n",
        replace="    retained_fraction: float | None = 1.0\n",
        defect="an audit that re-measured nothing reports that it kept everything",
        suites=("test_search::test_an_audit_re_measures_each_credited_medium_at_unseen_seeds",),
        item="winner's curse"),
    Mutation(
        id="audit-missing-medium-reads-zero",
        path="dytiscidae/evolution/auditor.py",
        find="                    if k in segs:           # a medium not re-run is not a zero\n"
             "                        held[k].append(float(segs[k].competence) / media[k])\n",
        replace="                    held[k].append(float(getattr(segs.get(k), 'competence', 0.0))"
                " / media[k])\n",
        defect="a medium the re-run did not measure is counted as a collapse to zero",
        suites=("test_search::test_an_audit_re_measures_each_credited_medium_at_unseen_seeds",),
        item="winner's curse"),
    # --- evidence: when a pass share certifies a bar (2026-10-06) -----------
    Mutation(
        id="no-model-verdict-reads-the-share-not-the-bound",
        path="dytiscidae/domain/evidence.py",
        find="    if still_hi < elite_lo:\n        return CERTIFIED",
        replace="    if still_k / max(still_n, 1) < elite_k / elite_n:\n        return CERTIFIED",
        defect="a bar is certified whenever still machines clear it less often "
               "than elites in the sample, however few of either were run",
        suites=("test_domain",), item="no-model gate"),
    Mutation(
        id="clopper-pearson-upper-is-two-sided",
        path="dytiscidae/domain/evidence.py",
        find="    alpha = 1.0 - confidence\n",
        replace="    alpha = (1.0 - confidence) / 2\n",
        defect="the one-sided bound is computed at the two-sided level, so every "
               "certificate is harder to earn than the stated confidence",
        suites=("test_domain",), item="no-model gate"),

    # --- promotion telemetry ------------------------------------------------
    Mutation(
        id="promote-drops-tier1-media",
        path="dytiscidae/evolution/loop.py",
        find='                                   "tier1_media": {\n',
        replace='                                   "tier1_media_dropped": {\n',
        defect="the promote event stops carrying the Tier-1 per-medium competence, "
               "so the Tier-1/Tier-2 gap can only be recovered by a join that "
               "breaks at every descriptor refit",
        suites=("test_search::test_promotion_spends_refinement_and_keeps_what_it_buys",), item="tier-gap telemetry"),

    Mutation(
        id="score-ignores-the-competence-floor",
        path="dytiscidae/evolution/loop.py",
        find="    floored = below_competence_floor(state.island, result)\n",
        replace="    floored = False\n",
        defect="a design competent in none of its island's media is ranked like "
               "any other, so a machine that does nothing stands where the "
               "window puts it",
        suites=("test_search::test_the_scalar_stands_at_zero_below_the_competence_floor",), item="blend wiring"),
    Mutation(
        id="mission-weight-back-in-the-blend",
        path="dytiscidae/evolution/loop.py",
        find="    mission_weight: float = 0.0\n",
        replace="    mission_weight: float = 0.30\n",
        defect="0.3 of the scalar is a term that is zero for 97-100% of every "
               "island's window",
        suites=("test_search::test_the_scalar_stands_at_zero_below_the_competence_floor",), item="blend wiring"),
    Mutation(
        id="place-forgets-the-floor",
        path="dytiscidae/evolution/loop.py",
        find='objectives=obj, at_floor=sc["at_floor"])',
        replace="objectives=obj)",
        defect="the archive is never told a design is at the floor, so it "
               "displaces a ranked incumbent on the strength of a tie",
        suites=("test_search::test_the_scalar_stands_at_zero_below_the_competence_floor",), item="blend wiring"),
    Mutation(
        id="dry-run-forgets-the-floor",
        path="dytiscidae/evolution/loop.py",
        find='sc["obj"], at_floor=sc["at_floor"])',
        replace='sc["obj"])',
        defect="the refinement funnel's dry status calls a floored design a "
               "replacement the real placement then refuses",
        suites=("test_search::test_the_scalar_stands_at_zero_below_the_competence_floor",), item="blend wiring"),
    Mutation(
        id="verify-offers-verified-designs",
        path="dytiscidae/evolution/loop.py",
        find="    for elite in curator.promotion_candidates(3):",
        replace="    for elite in sorted(archive.cells.values(), key=lambda e: -e.fitness)[:3]:",
        defect="the verification round takes the top three by fitness, so two "
               "verified elites at the top leave it one design to try",
        suites=("test_search::test_verification_offers_designs_not_yet_verified",), item="blend wiring"),
    Mutation(
        id="reeval-reuses-the-stored-seed",
        path="dytiscidae/evolution/loop.py",
        find='["reeval"], elite, int(rng.integers(1 << 30))))',
        replace='["reeval"], elite, int(elite.meta.get("eval_seed") or 0)))',
        defect="a re-evaluation runs at the seed that scored the elite, so it "
               "measures the same draw again and the median of its draws is one draw",
        suites=("test_search::test_an_archived_elite_is_re_run_at_a_fresh_seed_and_kept_at_its_median_cpu",
                "test_search::test_an_elite_is_re_measured_at_fresh_draws_and_kept_at_its_median"), item="re-evaluation"),
    Mutation(
        id="reeval-commits-to-the-window",
        path="dytiscidae/evolution/loop.py",
        find="commit=False, at_cell=elite.cell)",
        replace="commit=True, at_cell=elite.cell)",
        defect="a re-evaluation feeds the curriculum window, the judge and the "
               "descriptor buffer, so one design is counted once per re-run",
        suites=("test_search::test_an_archived_elite_is_re_run_at_a_fresh_seed_and_kept_at_its_median_cpu",
                "test_search::test_an_elite_is_re_measured_at_fresh_draws_and_kept_at_its_median"), item="re-evaluation"),
    Mutation(
        id="reeval-records-nothing",
        path="dytiscidae/evolution/loop.py",
        find="    out = state.archive.record_draw(elite, draw)\n",
        replace='    out = {"n": 1, "median_base": 0.0, "removed": []}\n',
        defect="a re-evaluation is run and logged but never reaches the elite's "
               "buffer, so its score stays the single draw it began with",
        suites=("test_search::test_an_archived_elite_is_re_run_at_a_fresh_seed_and_kept_at_its_median_cpu",
                "test_search::test_an_elite_is_re_measured_at_fresh_draws_and_kept_at_its_median"), item="re-evaluation"),

    # --- the learner's arithmetic -----------------------------------------
    Mutation(
        id="gae-drops-the-recursion",
        path="dytiscidae/learning/ppo.py",
        find="                last = delta + self.gamma * self.lam * last",
        replace="                last = delta",
        defect="GAE degenerates to the one-step TD error at every lambda",
        suites=("test_ppo",), item="C/D advantage"),
    Mutation(
        id="gae-bootstraps-past-the-end",
        path="dytiscidae/learning/ppo.py",
        find="                nxt = val[i + 1] if i + 1 < n else 0.0",
        replace="                nxt = val[i + 1] if i + 1 < n else val[i]",
        defect="the terminal bootstrap is the last value instead of zero",
        suites=("test_ppo",), item="D target/bootstrap"),
    Mutation(
        id="return-loses-the-baseline",
        path="dytiscidae/learning/ppo.py",
        find="            RET.append(adv + val)",
        replace="            RET.append(adv)",
        defect="the value target is the advantage, not advantage plus value",
        suites=("test_ppo",), item="D return"),
    Mutation(
        id="gamma-ignored-in-shaping",
        path="dytiscidae/learning/ppo.py",
        find="                rew = rew + self.shaping * (self.gamma * nxt - phi)",
        replace="                rew = rew + self.shaping * (nxt - phi)",
        defect="potential shaping stops telescoping, so it changes the optimum",
        suites=("test_ppo",), item="D reward shaping"),
    Mutation(
        id="terminal-scale-disabled",
        path="dytiscidae/learning/ppo.py",
        find="        return {k: max(float(np.std(v)), 0.05) for k, v in by.items()}",
        replace="        return {k: 1.0 for k, v in by.items()}",
        defect="per-segment-kind reward scaling is off; water sets the gradient",
        suites=("test_ppo",), item="D reward"),

    # --- the update -------------------------------------------------------
    Mutation(
        id="clip-does-not-clip",
        path="dytiscidae/learning/ppo.py",
        find="                ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()",
        replace="                ratio * a, ratio * a).mean()",
        defect="PPO's clipped surrogate becomes the unclipped one",
        suites=("test_ppo",), item="J/PPO clip"),
    Mutation(
        id="policy-loss-sign-flipped",
        path="dytiscidae/learning/ppo.py",
        find="            pi_loss = -torch.min(",
        replace="            pi_loss = torch.min(",
        defect="the policy ascends away from the reward",
        suites=("test_ppo",), item="C gradient"),
    Mutation(
        id="optimiser-never-steps",
        path="dytiscidae/learning/ppo.py",
        find="            opt.step()",
        replace="            pass  # opt.step()",
        defect="the update computes a gradient and never applies it",
        suites=("test_ppo",), item="C parameter update"),
    Mutation(
        id="anneal-never-reaches-the-optimiser",
        path="dytiscidae/learning/ppo.py",
        find="        group[\"lr\"] = lr_now",
        replace="        group[\"lr\"] = float(lr)",
        defect="the annealed rate is reported but the optimiser keeps the base rate",
        suites=("test_ppo",), item="C scheduler"),
    Mutation(
        id="entropy-reverted-to-cross-entropy",
        path="dytiscidae/learning/ppo.py",
        find="        return self.entropy_of(self.latent(obs))",
        replace="        return -self._log_prob_under(\n"
                "            self.latent(obs), torch.tanh(self.latent(obs).sample())).mean()",
        defect="the entropy bonus goes back to having no expected gradient",
        suites=("test_ppo",), item="J regression / D entropy"),
    Mutation(
        id="kl-reverted-to-the-naive-estimator",
        path="dytiscidae/learning/ppo.py",
        find="                kl = float((logr.exp() - 1.0 - logr).mean().item())",
        replace="                kl = float((-logr).mean().item())",
        defect="the KL that target_kl stops on can go negative again",
        suites=("test_ppo",), item="J regression"),
    Mutation(
        id="non-finite-guard-removed",
        path="dytiscidae/learning/ppo.py",
        find="    if nonfinite:\n        return {\"transitions\": n,",
        replace="    if False:\n        return {\"transitions\": n,",
        defect="a NaN reward is consumed and poisons every weight again",
        suites=("test_ppo",), item="J regression / E NaN"),

    # --- the policy -------------------------------------------------------
    Mutation(
        id="tanh-jacobian-dropped",
        path="dytiscidae/learning/ppo.py",
        find="        return (base.log_prob(u).sum(-1)\n"
             "                - torch.log1p(-a.pow(2) + 1e-6).sum(-1))",
        replace="        return base.log_prob(u).sum(-1)",
        defect="the squashed density loses its change-of-variables term",
        suites=("test_ppo",), item="C correctness"),
    Mutation(
        id="observation-normaliser-does-not-scale",
        path="dytiscidae/learning/ppo.py",
        find="            (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -10.0, 10.0)",
        replace="            (obs - self.obs_mean), -10.0, 10.0)",
        defect="observations are centred but not scaled",
        suites=("test_ppo",), item="B numerical oracle"),
    Mutation(
        id="running-variance-is-wrong",
        path="dytiscidae/learning/ppo.py",
        find="        m2 = (self.obs_var * self.obs_count + b_var * b_n\n"
             "              + delta.pow(2) * self.obs_count * b_n / total)",
        replace="        m2 = self.obs_var * self.obs_count + b_var * b_n",
        defect="the parallel variance update drops its correction term",
        suites=("test_ppo",), item="B numerical oracle"),

    # --- reproducibility --------------------------------------------------
    Mutation(
        id="torch-seed-removed-from-run-search",
        path="dytiscidae/evolution/loop.py",
        find="        _torch.manual_seed(int(cfg.seed))",
        replace="        pass  # _torch.manual_seed(int(cfg.seed))",
        defect="the shared policy's initial weights stop following --seed",
        suites=("test_ppo",), item="J regression / H seed"),
    Mutation(
        id="learner-rng-not-checkpointed",
        path="dytiscidae/evolution/loop.py",
        find="    if state.learner_rng is not None:\n"
             "        payload[\"learner_rng_state\"] = state.learner_rng.bit_generator.state",
        replace="    if False:\n"
                "        payload[\"learner_rng_state\"] = state.learner_rng.bit_generator.state",
        defect="a resume takes a different sequence of gradient steps",
        suites=("test_ppo",), item="J regression / H random state"),
    Mutation(
        id="torch-rng-not-checkpointed",
        path="dytiscidae/evolution/loop.py",
        find="            payload[\"torch_rng_state\"] = (\n                _torch.get_rng_state().numpy().copy())",
        replace="            payload[\"torch_rng_state\"] = None",
        defect="a resume explores with a different noise stream",
        suites=("test_ppo",), item="J regression / H random state"),
    Mutation(
        id="update-shuffles-from-the-global-stream",
        path="dytiscidae/learning/ppo.py",
        find="    shuffler = rng if rng is not None else np.random.default_rng()",
        replace="    shuffler = np.random.default_rng()",
        defect="the caller's stream is ignored, so the update is not reproducible",
        suites=("test_ppo",), item="H reproducibility"),

    # --- physics ----------------------------------------------------------
    Mutation(
        id="hull-buckling-exponent",
        path="dytiscidae/physics/structure.py",
        find="    return knockdown * (n**2 - 1) * e_prime * wall**3 / (",
        replace="    return knockdown * (n**2 - 1) * e_prime * wall**2 / (",
        defect="the buckling allowable loses its cubic wall-thickness dependence",
        suites=("test_hull_buckling", "test_math"), item="C formula"),
    Mutation(
        id="jet-thrust-is-linear-in-flow",
        path="dytiscidae/physics/jet.py",
        find="        thrust_mag = coeff * rho * q * np.abs(q) / area",
        replace="        thrust_mag = coeff * rho * np.abs(q) / area",
        defect="jet thrust stops being a momentum flux",
        suites=("test_jet_energy",
                "test_physics::test_jet_thrust_matches_momentum_flux"),
        item="C formula"),

    # --- the variation operators ------------------------------------------
    #
    # `mut_gait` exists only to be a *joint* move; every way it can quietly
    # become axis-aligned looks identical to every other genome test, because
    # the children still build and still differ from their parents.
    Mutation(
        id="gait-drops-phase-and-rest",
        path="dytiscidae/core/genome.py",
        find="        part.phase_offset = float(rng.uniform(0.0, 2 * math.pi))\n"
             "        part.neutral = float(rng.uniform(0.2, 0.8))",
        replace="        pass  # phase and rest left where the parent had them",
        defect="mut_gait degrades to an amplitude-and-frequency move, so two of "
               "the four coordinates the thrust sweep needed never move",
        suites=("test_search::test_the_gait_operator_moves_every_coordinate_at_once",), item="arch39 S jointness"),
    Mutation(
        id="gait-touches-one-part",
        path="dytiscidae/core/genome.py",
        find="    g.flap_frequency = float(rng.uniform(1.5, 12.0))\n"
             "    for part in movable:",
        replace="    g.flap_frequency = float(rng.uniform(1.5, 12.0))\n"
                "    for part in movable[:1]:",
        defect="mut_gait resamples one part rather than the actuated set, which "
               "is what mut_stroke already did",
        suites=("test_search::test_the_gait_operator_moves_every_coordinate_at_once",), item="arch39 S jointness"),

    Mutation(
        id="continuous-mission-ignores-auto-reset",
        path="dytiscidae/envs/mission.py",
        find="            if int(env.data.warning[_bad].number) > bad0:\n"
             "                result.survived = False\n"
             "                result.failure = \"unstable\"\n"
             "                break\n",
        replace="",
        defect="a MuJoCo auto-reset teleports the machine mid-mission and the "
               "mission carries on across the jump (measured: survived=True, "
               "1500 of 1500 steps after a reset at step 400)",
        suites=("test_search::test_a_mujoco_auto_reset_ends_the_continuous_mission",), item="QACC auto-reset"),

    Mutation(
        id="island-best-reads-the-best-domain",
        path="dytiscidae/evolution/islands.py",
        find="    return float(np.min(comps)) ** 0.5 * float(np.mean(comps))\n\n\n"
             "@dataclass(eq=False)",
        replace="    return float(np.max(comps)) ** 0.5 * float(np.mean(comps))\n\n\n"
                "@dataclass(eq=False)",
        defect="an island's best is judged on its strongest own domain, so a "
               "one-medium specialist wins a pairing island",
        suites=("test_search::test_an_islands_best_is_judged_on_its_own_domains",), item="per-island best"),

    Mutation(
        id="triphibian-island-pays-the-best-medium",
        path="dytiscidae/evolution/islands.py",
        find="    inv = sum(1.0 / (max(float(v), 0.0) + e) for v in vals)\n"
             "    return float(3.0 / inv - e)",
        replace="    return float(max(vals))",
        defect="the triphibian island pays the best medium, so a one-medium "
               "specialist (arch45's fitness-1.0 design: air 0, water 0.337, "
               "land 0.003) outranks a machine that does all three",
        suites=("test_search::test_the_triphibian_island_pays_the_weakest_medium",),
        item="triphibian island"),

    Mutation(
        id="triphibian-ladder-reads-the-best-medium",
        path="dytiscidae/evolution/curriculum.py",
        find="    if stage <= 0:\n        return c2",
        replace="    if stage <= 0:\n        return c1",
        defect="the triphibian island's stage 0 reads the best medium, so 75% of "
               "its selection (the curriculum half at stage 0) pays a specialist",
        suites=("test_search::test_the_triphibian_island_pays_the_weakest_medium",),
        item="triphibian island"),

    Mutation(
        id="triphibian-curriculum-is-the-shared-ladder",
        path="dytiscidae/evolution/islands.py",
        find='                      weakest=spec.get("objective") == "weakest")',
        replace="                      weakest=False)",
        defect="the triphibian island is built with the shared ladder, whose "
               "stages 0-3 pay the best one or two media",
        suites=("test_search::test_the_triphibian_island_pays_the_weakest_medium",),
        item="triphibian island"),

    Mutation(
        id="competence-floor-at-zero",
        path="dytiscidae/evolution/islands.py",
        find='                and float(getattr(segs[d], "competence", 0.0)) >= COMPETENCE_FLOOR[d]):',
        replace='                and float(getattr(segs[d], "competence", 0.0)) >= 0.0):',
        defect="the competence floor is zero, so every design that has a segment "
               "clears it and the floor flags nothing",
        suites=("test_search::test_a_design_competent_in_nothing_its_island_reads_is_below_the_floor",),
        item="competence floor"),

    Mutation(
        id="competence-floor-reads-every-medium",
        path="dytiscidae/evolution/islands.py",
        find="    for d in island_media(island):\n        if (d in segs",
        replace='    for d in ("air", "water", "land"):\n        if (d in segs',
        defect="the competence floor reads all three media for every island, so a "
               "water specialist is excused by competence in a medium it does not read",
        suites=("test_search::test_a_design_competent_in_nothing_its_island_reads_is_below_the_floor",),
        item="competence floor"),

    Mutation(
        id="island-archive-read-through-the-merge",
        path="dytiscidae/ops/run.py",
        find="    if island is not None:\n        if island not in names:",
        replace="    if False:\n        if island not in names:",
        defect="one island's archive is read through the cross-island merge, so "
               "an elite that lost its cell to another island is never filmed",
        suites=("test_search::test_one_islands_archive_is_read_alone_not_through_the_merge",), item="per-island best"),

    Mutation(
        id="floor-candidate-joins-the-front",
        path="dytiscidae/evolution/archive.py",
        find="        if cand.at_floor:\n            return \"rejected\", None\n",
        replace="",
        defect="a design below the competence floor joins a ranked incumbent's "
               "front on margins alone, so a machine that cannot move holds a cell",
        suites=("test_search::test_margins_alone_cannot_fill_a_cell",), item="competence floor"),

    Mutation(
        id="floor-member-survives-a-ranked-entry",
        path="dytiscidae/evolution/archive.py",
        find="        kept = [e for e in ranked if not self._dominates(obj, e.objectives)]",
        replace="        kept = [e for e in front if not self._dominates(obj, e.objectives)]",
        defect="a ranked design entering a cell leaves the floor members beside "
               "it, so large margins keep a competence-zero machine in the front",
        suites=("test_search::test_margins_alone_cannot_fill_a_cell",), item="competence floor"),

    Mutation(
        id="representative-is-the-max-draw",
        path="dytiscidae/evolution/archive.py",
        find="        med = draws[order[(len(draws) - 1) // 2]]",
        replace="        med = draws[order[-1]]",
        defect="an elite's score is its best draw, so re-evaluation can only "
               "raise it and the winner's curse the buffer exists to remove stays",
        suites=("test_search::test_a_cells_score_is_the_median_of_its_draws",), item="draw buffer"),

    Mutation(
        id="draw-buffer-is-unbounded",
        path="dytiscidae/evolution/archive.py",
        find="        if len(draws) > d:\n            del draws[: len(draws) - d]\n",
        replace="",
        defect="an elite keeps every draw it was ever given, so its checkpoint "
               "grows with each re-evaluation and a stale draw never ages out",
        suites=("test_search::test_a_cells_score_is_the_median_of_its_draws",), item="draw buffer"),

    Mutation(
        id="rescore-leaves-the-front-alone",
        path="dytiscidae/evolution/archive.py",
        find="        removed = self._settle(elite.cell)",
        replace="        removed = []",
        defect="a re-evaluated score that is now dominated stays in its cell's "
               "front, so the front is a record of first draws",
        suites=("test_search::test_a_cells_score_is_the_median_of_its_draws",), item="draw buffer"),

    Mutation(
        id="curriculum-reads-every-medium",
        path="dytiscidae/evolution/curriculum.py",
        find="    if domains is not None:\n        segs = {d: s for d, s in segs.items() if d in domains}",
        replace="    if False:\n        segs = {d: s for d, s in segs.items() if d in domains}",
        defect="a specialist island's curriculum pays for another medium again, "
               "so the air and land islands fill with water machines",
        suites=("test_search::test_a_specialist_islands_curriculum_reads_only_its_own_medium",), item="island purity"),

    Mutation(
        id="standing-ranks-ties-at-or-below",
        path="dytiscidae/evolution/curriculum.py",
        find="    return float(np.mean(np.asarray(window, float) < x))",
        replace="    return float(np.mean(np.asarray(window, float) <= x))",
        defect="a tie takes its own mass as rank, so a design that equals the "
               "window's mode stands high on a score everyone shares",
        suites=("test_search::test_a_score_of_zero_stands_at_zero",), item="floor rank"),

    Mutation(
        id="zero-takes-the-young-window-half",
        path="dytiscidae/evolution/curriculum.py",
        find="    if x <= ZERO_SCORE:\n        return 0.0\n    if len(window) < min_n:",
        replace="    if len(window) < min_n:",
        defect="a raw score of zero stands at 0.5 while the stage window is "
               "young, so doing nothing is paid the median early in a run",
        suites=("test_search::test_a_score_of_zero_stands_at_zero",), item="floor rank"),

    Mutation(
        id="standing-ranks-against-the-floor",
        path="dytiscidae/evolution/curriculum.py",
        find="        return [e for e in w if not (len(e) > 3 and e[3])]",
        replace="        return list(w)",
        defect="the quantile is taken over the whole window, so with most of it "
               "at the floor every competent design stands at 0.95-1.0 and the "
               "scalar separates them by 0.05",
        suites=("test_search::test_standing_ranks_only_against_designs_above_the_floor",), item="floor rank"),

    Mutation(
        id="bandit-without-an-exploration-floor",
        path="dytiscidae/evolution/curator.py",
        find="                 epsilon: float = 0.2) -> None:",
        replace="                 epsilon: float = 0.0) -> None:",
        defect="the tilted argmax alone decides every slot, so a non-structural "
               "operator with a good mean can go unpicked for 80 generations",
        suites=("test_search::test_no_operator_can_go_dormant_under_the_structural_tilt",), item="operator dormancy"),

    Mutation(
        id="tier2-caps-fitness-again",
        path="dytiscidae/evolution/curator.py",
        find='        elite.meta["tier2_fitness"] = float(tier2_fitness)\n',
        replace='        elite.meta["tier2_fitness"] = float(tier2_fitness)\n'
                '        elite.fitness = min(elite.fitness, tier2_fitness)\n',
        defect="a failed Tier-2 read caps the design's fitness, so the best design "
               "becomes the worst parent and the last migrant (ROADMAP 2026-10-10 N4)",
        suites=("test_search::test_a_tier2_failure_does_not_make_the_best_design_the_worst_parent",), item="tier-2 flag"),

    Mutation(
        id="promotion-pool-keeps-verified-elites",
        path="dytiscidae/evolution/curator.py",
        find="        pool = [e for e in self.archive.cells.values() if e.tier < 2]",
        replace="        pool = list(self.archive.cells.values())",
        defect="the promotion pool offers designs already verified, so the Tier-2 "
               "budget is spent twice on the same elite",
        suites=("test_search::test_a_tier2_failure_does_not_make_the_best_design_the_worst_parent",), item="tier-2 flag"),

    Mutation(
        id="checkpoint-asks-git-at-every-write",
        path="dytiscidae/ops/checkpoint.py",
        find="    if _PROCESS_SHA is not None:\n        return _PROCESS_SHA\n",
        replace="",
        defect="each checkpoint names whatever HEAD is when it is written, not "
               "the code the process is running",
        suites=("test_search::test_a_checkpoint_names_the_commit_the_process_started_from",), item="provenance"),

    Mutation(
        id="stale-kernel-used-anyway",
        path="dytiscidae/envs/batchroll.py",
        find='        if state == "stale":\n            _USABLE = (False, why)\n            return _USABLE\n',
        replace="",
        defect="a GPU kernel older than its source is used to score anyway, so "
               "the search and every verification run different physics",
        suites=("test_search::test_a_kernel_older_than_its_source_is_not_usable",), item="kernel freshness"),

    # --- the task: each phase scored on its own purpose (2026-09-21) --------
    Mutation(
        id="cruise-pays-for-speed-in-any-direction",
        path="dytiscidae/envs/triphibian.py",
        find="                    along = float(v @ h)",
        replace="                    along = float(np.linalg.norm(v))",
        defect="a cruise phase pays for speed whatever its direction, so drifting "
               "across the commanded heading is progress again",
        suites=("test_physics",), item="task tracking"),
    Mutation(
        id="turn-counts-slowing-down",
        path="dytiscidae/envs/triphibian.py",
        find="                n = np.array([-ha[1], ha[0]])\n"
             "                if float(hb @ n) < 0.0:\n"
             "                    n = -n",
        replace="                n = (hb - ha) / max(float(np.linalg.norm(hb - ha)), 1e-9)",
        defect="a turn is read along the difference of the two headings, so a "
               "body slowing down on its launch heading scores as turning",
        suites=("test_physics",), item="task response"),
    Mutation(
        id="gain-is-ignored",
        path="dytiscidae/control/cpg.py",
        find="        if gain != 1.0:\n            out.amplitude = out.amplitude * float(gain)",
        replace="        if False:\n            out.amplitude = out.amplitude * float(gain)",
        defect="the gain channel is read and never applied, so stopping is still "
               "outside the action space",
        suites=("test_physics::test_the_gait_gain_can_stop_a_machine",),
        item="gait gain"),
    Mutation(
        id="search-cli-refine-default-two",
        path="dytiscidae/ops/run.py",
        find='p.add_argument("--refine-steps", type=int, default=0,',
        replace='p.add_argument("--refine-steps", type=int, default=2,',
        defect="a search launched with no flags spends two (1+1)-ES steps per "
               "candidate selected on the draw they report, the bias ROADMAP M1 "
               "measured (8-23% of the accepted gain survives a fresh draw)",
        suites=("test_physics::test_the_search_cli_defaults_are_the_stored_run_configuration",),
        item="cli defaults"),
    Mutation(
        id="cli-mission-weight-default-back",
        path="dytiscidae/ops/run.py",
        find='p.add_argument("--mission-weight", type=float, default=0.0,',
        replace='p.add_argument("--mission-weight", type=float, default=0.30,',
        defect="a search launched with no flags weights the mission term at 0.30 "
               "again, the quantile blend ARCH51 removed because mission_fraction "
               "is zero for 97-100% of every window (ROADMAP N2)",
        suites=("test_physics::test_the_search_cli_defaults_are_the_stored_run_configuration",),
        item="cli defaults"),
    Mutation(
        id="config-file-loses-to-cli-defaults",
        path="dytiscidae/ops/run.py",
        find="        if name in raw and not keep and mine == getattr(default, name):\n",
        replace="        if False:\n",
        defect="search --config keeps the command line's defaults, so a relaunch "
               "from an export silently runs a different configuration",
        suites=("test_physics::test_a_config_export_relaunches_the_same_search",), item="config export"),
    Mutation(
        id="config-file-picks-the-run-dir",
        path="dytiscidae/ops/run.py",
        find="        keep = (name in _LAUNCH_FIELDS\n",
        replace="        keep = (False\n",
        defect="search --config without --run writes into the recorded run's "
               "directory, and may inherit its resume flag",
        suites=("test_physics::test_a_config_export_relaunches_the_same_search",), item="config export"),
    Mutation(
        id="postrun-failure-is-quiet",
        path="dytiscidae/ops/run.py",
        find="    ok = rc == 0 and report\n",
        replace="    ok = True\n",
        defect="a post-run that leaves no report is recorded as fine, so a run "
               "ends without films or charts and nothing says so (arch47, arch48)",
        suites=("test_physics::test_a_failed_postrun_is_loud",), item="post-run"),
    Mutation(
        id="child-inherits-c-environment",
        path="dytiscidae/ops/run.py",
        find='        env.pop(k, None)\n    return env\n',
        replace='        env.pop(k, None)\n    return None\n',
        defect="a child Python inherits the kernel's C-level PYTHONEXECUTABLE and "
               "starts as the system interpreter, so post-run dies on import numpy",
        suites=("test_physics::test_a_child_python_is_the_venv_after_the_kernel_poisons_the_environment",),
        item="post-run environment"),
    Mutation(
        id="single-path-drops-the-gain",
        path="dytiscidae/control/cpg.py",
        find="        return a[:n_modes], gait_gain(a[n_modes])",
        replace="        return a[:n_modes], 1.0",
        defect="the single-machine paths -- Tier-2, the mission, every film -- drive "
               "at gain one while the batched pool honours it",
        suites=("test_search::test_the_gait_gain_drives_the_same_on_every_path",),
        item="gait gain"),
    Mutation(
        id="batched-drops-the-shared-gain",
        path="dytiscidae/envs/batchroll.py",
        find="                            gain = (gain or 0.0) + float(a_np[TWIST_DIM])",
        replace="                            pass",
        defect="the pool that scores the archive ignores the shared policy's gain "
               "action while the single path applies it",
        suites=("test_search::test_the_gait_gain_drives_the_same_on_every_path",),
        item="gait gain"),
    Mutation(
        id="summed-policy-drops-the-own-gain",
        path="dytiscidae/envs/evaluate.py",
        find="                gain, has_gain = gain + float(a[self.n_modes]), True",
        replace="                pass",
        defect="the single path drops the per-candidate policy's gain intent",
        suites=("test_search::test_the_gait_gain_drives_the_same_on_every_path",),
        item="gait gain"),
    Mutation(
        id="batched-never-records-the-gain",
        path="dytiscidae/envs/batchroll.py",
        find="            if r[\"gain\"] is not None:\n                r[\"gains\"].append(r[\"gain\"])",
        replace="            pass",
        defect="the path that scores the archive publishes no gain, so a run cannot "
               "see whether its controllers throttle when told to stop",
        suites=("test_search::test_the_gait_gain_drives_the_same_on_every_path",),
        item="gait gain"),
    Mutation(
        id="audit-perturbs-another-seed",
        path="dytiscidae/evolution/auditor.py",
        find="                    alt = reevaluate(seed=seed, perturb={key: factor})",
        replace="                    alt = reevaluate(seed=0, perturb={key: factor})",
        defect="the perturbed re-run uses seed 0 -- another scatter and another task "
               "-- so the audit's ratio measures the draw, not the perturbation",
        suites=("test_search::test_an_audit_perturbs_the_scored_experiment_and_nothing_else",),
        item="audit"),
    Mutation(
        id="audit-base-is-the-record",
        path="dytiscidae/evolution/loop.py",
        find="            mission_fraction = float(reevaluate().mission_fraction)",
        replace="            mission_fraction = float(elite.meta.get(\"mission_fraction\", 0.0))",
        defect="the audit divides a re-run by the recorded score, which another path "
               "and another network earned, so the ratio is not the perturbation's",
        suites=("test_search::test_an_audit_perturbs_the_scored_experiment_and_nothing_else",),
        item="audit"),
    Mutation(
        id="air-height-counts-floating",
        path="dytiscidae/envs/triphibian.py",
        find="                        ab = (np.ones(top - lo, bool) if airborne is None\n"
             "                              else np.asarray(airborne, bool)[lo:top])",
        replace="                        ab = np.ones(top - lo, bool)",
        defect="holding height no longer asks whether the machine is in the air, "
               "so a body that fell into the sea floats to full marks",
        suites=("test_physics",), item="task response"),
    Mutation(
        id="hold-ignores-sinking",
        path="dytiscidae/envs/triphibian.py",
        find="                               score=at * still * level * served)",
        replace="                               score=at * still * served)",
        defect="a hold stops asking whether the machine is still sinking, so a hull "
               "passing through the commanded depth at 0.1 m/s scores for holding it",
        suites=("test_physics",), item="task response"),
    Mutation(
        id="turn-counts-stopping",
        path="dytiscidae/envs/triphibian.py",
        find="                gain = min(float(b[\"v\"] @ n), float((b[\"v\"] - a[\"v\"]) @ n))",
        replace="                gain = float((b[\"v\"] - a[\"v\"]) @ n)",
        defect="a machine that slid away from the commanded side and stopped scores "
               "the stop as a turn toward it",
        suites=("test_physics",), item="task response"),
    Mutation(
        id="turn-ignores-airborne",
        path="dytiscidae/envs/triphibian.py",
        find="                        * min(a.get(\"airborne\", 1.0), b.get(\"airborne\", 1.0)))",
        replace="                        * 1.0)",
        defect="a turn is scored whether or not the machine is still in the air",
        suites=("test_physics",), item="task response"),
    Mutation(
        id="hold-ignores-motion",
        path="dytiscidae/envs/triphibian.py",
        find="                    still = float(np.clip(1.0 - e_m / band_m, 0.0, 1.0))",
        replace="                    still = 1.0",
        defect="a hold phase stops penalising movement, so a hull sinking through "
               "the commanded depth on the way past is holding it",
        suites=("test_physics",), item="task hold"),
    Mutation(
        id="hold-scale-is-absolute",
        path="dytiscidae/envs/triphibian.py",
        find="                    scale = max(abs(float(ph.depth) - d_start), self.DEPTH_BAND)\n"
             "                    band_m",
        replace="                    scale = 10.0\n"
                "                    band_m",
        defect="the hold's depth error is judged against a fixed band instead of "
               "the descent the command asked for, so a hull that sits where it "
               "was released earns most of it",
        suites=("test_physics",), item="task hold"),
    Mutation(
        id="land-stop-adds-to-the-walk",
        path="dytiscidae/envs/triphibian.py",
        find="            task = progress * (0.5 + 0.5 * stop[\"score\"])",
        replace="            task = 0.5 * progress + 0.5 * stop[\"score\"]",
        defect="stopping on land adds to the walk instead of qualifying it, and a "
               "rock -- which stops perfectly -- scores half",
        suites=("test_physics",), item="task land"),
    Mutation(
        id="controller-is-not-told-the-task",
        path="dytiscidae/envs/triphibian.py",
        find="                self.task_channels(R, ph),",
        replace="                np.zeros(6),",
        defect="the observation stops carrying the command, so the controller is "
               "scored on a purpose it cannot see -- the defect this whole change "
               "exists to remove",
        suites=("test_physics",), item="task observed"),
    Mutation(
        id="evaluators-ask-different-tasks",
        path="dytiscidae/envs/batchroll.py",
        find="        task = schedule_for(dom, np.random.default_rng(task_seed(scatter_seed)))",
        replace="        task = schedule_for(dom)",
        defect="the batched path asks every machine the default task while the "
               "single path asks the drawn one, so the search scores one experiment "
               "and every verification runs another",
        suites=("test_search::test_the_two_evaluation_paths_score_the_same_machine_the_same",), item="path agreement"),

    # --- the film is the evaluation (2026-09-21) --------------------------
    Mutation(
        id="pool-drops-the-shared-policy-from-rescores",
        path="dytiscidae/envs/actors.py",
        find="        spec = None\n        if shared is not None:\n",
        replace="        spec = None\n        if shared is not None and buffer is not None:\n",
        defect="the actor pool ships the shared policy to its workers only with a "
               "rollout buffer, so every re-score and refinement trial -- the "
               "numbers the archive keeps -- is scored without it",
        suites=("test_search::test_sharding_a_generation_does_not_change_a_score",), item="film = evaluation"),
    Mutation(
        id="level-gate-removed-from-the-task",
        path="dytiscidae/envs/triphibian.py",
        find="                            if not self.flies_level():\n"
             "                                flight = 0.0\n",
        replace="",
        defect="the air task pays holding height to any airframe that glides "
               "level for a few seconds, whatever its actuators deliver (AD)",
        suites=("test_physics::test_holding_height_is_flight_only_if_the_actuators_can",),
        item="AD level gate"),
    Mutation(
        id="level-gate-removed-from-the-ladder",
        path="dytiscidae/evolution/judge.py",
        find='        ("flies_level", "level_margin", 0.7),\n',
        replace="",
        defect="the flight rungs read trajectories only, so a glider launched at "
               "trim climbs them on its airframe (AD)",
        suites=("test_physics::test_holding_height_is_flight_only_if_the_actuators_can",),
        item="AD level gate"),
    Mutation(
        id="single-path-crossing-unscattered",
        path="dytiscidae/envs/transitions.py",
        find="    env.scatter(np.random.default_rng(transition_scatter_seed(kind)))\n",
        replace="",
        defect="Tier-2, films and probes start every crossing from the bare "
               "placement while the search scored a scattered one",
        suites=("test_search::test_the_two_evaluation_paths_score_the_same_machine_the_same",),
        item="path agreement"),
    Mutation(
        id="batched-entrainment-unreacted",
        path="dytiscidae/envs/batchroll.py",
        find="                if sol_i._prev_mbody is not None:\n"
             "                    entrainment_reaction(",
        replace="                if False:\n"
                "                    entrainment_reaction(",
        defect="the search's path lets entrained water create momentum while "
               "verification reacts it, so the two score different swimmers (AK)",
        suites=("test_search::test_the_two_evaluation_paths_score_the_same_machine_the_same",),
        item="path agreement"),
    Mutation(
        id="batched-transition-ignores-its-start",
        path="dytiscidae/envs/batchroll.py",
        find="                                   back=float((spec.transition_back or {}).get(kind, 0.0)))",
        replace="                                   back=0.0)",
        defect="the path the search scores with starts every probe at the "
               "interface while verification starts it where the curriculum "
               "says, so the two score different crossings (Y/O)",
        suites=("test_search::test_the_distance_curriculum_steps_back_only_on_evidence",),
        item="Y/O distance"),
    Mutation(
        id="distance-counts-other-distances",
        path="dytiscidae/evolution/curriculum.py",
        find="            if kind in self.kinds and abs(float(getattr(tr, \"start_back\", 0.0))\n"
             "                                          - self.back.get(kind, 0.0)) < 1e-9:",
        replace="            if kind in self.kinds:",
        defect="crossings made from an easier start count as evidence for the "
               "harder one, so the start runs away from what anyone can cross (Y/O)",
        suites=("test_search::test_the_distance_curriculum_steps_back_only_on_evidence",),
        item="Y/O distance"),
    Mutation(
        id="funnel-refines-everyone",
        path="dytiscidae/evolution/loop.py",
        find="            if w is None or w.size == 0 or (chosen is not None and not chosen[i]):",
        replace="            if w is None or w.size == 0:",
        defect="the refinement funnel is recorded and ignored: every candidate "
               "still enters every step's batch, so it saves nothing (AK)",
        suites=("test_search::test_the_refinement_funnel_refines_only_what_it_selects",),
        item="AK funnel"),
    Mutation(
        id="exploration-noise-per-shard",
        path="dytiscidae/envs/batchroll.py",
        find="        if shared is None or buffer is None:\n            return None\n",
        replace="        return None\n",
        defect="the learning rollout draws its noise from torch's stream seeded "
               "per shard, so a machine explores differently in another shard "
               "and the pool's shape changes what the learner sees (AJ)",
        suites=("test_search::test_sharding_a_generation_does_not_change_a_score",),
        item="AJ reproducibility"),
    Mutation(
        id="balance-reads-dof-not-rotors",
        path="dytiscidae/envs/actors.py",
        find="    return 18.0 + 1.7 * rotors\n",
        replace="    return 35.0 + 0.9 * float(getattr(pheno, \"n_actuated\", 0))\n",
        defect="the balanced queue predicts a machine's cost from DOF, which "
               "explains 7-19% of evaluation wall where rotors explain 25-42%, "
               "so a heavy shard can go last (AJ, 2026-10-03)",
        suites=("test_search::test_the_pool_queues_and_balances_by_rotors",),
        item="AJ cost model"),
    Mutation(
        id="queue-results-in-shard-order",
        path="dytiscidae/envs/actors.py",
        find="            for i, r in zip(idx, res):\n                results[i] = r\n",
        replace="            for i, r in zip(sorted(range(n))[len([x for x in results if x is not None]):], res):\n"
                "                results[i] = r\n",
        defect="results come back in shard order rather than to the machines "
               "that earned them once shards are not contiguous (AJ)",
        suites=("test_search::test_sharding_a_generation_does_not_change_a_score",),
        item="AJ queue"),
    Mutation(
        id="pool-map-out-of-order",
        path="dytiscidae/envs/actors.py",
        find="            return [f.result() for f in futures]\n        except Exception as exc:",
        replace="            return [f.result() for f in reversed(futures)]\n        except Exception as exc:",
        defect="a round's Tier-2 results come back to the wrong elites once they "
               "run on the workers (AL)",
        suites=("test_search::test_sharding_a_generation_does_not_change_a_score",),
        item="AL promotion"),
    Mutation(
        id="promotion-re-identifies",
        path="dytiscidae/evolution/loop.py",
        find='        bases = MobilityBasis.bases_from_record(elite.meta.get("mobility_basis"))',
        replace="        bases = {}",
        defect="a promotion re-identifies the elite at a fresh seed and hands "
               "Tier-2 a basis its Tier-1 score was not earned with (AL)",
        suites=("test_search::test_promotion_spends_refinement_and_keeps_what_it_buys",),
        item="AL promotion"),
    Mutation(
        id="identify-one-means-identify-all",
        path="dytiscidae/envs/batchroll.py",
        find="        wanted = [i for i in live if identify_axes[i]]",
        replace="        wanted = list(live) if any(identify_axes) else []",
        defect="one candidate due for identification identifies the whole batch, so "
               "identify_axes_every > 1 does not do what its name says (AN)",
        suites=("test_search::test_sharding_a_generation_does_not_change_a_score",),
        item="AN identify cadence"),
    Mutation(
        id="identify-list-not-sliced-per-shard",
        path="dytiscidae/envs/actors.py",
        find='                kw["identify_axes"] = [kwargs["identify_axes"][i] for i in idx]',
        replace='                kw["identify_axes"] = list(kwargs["identify_axes"])',
        defect="every shard receives the whole batch's identify list, so shard j "
               "identifies by shard 0's wishes (AN)",
        suites=("test_search::test_sharding_a_generation_does_not_change_a_score",),
        item="AN identify cadence"),
    Mutation(
        id="trim-probes-the-live-state",
        path="dytiscidae/envs/triphibian.py",
        find="        m, d = self.model, mj.MjData(self.model)",
        replace="        m, d = self.model, self.data",
        defect="measuring trim resets the simulation being run, so the first air "
               "reset of each body loses its spawn offset and later ones do not",
        suites=("test_physics",), item="film = evaluation"),
    Mutation(
        id="clearance-reads-render-geometry",
        path="dytiscidae/envs/triphibian.py",
        find="            g = np.nonzero((m.geom_bodyid != 0)\n"
             "                           & ((m.geom_contype != 0) | (m.geom_conaffinity != 0)))[0]",
        replace="            g = np.nonzero(m.geom_bodyid != 0)[0]",
        defect="clearance counts the camera's wing strips as the machine's lowest "
               "point, so a model built for filming flies a different air segment",
        suites=("test_physics",), item="film = evaluation"),
    Mutation(
        id="film-ignores-the-evaluation-seed",
        path="dytiscidae/viz/film.py",
        find='    seed = int(meta.get("eval_seed") or 0)',
        replace="    seed = 0",
        defect="the film re-runs the evaluation from a different initial condition "
               "and stamps it with the recorded score anyway",
        suites=("test_search::test_a_film_reproduces_the_scored_experiment",), item="film = evaluation"),

    Mutation(
        id="reversal-counts-agreement",
        path="dytiscidae/envs/triphibian.py",
        find="    return rate, float(np.mean(dots < 0.0))",
        replace="    return rate, float(np.mean(dots > 0.0))",
        defect="command_reversal counts steps that continue in the same direction, "
               "so a smooth controller reads as chattering and a chattering one as "
               "smooth",
        suites=("test_physics",), item="Y chatter"),
    Mutation(
        id="action-rate-penalty-ignored",
        path="dytiscidae/envs/triphibian.py",
        find="                chatter = 1.0 / (1.0 + self.action_rate_penalty * stats[0])",
        replace="                chatter = 1.0",
        defect="the command-rate penalty is configured and never applied",
        suites=("test_physics",), item="Y chatter"),

    Mutation(
        id="refit-gate-never-keeps",
        path="dytiscidae/evolution/descriptors.py",
        find="            if self.keep_if_overlap > 0.0 and self.last_overlap >= self.keep_if_overlap:",
        replace="            if False:",
        defect="the refit gate is configured and never keeps the axes, so every "
               "refit re-bins and merges the archive as before",
        suites=("test_search::test_a_refit_that_changes_nothing_can_be_skipped",), item="P refit"),

    Mutation(
        id="batched-fluid-diagnostics-stay-default",
        path="dytiscidae/envs/batchroll.py",
        find='                e.solver.diag.mean_submerged = float(o["subf"][pa:pb].mean())',
        replace="                pass",
        defect="the batched path leaves diag.mean_submerged at 0.0, so the "
               "policy the search scores is told it is dry however deep it is",
        suites=("test_search::test_the_batched_path_tells_the_policy_it_is_wet",), item="path agreement"),

    Mutation(
        id="stage-one-reads-gross-measurements",
        path="dytiscidae/evolution/curriculum.py",
        find='        for dom, key in (("air", "cruise_progress"), ("water", "cruise_progress"),',
        replace='        best = float(np.clip(1.0 - meas.get("air", {}).get("sink_rate", 9.9) / 3.0, 0.0, 1.0))\n'
                '        for dom, key in (("water", "cruise_progress"),',
        defect="the curriculum's directed stage pays for gliding from the 30 m "
               "launch again, which a machine with its actuators off also does",
        suites=("test_search::test_a_specialist_islands_curriculum_reads_only_its_own_medium",), item="sinking is not a capability"),

    # --- why nothing flew, 2026-09-23 -------------------------------------
    Mutation(
        id="strip-arm-from-frame-origin",
        path="dytiscidae/physics/fluid.py",
        find="        v_elem = v_org + _cross3(omega, pos - xipos[p.body_id])",
        replace="        v_elem = v_org + _cross3(omega, pos - xpos[p.body_id])",
        defect="the strip lever arm starts at the body frame while the velocity "
               "is at the centre of mass: flapping strips move 1.8-3.6x too fast",
        suites=("test_physics",), item="fluid kinematics"),
    Mutation(
        id="reversed-flow-mirrors-incidence",
        path="dytiscidae/physics/fluid.py",
        find="        alpha = np.where(rev, -alpha, alpha)\n",
        replace="",
        defect="flow over the trailing edge is folded by a mirror, so a plate "
               "swept backwards lifts the same way as forwards",
        suites=("test_physics",), item="fluid incidence"),
    Mutation(
        id="kramer-opposes-pitch-up",
        path="dytiscidae/physics/fluid.py",
        find="            -self.c_rot * rho * U * omega_s * p.chord**2 * p.dr,",
        replace="            self.c_rot * rho * U * omega_s * p.chord**2 * p.dr,",
        defect="rotational lift has the wrong sign and opposes nose-up pitching",
        suites=("test_physics",), item="fluid rotation"),
    Mutation(
        id="servo-without-feed-forward",
        path="dytiscidae/envs/triphibian.py",
        find="        return tgt + self.servo_lead[: len(tgt)] * rate",
        replace="        return tgt",
        defect="the position servo lags 75 ms and a 7.3 Hz stroke reaches 27%",
        suites=("test_physics",), item="actuation"),
    Mutation(
        id="cpg-phase-on-absolute-clock",
        path="dytiscidae/control/cpg.py",
        find="            self._psi += 2.0 * np.pi * (self._f_last - f) * t",
        replace="            pass",
        defect="a frequency change jumps the stroke phase by 2 pi df t",
        suites=("test_physics",), item="control continuity"),
    Mutation(
        id="a-fall-pays-for-flight",
        path="dytiscidae/envs/triphibian.py",
        find='                res.parts = {"gate": float(credit * frac), "control": 0.0}\n'
             '                return 0.0',
        replace='                res.parts = {"gate": float(credit * frac), "control": 0.10}\n'
                '                return float(credit * frac * 0.10)',
        defect="a body that reaches the sea from the launch in under 2.8 s is "
               "paid for flight, more than one that stayed up longer",
        suites=("test_physics",), item="sinking is not a capability"),

    # --- the fluid model's open items, closed 2026-09-23..26 ---------------
    Mutation(
        id='lift-drag-explicit',
        path='dytiscidae/physics/fluid.py',
        find='        self.implicit_damping = True\n',
        replace='        self.implicit_damping = False\n',
        defect='lift and drag integrated explicitly: the corrected added mass runs the teal away in water',
        suites=('test_physics',), item='F-03'),
    Mutation(
        id='weight-cancelled-at-strips',
        path='dytiscidae/physics/fluid.py',
        find='    fb[:, 2] += m_body * GRAVITY\n',
        replace='    pass\n',
        defect="the entrained fluid's weight is not cancelled at the centre of mass",
        suites=('test_physics',), item='F-14'),
    Mutation(
        id='limiter-scales-nothing',
        path='dytiscidae/physics/fluid.py',
        find='        fb *= limit / total\n',
        replace='        pass\n',
        defect='the per-machine limiter flags but does not bound the load',
        suites=('test_physics',), item='F-04'),
    Mutation(
        id='stall-over-sixteen-degrees',
        path='dytiscidae/physics/fluid.py',
        find='SEPARATION_COMPLETE = np.radians(6.0)',
        replace='SEPARATION_COMPLETE = np.radians(16.0)',
        defect='the handover is wide again and the drop after stall shrinks',
        suites=('test_stall_blend',), item='F-08'),
    Mutation(
        id='lev-never-from-rotation',
        path='dytiscidae/physics/fluid.py',
        find='    return 1.0 - _smoothstep((ro - rlo) / (rhi - rlo))',
        replace='    return np.zeros_like(ro)',
        defect='a revolving wing gets no leading-edge vortex: the robofly reads CL 1.1',
        suites=('test_physics',), item='F-02'),
    Mutation(
        id='no-wagner-lag',
        path='dytiscidae/physics/fluid.py',
        find='        alpha_e = alpha * (1.0 - a1 - a2) + a1 * self.x[0] + a2 * self.x[1]',
        replace='        alpha_e = alpha',
        defect='an impulsively started strip carries its whole circulation at once',
        suites=('test_physics',), item='F-13'),
    Mutation(
        id='inflow-disc-halved',
        path='dytiscidae/physics/fluid.py',
        find='            k = T / (2.0 * rho * self.area)',
        replace='            k = T / (rho * self.area)',
        defect='the momentum inflow is sqrt(2) too large in hover',
        suites=('test_physics',), item='F-13'),
    Mutation(
        id='slam-four-times',
        path='dytiscidae/physics/structure.py',
        find='    k = min((math.pi / (2.0 * math.tan(dr))) ** 2, 250.0)',
        replace='    k = min((math.pi / math.tan(dr)) ** 2, 250.0)',
        defect="Wagner's slam peak is 4x too high",
        suites=('test_physics',), item='S-02'),
    Mutation(
        id='propellers-one-handed',
        path='dytiscidae/physics/rotor.py',
        find='        td = (self.handed[rows] * sgn)[:, None] * ax\n',
        replace='        td = sgn[:, None] * ax\n',
        defect='a mirrored propeller is the same hand: half the rotors push down',
        suites=('test_physics',), item='rotor'),
    Mutation(
        id='universal-built-as-hinge',
        path='dytiscidae/core/mjcf.py',
        find='            if s.part.joint == "universal":\n',
        replace='            if False:\n',
        defect='a universal joint is a plain hinge again: no wing can feather',
        suites=('test_physics',), item='AB'),
    Mutation(
        id='height-from-endpoints',
        path='dytiscidae/envs/triphibian.py',
        find='                                sink = max(sink, float(drops.max()))',
        replace='                                sink = sink',
        defect='a drop-and-recover trajectory reads as holding height',
        suites=('test_physics',), item='AE'),
    Mutation(
        id='stage-one-pays-coast',
        path='dytiscidae/evolution/curriculum.py',
        find='                v *= float(np.clip(meas.get("air", {}).get("airborne_fraction", 0.0), 0.0, 1.0))',
        replace='                v *= 1.0',
        defect="curriculum stage 1 pays the launch's coast after the machine is in the sea",
        suites=('test_physics',), item='AE'),

    # --- the rotor lookup, vectorised, 2026-10-03 ----------------------------
    Mutation(
        id='rotor-batch-one-table',
        path='dytiscidae/physics/rotor.py',
        find='    r = np.arange(len(om)) if rows is None else rows\n',
        replace='    r = (np.arange(len(om)) if rows is None else rows)[:1].repeat(len(om))\n',
        defect="every rotor of a batch is looked up in the first rotor's table",
        suites=('test_physics::test_the_rotor_batch_is_the_per_rotor_loop',), item='perf'),
    Mutation(
        id='rotor-table-one-sum',
        path='dytiscidae/physics/rotor.py',
        find='    T = np.array([np.sum(row) for row in dT * dr])\n',
        replace='    T = np.full(len(om), np.sum(dT * dr) / len(om))\n',
        defect="the vectorised table build averages thrust over the grid instead of per point",
        suites=('test_physics::test_the_rotor_batch_is_the_per_rotor_loop',), item='perf'),
    Mutation(
        id='clearance-batch-first-slice',
        path='dytiscidae/envs/triphibian.py',
        find='            e._clear_memo = (e._clearance_key(), float(np.min(diff[at:at + g.size])))\n',
        replace='            e._clear_memo = (e._clearance_key(), float(np.min(diff[:g.size])))\n',
        defect="every machine in a batch is given the first machine's clearance",
        suites=('test_physics::test_the_clearance_of_a_batch_is_each_machines_own',), item='AM'),
    Mutation(
        id='clearance-memo-keyed-on-time',
        path='dytiscidae/envs/triphibian.py',
        find='        return (d.time, d.geom_xpos.tobytes(), d.geom_xmat.tobytes(),\n                d.xpos[self.root_body].tobytes())\n',
        replace='        return (d.time,)\n',
        defect="the clearance memo answers for a restored or re-placed state at the same clock",
        suites=('test_physics::test_the_clearance_of_a_batch_is_each_machines_own',), item='AM'),
    # --- the host-side speed-up, 2026-09-27 ----------------------------------
    Mutation(
        id='batched-power-one-sum',
        path='dytiscidae/physics/energy.py',
        find='                p += float(np.sum(terms[a:b]))\n',
        replace='                p += float(np.sum(terms))\n',
        defect="every machine is charged for the whole batch's actuators",
        suites=('test_physics',), item='perf'),
    Mutation(
        id='caller-moves-state-under-launch',
        path='dytiscidae/envs/batchroll.py',
        find='                for i, e in enumerate(envs):\n                    acc[i] += e.body_twist()\n',
        replace='                for i, e in enumerate(envs):\n                    e._mj.mj_forward(e.model, e.data)\n                    acc[i] += e.body_twist()\n',
        defect='a caller refreshes the kinematics between steps, after the early launch read them',
        suites=('test_search::test_the_early_fluid_launch_changes_nothing',), item='perf'),
    # --- arch43's pool, 2026-09-26 --------------------------------------------
    Mutation(
        id='nan-observation-reaches-policy',
        path='dytiscidae/envs/batchroll.py',
        find='    return bool(np.all(np.isfinite(obs)))\n',
        replace='    return True\n',
        defect='a NaN observation reaches the shared policy and raises out of the batch',
        suites=('test_search::test_a_nan_observation_fails_the_rollout_not_the_batch',), item='arch43'),
    # --- AG and AI, 2026-09-26 ---------------------------------------------
    Mutation(
        id='level-rig-spends-battery',
        path='dytiscidae/envs/triphibian.py',
        find='            self.budget = budget\n',
        replace='            pass\n',
        defect='measuring the level margin drains the battery the segments then fly on',
        suites=('test_physics',), item='AG'),
    Mutation(
        id='rotor-torque-off-by-radius',
        path='dytiscidae/physics/rotor.py',
        find='        out += frac * np.array([n * c_t, n * R * c_q])',
        replace='        out += frac * np.array([n * c_t, n * c_q])',
        defect="the rotor table's torque loses a factor of the radius",
        suites=('test_physics',), item='AI'),
    Mutation(
        id='mut-rotor-never-adds',
        path='dytiscidae/core/genome.py',
        find='        _new_rotor(bare[int(rng.integers(len(bare)))], rng)\n        return True',
        replace='        return False',
        defect='the search can never put a propeller on a design',
        suites=('test_physics',), item='AI'),

    Mutation(
        id="rotor-thrust-at-start-of-step",
        path="dytiscidae/physics/rotor.py",
        find="            omega_e = np.where((w_ < 1e-6) | (c < 1e-12), omega, sgn * w2)\n",
        replace="            omega_e = omega\n",
        defect="rotor thrust is taken at the start-of-step spin, and a propeller entering water is fired out of it",
        suites=("test_physics",), item="rotor in water"),
    Mutation(
        id="entrainment-not-reacted",
        path="dytiscidae/physics/fluid.py",
        find="            if self._prev_mbody is not None:\n"
             "                entrainment_reaction(",
        replace="            if False:\n"
                "                entrainment_reaction(",
        defect="added mass grows in the mass matrix with no -dm/dt v reaction, so "
               "every step of entrainment creates dm v of momentum: the ray "
               "accelerates to 21.9 m/s after entering at 8 (AK)",
        suites=("test_physics::test_entry_shock_is_hydrodynamic_not_a_speed_limit",),
        item="AK water entry"),
    Mutation(
        id="damping-stale-at-the-surface",
        path="dytiscidae/physics/fluid.py",
        find="        return self._stale or self._count % self.REFRESH_EVERY == 0",
        replace="        return self._count % self.REFRESH_EVERY == 0",
        defect="the implicit damping is refreshed on its 4-step cadence only, so "
               "first contact with water can run three steps of water loads "
               "against a B formed in air: a strut joint thrown to 100 rad/s (AK)",
        suites=("test_physics::test_entry_shock_is_hydrodynamic_not_a_speed_limit",),
        item="AK water entry"),

    Mutation(
        id="steps-not-counted",
        path="dytiscidae/envs/triphibian.py",
        find="        self.steps_run += 1\n",
        replace="",
        defect="an evaluation reports no physics steps, and a generation's cost cannot be attributed",
        suites=("test_physics",), item="generation cost"),

    Mutation(
        id="rig-steps-charged-to-segments",
        path="dytiscidae/envs/triphibian.py",
        find="            self.rig_steps += self.steps_run - steps0\n",
        replace="",
        defect="level_margin's rig steps are counted as segment steps",
        suites=("test_physics",), item="generation cost"),

    Mutation(
        id="cost-untimed-counts-subphases",
        path="dytiscidae/evolution/loop.py",
        find='        timed = sum(v for k, v in self.seconds.items() if "." not in k)\n',
        replace="        timed = sum(self.seconds.values())\n",
        defect="a sub-phase is subtracted twice and untimed goes negative",
        suites=("test_physics",), item="generation cost"),

    Mutation(
        id="merged-rescore-halves-swapped",
        path="dytiscidae/evolution/loop.py",
        find="            results, first = both[:k], (first[0], first[1], both[k:])\n",
        replace="            results, first = both[k:], (first[0], first[1], both[:k])\n",
        defect="the merged batch hands the trials' scores to the re-score and back",
        suites=("test_search::test_merging_the_rescore_with_the_first_refinement_changes_nothing",),
        item="merged re-score"),

    Mutation(
        id="mean-not-padded",
        path="dytiscidae/learning/ppo.py",
        find="                blocks = [obs[j:j + MEAN_MIN_ROWS]\n"
             "                          for j in range(0, obs.shape[0], MEAN_MIN_ROWS)]\n",
        replace="                blocks = [obs]\n",
        defect="a row's action on the mean depends on how many rows share its batch",
        suites=("test_ppo",), item="merged re-score"),

    # --- the job layer ----------------------------------------------------
    Mutation(
        id="job-accepts-any-transition",
        path="dytiscidae/domain/job.py",
        find="        if to not in _ALLOWED[self.status]:",
        replace="        if False:",
        defect="the job lifecycle stops rejecting impossible transitions",
        suites=("test_domain",), item="A/D state machine"),

    # --- the index --------------------------------------------------------
    Mutation(
        id="index-generator-drifts",
        path="tools/index_gen.py",
        find="             \"is in `CLAUDE.md`.\", \"\"]",
        replace="             \"is in `CLAUDE.md`.\", \"\", \"drifted\"]",
        defect="the committed index stops matching what the source generates",
        suites=("test_index",), item="A/J index gate"),
    # --- the critic's label and calibration (2026-10-03) ------------------
    # --- crossings (ARCH46_SPEC §8, 2026-10-03) ------------------------------
    Mutation(
        id="crossing-hold-gate-off",
        path="dytiscidae/envs/transitions.py",
        find="        r.crossed = cross >= 0 and not r.failure and r.hold >= HOLD_PASS",
        replace="        r.crossed = cross >= 0 and not r.failure",
        defect="a crossing made before the command counts: a falling body crosses air_to_water",
        suites=("test_search::test_a_crossing_is_commanded_and_a_still_machine_makes_none",),
        item="ARCH46 8"),
    Mutation(
        id="crossing-passes-through-and-counts",
        path="dytiscidae/envs/transitions.py",
        find="        elif cross >= 0 and (medium_of(env) is not self.target",
        replace="        elif False and (medium_of(env) is not self.target",
        defect="reaching the target and leaving it again counts as a crossing",
        suites=("test_search::test_a_crossing_is_commanded_and_a_still_machine_makes_none",),
        item="ARCH46 8"),
    Mutation(
        id="crossing-told-to-go-from-the-start",
        path="dytiscidae/envs/transitions.py",
        find="        return self.start if i < self.hold_steps else self.target",
        replace="        return self.target",
        defect="the controller observes the target during the hold, so it is never told to stay",
        suites=("test_search::test_a_crossing_is_commanded_and_a_still_machine_makes_none",),
        item="ARCH46 8"),
    Mutation(
        id="crossing-aloft-means-no-contact",
        path="dytiscidae/envs/transitions.py",
        find="            aloft = float(np.clip(r.aloft_fraction, 0.0, 1.0))",
        replace="            aloft = float(np.clip(r.airborne_fraction, 0.0, 1.0))",
        defect="a body floating in water has no contacts and earns 0.300 of water_to_air",
        suites=("test_search::test_a_crossing_is_commanded_and_a_still_machine_makes_none",),
        item="ARCH46 8"),
    Mutation(
        id="crossing-height-counts-the-placement",
        path="dytiscidae/envs/transitions.py",
        find="            height = float(np.clip(r.go_peak_clearance / 0.5, 0.0, 1.0))",
        replace="            height = float(np.clip(r.peak_clearance / 0.5, 0.0, 1.0))",
        defect="the placement gap counts as height, paying a sitting machine a takeoff",
        suites=("test_search::test_an_attempted_takeoff_outscores_never_leaving_the_ground",),
        item="ARCH46 8"),
    Mutation(
        id="crossing-scatter-not-reseated",
        path="dytiscidae/envs/transitions.py",
        find="    reseat_after_scatter(env, kind)\n    r.survivable_entry_speed",
        replace="    r.survivable_entry_speed",
        defect="scatter poses the joints after placement and drops a land start from 0.5 m",
        suites=("test_search::test_an_attempted_takeoff_outscores_never_leaving_the_ground",),
        item="ARCH46 8"),
    Mutation(
        id="water_to_land-starts-dry",
        path="dytiscidae/envs/transitions.py",
        find="        env.data.qpos[2] = -WATER_START_DEPTH\n",
        replace="        env.data.qpos[2] = env._clear_of_terrain(8.0 - back, 0.0, 0.0, gap=0.02)\n",
        defect="the old placement: the machine starts dry on the ramp and is credited for settling in",
        suites=("test_search::test_a_crossing_is_commanded_and_a_still_machine_makes_none",),
        item="ARCH46 8"),
    # --- the ninth instance: still crossings left after 10-03 (2026-10-04) ---
    Mutation(
        id="crossing-air-hold-reads-height-only",
        path="dytiscidae/envs/transitions.py",
        find="            loss = max(self.z0 - self.z_min, e_loss)\n",
        replace="            loss = self.z0 - self.z_min\n",
        defect="a glider coasting on its launch speed holds height for 1.5 s and then "
               "glides in: arch46's still gliders cross air_to_water again",
        suites=("test_search::test_the_bodies_that_crossed_held_still_in_arch46_cross_nothing",),
        item="ARCH46 8"),
    Mutation(
        id="crossing-land-is-any-ground-contact",
        path="dytiscidae/envs/transitions.py",
        find="    if medium_of(env) is not Domain.LAND:\n        return False\n",
        replace="    if medium_of(env) is Domain.LAND:\n        return True\n",
        defect="a float whose root rides above the waterline while its hull rests on the "
               "submerged ramp is on land: arch46's still floats cross water_to_land",
        suites=("test_search::test_the_bodies_that_crossed_held_still_in_arch46_cross_nothing",),
        item="ARCH46 8"),
    Mutation(
        id="crossing-shore-progress-counts-the-hold",
        path="dytiscidae/envs/transitions.py",
        find="                (float(env.root_pos()[0]) - self.x_go) / (SHORE_X - self.x0), 0.0, 1.0))",
        replace="                (float(env.root_pos()[0]) - self.x0) / (SHORE_X - self.x0), 0.0, 1.0))",
        defect="drift during the hold is paid as going: a still amphibian capsizing "
               "shoreward earns 0.136 of a graded water_to_land",
        suites=("test_search::test_the_bodies_that_crossed_held_still_in_arch46_cross_nothing",),
        item="ARCH46 8"),
    Mutation(
        id="crossing-aborted-probe-keeps-its-hold",
        path="dytiscidae/envs/transitions.py",
        find="        r.hold = 0.0 if aborted else self.hold_score()\n",
        replace="        r.hold = self.hold_score()\n",
        defect="a probe that blew up or went flat keeps its hold, so its graded approach "
               "pays the jump: 0.54 to arch46 elite 82 held still",
        suites=("test_search::test_a_crossing_is_commanded_and_a_still_machine_makes_none",),
        item="ARCH46 8"),
    Mutation(
        id="still-machine-leaves-rotors-spinning",
        path="dytiscidae/envs/triphibian.py",
        find="                off[k] = float(self.cpg.lo[k])\n",
        replace="                pass\n",
        defect="the still machine keeps its rotors at their throttle (the 2026-10-04 "
               "still arm): two water_to_air 'still' crossers were rotor-driven",
        suites=("test_search::test_the_bodies_that_crossed_held_still_in_arch46_cross_nothing",),
        item="ARCH46 8"),
    Mutation(
        id="refit-trigger-reads-a-stale-alias",
        path="dytiscidae/evolution/loop.py",
        find="    start_gen = load_state(state) if getattr(cfg, \"resume\", False) else 0",
        replace="    _stale = state.descriptors\n"
                "    start_gen = load_state(state) if getattr(cfg, \"resume\", False) else 0\n"
                "    if _stale is not None: state.descriptors = _stale",
        defect="the arch24 resume bug: the refit trigger reads descriptors captured "
               "before load_state replaced them, so a resumed run never refits",
        suites=("test_search::test_learned_axes_survive_resume",),
        item="arch24 resume"),
    Mutation(
        id="tier1_5-ignores-the-floor",
        path="dytiscidae/envs/evaluate.py",
        find="    dom = domain or weakest_domain(competences or {}, floor=LEG_COMPETENCE_BAR)",
        replace="    dom = domain or weakest_domain(competences or {})",
        defect="Tier-1.5 runs a medium the design scored 0 in, so its retention "
               "is 0/0 -- arch45, 124 of 147 promotions",
        suites=("test_search::test_a_long_leg_runs_on_promotion_candidates_only",),
        item="ARCH46 2a"),
    Mutation(
        id="tier2-probe-legs-enter-the-mission",
        path="dytiscidae/envs/evaluate.py",
        find="            r.probe_segments[dom.value] = env.rollout(",
        replace="            r.segments[dom.value] = env.rollout(",
        defect="label-only Tier-2 legs land in segments, which fitness scores",
        suites=("test_search::test_tier2_probe_legs_label_without_scoring",),
        item="ARCH46 2b"),
    Mutation(
        id="critic-ignores-probe-legs",
        path="dytiscidae/evolution/critic.py",
        find="    seg = {**(getattr(result, \"probe_segments\", None) or {}),",
        replace="    seg = {**({}),",
        defect="the probe legs run and the critic never reads them",
        suites=("test_search::test_tier2_probe_legs_label_without_scoring",),
        item="ARCH46 2b"),
    Mutation(
        id="critic-unmeasured-medium-is-zero",
        path="dytiscidae/evolution/critic.py",
        find="        [float(seg[k].competence) if k in seg else np.nan for k in CRITIC_TARGETS],",
        replace="        [float(seg[k].competence) if k in seg else 0.0 for k in CRITIC_TARGETS],",
        defect="a medium Tier-2 never ran (legs after a failure) is labelled as "
               "measured zero, teaching that Tier-2 destroys what it never saw",
        suites=("test_search::test_critic_learns_from_a_cheap_score_of_zero",),
        item="critic label"),
    Mutation(
        id="critic-refuses-a-zero-cheap-score",
        path="dytiscidae/evolution/critic.py",
        find="        if features:\n            self.label(np.asarray(features, float), expensive_outcome(result))",
        replace="        if features and features[0] > 1e-4:\n            self.label(np.asarray(features, float), expensive_outcome(result))",
        defect="the arch45 gate: promotions with a zero Tier-1 mission are not "
               "labelled, which dropped 128 of 147 and every Tier-2 success",
        suites=("test_search::test_critic_learns_from_a_cheap_score_of_zero",), item="critic label"),
    Mutation(
        id="critic-exploit-keeps-its-numbers",
        path="dytiscidae/evolution/critic.py",
        find="    if result is None or getattr(result, \"exploit\", \"\"):",
        replace="    if result is None:",
        defect="a Tier-2 exploit is labelled with the scores it faked",
        suites=("test_search::test_critic_learns_from_a_cheap_score_of_zero",), item="critic label"),
    Mutation(
        id="critic-calibrates-in-sample-on-the-residual",
        path="dytiscidae/evolution/critic.py",
        find="            skills[j] = _skill(np.clip(pj, -1.0, 1.0) + cheap, cheap, yj + cheap)",
        replace="            skills[j] = max(_corr(Zj @ W[:, j] + bias[j], yj), 0.0)",
        defect="calibration read on the residual, in sample: -cheap is a feature, "
               "so noise labels look 0.70 calibrated",
        suites=("test_search::test_critic_learns_the_exploit_signature",), item="critic calibration"),
    Mutation(
        id="critic-takes-credit-for-the-cheap-score",
        path="dytiscidae/evolution/critic.py",
        find="    return float(np.clip(_corr(predicted, observed) - max(_corr(cheap, observed), 0.0),",
        replace="    return float(np.clip(_corr(predicted, observed) - 0.0,",
        defect="calibration counts the cheap score's own accuracy as the critic's skill",
        suites=("test_search::test_critic_learns_from_a_cheap_score_of_zero",), item="critic calibration"),
    Mutation(
        id="critic-can-raise-a-score",
        path="dytiscidae/evolution/critic.py",
        find="        shortfall = float(np.clip(-self.predict(features), 0.0, 1.0))",
        replace="        shortfall = float(np.clip(-self.predict(features), -1.0, 1.0))",
        defect="a predicted positive gap raises the cheap score: a second objective",
        suites=("test_search::test_critic_learns_the_exploit_signature",), item="critic bound"),

    # --- GRPO for the shared policy (ROADMAP N, 2026-10-03) ---------------
    Mutation(
        id="grpo-advantage-across-the-whole-batch",
        path="dytiscidae/learning/grpo.py",
        find="        members.setdefault(k, []).append(i)",
        replace="        members.setdefault(0, []).append(i)",
        defect="the group baseline is the whole batch's mean and spread, so a "
               "body's luck is back in the advantage and GRPO is a worse PPO",
        suites=("test_ppo",), item="N group advantage"),
    Mutation(
        id="grpo-group-ignores-the-segment-kind",
        path="dytiscidae/learning/grpo.py",
        find="            by.setdefault((t.group, t.tag), []).append(t)",
        replace="            by.setdefault((t.group, \"\"), []).append(t)",
        defect="a body's water competence is compared with its crossing quality",
        suites=("test_ppo",), item="N group key"),
    Mutation(
        id="grpo-std-floor-removed",
        path="dytiscidae/learning/grpo.py",
        find="STD_EPS = 0.05\n\n\ndef group_advantages",
        replace="STD_EPS = 0.0\n\n\ndef group_advantages",
        defect="float noise in a group that did nothing is stretched to a unit "
               "advantage, and a group of identical returns divides 0 by 0",
        suites=("test_ppo",), item="N group floor"),
    Mutation(
        id="grpo-rows-renormalised-with-the-batch",
        path="dytiscidae/learning/ppo.py",
        find="            adv = torch.cat([(a_ord - a_ord.mean()) / (a_ord.std() + 1e-8),\n"
             "                             adv[n_ord:]])",
        replace="            adv = (adv - adv.mean()) / (adv.std() + 1e-8)",
        defect="a mixed update standardises the group rows with the ordinary "
               "ones, adding a different kind of number's mean and scale back",
        suites=("test_ppo",), item="N mixed batch"),
    Mutation(
        id="return-split-drops-the-discount",
        path="dytiscidae/learning/ppo.py",
        find="            p[2] += float(disc @ term)\n",
        replace="            p[2] += float(term.sum())\n",
        defect="the terminal part of the reported return is undiscounted, so the "
               "shaping share N10 reads is computed against the wrong denominator",
        suites=("test_ppo::test_the_return_splits_into_shaping_and_terminal",),
        item="N10 return split"),
    Mutation(
        id="return-split-omits-shaping",
        path="dytiscidae/learning/ppo.py",
        find="            shp = rew - term\n",
        replace="            shp = np.zeros(n)\n",
        defect="the reported shaping part is always zero, so N10 would read the "
               "learner as pure competence whatever the shaping does",
        suites=("test_ppo::test_the_return_splits_into_shaping_and_terminal",),
        item="N10 return split"),
    Mutation(
        id="grpo-rollouts-leak-into-the-archive",
        path="dytiscidae/evolution/loop.py",
        find="    del results                      # learning-only: no score leaves this function",
        replace="    for _k, _r in enumerate(results):\n"
                "        _ph, _res, _ct = evaluated[chosen[groups[_k]]]\n"
                "        _place(state, _ph.genome, _ph, _r, _ct, None, [\"grpo\"])",
        defect="the learning-only rollouts are filed as candidates, so a lucky "
               "exploration sample can become an elite's score",
        suites=("test_search::test_grpo_rollouts_never_reach_the_archive",),
        item="N learning-only"),
    Mutation(
        id="grpo-rollouts-count-as-evaluations",
        path="dytiscidae/evolution/loop.py",
        find="    del results                      # learning-only: no score leaves this function",
        replace="    state.evaluated += len(results)",
        defect="the learning-only rollouts advance the evaluation counter, which "
               "names genomes and sets the identification cadence",
        suites=("test_search::test_grpo_rollouts_never_reach_the_archive",),
        item="N learning-only"),
    Mutation(
        id="grpo-pool-drops-the-group-ids",
        path="dytiscidae/envs/actors.py",
        find="                kw[\"groups\"] = [kwargs[\"groups\"][i] for i in idx]",
        replace="                pass",
        defect="a body split across two shards comes back ungrouped, so its "
               "trajectories are dropped from GRPO and the stage buys nothing",
        suites=("test_search::test_grpo_group_ids_survive_the_shard_split",),
        item="N pool"),
)


# --------------------------------------------------------------------------


def copy_tree(dest: Path) -> None:
    """Every tracked file, including uncommitted edits to tracked files."""
    names = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                           capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    tar = subprocess.Popen(
        ["tar", "--null", "-T", "-", "-cf", "-"], cwd=ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    untar = subprocess.Popen(["tar", "-xf", "-", "-C", str(dest)],
                             stdin=tar.stdout)
    tar.stdout.close()
    tar.stdin.write(names)
    tar.stdin.close()
    untar.communicate()
    if untar.returncode:                                     # pragma: no cover
        raise RuntimeError("copying the tree failed")
    # The compiled GPU kernel is untracked, so the copy above leaves it behind,
    # and without it every test that drives the batched evaluator *skips* --
    # which exits 0, which this harness reads as "no suite failed".  A mutant
    # in the batched path therefore survived every run for a reason that had
    # nothing to do with the tests: they never ran.  Link it in, read-only use.
    # Its freshness is still checked against the copy's own `mojo/src`.
    build = ROOT / "mojo" / "build"
    if build.exists() and not (dest / "mojo" / "build").exists():
        (dest / "mojo").mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.realpath(build), dest / "mojo" / "build")


def apply_mutation(tree: Path, m: Mutation) -> str:
    """Apply it, or return why it could not be applied exactly once."""
    p = tree / m.path
    if not p.exists():
        return f"{m.path} does not exist"
    text = p.read_text(encoding="utf-8")
    n = text.count(m.find)
    if n != 1:
        return f"the target text appears {n} times in {m.path}, expected 1"
    p.write_text(text.replace(m.find, m.replace), encoding="utf-8")
    return ""


#: Driver for ``module::function``.  The suites keep their failures in a module
#: -level ``FAILURES`` list and print through ``check``, so one function can be
#: run on its own by importing the module and reading that list afterwards.
_ONE_FUNCTION = """
import importlib.util, sys, traceback
sys.path.insert(0, {tree!r})
# By path, not by package: tests/ has no __init__.py, so it is not importable
# as `tests.<module>` and pretending otherwise fails before the mutation is
# ever exercised -- which would read as a survivor.
spec = importlib.util.spec_from_file_location(
    "suite_under_test", {tree!r} + "/tests/{module}.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
before = len(getattr(mod, "FAILURES", []))
try:
    getattr(mod, "{func}")()
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
fails = getattr(mod, "FAILURES", [])[before:]
if fails:
    print("[FAIL] " + "; ".join(fails))
    raise SystemExit(1)
raise SystemExit(0)
"""


#: Per-suite wall-clock limits.  `test_search` takes ~25 min with the GPU
#: kernel present, so a flat 900 s ended every run of it early -- see
#: ``evaluate`` for why that used to be reported as a catch.
SUITE_TIMEOUT = {"test_search": 3600}


def run_suite(tree: Path, suite: str, timeout: int | None = None) -> tuple:
    """(returncode, stdout+stderr).  Same invocation a contributor uses.

    ``module::function`` runs one test function instead of the whole file.
    That exists because two suites here -- ``test_physics`` and ``test_search``
    -- stop partway on a machine with no GPU fluid extension, and without this
    every mutation in the code they cover would be reported as surviving when
    what actually happened is that the check never ran.  "Not run" and "did not
    catch it" are different results and this is what keeps them apart.
    """
    env = dict(os.environ, PYTHONPATH=str(tree), MUJOCO_GL="disable")
    if "::" in suite:
        module, func = suite.split("::", 1)
        script = _ONE_FUNCTION.format(tree=str(tree), module=module, func=func)
        cmd = [PYTHON, "-c", script]
    else:
        cmd = [PYTHON, str(tree / "tests" / f"{suite}.py")]
    try:
        out = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True,
                             text=True, timeout=timeout or SUITE_TIMEOUT.get(suite, 900))
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout or SUITE_TIMEOUT.get(suite, 900)}s"


def failing_checks(output: str) -> list:
    return [ln.strip() for ln in output.splitlines() if "[FAIL]" in ln]


def evaluate(m: Mutation, workdir: Path) -> Result:
    t0 = time.time()
    tree = workdir / m.id
    copy_tree(tree)
    problem = apply_mutation(tree, m)
    if problem:
        return Result(m, "MISAPPLIED", detail=problem,
                      seconds=time.time() - t0)

    for suite in m.suites:
        rc, out = run_suite(tree, suite)
        # A suite that ran out of time failed nothing: it is evidence of
        # neither outcome.  It used to fall through to "caught", and with the
        # kernel linked in, `test_search` outlasted the old 900 s limit on
        # every mutant -- three batched-path mutations were reported caught at
        # 900.2 s, 900.1 s and 900.2 s with no failing check between them.
        if rc == 124 and out.startswith("timed out"):
            return Result(m, "TIMEOUT", by=suite, detail=out,
                          seconds=time.time() - t0)
        if rc != 0:
            fails = failing_checks(out)
            detail = fails[0] if fails else out.strip().splitlines()[-1][:160]
            return Result(m, "caught", by=suite, detail=detail,
                          seconds=time.time() - t0, failing_checks=fails)
    return Result(m, "SURVIVED",
                  detail="no named suite failed: " + ", ".join(m.suites),
                  seconds=time.time() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="run mutations whose id contains this")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default="", help="write results here")
    args = ap.parse_args(argv)

    chosen = [m for m in MUTATIONS if args.only in m.id]
    if args.list:
        for m in chosen:
            print(f"{m.id:42} {m.path:34} {m.item}")
        return 0
    if not chosen:
        print(f"no mutation matches {args.only!r}")
        return 1

    print(f"{len(chosen)} mutations, each on its own copy of the tree\n")
    workdir = Path(tempfile.mkdtemp(prefix="dyt-mutate-"))
    results = []
    try:
        for m in chosen:
            r = evaluate(m, workdir)
            results.append(r)
            mark = {"caught": "caught  ", "SURVIVED": "SURVIVED",
                    "MISAPPLIED": "MISAPPLD", "ERROR": "ERROR   ",
                    "TIMEOUT": "TIMEOUT "}[r.status]
            where = f" by {r.by}" if r.by else ""
            print(f"  [{mark}] {m.id:42} {r.seconds:5.1f}s{where}")
            if r.status == "caught":
                print(f"             {r.detail[:150]}")
            elif r.status != "caught":
                print(f"             {r.detail[:150]}")
            # Free the copy as we go: 24 trees is a lot of disk.
            shutil.rmtree(workdir / m.id, ignore_errors=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    caught = sum(1 for r in results if r.status == "caught")
    survived = [r for r in results if r.status == "SURVIVED"]
    bad = [r for r in results if r.status in ("MISAPPLIED", "ERROR", "TIMEOUT")]

    print("\n" + "=" * 70)
    print(f"caught {caught}/{len(results)}"
          + (f", {len(survived)} SURVIVED" if survived else "")
          + (f", {len(bad)} could not be applied" if bad else ""))
    for r in survived:
        print(f"  SURVIVED  {r.mutation.id}: {r.mutation.defect}")
    for r in bad:
        print(f"  {r.status}  {r.mutation.id}: {r.detail}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"id": r.mutation.id, "path": r.mutation.path,
              "defect": r.mutation.defect, "item": r.mutation.item,
              "status": r.status, "by": r.by, "detail": r.detail,
              "seconds": round(r.seconds, 2)} for r in results],
            indent=2) + "\n")
        print(f"\nwrote {args.json}")

    # A survivor is a reportable hole, not a crash; a misapplied mutation, or a
    # suite that ran out of time, is a broken measurement and must be loud.
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
