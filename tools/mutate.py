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
        id="island-archive-read-through-the-merge",
        path="dytiscidae/ops/run.py",
        find="    if island is not None:\n        if island not in names:",
        replace="    if False:\n        if island not in names:",
        defect="one island's archive is read through the cross-island merge, so "
               "an elite that lost its cell to another island is never filmed",
        suites=("test_search::test_one_islands_archive_is_read_alone_not_through_the_merge",), item="per-island best"),

    Mutation(
        id="curriculum-reads-every-medium",
        path="dytiscidae/evolution/curriculum.py",
        find="    if domains is not None:\n        segs = {d: s for d, s in segs.items() if d in domains}",
        replace="    if False:\n        segs = {d: s for d, s in segs.items() if d in domains}",
        defect="a specialist island's curriculum pays for another medium again, "
               "so the air and land islands fill with water machines",
        suites=("test_search::test_a_specialist_islands_curriculum_reads_only_its_own_medium",), item="island purity"),

    Mutation(
        id="bandit-without-an-exploration-floor",
        path="dytiscidae/evolution/curator.py",
        find="                 epsilon: float = 0.2) -> None:",
        replace="                 epsilon: float = 0.0) -> None:",
        defect="the tilted argmax alone decides every slot, so a non-structural "
               "operator with a good mean can go unpicked for 80 generations",
        suites=("test_search::test_no_operator_can_go_dormant_under_the_structural_tilt",), item="operator dormancy"),

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
        find="        elif cross >= 0 and medium_of(env) is not self.target:",
        replace="        elif False:",
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
