# Air L1: a bar referenced to the null, not to the population — 2026-10-19, b0ab10a

Context: E2's air L1 bar could not be set from arch51's population (ROADMAP
2026-10-19): no elite reaches air 0.012 and every rung an elite clears, the still
machine clears as often. The user asked whether a statistical, dynamic threshold
is admissible under rule 5. Answer (this file): rule 5 *requires* thresholds from
the measured distribution; what it forbids is a bar the inert machine also clears.
A population quantile always yields a number, so it is not a bar. A bar defined
against the null (what a machine that cannot read the command scores) is.

## Frozen before the run

**0 Question.** Does any arch51 elite turn toward the commanded side more often
than a machine that cannot read the command would, over K fresh draws? Decision:
whether air L1 can be stated as a null-referenced bar today, and whether the next
work is turn *authority* (bodies) or turn *control* (policy).

**1 Prior art.** `experiments/no_model_gate/results_arch51.json` (0/139 at air
0.012; rungs leak). ROADMAP "Air on a commanded difference": 13 arch49 gannets,
mean signed turn negative in 12/13, one draw each. `tasks.mirror`,
`triphibian.py:2160-2197` (`pair_turn`, `pair_turn_mean`, `pair_turn_factor`).
The policy observes the heading error as (cos, sin) (`task_channels`), so the
command is visible to it. Rivals:
- R1 *control*: bodies can turn, the policy never learned to map heading error to
  a turning action (the search's air credit was zero throughout arch51, so PPO had
  no air reward: `return_by_tag` air 0).
- R2 *authority*: the air bodies have no actuator pattern that yaws/rolls them;
  no policy could turn them.
- R3 *instrument*: `pair_turn_mean` sign or the mirror is wrong, or the factor
  (served x airborne) is 0 so nothing is measurable.
- R4 *nothing*: the signed response is noise; signs split 50/50 across draws.

**2 Mechanism.** The pair mean cancels any response that does not depend on the
command: an open-loop or still machine reads exactly 0 (the heading-pair test,
|mean| < 1e-12). A positive mean on a draw means the second-half velocity moved
toward the commanded side *in both mirrored halves*, which only a command-reading
controller can produce. Over K draws the sign of a non-reader is undefined (0) and
of a random reader is a fair coin.

**3 Predictions.** Per elite, K = 8 fresh draws (seeds 2026101900..07, never used
by the search), sign test on `pair_turn_mean` over the measurable draws, one-sided
binomial p < 0.05 (8/8: p = 0.0039; 7/8: 0.035; 6/7: 0.0625 fails).
- Family: under R4 with 139 elites all measurable, expected screen passes 139 x
  0.035 = 4.9 (95% upper ~ 9). **A screen pass is a result only if it replicates**
  on 8 further draws (seeds ..10..17); expected replicated passes under R4: 0.2.
- R1/R2 both predict **0 replicated passes** (bodies can't, or policy didn't).
  They are separated by experiment B below, not by A.
- R3 predicts measurable draws < K/2 for most elites, or a still machine with a
  nonzero mean.
- Falsified (the "nothing is there" reading) if >= 1 elite replicates, or if the
  count of screen passes exceeds 9.
- Magnitude, reported not gated: mean `pair_turn_mean` and its 95% CI per elite.

**4 Measurement.** Metric: `segments['air'].measurements['pair_turn_mean']` from
`batchroll.evaluate_tier1_batch` (the search's path), elite arm = own policy +
the network that scored its generation, as the gate does. Inert probe: the still
arm (`held_still_params`, no policy) on the same draws must read |mean| < 1e-9 on
every measurable draw. Missing vs zero: a draw with `pair_turn_factor` == 0 (not
served or not airborne) is *unmeasured*, excluded from n; an elite with n < 4 is
`unmeasured`, not failed. Noise floor: the per-elite spread of `pair_turn_mean`
over the 8 draws, reported. Threshold provenance: p < 0.05 against the null of a
fair coin; no population quantile anywhere.

**5 Design.** All 139 merged arch51 elites (same population as the gate). Read
once after the screen, once after replication of the screen passes. Stopping:
fixed K. What varies: nothing but the draw. Controls: still arm (inert), and the
R4 family-wise count. Power: at K = 8 a true reader with P(right side) = 0.9
passes 8/8 or 7/8 with prob 0.81; replication 0.66 — a weak reader (0.7) is
missed (0.26 x 0.26), stated.

**Experiment B (pre-registered, run after A, on the elites A found airborne in
>= 4 draws, capped at 20 by recorded air rung):** turn authority without
learning. For each body, 32 random linear own policies (weights ~ N(0, 0.5),
the search's `Policy(hidden=0, gain=cfg)`), no shared network, one draw each
(seed 2026101920); the top 3 by `pair_turn_mean` re-run at 4 fresh draws.
- R2 predicts: no body's best replicated mean exceeds 0.1 (a 10% turn).
- R1 predicts: >= 1 body in 3 reaches a replicated mean > 0.1 — authority exists
  and the policy is what failed. Falsifier for R1: 0 bodies.

## Appended after

**7 (found at the smoke run, before any full data; written 2026-10-19 before
the screen finished).** Two instrument facts, from 2 elites x 2 draws:
- The still machine's `pair_turn_mean` is **not zero**: -0.051 and -0.009 on
  elite 50's body. Reason, from `triphibian.py:2176-2189`: the gain is
  `min(v_b . n, (v_b - v_a) . n)` and the mirror flips `n`, so the second half
  reads `min(-a, -b) = -max(a, b)` and the pair mean of a non-reader is
  `-0.5 |a - b| <= 0`, never 0. The heading-pair test's "cancels to 1e-12" is for
  water/land's linear progress, not for air's clipped gain. Consequence: under
  the mirror a non-reader cannot score positive, so the pre-registered fair-coin
  null is *conservative* (harder on elites than the true null, P(+) = 0 for a
  deterministic non-reader). The frozen test stands; the count of elites with
  >= 1 positive measurable draw is reported beside it as the strict-null read.
  This also explains ROADMAP's "mean signed turn negative in 12 of 13 gannets":
  the sign was the statistic's, not the gannets'.
- Both elites had `pair_turn_factor` = 0 on every draw (one draw "battery
  exhausted"): not airborne when the turn was commanded (the command comes at
  0.5 x 8 s; the best-by-mission elite's `airborne_seconds` is 2.6). If this
  holds across the 139, A's verdict is `unmeasured` for the population and R3's
  third form ("nothing is measurable") wins; experiment B's population (airborne
  in >= 4 draws) would then be empty, and B needs a new file.

**1/7 addendum (2026-10-19, during the screen, from `runs/*/events.jsonl`).**
Why no arch51 elite is airborne when the turn is commanded: the run never held a
glider. The gannet seed was `rejected` at gen 0 on every island (`air` 0.0 under
the paired score, `at_floor` 1); gannets were evaluated 8 times in arch51 (the 8
seeds) against 1,172 in arch49, and arch51's air island was eels (634) and randoms
(508). Under `air_task = turn x (0.5 + 0.5 height)` a glider that does not turn
scores 0, the floor (N5) rejects a 0 from the archive, so the only lineage that
stays aloft through the command phase was gone before any policy could learn to
turn it. The 14 arch51 evaluations with a positive pair mean (up to 0.447) are
rotor eels spinning at 13-65 rad/s (`MAX_SPIN_RATE` 6.28), gated to 0.05 credit
by `air_gates`; they are tumbles, not reads. arch51 evaluations airborne through
the command phase (`airborne_fraction` >= 0.5): 276 of 10,388; arch49: 379 of
4,749, 311 of them gannets. (`airborne_seconds` is written only on the short-hop
branch and caps at 2.796 s; the long branch writes `airborne_fraction`.)
Experiment B as frozen has an empty population in arch51; B' below replaces it.

**B' (pre-registered 2026-10-19, before B' data): turn authority on bodies that
glide.** Population: arch49's archive elites with `body_plan` gannet, no rotors,
air rung >= 6 (`stays_up`), capped at 12 by recorded air. Method as B: 32 random
linear own policies per body (weights ~ N(0, 0.5), `Policy(hidden=0)`), no
shared network, one draw (seed 2026101920), top 3 by `pair_turn_mean` re-run at
4 fresh draws (2026101921..24). The strict null (station 7) is that a non-reader
reads <= 0, so a replicated mean > 0.1 is a read with 10% of a perfect turn.
- R2 (no authority): no body's best replicated mean > 0.1.
- R1 (authority, policy failed): >= 1 body's best replicated mean > 0.1.
- R3 check: the same bodies held still must read <= 0 on every measurable draw.
Decision: R1 -> the air problem is admission (keep gliders at the floor so the
turn can be learned), not actuation; R2 -> the gannet airframe needs a control
surface before any air score can move.
