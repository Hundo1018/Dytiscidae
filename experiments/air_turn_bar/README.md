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

**C (pre-registered 2026-10-19, before any C data; built as `floor_tiebreak`,
commit c7c92f8): does keeping the glider at the floor keep a glider in the
archive?** Two arms, the air island alone, 40 generations, arch51's
configuration (batch 16, 4 workers, 8 s, refine 0, shared policy, gait gain,
mission weight 0, reeval 4/8), seed 20261019, pinned worktree; one arm with
`--floor-tiebreak`, one without. GPU single-lane: after the screen and B'.
Predicted wall 40 x 110-150 s = 1.3-1.7 h per arm.
- C1 (mechanism): at gen 40 the air archive holds >= 3 elites with
  `airborne_fraction` >= 0.5 in the on arm and 0 in the off arm (arch51: 0 of
  139 overall; its gannet seed was rejected at gen 0 on every island).
  *Falsified* if the on arm holds 0.
- C2 (lineage): gannet evaluations after gen 8: on arm > 8, off arm <= 8
  (arch51: 8 in 900 generations, the seeds).
- C3 (the point, reported, not predicted at 40 gens): evaluations with air
  competence >= 0.012, and evaluations with `pair_turn_mean` > 0 at
  `pair_turn_factor` > 0.5 and `spin_rate` < 6.28, per arm. B' says whether a
  random linear policy can turn a gannet at all; if B' is R1, C3 > 0 in the on
  arm is the first air gradient the search has had under the paired score.
- One seed per arm: C1 and C2 are checks of a mechanism that is deterministic
  in direction, not effect sizes. Nothing here is comparable with arch51 on
  `fitness`; `airborne_fraction` and air competence are.

## A, appended after the screen (2026-10-19, unit `atb-screen`, 7218 s)

**6.** Ran as designed: 139 elites x 8 fresh draws (2026101900..07) x 2 arms,
`experiments/air_turn_bar/screen.json` and `.rows.jsonl`, main at c7c92f8
(the floor-tiebreak code is off and not on this path).

**7.** Of 1,112 elite draws, 3 had `pair_turn_factor` > 0; 533 ended "battery
exhausted", 64 "diverged", 48 "unstable". The still arm had 3 measurable draws
(all on elite 3's body), pair means -0.007, -0.007, -0.092: **all <= 0**, as the
strict null requires. Two elite draws (134 at seed ..00, 40 at ..03) carry a
single half (`pair_turn` 1.02 and -1.96, factor 1.0 and 0.43) and no pair mean:
the mirrored half was not measurable, and `triphibian.py` then scores the
*unpaired* half (`if p is not None and "turn" in p` else the single `signed`),
so elite 134 took air 0.025 from one half with nothing to cancel against. An
unpaired half is "could not measure", not a measurement (rule 4); left as an
open item below, not fixed here.

**8/9.** Against the frozen text: verdict `unmeasured` for 139 of 139 (138 with
0 measurable draws, 1 with 1); 0 screen passes, 0 elites with a positive
measurable draw. **R3's third form holds: nothing in arch51's archive is
airborne when the turn is commanded**, so the null-referenced bar cannot be
read on this population. R1/R2 are untouched by A (as frozen); B' decides them.
The bar itself stands: a non-reader reads <= 0 (3/3 still draws), so any
elite with a positive pair mean over K draws is evidence of reading the
command, and the frozen sign test is the conservative form of it.

**10.** Before: the air problem was a scoring/threshold question. After: the
archive holds no body to score; the floor plus first-come emptied the air
island of gliders at gen 0 (addendum above). What would move it back: C's on
arm holding 0 airborne elites at gen 40.

## B', appended after the run (2026-10-19, 859 s, `authority.json`)

**6.** 8 arch49 gannet gliders (rung >= 6, no rotors; 12 were asked for, 8
exist), 32 random linear own policies each at sigma 0.5, one draw, top 3 by pair
mean replicated at 4 fresh draws; the still machine on the same draws.

**7.** Held still, 5 of 8 bodies are airborne through the command on >= 1 of
the 4 replication draws (159: 4/4, 74: 4/4, 80: 4/4, 161: 1/4, 186: 1/4) and
every still pair mean is <= 0 (max -0.034 .. -0.293): the strict null holds on
gliders too. Under random policies at sigma 0.5, 6 of 8 bodies had **0 of 32**
measurable draws, including 159, 161 and 186 that glide when held still: a
random policy of that size takes the glider out of the air before the
command. Body 74: 6/32 measurable, 3 positive, best 0.307 on the screen draw;
its top 3 replicated at 0.007 (n = 3). Body 80: 3/32, best 0.111; replicated
-0.037.

**8/9.** Against the frozen text: "R2 predicts: no body's best replicated mean
exceeds 0.1" -- **0 of 8**, R2's prediction held, R1's falsifier fired. But
station 4 was weak: the probe measured "random sigma-0.5 linear policies", and
most of those crash the glide, so the read is "a random policy does not turn a
gannet", not "a gannet cannot be turned". The single-draw 0.307 that did not
replicate is the winner's curse the design anticipated. **Classification:
confirmed for R2 as frozen; authority at a policy scale that keeps the glide is
untested.**

**B'' (pre-registered now, before data): the same probe at sigma 0.1 and
0.02** on the same 8 bodies, same seeds. R2 predicts 0 bodies > 0.1 at both;
R1 predicts >= 1 at one of them with measurable draws >= 8/32 (the glide kept).
Queued after C (one GPU lane).

## B'', appended after the run (2026-10-20, `authority_s0.1.json`, `authority_s0.02.json`)

**6/7.** Same 8 bodies, 32 policies, same seeds. Measurable draws per body
rose as sigma fell: sigma 0.5 -> 0.1 -> 0.02 gave body 159 0 -> 1 -> 18 of 32,
body 74 6 -> 14 -> 25, body 80 3 -> 2 -> 19. Every still pair mean <= 0.

**8/9.** Sigma 0.1: 0 bodies with a replicated mean > 0.1 (74: screen 0.294,
replicated 0.018 at n = 1). **Sigma 0.02: body 159 (recorded air 0.422, rung 7)
replicated at 0.119 over 4 of 4 fresh draws** with 18/32 measurable on the
screen; body 80's screen 0.446 replicated at 0.05 (n = 4), body 161's 0.082 at
n = 1. Against the frozen text: "R1 predicts >= 1 at one of them with
measurable draws >= 8/32" -- **R1 confirmed at sigma 0.02**, R2 refuted. A
gannet airframe can be turned 12% of a perfect turn by a linear policy that
reads the heading error, without learning; the policy scale that does it is
25x smaller than the search's sigma-0.5 initialisation would suggest.

## C, appended after the run (2026-10-20, units `atb-chain`, `runs/atb_c_off`, `runs/atb_c_on`)

**6.** Both arms ran as designed: `checkpoint.json` provenance has
`islands ['air']`, `floor_tiebreak` False / True, `gait_gain` True,
`reeval_per_generation` 4, seed 20261019; 40 generations each, 5,220 s and
5,023 s (131 and 126 s per generation; arch51 146). Worktree `../atb-tree`
at 1fc303e.

**7.** C1's archive count matches elites to their evaluation by (gen,
eval_seed); 28 (off) and 15 (on) elites did not match (re-evaluated elites
carry their median draw's seed), so C1's counts are lower bounds.

**8/9, against the frozen text.**
| read | predicted | C-off | C-on | verdict |
|---|---|---|---|---|
| C1 archive elites airborne >= 0.5 at gen 40 | on >= 3, off 0 | >= 5 | >= 5 | on half met; off half **missed** (early floor members are airborne too; arch51's 0/139 was the end of 900 gens) |
| C2 gannet evaluations after gen 8 | on > 8, off <= 8 | 0 (1 in all: the seed, `rejected` at gen 0) | 140 (165 in all; the seed `improved` at gen 0, by the tiebreak) | **both met** |
| C3 evaluations with air >= 0.012 | reported | 0 | 3: gen 16 (0.014), 22 (0.013), **27 (0.111)**, all gannets, no rotors | first air credit under the paired score |
| C3 clean positive reads (pair mean > 0, factor > 0.5, spin < 6.28) | reported | 0 | 3, the same three; gen 27: pair mean 0.204, airborne 1.0, spin 0.67, no air gate | |
| air archive at gen 40 | | 52 elites, 0 gannets, max air 0 | 46 elites, 14 gannets, max air 0.111 | |

The gen-27 gannet (operators `duplicate_part`, `body_field`) is the on arm's
best elite (fitness 0.5, above the floor). **Cross-validated (station 11):**
the post-run film re-ran it through the single-machine path at its eval seed
and reproduced air 0.111 = 0.111, water 0.033, land 0.065 (`runs/atb_c_on/postrun.log`).
Fresh-draw retention inside the run (N11 re-evaluations of gannet elites):
gen 11 air 0.147, 19 0.5 -> 0.0, 23 0.5 -> 0.5 (air 0.107), 27 0.0 -> 0.5 (air
0.097 at a fresh seed), 28 and 29 0.5 -> 0.0. The read is real on some draws
and absent on others: a weak reader, which is what B'' found (0.119 at sigma
0.02). The screen (A's K = 8 fresh draws) on this archive is queued
(`screen_c_on.json`) and the no-model gate on it (`gate_c_on.json`).

**10.** Before: air had no gradient under the paired score because nothing was
aloft when the command came, and the floor had removed the gliders. After: with
the tiebreak the glider lineage survives at the floor, and within 27
generations the search produced a gannet that turns toward the command on a
fresh seed and holds air 0.111 through the other evaluation path. What would
move it back: the screen on C-on reading 0 elites with a positive measurable
draw, or the gate certifying nothing against the still machine.

**12.** Reproducers: `runs/atb_chain.sh` (C), `runs/atb_chain2.sh` (B''),
`experiments/air_turn_bar/{run,authority}.py`; commits c7c92f8 (code) and
1fc303e (worktree).

## Exploration read (pre-registered 2026-10-21, before data; air list item 6, the learner's half)

`exploration.py`: the 8 arch49 gannet gliders (as B'), own policy none, the shared
network's actor output zeroed so only its sampling noise acts, `log_std` in
{-0.5 (the search's init), -1.5, -2.5, -4.0}, 32 sampled copies per body per value at
one seed, bases stored. Read: the share of copies aloft through the command
(`pair_turn_factor` > 0.5) and the share of those with a positive pair mean.
- Prediction (from B''): at -0.5 (std 0.61) the measurable share is <= 25%; at -2.5
  (std 0.08) it is >= 50%. *Falsified* if -0.5 reaches 50% or -2.5 stays under 25%.
- Decision: if confirmed, arch55 starts the shared policy at the log_std whose
  measurable share first exceeds 50% (an init, not a clamp; PPO still learns it).
- Positive share is reported, not predicted: noise cannot read the command.

## Exploration read, appended after the run (2026-10-22, `exploration.json`, 2,930 s)

8 gannets x 32 sampled copies per `log_std`:

| log_std (std) | measurable | positive of measurable | bodies with >= 16/32 measurable |
|---|---|---|---|
| -0.5 (0.61), the search's init | 9/256 (3.5%) | 0 | 0 |
| -1.5 (0.22) | 75/256 (29%) | 13 (17%) | 2 (159, 74) |
| -2.5 (0.08) | 103/256 (40%) | 9 (9%) | 3 (159, 74, 80) |
| -4.0 (0.02) | 98/256 (38%) | 0 | 3 |

Against the frozen text: "-0.5 <= 25%" held (3.5%); "-2.5 >= 50%" **missed** (40%);
neither falsifier fired. Three bodies (161, 175, 118) are measurable at no std: not
gliders at this seed. The decision rule ("the log_std whose share first exceeds 50%")
selects nothing; the measurable share saturates at -2.5 and the positive share peaks
at -1.5. **Decision, typed and said so:** arch55 initialises `log_std` at -2.0, between
the two, as a `--policy-log-std` flag; PPO still learns the parameter.
