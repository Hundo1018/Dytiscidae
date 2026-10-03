# arch46 — design spec

Written 2026-10-03, after arch45 (900 generations, `runs/arch45`). It answers
an outside proposal that combined SAIL, multi-fidelity residual models,
Deb / constrained MAP-Elites, T-DominO, RUDDER, AURORA / CVT-MAP-Elites and
PGA-MAP-Elites. This document maps each of those onto this code base. For each
one it says whether arch45's data supports it, what already exists, and how it
would be falsified.

Ranked by the 2026-09-30 rule: build cost first, then loop speed, then what it
does for the search. Every number below was measured from
`runs/arch45/events.jsonl` on 2026-10-03, with the script quoted in its item.

---

## 0. What arch45 measured, and which of the proposal's premises hold

| claim in the proposal | measured | verdict |
|---|---|---|
| the critic only labels when Tier-1 mission > 1e-4 | `loop.py` gate; 147 promotions, **128 dropped**, 19 kept, all labelled 0.0 | **true, and worse**: final report `fitted=0, fits=0, calibration=0.0`. The critic did nothing for 900 generations |
| it learns a Tier-2/Tier-1 ratio, ill-posed at zero | `critic.label(…, r2.mission_fraction / cheap)` | true |
| `Tier-1 = 0, Tier-2 = 0.1667` is the key counter-example of cheap evaluation missing a success | 4 promotions, all exactly 0.1667 | **false as stated**. Tier-2's mission is `completed_legs / 6` from a **random start domain**, stopping at the first leg below 0.15. 1/6 means "the first leg drawn was a medium this body can do". Tier-1's mission is gated on the weakest medium. The two are different quantities |
| "13,924 evaluations, 1 non-zero mission" | 13,924 evaluations, **169** with mission > 0, 94 above 1e-4 | false. The scarcity point stands (1.2%) |
| infeasibility rises with part count, 23.6% at 2 parts → 80.8% at 10 | 23.6% at 2, 46.2% at 4, 57.4% at 6, 67.4% at 8, **78.1%** at 10, 100% at 14+ (n=11). Mean parts 4.2 → 6.3, infeasible 37.1% → 60.5%, first vs last 100 gens | true; the arch34 finding (ROADMAP item E), reproduced |
| refits cause a coverage sawtooth | 34 refits, median 85 cells merged (435 → 343), at a median subspace overlap of **0.974** | true. The fix already exists (§3) |
| fitness ~0.75 while the mission is not completed | median fitness 0.714 (p90 0.93). Share of evaluations with competence > 0.15: air **1.1%**, water 9.0%, land 6.5%; none in all three. A fitness-1.0 generalist at gen ≥ 800 had air 0.000, water 0.337, land 0.003, mission 0 | true, and it is the largest open question (§4) |

Two more measurements the proposal did not have:

- **Tier-1.5 measured nothing in 124 of 147 promotions.** It runs the
  *weakest* medium (`evaluate.weakest_domain`), and for these bodies the
  weakest medium's Tier-1 competence is 0. Retention is then undefined, so only
  23 promotions produced a retention value, with median 0.0. Tier-1.5 picked
  air in 100 of 147.
- **Tier-2 never runs the media after the first failure.** 143 of 147
  promotions failed their first leg, so most Tier-2 results contain exactly one
  medium.

---

## 1. Done: the critic learns per-medium residuals — **built 2026-10-03, unrun**

This is the proposal's first item, corrected by §0.

- **Target.** The target is `expensive − cheap` for each of air, water and land
  competence (`critic.CRITIC_TARGETS`), not the mission. Both tiers use the same
  scorer for competence; they do not use the same definition of mission.
- **No gate.** Every promotion is a label. An auditor invalidation is labelled
  as losing every cheap score. A Tier-2 exploit is labelled as retaining
  nothing.
- **Unmeasured is not zero.** A medium Tier-2 never ran is stored as NaN, and
  each target is fitted only on its measured rows. Scoring it as 0 would teach
  the critic that Tier-2 destroys competence it never looked at.
- **Calibration is skill over the cheap score.** It is computed out of fold, on
  the implied expensive outcome, as `corr(L + Δ̂, H) − corr(L, H)`. Two holes
  were found while building this, and both are now tests:
  - Calibrating on the residual in sample read **0.70 on pure-noise labels**,
    because the residual contains `−L` and `L` is a feature.
  - Calibrating on `H` alone credits the critic with Tier-1's own accuracy.
- **Unchanged.** The discount is still one-directional and bounded. A predicted
  *positive* gap moves no score.
- **Telemetry.** The `promote` event now carries `tier2_media`, so the next
  run's labels can be replayed offline. arch45's cannot, because the per-medium
  Tier-2 values were never written.
- **Resume.** A checkpoint's ratio labels are dropped on restore and counted
  (`dropped_legacy`).
- **Held by** `test_critic_learns_from_a_cheap_score_of_zero` and six mutations
  in `tools/mutate.py` (`critic-*`), all caught.

**Pre-registered read.** At ~0.16 promotions per generation, `min_samples=60` is
reached near generation 370, and each medium sees about a third of the labels.

- **Predicted:** by generation 500, at least one target has a per-target skill
  above 0.05, and `calibration_by_target` is non-empty.
- **Falsified if:** every target's skill is ≤ 0.05 at generation 600. That
  would mean the 16 cheap features do not carry the Tier-1/Tier-2 gap, and no
  surrogate built on them (§6) is worth building.

---

## 2. Tier-1.5 and Tier-2 label what they can — **low cost, no loop cost**

The cheapest way to get more labels is to stop wasting the expensive runs that
already happen.

**2a. Tier-1.5 runs a medium with something to retain.** In
`loop._verify_and_label`, run the weakest medium *among those with Tier-1
competence > 0.15*. Fall back to the weakest overall only when none qualifies.
The 0.15 is the Tier-2 leg-failure bar, so this adds no new threshold.

- Change: one function next to `evaluate.weakest_domain`.
- **Prediction:** retention is defined for ≥ 60% of promotions (arch45: 23 of
  147, 16%).
- Mutation: revert to `weakest_domain`. It must be caught by a fixture with
  competences `(0, 0.3, 0.6)`.

**2b. Tier-2 runs one leg in every medium, for labelling only.**
`mission_fraction` keeps its meaning: it stops at the first failure. Each medium
not yet visited gets one extra compressed leg, which goes into `segments` and is
marked `label_only`. That turns ~1 label per promotion into 3.

- Cost: up to two extra 25 s legs per promotion. Read AL's `stage_wall` before
  and after.
- **Prediction:** labels per medium triple, and the critic reaches
  `min_samples` before generation 150 instead of ~370.
- **Falsified (as worth its cost) if:** Tier-2's stage wall grows more than 10%
  of a generation's wall.
- Mutation: leaving `label_only` legs inside `mission_fraction`. It must be
  caught by a fixture whose first leg fails and whose second would pass.

---

## 3. Refit: switch on the guard that exists — **trivial; the proposal's dual archive is not needed**

`descriptor_keep_if_overlap` (`descriptors.py:187`) already skips a refit whose
subspace overlaps the current one above a threshold. It ran at 0.0 (off) in
arch45. Measured on arch45's 33 refits that carry an overlap:

| threshold | refits skipped | merges avoided |
|---|---|---|
| 0.90 | 25 / 33 | 2238 / 3066 |
| 0.95 | 19 / 33 | 1601 / 3066 |
| 0.97 | 17 / 33 | 1424 / 3066 |

The proposal's stable/experimental two-archive design solves the same problem
with new machinery. Reuse the existing guard. Set it at **0.95**, the threshold
that skips a refit only when its axes have barely moved, which is the median
case. Overlap and merges correlate at only −0.45, so a merge is not purely a
cost of axis motion. Some refits merge many cells even at high overlap.

- **Prediction:** per-island filled cells between consecutive refits stop
  falling across the run. In arch34 the archive size at each refit fell
  105 → 77.
- **Falsified if:** coverage still sawtooths by more than 3 points at skipped
  refits. That would mean the merges come from re-binning, not from the axes.
- No new mutation is needed: `tools/mutate.py` already holds the guard
  (line 692), and `test_search.py` tests it.

---

## 4. Measure before reshaping: what does fitness pay for? — **measurement, cheap**

The proposal puts T-DominO on the generalist island to stop a scalar from
hiding a weak medium. arch45 shows something more basic. Fitness is
`blend(island_q, curriculum_q, mission_q)` (`loop.py`, `score_parts`), all
rung- and curriculum-relative. It reached 1.0 on a body whose best medium was
water at 0.337, with mission 0. Correlation between fitness and the best medium's
competence, per island at gen ≥ 800, ranged from 0.33 (air) to 0.73 (amphibian).

Before replacing the scalar, decompose it. For the archive at gens 300, 600
and 899, report how much of `fitness` each of `island_q`, `curriculum_q` and
`mission_q` carries. Check whether `mission_q = 1.0` at `stage_name = chain`
can coexist with `mission_fraction = 0`, and why.

- The decomposition decides between three fixes: a gate (the project's
  standing answer), T-DominO, or neither.
- **Not built until read.** The lesson that applies is "when a score pays for
  doing nothing, first ask whether the machine was ever told what to do".

---

## 5. Energy: feasibility first in the cell's front — **low-medium; ROADMAP item E**

The archive already keeps a per-cell Pareto front over `(mission, structure,
energy)` (`archive.add` → `_verdict`, MOME). Energy is an *objective* there, so
a large mission gain can always buy an infeasible design a place on the front.
Deb's rule turns that into a *constraint*:

- feasible beats infeasible;
- between two infeasible designs, smaller `max(0, −energy_margin)` wins;
- between two feasible designs, the existing dominance applies.

Changes: `_verdict` and `_representative` in `archive.py`, behind
`--energy-constraint` (off by default, like every arch45 item).

- **Prediction:** the infeasible share over the last 100 generations is below
  the first 100's (arch45: 37.1% → 60.5%), and infeasibility at 8+ parts falls
  below 50% (arch45: 67–100%).
- **Falsified if:** the infeasible share still rises with generation. That would
  mean the pressure comes from the variation operators, not from replacement.
- **Watch:** coverage. With 37–60% infeasible, some cells will hold only
  infeasible designs, and the rule must not empty them. An infeasible design
  stays when nothing feasible is there.
- Mutation: swap the first two rules. It must be caught by a fixture with one
  feasible low-mission design and one infeasible high-mission design in the
  same cell.

This is a selection change and §1–§3 are not, so it is an arm, not part of the
bundle (see "Bundle changes that measure different things").

---

## 6. SAIL, reduced to what keeps the critic's invariant — **medium; conditional on §1's read**

SAIL's acquisition `μ + κσ` *raises* scores for uncertain designs. The critic is
built so that it can never raise a score, because a score-raising model becomes
a second objective for the population to exploit. This project has paid for
seven such exploits. So the acquisition goes where it cannot be optimised
against: **choosing what Tier-2 verifies.**

- Today `_verify_and_label` takes the top 3 by fitness.
- Change: take 2 by fitness and 1 by `promise = max(Δ̂, 0) + κ·σ̂`. `σ̂` is the
  ridge posterior standard deviation, which is cheap for a linear model. The
  third candidate is the one the critic thinks Tier-1 *undersells*, or knows
  least about.
- **Only if §1's read shows skill > 0.05.** A critic without skill has no
  promise to rank by.
- **Prediction:** the acquisition-picked promotion has a higher mean Tier-2
  per-medium competence than the third-by-fitness one did. Compare against
  arch45's third pick.
- **Falsified if:** it is no better over ≥ 30 rounds.

Not built: a GP or a multi-head surrogate. With 147 labels per run, the bottleneck
is label count (§2), not model class. The proposal's 6 s / 24 s / 60 s / Tier-2
fidelity ladder is also out for now. A 24 s rung costs a full leg on every
candidate that reaches it, and arch45's Tier-1.5 shows the 60 s rung has not yet
told us anything (§0).

---

## 7. The learner: PGA-MAP-Elites after the objective, RUDDER after that — **high**

ROADMAP Phase 3 names PGA-MAP-Elites as *the indicated direction* for the
learner, and RUDDER as a later replacement for the hand-written potential. The
proposal ranks PGA last, for a reason this spec accepts. A gradient operator
climbs whatever fitness is, and §4 shows fitness is not yet competence. So
**PGA waits on §4's read**, and RUDDER waits on PGA. Phase 3's argument for PGA,
that four null results mean there is no shared controller of this class to find,
is unchanged. Only the order moves.

The proposal's stage-progress potential (`Φ = stage progress + distance to next
transition + …`) overlaps Y/O, the transition-distance curriculum, which is
built and off. Use Y/O rather than a second potential.

---

## The bundle for arch46

Bundle §1 + §2 + §3. Each one changes what the run *measures*, not what it
*selects*, and their reads do not overlap. §5 is an arm on its own. §6 starts
only after §1's generation-500 read, and §7 only after §4.

| item | read | falsified if |
|---|---|---|
| §1 critic | per-target skill at gen 500 | all ≤ 0.05 at gen 600 |
| §2a Tier-1.5 | share of promotions with defined retention | < 60% |
| §2b Tier-2 labels | gen at which the critic first fits | > 150, or Tier-2 wall up > 10% |
| §3 refit guard | per-island filled cells across skipped refits | sawtooth > 3 pts remains |
| §4 fitness | decomposition at 300 / 600 / 899 | n/a, a measurement |
| §5 Deb (arm) | infeasible share, last vs first 100 gens | still rising |
