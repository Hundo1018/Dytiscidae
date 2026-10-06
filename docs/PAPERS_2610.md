# Three October papers, mapped onto this code base

Written 2026-10-06, after arch48. The user asked to read three papers and apply
them. Their extractions are in `docs/papers/`:

| id | paper | domain |
|---|---|---|
| [2610.01612](papers/2610.01612.md) | ReCo: response-consistent locomotion with policy-aware MPC (NUS) | legged manipulation, PPO + MPC |
| [2610.02740](papers/2610.02740.md) | Prospective Hindsight: self-calibrating RL via prediction-reality gaps (Salesforce) | LLM-agent RL |
| [2610.03631](papers/2610.03631.md) | NeutronGym: physics-graded instrument design for LLM agents (ORNL) | LLM agents, simulator-graded design |

None of the three is about quality-diversity search, morphology or flapping
flight. Each idea was searched for in the code first (CLAUDE.md, "Before adding
anything"), then either measured on arch48 or built.

**The result is two measurements, not a new method.** Applying NeutronGym's
admission gate and Prospective Hindsight's prediction-vs-outcome table to arch48
found two things:

1. **Tier-1 credit in water and land belongs to one task draw** (§3). A fresh
   seed alone takes Tier-1 water passes from 15/16 to 3/16 and land from 13/16
   to 0/16, before any Tier-2 difference is applied. Elites earn their credit on
   one to three of eight headings. This is NeutronGym's "winner's curse"
   (failure 3), and it is why Tier-1 does not predict Tier-2 (Spearman −0.09
   water, 0.02 land) and why the critic has no skill.
2. **Water competence does not separate the elites from still machines** (§1).
   At bar 0.15: elites 45/200, still 35/200, unsearched base gait 42/200.
   Land does separate (45 / 12 / 29).

ReCo's response model was measured and dropped (§4).

---

## 1. NeutronGym's admission gate: a pass share is evidence only through its bound

**Paper (§5.1, App. H):** before a pass rate counts as capability, a task family
faces probes that use no model: the best fixed answer, a rule that reads the
prompt, and each instance's solution applied to the others. A family fails if
the one-sided 95% Clopper-Pearson upper bound on the best probe's pass share
exceeds a typed 20%.

**Existing before this:** the still-machine rule (CLAUDE.md, "Designing a
measurement" 1, 6 and 7) and `experiments/transition_distance/`. That
experiment covers crossings and air launch heights only ("0 of 218"). Nothing
computed a bound.

**Built:**
- `dytiscidae/domain/evidence.py`: `clopper_pearson`, `no_model_verdict` and
  `distinct_share`, in pure Python.
  - Held by `tests/test_domain.py`, which checks against scipy's beta
    quantiles to 1e-6.
  - Two mutations, both caught.
- **Changed from the paper:** the ceiling is not typed. A bar is *certified*
  when the still machines' upper bound is strictly below the elites' lower bound
  on the same bar, which is what rule 5 asks for.
- `experiments/no_model_gate/run.py` runs every Tier-1 bar through the search's
  batched path with three arms:
  - `elite`;
  - `still` (`held_still_params`);
  - `base` (the body's base gait, no policy, no network): the paper's
    "fixed answer".

**Measured (arch48, n = 200 merged elites, 3206 s; the triphibian island alone,
n = 84, 975 s).** 68 bars were enumerated from the code: ladder rungs
(`judge.py:69`), competence bars (`curriculum.py:66,106`,
`evaluate.py:403`), weakest-medium, stage and crossing bars.

| verdict against `still` | bars |
|---|---|
| certified | 10: land `moves`, `climbs_slope`; land competence 0.012 through 0.30; `c2 >= 0.055`; stages 0 and 1 |
| leak | 28 |
| underpowered | 2 |
| unmeasured (no elite clears it) | 28: air rungs 9 and up, water 5-6, every transition bar, all four crossings |

Against `base`: 0 certified, 40 leak.

Recounted by hand from the per-elite rows:

| bar | elite | still | base | verdict |
|---|---|---|---|---|
| water >= 0.012 | 121 | 103 | 112 | leak |
| water >= 0.15 | 45 | 35 | 42 | leak |
| water >= 0.25 | 23 | 20 | 21 | leak |
| land >= 0.15 | 45 | 12 | 29 | certified |
| land >= 0.25 | 26 | 5 | 14 | certified |

The median elite-minus-still water competence is 0.0001.

**The water leak was verified by a fresh judge**
(`experiments/no_model_gate/verify/`).
- The still arm's `data.ctrl` is constant (max |Δctrl| = 0), and no policy or
  network acts.
- 4 of 5 sampled still passers reproduce exactly on the numpy path.
- Every one is dense (density ratio 1.4-2.0). It sinks at 0.5-0.6 m/s and glides
  sideways at 0.2-0.4 m/s. Hold scores 0.000 in all 20 runs; the whole credit is
  cruise progress (`triphibian.py:1967`, combined at 2116).
- Scatter's velocity kick is *not* the source. Whether the glide lines up with
  the drawn heading is luck: one body glides the other way with scatter off.

Two ways `held_still_params` is not quite still:
- the servos snap the joints from the gait pose to the held offset in the first
  0.5 s (peak 4.4 rad/s);
- a stopped rotor windmills in the flow (38 rad/s, 0.5 N).

`triphibian.py:2069-2085` says water progress is "mostly passive, which the rule
allows". The 2026-09-21 rule was that a glide is a capability and sinking must be
*chosen*. The gate shows that this is not enforced: a body that cannot choose
earns as much as the elites.

The other leaks:
- water `dives` and `goes_deep` (dense bodies sink);
- land `stirs`: an upright still body reads 0.10 m/s peak speed;
- all four takeoff rungs;
- air rungs 1-4, which count the same in all three arms: they measure the
  airframe, not behaviour;
- stage 4's `mission_fraction > 0`, which passes at 1.6e-5.

`three_media` on the triphibian island: elite 1/84, still 0/84 (underpowered),
base 1/84 (a different gannet).

## 2. Concentration: how many distinct designs are behind the passes

**Paper (Table 7):** a family passed 141 times with 42 distinct designs, and
only the count caught it. **Built:** `distinct_share`, reported by the gate per
bar, keyed by body plan. The 35 still water passers are spread over four plans
(gannet 19, beetle 9, eel 4, random 3): the leak is not one body.

## 3. Prospective Hindsight's surprise table, as a Tier-1 → Tier-2 diagnostic

**Paper (§3):** the model predicts its own outcome before the verifier scores
it, and samples where prediction and outcome disagree get loss weight 1 + α.
The evidence is thin:
- no variance in the tables;
- a random-reweighting control also moves the metrics;
- the shared-parameter mechanism is not isolated.

**Not built: the loss weighting.** Its predictor shares the policy's parameters.
Here the critic is a separate ridge model with calibration skill 0.0 on all
three media in arch48, so surprise-weighting with it would weight every sample.

**Built instead:**
- The `promote` event now carries `tier1_media` beside `tier2_media`. It is None
  where meta has no value, never a defaulted 0.
- It is held by a `test_search.py` check and the mutation
  `promote-drops-tier1-media`.
- `experiments/tier_gap/` reads arch45-48 events. For past runs it joins promote
  events to evaluate events by (island, cell); the join rate is 0.39-0.68.

**Measured** (arch48, bar 0.15, legs that ran, n = 121). "Overconfident failure"
means Tier-1 passed and Tier-2 failed.

| medium | Tier-1 pass, Tier-2 fail | Spearman |
|---|---|---|
| water | 35/36 [0.87, 1.00] | -0.09 |
| land | 30/35 [0.72, 0.94] | 0.02 |
| air | — | — (Tier-2 air is exactly 0 for all 121) |

arch47 is the same: water 33/33, land 24/28.

**Decomposed** (`experiments/tier_gap_ablation/`, 6128 s). The ladder walks
from "Tier-1 as scored" to "the Tier-2 leg", one difference at a time:

| rung | water passes | land passes |
|---|---|---|
| R0, Tier-1 as scored (reproduces: water 56/61, land 48/61 within 0.005) | 15/16 | 13/16 |
| **R0s, a fresh seed and nothing else** | **3/16** | **0/16** |
| R1, offline path | 2/16 | 0/16 |
| R2-R6: leg length, default task, scatter, disturbances, network | 0-1/10 | 0-4/10 |

- The real `evaluate_tier2` matches the last rung: land correlation 0.999,
  water 0.96.
- The path makes no difference: R1 vs R0s has correlation 0.98 and median
  difference 0.
- The seed's task draw carries the credit, almost all of it through the heading.
  Setting only the heading to the default π/2 leaves 1/10 water passes and 2/10
  land passes.
- In an eight-heading sweep, each pass earns >= 0.15 on 1-3 headings and nothing
  outside them. Six of ten water passes peak at heading 0: they swim forward and
  do not steer.
- At Tier-1's own draw the passes beat the still machine by +0.17 (water) and
  +0.20 (land). At a fresh seed the median difference is 0.000.

**Mechanism:**
- Tier-1 draws one heading per generation (`tasks.py:206`) and shares it across
  the 16 candidates, and an archive keeps the best of each draw.
- Elites are therefore selected for the heading they happened to face. That is
  the best of 16 on a draw, which is NeutronGym's winner's curse.
- Tier-2 then asks for a different heading.
- The critic's residual (Tier-2 minus Tier-1) is mostly this luck, which
  explains its zero skill.

**The verification that should have caught this was inert.**
- The auditor's held-out-seed check re-ran only when `mission_fraction > 1e-6`.
- arch48's mission was 0 in every generation, so all 32 audits reported
  `retained 1.0` without re-running anything.
- **Fixed:**
  - The held-out seeds now also re-measure each medium the design was credited
    in (competence >= `held_out_floor` 0.05).
  - The audit publishes `held_out` per medium. The auditor report publishes
    `mean_held_out`.
  - `retained` is None when nothing was re-measured.
  - A collapse is a *note*, not an invalidation, so selection is unchanged and
    arch49 stays comparable with arch48.
  - Held by `test_an_audit_re_measures_each_credited_medium_at_unseen_seeds`
    and three mutations, all caught.
  - Cost: two Tier-1 evaluations per audit, about 32 audits per 300
    generations.

## 4. ReCo's response model: does a machine follow the command it is given?

**Paper (§III-B, C):** shape the policy so its closed-loop response is
low-order and consistent across bodies. Then fit a first-order lag (gain k,
time constant τ) per channel and plan over it.

**Existing before this:** `_turn_authority` (`envs/triphibian.py:378`)
correlates commanded and realised angular body rates. That is the static form
of the idea, and it is already in the air score.

**Measured** (`experiments/step_response/`, all 200 arch48 elites × 3 media,
offline path, 1165 s). The fitted model is r[t+1] = a·r[t] + b·c[t]. ΔR² is how
much the command explains beyond momentum.

| medium | ΔR² vs competence | `_turn_authority` vs competence | ΔR² after turn_authority |
|---|---|---|---|
| air | -0.09 | +0.05 | -0.09 |
| water | -0.02 | -0.07 | -0.03 |
| land | -0.03 | +0.03 | -0.03 |

All are Spearman, all p > 0.2. Median ΔR², real vs a shuffled-command control:
- air 0.0022 vs 0.0017;
- water 0.0016 vs 0.0011;
- land 0.0018 vs 0.0020.

**Dropped:** no measurement or descriptor is published. A side reading: the
policy's angular commands explain about 0.2% of the realised body rate.

**Not built: the cross-body consistency penalty.** It needs body randomisation
that the search does not have, and it would be a comparability boundary.

## 5. Already present

- **Within-group reward spread** (NeutronGym §5.2): `reward_by_tag`
  {tag: [mean, std, n]} per PPO update, and GRPO's `grpo_zero_groups`.
- **Tier-2 at a fresh seed:** `_verify_and_label` draws `seed_2` afresh. That
  is why Tier-2 sees the gap.
- **Gate-then-score depth credit:** the rungs, and the Tier-0 feasibility gate.

---

## Fixed, 2026-10-06 (second pass, on the user's "修正")

Items 1, 2 and 4 of the list below are closed by one change; item 3 is
narrowed. The evidence is in `experiments/heading_pair/README.md`.

**The antipodal heading pair** (`tasks.antipode`, `evaluate.run_segment`, and
the same sequence in `batchroll.evaluate_tier1_batch`):
- Water and land are each run twice from one initial state: at the drawn
  heading and at its opposite. The environment's random stream is rewound
  between the halves.
- The cruise term reads the mean of the two halves' signed speeds along their
  headings. Tracking takes the worse half's, hold/stop the mean of the two.
- A velocity that does not depend on the command cancels exactly. That covers a
  dense body sinking along a glide, a body that swims forward without
  steering, the servo snap and a windmilling rotor.
- Still machines and open-loop gaits score |mean| <= 2.8e-17 on both paths.
  An open-loop gait cannot read the command, so it can no longer earn water or
  land cruise progress.
- Tier-1.5 and Tier-2 run the same pair. Tier-2 now draws its headings instead
  of fixing them at π/2.
- Cost: about +18% physics steps per generation (192k on top of about 1.05M).

**Re-gated on arch48** (the same 200 elites, which were selected under the old
score):

| bar | before | after |
|---|---|---|
| water >= 0.012 (elite / still / base) | 121 / 103 / 112, leak | 41 / 2 / 2, certified against both |
| water >= 0.15 | 45 / 35 / 42, leak | 2 / 0 / 0, underpowered |
| land >= 0.15 | 45 / 12 / 29 | 7 / 0 / 0, certified against both |

- Every competence bar, which is what selection pays for, is now certified or
  underpowered. None leaks.
- Against the unsearched base gait, nothing was certified before; now every
  measured competence bar is.
- 20-23% of the elites keep nonzero water or land competence, so selection
  still has a gradient to climb.

**The auditor's perturbation check** had the same defect as its held-out check:
it ran only on a nonzero mission. It now also measures each credited medium,
as a note.

**Not changed:** the judge's ladder rungs that a passive body can stand on:
- water depth;
- land posture;
- takeoff.

They are telemetry and scout input, not fitness, and redesigning them against a
commanded difference needs its own measurement. Until then, read their counts
as state, not capability. Air rungs 1-4 measure the airframe by design.

**Comparability:** water and land competence, mission_fraction, and everything
that reads them (critic, scout, curriculum, islands) are not comparable across
this change.

**Pre-registered reads for the next run:**
1. The auditor's `mean_held_out` per medium is >= 0.5. arch48's elites, at a
   fresh seed under the old score, would read about 0.2 (3/16 of 15/16).
2. Tier-1 vs Tier-2 Spearman per medium on `tier1_media` / `tier2_media` is
   > 0.3 in water and land. It was −0.09 / 0.02.
3. `experiments/no_model_gate` on that run certifies every competence bar the
   elites clear in >= 10% of cases.

## The work list this sets

Ranked by the 2026-09-30 rule: build cost, then loop speed, then what it does
for the search. Items 1-3 change what selection pays for, so each is a
comparability boundary.

1. **Score Tier-1 on more than one heading per candidate, or one draw per
   candidate rather than one per generation.**
   - Without this, every Tier-1 score in water and land is a measurement of the
     draw. The critic, the scout and the curriculum all learn from it.
   - Cheapest form: an independent draw per candidate. It costs nothing, and
     stops 16 candidates competing on one heading.
   - Unbiased form: k headings averaged. It costs k×, and the right k is
     unmeasured: sweep it, as the pool shape was swept.
   - Incumbents keep the score of their own lucky draw either way. So pair it
     with re-scoring the incumbent when a newcomer challenges its cell, or with
     the held-out audit's read.
   - Tier-2's default task fixes the heading at π/2. A verifier at one heading
     has the same defect, so it should draw too.
2. **Make water cruise progress chosen, as land's crossings were (rule 7).**
   - A dense still body earns water credit by sinking at an angle.
   - The cruise term needs the same shape as the crossing fix: gate it on
     progress beyond the still twin's, or on a commanded change of direction.
   - This is the user's 2026-09-21 rule, so it is a decision for the user, not
     a coefficient.
3. **Retire or re-specify the bars the gate found leaking:**
   - water `dives` and `goes_deep`;
   - land `stirs`;
   - the four takeoff rungs;
   - air rungs 1-4 (airframe, not behaviour);
   - stage 4's `mission_fraction > 0`.

   Then re-run `experiments/no_model_gate` until each is certified or
   withdrawn.
4. **Make `held_still_params` still:**
   - hold the joints at the gait pose rather than snapping them to the base
     offset;
   - brake rotors rather than commanding zero speed.

   Both are small. They change the still arm and not the search, so they come
   before 2 and 3 are measured again.
5. **Read arch49's `held_out` audits.** If the per-medium mean is below 0.35,
   item 1 has its pre-registered confirmation inside the run that tests it.
