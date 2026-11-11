# Why no machine does the three-media mission — frozen before the evidence pass

2026-11-10 21:46 +08, commit a7770c2. Written from the ROADMAP's dated head
(lines 1-385), CLAUDE.md and the memory index only, BEFORE the forensics,
code and literature reports under `reports/` were read. Nothing below this
line is edited after the reports return; outcomes go in `DIAGNOSIS.md`.

Constraint from the user (2026-11-10): no new training until the analysis can
support a plan that reaches normal triphibian locomotion within three full runs.

## 0 Question, and the decision that hangs on it

Why has the continuous mission (air, water, land and the crossings between
them) read 0 on every run from arch34 to arch60, and what is the smallest set
of changes after which three full runs are enough?

Decision: what gets built and measured offline before any launch, and what
the (at most three) runs are.

## 1 Rival explanations

- **R1 physics.** No body in the reachable design space can do all three media
  and the crossings under the actuator, power and fluid model.
- **R2 control authority.** The controller parameterisation (gait parameters
  plus whatever feedback the policy has) cannot produce commanded manoeuvres
  (a turn in the air, hold-then-go at a boundary) on any body.
- **R3 learner starvation.** The shared PPO sees too few transitions per
  (body, task) and its bodies change under it, so no closed-loop skill forms.
- **R4 search signal.** Selection does not accumulate (one-draw noise,
  quantile fitness, Tier-2 inversion); the archive is about as good as random
  sampling.
- **R5 chain sparsity.** The mission is a conjunction of several rare events
  and nothing pays for a partial chain or starts a crossing from a state the
  machine can reach.
- **R6 instrument.** The gates added against still-machine leaks leave a
  passing set so small that a purpose-built reference fails too, or the
  mission evaluation has a defect.
- **R7 nothing wrong, too short.** More generations of the same would get there.

## 2 Mechanism I hold before reading (prior weights)

R5+R3 0.35, R2 0.20, R1 0.20, R4 0.15, R6 0.08, R7 0.02.

Stated as a mechanism: the mission needs a closed-loop controller that
sequences skills; the system has one small shared policy fed a few thousand
transitions per generation across hundreds of changing bodies, an archive that
scores each body on one draw of one medium, and no start state that puts a
competent body at a boundary. A crossing is therefore never attempted from a
state where it could succeed, so its reward is identically zero and neither
the learner nor the search has a gradient toward it.

## 3 Predictions on the stored data (each can fail)

1. **R7 dies if** mission and every crossing count are exactly 0 (not small and
   rising) in every run arch47-arch60. Predicted: exactly 0 in >= 95% of
   generations of every run; no run shows a rising crossing count.
2. **R5 predicts** crossings are not failed, they are not started: the share of
   crossing attempts that satisfy the hold precondition is < 5% pooled over
   arch47-arch60. If attempts start at > 30% and fail in the go phase, R5 is
   wrong and R1/R2 take its weight.
3. **R5 predicts** competence does not co-occur: of final-archive elites pooled
   over arch51-arch59, < 3% clear 0.012 in two media and < 0.5% in three.
   If >= 10% clear two media, the stepping stones exist and R5 is weakened.
4. **R3 predicts** the learner's per-tag return is flat: in each run, the
   slope of `return_by_tag` over generations is inside its own spread for air
   and for every crossing tag, and total PPO transitions per run are below
   2e7. If air return rises by more than twice its spread and the mission is
   still 0, R3 is not the binding cause.
5. **R1 (weak form) dies if** the stored references show one body flying the
   air segment, swimming and moving on land above each medium's floor under
   some controller. I expect the rotor reference to fly (0.96-0.99, recorded)
   and expect NOT to find any stored measurement of that same body in water
   and on land: predicted "unmeasured", which is itself a finding.
6. **R2 predicts** on a fixed, flight-capable body the learner does not reach
   a commanded turn with ample samples (>= 1e6 transitions). The stored M4 /
   `experiments/fixed_body_learning` result decides: water learned, air not.
   If its air arm had < 1e5 transitions, the read is "unmeasured", not R2.
7. **R4 predicts** random bodies match the archive at matched evaluations
   (recorded once: 1.94x on competence for arch49 gen 150). R4 is weakened if
   the later runs (arch54, arch56) beat matched random sampling by > 2x on the
   certified air criterion.
8. **R6 predicts** the purpose-built references (rotor control, any scripted
   crossing) score 0 on the gated crossing measures while physically crossing.
   Predicted: no such reference was ever run through `CrossingTracker`
   ("unmeasured").

## 4 What "normal triphibian locomotion" has to mean (to be fixed in DIAGNOSIS.md)

One body with one controller, in the continuous mission evaluation, on fresh
seeds: commanded locomotion in each of the three media above that medium's
certified floor, and every crossing the mission asks for made on command.
Bar referenced to the null (rule 9): the still machine and the base gait at 0
on the same seeds; the film reproduces the score. The repo's own E2 levels are
used if they already say this.

## 5 Design of this analysis

- Evidence: stored runs arch43-arch60 (`runs/`), the ROADMAP's measurements,
  `experiments/*/results`, the code. No search, no PPO run.
- Controls: still machine and base gait wherever a stored read has them.
- Comparability: fitness not comparable before arch51; water/land not
  comparable across mujoco 3.15 (arch57 on?) and 2026-10-06; air not across
  2026-10-10. Pooled reads stay inside one regime or are labelled.
- Stopping rule: the diagnosis is accepted when it explains four facts at
  once: (i) mission exactly 0, not small; (ii) random bodies beat the archive;
  (iii) air credit on < 5% of fresh draws; (iv) a rotor control flies the air
  segment while rotor elites do not.
