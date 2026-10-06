# The antipodal heading pair: does it close the leaks it was built for?

2026-10-06. Water and land are scored on a pair of rollouts from one initial
state: the drawn heading and its opposite (`tasks.antipode`,
`evaluate.run_segment`, and the same sequence in `batchroll.evaluate_tier1_batch`).
The progress term reads the mean of the two signed speeds along the commanded
headings, so motion the command did not choose cancels. Why: docs/PAPERS_2610.md
§1 and §3.

## 1. A still machine and an open-loop gait cancel exactly

`probe_still.py` runs eel and gannet, held still (`held_still_params`) and at
their base gait with no policy, through both evaluators at seed 3. The test
`test_the_heading_pair_cancels_what_the_command_did_not_choose` holds it.

- Pair mean signed progress: |mean| <= 2.8e-17 on every medium, both paths.
- Paired competence: 0.000000 everywhere, single path vs batch identical.

An open-loop gait cannot read the command, so it cannot earn cruise progress
in water or on land. Only a controller that sees the task can (CLAUDE.md
"Designing a measurement", rule 3).

## 2. The no-model gate on arch48's elites, before and after

The command is `experiments/no_model_gate/run.py --run runs/arch48`. It used the
same 200 merged-archive elites and the search's batched path, with
`--out experiments/heading_pair/gate_arch48_paired.json`. The elites were
selected under the old score, so the elite column is a lower bound on what a
search under the pair would find.

| bar | before: elite / still / base | after: elite / still / base | verdict vs still, before → after |
|---|---|---|---|
| water >= 0.012 | 121 / 103 / 112 | 41 / 2 / 2 | leak → **certified** (also vs base) |
| water >= 0.055 | 88 / 74 / 81 | 12 / 0 / 0 | leak → **certified** (also vs base) |
| water >= 0.15 | 45 / 35 / 42 | 2 / 0 / 0 | leak → underpowered |
| land >= 0.012 | 78 / 46 / 74 | 46 / 0 / 0 | certified → certified (now also vs base) |
| land >= 0.055 | 60 / 21 / 51 | 23 / 0 / 0 | certified → certified (now also vs base) |
| land >= 0.15 | 45 / 12 / 29 | 7 / 0 / 0 | certified → certified (now also vs base) |
| stage 1 pass | 61 / 23 / 46 | 12 / 2 / 5 | certified → certified |

The "before" counts are from `experiments/no_model_gate/results_arch48.json`.

Every competence bar is now certified or underpowered, and none leaks. These
bars are what selection pays for (`islands.island_score`,
`curriculum.stage_score`). Against the unsearched base gait, nothing was
certified before. Now every measured competence bar is.

The gradient survives. 41 of 200 elites (20.5%) still score >= 0.012 in water,
and 46 of 200 (23%) on land, under a score they were not selected for.

## 3. What still leaks, and why it is left

The judge's ladder rungs still leak (table in `gate_arch48_paired_table.md`):

- the water depth rungs `submerges`..`goes_deep`: still 147/119/85/37 against
  elite 136/91/58/26;
- the land posture rungs `stays_upright` and `supports_itself`;
- all four takeoff rungs;
- air rungs 1-7.

The rungs are not in selection. `judge.score` goes to `meta["rungs"]` and
`meta["judged"]` (telemetry), the scout's feature vector, and the auditor's
veto. Fitness is the island and curriculum scores, which read competence.

- **Air 1-4 (`lift_margin`)** measure the airframe on purpose. That is how
  arch37 doubled the flyable share, so they are not a defect. They are exempt.
- **Water depth, land posture and takeoff** are rungs a passive body can stand
  on.
  - A dense body sinks.
  - A body lying still is upright and in contact.
  - Three eel aerial_divers clear `climbs_out` in every arm.
- What they need is a re-specification against a commanded difference, for
  example two commanded depths. That should be measured before it is designed,
  as this pair was.
- Until then, report rung counts as **state, not capability** for these rungs.

Land `stirs` moved from leak to certified, but only just. Elites went 70 → 74
against still at 51, so the elites' lower bound is 0.313 against the still
machines' upper bound of 0.311. Read it as a tie. `moves` and `climbs_slope`
stay certified (58 / 23 against still 20 / 1).
