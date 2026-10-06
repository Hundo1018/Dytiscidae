# Tier-1 to Tier-2 ablation ladder (arch48, water and land)

Measured 2026-10-06 on `runs/arch48`. Question (`experiments/tier_gap/README.md`): 35 of 36
promotions with Tier-1 water >= 0.15 scored < 0.15 on the Tier-2 water leg. Does Tier-1 pay
for something Tier-2 shows is not there (a leak in the search's score), or does Tier-2
measure something else (a defect in the verifier and the critic's labels)?

**Answer: the first. Tier-1's credit is a property of the one task draw the generation scored
it on, mostly the commanded heading. Move the draw and the credit goes, with no Tier-2
difference involved, and the end of the ladder is what the real `evaluate_tier2` returns.**

Everything below is `run.py` (rungs) and `analyze.py` (tables), launched by `stageB..E.sh`
(stage A was `--n-max 16 --rungs R0,R0s,R1 --still R0s`). Raw values: `raw.json`; per-rung
wall: `raw.walls.json`; selection and recorded scores: `results.json`. Simulation wall 6128 s
over 37 rung-runs, one process at a time under `MemoryMax=3500M`.

```bash
PYTHONPATH=. .venv/bin/python experiments/tier_gap_ablation/analyze.py R0 R0s R1 R2b R3b R4b R5b R6b
PYTHONPATH=. .venv/bin/python experiments/tier_gap_ablation/analyze.py headings
```

## Elites

The 621 elites of `runs/arch48/archive_<island>.pkl` (de-duplicated: seeding files one design
on every island). The merged `archive.pkl` has 200 and 4 `policy_promoted`; the per-island
files have 28. **A "pass" is an archive elite with recorded Tier-1 `meta[medium] >= 0.15`,
not a promotion event** (the events carry no design id): 88 water, 104 land. Per medium, n
passes plus the same number of random controls (recorded < 0.15), `--n-max 16`: 16 + 16 for
R0, R0s, R1; a fixed random 10 + 10 (`--sub 10`) for everything else. 4 of the 61 elites run
were promoted. **All 28 `policy_promoted` in the archives equal `policy` exactly**: promotion
refinement accepts a step only if `mission_fraction` rises (`loop.py:875`) and it is 0 for
every arch48 elite, so the "refined controller" difference is a no-op here.

## Rung 0 reproduces the recorded Tier-1 score

R0 = `batchroll`'s per-domain loop at the elite's own `eval_seed`, all three domains in cycle
order, own stored policy and mobility basis, the network that scored its generation
(`scoring_networks/gen<N>.npz`), 8 s, no disturbances. Absolute error against the archive
(61 elites; every pass and control, 3 media):

| medium | within 0.005 | median abs error | max abs error |
|---|---|---|---|
| air | 61/61 | 0.0 | 0.0005 |
| water | 56/61 | 0.0002 | 0.165 |
| land | 48/61 | 0.00005 | 0.347 |

Among the passes: water 15/16 and land 13/16 are still >= 0.15 at R0. The misses are single
elites whose result changes by a lot (e.g. land 0.407 -> 0.060); I did not find why. R0 here
runs each (generation, seed) group alone, the search scored 16 candidates in shards of 2, and
the same elites through the numpy path at the same seed (R0o, 10 + 10) give water 10/10 and
land 9/10 passes with |R0 - R0o| median 0 and > 0.02 in 1/38 (water) and 4/38 (land). So
the reproduction is good, not exact; the misses are chaotic cases, not a systematic offset.
`R0p` (the cheaper "prime" form used for the alone-from-R0 rows) reproduces R0 to the digit.

## The ladder

Median over elites of the per-elite median over the 3 fresh seeds (910011, 910022, 910033)
for every row except R0 (the generation's seed, deterministic) and R7. `share` is the
fraction >= 0.15. Spearman is over all 20 elites against R0 (n is small: read the sign and
the order of magnitude). Cumulative: each row keeps the changes above it. 10 passes + 10
controls per medium.

| rung | what changes | water mean pos | water share pos / neg | water rho | land mean pos | land share pos / neg | land rho |
|---|---|---|---|---|---|---|---|
| R0 | Tier-1 as scored | 0.203 | 9/10, 0/10 | 1.00 | 0.205 | 8/10, 0/10 | 1.00 |
| **R0s** | **fresh seed (task, scatter, reset noise)** | **0.082** | **3/10**, 0/10 | **0.12** | **0.017** | **0/10**, 0/10 | **0.04** |
| R1 | + offline path (`env.rollout`, SummedPolicy) | 0.064 | 2/10, 0/10 | 0.27 | 0.010 | 0/10, 0/10 | -0.16 |
| R2b | + 25 s leg | 0.060 | 0/10, 0/10 | 0.48 | 0.016 | 0/10, 0/10 | 0.27 |
| R3b | + default task | 0.044 | 1/10, 0/10 | -0.02 | 0.040 | 1/10, 0/10 | 0.29 |
| R4b | + no scatter | 0.038 | 1/10, 0/10 | -0.09 | 0.050 | 1/10, 0/10 | 0.17 |
| R5b | + sea state, current, wind | 0.022 | 0/10, 0/10 | -0.13 | 0.045 | 1/10, 0/10 | 0.23 |
| R6b | + final network (not the scoring one) | 0.023 | 0/10, 1/10 | -0.16 | 0.140 | 4/10, 0/10 | 0.34 |
| R7 | real `evaluate_tier2`, same controller (n = 6, 1 seed) | see below | | | | | |

On all 16 + 16 (R0, R0s, R1): water passes 15/16 -> 3/16 -> 2/16, Spearman vs R0 1.00 / 0.10
/ 0.20; land 13/16 -> 0/16 -> 0/16, 1.00 / 0.09 / 0.05. R1 vs R0s, paired over all 192
(medium, elite, seed) cells: median |difference| 0, correlation 0.977 (water) and 0.988
(land): **the offline path is not a difference**. R2b..R6b run on the batched path ("b") for
wall (an offline 25 s leg costs 7 s; 5 rungs x 60 cells x 3 seeds did not fit); R7 is the
offline check.

**Where it collapses: R0 -> R0s, the seed, before anything Tier-2-specific is applied.** Water
passes 15/16 -> 3/16, land 13/16 -> 0/16, with the same path, leg, task *kind*, scatter,
network and no disturbances. Later rungs shave a little more off water (mean 0.082 -> 0.022)
and none off land.

**R7 against R6b** (real `evaluate_tier2(label_all_media=True)` with the R6 controller,
seed 910011, 6 elites per medium, per-elite values): land 0.073/0.071, 0.133/0.142, 0.0/0.0,
0.008/0.009, 0.002/0.002, 0.014/0.014 (R6b/R7), correlation 0.999; water correlation 0.963, one
elite 0.067 vs 0.136 and five within 0.002. The ladder's end point is the verifier.

R6 and R7 use the run's final network, because the promotion-time network is not stored per
promotion. The real Tier-2 saw the network of the promotion generation. R6b land (0.140
mean) is therefore likely more favourable than what Tier-2 saw; the water rows are unaffected.

## Which part of the draw: heading

Same seed except one part, 10 + 10, passes >= 0.15 (water / land):

| change from R0 | water | land |
|---|---|---|
| none (R0) | 9/10 | 8/10 |
| reset noise fresh (`S_env`) | 9/10 | 5/10 |
| scatter fresh (`S_scatter`) | 7/10 | 4/10 |
| **task draw fresh (`S_task`)** | **1/10** | **1/10** |
| task: depth set to default (`T_depth`) | 8/10 | 8/10 |
| task: hold/cruise order set to default (`T_order`) | 5/10 | 7/10 |
| **task: heading set to default pi/2 (`T_heading`)** | **1/10** | **2/10** |

Eight fixed headings, everything else the generation's draw (`H0..H7`, -pi .. 3pi/4):

| | cells >= 0.15 | passes per elite at >= 0.15 of 8 |
|---|---|---|
| water, passes | 21/80 | 1,1,2,2,2,2,2,3,3,3 |
| water, controls | 4/80 | 0,0,0,0,0,0,1,1,1,1 |
| land, passes | 16/80 | 0,0,1,2,2,2,2,2,2,3 |
| land, controls | 9/80 | 0,0,0,0,0,0,0,2,3,4 |

Each elite earns its credit on a band of one to three headings of eight and nothing outside
it. Six of ten water passes have their band at heading 0 (+x, the body's forward direction at
spawn): they swim forward and do not steer. `analyze.py headings` prints every elite.
Tier-1 draws one heading per generation, uniform on the circle, and shares it across the
16 candidates; an archive keeps the best. Tier-2's default task has a fixed heading
(`tasks.py:98` `DEFAULT_HEADING = pi/2`), hold first, depth 5.75; at heading pi/2 one of ten
water passes is above 0.15.

Fresh draws, all 48 (elite, seed) cells per group: passes reach >= 0.15 in 9/48 (water) and
6/48 (land); controls 4/48 and 3/48; pass mean 0.071 / 0.062 against control 0.039 / 0.024.
Tier-1 against Tier-1 at another draw correlates 0.10 (water) / 0.09 (land), as Tier-1
against Tier-2 does (-0.09 / 0.02, `tier_gap`).

## Each difference alone

From R0 (the generation's own seed kept, so only the named thing moves; `B_*`), passes >= 0.15
of 10 (controls in brackets); and from R0s (fresh seeds; `A_*`):

| alone | water from R0 | land from R0 | water from R0s | land from R0s |
|---|---|---|---|---|
| (base) | 9 (0) | 8 (0) | 3 (0) | 0 (0) |
| default task | **0** (1) | **2** (1) | 1 (1) | 0 (0) |
| 25 s leg | **2** (1) | 4 (0) | 0 (0), as R2b | 0 (0), as R2b |
| no scatter | 5 (1) | 5 (0) | 2 (0) | 1 (1) |
| disturbances | 7 (0) | 5 (0) | 2 (0) | 1 (0) |
| final network | 5 (1) | 5 (0) | 2 (1) | 0 (0) |
| offline path (`R0o`) | 10 (0) | 9 (0) | | |

Task and, in water, the leg length each remove most of the credit alone. Scatter, the
network and (on land) disturbances each remove about half, which is what reset noise alone
does on land (`S_env` 5/10): a competence selected as the best of a generation on one exact
realisation does not survive a different realisation of anything. The ladder's order
therefore does not matter for the verdict (everything is already gone by R0s).

## Still machine (`held_still_params`, rotors stopped)

Passes, 10 per medium (the same 10 in every row), water / land. "Above still" is a paired
difference > 0.02.

| rung | pass mean, actuated | pass mean, still | still machine >= 0.15 | median (actuated - still) | elites above still |
|---|---|---|---|---|---|
| R0 (Tier-1's draw) | 0.203 / 0.205 | 0.065 / 0.009 | 1/10 / 0/10 | +0.167 / +0.195 | 8/10 / 9/10 |
| R0s (the collapse rung) | 0.082 / 0.017 | 0.051 / 0.001 | 1/10 / 0/10 | 0.000 / 0.000 | 3/10 / 2/10 |
| R6b (end of the ladder) | 0.023 / 0.140 | 0.000 / 0.000 | 0/10 / 0/10 | 0.000 / +0.087 | 2/10 / 6/10 |

**Tier-1's credit was above the still machine at its own draw** (8 of 10 water and 9 of 10
land passes, by 0.17-0.20), so it is not a still-machine leak (CLAUDE.md, "Designing a
measurement", rules 6-7). At a fresh draw the passes are no longer distinguishable from
the still machine (median difference 0.000 in both media; the final network on land, R6b,
is the one place they stay above it). One caveat in water: a still
machine reaches 0.15 at Tier-1's draw for 2 of 10 controls (the passive cruise progress the
water score allows by design).

## All the differences, with where they are

1. Path: batched (`batchroll.py:906`) vs `TriphibianEnv.rollout`. Not a difference (above).
2. Leg: 8 s (`--segment-seconds`) vs 300 / 12 = 25 s (`evaluate.py:553`). Water: yes (B_leg 2/10).
3. Task: generation's draw (`batchroll.py:1065`, `tasks.py:189`) vs default schedule
   (`triphibian.py:1855` `_arm_task`; `tasks.py:98-203`: heading pi/2, hold first, depth
   5.75). **The dominant one, through the heading.**
4. Scatter: `env.scatter` after reset (`batchroll.py:1068`, `evaluate.py:348`) vs none in
   `evaluate_tier2` (`evaluate.py:558-565`). About half alone, nothing at fresh seeds.
5. Disturbances: `evaluate.py:527-541`. Small.
6. Controller: promotion refinement (`loop.py:2503`) is a no-op in arch48 (28/28 promoted
   policies equal the archived one), so the only controller difference is below.
7. **Network**: Tier-2 runs `_with_shared(state, ctrl2)` (`loop.py:2657`), the live network at
   promotion; Tier-1's recorded score used the network of its own generation
   (`scoring_networks/gen<N>.npz`). Not stored per promotion; I used the final network (R6).
8. **Seed**: Tier-2's `seed_2` is a fresh random draw per promotion (`loop.py:2658`), and
   `evaluate_tier2` draws sea, current, wind, start order and env seed from it. Tier-1's seed
   is the generation's, shared by 16 candidates and by the selection. This is the leak.
9. `evaluate_tier2` does not set `env.air_launch_height` (`evaluate.py:306` and
   `batchroll.py:968` do): air only, no effect when the spec value is None. Air was not
   run here; Tier-2 air is 0 for all 121 and needs its own look (a 25 s leg from a 30 m drop
   against 8 s).
10. A Tier-2 medium that the chain reached is the mean over its legs, otherwise the one probe
    leg (`evaluate.py:575`, `critic.py:115`); I ran one leg, as the probe does.

## Verdict

**Tier-1 leak, not a Tier-2 defect.** The Tier-1 score of a water or land elite in this run
is its competence on one commanded heading (and one hold/cruise order), picked as the best
of 16 on that draw. Elites are heading-specific (a band of 1-3 headings of 8), so a
single-draw score is a lottery: passes fall from 15/16 to 3/16 (water) and 13/16 to 0/16
(land) when the seed alone changes, and Tier-1 against Tier-1 at another draw (0.10, 0.09)
correlates as poorly as Tier-1 against Tier-2. Tier-2 is another draw (plus a 25 s leg, which
costs water more): `R7` equals the R6b rung, and R6b is where a fresh Tier-1 lands. The
critic's labels are that draw's noise, which fits its calibration skill of 0.0.

For the next run's list: score Tier-1 on several headings per candidate (or one per
candidate, not one per generation), and make Tier-2 report the same distribution; a
verifier at one fixed heading has the problem it is diagnosing. Not done here: the air
medium, the promotion-time network, a draw-count sweep to say how many headings are enough.

## Decisions I made that the brief did not specify

* Elites are archive elites (recorded >= 0.15), not promotion events; 4 of 61 were promoted.
* 16 + 16 for the cheap rungs, 10 + 10 for the rest, from a seeded random subsample, to fit the
  budget (planned <= 2 h; sim wall 1.7 h, about 2.3 h elapsed with builds and waits).
* Fresh seeds are 910011/22/33, all of one seed (env, scatter, task, disturbances) for the
  ladder; the `S_*` rows move one part.
* R2..R6 ran on the batched path ("b") after R1 showed the path equal; R7 is offline.
* R6 uses the final network, `policy_promoted` where present (28/28 identical anyway).
* Added rungs the brief did not name: `S_*`, `T_*`, `H*`, `R0o`, `R0p`, `B_*`.
* `run.py` only imports from `dytiscidae`; nothing under `dytiscidae/`, `runs/` or `docs/`
  was edited. The cheap suites: all seven print their success line (`test_index` failed once,
  mid-run, on the uncommitted `dytiscidae/` changes already in the tree, and passed after
  `docs/index/` was regenerated by someone else; this directory is not indexed).
