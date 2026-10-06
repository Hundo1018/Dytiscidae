# No-model admission gate on every Tier-1 bar -- arch48 -- 2026-10-06

Applies NeutronGym's no-model admission (`docs/papers/2610.03631.md` §3.5) to the
still-machine rule of CLAUDE.md "Designing a measurement". Until now the rule was checked
by hand and only on crossings; this runs it on every bar a Tier-1 evaluation can clear.

## Frozen before the run

**Question.** For each bar the search rewards, does a machine that uses no model clear it as
often as the elites do? Arms, on the same body, through `batchroll.evaluate_tier1_batch`
(three medium segments with the run's scatter and task draw, then the four transitions)
under the network that scored each elite (`scoring_networks/gen<gen>.npz`):

- `elite`: its own gait, own policy, scoring network (reproduces the archive: below).
- `still`: `TriphibianEnv.held_still_params()` (every actuator still, rotors stopped), no policy.
- `base`: the body's base CPG gait at its base amplitudes, no policy, no shared network
  (NeutronGym's "fixed answer": the gait that was never searched).

**Verdict** (`dytiscidae/domain/evidence.py`, one-sided 95% Clopper-Pearson):
`certified` iff the arm's upper bound < the elites' lower bound; `leak` if the arm clears the
bar and is not separable; `underpowered` if it never clears but the counts cannot separate;
`unmeasured` if no elite clears. **A `leak` means "not separable at these counts", not "proved
passive".** Two verdicts per bar: against `still` (the decision) and against `base`.

**Bars, enumerated from the code** (`enumerate_bars()`; 68 bars, none typed here):

| family | bars | source |
|---|---|---|
| ladder rungs `rung_reached(domain) >= r` | air 15, water 6, land 6, takeoff 4, transition 8 | `dytiscidae/evolution/judge.py:69` (`LADDER`), `judge.py:299` (`rung_reached`); the `takeoff` ladder is defined but not judged by `_score_candidate` (`loop.py:1029`), read from the land segment's `takeoff_height`; the `transition` ladder reads `TransitionSet.component_means()` + `crossed_fraction` as `_score_candidate` does |
| per-medium competence `>= t`, t in {0.012, 0.055, 0.15, 0.25, 0.30, 0.35} | 3 media x 6 | 0.012 `curriculum.py:106` `WEAKEST_BARS[1]` (= three_media, `loop.py:1881`; = 2 x `islands.py:135` `TRIPHIBIAN_FLOOR`); 0.055 `WEAKEST_BARS[0]`; 0.15 `evaluate.py:403` `LEG_COMPETENCE_BAR`; 0.25, 0.30, 0.35 `curriculum.py:66` `STAGES` |
| weakest-medium bars | `c2 >= 0.055`, `c3 >= 0.012` | `curriculum.py:106`, `loop.py:1797,1881` |
| curriculum stage pass (stages 0-3), under the elite's own island curriculum | 4 | `islands.curriculum_for` + `Curriculum.bar` (`curriculum.py:346`); std bars 0.25/0.35/0.35/0.30, triphibian 0.055/0.012/0.35/0.30 |
| stage 4 | `mission_fraction > 0` (the stage-4 bar is 0.0) | `curriculum.py:66` |
| crossings | `crossed` per kind (4) | `transitions.CrossingTracker` |

Not covered: the judge's ratchet bars (`judge.py` `HEADLINE`: continuous, they move with the
population), Tier-2 verification, the auditor, and the critic.

## What ran

Command (`experiments/no_model_gate/run.py`, `--selftest` checks `gate_table` without a simulator):

    systemd-run --user --unit nmg-arch48 -p MemoryMax=3500M env MUJOCO_GL=disable PYTHONPATH=. \
        DYTISCIDAE_KERNEL_DIR=<main>/mojo/build .venv/bin/python -u experiments/no_model_gate/run.py \
        --run runs/arch48 --out experiments/no_model_gate/results_arch48.json
    ... --island triphibian --out experiments/no_model_gate/results_arch48_triphibian.json

One process, no pool, no kill. A 10-elite probe took 211 s (21 s/elite); the full run took
3206 s for n = 200 (merged archive = every elite `load_run_archive` returns, the
same loading as `experiments/transition_distance`; the merged archive keeps one occupant per cell, so
it does not contain every island's elites) and the triphibian island's own archive 975 s for n = 84.
Both are every elite, no subsetting. Reproduction of the stored scores by the elite arm (|gap| <= 0.005):
merged air 200/200, water 192/200, land 178/200;
triphibian island air 84/84, water 81/84, land 74/84. Bodies (merged): {'gannet': 72, 'beetle': 63, 'random': 23, 'eel': 27, 'ray': 2, 'medusa': 13}.
Per-elite per-arm bar booleans, competences and failures: `results_*.json.rows.jsonl`; per-island counts: `per_island` in the JSON.

## Result: merged archive, n = 200

Columns: elite share k/n with its lower 95% bound; still and base with their upper bounds; verdict
of still and of base against the elites; distinct body plans / passes for the elites' and the still
machines' passes. Rows nobody clears are omitted (all `unmeasured`, in the JSON).

| bar | elite k/n (lo) | still k/n (hi) | base k/n (hi) | still | base | plans elite | plans still |
|---|---|---|---|---|---|---|---|
| rung:air>=1:makes_lift | 188/200 (0.905) | 188/200 (0.965) | 188/200 (0.965) | leak | leak | 6/188 | 6/188 |
| rung:air>=2:carries_a_third | 174/200 (0.824) | 174/200 (0.907) | 174/200 (0.907) | leak | leak | 6/174 | 6/174 |
| rung:air>=3:nearly_flies | 134/200 (0.611) | 134/200 (0.725) | 134/200 (0.725) | leak | leak | 5/134 | 5/134 |
| rung:air>=4:carries_itself | 118/200 (0.530) | 118/200 (0.648) | 118/200 (0.648) | leak | leak | 4/118 | 4/118 |
| rung:air>=5:leaves_surface | 84/200 (0.361) | 81/200 (0.465) | 88/200 (0.501) | leak | leak | 4/84 | 4/81 |
| rung:air>=6:stays_up | 10/200 (0.027) | 8/200 (0.071) | 10/200 (0.083) | leak | leak | 3/10 | 2/8 |
| rung:air>=7:glides | 2/200 (0.002) | 1/200 (0.023) | 3/200 (0.038) | leak | leak | 1/2 | 1/1 |
| rung:air>=8:holds_height | 1/200 (0.000) | 0/200 (0.015) | 1/200 (0.023) | underpowered | leak | 1/1 | - |
| rung:water>=1:submerges | 138/200 (0.632) | 147/200 (0.786) | 140/200 (0.753) | leak | leak | 6/138 | 6/147 |
| rung:water>=2:dives | 92/200 (0.400) | 119/200 (0.653) | 91/200 (0.516) | leak | leak | 6/92 | 6/119 |
| rung:water>=3:reaches_depth | 62/200 (0.256) | 85/200 (0.486) | 62/200 (0.368) | leak | leak | 5/62 | 5/85 |
| rung:water>=4:goes_deep | 26/200 (0.093) | 37/200 (0.236) | 26/200 (0.176) | leak | leak | 5/26 | 5/37 |
| rung:land>=1:stays_upright | 162/200 (0.759) | 172/200 (0.899) | 158/200 (0.836) | leak | leak | 6/162 | 6/172 |
| rung:land>=2:supports_itself | 141/200 (0.647) | 167/200 (0.877) | 141/200 (0.758) | leak | leak | 6/141 | 6/167 |
| rung:land>=3:stirs | 70/200 (0.294) | 51/200 (0.311) | 71/200 (0.415) | leak | leak | 5/70 | 5/51 |
| rung:land>=4:moves | 54/200 (0.219) | 20/200 (0.142) | 54/200 (0.326) | certified | leak | 5/54 | 5/20 |
| rung:land>=5:climbs_slope | 23/200 (0.080) | 1/200 (0.023) | 25/200 (0.170) | certified | leak | 3/23 | 1/1 |
| rung:land>=6:walks | 3/200 (0.004) | 0/200 (0.015) | 1/200 (0.023) | underpowered | leak | 2/3 | - |
| rung:takeoff>=1:unweights | 43/200 (0.168) | 26/200 (0.176) | 39/200 (0.247) | leak | leak | 5/43 | 4/26 |
| rung:takeoff>=2:hops | 22/200 (0.076) | 11/200 (0.089) | 14/200 (0.107) | leak | leak | 4/22 | 3/11 |
| rung:takeoff>=3:clears | 7/200 (0.017) | 4/200 (0.045) | 6/200 (0.058) | leak | leak | 4/7 | 2/4 |
| rung:takeoff>=4:climbs_out | 3/200 (0.004) | 3/200 (0.038) | 3/200 (0.038) | leak | leak | 1/3 | 1/3 |
| comp:air>=0.012 | 5/200 (0.010) | 1/200 (0.023) | 3/200 (0.038) | leak | leak | 1/5 | 1/1 |
| comp:air>=0.055 | 1/200 (0.000) | 1/200 (0.023) | 2/200 (0.031) | leak | leak | 1/1 | 1/1 |
| comp:water>=0.012 | 121/200 (0.545) | 103/200 (0.575) | 112/200 (0.619) | leak | leak | 6/121 | 6/103 |
| comp:water>=0.055 | 88/200 (0.381) | 74/200 (0.430) | 81/200 (0.465) | leak | leak | 5/88 | 6/74 |
| comp:water>=0.15 | 45/200 (0.177) | 35/200 (0.225) | 42/200 (0.263) | leak | leak | 5/45 | 4/35 |
| comp:water>=0.25 | 23/200 (0.080) | 20/200 (0.142) | 21/200 (0.148) | leak | leak | 4/23 | 3/20 |
| comp:water>=0.3 | 12/200 (0.035) | 14/200 (0.107) | 17/200 (0.125) | leak | leak | 2/12 | 3/14 |
| comp:water>=0.35 | 7/200 (0.017) | 8/200 (0.071) | 9/200 (0.077) | leak | leak | 2/7 | 2/8 |
| comp:land>=0.012 | 78/200 (0.332) | 46/200 (0.284) | 74/200 (0.430) | certified | leak | 6/78 | 5/46 |
| comp:land>=0.055 | 60/200 (0.247) | 21/200 (0.148) | 51/200 (0.311) | certified | leak | 6/60 | 5/21 |
| comp:land>=0.15 | 45/200 (0.177) | 12/200 (0.095) | 29/200 (0.192) | certified | leak | 6/45 | 3/12 |
| comp:land>=0.25 | 26/200 (0.093) | 5/200 (0.052) | 14/200 (0.107) | certified | leak | 5/26 | 3/5 |
| comp:land>=0.3 | 15/200 (0.047) | 3/200 (0.038) | 13/200 (0.101) | certified | leak | 5/15 | 2/3 |
| comp:land>=0.35 | 11/200 (0.031) | 3/200 (0.038) | 11/200 (0.089) | leak | leak | 5/11 | 2/3 |
| weakest:c2>=0.055 | 22/200 (0.076) | 7/200 (0.065) | 22/200 (0.153) | certified | leak | 5/22 | 3/7 |
| weakest:c3>=0.012 | 0/200 (0.000) | 0/200 (0.015) | 1/200 (0.023) | unmeasured | unmeasured | - | - |
| stage0:pass | 33/200 (0.123) | 11/200 (0.089) | 19/200 (0.136) | certified | leak | 5/33 | 4/11 |
| stage1:pass | 61/200 (0.251) | 23/200 (0.159) | 46/200 (0.284) | certified | leak | 6/61 | 5/23 |
| stage4:mission>0 | 4/200 (0.007) | 6/200 (0.058) | 3/200 (0.038) | leak | leak | 2/4 | 3/6 |

27 further bars were cleared by no elite, no still machine and no base machine out of n = 200 (all `unmeasured`); they are in the JSON.

Verdict counts over the 68 bars: still {'leak': 28, 'underpowered': 2, 'unmeasured': 28, 'certified': 10}; base {'leak': 40, 'unmeasured': 28}.

## Result: the triphibian island's own archive, n = 84 (the bars the island is paid on)

| bar | elite k/n (lo) | still k/n (hi) | base k/n (hi) | still | base | plans elite | plans still |
|---|---|---|---|---|---|---|---|
| comp:land>=0.055 | 9/84 (0.057) | 1/84 (0.055) | 17/84 (0.288) | certified | leak | 3/9 | 1/1 |
| comp:land>=0.15 | 8/84 (0.048) | 0/84 (0.035) | 7/84 (0.151) | certified | leak | 3/8 | - |
| weakest:c2>=0.055 | 4/84 (0.016) | 1/84 (0.055) | 5/84 (0.121) | leak | leak | 3/4 | 1/1 |
| weakest:c3>=0.012 | 1/84 (0.001) | 0/84 (0.035) | 1/84 (0.055) | underpowered | leak | 1/1 | - |
| stage0:pass | 4/84 (0.016) | 1/84 (0.055) | 5/84 (0.121) | leak | leak | 3/4 | 1/1 |
| stage1:pass | 1/84 (0.001) | 0/84 (0.035) | 1/84 (0.055) | underpowered | leak | 1/1 | - |
| stage4:mission>0 | 3/84 (0.010) | 5/84 (0.121) | 2/84 (0.073) | leak | leak | 2/3 | 2/5 |

Full table: `results_arch48_triphibian_table.md`.

## What the data say

- **Certified against still (10 of 68):** `land` rungs `moves` and `climbs_slope`, `comp:land >= 0.012 .. 0.30`,
  `weakest:c2 >= 0.055`, `stage0` and `stage1` pass. All are land-driven: a still machine clears
  them 2-4x less often than the elites.
- **Leaks (28), by mechanism.** (1) *Body-property rungs*: `rung:air>=1..4` (`lift_margin`, the airframe's
  static lift) are cleared by identical counts in all three arms (188/174/134/118 of 200); the ladder
  itself says so (`judge.py` comment above `flaps_forward`). (2) *Sinking is a dive*: `water>=1..4`
  read `depth_gain`, ungated, so a dense still body that sinks clears them more often than the elites
  (`dives` 119 still vs 92 elite; `goes_deep` 37 vs 26 of 200; e.g. eel, density ratio 2.2, sinks 13 m with
  its actuators off). Every `comp:water` threshold leaks (0.15: 35 vs 45; 0.35: 8 vs 7). (3) *Land `stirs`*
  (`land_peak_speed >= 0.08`) 51 still vs 70 elite: upright still bodies (gannet 7 and 9, eel 13)
  read 0.10-0.11 m/s over their best one-second window with `land_speed` 0.002-0.012, so the "posture gate
  reads exactly 0.0 for a passive machine" note in `judge.py` (`stirs` comment) does not hold on this
  population. `stays_upright` and `supports_itself` leak too (162 vs 172, 141 vs 167). (4) *Takeoff* rungs: all four are cleared by still
  machines (`climbs_out` 3 vs 3, the same 3 `eel` aerial_divers in every arm, 0.8 m takeoff from a body
  that was not asked to). (5) `stage4: mission_fraction > 0` is cleared by 6 still machines against 4 elites, but
  only at epsilon (largest value in any arm 1.6e-5, still max 7.8e-6): the stage-4 bar of 0.0 is not a bar.
  (6) `comp:air >= 0.012 / 0.055`: elites 5 and 1 of 200, still 1 and 1, base 3 and 2.
- **Underpowered (2):** `air>=8 holds_height` (1 elite), `land>=6 walks` (3 elites); not a certificate.
- **Unmeasured (28):** every air rung from `flaps_forward` up, water `holds_depth`/`manoeuvres`, the whole
  `transition` ladder, all four crossings (0 of 200 in every arm, as `runs/arch48_notes.md` says of the
  mission), and `three_media` on the merged archive (0 elites).
- **The island's own bar, `weakest:c3 >= 0.012` (three_media), on the 84 triphibian-island elites:** 1 elite
  (index 28, gannet, air 0.036 / water 0.095 / land 0.651), still 0/84 (upper 0.035), base 1/84: a different
  gannet's *base gait* (index 4: air 0.084 / water 0.024 / land 0.109) clears all three media, so the
  verdict against base is `leak` and against still `underpowered` (n = 1 elite cannot separate). The
  elite's own base gait scores air 0.0 and water 0.094, so what the search added over the base gait on
  that body is land (0.651 against 0.130) and air (0.036 against 0).
- Base-vs-elite: on this run the search added measurably to a body's base gait only on land
  (`comp:land >= 0.055`: elite 60, base 51 of 200; `moves` 54 vs 54; `climbs_slope` 23 vs 25: base `leak` on all, i.e. the base gait clears them as often);
  on air and water the base gait clears the same bars at the same rate (`dives` 91 base vs 92 elite).

## Decisions made without being told

- Elites = `load_run_archive(run)` merged (n = 200), plus the triphibian island's own archive as a
  second run because the merge drops the island's one three_media elite.
- `evaluate_tier1_batch` with `MissionSpec()` defaults and `identify_axes=False`, as `rescore.py` does; the
  arms differ only in controller/network. The `still` arm uses the elite's stored mobility basis (unused with no policy).
- Stage bars are evaluated under each elite's own island curriculum (the still twin gets the same island), so
  the threshold of a pooled `stageN:pass` row is the island's own (std or weakest).
- `takeoff` ladder included though the search does not judge it; `rung >= r` is the cumulative `rung_reached`.
- `gate_table` is thin glue over `evidence.py`; it has a `--selftest` in the script, not a test file under `tests/`.
